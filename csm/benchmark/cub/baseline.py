from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')


import scipy.io as sio
import torchvision
import torch
import numpy as np
import os.path as osp
import os
from absl import flags
from absl import app
from ..cub import pck_eval, bench_eval
from ...nnutils import test_utils
from ...data import cub as cub_data
from ...nnutils import cub_loss_utils as loss_utils
from ...nnutils import icn_net, geom_utils
from ...utils import render_utils
from ...nnutils.nmr import NeuralRenderer
from ...utils import bird_vis
import pymesh
from ...nnutils import geom_utils
from ...utils.visualizer import Visualizer
from ...utils import cub_parse
from ...utils import mesh
from ...utils import visutil
from ...utils import transformations
from ...utils import visdom_render
import pdb
import json
import matplotlib.pyplot as plt
from ...utils import metrics
import pprint

"""
Script for testing on CUB for camera prediction on cubs.
Sample usage: nice -n 20 python -m icn.benchmark.cub.baseline --n_data_workers=1 --name=birds_gt_camera_baseline --max_eval_iter=10000 --use_html=True --visualize=False --visuals_freq=10 --split=val --num_train_epoch=200 --batch_size=16
"""


cm = plt.get_cmap('jet')
# from matplotlib import set_cmap
flags.DEFINE_boolean('visualize', False, 'if true visualizes things')
flags.DEFINE_integer('seed', 0, 'seed for randomness')
opts = flags.FLAGS
# color_map = cm.jet(0)
kp_eval_thresholds = [0.05, 0.1, 0.2]


class CSPTester(test_utils.Tester):

    def define_model(self,):
        opts = self.opts
        self.img_size = opts.img_size
        self.upsample_img_size = ((opts.img_size // 64) * (2**6), (opts.img_size // 64) * (2**6))
        self.offset_z = 5.0
        self.model = icn_net.CamNet(opts)
        self.load_network(self.model, 'pred', self.opts.num_train_epoch)
        self.model.cuda()
        self.model.eval()

        self.uv2points = cub_parse.UVTo3D(self.mean_shape)
        self.model_obj = pymesh.form_mesh(self.mean_shape['verts'].data.cpu(
        ).numpy(), self.mean_shape['faces'].data.cpu().numpy())
        self.grid = cub_parse.get_sample_grid(self.upsample_img_size).repeat(
            opts.batch_size, 1, 1, 1).to(self.device)
        self.model_obj_path = osp.join(
            self.opts.cachedir, 'cub', 'model', 'mean_bird.obj')
        self.init_render()
        self.default_uv_map = self.grid[0:1, :, :, :]
        # self.render_mean_bird_with_uv()
        return

    def init_render(self, ):
        opts = self.opts
        faces_np = self.mean_shape['faces'].data.cpu().numpy()
        verts_np = self.mean_shape['sphere_verts'].data.cpu().numpy()
        self.keypoint_cmap = [cm(i * 17) for i in range(15)]
        vis_rend = bird_vis.VisRenderer(opts.img_size, faces_np)
        uv_sampler = mesh.compute_uvsampler(
            verts_np, faces_np, tex_size=opts.tex_size)
        uv_sampler = torch.from_numpy(uv_sampler).float().cuda()
        self.uv_sampler = uv_sampler.view(-1, len(faces_np), opts.tex_size * opts.tex_size, 2)
        
        self.verts_obj = self.mean_shape['verts']
        self.visdom_renderer = visdom_render.VisdomRenderer(
            vis_rend, self.verts_obj, self.uv_sampler, self.offset_z,
            self.mean_shape_np, self.model_obj_path, self.keypoint_cmap, self.opts)

        texture_image_uv = cub_parse.get_sample_grid((256, 256)) * 0.5 + 0.5
        texture_image_uv = torch.cat([texture_image_uv[:, :, None, 0], texture_image_uv[:, :, None, 1],  texture_image_uv[:, :, None, 0]*0 + 1], dim=-1)  # U, V, Visible
        texture_image_uv = texture_image_uv.permute(2, 0, 1).cuda()
        self.default_texture_image_uv = texture_image_uv
        self.visdom_renderer.vis_rend.set_light_status(False)
        return

    def init_dataset(self,):
        opts = self.opts
        self.dataloader = cub_data.cub_dataloader(opts)
        
        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        mean_file_name = 'mean_shape.mat'
        if opts.cmr_mean_shape:
            mean_file_name = 'mean_cmr_shape.mat'

        self.mean_shape = cub_parse.load_mean_shape(
            osp.join(opts.cub_cache_dir, 'uv', mean_file_name), self.device)
        self.mean_shape_np = sio.loadmat(
            osp.join(opts.cub_cache_dir, 'uv', mean_file_name))
        return

    def set_input(self, batch):
        opts = self.opts
        input_imgs = batch['img'].type(self.Tensor)
        mask = batch['mask'].type(self.Tensor)
        for b in range(input_imgs.size(0)):
            input_imgs[b] = self.resnet_transform(input_imgs[b])
        self.inds = [k.item() for k in batch['inds']]
        self.input_img_tensor = input_imgs.to(self.device)
        self.mask = mask.to(self.device)

        self.codes_gt = {}
        self.codes_gt['mask'] = self.mask
        self.codes_gt['img'] = self.input_img_tensor
        self.kp_uv = batch['kp_uv'].type(self.Tensor).to(self.device)
        self.codes_gt['kp_uv'] = self.kp_uv
        self.codes_gt['kp'] = batch['kp'].type(self.Tensor).to(self.device)

        cam_pose = batch['sfm_pose'].type(self.Tensor)
        self.cam_pose = cam_pose.to(self.device)
        self.codes_gt['cam'] = self.cam_pose

        kps_vis = self.codes_gt['kp'][..., 2] > 0
        kps_ind = (self.codes_gt['kp'] * 0.5 + 0.5) * \
            self.input_img_tensor.size(-1)
        self.codes_gt['kps_vis'] = kps_vis
        self.codes_gt['kps_ind'] = kps_ind
        return

    def find_nearest_point_on_mask(self, mask, x, y):
        img_H = mask.size(0)
        img_W = mask.size(1)
        non_zero_inds = torch.nonzero(mask)
        distances = (non_zero_inds[:, 0] - y)**2 + (non_zero_inds[:, 1] - x) ** 2
        min_dist, min_index = torch.min(distances, dim=0)
        min_index = min_index.item()
        return non_zero_inds[min_index][1].item(), non_zero_inds[min_index][0].item()

    def forward(self, ):
        feed_dict = {}
        feed_dict['img'] = self.input_img_tensor
        feed_dict['mask'] = self.mask
        codes_pred = self.model.forward(feed_dict)
        self.codes_pred = codes_pred
        return

    def predict(self, ):
        # Render UV image using the camera pose.
        # Check which UVs are near the keypoints
        self.forward()
        visdom_renderer = self.visdom_renderer

        # uv_map = self.codes_gt['uv_map']
        imgs = self.codes_gt['img']
        mask = self.codes_gt['mask']
        if opts.use_gt_cam:
            camera = self.codes_gt['cam']
        else:
            camera = self.codes_pred['cam']
        return 

    def evaluate(self, ):
        pred_cam = self.codes_pred['cam'].data.cpu()
        gt_cam = self.codes_gt['cam'].data.cpu()
        quat_err  = [metrics.quat_dist(q_gt, q_pred) for q_gt, q_pred in zip(gt_cam[:,3:], pred_cam[:,3:])]
        trans_err = metrics.trans_error(gt_cam[:,1:3], pred_cam[:,1:3])
        scale_err = metrics.scale_error(gt_cam[:,None,0], pred_cam[:,None,0])
        return scale_err, trans_err, quat_err

    def visuals_to_save(self, total_steps, count=None):
        visdom_renderer = self.visdom_renderer
        opts = self.opts

        batch_visuals = []
        mask = self.codes_gt['mask']
        img = self.codes_gt['img']

        if count is None:
            count = len(img)


        if opts.use_gt_cam:
            camera = self.codes_gt['cam']
            camera_gt = self.codes_gt['cam']
        else:
            camera_gt = self.codes_gt['cam']
            camera = self.codes_pred['cam']

        results_dir = osp.join(opts.result_dir, "{}".format(opts.split),
                               "{}".format(total_steps))
        for b in range(count):
            visuals = {}
            visuals['z_img'] = visutil.tensor2im(
                visutil.undo_resnet_preprocess(img.data[b, None, :, :, :]))

            image_u, _, _, _ = visdom_renderer.render_default_uv_map(camera[b])
            mean_mask = image_u[:,:, 2]
            mean_mask = (mean_mask > 128).astype(np.float32)
            visuals['z_img_mask'] = visdom_renderer.render_mask_boundary(visuals['z_img'], mean_mask)

            image_u, _, _, _ = visdom_renderer.render_default_uv_map(camera_gt[b])
            mean_mask = image_u[:,:, 2]
            mean_mask = (mean_mask > 128).astype(np.float32)
            visuals['z_img_mask_gt'] = visdom_renderer.render_mask_boundary(visuals['z_img'], mean_mask)


            visuals['img_kp'] = bird_vis.draw_keypoint_on_image(
                visuals['z_img'], self.codes_gt['kps_ind'][b],
                self.codes_gt['kps_vis'][b], self.keypoint_cmap)
            visuals['z_mask'] = visutil.tensor2im(
                mask.data.repeat(1, 3, 1, 1)[b, None, :, :, :])
            visuals['ind'] = "{:04}".format(self.inds[b])
            batch_visuals.append(visuals)

        return batch_visuals

    def test(self,):
        opts = self.opts
        bench_stats = {'quat': [], 'trans': [], 'scale': [], 'inds' : [] }
        n_iter = opts.max_eval_iter if opts.max_eval_iter > 0 else len(self.dataloader)
        result_path = osp.join(
            opts.results_dir, 'results_cam_{}.mat'.format(n_iter))
        print('Writing to %s' % result_path)
        self.visualizer = Visualizer(opts)
        visualizer = self.visualizer
        # pdb.set_trace()
        if not osp.exists(result_path):
            from itertools import izip
            for i, batch in enumerate(self.dataloader,1):
                if i % 100 == 0:
                    print('{}/{} evaluation iterations.'.format(i, n_iter))
                if opts.max_eval_iter > 0 and (i >= opts.max_eval_iter):
                    break
                self.set_input(batch)
                self.predict()
                err_scale, err_trans, err_quat = self.evaluate()
                inds = self.inds
                if opts.visualize and (i % opts.visuals_freq == 0):
                    visualizer.save_current_results(i, self.visuals_to_save(i))
                bench_stats['quat'].append(err_quat)
                bench_stats['trans'].append(err_trans)
                bench_stats['scale'].append(err_scale)
                bench_stats['inds'].extend(inds)

            bench_stats['quat'] = np.concatenate(bench_stats['quat'])
            bench_stats['trans'] = np.concatenate(bench_stats['trans'])
            bench_stats['scale'] = np.concatenate(bench_stats['scale'])
            
            sio.savemat(result_path, bench_stats)
        else:
            bench_stats = sio.loadmat(result_path)

        json_file = osp.join(opts.results_dir, 'stats_{}.json'.format(n_iter))
        intervals = {'quat' : [10, 20, 30], 'trans': [0.01, 0.05, 0.1, 0.2, 0.3], 'scale': [0.05, 0.1, 0.15, 0.2]} 
        stats = bench_eval.camera_benchmark(intervals,bench_stats,)

        for key in intervals:
            print('{} : Median : {} , Mean : {}'.format(key, stats[key]['median'], stats[key]['mean']))

        for key in intervals:
            print(key)
            pprint.pprint(intervals[key])
            pprint.pprint(stats[key]['acc'])

        with open(json_file, 'w') as f:
            json.dump(stats, f)

        return



def main(_):
    # opts.n_data_workers = 0 opts.batch_size = 1 print = pprint.pprint
    # pdb.set_trace()
    opts.results_dir = osp.join(opts.results_dir_base, opts.name,  '%s' % (opts.split), 'epoch_%d' % opts.num_train_epoch)
    opts.result_dir = opts.results_dir
    if not osp.exists(opts.results_dir):
        print('writing to %s' % opts.results_dir)
        os.makedirs(opts.results_dir)

    seed = opts.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    tester = CSPTester(opts)
    tester.init_testing()
    tester.test()


if __name__ == '__main__':
    app.run(main)
