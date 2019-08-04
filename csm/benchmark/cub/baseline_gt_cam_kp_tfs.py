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
from ..cub import pck_eval
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

import pprint

"""
Script for testing on CUB.

Sample usage: nice -n 20 python -m icn.benchmark.baseline.evaluate --n_data_workers=1 --name=birds_gt_camera_baseline --max_eval_iter=10000 --use_html=True --visualize=True --visuals_freq=10 --split=val

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
        self.uv2points = cub_parse.UVTo3D(self.mean_shape)
        self.model_obj = pymesh.form_mesh(self.mean_shape['verts'].data.cpu(
        ).numpy(), self.mean_shape['faces'].data.cpu().numpy())
        self.grid = cub_parse.get_sample_grid(self.upsample_img_size).repeat(
            opts.batch_size, 1, 1, 1).to(self.device)
        self.model_obj_path = osp.join(
            self.opts.cachedir, 'cub', 'model', 'mean_bird.obj')
        self.init_render()
        self.kp_names = self.dl_img1.dataset.sdset.kp_names
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
        self.dl_img1 = cub_data.cub_test_pair_dataloader(opts, 1)
        self.dl_img2 = cub_data.cub_test_pair_dataloader(opts, 2)
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
        batch = cub_parse.collate_pair_batch(batch)
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
        self.codes_gt['cam_gt'] = self.cam_pose

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

    def predict(self, ):
        # Render UV image using the camera pose.
        # Check which UVs are near the keypoints
        visdom_renderer = self.visdom_renderer

        # uv_map = self.codes_gt['uv_map']
        imgs = self.codes_gt['img']
        mask = self.codes_gt['mask']
        camera = self.codes_gt['cam_gt']
        

        mean_masks = []
        uv_maps =[]
        for b in range(len(imgs)):
            image_u, _, image_v, _ = visdom_renderer.render_default_uv_map(camera[b])
            mask = image_u[:,:,None, 2]
            mask = (torch.from_numpy(mask) > 128).cuda().float()[:,:,0]
            image_u = torch.from_numpy(image_u).cuda().float()/255
            image_v = torch.from_numpy(image_v).cuda().float()/255
            uv_map = torch.cat([image_u[:,:,None,0], image_v[:,:,None,0]], dim=-1)
            mean_masks.append(mask)
            uv_maps.append(uv_map)

        mean_masks = torch.stack(mean_masks)
        uv_maps = torch.stack(uv_maps)
        kps_vis = self.codes_gt['kps_vis']
        kps_uv = 0*self.codes_gt['kp'][:,:,0:2]
        kps_ind = self.codes_gt['kps_ind'].long()
        kps_ind_modified = 0*kps_ind
        for bx in range(len(kps_vis)):
            for kx in range(len(kps_vis[b])):
                rx = kps_ind[bx][kx][1]
                cx = kps_ind[bx][kx][0]
                if mean_masks[bx, rx, cx] > 0.5:
                    kps_uv[bx, kx] = uv_maps[bx, rx, cx]
                    kps_ind_modified[bx][kx][1] = rx
                    kps_ind_modified[bx][kx][0] = cx
                else:
                    cx, rx = self.find_nearest_point_on_mask(mean_masks[bx], cx, rx)
                    kps_ind_modified[bx][kx][1] = rx
                    kps_ind_modified[bx][kx][0] = cx
                    
                    # Find the nearest point on the mask
                    kps_uv[bx, kx] = uv_maps[bx, rx, cx]

        kps_3d = self.uv2points.forward(kps_uv.view(-1, 2))
        kps_3d = kps_3d.view(kps_uv.size(0), kps_uv.size(1), 3)
        self.codes_pred = {}
        self.codes_pred['kps_uv'] = kps_uv[:, :, 0:2]
        self.codes_pred['kps_3d'] = kps_3d
        self.codes_pred['uv_maps'] = uv_maps
        self.codes_pred['mean_masks'] = mean_masks
        self.codes_pred['kps_ind_modif'] = kps_ind_modified
        return kps_uv, kps_3d, kps_vis

    def map_kp_img1_to_img2(self, vis_inds, kps1_3d, kps2, mask2, camera2,):
        
        transfer_kps = torch.clamp((geom_utils.orthographic_proj_withz(kps1_3d.unsqueeze(0), camera2.unsqueeze(0), offset_z=5.0).squeeze()[:, 0:2] + 1)*.5 * 255, min=0, max=255).long()
        # kps2 = (kps2 + 1)*0.5*255
        
        kp_mask = torch.zeros([len(kps2)]).cuda()
        kp_mask[vis_inds] = 1
        for i in range(len(transfer_kps)):
            if mask2[transfer_kps[i][1], transfer_kps[i][0]] < 0.5:
                transfer_kps[i][1], transfer_kps[i][0] = self.find_nearest_point_on_mask(mask2, transfer_kps[i][1], transfer_kps[i][0])

        kp_transfer_error = kp_mask[:, None] * ((transfer_kps.float() - kps2[:,0:2]))
        kp_transfer_error = torch.norm(kp_transfer_error, dim=1)
        return transfer_kps, torch.stack([kp_transfer_error, kp_mask], dim=1)

    def render_transfer_kps_imgs(self, img1, img2, kps1, kps2, transfer_kps12, transfer_kps21, common_kps):
        visuals = {}
        common_vis = kps1[:,0]*0
        common_vis[common_kps] = 1

        img1_tfs = bird_vis.draw_keypoint_on_image(img1, kps1, 
            common_vis, self.keypoint_cmap)
        img2_tfs = bird_vis.draw_keypoint_on_image(img2, kps2, 
            common_vis, self.keypoint_cmap)

        img_tfs12 = bird_vis.draw_keypoint_on_image(img2, transfer_kps12, 
            common_vis, self.keypoint_cmap)
        img_tfs21 = bird_vis.draw_keypoint_on_image(img1, transfer_kps21, 
            common_vis, self.keypoint_cmap)

        visuals['tfs_a_img1'] = img1_tfs
        visuals['tfs_d_img2'] = img2_tfs
        visuals['tfs_b_1to2'] = img_tfs12
        visuals['tfs_c_2to1'] = img_tfs21
    
        return  visuals
    
    ## uv_map only has 2 channels at the end. Fill the last channel with ones.
    def convert_uv_map_to_img(self, uv_map):
        uv_map = torch.cat([uv_map, uv_map[...,None, 0]*0], dim=-1)
        return uv_map


    def evaluate(self, ):
        common_kp_indices = torch.nonzero(self.codes_gt['kp'][0, :, 2] * self.codes_gt['kp'][1, :, 2] > 0.5)
        self.codes_pred['common_kps'] = common_kp_indices
        mask = self.codes_gt['mask']
        kps3d = self.codes_pred['kps_3d']
        kp = self.codes_gt['kp']
        kp_inds = self.codes_gt['kps_ind']
        transfer_kps12, error_kps12 = self.map_kp_img1_to_img2(common_kp_indices, kps3d[0], kp_inds[1], mask[1],  self.codes_gt['cam_gt'][1],)
        transfer_kps21, error_kps21 = self.map_kp_img1_to_img2(common_kp_indices, kps3d[1], kp_inds[0],  mask[0], self.codes_gt['cam_gt'][0])
        # transfer_kps12, error_kps12 = self.map_kp_img1_to_img2(common_kp_indices, kps3d[0], kp[0], mask[0],  self.codes_gt['cam_gt'][0],)
        # transfer_kps21, error_kps21 = self.map_kp_img1_to_img2(common_kp_indices, kps3d[1], kp[1],  mask[1], self.codes_gt['cam_gt'][1])
        self.codes_pred['tfs_12'] = transfer_kps12
        self.codes_pred['tfs_21'] = transfer_kps21
        return visutil.torch2numpy(transfer_kps12), visutil.torch2numpy(error_kps12), visutil.torch2numpy(transfer_kps21), visutil.torch2numpy(error_kps21)
    
    def visuals_to_save(self, total_steps):
        ## For each image, render the keypoints in 3D?
        visdom_renderer = self.visdom_renderer
        mask = self.codes_gt['mask']
        img = self.codes_gt['img']
        kps_uv = self.codes_pred['kps_uv']
        kps_3d = self.codes_pred['kps_3d']
        camera = self.codes_gt['cam_gt']
        kps_ind = self.codes_gt['kps_ind']
        ## For each image, show how keypoints transfer to location of the mask
        visuals = {}
        visuals['z_img1'] = visutil.tensor2im(visutil.undo_resnet_preprocess(
            img.data[0, None, :, :, :]))
        visuals['z_img2'] = visutil.tensor2im(visutil.undo_resnet_preprocess(
            img.data[1, None, :, :, :]))
        mean_masks = visutil.torch2numpy(self.codes_pred['mean_masks'])
        visuals['z_img1'] = visdom_renderer.render_mask_boundary(visuals['z_img1'], mean_masks[0])
        visuals['z_img2'] = visdom_renderer.render_mask_boundary(visuals['z_img2'], mean_masks[1])

        visuals_tfs = self.render_transfer_kps_imgs(visuals['z_img1'], visuals['z_img2'], kps_ind[0], kps_ind[1], 
            self.codes_pred['tfs_12'], self.codes_pred['tfs_21'], self.codes_pred['common_kps'] )
        visuals.update(visuals_tfs)
        
        
        visuals['img_kp1'] = bird_vis.draw_keypoint_on_image(visuals['z_img1'],
            self.codes_gt['kps_ind'][0], self.codes_gt['kps_vis'][0], self.keypoint_cmap)
        visuals['img_kp1_new'] = bird_vis.draw_keypoint_on_image(visuals['z_img1'],
            self.codes_pred['kps_ind_modif'][0], self.codes_gt['kps_vis'][0], self.keypoint_cmap)
        visuals['ind'] = "{:04}".format(self.inds[0])
        visuals['img_kp2'] = bird_vis.draw_keypoint_on_image(visuals['z_img2'],
            self.codes_gt['kps_ind'][1], self.codes_gt['kps_vis'][1], self.keypoint_cmap)
        
        visuals['img_kp2_new'] = bird_vis.draw_keypoint_on_image(visuals['z_img2'],
            self.codes_pred['kps_ind_modif'][1], self.codes_gt['kps_vis'][1], self.keypoint_cmap)
        
        kp_ms = visdom_renderer.render_gt_kps_heatmap(kps_uv[0],  camera[0], suffix='_img1')
        visuals.update(kp_ms)
        kp_ms = visdom_renderer.render_gt_kps_heatmap(kps_uv[1],  camera[1], suffix='_img2')
        visuals.update(kp_ms)

        uv_maps = self.convert_uv_map_to_img(self.codes_pred['uv_maps'])
        visuals['uv_map_img1'] = visutil.tensor2im(uv_maps[0,None,:,:,:].permute(0, 3,1,2))
        visuals['uv_map_img2'] = visutil.tensor2im(uv_maps[1,None,:,:,:].permute(0, 3,1,2))
       
     
        return [visuals]


    def test(self,):
        opts = self.opts
        bench_stats_m1 = {'transfer': [], 'kps_err': [], 'pair': [], }

        n_iter = opts.max_eval_iter if opts.max_eval_iter > 0 else len(
            self.dl_img1)
        result_path = osp.join(
            opts.results_dir, 'results_{}.mat'.format(n_iter))
        print('Writing to %s' % result_path)
        self.visualizer = Visualizer(opts)
        visualizer = self.visualizer
        if not osp.exists(result_path):
            # n_iter = len(self.dl_img1)
            from itertools import izip
            for i, batch in enumerate(izip(self.dl_img1, self.dl_img2)):
                if i % 100 == 0:
                    print('{}/{} evaluation iterations.'.format(i, n_iter))
                if opts.max_eval_iter > 0 and (i >= opts.max_eval_iter):
                    break
                # batch = [batch[0], batch[0]]
                self.set_input(batch)
                self.predict()
                transfer_kps12, error_kps12, transfer_kps21, error_kps21 = self.evaluate()
                # inds = self.inds.cpu().numpy()
                if opts.visualize and (i % opts.visuals_freq == 0):
                    visualizer.save_current_results(i, self.visuals_to_save(i))
                bench_stats_m1['transfer'].append(transfer_kps12)
                bench_stats_m1['transfer'].append(transfer_kps21)
                bench_stats_m1['kps_err'].append(error_kps12)
                bench_stats_m1['kps_err'].append(error_kps21)
                bench_stats_m1['pair'].append(
                    (self.inds[0], self.inds[1]))
                bench_stats_m1['pair'].append(
                    (self.inds[1], self.inds[0]))

            bench_stats_m1['transfer'] = np.stack(bench_stats_m1['transfer'])
            bench_stats_m1['kps_err'] = np.stack(bench_stats_m1['kps_err'])
            bench_stats_m1['pair'] = np.stack(bench_stats_m1['pair'])

            bench_stats = {}
            bench_stats['m1'] = bench_stats_m1
            sio.savemat(result_path, bench_stats)
        else:
            bench_stats = sio.loadmat(result_path)
            bench_stats_m1 = {}
            bench_stats_m1['pair'] = bench_stats['m1']['pair'][0][0]
            bench_stats_m1['kps_err'] = bench_stats['m1']['kps_err'][0][0]
            bench_stats_m1['transfer'] = bench_stats['m1']['transfer'][0][0]

        json_file = osp.join(opts.results_dir, 'stats_m1_{}.json'.format(n_iter))

        stats_m1 = pck_eval.benchmark_all_instances(kp_eval_thresholds, self.kp_names, bench_stats_m1, opts.img_size)
        stats = stats_m1
        print(' Method 1 | Keypoint | Median Err | Mean Err | STD Err')
        pprint.pprint(zip(stats['kp_names'], stats['median_kp_err'], stats['mean_kp_err'], stats['std_kp_err']))
        print('PCK Values')
        pprint.pprint(stats['interval'])
        pprint.pprint(stats['pck'])

        mean_pck = {}
        for i, thresh  in enumerate(stats['interval']):
            mean_pck[thresh] = []
            for kp_name in self.kp_names:
                mean_pck[thresh].append(stats['pck'][kp_name][i])

        mean_pck = {k: np.mean(np.array(t)) for k,t in mean_pck.items()}
        pprint.pprint('Mean PCK  ')
        pprint.pprint(mean_pck)

        with open(json_file, 'w') as f:
            json.dump(stats, f)

        return

    def plot_mean_var_ellipse(self, means, variances):

        from matplotlib.patches import Ellipse
        import matplotlib.pyplot as plt
        ax = plt.subplot(111, aspect='equal')

        for ix in range(len(means)):
            ell = Ellipse(xy=(means[ix][0], means[ix][1]),
                          width=variances[ix][0], height=variances[ix][1],
                          angle=0)
            color = self.keypoint_cmap[ix] * 25
            ell.set_facecolor(color[0:3])
            ell.set_alpha(0.4)
            ax.add_artist(ell)
        ax.grid(True, which='both')
        plt.scatter(means[:, 0], means[:, 1])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('on')
        for i in range(len(means)):
            ax.annotate('{}'.format(i + 1), (means[i, 0], means[i, 1]))
        plt.savefig('uv_errors.png')
        return


def main(_):
    # opts.n_data_workers = 0 opts.batch_size = 1 print = pprint.pprint
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
