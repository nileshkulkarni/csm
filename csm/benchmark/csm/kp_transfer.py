from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pprint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import pdb
from ...utils import visdom_render
from ...utils import transformations
from ...utils import visutil
from ...utils import mesh
from ...utils import cub_parse
from ...utils.visualizer import Visualizer
from ...nnutils import geom_utils
import pymesh
from ...utils import bird_vis
from ...nnutils.nmr import NeuralRenderer
from ...utils import render_utils
from ...nnutils import icn_net, geom_utils
from ...nnutils import loss_utils as loss_utils
from ...data import cub as cub_data
from ...data import p3d as p3d_data
from ...nnutils import test_utils

"""
Script for testing on CUB.

Sample usage: python -m cmr.benchmark.csp_keypoint --split val --name
<model_name> --num_train_epoch <model_epoch>
"""

from . import pck_eval
from absl import app
from absl import flags
import os
import os.path as osp
import numpy as np
import torch
import torchvision
import scipy.io as sio


cm = plt.get_cmap('jet')
# from matplotlib import set_cmap
flags.DEFINE_boolean('visualize', False, 'if true visualizes things')
flags.DEFINE_integer('seed', 0, 'seed for randomness')
flags.DEFINE_boolean('pose_dump', True, 'scale_trans_predictions dumped to a file')
flags.DEFINE_boolean('mask_dump', True, 'dump seg mask to file')
flags.DEFINE_string('quat_predictions_path', None, 'Load pose annotations')
flags.DEFINE_string('mask_predictions_path', None, 'Load mask annotations')
flags.DEFINE_boolean('robust', False, 'evaluate using a roboust measure')
flags.DEFINE_string('dataset', 'cub', 'Evaulate on birds')

opts = flags.FLAGS
# color_map = cm.jet(0)
kp_eval_thresholds = [0.05, 0.1, 0.2]


class CSPTester(test_utils.Tester):

    def define_model(self,):
        opts = self.opts
        self.img_size = opts.img_size
        self.model = icn_net.ICPNet(opts)
        self.load_network(self.model, 'pred', self.opts.num_train_epoch)
        self.mask_preds = None
        if opts.mask_predictions_path is not None:
            print('populating mask for birds')
            self.mask_preds = sio.loadmat(opts.mask_predictions_path)

        self.model.cuda()
        self.model.eval()
        self.upsample_img_size = (
            (opts.img_size // 64) * (2**6), (opts.img_size // 64) * (2**6))
        self.camera_solver = geom_utils.CameraSolver(self.Tensor, self.device)
        self.offset_z = 5.0
        self.uv2points = cub_parse.UVTo3D(self.mean_shape)
        self.model_obj = pymesh.form_mesh(self.mean_shape['verts'].data.cpu(
        ).numpy(), self.mean_shape['faces'].data.cpu().numpy())
        self.model_obj_path = osp.join(
            self.opts.cachedir, 'cub', 'model', 'mean_bird.obj')
        self.grid = cub_parse.get_sample_grid(self.upsample_img_size).repeat(
            opts.batch_size * 2, 1, 1, 1).to(self.device)


        self.init_render()
        self.kp_names = self.dl_img1.dataset.sdset.kp_names

        self.renderer_mask = NeuralRenderer(opts.img_size)
        self.hypo_mask_renderers = [NeuralRenderer(opts.img_size) for _ in range(opts.num_hypo_cams)]

        self.renderer_depth = NeuralRenderer(opts.img_size)
        self.hypo_depth_renderers = [NeuralRenderer(opts.img_size) for _ in range(opts.num_hypo_cams)]
        # self.render_mean_bird_with_uv()
        if opts.pose_dump:
            self.scale_trans_preds = {}  # iter, pair_id, pose_1, pose_2
            self.quat_preds = {}  # iter, pair_id, pose_1, pose_2
        if opts.mask_dump:
            self.mask_preds = {}
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
        self.uv_sampler = uv_sampler.view(-1, len(faces_np),
                                          opts.tex_size * opts.tex_size, 2)

        self.verts_obj = self.mean_shape['verts']
        self.visdom_renderer = visdom_render.VisdomRenderer(
            vis_rend, self.verts_obj, self.uv_sampler, self.offset_z,
            self.mean_shape_np, self.model_obj_path, self.keypoint_cmap, self.opts)
        return

    def init_dataset(self,):
        opts = self.opts
        if opts.dataset == 'cub':
            print('Loading the Birds dataset')
            dataloader_fn = cub_data.cub_test_pair_dataloader
        elif opts.dataset == 'p3d':
            print('Loading the p3d dataset {}'.format(opts.p3d_class))
            dataloader_fn = p3d_data.p3d_test_pair_dataloader
        else:
            assert False, 'Incorrect dataset type, {}'.format(opts.dataset)

        self.dl_img1 = dataloader_fn(opts, 1)
        self.dl_img2 = dataloader_fn(opts, 2)

        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        if opts.dataset == 'p3d':
            if opts.p3d_class == 'car':
                mpath = osp.join(opts.p3d_cache_dir, '../shapenet/', 'car', 'shape.mat')
            else:
                mpath = osp.join(opts.p3d_cache_dir, '../shapenet/', opts.p3d_class, 'shape.mat')
        elif opts.dataset == 'cub':
            mpath = osp.join(opts.cub_cache_dir, '../shapenet/', 'bird', 'shape.mat')

        print('Loading Mean shape from {}'.format(mpath))
        self.mean_shape = cub_parse.load_mean_shape(mpath, self.device)
        self.mean_shape_np = sio.loadmat(mpath)

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

    def predict(self,):
        opts = self.opts
        feed_dict = {}
        feed_dict['img'] = self.input_img_tensor
        feed_dict['mask'] = self.mask

        codes_pred = self.model.forward(feed_dict)
        b_size = len(self.mask)
        ratio = self.upsample_img_size[1] * 1.0 / self.upsample_img_size[0]

        mask = torch.nn.functional.grid_sample(
            self.mask.unsqueeze(1), self.grid[0:b_size])
        img = torch.nn.functional.grid_sample(
            self.input_img_tensor, self.grid[0:b_size])

        kps_vis = self.codes_gt['kps_vis']
        kps_uv = 0 * self.codes_gt['kp'][:, :, 0:2]
        kps_ind = self.codes_gt['kps_ind'].long()
        kps_ind_modified = 0 * kps_ind
        uv_maps = codes_pred['uv_map']
        for bx in range(len(kps_vis)):
            for kx in range(len(kps_vis[bx])):
                rx = kps_ind[bx][kx][1]
                cx = kps_ind[bx][kx][0]
                kps_uv[bx, kx] = uv_maps[bx, rx, cx]

        self.codes_pred = codes_pred
        if self.mask_preds is not None and not opts.mask_dump:
            self.codes_pred['seg_mask'] = self.populate_mask_from_file().squeeze()
        else:
             self.dump_predictions()
        return

    def dump_predictions(self,):
        opts = self.opts
        iter_index = "{:05}".format(self.iter_index)
        if opts.pose_dump:
            codes_pred = self.codes_pred
            camera = codes_pred['cam'].data.cpu().numpy()
            pose1 = {'scale_p1': camera[0, 0], 'trans_p1': camera[0, 1:3]}
            pose2 = {'scale_p2': camera[1, 0], 'trans_p2': camera[1, 1:3]}
            pose = pose1
            pose.update(pose2)
            pose['ind1'] = self.inds[0]
            pose['ind2'] = self.inds[1]
            self.scale_trans_preds[iter_index] = pose

            pose1 = {'quat_p1': camera[0, 3:7]}
            pose2 = {'quat_p2': camera[1, 3:7]}
            pose = pose1
            pose.update(pose2)
            self.quat_preds[iter_index] = pose

        if opts.mask_dump:
            mask_np = self.codes_pred['seg_mask'].data.cpu().numpy()
            mask = {}
            mask['mask_1'] = mask_np[0]
            mask['mask_2'] = mask_np[1]
            self.mask_preds[iter_index] = mask

    def populate_pose_from_file(self,):
        iter_index = "{:05}".format(self.iter_index)
        st = self.scale_trans_preds[iter_index]
        quat = self.quat_preds[iter_index]
        p1_s = np.array([st['scale_p1']])
        p2_s = np.array([st['scale_p2']])
        p1_t = st['trans_p1']
        p2_t = st['trans_p2']
        p1_q = quat['quat_p1'][0, 0][0]
        p2_q = quat['quat_p2'][0, 0][0]
        camera1 = np.concatenate([p1_s, p1_t, p1_q], axis=0)
        camera2 = np.concatenate([p2_s, p2_t, p2_q], axis=0)
        camera = np.stack([camera1, camera2], axis=0)
        return torch.from_numpy(camera.copy()).float().type(self.Tensor)

    def populate_mask_from_file(self,):
        iter_index = "{:05}".format(self.iter_index)
        masks = self.mask_preds[iter_index]
        mask1 = masks['mask_1'][0, 0]
        mask2 = masks['mask_2'][0, 0]
        mask = np.stack([mask1, mask2])
        return torch.from_numpy(mask).float().type(self.Tensor)

    def find_nearest_point_on_mask(self, mask, x, y):
        img_H = mask.size(0)
        img_W = mask.size(1)
        non_zero_inds = torch.nonzero(mask)
        distances = (non_zero_inds[:, 0] - y)**2 + (non_zero_inds[:, 1] - x) ** 2
        min_dist, min_index = torch.min(distances, dim=0)
        min_index = min_index.item()
        return non_zero_inds[min_index][1].item(), non_zero_inds[min_index][0].item()

    def map_kp_img1_to_img2(self, vis_inds, kps1, kps2, uv_map1, uv_map2, mask1, mask2):
        kp_mask = torch.zeros([len(kps1)]).cuda()
        kp_mask[vis_inds] = 1
        kps1 = kps1.long()

        kps1_vis = kps1[:, 2] > 200
        img_H = uv_map2.size(0)
        img_W = uv_map2.size(1)
        kps1_uv = uv_map1[kps1[:, 1], kps1[:, 0], :]

        kps1_3d = geom_utils.project_uv_to_3d(self.uv2points, kps1_uv[None, None, :, :])
        uv_points3d = geom_utils.project_uv_to_3d(self.uv2points, uv_map2[None, :, :, :])

        # kps1_3d = self.uv2points.forward()
        # uv_map2_3d = self.uv2points.forward()
        distances3d = torch.sum((kps1_3d.view(-1, 1, 3) - uv_points3d.view(1, -1, 3))**2, -1).sqrt()

        distances3d = distances3d + (1 - mask2.view(1, -1)) * 1000
        distances = distances3d
        min_dist, min_indices = torch.min(distances.view(len(kps1), -1), dim=1)
        min_dist = min_dist + (1 - kps1_vis).float() * 1000
        transfer_kps = torch.stack(
            [min_indices % img_W, min_indices // img_W], dim=1)

        kp_transfer_error = torch.norm((transfer_kps.float() - kps2[:, 0:2]), dim=1)
        return transfer_kps, torch.stack([kp_transfer_error, kp_mask, min_dist], dim=1)


    def evaluate(self,):
        # Collect keypoints that are visible in both the images. Take keypoints
        # from one image --> Keypoints in second image.
        common_kp_indices = torch.nonzero(
            self.codes_gt['kp'][0, :, 2] * self.codes_gt['kp'][1, :, 2] > 0.5)
        kps_ind = self.codes_gt['kps_ind']
        kps = self.codes_gt['kp']  # -1 to 1
        uv_map = self.codes_pred['uv_map']
        self.codes_pred['common_kps'] = common_kp_indices

        mask = (self.codes_pred['seg_mask'] > 0.5).float()

        transfer_kps12, error_kps12 = self.map_kp_img1_to_img2(
            common_kp_indices, kps_ind[0], kps_ind[1], uv_map[0], uv_map[1], mask[0], mask[1])
        transfer_kps21, error_kps21 = self.map_kp_img1_to_img2(
            common_kp_indices, kps_ind[1], kps_ind[0], uv_map[1], uv_map[0], mask[1], mask[0])

        kps1 = visutil.torch2numpy(kps_ind[0])
        kps2 = visutil.torch2numpy(kps_ind[1])

        self.codes_pred['tfs_12'] = transfer_kps12
        self.codes_pred['tfs_21'] = transfer_kps21

        return visutil.torch2numpy(transfer_kps12), visutil.torch2numpy(error_kps12), visutil.torch2numpy(transfer_kps21), visutil.torch2numpy(error_kps21), kps1, kps2

    def get_current_visuals(self,):
        visuals = self.visuals_to_save(self.total_steps, count=1)[0]
        visuals.pop('ind')
        return visuals

    def visuals_to_save(self, total_steps):
        visdom_renderer = self.visdom_renderer
        opts = self.opts
        batch_visuals = []
        mask = self.codes_gt['mask']
        img = self.codes_gt['img']
        uv_map = self.codes_pred['uv_map']
        results_dir = osp.join(opts.result_dir, "{}".format(
            opts.split), "{}".format(total_steps))
        if not osp.exists(results_dir):
            os.makedirs(results_dir)

        if opts.use_gt_cam:
            camera = self.codes_gt['cam']
        else:
            camera = self.codes_pred['cam']

        for b in range(len(img)):
            visuals = {}
            visuals['z_img'] = visutil.tensor2im(visutil.undo_resnet_preprocess(
                img.data[b, None, :, :, :]))
            # pdb.set_trace()
            visuals['img_kp'] = bird_vis.draw_keypoint_on_image(visuals['z_img'], self.codes_gt['kps_ind'][
                                                                b],  self.codes_gt['kps_vis'][b], self.keypoint_cmap)
            visuals['z_mask'] = visutil.tensor2im(
                mask.data.repeat(1, 3, 1, 1)[b, None, :, :, :])
            visuals['uv_x'], visuals['uv_y'] = render_utils.render_uvmap(
                mask[b], uv_map[b].data.cpu())
            # visuals['model'] =
            # (self.render_model_using_cam(self.codes_pred['cam'][b])*255).astype(np.uint8)
            visuals['texture_copy'] = bird_vis.copy_texture_from_img(
                mask[b], img[b],                                        self.codes_pred['xy_map'][b])

            texture_vps = visdom_renderer.render_model_using_nmr(uv_map.data[b], img.data[b], mask.data[b],
                                                                 camera[b], upsample_texture=True)
            visuals.update(texture_vps)
            # texture_uv = visdom_renderer.render_model_uv_using_nmr(
            #     uv_map.data[b], mask.data[b], camera[b])
            # visuals.update(texture_uv)
            visuals['ind'] = "{:04}".format(self.inds[b])
            texture_kp = visdom_renderer.render_kps_heatmap(uv_map.data[b], self.codes_gt['kps_ind'][b], self.codes_gt[
                'kps_vis'][b], camera[b])
            visuals.update(texture_kp)
            texture_gt_kp = visdom_renderer.render_gt_kps_heatmap(
                self.codes_gt['kp_uv'][b], camera[b])
            # visuals.update(texture_gt_kp)

            uv_contour = visdom_render.render_UV_contour(visuals['z_img'], uv_map.data[b].cpu(), mask.data[b].cpu())
            visuals['contour'] = uv_contour
            if opts.pred_xy_cycle:
                texture_cycle = visdom_renderer.render_cycle_images(
                    self.codes_pred['cycle_xy_map_mask'][b],
                    self.codes_pred['cycle_xy_map'][b], img.data[b],
                    mask.data[b], camera[b])
                visuals.update(texture_cycle)

            if opts.multiple_cam_hypo:
                vis_cam_hypotheses = visdom_renderer.render_all_hypotheses(self.codes_pred['cam_hypotheses'][b],
                                                                           self.codes_pred['cam_probs'][b],
                                                                           self.codes_pred['cam_sample_inds'][b].item(),
                                                                           self.codes_gt['cam'][b],
                                                                           [self.loss_factors['per_hypo_loss'][b]])
                visuals.update(vis_cam_hypotheses)
            steal_visuals = self.steal_colors()
            visuals.update(steal_visuals)

            # steal_visuals_cyc = self.steal_colors_cyc()
            # visuals.update(steal_visuals_cyc)

            batch_visuals.append(visuals)
            bird_vis.save_obj_with_texture('{:04}'.format(self.inds[b]), results_dir, visuals[
                                           'texture_img'], self.mean_shape_np)

            # bird_vis.save_obj_with_texture('{:04}'.format(self.inds[b]),
            # results_dir, visuals['texture_img'], self.mean_shape_np)

        ## transfer key point results:
        mask = self.codes_gt['mask']
        img = self.codes_gt['img']
        kps_ind = self.codes_gt['kps_ind']
        codes_pred  =self.codes_pred
        codes_gt = self.codes_gt
            
        visuals_tfs = bird_vis.render_transfer_kps_imgs(self.keypoint_cmap, batch_visuals[0]['z_img'], batch_visuals[1]['z_img'], kps_ind[0], kps_ind[1], 
            self.codes_pred['tfs_12'], self.codes_pred['tfs_21'], self.codes_pred['common_kps'] )
        batch_visuals[0].update(visuals_tfs)
        batch_visuals[1].update(visuals_tfs)

        return batch_visuals


    def test(self,):
        opts = self.opts
        bench_stats_m1 = {'kps1': [], 'kps2': [], 'transfer': [], 'kps_err': [], 'pair': [], }
        bench_stats_m2 = {'transfer': [], 'kps_err': [], 'pair': [], }

        n_iter = opts.max_eval_iter if opts.max_eval_iter > 0 else len(
            self.dl_img1)
        result_path = osp.join(
            opts.results_dir, 'results_{}.mat'.format(n_iter))
        print('Writing to %s' % result_path)
        self.visualizer = Visualizer(opts)
        visualizer = self.visualizer
        bench_stats = {}
        self.iter_index = None
        if not osp.exists(result_path) or opts.force_run:
            from itertools import izip
            for i, batch in enumerate(izip(self.dl_img1, self.dl_img2)):
                self.iter_index = i

                if i % 100 == 0:
                    print('{}/{} evaluation iterations.'.format(i, n_iter))
                if opts.max_eval_iter > 0 and (i >= opts.max_eval_iter):
                    break
                self.set_input(batch)
                self.predict()
                transfer_kps12, error_kps12, transfer_kps21, error_kps21, kps1, kps2 = self.evaluate()
                if opts.visualize and (i % opts.visuals_freq == 0):
                    visualizer.save_current_results(i, self.visuals_to_save(i))

                bench_stats_m1['transfer'].append(transfer_kps12)
                bench_stats_m1['kps_err'].append(error_kps12)
                bench_stats_m1['kps1'].append(kps1)
                bench_stats_m1['kps2'].append(kps2)
                bench_stats_m1['pair'].append(
                    (self.inds[0], self.inds[1]))

                bench_stats_m1['transfer'].append(transfer_kps21)
                bench_stats_m1['kps_err'].append(error_kps21)
                bench_stats_m1['kps1'].append(kps2)
                bench_stats_m1['kps2'].append(kps1)
                bench_stats_m1['pair'].append(
                    (self.inds[1], self.inds[0]))

            bench_stats_m1['kps1'] = np.stack(bench_stats_m1['kps1'])
            bench_stats_m1['kps2'] = np.stack(bench_stats_m1['kps2'])
            bench_stats_m1['transfer'] = np.stack(bench_stats_m1['transfer'])
            bench_stats_m1['kps_err'] = np.stack(bench_stats_m1['kps_err'])
            bench_stats_m1['pair'] = np.stack(bench_stats_m1['pair'])
            bench_stats['m1'] = bench_stats_m1

            if opts.pose_dump:
                pose_file = osp.join(opts.results_dir, 'scale_trans_dump_{}.mat'.format(n_iter))
                sio.savemat(pose_file, self.scale_trans_preds)

                pose_file = osp.join(opts.results_dir, 'quat_dump_{}.mat'.format(n_iter))
                sio.savemat(pose_file, self.quat_preds)

            if opts.mask_dump:
                mask_file = osp.join(opts.results_dir, 'mask_dump_{}.mat'.format(n_iter))
                sio.savemat(mask_file, self.mask_preds)

            sio.savemat(result_path, bench_stats)

        else:
            bench_stats = sio.loadmat(result_path)
            bench_stats_m1 = {}
            bench_stats_m1['pair'] = bench_stats['m1']['pair'][0][0]
            bench_stats_m1['kps_err'] = bench_stats['m1']['kps_err'][0][0]
            bench_stats_m1['transfer'] = bench_stats['m1']['transfer'][0][0]
            bench_stats_m1['kps1'] = bench_stats['m1']['kps1'][0][0]
            bench_stats_m1['kps2'] = bench_stats['m1']['kps2'][0][0]

        dist_thresholds = [1e-4, 1e-3,0.25*1e-2, 0.5*1e-2, 0.75*1e-2, 1E-2, 1E-1, 0.2, 0.3, 0.4, 0.5, 0.6, 10]
        pck_eval.run_evaluation(bench_stats_m1, n_iter, opts.results_dir, opts.img_size, self.kp_names, dist_thresholds)
        return


def main(_):
    # opts.n_data_workers = 0 opts.batch_size = 1 print = pprint.pprint
    opts.batch_size = 1
    opts.results_dir = osp.join(opts.results_dir_base, opts.name,  '%s' %
                                (opts.split), 'epoch_%d' % opts.num_train_epoch)
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
