from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pprint
import matplotlib.pyplot as plt
import pdb
from ...utils import visdom_renderer
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
from ...nnutils import cub_loss_utils as loss_utils
from ...data import cub as cub_data
from ...nnutils import test_utils
"""
Script for testing on CUB.

Sample usage:
python -m cmr.benchmark.csp_keypoint --split val --name <model_name> --num_train_epoch <model_epoch>
"""


from absl import app
from absl import flags
import os
import os.path as osp
import numpy as np
import torch
import torchvision
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')

cm = plt.get_cmap('jet')
# from matplotlib import set_cmap
flags.DEFINE_boolean('visualize', False, 'if true visualizes things')

opts = flags.FLAGS


class CSPTester(test_utils.Tester):
    def define_model(self,):
        opts = self.opts
        self.img_size = opts.img_size
        self.model = icn_net.ICPNet(opts)
        self.load_network(self.model, 'pred', self.opts.num_train_epoch)
        self.model.cuda()
        self.upsample_img_size = (
            (opts.img_size//64)*(2**6), (opts.img_size//64)*(2**6))
        self.camera_solver = geom_utils.CameraSolver(self.Tensor, self.device)
        self.offset_z = 5.0
        self.uv2points = cub_parse.UVTo3D(self.mean_shape)
        self.model_obj = pymesh.form_mesh(self.mean_shape['verts'].data.cpu(
        ).numpy(), self.mean_shape['faces'].data.cpu().numpy())
        self.model_obj_path = osp.join(
            self.opts.cachedir, 'cub', 'model', 'mean_bird.obj')
        self.grid = cub_parse.get_sample_grid(self.upsample_img_size).repeat(
            opts.batch_size, 1, 1, 1).to(self.device)
        self.init_render()
        # self.verts_obj = self.mean_shape['verts']
        # faces_np = self.mean_shape['faces'].data.cpu().numpy()
        # verts_np = self.mean_shape['verts'].data.cpu().numpy()
        # self.vis_rend = bird_vis.VisRenderer(opts.img_size, faces_np)
        # uv_sampler = mesh.compute_uvsampler(verts_np, faces_np, tex_size=opts.tex_size)
        # uv_sampler = torch.from_numpy(uv_sampler).float().cuda()
        # self.uv_sampler = uv_sampler.view(-1, len(faces_np), opts.tex_size*opts.tex_size, 2)
        self.model.eval()

        # self.render_mean_bird_with_uv()
        return

    def init_render(self, ):
        self.keypoint_cmap = [cm(i*17) for i in range(15)]
        vis_rend = bird_vis.VisRenderer(opts.img_size, faces_np)
        faces_np = self.mean_shape['faces'].data.cpu().numpy()
        verts_np = self.mean_shape['sphere_verts'].data.cpu().numpy()
        uv_sampler = mesh.compute_uvsampler(
            verts_np, faces_np, tex_size=opts.tex_size)
        uv_sampler = torch.from_numpy(uv_sampler).float().cuda()
        self.uv_sampler = uv_sampler.view(-1, len(faces_np),
                                          opts.tex_size*opts.tex_size, 2)

        self.verts_obj = self.mean_shape['verts']
        self.visdom_renderer = visdom_renderer.VisdomRenderer(vis_rend, self.verts_obj,
                                                              self.uv_sampler, self.offset_z,
                                                              self.mean_shape_np, self.model_obj_path,
                                                              self.keypoint_cmap, self.opts)

    def init_dataset(self,):
        self.dataloader = cub_data.cub_dataloader(opts, shuffle=True)
        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        self.mean_shape = cub_parse.load_mean_shape(
            osp.join(opts.cub_cache_dir, 'uv', 'mean_shape.mat'), self.device)
        self.mean_shape_np = sio.loadmat(
            osp.join(opts.cub_cache_dir, 'uv', 'mean_shape.mat'))
        return

    def set_input(self, batch):

        opts = self.opts
        input_imgs = batch['img'].type(self.Tensor)
        mask = batch['mask'].type(self.Tensor)
        for b in range(input_imgs.size(0)):
            input_imgs[b] = self.resnet_transform(input_imgs[b])
        self.inds = batch['inds']
        self.input_img_tensor = input_imgs.to(self.device)
        self.mask = mask.to(self.device)
        self.anchor_inds = batch['anchor'].long().to(self.device)
        self.neg_inds = batch['neg_inds'].long().to(self.device)
        self.pos_inds = batch['pos_inds'].long().to(self.device)
        self.codes_gt = {'anchor': self.anchor_inds,
                         'neg_ind': self.neg_inds, 'pos_ind': self.neg_inds}
        self.kp_uv = batch['kp_uv'].type(self.Tensor).to(self.device)
        self.codes_gt['kp_uv'] = self.kp_uv
        self.codes_gt['kp'] = batch['kp'].type(self.Tensor).to(self.device)

        cam_pose = batch['sfm_pose'].type(self.Tensor)
        self.cam_pose = cam_pose.to(self.device)
        self.codes_gt['cam_gt'] = self.cam_pose

        kps_vis = self.codes_gt['kp'][..., 2] > 0
        kps_ind = (self.codes_gt['kp']*0.5 + 0.5) * \
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
        ratio = self.upsample_img_size[1]*1.0/self.upsample_img_size[0]
        mask = torch.nn.functional.grid_sample(
            self.mask.unsqueeze(1), self.grid[0:b_size])
        img = torch.nn.functional.grid_sample(
            self.input_img_tensor, self.grid[0:b_size])

        self.codes_gt['img'] = img
        self.codes_gt['mask'] = mask
        self.codes_gt['xy_map'] = torch.cat(
            [self.grid[0:b_size, :, :, None,  0]*ratio, self.grid[0:b_size, :, :, None,  1]], dim=-1)

        points3d = geom_utils.project_uv_to_3d(
            self.uv2points, codes_pred['uv_map'])
        codes_pred['points_3d'] = points3d.view(
            b_size, self.upsample_img_size[0], self.upsample_img_size[1], 3)

        if opts.cam_compute_ls:  # Computes a camera using a linear system.
            # This camera is predicted using Neural Network
            codes_pred['cam'] = codes_pred['cam']

        codes_pred['project_points_cam_pred'] = geom_utils.project_3d_to_image(
            points3d,  codes_pred['cam'], self.offset_z)[..., 0:2]
        codes_pred['project_points_cam_pred'] = codes_pred['project_points_cam_pred'].view(
            self.codes_gt['xy_map'].size())
        if opts.use_gt_cam:
            project_points = geom_utils.project_3d_to_image(
                points3d, self.codes_gt['cam_gt'], self.offset_z)[..., 0:2]
            codes_pred['project_points'] = project_points.view(
                self.codes_gt['xy_map'].size())
            codes_pred['cam'] = self.codes_gt['cam_gt']
        else:
            codes_pred['project_points'] = codes_pred['project_points_cam_pred']

        codes_pred['xy_map'] = codes_pred['project_points']

        self.codes_pred = codes_pred

        return

    '''
    There 15 possible keypoints on every birds.
    kp_uv_locations  15 x 2
    '''

    def get_keypoint_color(self, kpx):
        return cm(kpx*17)

    # def render_gt_kps_heatmap(self, kp3d_uv, camera, uv_sampler):
    #     tex_size = self.opts.tex_size
    #     other_vps = False
    #     uv_H = 256
    #     uv_W = 256
    #     all_visuals = []
    #     default_tex = bird_vis.create_kp_heat_map_texture(uv_H, uv_W)
    #     kp_textures = []

    #     for kpx, kp_uv in enumerate(kp3d_uv):
    #         visuals = {}
    #         kp_uv = kp_uv*255
    #         uv_cords = [int(kp_uv[0]), int(kp_uv[1])]
    #         kp_color = self.keypoint_cmap[kpx]
    #         texture = bird_vis.create_kp_heat_map_texture(
    #             uv_H, uv_W, uv_cords[0], uv_cords[1], color=kp_color)
    #         kp_textures.append(texture)
    #     # kp_textures =
    #     kp_textures = np.stack(kp_textures, axis=0)
    #     default_mask = (0 == np.max(kp_textures[:, 3, None, :, :], axis=0))
    #     average = np.sum(kp_textures[:, 3, None, :, :], axis=0) + default_mask

    #     texture = np.sum(kp_textures, axis=0)/average
    #     texture = default_tex * default_mask + texture*(1-default_mask)
    #     texture = texture[0:3, :, :]
    #     texture = torch.from_numpy(texture).float().cuda()
    #     # renderer, vert, camera, uv_sampler, texture_image, tex_size, other_vps
    #     texture_ms, texture_img = bird_vis.wrap_texture_and_render(
    #         self.vis_rend, self.verts_obj, camera, uv_sampler, texture, tex_size, other_vps=True)
    #     visuals = {}
    #     visuals['texture_kp_zgt'] = visutil.image_montage(
    #         list(texture_ms), nrow=2)
    #     visuals['texture_kp_img_zgt'] = (texture_img*255).astype(np.uint8)
    #     return visuals

    # def render_kps_heatmap(self, uv_map, kps_ind, kps_vis, camera, uv_sampler):
    #     tex_size = self.opts.tex_size
    #     other_vps = False
    #     uv_H = 256
    #     uv_W = 256
    #     all_visuals = []
    #     default_tex = bird_vis.create_kp_heat_map_texture(uv_H, uv_W)
    #     kp_textures = []

    #     for kpx, (kp_ind, kp_vis) in enumerate(zip(kps_ind, kps_vis)):
    #         visuals = {}
    #         if kp_vis:
    #             y_cord = int(kp_ind[1])
    #             x_cord = int(kp_ind[0])

    #             uv_cords = torch.round(
    #                 uv_map[y_cord-2:y_cord+2, x_cord-2:x_cord+2, :]*255)
    #             uv_cords, _ = torch.median(uv_cords.view(-1, 2), dim=0)
    #             uv_cords = [int(uv_cords[0]), int(uv_cords[1])]
    #             kp_color = self.keypoint_cmap[kpx]
    #             texture = bird_vis.create_kp_heat_map_texture(
    #                 uv_H, uv_W, uv_cords[0], uv_cords[1], color=kp_color)
    #             kp_textures.append(texture)
    #     # kp_textures =
    #     kp_textures = np.stack(kp_textures, axis=0)
    #     default_mask = (0 == np.max(kp_textures[:, 3, None, :, :], axis=0))
    #     average = np.sum(kp_textures[:, 3, None, :, :], axis=0) + default_mask

    #     texture = np.sum(kp_textures, axis=0)/average
    #     texture = default_tex * default_mask + texture*(1-default_mask)
    #     texture = texture[0:3, :, :]
    #     texture = torch.from_numpy(texture).float().cuda()
    #     # renderer, vert, camera, uv_sampler, texture_image, tex_size, other_vps
    #     texture_ms, texture_img = bird_vis.wrap_texture_and_render(
    #         self.vis_rend, self.verts_obj, camera, uv_sampler, texture, tex_size, other_vps=True)
    #     visuals = {}
    #     visuals['texture_kp'] = visutil.image_montage(list(texture_ms), nrow=2)
    #     visuals['texture_kp_img'] = (texture_img*255).astype(np.uint8)
    #     return visuals

    def evaluate(self,):
        b_size = len(self.codes_gt['img'])

        kps_vis = self.codes_gt['kps_vis']
        kps_ind = torch.round(self.codes_gt['kps_ind']).long()
        batch_kps_uv = []
        batch_kps_3d = []
        uv_map = self.codes_pred['uv_map']
        points3d = self.codes_pred['points_3d']
        for b in range(b_size):
            kps_uv = uv_map[b, kps_ind[b][:, 1], kps_ind[b][:, 0], :]
            kps_3d = points3d[b, kps_ind[b][:, 1], kps_ind[b][:, 0], :]
            batch_kps_uv.append(kps_uv)
            batch_kps_3d.append(kps_3d)

        kps_3d = [temp.data.cpu().numpy() for temp in batch_kps_3d]
        kps_uv = [temp.data.cpu().numpy() for temp in batch_kps_uv]
        kps_vis = [temp.data.cpu().numpy() for temp in kps_vis]

        # UV location of all keypoints.
        # 3D location of all keypoints.
        # Return mean location and variance in uv space, and 3D space for every keypoint
        return kps_vis, kps_uv, kps_3d

    # def visualize(self, outputs, batch):
    #     camera = self.codes_gt['cam_gt'][0]
    #     visuals = self.render_kps_heatmap(self.codes_gt['kps_ind'][0], self.codes_gt['kps_vis'][0],camera, self.uv_sampler)
    #     return

    def get_current_visuals(self,):
        visuals = {}
        outputs = self.self.codes_gt
        camera = self.codes_gt['cam_gt'][0]
        self.visuals = self.render_kps_heatmap(
            outputs['kps_ind'][0], outputs['kps_vis'][0], camera, self.uv_sampler)
        mask = self.codes_gt['mask']
        visuals['img'] = visutil.tensor2im(visutil.undo_resnet_preprocess(
            self.codes_gt['img'].data))
        visuals['z_mask'] = visutil.tensor2im(mask.data.repeat(1, 3, 1, 1))
        return visuals

    # def render_model_using_nmr(self, uv_map, img, mask, cam,):
    #     visuals = {}
    #     texture_ms, texture_img = bird_vis.render_model_with_texture(
    #         self.vis_rend, self.verts_obj, 256, 256, uv_map, img, mask, cam, self.uv_sampler, other_vps=True, undo_resnet=True)
    #     visuals['texture_ms'] = visutil.image_montage(list(texture_ms), nrow=2)
    #     visuals['texture_img'] = (texture_img*255).astype(np.uint8)
    #     return visuals

    # def render_model_uv_using_nmr(self, uv_map, mask, cam,):
    #     u_image = uv_map[:, :, None, 0].permute(2, 1, 0).repeat(3, 1, 1)
    #     v_image = uv_map[:, :, None, 1].permute(2, 1, 0).repeat(3, 1, 1)
    #     visuals = {}
    #     img_pred_uu, _ = bird_vis.render_model_with_texture(
    #         self.vis_rend, self.verts_obj, 256, 256, uv_map, u_image, mask, cam, self.uv_sampler, other_vps=True)
    #     img_pred_vv, _ = bird_vis.render_model_with_texture(
    #         self.vis_rend, self.verts_obj, 256, 256, uv_map, v_image, mask, cam, self.uv_sampler, other_vps=True)
    #     visuals['z_texture_uu'] = visutil.image_montage(
    #         list(img_pred_uu), nrow=2)
    #     visuals['z_texture_vv'] = visutil.image_montage(
    #         list(img_pred_vv), nrow=2)

    #     img_pred_ms_u, _ = bird_vis.render_model_with_uv_greyscale_map(
    #         self.vis_rend, self.verts_obj, 256, 256, cam, self.uv_sampler, other_vps=True, uv_dim=0)
    #     visuals['u_ms'] = visutil.image_montage(list(img_pred_ms_u), nrow=2)
    #     img_pred_ms_v, _ = bird_vis.render_model_with_uv_greyscale_map(
    #         self.vis_rend, self.verts_obj, 256, 256, cam, self.uv_sampler, other_vps=True, uv_dim=1)
    #     visuals['v_ms'] = visutil.image_montage(list(img_pred_ms_v), nrow=2)
    #     return visuals

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

        for b in range(len(img)):
            visuals = {}
            visuals['z_img'] = visutil.tensor2im(visutil.undo_resnet_preprocess(
                img.data[b, None, :, :, :]))
            # pdb.set_trace()
            visuals['img_kp'] = bird_vis.draw_keypoint_on_image(
                visuals['z_img'], self.codes_gt['kps_ind'][b],  self.codes_gt['kps_vis'][b], self.keypoint_cmap)
            visuals['z_mask'] = visutil.tensor2im(
                mask.data.repeat(1, 3, 1, 1)[b, None, :, :, :])
            visuals['uv_x'], visuals['uv_y'] = render_utils.render_uvmap(
                mask[b], uv_map[b].data.cpu())
            # visuals['model'] = (self.render_model_using_cam(self.codes_pred['cam'][b])*255).astype(np.uint8)
            visuals['texture_copy'] = bird_vis.copy_texture_from_img(
                mask[b], img[b], self.codes_pred['xy_map'][b])
            texture_vps = visdom_renderer.render_model_using_nmr(
                uv_map.data[b], img.data[b], mask.data[b], self.codes_pred['cam'][b])
            visuals.update(texture_vps)
            texture_uv = visdom_renderer.render_model_uv_using_nmr(
                uv_map.data[b], mask.data[b], self.codes_pred['cam'][b])
            visuals.update(texture_uv)
            visuals['ind'] = "{:04}".format(self.inds[b])
            texture_kp = self.render_kps_heatmap(
                uv_map.data[b], self.codes_gt['kps_ind'][b], self.codes_gt['kps_vis'][b], self.codes_pred['cam'][b], self.uv_sampler)
            visuals.update(texture_kp)
            texture_gt_kp = visdom_renderer.render_gt_kps_heatmap(
                self.codes_gt['kp_uv'][b], self.codes_pred['cam'][b], self.uv_sampler)
            visuals.update(texture_gt_kp)
            batch_visuals.append(visuals)
            bird_vis.save_obj_with_texture('{:04}'.format(
                self.inds[b]), results_dir, visuals['texture_kp_img'], self.mean_shape_np)
            # bird_vis.save_obj_with_texture('{:04}'.format(self.inds[b]), results_dir, visuals['texture_img'], self.mean_shape_np)
        return batch_visuals

    def test(self,):
        opts = self.opts
        bench_stats = {'kps_vis': [], 'kps_uv': [], 'kps_3d': [], 'inds': []}
        result_path = osp.join(opts.results_dir, 'results.mat')
        print('Writing to %s' % result_path)
        self.visualizer = Visualizer(opts)
        visualizer = self.visualizer
        if not osp.exists(result_path):
            n_iter = len(self.dataloader)
            for i, batch in enumerate(self.dataloader):
                if i % 100 == 0:
                    print('{}/{} evaluation iterations.'.format(i, n_iter))
                if opts.max_eval_iter > 0 and (i >= opts.max_eval_iter):
                    break
                self.set_input(batch)
                self.predict()
                inds = self.inds.cpu().numpy()
                if opts.visualize and (i % opts.visuals_freq == 0):
                    visualizer.save_current_results(i, self.visuals_to_save(i))
                kps_vis, kps_uv, kps_3d = self.evaluate()
                bench_stats['kps_vis'].extend(kps_vis)
                bench_stats['kps_uv'].extend(kps_uv)
                bench_stats['kps_3d'].extend(kps_3d)
                bench_stats['inds'].append(inds)

            bench_stats['kps_vis'] = np.stack(bench_stats['kps_vis'])
            bench_stats['kps_uv'] = np.stack(bench_stats['kps_uv'])
            bench_stats['kps_3d'] = np.stack(bench_stats['kps_3d'])
            bench_stats['inds'] = np.concatenate(bench_stats['inds'])
            sio.savemat(result_path, bench_stats)
        else:
            bench_stats = sio.loadmat(result_path)

        pdb.set_trace()
        stats = {}

        valid_kps_uv = []
        valid_kps_3d = []

        for kx in range(15):
            vis_inds = np.where(bench_stats['kps_vis'][:, kx].squeeze())
            valid_kps_uv.append(bench_stats['kps_uv'][vis_inds[0], kx, :])
            valid_kps_3d.append(bench_stats['kps_3d'][vis_inds[0], kx, :])

        mean_kps_uv = []
        mean_kps_3d = []
        var_kps_uv = []
        var_kps_3d = []

        for kx in range(15):
            mean_kps_uv.append(np.mean(valid_kps_uv[kx], axis=0).round(4))
            mean_kps_3d.append(np.mean(valid_kps_3d[kx], axis=0).round(4))
            var_kps_uv.append(np.var(valid_kps_uv[kx], axis=0).round(4))
            var_kps_3d.append(np.var(valid_kps_3d[kx], axis=0).round(4))

        mean_kps_uv = np.stack(mean_kps_uv)
        mean_kps_3d = np.stack(mean_kps_3d)
        var_kps_uv = np.stack(var_kps_uv)
        var_kps_3d = np.stack(var_kps_3d)

        stats['mean_kp_uv'] = mean_kps_uv
        stats['mean_kps_3d'] = mean_kps_3d
        stats['var_kp_uv'] = var_kps_uv
        stats['var_kps_3d'] = var_kps_3d

        pprint.pprint(stats)
        self.plot_mean_var_ellipse(mean_kps_uv, var_kps_uv)
        # mean_iou = bench_stats['ious'].mean()

        # n_vis_p = np.sum(bench_stats['kp_vis'], axis=0)
        # n_correct_p_pt1 = np.sum(
        #     (bench_stats['kp_errs'] < 0.1) * bench_stats['kp_vis'], axis=0)
        # n_correct_p_pt15 = np.sum(
        #     (bench_stats['kp_errs'] < 0.15) * bench_stats['kp_vis'], axis=0)
        # pck1 = (n_correct_p_pt1 / n_vis_p).mean()
        # pck15 = (n_correct_p_pt15 / n_vis_p).mean()
        # print('%s mean iou %.3g, pck.1 %.3g, pck.15 %.3g' %
        #       (osp.basename(result_path), mean_iou, pck1, pck15))
    def plot_mean_var_ellipse(self, means, variances):

        from matplotlib.patches import Ellipse
        import matplotlib.pyplot as plt
        ax = plt.subplot(111, aspect='equal')

        for ix in range(len(means)):
            ell = Ellipse(xy=(means[ix][0], means[ix][1]),
                          width=variances[ix][0], height=variances[ix][1],
                          angle=0)
            color = self.keypoint_cmap[ix]*25
            ell.set_facecolor(color[0:3])
            ell.set_alpha(0.4)
            ax.add_artist(ell)
        ax.grid(True, which='both')
        plt.scatter(means[:, 0], means[:, 1])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('on')
        for i in range(len(means)):
            ax.annotate('{}'.format(i+1), (means[i, 0], means[i, 1]))
        plt.savefig('uv_errors.png')
        return


def main(_):
    # opts.n_data_workers = 0
    # opts.batch_size = 1
    # print = pprint.pprint
    opts.results_dir = osp.join(opts.results_dir_base, '%s' % (opts.split),
                                opts.name, 'epoch_%d' % opts.num_train_epoch)
    opts.result_dir = opts.results_dir
    if not osp.exists(opts.results_dir):
        print('writing to %s' % opts.results_dir)
        os.makedirs(opts.results_dir)

    torch.manual_seed(0)
    tester = CSPTester(opts)
    tester.init_testing()
    tester.test()


if __name__ == '__main__':
    app.run(main)
