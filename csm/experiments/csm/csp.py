from __future__ import absolute_import, division, print_function


import multiprocessing
import os
import os.path as osp
import pdb
import threading
from collections import OrderedDict, defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pymesh
import scipy.io as sio
import scipy.misc

import torch
import torchvision
from absl import app, flags
from torch.autograd import Variable
from ...utils import mesh
from ...utils import metrics
from ...data import cub as cub_data
from ...data import imagenet as imnet_data
from ...data import p3d as p3d_data
from ...nnutils import loss_utils as loss_utils
from ...nnutils import geom_utils, icn_net, train_utils
from ...nnutils import net_blocks as nb 
from ...nnutils.nmr import NeuralRenderer
from ...utils import (bird_vis, cub_parse, mesh, render_utils, transformations,
                      visdom_render, visutil)

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', '..', 'cachedir')
flags.DEFINE_string('cachedir', cache_path, 'Cachedir')
flags.DEFINE_string('rendering_dir', osp.join('cache_path', 'rendering'),
                    'Directory where intermittent renderings are saved')
flags.DEFINE_string('result_dir', osp.join('cache_path', 'results'),
                    'Directory where intermittent renderings are saved')
flags.DEFINE_string('dataset', 'cub', 'cub or imnet or p3d')
flags.DEFINE_integer('seed', 0, 'seed for randomness')

cm = plt.get_cmap('jet')

class CSPTrainer(train_utils.Trainer):
    def define_model(self, ):
        opts = self.opts
        self.img_size = opts.img_size
        
        self.model = icn_net.ICPNet(opts)
       
        if opts.num_pretrain_epochs > 0:
            self.load_network(self.model, 'pred',
                              self.opts.num_pretrain_epochs)
        
        self.upsample_img_size = ((opts.img_size // 64) * (2**6),
                                  (opts.img_size // 64) * (2**6))

        self.grid = cub_parse.get_sample_grid(self.upsample_img_size).repeat(
            1, 1, 1, 1).to(self.device)

        self.offset_z = 5.0
        self.model.to(self.device)
        self.uv2points = cub_parse.UVTo3D(self.mean_shape)
        
        self.model_obj = pymesh.form_mesh(
            self.mean_shape['verts'].data.cpu().numpy(),
            self.mean_shape['faces'].data.cpu().numpy())
        model_obj_dir = osp.join(self.save_dir, 'model')

        visutil.mkdir(model_obj_dir)
        self.model_obj_path = osp.join(model_obj_dir, 'template.obj')
        pymesh.meshio.save_mesh(self.model_obj_path, self.model_obj)

        self.init_render()
        self.renderer_mask = NeuralRenderer(opts.img_size)
        self.hypo_mask_renderers = [NeuralRenderer(opts.img_size) for _ in range(opts.num_hypo_cams)]
        self.hypo_mask_renderers_2 = NeuralRenderer(opts.img_size)
        self.renderer_depth = NeuralRenderer(opts.img_size)
        self.hypo_depth_renderers = [NeuralRenderer(opts.img_size) for _ in range(opts.num_hypo_cams)]
        self.hypo_depth_renderers_2 = NeuralRenderer(opts.img_size)
        return

    def init_render(self, ):
        opts = self.opts
        self.keypoint_cmap = [cm(i * 17) for i in range(15)]
        faces_np = self.mean_shape['faces'].data.cpu().numpy()
        verts_np = self.mean_shape['sphere_verts'].data.cpu().numpy()
        uv_sampler = mesh.compute_uvsampler(
            verts_np, faces_np, tex_size=opts.tex_size) # F x tex_size x tex_size x 3
        uv_sampler = torch.from_numpy(uv_sampler).float().cuda()
        self.uv_sampler = uv_sampler.view(-1, len(faces_np),
                                          opts.tex_size * opts.tex_size, 2)


        self.verts_obj = self.mean_shape['verts']
        vis_rend = bird_vis.VisRenderer(opts.img_size, faces_np)
        self.visdom_renderer = visdom_render.VisdomRenderer(
            vis_rend, self.verts_obj, self.uv_sampler, self.offset_z,
            self.mean_shape_np, self.model_obj_path, self.keypoint_cmap, self.opts)
        return

    def init_dataset(self, ):
        opts = self.opts
        dataloader_fn = None
        if opts.dataset == 'cub':
            dataloader_fn = cub_data.cub_dataloader
        elif opts.dataset == 'p3d':
            dataloader_fn = p3d_data.p3d_dataloader
        elif opts.dataset == 'imnet':
            dataloader_fn = imnet_data.imnet_dataloader

        self.dataloader = dataloader_fn(opts)
        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if opts.dataset == 'cub':
            mpath = osp.join(opts.cub_cache_dir, '../shapenet/', 'bird', 'shape.mat')
        elif opts.dataset == 'p3d':
            if opts.p3d_class == 'car':
                mpath = osp.join(opts.p3d_cache_dir, '../shapenet/', 'car', 'shape.mat')
            else:
                mpath = osp.join(opts.p3d_cache_dir, '../shapenet/', opts.p3d_class, 'shape.mat')
        elif opts.dataset == 'imnet':
            mpath = osp.join(opts.imnet_cache_dir, '../shapenet/', opts.imnet_class, 'shape.mat')

        print('Loading Mean shape from {}'.format(mpath))
        self.mean_shape = cub_parse.load_mean_shape(mpath, self.device)
        self.mean_shape_np = sio.loadmat(mpath)
        self.mean_shape_pred_np = sio.loadmat(mpath)
        return

    def set_input(self, batch):
        opts = self.opts
        input_imgs = batch['img'].type(self.Tensor)
        mask = batch['mask'].type(self.Tensor)
        for b in range(input_imgs.size(0)):
            input_imgs[b] = self.resnet_transform(input_imgs[b])

        self.inds = [k.item() for k in batch['inds']]

        if opts.flip_train:
            self.inds.extend([k+10000 for k in self.inds])
        self.input_img_tensor = input_imgs.to(self.device)
        self.mask = mask.to(self.device)

        if opts.flip_train:
            flip_imgs = batch['flip_img'].type(self.Tensor)
            for b in range(flip_imgs.size(0)): 
                flip_imgs[b] = self.resnet_transform(flip_imgs[b])
            self.flip_imgs_tensor = flip_imgs.to(self.device)
            self.flip_mask = batch['flip_mask'].type(self.Tensor).to(self.device)

        cam_pose = batch['sfm_pose'].type(self.Tensor)
        self.cam_pose = cam_pose.to(self.device)
        self.codes_gt = {}
        self.kp_uv = batch['kp_uv'].type(self.Tensor).to(self.device)
        self.codes_gt['kp_uv'] = self.kp_uv
        self.codes_gt['kp'] = batch['kp'].type(self.Tensor).to(self.device)
        self.codes_gt['cam'] = self.cam_pose
        if opts.flip_train:
            new_pose =self.codes_gt['cam'][...,None,:]
            new_pose = self.reflect_cam_pose(new_pose).squeeze(1)
            self.codes_gt['cam'] = torch.cat([self.codes_gt['cam'], new_pose])

        kps_vis = self.codes_gt['kp'][..., 2] > 0
        kps_ind = (
            self.codes_gt['kp'] * 0.5 + 0.5) * self.input_img_tensor.size(-1)
        self.codes_gt['kps_vis'] = kps_vis
        self.codes_gt['kps_ind'] = kps_ind
        return

    def define_criterion(self):
        opts = self.opts
        self.smoothed_factor_losses = {
            'reproject': 0.0,
            'render_mask' : 0.0,
            'quat_err': 0.0,
            'dist_entropy':0.0,
            'rot_mag' : 0.0,
            'depth' : 0.0,
            'seg_mask' :0.0,
        }
        # self.smoothed_factor_losses = defaultdict(float)

    def reflect_cam_pose(self, cam_pose):
        new_cam_pose = cam_pose * torch.FloatTensor([1, -1, 1, 1, 1, -1, -1]).view(1,1,-1).cuda()
        return new_cam_pose


    def flip_train_predictions_swap(self, codes_pred, true_size):
        ## Copy cam
        ## Copy Cam Probs
        keys_to_copy = ['cam_probs', 'cam_sample_inds']
        for key in keys_to_copy:
            codes_pred[key]= torch.cat([codes_pred[key][:true_size], codes_pred[key][:true_size]])

        ## mirror rotation
        new_cam_pose = self.reflect_cam_pose(codes_pred['cam'][:true_size, None,:]).squeeze(1)
        if not (codes_pred['cam'][:true_size].shape == new_cam_pose.shape):
            pdb.set_trace()
        codes_pred['cam'] = torch.cat([codes_pred['cam'][:true_size], new_cam_pose])
        new_cam_hypos = self.reflect_cam_pose(codes_pred['cam_hypotheses'][:true_size])
        codes_pred['cam_hypotheses'] = torch.cat([codes_pred['cam_hypotheses'][:true_size], new_cam_hypos])
        return codes_pred


    def forward(self, ):
        opts = self.opts
        feed_dict = {}
        feed_dict['img'] = self.input_img_tensor
        feed_dict['mask'] = self.mask

        if opts.flip_train:
            feed_dict['img'] = torch.cat([self.input_img_tensor, self.flip_imgs_tensor])
            feed_dict['mask'] = torch.cat([self.mask, self.flip_mask])

        codes_pred = self.model.forward(feed_dict)
        b_size = len(feed_dict['img'])
        if opts.flip_train:
            codes_pred = self.flip_train_predictions_swap(codes_pred, true_size=len(self.mask))

        grid = self.grid.repeat(b_size,1,1,1)
        mask = torch.nn.functional.grid_sample(feed_dict['mask'].unsqueeze(1),
                                               grid)
        img = torch.nn.functional.grid_sample(feed_dict['img'], grid)

        self.codes_gt['img'] = img
        self.codes_gt['mask'] = mask
        self.codes_gt['xy_map'] = grid

        points3d = geom_utils.project_uv_to_3d(self.uv2points, codes_pred['uv_map'])
        codes_pred['points_3d'] = points3d.view(b_size, self.upsample_img_size[0], self.upsample_img_size[1], 3)

        if opts.use_gt_quat and opts.pred_cam:
            codes_pred['cam'] = torch.cat([codes_pred['cam'][:,0:3], self.codes_gt['cam'][:,3:7]], dim=1)
        else:
            codes_pred['cam'] = codes_pred['cam']

        codes_pred['project_points_cam_pred'] = geom_utils.project_3d_to_image(points3d, codes_pred['cam'], self.offset_z)
        codes_pred['project_points_cam_z'] = (codes_pred['project_points_cam_pred'][...,2] - self.cam_location[2]).view(self.codes_gt['xy_map'][...,0].size())
        codes_pred['project_points_cam_pred'] = codes_pred['project_points_cam_pred'][..., 0:2].view(self.codes_gt['xy_map'].size())
        
        if opts.multiple_cam_hypo:
            codes_pred['project_points_all_hypo'] = []
            codes_pred['project_points_z_all_hypo'] = []
            cams_all_hypo = codes_pred['cam_hypotheses']
            for cx in range(cams_all_hypo.size(1)):
                project_points_cam_cx = geom_utils.project_3d_to_image(points3d, cams_all_hypo[:,cx], self.offset_z)
                project_points_cam_z = (project_points_cam_cx[...,2] - self.cam_location[2]).view(self.codes_gt['xy_map'][...,0].size())
                project_points_cam_cx = project_points_cam_cx[...,0:2].view(self.codes_gt['xy_map'].size())
                codes_pred['project_points_all_hypo'].append(project_points_cam_cx)
                codes_pred['project_points_z_all_hypo'].append(project_points_cam_z)
            codes_pred['project_points_all_hypo'] = torch.stack(codes_pred['project_points_all_hypo'], 1)
            codes_pred['project_points_z_all_hypo'] = torch.stack(codes_pred['project_points_z_all_hypo'], 1)
            _, max_probs_inds = torch.max(codes_pred['cam_probs'], dim=1)
            codes_pred['cam'] = torch.gather(cams_all_hypo, dim=1, index=max_probs_inds.view(-1,1,1).repeat(1,1,7)).squeeze()
            codes_pred['cam_sample_inds'] = max_probs_inds.unsqueeze(-1)


        codes_pred['project_points'] = codes_pred['project_points_cam_pred']

        ## Render mean-shape and L2 Loss on mask.
        if opts.render_mask:
            camera = codes_pred['cam']
            # camera = torch.cat([camera[:, 0:3], camera[:, 3:7].detach()], dim=-1)
            faces = self.mean_shape['faces'][None,...].repeat(b_size,1,1)
            verts = self.mean_shape['verts'][None,...].repeat(b_size,1,1)
            mask_pred = self.renderer_mask.forward(verts, faces, camera)
            codes_pred['mask'] = mask_pred

            if opts.multiple_cam_hypo:
                codes_pred['mask_all_hypo'] = []
                cams_all_hypo = codes_pred['cam_hypotheses']
                for cx in range(cams_all_hypo.size(1)):
                    mask_pred = self.hypo_mask_renderers[cx].forward(verts, faces, cams_all_hypo[:,cx])
                    codes_pred['mask_all_hypo'].append(mask_pred)
                codes_pred['mask_all_hypo'] = torch.stack(codes_pred['mask_all_hypo'], 1)

        if opts.render_depth:
            camera = codes_pred['cam']
            faces = self.mean_shape['faces'][None,...].repeat(b_size,1,1)
            verts = self.mean_shape['verts'][None,...].repeat(b_size,1,1)
            depth_pred = self.renderer_depth.forward(verts, faces, camera, depth_only=True)

            codes_pred['depth'] = depth_pred

            if opts.multiple_cam_hypo:
                codes_pred['depth_all_hypo'] = []
                cams_all_hypo = codes_pred['cam_hypotheses']
                for cx in range(cams_all_hypo.size(1)):
                    depth_pred = self.hypo_depth_renderers[cx].forward(verts, faces, cams_all_hypo[:,cx], depth_only=True)
                    codes_pred['depth_all_hypo'].append(depth_pred)
                codes_pred['depth_all_hypo'] = torch.stack(codes_pred['depth_all_hypo'], 1)

        codes_pred['xy_map'] = codes_pred['project_points']

        codes_pred['iter'] = self.real_iter
        self.total_loss, self.loss_factors = loss_utils.code_loss(
            self.codes_gt, codes_pred, opts)


        codes_pred['quat_err'] = np.array([metrics.quat_dist(pred, gt) for pred, gt in zip(codes_pred['cam'][:, 3:].data.cpu(), self.cam_pose[:, 3:].data.cpu())])
        self.loss_factors['quat_err'] = np.mean(codes_pred['quat_err'])

        # self.loss_factors['cam_loss'] = torch.FloatTensor([0.0])
        for k in self.smoothed_factor_losses.keys():
            if 'var' in k or 'entropy' in k:
                if k in self.loss_factors.keys():
                    self.smoothed_factor_losses[k] = self.loss_factors[k].item()
            else:
                if k in self.loss_factors.keys():
                    self.smoothed_factor_losses[
                        k] = 0.99 * self.smoothed_factor_losses[
                            k] + 0.01 * self.loss_factors[k].item()

        self.codes_pred = codes_pred
        return


    def get_current_visuals(self, ):
        visuals = self.visuals_to_save(self.total_steps, count=1)[0]
        visuals.pop('ind')
        return visuals

    def visuals_to_save(self, total_steps, count=None):
        visdom_renderer = self.visdom_renderer
        opts = self.opts
        mean_shape_np = self.mean_shape_np

        if count is None:
            count = min(opts.save_visual_count, len(self.codes_gt['img']))

        batch_visuals = []
        mask = self.codes_gt['mask']
        img = self.codes_gt['img']
        uv_map = self.codes_pred['uv_map']
        camera = self.codes_pred['cam']

        results_dir = osp.join(opts.result_dir, "{}".format(opts.split),
                               "{}".format(total_steps))

        visual_ids = list(range(count))
        if count > 1 and opts.flip_train:
            offset = len(self.codes_gt['img'])//2
            visual_ids = list(range(count//2))
            visual_ids= [val for id1 in visual_ids for val in (id1, id1 + offset)]

        for b in visual_ids:
            visuals = {}
            if opts.render_mask:
                visuals['mask_render'] = visutil.tensor2im(self.codes_pred['mask'][b,None, None,...].repeat(1,3,1,1).data.cpu())

            visuals['z_img'] = visutil.tensor2im(
                visutil.undo_resnet_preprocess(img.data[b, None, :, :, :]))
            if opts.render_depth and opts.multiple_cam_hypo:
                depth_map = (self.codes_pred['depth'][b]*self.codes_pred['mask'][b])[None, None,...].repeat(1,3,1,1).data.cpu()
                all_depth_hypo = (self.codes_pred['mask_all_hypo'][b]*self.codes_pred['depth_all_hypo'][b])[:,None,:,:].repeat(1,3,1,1).data.cpu()/50.0
                all_depth_hypo = (all_depth_hypo.numpy()*255).astype(np.uint8).transpose(0,2,3,1)
                visuals['all_depth'] = visutil.image_montage(all_depth_hypo, nrow=3)


            visuals['z_mask'] = visutil.tensor2im(
                mask.data.repeat(1, 3, 1, 1)[b, None, :, :, :])
            visuals['uv_x'], visuals['uv_y'] = render_utils.render_uvmap(
                mask[b], uv_map[b].data.cpu())

            visuals['texture_copy'] = bird_vis.copy_texture_from_img(
                mask[b], img[b], self.codes_pred['xy_map'][b])
            texture_vps = visdom_renderer.render_model_using_nmr(
                uv_map.data[b], img.data[b], mask.data[b], camera[b], upsample_texture=True)
            visuals.update(texture_vps)
            texture_uv = visdom_renderer.render_model_uv_using_nmr(
                uv_map.data[b], mask.data[b], camera[b])
            visuals.update(texture_uv)
            visuals['ind'] = "{:04}".format(self.inds[b])

            if opts.render_depth and opts.multiple_cam_hypo:
                visuals['depth_loss'] = visdom_renderer.visualize_depth_loss_perpixel(self.loss_factors['depth_loss_all_hypo_vis'][b])

            if opts.pred_mask:
                visuals['pred_mask'] = visutil.tensor2im(self.codes_pred['seg_mask'].data.repeat(1, 3, 1, 1)[b, None, :, :, :])

    
            if opts.multiple_cam_hypo:
                vis_cam_hypotheses = visdom_renderer.render_all_hypotheses(self.codes_pred['cam_hypotheses'][b],
                                                                           self.codes_pred['cam_probs'][b],
                                                                           self.codes_pred['cam_sample_inds'][b].item(),
                                                                           self.codes_gt['cam'][b],
                                                                           None)
                visuals.update(vis_cam_hypotheses)

            batch_visuals.append(visuals)
            if count != 1:
                bird_vis.save_obj_with_texture('{:04}'.format(
                    self.inds[b]), results_dir, visuals['texture_img'],
                    mean_shape_np)
        return batch_visuals

    def get_current_points(self, ):
        pts_dict = {}
        return pts_dict

    def get_current_scalars(self, ):
        loss_dict = {
            'total_loss': self.smoothed_total_loss,
            'iter_frac': self.real_iter * 1.0 / self.total_steps
        }
        # loss_dict['grad_norm'] = self.grad_norm
        for k in self.smoothed_factor_losses.keys():
            # if np.abs(self.smoothed_factor_losses[k]) > 1E-5 or 'loss_{}'.format(k) in loss_dict.keys():
            loss_dict['loss_' + k] = self.smoothed_factor_losses[k]
        return loss_dict


FLAGS = flags.FLAGS


def main(_):
    
    seed = FLAGS.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    FLAGS.img_height = FLAGS.img_size
    FLAGS.img_width = FLAGS.img_size
    FLAGS.cache_dir = cache_path
    FLAGS.rendering_dir = osp.join(FLAGS.cache_dir, 'rendering', FLAGS.name)
    FLAGS.result_dir = osp.join(FLAGS.cache_dir, 'result', FLAGS.name)
    trainer = CSPTrainer(FLAGS)
    trainer.init_training()
    trainer.train()
    pdb.set_trace()


if __name__ == '__main__':
    app.run(main)
