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
from ...nnutils import cub_loss_utils as loss_utils
from ...data import cub as cub_data
from ...data import p3d as p3d_data
from ...nnutils import test_utils
from ...external.pytorch_knn_cuda import knn_pytorch
from torch.autograd import Variable, Function
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
flags.DEFINE_string('st_predictions_path', None, 'Load pose annotations')
flags.DEFINE_boolean('robust', False, 'evaluate using a roboust measure')
flags.DEFINE_string('dataset', 'cub', 'Evaulate on birds')

opts = flags.FLAGS
# color_map = cm.jet(0)
kp_eval_thresholds = [0.05, 0.1, 0.2]


class KNearestNeighbor(Function):
  """ Compute k nearest neighbors for each query point.                                      
  """
  def __init__(self, k):
    self.k = k

  def forward(self, ref, query):
    ref = ref.float().cuda()
    query = query.float().cuda()

    inds = torch.empty(query.shape[0], self.k, query.shape[2]).long().cuda()                 

    # make sure inputs are contiguous
    knn_pytorch.knn(ref.contiguous(), query.contiguous(), inds.contiguous())                 

    return inds


class CSPTester(test_utils.Tester):

    def define_model(self,):
        opts = self.opts
        self.img_size = opts.img_size
        self.model = icn_net.ICPNet(opts)
        self.load_network(self.model, 'pred', self.opts.num_train_epoch)
        self.mask_preds = None
        if  opts.mask_predictions_path is not None:
            print('populating mask for birds')
            self.mask_preds = sio.loadmat(opts.mask_predictions_path)
        # self.scale_trans_preds = sio.loadmat(opts.st_predictions_path)
        # self.quat_preds = sio.loadmat(opts.quat_predictions_path)
        self.knn = KNearestNeighbor(1)

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
            opts.batch_size*2, 1, 1, 1).to(self.device)

        self.triangle_loss_fn = loss_utils.LaplacianLoss(self.mean_shape['faces'].unsqueeze(0))
        
        self.init_render()
        self.kp_names = self.dl_img1.dataset.sdset.kp_names

        self.renderer_mask = NeuralRenderer(opts.img_size)
        self.hypo_mask_renderers = [NeuralRenderer(opts.img_size) for _ in range(opts.num_hypo_cams)]

        self.renderer_depth = NeuralRenderer(opts.img_size)
        self.hypo_depth_renderers = [NeuralRenderer(opts.img_size) for _ in range(opts.num_hypo_cams)]
        # self.render_mean_bird_with_uv()
        if opts.pose_dump:
            self.scale_trans_preds = {} ## iter, pair_id, pose_1, pose_2
            self.quat_preds = {} ## iter, pair_id, pose_1, pose_2
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

    # def init_dataset(self,):
    #     opts = self.opts
    #     self.dl_img1 = cub_data.cub_test_pair_dataloader(opts, 1)
    #     self.dl_img2 = cub_data.cub_test_pair_dataloader(opts, 2)
    #     self.resnet_transform = torchvision.transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225])

    #     if not opts.honest_mean_shape:
    #         mean_file_name = 'mean_shape.mat'
    #         if opts.cmr_mean_shape:
    #             mean_file_name = 'mean_cmr_shape.mat'
    #         mpath = osp.join(opts.cub_cache_dir, 'uv', mean_file_name)
    #     else:
    #         mpath = osp.join(opts.cub_cache_dir, '../shapenet/', 'bird3', 'mean_shape.mat')

    #     self.mean_shape = cub_parse.load_mean_shape(mpath, self.device)
    #     self.mean_shape_np = sio.loadmat(mpath)
    #     return

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
            if not opts.honest_mean_shape:
                mpath = osp.join(opts.p3d_cache_dir, 'uv', mean_file_name)
            else:
                if opts.p3d_class=='car':
                    mpath = osp.join(opts.p3d_cache_dir, '../shapenet/', 'car2', 'mean_shape.mat')
                else:
                    mpath = osp.join(opts.p3d_cache_dir, '../shapenet/', opts.p3d_class, 'mean_shape.mat')

        else:
            if not opts.honest_mean_shape:
                mean_file_name = 'mean_shape.mat'
                if opts.cmr_mean_shape:
                    mean_file_name = 'mean_cmr_shape.mat'
                mpath = osp.join(opts.cub_cache_dir, 'uv', mean_file_name)
            else:
                mpath = osp.join(opts.cub_cache_dir, '../shapenet/', 'bird2', 'mean_shape.mat')

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


        if opts.uv_to_3d_pred:
            verts_3d = self.model.uv23d_pred(self.verts_uv)
            codes_pred['verts_3d'] = verts_3d
            self.uv2points.set_3d_verts(verts_3d, self.verts_uv)
        else:
            verts_3d = self.mean_shape['verts']
            codes_pred['verts_3d'] = verts_3d
            codes_pred['def'] = 0*codes_pred['verts_3d'].unsqueeze(0)

        self.codes_gt['img'] = img
        self.codes_gt['mask'] = mask
        self.codes_gt['xy_map'] = torch.cat([self.grid[0:b_size, :, :, None,  0] *
                                             ratio, self.grid[0:b_size, :, :, None,  1]], dim=-1)

        points3d = geom_utils.project_uv_to_3d(
            self.uv2points, codes_pred['uv_map'])
        codes_pred['points_3d'] = points3d.view(
            b_size, self.upsample_img_size[0], self.upsample_img_size[1], 3)
        codes_pred['iter'] = 1
        if opts.cam_compute_ls:  # Computes a camera using a linear system.
            # This camera is predicted using Neural Network
            codes_pred['cam'] = codes_pred['cam']

        
        codes_pred['project_points_cam_pred'] = geom_utils.project_3d_to_image(points3d, codes_pred['cam'], self.offset_z)
        codes_pred['project_points_cam_z'] = (codes_pred['project_points_cam_pred'][...,2] - self.cam_location[2]).view(self.codes_gt['xy_map'][...,0].size())
        codes_pred['project_points_cam_pred'] = codes_pred['project_points_cam_pred'][..., 0:2].view(self.codes_gt['xy_map'].size())
        

        if opts.evaluate_all_hypotheses:
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


        if opts.render_mask:
            camera = self.codes_gt['cam'] if opts.use_gt_cam else codes_pred['cam']
            # camera = torch.cat([camera[:, 0:3], camera[:, 3:7].detach()], dim=-1)
            faces = self.mean_shape['faces']
            V = verts_3d[None, ...] + codes_pred['def']
            mask_pred = self.renderer_mask.forward(V, faces[None,...].repeat(b_size,1,1), camera)
            codes_pred['mask'] = mask_pred

            if opts.evaluate_all_hypotheses:
                codes_pred['mask_all_hypo'] = []
                cams_all_hypo = codes_pred['cam_hypotheses']
                for cx in range(cams_all_hypo.size(1)):
                    mask_pred = self.hypo_mask_renderers[cx].forward(V, faces[None,...].repeat(b_size,1,1), cams_all_hypo[:,cx])
                    codes_pred['mask_all_hypo'].append(mask_pred)
                codes_pred['mask_all_hypo'] = torch.stack(codes_pred['mask_all_hypo'], 1)

        if opts.render_depth:
            camera = self.codes_gt['cam'] if opts.use_gt_cam else codes_pred['cam']
            faces = self.mean_shape['faces']
            V = verts_3d[None, ...] + codes_pred['def']
            depth_pred = self.renderer_depth.forward(V, faces[None,...].repeat(b_size,1,1), camera, depth_only=True)
            codes_pred['depth'] = depth_pred
            if opts.evaluate_all_hypotheses:
                codes_pred['depth_all_hypo'] = []
                cams_all_hypo = codes_pred['cam_hypotheses']
                for cx in range(cams_all_hypo.size(1)):
                    depth_pred = self.hypo_depth_renderers[cx].forward(V, faces[None,...].repeat(b_size,1,1), cams_all_hypo[:,cx], depth_only=True)
                    codes_pred['depth_all_hypo'].append(depth_pred)
                codes_pred['depth_all_hypo'] = torch.stack(codes_pred['depth_all_hypo'], 1)

        if opts.use_gt_cam:
            project_points = geom_utils.project_3d_to_image(
                points3d, self.codes_gt['cam'], self.offset_z)[..., 0:2]
            codes_pred['project_points'] = project_points.view(
                self.codes_gt['xy_map'].size())
            codes_pred['cam'] = self.codes_gt['cam']
        else:
            codes_pred['project_points'] = codes_pred['project_points_cam_pred']

        codes_pred['xy_map'] = codes_pred['project_points']


        self.total_loss, self.loss_factors = loss_utils.code_loss(
            self.codes_gt, codes_pred, opts,  self.triangle_loss_fn)

        if opts.minimize_least and opts.evaluate_all_hypotheses:
            codes_pred['cam_sample_inds'] = self.loss_factors['cam_sample_inds'].unsqueeze(1)
            codes_pred['cam'] = torch.gather(codes_pred['cam_hypotheses'], 1, codes_pred['cam_sample_inds'].view(-1,1,1).repeat(1,1,7)).squeeze()
            mask_pred = self.renderer_mask.forward(V, faces[None,...].repeat(b_size,1,1), codes_pred['cam'])
            codes_pred['mask'] = mask_pred



        kps_vis = self.codes_gt['kps_vis']
        kps_uv = 0*self.codes_gt['kp'][:,:,0:2]
        kps_ind = self.codes_gt['kps_ind'].long()
        kps_ind_modified = 0*kps_ind
        uv_maps = codes_pred['uv_map']
        for bx in range(len(kps_vis)):
            for kx in range(len(kps_vis[bx])):
                rx = kps_ind[bx][kx][1]
                cx = kps_ind[bx][kx][0]
                kps_uv[bx, kx] = uv_maps[bx, rx, cx]

        kps_3d = self.uv2points.forward(kps_uv.view(-1, 2))
        kps_3d = kps_3d.view(kps_uv.size(0), kps_uv.size(1), 3)
        codes_pred['kps_3d'] = kps_3d
        
        self.codes_pred = codes_pred

        self.dump_predictions()
        if not opts.mask_dump:
            self.codes_pred['mask'] = self.populate_mask_from_file().squeeze()

        # camera = self.populate_pose_from_file()
        # self.codes_pred['cam'][:,3:7] = camera[:,3:7]
        return

    def dump_predictions(self,):
        opts = self.opts
        iter_index = "{:05}".format(self.iter_index)
        if opts.pose_dump:
            codes_pred = self.codes_pred
            camera = codes_pred['cam'].data.cpu().numpy()
            pose1 = {'scale_p1':camera[0,0] , 'trans_p1':camera[0,1:3]}
            pose2 = {'scale_p2':camera[1,0] , 'trans_p2':camera[1,1:3]}
            pose = pose1
            pose.update(pose2)
            pose['ind1'] = self.inds[0]
            pose['ind2'] = self.inds[1]
            self.scale_trans_preds[iter_index] = pose


            pose1 = {'quat_p1':camera[0,3:7]}
            pose2 = {'quat_p2':camera[1,3:7]}
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
        p1_q = quat['quat_p1'][0,0][0]
        p2_q = quat['quat_p2'][0,0][0]
        camera1 = np.concatenate([p1_s, p1_t, p1_q], axis=0)
        camera2 = np.concatenate([p2_s, p2_t, p2_q], axis=0)
        camera = np.stack([camera1, camera2], axis=0)
        return torch.from_numpy(camera.copy()).float().type(self.Tensor)

    def populate_mask_from_file(self,):
        iter_index = "{:05}".format(self.iter_index)
        masks = self.mask_preds[iter_index]
        mask1 = masks['mask_1'][0,0]
        mask2 = masks['mask_2'][0,0]
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



    def collect_results(self, ):
        codes_pred = self.codes_pred
        codes_gt = self.codes_gt

        # bwk_warp12 = self.compute_warp_robust(codes_pred['uv_map'][0], codes_pred['uv_map'][1], codes_pred['seg_mask'][0], codes_pred['seg_mask'][1])
        # bwk_warp21 = self.compute_warp_robust(codes_pred['uv_map'][1], codes_pred['uv_map'][0], codes_pred['seg_mask'][0], codes_pred['seg_mask'][1])
        
        # bwk_warp12 = bwk_warp21.numpy()
        # bwk_warp21 = bwk_warp21.numpy()
        kps  = codes_gt['kp'].data.cpu().numpy()
        uv_map = codes_pred['uv_map'].data.cpu().numpy()
        seg_mask = codes_pred['seg_mask'].data.cpu().numpy()
        gt_mask = codes_gt['mask'].data.cpu().numpy()
        cam = codes_pred['cam'].data.cpu().numpy()
        # gt_cam = codes_pred['gt_cam'].data.cpu().numpy()
        img = codes_gt['img']
        img = visutil.undo_resnet_preprocess(img)
        img = img.permute(0,2,3,1).data.cpu().numpy()

        tfs12, err12, tfs21, err21 = self. evaluate_m1() ## gets you the transfer points. Dump them and visualize later :)
        kps_ind = self.codes_gt['kps_ind'].data.cpu().numpy()
        # visutil.torch2numpy(transfer_kps12), visutil.torch2numpy(error_kps12), visutil.torch2numpy(transfer_kps21), visutil.torch2numpy(error_kps21)
        # result1 = {'kp': kps[0], 'uv_map':uv_map[0], 'seg_mask': seg_mask[0], 'gt_mask': gt_mask[0], 'cam':cam[0], 'img':img[0]}
        # result2 = {'kp': kps[1], 'uv_map':uv_map[1], 'seg_mask': seg_mask[1], 'gt_mask': gt_mask[1], 'cam':cam[1], 'img':img[1]}
        # results = {'bwk_warp12': bwk_warp12, 'bwk_warp21': bwk_warp21, 'kp': kps, 'uv_map':uv_map,
        #            'seg_mask': seg_mask, 'gt_mask': gt_mask, 'cam': cam, 'img':img}
        # pdb.set_trace()
        results = {'kp': kps, 'uv_map':uv_map, 'seg_mask': seg_mask, 'gt_mask': gt_mask, 'cam': cam,
                   'img':img, 'tfs12': tfs12, 'err12': err12,
                    'tfs21': tfs21, 'err21':err21, 'kps_ind1':kps_ind[0], 'kps_ind2':kps_ind[1]}
        
        results['ind1'] = self.inds[0]  
        results['ind2'] = self.inds[1]
        return results

    def compute_warp_robust(self, uv_map1, uv_map2, mask1, mask2):
        H,W = uv_map1.shape[0], uv_map2.shape[1]
        
        uv_map1_3d = geom_utils.project_uv_to_3d(self.uv2points, uv_map1[None,:,:,:]).view(H,W,3).data.cpu()
        uv_map2_3d = geom_utils.project_uv_to_3d(self.uv2points, uv_map2[None,:,:,:]).view(H,W,3).data.cpu()

        # uv_map2_3d = uv_map2_3d.view(-1, 3)
        # uv_map1_3d = uv_map1_3d.view(-1, 3)
        # uv_map1_3d = uv_map1_3d.reshape(H,W, 3)
        # uv_map2_3d = uv_map2_3d.reshape(H*W, 3)
        mask1 = (mask1 > 0.5).float().cpu()
        mask1_np = mask1.data.cpu().numpy()
        non_zero_inds = np.where(mask1_np.squeeze()>0.5)
        indices  = []
        warp = torch.zeros(H,W,2)
        
        for hx, wx in zip(non_zero_inds[0], non_zero_inds[1]):
            dist = torch.norm(uv_map1_3d[hx, wx, : ][None,None,:] - uv_map2_3d[:, :, :], dim=2)
            _,min_index = torch.min(dist.view(-1), 0)
            min_index = min_index.item()
            warp[hx,wx,0] = min_index - (min_index//W)*W ## x cord
            warp[hx,wx,1] = min_index//W ## y coord

        default_grid = (self.grid*0.5+ 0.5)*H
        default_grid = default_grid[0].cpu()
        warp = warp * mask1[0,:,:,None] + default_grid * (1-mask1[0,:,:,None])
        return warp



    def map_kp_img1_to_img2_robust(self, vis_inds, kps1, kps2, uv_map1, uv_map2, mask1, mask2):
        kp_mask = torch.zeros([len(kps1)]).cuda()
        kp_mask[vis_inds] = 1
        kps1 = kps1.long()

        img_H = uv_map2.size(0)
        img_W = uv_map2.size(1)
        kps1_uv = uv_map1[kps1[:, 1], kps1[:, 0], :]

        if False:
            distances3d = geom_utils.compute_distance_in_uv_sapce(kps1_uv.view(-1,2), uv_map2.view(-1, 2))
        
        if True:
            kps1_3d = geom_utils.project_uv_to_3d(self.uv2points, kps1_uv[None,None,:,:])
            uv_points3d = geom_utils.project_uv_to_3d(self.uv2points, uv_map2[None,:,:,:])

            # kps1_3d = self.uv2points.forward()
            # uv_map2_3d = self.uv2points.forward()
            distances3d = torch.sum((kps1_3d.view(-1,1,3)  - uv_points3d.view(1,-1, 3))**2, -1).sqrt()
        
            distances3d = distances3d + (1-mask2.view(1,-1))*1000
            distances = distances3d

        if False:
            distances = torch.exp(-10*distances)
            distances = torch.softmax(distances, dim=1)
            assert torch.sum(distances.sum(1) > 0.99).item() == len(distances), 'we might be hitting a few nans due to high exponent, {}'.format(self.iter_index)
            
            grid_default = self.grid[0].view(1, -1,2)
            pixels_indices = distances[:,:,None] * grid_default

            pixels_indices = pixels_indices.sum(1)

            transfer_kps =  (pixels_indices * 0.5 + 0.5)*img_H

        if True:
            min_dist, min_indices = torch.min(distances.view(len(kps1), -1), dim=1)
            transfer_kps = torch.stack(
            [min_indices % img_W, min_indices // img_W], dim=1)

        # transfer_kps = torch.stack(
        #     [min_indices % img_W, min_indices // img_W], dim=1)
        kp_transfer_error = torch.norm(
            kp_mask[:, None] * (transfer_kps.float() - kps2[:, 0:2]), dim=1)
        return transfer_kps, torch.stack([kp_transfer_error, kp_mask], dim=1)


    '''
    There 15 possible keypoints on every birds. kp_uv_locations  15 x 2
    '''

    def map_kp_img1_to_img2(self, vis_inds, kps1, kps2, uv_map1, uv_map2, mask1, mask2):
        kp_mask = torch.zeros([len(kps1)]).cuda()
        kp_mask[vis_inds] = 1
        kps1 = kps1.long()
        # pdb.set_trace()
        uv_map1 = uv_map1 + 10000 * (1-mask1).permute(1,2,0)
        uv_map2 = uv_map2 + 10000 * (1-mask2).permute(1,2,0)
        
        kps1_uv = uv_map1[kps1[:, 1], kps1[:, 0], :]

        # Now find the nearest locations in uv_map2
        distances = (uv_map2[None, :, :, :] - kps1_uv[:, None, None, :])**2
        distances = distances.sum(-1)
        min_dist, min_indices = torch.min(distances.view(len(kps1), -1), dim=1)

        img_H = uv_map2.size(0)
        img_W = uv_map2.size(1)
        transfer_kps = torch.stack(
            [min_indices % img_W, min_indices // img_W], dim=1)
        kp_transfer_error = torch.norm(
            kp_mask[:, None] * (transfer_kps.float() - kps2[:, 0:2]), dim=1)
        return transfer_kps, torch.stack([kp_transfer_error, kp_mask], dim=1)


    def evaluate_m1_via_shape(self,):
        # Collect keypoints that are visible in both the images. Take keypoints
        # from one image --> Keypoints in second image.
        common_kp_indices = torch.nonzero(
            self.codes_gt['kp'][0, :, 2] * self.codes_gt['kp'][1, :, 2] > 0.5)
        kps_ind = self.codes_gt['kps_ind']
        kps = self.codes_gt['kp'] ## -1 to 1 
        uv_map = self.codes_pred['uv_map']
        kps3d = self.codes_pred['kps_3d']
        mask = (self.codes_pred['seg_mask'] > 0.5).float().squeeze()
        camera = self.codes_pred['cam']
      
        transfer_kps12, error_kps12 = self.map_kp_img1_to_img2_via_shape(
            common_kp_indices, kps3d[0], kps_ind[1], mask[1], camera[1])
        transfer_kps21, error_kps21 = self.map_kp_img1_to_img2_via_shape(
            common_kp_indices, kps3d[1], kps_ind[0], mask[0], camera[0])
        return visutil.torch2numpy(transfer_kps12), visutil.torch2numpy(error_kps12), visutil.torch2numpy(transfer_kps21), visutil.torch2numpy(error_kps21)

    def evaluate_m1(self,):
        # Collect keypoints that are visible in both the images. Take keypoints
        # from one image --> Keypoints in second image.
        common_kp_indices = torch.nonzero(
            self.codes_gt['kp'][0, :, 2] * self.codes_gt['kp'][1, :, 2] > 0.5)
        kps_ind = self.codes_gt['kps_ind']
        kps = self.codes_gt['kp'] ## -1 to 1 
        uv_map = self.codes_pred['uv_map']
        # mask = self.codes_gt['mask']
        mask = (self.codes_pred['seg_mask'] > 0.5).float()
        if self.opts.robust:
            transfer_kps12, error_kps12 = self.map_kp_img1_to_img2_robust(
                common_kp_indices, kps_ind[0], kps_ind[1], uv_map[0], uv_map[1], mask[0], mask[1])
            transfer_kps21, error_kps21 = self.map_kp_img1_to_img2_robust(
            common_kp_indices, kps_ind[1], kps_ind[0], uv_map[1], uv_map[0], mask[1], mask[0])
        else:
            transfer_kps12, error_kps12 = self.map_kp_img1_to_img2(
                common_kp_indices, kps_ind[0], kps_ind[1], uv_map[0], uv_map[1], mask[0], mask[1])
            transfer_kps21, error_kps21 = self.map_kp_img1_to_img2(
            common_kp_indices, kps_ind[1], kps_ind[0], uv_map[1], uv_map[0], mask[1], mask[0])
        return visutil.torch2numpy(transfer_kps12), visutil.torch2numpy(error_kps12), visutil.torch2numpy(transfer_kps21), visutil.torch2numpy(error_kps21)


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
            bird_vis.save_obj_with_texture('{:04}'.format(self.inds[b]), results_dir, visuals['texture_img'], self.mean_shape_np)
            
            # bird_vis.save_obj_with_texture('{:04}'.format(self.inds[b]),
            # results_dir, visuals['texture_img'], self.mean_shape_np)
       
        return batch_visuals

    def steal_colors(self, upsample_texture=True):
        
        visdom_renderer = self.visdom_renderer
        visuals = {}
        codes_pred = self.codes_pred
        img1 = self.codes_gt['img'][0]
        img2 = self.codes_gt['img'][1]
        img1 = img1.unsqueeze(0)
        img1 = visutil.undo_resnet_preprocess(img1).squeeze()
        img2 = img2.unsqueeze(0)
        img2 = visutil.undo_resnet_preprocess(img2).squeeze()
        visuals['tfs_a_img1'] = visutil.tensor2im(img1.unsqueeze(0))
        visuals['tfs_d_img2'] = visutil.tensor2im(img2.unsqueeze(0))


        mask1 = self.codes_gt['mask'][0]
        mask2 = self.codes_gt['mask'][1]

        uv_map1 = codes_pred['uv_map'][0]
        uv_map2 = codes_pred['uv_map'][1]

        if upsample_texture:
            img1, mask1, uv_map1 = bird_vis.upsample_img_mask_uv_map(img1.squeeze(0), mask1, uv_map1)
            img2, mask2, uv_map2 = bird_vis.upsample_img_mask_uv_map(img2.squeeze(0), mask2, uv_map2)


        img1_np = visutil.torch2numpy(img1)
        img2_np = visutil.torch2numpy(img2)

        uv_map1_np = visutil.torch2numpy(uv_map1)
        uv_map2_np = visutil.torch2numpy(uv_map2)
        texture1 = bird_vis.create_texture_image_from_uv_map(
            256, 256, uv_map1_np, img1_np, mask1.cpu().numpy())
        texture2 = bird_vis.create_texture_image_from_uv_map(
            256, 256, uv_map2_np, img2_np, mask2.cpu().numpy())

        texture1 = torch.from_numpy(texture1).float().cuda()
        texture2 = torch.from_numpy(texture2).float().cuda()

        tsf2to1 = bird_vis.copy_texture_using_uvmap(mask1, texture2, uv_map1)
        tsf1to2 = bird_vis.copy_texture_using_uvmap(mask2, texture1, uv_map2)
        tsf1to1 = bird_vis.copy_texture_using_uvmap(mask1, texture1, uv_map1)
        tsf2to2 = bird_vis.copy_texture_using_uvmap(mask2, texture2, uv_map2)
        visuals = {}
        # visuals['tfs_1to1'] = tsf1to1
        visuals['tfs_b_1to2'] = tsf1to2
        visuals['tfs_c_2to1'] = tsf2to1
        # visuals['tfs_2to2'] = tsf2to2

        visuals['tfs_a_img1'] = visutil.tensor2im(img1.unsqueeze(0))
        visuals['tfs_d_img2'] = visutil.tensor2im(img2.unsqueeze(0))

        return visuals

    def test(self,):
        opts = self.opts
        bench_stats_m1 = {'transfer': [], 'kps_err': [], 'pair': [], }
        bench_stats_m2 = {'transfer': [], 'kps_err': [], 'pair': [], }
        
        n_iter = opts.max_eval_iter if opts.max_eval_iter > 0 else len(
            self.dl_img1)
        result_path = osp.join(
            opts.results_dir, 'vis_dump{}.mat'.format(n_iter))
        print('Writing to %s' % result_path)
        self.visualizer = Visualizer(opts)
        visualizer = self.visualizer
        bench_stats = {}
        self.iter_index = None
        self.database = {opts.dataset:[]}
        self.database['mean_shape'] = self.mean_shape
        
        
        if not osp.exists(result_path) or opts.force_run:
            # n_iter = len(self.dl_img1)
            from itertools import izip
            for i, batch in enumerate(izip(self.dl_img1, self.dl_img2)):
                self.iter_index = i
                if i % 100 == 0:
                    print('{}/{} evaluation iterations.'.format(i, n_iter))
                if opts.max_eval_iter > 0 and (i >= opts.max_eval_iter):
                    break
                self.set_input(batch)
                self.predict()
                collected_data = self.collect_results()

                self.database[opts.dataset].append(collected_data)

        sio.savemat(result_path, self.database)
        # pdb.set_trace()

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
    opts.batch_size = 1
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
