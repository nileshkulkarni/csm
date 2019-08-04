'''
Loss building blocks.
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
from absl import flags
import numpy as np
import pdb
import itertools
from . import geom_utils
from ..utils import visutil

flags.DEFINE_float('reproject_loss_wt', 1, 'Reprojection loss bw uv and  loss weight.')
flags.DEFINE_float('mask_loss_wt', 1.0, 'Mask loss wt.')
flags.DEFINE_float('seg_mask_loss_wt', 1.0, 'Predicted Seg Mask loss wt.')
flags.DEFINE_float('depth_loss_wt', 1.0, 'Depth loss wt.')
flags.DEFINE_float('reg_loss_wt', 10, 'Regularzation Def loss wt.')
flags.DEFINE_float('ent_loss_wt', 0.05, 'Cam loss wt.')
flags.DEFINE_float('reg_rotation_wt', 1, 'Margin for contrastive hinge loss')
flags.DEFINE_boolean('ignore_mask_gcc', False, 'ignore mask in reproject loss')
flags.DEFINE_boolean('ignore_mask_vis', False, 'ignore mask in visbility loss')
flags.DEFINE_float('warmup_pose_iter', 0, 'Warm up iter for pose prediction')

'''
    feat1 : B x H x W x feat_dim
    feat2 : B x H x W x feat_dim
    mask  : B x H x W
'''

NC2_perm = list(itertools.permutations(range(8), 2))
NC2_perm =  torch.LongTensor(zip(*NC2_perm)).cuda()

def depth_loss_fn(depth_render, depth_pred, mask):
    loss = torch.nn.functional.relu(depth_pred-depth_render).pow(2) * mask
    loss = loss.view(loss.size(0), -1).mean(-1)
    return loss

def depth_loss_fn_vis(depth_render, depth_pred, mask):
    loss = torch.nn.functional.relu(depth_pred-depth_render)* mask
    return loss

def mask_loss_fn(mask_pred, mask_gt):
    loss = torch.nn.functional.mse_loss(mask_pred, mask_gt, reduce=False)
    loss = loss.view(loss.size(0), -1).mean(-1)
    return loss



def reproject_loss_l2(project_points, grid_points, mask):
    non_mask_points = mask.view(mask.size(0), -1).mean(1)
    mask = mask.unsqueeze(-1).repeat(1, 1, 1, project_points.size(-1))
    loss = (mask * project_points - mask * grid_points).pow(2).sum(-1).view(mask.size(0), -1).mean(1)
    loss = loss / (non_mask_points + 1E-10)
    return loss

def code_loss(codes_gt, codes_pred, opts, laplacian_loss_fn=None):
    total_loss = []
    loss_factors = {}
    seg_mask = codes_gt['mask'].squeeze()

    warmup_pose = False
    if opts.warmup_pose_iter > codes_pred['iter']:
        ## only train the pose predictor. Without training the probs.
        warmup_pose = True
    
    # Reprojection Loss
    project_points = codes_pred['project_points']
    if opts.ignore_mask_gcc:
        reproject_loss = reproject_loss_l2(project_points, codes_gt['xy_map'], seg_mask*0+1)
    else:
        reproject_loss = reproject_loss_l2(project_points, codes_gt['xy_map'], seg_mask)

    reproject_loss_pred_cam = reproject_loss_l2(codes_pred['project_points_cam_pred'], codes_gt['xy_map'], seg_mask)
    reproject_loss = reproject_loss.mean()
    reproject_loss_pred_cam = reproject_loss_pred_cam.mean()

    if opts.multiple_cam_hypo:
        reproject_loss_all_hypo = []
        for cx in range(codes_pred['cam_hypotheses'].size(1)):
            if opts.ignore_mask_gcc:
                reproject_loss_all_hypo.append(reproject_loss_l2(
                    codes_pred['project_points_all_hypo'][:, cx], codes_gt['xy_map'], seg_mask*0 + 1))
            else:
                reproject_loss_all_hypo.append(reproject_loss_l2(
                    codes_pred['project_points_all_hypo'][:, cx], codes_gt['xy_map'], seg_mask))

        reproject_loss_all_hypo = torch.stack(reproject_loss_all_hypo, dim=1)
        cam_probs = codes_pred['cam_probs']
        # if not opts.attach_uv_to_graph:
        #     cam_probs = cam_probs.data.clone()
        #     reproject_loss_all_hypo = reproject_loss_all_hypo.data.clone()

        reproject_loss_all_hypo = reproject_loss_all_hypo * cam_probs
        reproject_loss = reproject_loss_all_hypo.sum(1).mean()
        if warmup_pose:
            reproject_loss = 0*reproject_loss



    if opts.multiple_cam_hypo: ## regularize rotation.
        quats = codes_pred['cam_hypotheses'][:,:,3:7]
        quats_x = torch.gather(quats, dim=1, index=NC2_perm[0].view(1,-1,1).repeat(len(quats), 1, 4))
        quats_y = torch.gather(quats, dim=1, index=NC2_perm[1].view(1,-1,1).repeat(len(quats), 1, 4))
        inter_quats = geom_utils.hamilton_product(quats_x, geom_utils.quat_conj(quats_y))
        quatAng = geom_utils.quat2ang(inter_quats).view(len(inter_quats), opts.num_hypo_cams-1, -1)
        quatAng = -1*torch.nn.functional.max_pool1d(-1*quatAng.permute(0,2,1), opts.num_hypo_cams-1, stride=1).squeeze()
        loss_factors['rot_mag'] = opts.reg_rotation_wt * (np.pi- quatAng).mean()
        if warmup_pose and opts.az_ele_quat:
            loss_factors['rot_mag'] = 0*loss_factors['rot_mag']
            
        total_loss.append(loss_factors['rot_mag'])

    if opts.multiple_cam_hypo:  ## use entropy
        dist_entropy = -1*(-torch.log(codes_pred['cam_probs'] + 1E-9)*codes_pred['cam_probs']).sum(1).mean()
        if warmup_pose:
            dist_entropy = 0*dist_entropy

        total_loss.append(opts.ent_loss_wt*dist_entropy)
        loss_factors['dist_entropy'] = dist_entropy * opts.ent_loss_wt


    if opts.pred_mask:
        seg_mask_loss = torch.nn.functional.binary_cross_entropy(codes_pred['seg_mask'], codes_gt['mask'])
        loss_factors['seg_mask'] = opts.seg_mask_loss_wt*seg_mask_loss
        total_loss.append(loss_factors['seg_mask'])


    img_size = (codes_gt['xy_map'].size(0), codes_gt['xy_map'].size(1))
    triangle_loss = torch.zeros(1).mean().cuda()
    mask_loss = torch.zeros(1).mean().cuda()

    if opts.render_mask:
        render_mask_loss_fn = mask_loss_fn
        if opts.multiple_cam_hypo:
            mask_loss_all_hypo = []
            for cx in range(codes_pred['cam_hypotheses'].size(1)):
                mask_loss_all_hypo.append(render_mask_loss_fn(codes_pred['mask_all_hypo'][
                                          :, cx].unsqueeze(1), codes_gt['mask']))
            mask_loss_all_hypo = torch.stack(mask_loss_all_hypo, dim=1)
            cam_probs = codes_pred['cam_probs']
            if warmup_pose:
                cam_probs = cam_probs*0 + 1./cam_probs.shape[1]

            mask_loss = mask_loss_all_hypo * cam_probs
            mask_loss = mask_loss.sum(1).mean()
        else:
            mask_loss = render_mask_loss_fn(codes_pred['mask'].unsqueeze(1), codes_gt['mask'])
            pseudo_loss += mask_loss
            mask_loss = mask_loss.mean()

    depth_loss = torch.zeros(1).mean().cuda()
    depth_loss_all_hypo = torch.zeros(1).mean().cuda()
    depth_loss_all_hypo_vis = []

    if opts.render_depth:
        if opts.multiple_cam_hypo:
            depth_loss_all_hypo = []
            for cx in range(codes_pred['cam_hypotheses'].size(1)):

                actual_depth_at_pixels = torch.nn.functional.grid_sample(codes_pred['depth_all_hypo'][:,None,cx], 
                                                                         codes_pred['project_points_all_hypo'][:, cx].detach())
                actual_depth_at_pixels = actual_depth_at_pixels.squeeze()

                if opts.ignore_mask_vis:
                    depth_loss_all_hypo.append(depth_loss_fn(actual_depth_at_pixels,
                                                             codes_pred['project_points_z_all_hypo'][:,cx],
                                                             codes_gt['mask'][:,0]*0 + 1))
                    depth_loss_all_hypo_vis.append(
                        depth_loss_fn_vis(actual_depth_at_pixels,
                                        codes_pred['project_points_z_all_hypo'][:,cx],
                                        codes_gt['mask'][:,0]*0 + 1))
                else:
                    depth_loss_all_hypo.append(depth_loss_fn(actual_depth_at_pixels,
                                                             codes_pred['project_points_z_all_hypo'][:,cx],
                                                             codes_gt['mask'][:,0]))

                    depth_loss_all_hypo_vis.append(
                        depth_loss_fn_vis(actual_depth_at_pixels,
                                        codes_pred['project_points_z_all_hypo'][:,cx],
                                        codes_gt['mask'][:,0]))

            depth_loss_all_hypo = torch.stack(depth_loss_all_hypo, dim=1)
            depth_loss_all_hypo_vis = torch.stack(depth_loss_all_hypo_vis, dim=1)
            cam_probs = codes_pred['cam_probs']
            if warmup_pose:
                depth_loss_all_hypo = depth_loss_all_hypo*0 + 0

            depth_loss = depth_loss_all_hypo * cam_probs
            depth_loss = depth_loss.sum(1).mean()
            loss_factors['depth_loss_all_hypo'] = depth_loss_all_hypo
            loss_factors['depth_loss_all_hypo_vis'] = depth_loss_all_hypo_vis
        else:
            actual_depth_at_pixels = torch.nn.functional.grid_sample(codes_pred['depth'][:,None,...], codes_pred['project_points'].detach())
            actual_depth_at_pixels = actual_depth_at_pixels.squeeze()
            if opts.ignore_mask_vis:
                depth_loss = depth_loss_fn(actual_depth_at_pixels, codes_pred['project_points_cam_z'], codes_gt['mask'][:,0]*0 + 1)
            else:
                depth_loss = depth_loss_fn(actual_depth_at_pixels, codes_pred['project_points_cam_z'], codes_gt['mask'][:,0])
            depth_loss = depth_loss.mean()

   
    total_loss.append(reproject_loss * opts.reproject_loss_wt)
    total_loss.append(mask_loss * opts.mask_loss_wt)
    total_loss.append(depth_loss * opts.depth_loss_wt)

    loss_factors.update({
                        'reproject': reproject_loss * opts.reproject_loss_wt,
                        'render_mask': mask_loss * opts.mask_loss_wt,
                        'depth' : depth_loss * opts.depth_loss_wt
                        })

    total_loss = torch.stack(total_loss).sum()
    return total_loss, loss_factors
