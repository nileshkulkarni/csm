"""
Code borrowed from 
https://github.com/akanazawa/cmr/blob/master/utils/bird_vis.py
Visualization helpers specific to birds.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
import numpy as np
import os.path as osp
import cv2
import pdb
from . import cub_parse
from ..nnutils.nmr import NeuralRenderer
from ..utils import transformations
from . import visutil
import pdb

class VisRenderer(object):
    """
    Utility to render meshes using pytorch NMR
    faces are F x 3 or 1 x F x 3 numpy
    """

    def __init__(self, img_size, faces, t_size=3):
        self.renderer = NeuralRenderer(img_size)
        self.faces = Variable(
            torch.IntTensor(faces).cuda(), requires_grad=False)
        if self.faces.dim() == 2:
            self.faces = torch.unsqueeze(self.faces, 0)
        default_tex = np.ones((1, self.faces.shape[1], t_size, t_size, t_size,
                               3))
        blue = np.array([156, 199, 234.]) / 255.
        default_tex = default_tex * blue
        # Could make each triangle different color
        self.default_tex = Variable(
            torch.FloatTensor(default_tex).cuda(), requires_grad=False)
        # rot = transformations.quaternion_about_axis(np.pi/8, [1, 0, 0])
        # This is median quaternion from sfm_pose
        # rot = np.array([ 0.66553962,  0.31033762, -0.02249813,  0.01267084])
        # This is the side view:
        import cv2
        R0 = cv2.Rodrigues(np.array([np.pi / 3, 0, 0]))[0]
        R1 = cv2.Rodrigues(np.array([0, np.pi / 2, 0]))[0]
        R = R1.dot(R0)
        R = np.vstack((np.hstack((R, np.zeros((3, 1)))), np.array([0, 0, 0,
                                                                   1])))
        rot = transformations.quaternion_from_matrix(R, isprecise=True)
        cam = np.hstack([0.75, 0, 0, rot])
        self.default_cam = Variable(
            torch.FloatTensor(cam).cuda(), requires_grad=False)
        self.default_cam = torch.unsqueeze(self.default_cam, 0)

    def __call__(self, verts, cams=None, texture=None, rend_mask=False):
        """
        verts is |V| x 3 cuda torch Variable
        cams is 7, cuda torch Variable
        Returns N x N x 3 numpy
        """
        if texture is None:
            texture = self.default_tex
        elif texture.dim() == 5:
            # Here input it F x T x T x T x 3 (instead of F x T x T x 3)
            # So add batch dim.
            texture = torch.unsqueeze(texture, 0)
        if cams is None:
            cams = self.default_cam
        elif cams.dim() == 1:
            cams = torch.unsqueeze(cams, 0)

        if verts.dim() == 2:
            verts = torch.unsqueeze(verts, 0)

        verts = asVariable(verts)
        cams = asVariable(cams)
        texture = asVariable(texture)

        if rend_mask:
            rend = self.renderer.forward(verts, self.faces, cams)
            rend = rend.repeat(3, 1, 1)
            rend = rend.unsqueeze(0)
        else:
            rend = self.renderer.forward(verts, self.faces, cams, texture)

        rend = rend.data.cpu().numpy()[0].transpose((1, 2, 0))
        rend = np.clip(rend, 0, 1) * 255.0

        return rend.astype(np.uint8)

    def rotated(self, vert, deg, axis=[0, 1, 0], cam=None, texture=None):
        """
        vert is N x 3, torch FloatTensor (or Variable)
        """
        import cv2
        new_rot = cv2.Rodrigues(np.deg2rad(deg) * np.array(axis))[0]
        new_rot = convert_as(torch.FloatTensor(new_rot), vert)

        center = vert.mean(0)
        new_vert = torch.t(torch.matmul(new_rot,
                                        torch.t(vert - center))) + center
        # new_vert = torch.matmul(vert - center, new_rot) + center

        return self.__call__(new_vert, cams=cam, texture=texture)

    def diff_vp(self,
                verts,
                cam=None,
                angle=90,
                axis=[1, 0, 0],
                texture=None,
                kp_verts=None,
                new_ext=None,
                extra_elev=False):
        if cam is None:
            cam = self.default_cam[0]
        if new_ext is None:
            new_ext = [0.6, 0, 0]
        # Cam is 7D: [s, tx, ty, rot]
        import cv2
        cam = asVariable(cam)
        quat = cam[-4:].view(1, 1, -1)
        R = transformations.quaternion_matrix(
            quat.squeeze().data.cpu().numpy())[:3, :3]
        rad_angle = np.deg2rad(angle)
        rotate_by = cv2.Rodrigues(rad_angle * np.array(axis))[0]
        # new_R = R.dot(rotate_by)

        new_R = rotate_by.dot(R)
        if extra_elev:
            # Left multiply the camera by 30deg on X.
            R_elev = cv2.Rodrigues(np.array([np.pi / 9, 0, 0]))[0]
            new_R = R_elev.dot(new_R)
        # Make homogeneous
        new_R = np.vstack(
            [np.hstack((new_R, np.zeros((3, 1)))),
             np.array([0, 0, 0, 1])])
        new_quat = transformations.quaternion_from_matrix(
            new_R, isprecise=True)
        new_quat = Variable(torch.Tensor(new_quat).cuda(), requires_grad=False)
        # new_cam = torch.cat([cam[:-4], new_quat], 0)
        new_ext = Variable(torch.Tensor(new_ext).cuda(), requires_grad=False)
        new_cam = torch.cat([new_ext, new_quat], 0)

        rend_img = self.__call__(verts, cams=new_cam, texture=texture)
        if kp_verts is None:
            return rend_img
        else:
            kps = self.renderer.project_points(
                kp_verts.unsqueeze(0), new_cam.unsqueeze(0))
            kps = kps[0].data.cpu().numpy()
            return kp2im(kps, rend_img, radius=1)

    def set_bgcolor(self, color):
        self.renderer.set_bgcolor(color)

    def set_light_dir(self, direction, int_dir=0.8, int_amb=0.8):

        renderer = self.renderer.renderer.renderer
        renderer.light_direction = direction
        renderer.light_intensity_directional = int_dir
        renderer.light_intensity_ambient = int_amb

    def set_light_status(self, use_lights):
        renderer = self.renderer.renderer.renderer
        renderer.use_lights = use_lights
        return



def draw_keypoint_on_image(img, keypoints, keypoints_vis, color_map=None):
    img = img.copy()
    for kpx, (keypoint, vis) in enumerate(zip(keypoints,keypoints_vis)):
        if vis > 0:
            color  = (0,0,255)
            if color_map is not None:
                color = color_map[kpx]
            img = cv2.circle(img,(keypoint[0],keypoint[1]), 5, (color[0]*255,color[1]*255,color[2]*255), -1)
    return img



def write_on_image(img, text, location): ## location is x,y
    img = img.copy()
    color  = (0,0,255)
    img = cv2.putText(img,"{}".format(text), (location[0],location[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2)
    return img

def draw_keypoint_and_text_on_image(img, keypoints, keypoints_vis, color_map=None, text=None, text_col=None):
    img = img.copy()

    for kpx, (keypoint, vis) in enumerate(zip(keypoints,keypoints_vis)):
        if vis > 0:
            color  = (0,0,255)
            if color_map is not None:
                color = color_map[kpx]
            img = cv2.circle(img,(keypoint[0],keypoint[1]), 5, (color[0]*255,color[1]*255,color[2]*255), -1)
            color = (0,0,255)
            if text_col is not None:
                color = text_col[kpx]
            if text is not None:
                img = cv2.putText(img,"{}".format(text[kpx]), (keypoint[0],keypoint[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=1)

    return img



def save_obj_with_texture(name, results_path, texture_img, mean_shape):
    visutil.mkdir(results_path)
    verts  = np.round(mean_shape['verts'],5)
    uv_verts  = np.round(mean_shape['uv_verts'],5)
    faces = mean_shape['faces']
    obj_file = osp.join(results_path, '{}.obj'.format(name))
    with open(obj_file,'w') as f:
        f.write('mtllib {}.mtl\n'.format(name))

        ## write vertices
        for v in verts:
            f.write("v {} {} {}\n".format(v[0], v[1], v[2]))

        ## write texture vertices
        for v in uv_verts:
            f.write("vt {} {}\n".format(v[0], 1 - v[1]))

        f.write('usemtl bird\n')
        f.write('s 1\n')
        
        ## write faces
        faces = faces + 1
        for fv in faces:
            f.write('f {}/{}/ {}/{}/ {}/{}/ \n'.format(fv[0], fv[0], fv[1], fv[1], fv[2], fv[2]))
        # for fv in faces:
        #     f.write('f {}// {}// {}// \n'.format(fv[0]+1, fv[1]+1, fv[2]+1))


    ## mtl file
    mtl_file = osp.join(results_path, '{}.mtl'.format(name))
    with open(mtl_file,'w') as f:
        f.write('# Material Count: 1\n')
        f.write('newmtl bird\n')
        f.write('Ns 96.078431\n')
        f.write('Ka 1.000000 1.000000 1.000000\n')
        f.write('Kd 0.640000 0.640000 0.640000\n')
        f.write('Ks 0.500000 0.500000 0.500000\n')
        f.write('Ke 0.000000 0.000000 0.000000\n')
        f.write('Ni 1.00000\n')
        f.write('d 1.000000\n')
        f.write('illum 2\n')
        f.write('map_Kd {}.png\n'.format(name,))

    ## Dump the texture image
    # texture_img[:,:,0],  texture_img[:,:,2] = texture_img[:,:,2], texture_img[:,:,0]
    # texture_img = texture_img[:,:,[2,1,0]]
    # pdb.set_trace()
    visutil.save_image(texture_img, osp.join(results_path,'{}.png'.format(name)))
    return

def merge_textures(foreground, background,):
    '''
    3, H, W
    Assume foreground to have 1 in the A channel and 0 for the background.
    '''
    texture = foreground * (foreground[3,None,...] > 0.5) + background * (foreground[3,None,...] <0.5)
    return texture


def render_transfer_kps_imgs(keypoint_cmap, img1, img2, kps1, kps2, transfer_kps12, transfer_kps21, common_kps):
    visuals = {}
    common_vis = kps1[:,0]*0
    common_vis[common_kps] = 1

    img1_tfs = draw_keypoint_on_image(img1, kps1, 
        common_vis, keypoint_cmap)
    img2_tfs = draw_keypoint_on_image(img2, kps2, 
        common_vis, keypoint_cmap)

    img_tfs12 = draw_keypoint_on_image(img2, transfer_kps12, 
        common_vis, keypoint_cmap)
    img_tfs21 = draw_keypoint_on_image(img1, transfer_kps21, 
        common_vis, keypoint_cmap)

    visuals['tfs_a_img1'] = img1_tfs
    visuals['tfs_d_img2'] = img2_tfs
    visuals['tfs_b_1to2'] = img_tfs12
    visuals['tfs_c_2to1'] = img_tfs21

    return  visuals


def create_monocolor_texture(uvimg_H, uvimg_W, color=None):
    if color is None:
        color = [156, 199, 234., 0]
    default_tex = np.ones((uvimg_H, uvimg_W,4))
    blue = np.array(color) / 255.
    blue[3] = color[3]
    default_tex = default_tex * blue.reshape(1,1,-1)
    return  default_tex.transpose(2,0,1)

def create_kp_heat_map_texture(uvimg_H, uvimg_W,  u_cord=None, v_cord=None, color=None, transprancy=1):
    default_tex = create_monocolor_texture(uvimg_H, uvimg_W)
    if color is None:
        color = (1,0,0)
    box_size = 3
    if v_cord is not None and u_cord is not None:
        default_tex = default_tex*0
        default_tex[0, v_cord-box_size:v_cord+box_size, u_cord-box_size:u_cord+box_size] = color[0] 
        default_tex[1, v_cord-box_size:v_cord+box_size, u_cord-box_size:u_cord+box_size] = color[1]
        default_tex[2, v_cord-box_size:v_cord+box_size, u_cord-box_size:u_cord+box_size] = color[2]
        default_tex[3, v_cord-box_size:v_cord+box_size, u_cord-box_size:u_cord+box_size] = transprancy

    return default_tex

def upsample_img_mask_uv_map(img, mask, uv_map):
    uv_map = torch.nn.functional.upsample(uv_map.permute(2,0,1).unsqueeze(0), scale_factor=4, mode='bilinear')
    mask = torch.nn.functional.upsample(mask.unsqueeze(0), scale_factor=4, mode='nearest').squeeze(0)
    img = torch.nn.functional.upsample(img.unsqueeze(0), scale_factor=4, mode='nearest').squeeze(0)
    uv_map = uv_map.squeeze().permute(1,2,0)
    return img, mask, uv_map


'''
Can handle batch of images
uv_map = B x Uh x Vw x 2
mask = B x H x W x 1
'''
def batch_create_texture_mask_from_uv_map(uvimg_H, uvimg_W, uv_maps, masks, upsample_texture=True):
    bsize = len(uv_maps)
    uv_map_masks = []
    for i in range(bsize):
        img = masks[i]
        uv_map = uv_maps[i]
        mask = masks[i]
        if upsample_texture:
            _, mask, uv_map = upsample_img_mask_uv_map(img, mask, uv_map)
        texture_mask = torch_texture_mask_from_uv_map(uvimg_H, uvimg_W,  uv_map, mask)
        uv_map_masks.append(texture_mask)

    uv_map_masks = torch.stack(uv_map_masks)
    return uv_map_masks

'''
Cannot handle batch of images. Inputs are torch tensors.
'''
def torch_texture_mask_from_uv_map(uvimg_H, uvimg_W, uv_map, mask):
    count_tex = torch.zeros((uvimg_H, uvimg_W), dtype =uv_map.dtype).type(uv_map.type())
    uv_map_inds = uv_map.clone()
    uv_map_inds[:,:,0] = torch.clamp((uv_map[:,:,0] * uvimg_W).round(), 0, uvimg_W-1)
    uv_map_inds[:,:,1] = torch.clamp((uv_map[:,:,1] * uvimg_H).round(), 0, uvimg_H-1)
    uv_map_inds = uv_map_inds.long()

    non_zero_inds = torch.nonzero(mask.squeeze())
    uv_inds = uv_map_inds[non_zero_inds[:,0], non_zero_inds[:,1]]
    # pdb.set_trace()
    # for index in non_zero_inds:
    #     rx = index[0].item()
    #     cx = index[1].item()
    #     u_ind = uv_map_inds[rx, cx, 0]
    #     v_ind = uv_map_inds[rx, cx, 1]
    #     count_tex[v_ind, u_ind, 0] += 1
    count_tex[uv_inds[:,1], uv_inds[:,0]] = 1.0
    return count_tex.unsqueeze(0)

def create_texture_image_from_uv_map(uvimg_H, uvimg_W, uv_map, img, mask):
    default_tex = np.ones((uvimg_H, uvimg_W,3))
    blue = np.array([156, 199, 234.]) / 255.
    default_tex = default_tex * blue.reshape(1,1,-1)
    count_tex = np.zeros((uvimg_H, uvimg_W,1))
    uv_map_inds = uv_map.copy()
    uv_map_inds[:,:,0] = np.clip((uv_map[:,:,0] * uvimg_W).round(), 0, uvimg_W-1)
    uv_map_inds[:,:,1] = np.clip((uv_map[:,:,1] * uvimg_H).round(), 0, uvimg_H-1)
    uv_map_inds = uv_map_inds.astype(np.int32)

    non_zero_inds = np.where(mask.squeeze())

    for rx,cx in zip(*non_zero_inds):
        u_ind = uv_map_inds[rx, cx, 0]
        v_ind = uv_map_inds[rx, cx, 1]
        if count_tex[v_ind, u_ind, 0] == 0:
            default_tex[v_ind, u_ind,:] = img[:, rx,cx]
        else:
            default_tex[v_ind, u_ind,:] += img[:, rx,cx]
        count_tex[v_ind, u_ind, 0] += 1

    count_tex = count_tex + 1*(count_tex < 1)
    default_tex = default_tex / count_tex
    return default_tex.transpose(2,0,1)

def wrap_texture_and_render(renderer, vert, camera, uv_sampler, texture_image, tex_size, other_vps, lights=True ):
    sampled_texture = torch.nn.functional.grid_sample(texture_image.unsqueeze(0), uv_sampler)
    sampled_texture = sampled_texture.squeeze().permute(1,2,0)
    sampled_texture = sampled_texture.view(sampled_texture.size(0), tex_size, tex_size, 3)
    
    sampled_texture = sampled_texture.unsqueeze(3).repeat(1, 1, 1, tex_size, 1)
    renderer.set_light_dir([0, 1, -1], 0.4)
    img_pred = renderer(vert, camera, texture=sampled_texture)
    if not lights:
        renderer.set_light_status(lights)

    if other_vps:
        new_camera = camera.clone()
        new_camera[3] = 1
        new_camera[4:] *=0 
        vp1 = renderer.diff_vp(
            vert, new_camera, angle=90, axis=[0, 1, 0], texture=sampled_texture, extra_elev=True)
        vp2 = renderer.diff_vp(
            vert, new_camera, angle=180, axis=[0, 1, 0], texture=sampled_texture, extra_elev=True)
        vp3 = renderer.diff_vp(
            vert, new_camera, angle=180, axis=[1, 0, 0], texture=sampled_texture)
        return (img_pred, vp1, vp2, vp3), texture_image.cpu().numpy().transpose(1,2,0)
    else:
        return (img_pred), texture_image.cpu().numpy().transpose(1,2,0)

def render_model_with_uv_map(renderer, vert, uvimg_H, uvimg_W, camera, uv_sampler, tex_size=6, other_vps=False):
    texture_image = cub_parse.get_sample_grid((uvimg_H,uvimg_W)) * 0.5 + 0.5
    texture_image = torch.cat([texture_image[:,:,None, 0], texture_image[:,:,None, 0]*0, texture_image[:,:,None, 1]], dim=-1)
    texture_image = texture_image.permute(2,0,1).cuda()
    # pdb.set_trace()
    render_stuff, texture_image = wrap_texture_and_render(renderer, vert, camera, uv_sampler, texture_image, tex_size, other_vps)
    return render_stuff, texture_image

def render_model_default(renderer, vert, uvimg_H, uvimg_W, camera, uv_sampler, tex_size=6, other_vps=False, color=None):
    # texture_image = cub_parse.get_sample_grid((uvimg_H,uvimg_W)) * 0.5 + 0.5
    # texture_image = torch.cat([texture_image[:,:,None, 0], texture_image[:,:,None, 0]*0, texture_image[:,:,None, 1]], dim=-1)
    # texture_image = texture_image.permute(2,0,1).cuda()
    
    # pdb.set_trace()
    texture_image = torch.from_numpy(create_monocolor_texture(uvimg_H, uvimg_W, color=color)).float().cuda()[0:3]
    render_stuff, texture_image = wrap_texture_and_render(renderer, vert, camera, uv_sampler, texture_image, tex_size, other_vps)
    return render_stuff, texture_image



def render_model_with_uv_greyscale_map(renderer, vert, uvimg_H, uvimg_W, camera, uv_sampler, tex_size=6, other_vps=False, uv_dim=0):
    texture_image = cub_parse.get_sample_grid((uvimg_H,uvimg_W)) * 0.5 + 0.5
    texture_image = torch.cat([texture_image[:,:,None, uv_dim], texture_image[:,:,None, uv_dim], texture_image[:,:,None, uv_dim]*0 + 1], dim=-1)
    texture_image = texture_image.permute(2,0,1).cuda()
    
    render_stuff, texture_image = wrap_texture_and_render(renderer, vert, camera, uv_sampler, texture_image, tex_size, other_vps)
    return render_stuff, texture_image


def copy_texture_from_img(mask, img, xy_map):
    img = (visutil.undo_resnet_preprocess(img.unsqueeze(0))*mask).squeeze()
    img = img.permute(1,2,0)
    xy_map_inds = xy_map.clone()
    xy_map_inds[:,:,0] = (xy_map_inds[:,:,0] + 1)* (img.size(1)/2)
    xy_map_inds[:,:,1] = (xy_map_inds[:,:,1] + 1) * (img.size(0)/2)
    xy_map_inds = torch.clamp(xy_map_inds.long(), min=0, max=img.size(0) -1).long().view(-1,2)
    new_img =  img[xy_map_inds[:,1], xy_map_inds[:,0],:].view(img.shape)
    # new_img = img
    new_img = new_img.permute(2,0,1)
    # new_img = new_img * mask
    new_img =  (new_img*mask).unsqueeze(0)
    new_img = visutil.tensor2im(new_img)
    return new_img

def copy_texture_using_uvmap(mask, texture, uv_map):
    ## UV map is between 0 and 1. convert it to -1 to 1
    ## uv_map tells you what location on the texture this pixel gets the color.
    ## All inputs are torch.Tensors
    uv_map = uv_map.unsqueeze(0)
    img = torch.nn.functional.grid_sample(texture.unsqueeze(0), uv_map*2 - 1)
    img = img*mask.unsqueeze(0) ## basically ignore things outside the mask.
    img = visutil.tensor2im(img.data)
    return img

def copy_texture_using_cycle_uvmap(mask, texture, uv_map):
    ## UV map is between 0 and 1. convert it to -1 to 1
    ## uv_map tells you what location on the texture this pixel gets the color.
    ## All inputs are torch.Tensors
    uv_map = uv_map.unsqueeze(0)
    img = torch.nn.functional.grid_sample(texture.unsqueeze(0), uv_map*2 - 1)
    img = img*mask.unsqueeze(0) ## basically ignore things outside the mask.
    img = visutil.tensor2im(img.data)
    return img

def render_model_with_texture(renderer, vert, uvimg_H, uvimg_W, uv_map, img, mask, camera, uv_sampler, tex_size=6, other_vps=False, undo_resnet=True):
    uv_map = uv_map.data.cpu().numpy()
    if undo_resnet:
        img = img.unsqueeze(0)
        img = visutil.undo_resnet_preprocess(img).squeeze()
    img = img.data.cpu().numpy()
    # camera = camera.data.cpu().numpy()
    mask = mask.data.cpu().numpy()
    texture_image = create_texture_image_from_uv_map(uvimg_H, uvimg_W, uv_map, img, mask)
    texture_image = torch.from_numpy(texture_image).float().cuda()

    render_stuff, texture_image = wrap_texture_and_render(renderer, vert, camera, uv_sampler, texture_image, tex_size, other_vps)
    return render_stuff, texture_image


def asVariable(x):
    if type(x) is not torch.autograd.Variable:
        x = Variable(x, requires_grad=False)
    return x


def convert_as(src, trg):
    src = src.type_as(trg)
    if src.is_cuda:
        src = src.cuda(device=trg.get_device())
    if type(trg) is torch.autograd.Variable:
        src = Variable(src, requires_grad=False)
    return src


def convert2np(x):
    # import ipdb; ipdb.set_trace()
    # if type(x) is torch.autograd.Variable:
    #     x = x.data
    # Assumes x is gpu tensor..
    if type(x) is not np.ndarray:
        return x.cpu().numpy()
    return x


def tensor2mask(image_tensor, imtype=np.uint8):
    # Input is H x W
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = np.expand_dims(image_numpy, 2) * 255.0
    image_numpy = np.tile(image_numpy, (1, 1, 3))
    return image_numpy.astype(imtype)


def kp2im(kp, img, radius=None):
    """
    Input is numpy array or torch.cuda.Tensor
    img can be H x W, H x W x C, or C x H x W
    kp is |KP| x 2

    """
    kp_norm = convert2np(kp)
    img = convert2np(img)

    if img.ndim == 2:
        img = np.dstack((img, ) * 3)
    # Make it H x W x C:
    elif img.shape[0] == 1 or img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
        if img.shape[2] == 1:  # Gray2RGB for H x W x 1
            img = np.dstack((img, ) * 3)

    # kp_norm is still in [-1, 1], converts it to image coord.
    kp = (kp_norm[:, :2] + 1) * 0.5 * img.shape[0]
    if kp_norm.shape[1] == 3:
        vis = kp_norm[:, 2] > 0
        kp[~vis] = 0
        kp = np.hstack((kp, vis.reshape(-1, 1)))
    else:
        vis = np.ones((kp.shape[0], 1))
        kp = np.hstack((kp, vis))

    kp_img = draw_kp(kp, img, radius=radius)

    return kp_img


def draw_kp(kp, img, radius=None):
    """
    kp is 15 x 2 or 3 numpy.
    img can be either RGB or Gray
    Draws bird points.
    """
    if radius is None:
        radius = max(4, (np.mean(img.shape[:2]) * 0.01).astype(int))

    num_kp = kp.shape[0]
    # Generate colors
    import pylab
    cm = pylab.get_cmap('gist_rainbow')
    colors = 255 * np.array([cm(1. * i / num_kp)[:3] for i in range(num_kp)])
    white = np.ones(3) * 255

    image = img.copy()

    if isinstance(image.reshape(-1)[0], np.float32):
        # Convert to 255 and np.uint8 for cv2..
        image = (image * 255).astype(np.uint8)

    kp = np.round(kp).astype(int)

    for kpi, color in zip(kp, colors):
        # This sometimes causes OverflowError,,
        if kpi[2] == 0:
            continue
        cv2.circle(image, (kpi[0], kpi[1]), radius + 1, white, -1)
        cv2.circle(image, (kpi[0], kpi[1]), radius, color, -1)

    # import matplotlib.pyplot as plt
    # plt.ion()
    # plt.clf()
    # plt.imshow(image)
    # import ipdb; ipdb.set_trace()
    return image


def vis_verts(mean_shape, verts, face, mvs=None, textures=None):
    """
    mean_shape: N x 3
    verts: B x N x 3
    face: numpy F x 3
    textures: B x F x T x T (x T) x 3
    """
    from psbody.mesh.mesh import Mesh
    from psbody.mesh.meshviewer import MeshViewers
    if mvs is None:
        mvs = MeshViewers((2, 3))

    num_row = len(mvs)
    num_col = len(mvs[0])

    mean_shape = convert2np(mean_shape)
    verts = convert2np(verts)

    num_show = min(num_row * num_col, verts.shape[0] + 1)

    mvs[0][0].set_dynamic_meshes([Mesh(mean_shape, face)])
    # 0th is mean shape:

    if textures is not None:
        tex = convert2np(textures)
    for k in np.arange(1, num_show):
        vert_here = verts[k - 1]
        if textures is not None:
            tex_here = tex[k - 1]
            fc = tex_here.reshape(tex_here.shape[0], -1, 3).mean(axis=1)
            mesh = Mesh(vert_here, face, fc=fc)
        else:
            mesh = Mesh(vert_here, face)
        mvs[int(k % num_row)][int(k / num_row)].set_dynamic_meshes([mesh])


def vis_vert2kp(verts, vert2kp, face, mvs=None):
    """
    verts: N x 3
    vert2kp: K x N

    For each keypoint, visualize its weights on each vertex.
    Base color is white, pick a color for each kp.
    Using the weights, interpolate between base and color.

    """
    from psbody.mesh.mesh import Mesh
    from psbody.mesh.meshviewer import MeshViewer, MeshViewers
    from psbody.mesh.sphere import Sphere

    num_kp = vert2kp.shape[0]
    if mvs is None:
        mvs = MeshViewers((4, 4))
    # mv = MeshViewer()
    # Generate colors
    import pylab
    cm = pylab.get_cmap('gist_rainbow')
    cms = 255 * np.array([cm(1. * i / num_kp)[:3] for i in range(num_kp)])
    base = np.zeros((1, 3)) * 255
    # base = np.ones((1, 3)) * 255

    verts = convert2np(verts)
    vert2kp = convert2np(vert2kp)

    num_row = len(mvs)
    num_col = len(mvs[0])

    colors = []
    for k in range(num_kp):
        # Nx1 for this kp.
        weights = vert2kp[k].reshape(-1, 1)
        # So we can see it,,
        weights = weights / weights.max()
        cm = cms[k, None]
        # Simple linear interpolation,,
        # cs = np.uint8((1-weights) * base + weights * cm)
        # In [0, 1]
        cs = ((1 - weights) * base + weights * cm) / 255.
        colors.append(cs)

        # sph = [Sphere(center=jc, radius=.03).to_mesh(c/255.) for jc, c in zip(vert,cs)]
        # mvs[int(k/4)][k%4].set_dynamic_meshes(sph)
        mvs[int(k % num_row)][int(k / num_row)].set_dynamic_meshes(
            [Mesh(verts, face, vc=cs)])


def tensor2im(image_tensor, imtype=np.uint8, scale_to_range_1=False):
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    if scale_to_range_1:
        image_numpy = image_numpy - np.min(image_numpy, axis=2, keepdims=True)
        image_numpy = image_numpy / np.max(image_numpy)
    else:
        # Clip to [0, 1]
        image_numpy = np.clip(image_numpy, 0, 1)

    return (image_numpy * 255).astype(imtype)


def visflow(flow_img):
    # H x W x 2
    flow_img = convert2np(flow_img)
    from matplotlib import cm
    x_img = flow_img[:, :, 0]

    def color_within_01(vals):
        # vals is Nx1 in [-1, 1] (but could be larger)
        vals = np.clip(vals, -1, 1)
        # make [0, 1]
        vals = (vals + 1) / 2.
        # Append dummy end vals for consistent coloring
        weights = np.hstack([vals, np.array([0, 1])])
        # Drop the dummy colors
        colors = cm.plasma(weights)[:-2, :3]
        return colors

    # x_color = cm.plasma(x_img.reshape(-1))[:, :3]
    x_color = color_within_01(x_img.reshape(-1))
    x_color = x_color.reshape([x_img.shape[0], x_img.shape[1], 3])
    y_img = flow_img[:, :, 1]
    # y_color = cm.plasma(y_img.reshape(-1))[:, :3]
    y_color = color_within_01(y_img.reshape(-1))
    y_color = y_color.reshape([y_img.shape[0], y_img.shape[1], 3])
    vis = np.vstack([x_color, y_color])
    # import matplotlib.pyplot as plt
    # plt.ion()
    # plt.imshow(x_color)
    return vis


def visflow_jonas(flow_img, img_size):
    from ..utils.viz_flow import viz_flow
    # H x W x 2
    flow = convert2np(flow_img)

    # viz_flow expects the top left to be zero.
    # Conver to image coord
    flow = (flow + 1) * 0.5 * img_size

    flow_img = viz_flow(flow[:, :, 1], flow[:, :, 0])

    return flow_img


if __name__ == '__main__':

    # Test vis_vert2kp:
    from ..utils import mesh
    verts, faces = mesh.create_sphere()
    num_kps = 15
    num_vs = verts.shape[0]

    ind = np.random.randint(0, num_vs, num_vs)
    dists = np.stack([
        np.linalg.norm(verts - verts[np.random.randint(0, num_vs)], axis=1)
        for k in range(num_kps)
    ])
    vert2kp = np.exp(-.5 * (dists) / (np.random.rand(num_kps, 1) + 0.4))
    vert2kp = vert2kp / vert2kp.sum(1).reshape(-1, 1)

    vis_vert2kp(verts, vert2kp, faces)
