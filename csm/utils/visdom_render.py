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
from . import bird_vis
from ..utils import render_utils
import pdb
import matplotlib.pyplot as plt
import scipy.misc 
import uuid
import os

class VisdomRenderer(object):
    def __init__(self, renderer, verts, uv_sampler, offset_z, mean_shape_np,
                 model_obj_path, keypoint_cmap, opts):
        self.vis_rend = renderer
        self.verts_obj = verts
        self.uv_sampler = uv_sampler
        self.offset_z = offset_z
        self.uvimgH = 256
        self.uvimgW = 256
        self.mean_shape_np = mean_shape_np
        self.opts = opts
        self.model_obj_path = model_obj_path
        self.keypoint_cmap = keypoint_cmap

        self.Tensor = torch.cuda.FloatTensor
        
        return



    '''
        Set of per pixel loss for all hypothesis
    '''
    # import tempfile
    # import uuid
    def visualize_depth_loss_perpixel(self, per_pixel_losses):
        
        image = per_pixel_losses.unsqueeze(-1).data.cpu().numpy()
        temp = visutil.image_montage(image, nrow=3)
        plt.imshow(temp[:,:,0], cmap='hot', interpolation='nearest')
        plt.colorbar()
        filename = osp.join('/tmp/', "{}.png".format(str(uuid.uuid4())))
        try:
            plt.savefig(filename)
            image = scipy.misc.imread(filename)
        except Exception as e:
            image = (temp*0).astype(np.uint8)
        finally:
            os.remove(filename)
            plt.close()

        return image

    def render_mean_bird_with_uv(self,):
        opts = self.opts
        cam = torch.zeros(7).float().cuda()
        cam[3] = 1
        results_dir = osp.join(opts.result_dir, "{}".format(opts.split))
        visutil.mkdir(results_dir)
        visuals = {}
        _, uv_texture_img = bird_vis.render_model_with_uv_map(
            self.vis_rend, self.verts_obj, self.uvimgW, self.uvimgH, cam,
            self.uv_sampler)
        visuals['uv_texture_img'] = (uv_texture_img * 255).astype(np.uint8)
        bird_vis.save_obj_with_texture('mean_bird', results_dir,
                                       visuals['uv_texture_img'],
                                       self.mean_shape_np)
        return visuals

    def render_model_using_cam(self, camera):
        opts = self.opts
        img = render_utils.render_model(
            osp.join(opts.rendering_dir),
            self.model_obj_path,
            self.offset_z,
            camera.data.cpu().numpy(),
        )
        return img

    def render_model_using_nmr(
            self,
            uv_map,
            img,
            mask,
            cam,
            upsample_texture=True,
    ):
        visuals = {}
        uvimgH = self.uvimgH
        uvimgW = self.uvimgW
        if upsample_texture:
            img, mask, uv_map = bird_vis.upsample_img_mask_uv_map(img, mask, uv_map)

        texture_ms, texture_img = bird_vis.render_model_with_texture(
            self.vis_rend,
            self.verts_obj,
            uvimgW,
            uvimgH,
            uv_map,
            img,
            mask,
            cam,
            self.uv_sampler,
            other_vps=True,
            undo_resnet=True)
        visuals['texture_ms'] = visutil.image_montage(list(texture_ms), nrow=2)
        visuals['texture_img'] = (texture_img * 255).astype(np.uint8)
        return visuals

    def render_model_uv_using_nmr(
            self,
            uv_map,
            mask,
            cam,
    ):
        u_image = uv_map[:, :, None, 0].permute(2, 1, 0).repeat(3, 1, 1)
        v_image = uv_map[:, :, None, 1].permute(2, 1, 0).repeat(3, 1, 1)
        visuals = {}
        img_pred_uu, _ = bird_vis.render_model_with_texture(self.vis_rend, self.verts_obj, 256,
                                                            256,
                                                            uv_map,
                                                            u_image,
                                                            mask,
                                                            cam,
                                                            self.uv_sampler,
                                                            other_vps=True)
        img_pred_vv, _ = bird_vis.render_model_with_texture(
            self.vis_rend,
            self.verts_obj,
            256,
            256,
            uv_map,
            v_image,
            mask,
            cam,
            self.uv_sampler,
            other_vps=True)
        visuals['z_texture_uu'] = visutil.image_montage(
            list(img_pred_uu), nrow=2)
        visuals['z_texture_vv'] = visutil.image_montage(
            list(img_pred_vv), nrow=2)

        img_pred_ms_u, _ = bird_vis.render_model_with_uv_greyscale_map(
            self.vis_rend,
            self.verts_obj,
            256,
            256,
            cam,
            self.uv_sampler,
            other_vps=True,
            uv_dim=0)
        visuals['u_ms'] = visutil.image_montage(list(img_pred_ms_u), nrow=2)
        img_pred_ms_v, _ = bird_vis.render_model_with_uv_greyscale_map(
            self.vis_rend,
            self.verts_obj,
            256,
            256,
            cam,
            self.uv_sampler,
            other_vps=True,
            uv_dim=1)
        visuals['v_ms'] = visutil.image_montage(list(img_pred_ms_v), nrow=2)
        return visuals

    def render_gt_kps_heatmap(self, kp3d_uv, camera, suffix=''):
        uv_sampler = self.uv_sampler
        tex_size = self.opts.tex_size
        other_vps = False
        uv_H = 256
        uv_W = 256
        all_visuals = []
        default_tex = bird_vis.create_kp_heat_map_texture(uv_H, uv_W)
        kp_textures = []

        for kpx, kp_uv in enumerate(kp3d_uv):
            visuals = {}
            kp_uv = kp_uv * 255
            uv_cords = [int(kp_uv[0]), int(kp_uv[1])]
            kp_color = self.keypoint_cmap[kpx]
            texture = bird_vis.create_kp_heat_map_texture(
                uv_H, uv_W, uv_cords[0], uv_cords[1], color=kp_color)
            kp_textures.append(texture)
        # kp_textures =
        kp_textures = np.stack(kp_textures, axis=0)
        default_mask = (0 == np.max(kp_textures[:, 3, None, :, :], axis=0))
        average = np.sum(kp_textures[:, 3, None, :, :], axis=0) + default_mask

        texture = np.sum(kp_textures, axis=0) / average
        texture = default_tex * default_mask + texture * (1 - default_mask)
        texture = texture[0:3, :, :]
        texture = torch.from_numpy(texture).float().cuda()
        # renderer, vert, camera, uv_sampler, texture_image, tex_size, other_vps
        texture_ms, texture_img = bird_vis.wrap_texture_and_render(
            self.vis_rend,
            self.verts_obj,
            camera,
            uv_sampler,
            texture,
            tex_size,
            other_vps=True)
        visuals = {}
        visuals['texture_kp_zgt'+suffix] = visutil.image_montage(list(texture_ms), nrow=2)

        visuals['texture_kp_img_zgt'+suffix] = (texture_img * 255).astype(np.uint8)
        return visuals

    def render_kps_heatmap(self, uv_map, kps_ind, kps_vis, camera):
        uv_sampler = self.uv_sampler
        tex_size = self.opts.tex_size
        other_vps = False
        uv_H = self.uvimgH
        uv_W = self.uvimgW
        all_visuals = []
        default_tex = bird_vis.create_kp_heat_map_texture(uv_H, uv_W)
        kp_textures = []

        for kpx, (kp_ind, kp_vis) in enumerate(zip(kps_ind, kps_vis)):
            visuals = {}
            if kp_vis:
                y_cord = int(kp_ind[1])
                x_cord = int(kp_ind[0])
                # pdb.set_trace()
                uv_cords = torch.round(
                    uv_map[max(y_cord - 2,0):min(y_cord + 2, uv_map.size(0)), max(x_cord - 2, 0):min(x_cord + 2, uv_map.size(1)), :] *
                    255)
                # uv_cords = torch.round(uv_map[y_cord - 2:y_cord + 2, x_cord - 2:x_cord + 2, :] * 255)
                if uv_cords.numel() == 0:
                    continue
                    pdb.set_trace()
                uv_cords, _ = torch.median(uv_cords.view(-1, 2), dim=0)
                uv_cords = [int(uv_cords[0]), int(uv_cords[1])]
                kp_color = self.keypoint_cmap[kpx]
                texture = bird_vis.create_kp_heat_map_texture(
                    uv_H, uv_W, uv_cords[0], uv_cords[1], color=kp_color)
                kp_textures.append(texture)
        # kp_textures =
        kp_textures = np.stack(kp_textures, axis=0)
        default_mask = (0 == np.max(kp_textures[:, 3, None, :, :], axis=0))
        average = np.sum(kp_textures[:, 3, None, :, :], axis=0) + default_mask

        texture = np.sum(kp_textures, axis=0) / average
        texture = default_tex * default_mask + texture * (1 - default_mask)
        texture = texture[0:3, :, :]
        texture = torch.from_numpy(texture).float().cuda()
        # pdb.set_trace() renderer, vert, camera, uv_sampler, texture_image, tex_size, other_vps
        texture_ms, texture_img = bird_vis.wrap_texture_and_render(
            self.vis_rend,
            self.verts_obj,
            camera,
            uv_sampler,
            texture,
            tex_size,
            other_vps=True)
        visuals = {}
        visuals['texture_kp'] = visutil.image_montage(list(texture_ms), nrow=2)
        visuals['texture_kp_img'] = (texture_img * 255).astype(np.uint8)
        return visuals


    def render_all_hypotheses(self, cameras, probs=None, sample_ind=None, gt_cam=None,losses_per_hypo=None):
        uv_sampler = self.uv_sampler
        tex_size = self.opts.tex_size
        uv_H = self.uvimgH
        uv_W = self.uvimgW
        default_tex = bird_vis.create_monocolor_texture(uv_H, uv_W)[0:3,:,:]
        default_tex = torch.from_numpy(default_tex).float().cuda()
        renderings = []
        _, max_ind = torch.max(probs, dim=0)
        max_ind = max_ind.item()
        max_ind = sample_ind if sample_ind is not None else max_ind
        from . import metrics
        gt_quat = gt_cam[3:7].data.cpu()
        cam_errors = np.array([metrics.quat_dist(pred, gt_quat) for pred in cameras[:, 3:].data.cpu()])
        for cx, camera in enumerate(cameras):
            texture_ms, _ = bird_vis.wrap_texture_and_render(
                self.vis_rend,
                self.verts_obj,
                camera,
                uv_sampler,
                default_tex,
                tex_size,
                other_vps=False)
            if probs is not None :
                import cv2
                color = (0, 128, 128) if (cx == max_ind) else (255,0,0)
                texture_ms = cv2.putText(texture_ms,"P:{}".format(np.round(probs[cx].item(), 2)), (125,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2)
            if gt_cam is not None:
                texture_ms = cv2.putText(texture_ms,"E:{}".format(np.round(cam_errors[cx], 1)), (125,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2)
            if losses_per_hypo is not None:
                for lx, loss_per_hypo in enumerate(losses_per_hypo):
                    texture_ms = cv2.putText(texture_ms,"E:{}".format(np.round(loss_per_hypo[cx].item(), 2)), (125,90 + lx*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2)
            renderings.append(texture_ms)
        visuals = {}
        visuals['all_hypotheses'] = visutil.image_montage(renderings, nrow=3)
        return visuals

    def render_cycle_images(self, mask, xy_map, img, img_mask, camera,):
        '''
        Cannot handle batching, one example at a time.
        '''
        mask_np = mask.data.cpu().numpy()
        tex_size = self.opts.tex_size
        other_vps = False
        uv_H = 256
        uv_W = 256
        all_visuals = []
        default_tex = bird_vis.create_monocolor_texture(uv_H, uv_W)
        red_tex = bird_vis.create_monocolor_texture(
            uv_H, uv_W, color=[234., 0., 0.,
                               1])  # the last one tells transperancy

        mask_texture = mask_np * red_tex
        texture = mask_texture
        texture = (torch.from_numpy(texture)[0:3, :, :]).type(self.Tensor)
        texture_ms, texture_img = bird_vis.wrap_texture_and_render(
            self.vis_rend,
            self.verts_obj,
            camera,
            self.uv_sampler,
            texture,
            tex_size,
            other_vps=True)
        visuals = {}
        visuals['t_cyc_mask_ms'] = visutil.image_montage(
            list(texture_ms), nrow=2)
        visuals['t_cyc_mask_img'] = (texture_img * 255).astype(np.uint8)

        # pdb.set_trace()
        img = (visutil.undo_resnet_preprocess(img.unsqueeze(0))).squeeze()
        img = img * img_mask
        texture_pick = torch.nn.functional.grid_sample(
            img.unsqueeze(0),
            xy_map.permute(1, 2, 0).unsqueeze(0))
        texture_pick = texture_pick.data.squeeze().cpu().numpy()
        texture_pick = np.concatenate(
            [texture_pick, texture_pick[0, None, ...] * 0 + 1], axis=0)
        # pdb.set_trace()
        texture_pick = (mask_np > 0.5) * texture_pick
        texture = bird_vis.merge_textures(texture_pick, default_tex)
        texture = (torch.from_numpy(texture)[0:3, :, :]).type(self.Tensor)
        texture_ms, texture_img = bird_vis.wrap_texture_and_render(
            self.vis_rend,
            self.verts_obj,
            camera,
            self.uv_sampler,
            texture,
            tex_size,
            other_vps=True)
        visuals['t_cyc_pick_ms'] = visutil.image_montage(
            list(texture_ms), nrow=2)
        visuals['t_cyc_pick_img'] = (texture_img * 255).astype(np.uint8)
        return visuals

    def create_texture_from_cycle_xy_map(self, img, img_mask, xy_map, xy_map_mask):
        default_tex = bird_vis.create_monocolor_texture(self.uvimgH, self.uvimgW)
        img = (visutil.undo_resnet_preprocess(img.unsqueeze(0))).squeeze()
        img = img * img_mask
        texture_pick = torch.nn.functional.grid_sample(
            img.unsqueeze(0),
            xy_map.permute(1, 2, 0).unsqueeze(0))
        texture_pick = texture_pick.data.squeeze().cpu().numpy()
        texture_pick = np.concatenate(
            [texture_pick, texture_pick[0, None, ...] * 0 + 1], axis=0)
        mask_np = xy_map_mask.data.cpu().numpy()
        texture_pick = (mask_np > 0.5) * texture_pick
        texture = bird_vis.merge_textures(texture_pick, default_tex)
        texture = (torch.from_numpy(texture)[0:3, :, :]).type(self.Tensor)
        return texture

    def wrap_texture(self, texture, camera, lights, tex_size=6):
        img, _ = bird_vis.wrap_texture_and_render(self.vis_rend, self.verts_obj, camera, self.uv_sampler, texture, tex_size, other_vps=False, lights=lights)
        return img

    def set_mean_shape_verts(self, verts):
        # print('Updating mean shape')
        self.verts_obj = verts
        return

    def render_default_uv_map(self, camera, tex_size=6):
        image_u, u_texture_img = bird_vis.render_model_with_uv_greyscale_map(self.vis_rend, self.verts_obj, self.uvimgH, self.uvimgW, camera, self.uv_sampler, uv_dim=0, tex_size=tex_size)
        image_v, v_texture_img = bird_vis.render_model_with_uv_greyscale_map(self.vis_rend, self.verts_obj,self.uvimgH, self.uvimgW, camera, self.uv_sampler, uv_dim=1, tex_size=tex_size)
        return image_u, u_texture_img, image_v, v_texture_img


    def render_default_bird(self, camera, color=None, tex_size=6):
        # pdb.set_trace()
        image, texture_img = bird_vis.render_model_default(self.vis_rend, self.verts_obj, self.uvimgH, self.uvimgW, camera, self.uv_sampler, color=color, tex_size=tex_size)
        return image, texture_img



    def render_mask_boundary(self, img, mask):
        import cv2
        img_mask = np.stack([mask, mask, mask], axis=2)*255
        img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
        (thresh, im_bw) = cv2.threshold(img_mask.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
        im_bw = im_bw.astype(np.uint8)
        _, contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img =  np.ascontiguousarray(img, dtype=np.uint8)
        new_img = cv2.drawContours(img, contours, -1, (0,255,0), 3)
        return new_img

def render_UV_contour(img, uvmap, mask):
    mask = mask.permute(1,2,0).numpy().squeeze()
    import matplotlib.pyplot as plt
    import scipy.misc
    import os
    import tempfile
    
    cm1 = plt.get_cmap('jet')
    cm2 = plt.get_cmap('terrain')
    plt.imshow(img)
    plt.contour(uvmap[:,:,0].numpy()*mask, 10, linewidths = 1, cmap=cm1, vmin=0, vmax=1)
    plt.contour(uvmap[:,:,1].numpy()*mask, 10, linewidths = 1, cmap=cm2, vmin=0, vmax=1)

    plt.axis('off')
    temp_file = '/tmp/file_{}.jpg'.format(np.random.randint(10000000))
    plt.savefig(temp_file)
    image = scipy.misc.imread(temp_file)
    os.remove(temp_file)
    plt.close()
    return image


