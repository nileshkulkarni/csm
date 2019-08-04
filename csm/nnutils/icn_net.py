"""
Instance Correspondence net model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import os
import os.path as osp
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import pdb
from . import net_blocks as nb
from . import unet
from ..utils import cub_parse
from . import geom_utils
import math
#-------------- flags -------------#
#----------------------------------#
flags.DEFINE_integer('nz_feat', 200, 'Encoded feature size')
flags.DEFINE_integer('nz_UV_height', 256 // 3, 'image height')
flags.DEFINE_integer('nz_UV_width', 256 // 3, 'image width')
flags.DEFINE_integer('num_hypo_cams', 8, 'number of hypo cams')
flags.DEFINE_boolean('use_deconv', False, 'If true uses Deconv')
flags.DEFINE_string('upconv_mode', 'bilinear', 'upsample mode')
flags.DEFINE_boolean('multiple_cam_hypo', True, 'Multiple camera hypothesis')
flags.DEFINE_boolean('render_mask', True, 'Render mask')
flags.DEFINE_boolean('render_depth', True, 'Render depth')
flags.DEFINE_boolean('pred_cam', False, 'Predict camera  instead of computing camera from predictions')
flags.DEFINE_boolean('pred_mask', True, 'Predict camera  instead of computing camera from predictions')
flags.DEFINE_boolean('az_ele_quat', True, 'Predict camera as azi elev')
flags.DEFINE_float('scale_lr_decay', 0.05, 'Scale multiplicative factor')
flags.DEFINE_float('scale_bias', 1.0, 'Scale bias factor')

class ResNetConv(nn.Module):

    def __init__(self, n_blocks=4):
        super(ResNetConv, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.n_blocks = n_blocks

    def forward(self, x):
        n_blocks = self.n_blocks
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        if n_blocks >= 1:
            x = self.resnet.layer1(x)
        if n_blocks >= 2:
            x = self.resnet.layer2(x)
        if n_blocks >= 3:
            x = self.resnet.layer3(x)
        if n_blocks >= 4:
            x = self.resnet.layer4(x)
        return x

class AlexNetConv(nn.Module):

    def __init__(self, n_blocks=4):
        super(AlexNetConv, self).__init__()
        alexnet = torchvision.models.alexnet(pretrained=True)
        self.n_blocks = n_blocks
        self.net = nn.Sequential(*list(alexnet.features.children())[0:3*n_blocks+1])
    def forward(self, x):
        x = self.net.forward(x)
        return x

class VGG16Conv(nn.Module):

    def __init__(self, n_blocks=4):
        super(VGG16Conv, self).__init__()
        self.vgg16 =vgg16 = torchvision.models.vgg16(pretrained=True)
        self.block2layerid = [2, 4, 7, 9, 12, 16]
        self.net = nn.Sequential(*list(vgg16.features.children())[0:self.block2layerid[n_blocks]])

    def forward(self, x):
        x = self.net.forward(x)
        return x


class Encoder(nn.Module):
    """
    Current:
    Resnet with 4 blocks (x32 spatial dim reduction)
    Another conv with stride 2 (x64)
    This is sent to 2 fc layers with final output nz_feat.
    """

    def __init__(self, input_shape, n_blocks=4, nz_feat=100, batch_norm=True):
        super(Encoder, self).__init__()
        self.resnet_conv = ResNetConv(n_blocks=4)
        self.enc_conv1 = nb.conv2d(batch_norm, 512, 256, stride=2, kernel_size=4)
        nc_input = 256 * (input_shape[0] // 64) * (input_shape[1] // 64)
        self.enc_fc = nb.fc_stack(nc_input, nz_feat, 2)

        nb.net_init(self.enc_conv1)

    def forward(self, img):
        resnet_feat = self.resnet_conv.forward(img)
        self.resnet_feat = resnet_feat
        out_enc_conv1 = self.enc_conv1(resnet_feat)
        out_enc_conv1 = out_enc_conv1.view(img.size(0), -1)
        feat = self.enc_fc.forward(out_enc_conv1)

        return feat

class QuatPredictor(nn.Module):

    def __init__(self, nz_feat, nz_rot=4, classify_rot=False):
        super(QuatPredictor, self).__init__()
        self.pred_layer = nn.Linear(nz_feat, nz_rot)
        self.classify_rot = classify_rot

    def forward(self, feat):
        quat = self.pred_layer.forward(feat)
        if self.classify_rot:
            quat = torch.nn.functional.log_softmax(quat)
        else:
            quat = torch.nn.functional.normalize(quat)

        return quat

    def initialize_to_zero_rotation(self,):
        nb.net_init(self.pred_layer)
        self.pred_layer.bias = nn.Parameter(torch.FloatTensor([1,0,0,0]).type(self.pred_layer.bias.data.type()))
        return

class QuatPredictorAzEle(nn.Module):

    def __init__(self, nz_feat, dataset='others'):
        super(QuatPredictorAzEle, self).__init__()
        self.pred_layer = nn.Linear(nz_feat, 3)
        self.axis = torch.eye(3).float().cuda()
        self.dataset = dataset

    def forward(self, feat):
        angles = 0.1*self.pred_layer.forward(feat)
        angles = torch.tanh(feat)
        azimuth = math.pi/6 * angles[...,0]

        # # Birds
        if self.dataset == 'cub':
            elev = math.pi/2 * (angles[...,1])
            cyc_rot = math.pi/3 * (angles[...,2])
        else:
            # cars # Horse & Sheep
            elev = math.pi/9 * (angles[...,1])
            cyc_rot = math.pi/9 * (angles[...,2])

        q_az = self.convert_ax_angle_to_quat(self.axis[1], azimuth)
        q_el = self.convert_ax_angle_to_quat(self.axis[0], elev)
        q_cr = self.convert_ax_angle_to_quat(self.axis[2], cyc_rot)
        quat = geom_utils.hamilton_product(q_el.unsqueeze(1), q_az.unsqueeze(1))
        quat = geom_utils.hamilton_product(q_cr.unsqueeze(1), quat)
        return quat.squeeze(1)

    def convert_ax_angle_to_quat(self, ax, ang):
        qw = torch.cos(ang/2)
        qx = ax[0] * torch.sin(ang/2)
        qy = ax[1] * torch.sin(ang/2)
        qz = ax[2] * torch.sin(ang/2)
        quat = torch.stack([qw, qx, qy, qz], dim=1)
        return quat

    def initialize_to_zero_rotation(self,):
        nb.net_init(self.pred_layer)
        return

class ScalePredictor(nn.Module):

    def __init__(self, nz, bias=1.0, lr=0.05):
        super(ScalePredictor, self).__init__()
        self.pred_layer = nn.Linear(nz, 1)
        self.lr = lr
        self.bias = bias

    def forward(self, feat):
        scale = self.lr * self.pred_layer.forward(feat) + self.bias # b
        scale = torch.nn.functional.relu(scale) + 1E-12  # minimum scale is 0.0
        return scale



class TransPredictor(nn.Module):
    """
    Outputs [tx, ty] or [tx, ty, tz]
    """
    def __init__(self, nz, orth=True):
        super(TransPredictor, self).__init__()
        if orth:
            self.pred_layer = nn.Linear(nz, 2)
        else:
            self.pred_layer = nn.Linear(nz, 3)

    def forward(self, feat):
        trans = self.pred_layer.forward(feat)
        return trans

class Camera(nn.Module):

    def __init__(self, nz_input, az_ele_quat=False, scale_lr=0.05, scale_bias=1.0, dataset='others'):
        super(Camera, self).__init__()
        self.fc_layer = nb.fc_stack(nz_input, nz_input, 2)

        if az_ele_quat:
            self.quat_predictor = QuatPredictorAzEle(nz_input, dataset)
        else:
            self.quat_predictor = QuatPredictor(nz_input)

        self.prob_predictor = nn.Linear(nz_input, 1)
        self.scale_predictor = ScalePredictor(nz_input, lr=scale_lr, bias=scale_bias)
        self.trans_predictor = TransPredictor(nz_input)

    def forward(self, feat):
        feat = self.fc_layer(feat)
        quat_pred = self.quat_predictor.forward(feat)
        prob = self.prob_predictor(feat)
        scale = self.scale_predictor.forward(feat)
        trans = self.trans_predictor.forward(feat)
        return torch.cat([quat_pred, prob, scale, trans], dim=1)

    def init_quat_module(self,):
        self.quat_predictor.initialize_to_zero_rotation()



class MultiCamPredictor(nn.Module):

    def __init__(self, nc_input, ns_input, nz_channels, nz_feat=100, num_cams=8, 
                 aze_ele_quat=False, scale_lr=0.05, scale_bias=1.0, dataset='others'):
        super(MultiCamPredictor, self).__init__()

        self.fc = nb.fc_stack(nz_feat, nz_feat, 2, use_bn=False)
        self.scale_predictor = ScalePredictor(nz_feat)
        nb.net_init(self.scale_predictor)
        self.trans_predictor = TransPredictor(nz_feat)
        nb.net_init(self.trans_predictor)
        self.prob_predictor = nn.Linear(nz_feat, num_cams)
        self.camera_predictor = nn.ModuleList([Camera(nz_feat,aze_ele_quat, scale_lr=scale_lr,
                                                      scale_bias=scale_bias, dataset=dataset) for i in range(num_cams)])

        nb.net_init(self)
        for cx in range(num_cams):
            self.camera_predictor[cx].init_quat_module()

        self.quat_predictor = QuatPredictor(nz_feat)
        self.quat_predictor.initialize_to_zero_rotation()
        self.num_cams = num_cams

        base_rotation = torch.FloatTensor([0.9239, 0, 0.3827 , 0]).unsqueeze(0).unsqueeze(0) ##pi/4
        # base_rotation = torch.FloatTensor([ 0.7071,  0 , 0.7071,   0]).unsqueeze(0).unsqueeze(0) ## pi/2 
        base_bias = torch.FloatTensor([ 0.7071,  0.7071,   0,   0]).unsqueeze(0).unsqueeze(0)
        self.cam_biases = [base_bias]
        for i in range(1,self.num_cams):
            self.cam_biases.append(geom_utils.hamilton_product(base_rotation, self.cam_biases[i-1]))
        self.cam_biases = torch.stack(self.cam_biases).squeeze().cuda()
        return

    def forward(self, feat):
        feat = self.fc(feat)
        cameras = []
        for cx in range(self.num_cams):
            cameras.append(self.camera_predictor[cx].forward(feat))
        cameras = torch.stack(cameras, dim=1)
        quats = cameras[:, :, 0:4]
        prob_logits = cameras[:, :, 4]
        camera_probs = nn.functional.softmax(prob_logits, dim=1)

        scale = self.scale_predictor.forward(feat).unsqueeze(1).repeat(1, self.num_cams, 1)
        trans = self.trans_predictor.forward(feat).unsqueeze(1).repeat(1, self.num_cams, 1)
        scale = cameras[:,:,5:6]
        trans = cameras[:,:,6:8]
     
        bias_quats = self.cam_biases.unsqueeze(0).repeat(len(quats), 1, 1)
        new_quats = geom_utils.hamilton_product(quats, bias_quats)
        cam = torch.cat([scale, trans, new_quats, camera_probs.unsqueeze(-1)], dim=2)
        return self.sample(cam) + (quats,)

    def sample(self, cam):
        '''
            cams : B x num_cams x 8 Vector. Last column is probs.
        '''
        dist = torch.distributions.multinomial.Multinomial(probs=cam[:, :, 7])
        sample = dist.sample()
        sample_inds = torch.nonzero(sample)[:, None, 1]
        sampled_cam = torch.gather(cam, dim=1, index=sample_inds.unsqueeze(-1).repeat(1, 1, 8)).squeeze()[:, 0:7]
        return sampled_cam, sample_inds, cam[:, :, 7], cam[:, :, 0:7]


class ICPNet(nn.Module):

    def __init__(self, opts):
        super(ICPNet, self).__init__()
        self.opts = opts
        self.nc_encoder = 256
        self.uv_pred_dim = 3
        self.xy_pred_dim = 2
        self.unet_oc = self.uv_pred_dim +  1 ## uvmap + mask

        self.unet_gen = unet.UnetConcatGenerator(input_nc=3, output_nc=(
            self.unet_oc), num_downs=5,)
        self.unet_innermost = self.unet_gen.get_inner_most()
        self.img_encoder = Encoder((opts.img_size, opts.img_size), nz_feat=opts.nz_feat)

        self.grid = cub_parse.get_sample_grid((opts.img_size,opts.img_size)).cuda()

        if opts.multiple_cam_hypo:
            self.cam_predictor = MultiCamPredictor(512, 8, 128, nz_feat=opts.nz_feat,
                                                     num_cams=opts.num_hypo_cams, aze_ele_quat=opts.az_ele_quat,
                                                     scale_lr=opts.scale_lr_decay, scale_bias=opts.scale_bias,
                                                     dataset = opts.dataset)
        else:
            self.cam_predictor = Camera(opts.nz_feat,)


        img_size = (opts.img_height * 1.0, opts.img_width * 1.0)


    def forward(self, feats):
        return_stuff = {}
        
        img = feats['img']
        unet_output = self.unet_gen.forward(img)
        img_feat = self.img_encoder.forward(img)
        uv_map  = unet_output[:, 0:self.uv_pred_dim, :, :]
        mask = torch.sigmoid(unet_output[:, self.uv_pred_dim:, :, :])
        
        return_stuff['seg_mask'] = mask

        uv_map = torch.tanh(uv_map) * (1 - 1E-6)
        uv_map = torch.nn.functional.normalize(uv_map, dim=1, eps=1E-6)
        uv_map_3d = uv_map.permute(0, 2, 3, 1).contiguous()
        uv_map = geom_utils.convert_3d_to_uv_coordinates(uv_map.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        '''
        These check capture if the grad is nan
        '''
        if torch.sum(uv_map != uv_map) > 0:
            pdb.set_trace()
        if torch.max(uv_map) > 1.0:
            print('maximum index should be less that 1.0')
            pdb.set_trace()
        if torch.min(uv_map) < 0.0:
            print('minimum value should be greater that 0.0')
            pdb.set_trace()

        uv_map = uv_map.permute(0, 2, 3, 1).contiguous()

        if self.opts.multiple_cam_hypo:
            cam_sampled, sample_inds, cam_probs, all_cameras, base_quats = self.cam_predictor.forward(img_feat)
            cam = cam_sampled
            return_stuff['cam_hypotheses'] = all_cameras
            return_stuff['base_quats'] = base_quats[:,0]
        else:
            cam = self.cam_predictor.forward(img_feat) ## quat (0:4), prop(4:5), scale(5:6), trans(6:8)
            cam = torch.cat([cam[:,5:6], cam[:, 6:8], cam[:,0:4]],dim=1)
            sample_inds = torch.zeros(cam[:, None, 0].shape).long().cuda()
            cam_probs = sample_inds.float() + 1

        return_stuff['cam_sample_inds'] = sample_inds
        return_stuff['cam_probs'] = cam_probs
        return_stuff['cam'] = cam
        return_stuff['uv_map'] = uv_map
        return_stuff['uv_map_3d'] = uv_map_3d
        return return_stuff