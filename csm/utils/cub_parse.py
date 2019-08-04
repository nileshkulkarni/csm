from __future__ import division
from __future__ import print_function
import copy
import csv
import json
import numpy as np
import scipy.linalg
import scipy.io as sio
import os
import os.path as osp
import cPickle as pickle
import cPickle as pkl
import torch
from torch.autograd import Variable
from . import transformations
import pdb
from collections import defaultdict
import math
import torch.nn as nn
from ..nnutils import geom_utils


def append_bindex(kp_index):
    bIndex = torch.LongTensor(list(range(0,len(kp_index))))
    bIndex = bIndex[:, None, None]
    bIndex = bIndex.expand(kp_index[:,:,None,0].shape).type(kp_index.type())
    kp_index = torch.cat([bIndex, kp_index], dim=-1)
    return kp_index


def get_sample_grid(img_size):
    x = torch.linspace(-1, 1, img_size[1]).view(1, -1).repeat(img_size[0],1)
    y = torch.linspace(-1, 1, img_size[0]).view(-1, 1).repeat(1, img_size[1])
    grid = torch.cat((x.unsqueeze(2), y.unsqueeze(2)), 2)
    grid.unsqueeze(0)
    return grid


def collate_pair_batch(examples):
    batch = {}
    for key in examples[0]:
        if key in ['kp_uv', 'img', 'inds', 'neg_inds', 'mask', 'kp', 'pos_inds', 'sfm_pose', 'anchor']:
            batch[key] = torch.cat([examples[i][key] for i in range(len(examples))], dim=0)
    return batch




def normalize(point_3d):
    return point_3d/(1E-10 + np.linalg.norm(point_3d))


class UVTo3D(nn.Module):
    def __init__(self, mean_shape):
        super(UVTo3D, self).__init__()
        self.mean_shape = mean_shape
        self.verts_3d = mean_shape['verts']
        self.verts_uv = mean_shape['uv_verts']


    def compute_barycentric_coordinates(self, uv_verts, uv_points):
        verts = geom_utils.convert_uv_to_3d_coordinates(uv_verts)
        points = geom_utils.convert_uv_to_3d_coordinates(uv_points)
        vertA = verts[:,0,: ]
        vertB = verts[:,1,: ]
        vertC = verts[:,2,:]

        AB = vertB - vertA
        AC  = vertC - vertA
        BC  = vertC - vertB

        AP = points - vertA
        BP = points - vertB
        CP = points - vertC
        areaBAC = torch.norm(torch.cross(AB, AC, dim=1), dim=1)
        areaBAP = torch.norm(torch.cross(AB, AP, dim=1), dim=1)
        areaCAP = torch.norm(torch.cross(AC, AP, dim=1), dim=1)
        areaCBP = torch.norm(torch.cross(BC, BP, dim=1), dim=1)

        w = areaBAP/areaBAC
        v = areaCAP/areaBAC
        u = areaCBP/areaBAC
        barycentric_coordinates = torch.stack([u, v, w], dim=1)
        barycentric_coordinates = torch.nn.functional.normalize(barycentric_coordinates, p=1)
        return barycentric_coordinates



    def forward(self, uv):
        mean_shape = self.mean_shape
        uv_map_size = torch.Tensor([mean_shape['uv_map'].shape[1]-1, mean_shape['uv_map'].shape[0] -1]).view(1,2)
        uv_map_size = uv_map_size.type(uv.type())
        uv_inds = (uv_map_size * uv).round().long().detach()
        if torch.max(uv_inds) > 1000  or torch.min(uv_inds) < 0:
            print('Error in indexing')
            pdb.set_trace()
        face_inds = mean_shape['face_inds'][uv_inds[:,1], uv_inds[:,0]] 
        ## remember this. swaped on purpose. U is along the columns, V is along the rows
        face_vert_inds = mean_shape['faces'][face_inds,:]
        verts =  self.verts_3d 
        uv_verts = self.verts_uv
        face_verts = torch.stack([verts[face_vert_inds[:,0]], verts[face_vert_inds[:,1]], verts[face_vert_inds[:,2]]], dim=1)
        face_uv_verts = torch.stack([uv_verts[face_vert_inds[:,0]], uv_verts[face_vert_inds[:,1]], uv_verts[face_vert_inds[:,2]]], dim=1)
        bary_cord = self.compute_barycentric_coordinates(face_uv_verts, uv)
        # bary_cord = mean_shape['bary_cord'][uv_inds[:,0], uv_inds[:,1]]
        points3d = face_verts * bary_cord[:, :, None]
        points3d = points3d.sum(1)
        return points3d

    def set_3d_verts(self, verts_3d, verts_uv=None):
        assert verts_3d.shape == self.verts_3d.shape, 'shape does not match'
        self.verts_3d = verts_3d
        
        if verts_uv is not None:
            self.uv_verts = verts_uv
        return

def load_mean_shape(mean_shape_path, device='cuda:0'):
    if type(mean_shape_path) == str:
        mean_shape = sio.loadmat(mean_shape_path)
    else:
        mean_shape = mean_shape_path
    # mean_shape['bary_cord'] = torch.from_numpy(mean_shape['bary_cord']).float().to(device)
    mean_shape['uv_map'] = torch.from_numpy(mean_shape['uv_map']).float().to(device)
    mean_shape['uv_verts'] = torch.from_numpy(mean_shape['uv_verts']).float().to(device)
    mean_shape['verts'] = torch.from_numpy(mean_shape['verts']).float().to(device)
    mean_shape['sphere_verts'] = torch.from_numpy(mean_shape['sphere_verts']).float().to(device)
    mean_shape['face_inds'] = torch.from_numpy(mean_shape['face_inds']).long().to(device)
    mean_shape['faces'] = torch.from_numpy(mean_shape['faces']).long().to(device)
    return mean_shape