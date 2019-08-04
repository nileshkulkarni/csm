"""
Data loader for pascal VOC categories.
Should output:
    - img: B X 3 X H X W
    - kp: B X nKp X 2
    - mask: B X H X W
    - sfm_pose: B X 7 (s, tr, q)
    (kp, sfm_pose) correspond to image coordinates in [-1, 1]
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
import pdb
import scipy.io as sio
from absl import flags, app

import torch
from torch.utils.data.dataloader import default_collate
import itertools
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from ..utils import transformations

from . import base as base_data
# -------------- flags ------------- #
# ---------------------------------- #
curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'cachedir')

flags.DEFINE_string('p3d_dir', '/nfs.yoda/nileshk/CorrespNet/datasets/PASCAL3D+_release1.1', 'PASCAL Data Directory')
flags.DEFINE_string('p3d_anno_path', osp.join(cache_path, 'p3d'), 'Directory where pascal annotations are saved')
flags.DEFINE_string('p3d_class', 'car', 'PASCAL VOC category name')
flags.DEFINE_string('p3d_cache_dir', osp.join(cache_path, 'p3d'), 'P3D Data Directory')

opts = flags.FLAGS

# -------------- Dataset ------------- #
# ------------------------------------ #


class P3dDataset(base_data.BaseDataset):
    ''' 
    VOC Data loader
    '''
    def __init__(self, opts,):
        super(P3dDataset, self).__init__(opts,)
        self.img_dir = osp.join(opts.p3d_dir, 'Images')
        self.data_cache_dir = opts.p3d_cache_dir
        self.anno_path = osp.join(self.data_cache_dir, 'data', '{}_{}.mat'.format(opts.p3d_class, opts.split))
        self.anno_sfm_path = osp.join(self.data_cache_dir, 'sfm', '{}_{}.mat'.format(opts.p3d_class, opts.split))
        self.anno_train_sfm_path = osp.join(self.data_cache_dir, 'sfm', '{}_{}.mat'.format(opts.p3d_class, 'train'))

        self.kp_path = osp.join(
            opts.p3d_anno_path, 'data', '{}_kps.mat'.format(opts.p3d_class))
        self.anno_path = osp.join(
            opts.p3d_anno_path, 'data', '{}_{}.mat'.format(opts.p3d_class, opts.split))
        self.anno_sfm_path = osp.join(
            opts.p3d_anno_path, 'sfm', '{}_{}.mat'.format(opts.p3d_class, opts.split))

        # Load the annotation file.
        self.anno = sio.loadmat(
            self.anno_path, struct_as_record=False, squeeze_me=True)['images']
        self.anno_sfm = sio.loadmat(
            self.anno_sfm_path, struct_as_record=False, squeeze_me=True)['sfm_anno']
        self.kp_perm = sio.loadmat(
            self.kp_path, struct_as_record=False, squeeze_me=True)['kp_perm_inds'] - 1
        self.kp_names = sio.loadmat(
            self.kp_path, struct_as_record=False, squeeze_me=True)['kp_names'].tolist()
        # pdb.set_trace()
        # self.kp_perm = np.array([5, 6, 7, 8, 1, 2, 3, 4, 11, 12, 9, 10]) - 1
        # self.kp_names = ['left_back_trunk', 'right_back_wheel', 'right_front_light',
        #                  'right_front_wheel', 'right_back_trunk', 'left_back_wheel', 'left_front_light',
        #                  'left_front_wheel',  'upper_left_rearwindow', 'upper_right_windshield',
        #                  'upper_right_rearwindow',    'upper_left_windshield']
        self.kp3d = sio.loadmat(self.anno_train_sfm_path, struct_as_record=False,
                                squeeze_me=True)['S'].transpose().copy()

        opts.num_kps = len(self.kp_perm)
        self.num_imgs = len(self.anno)
        print('%d images' % self.num_imgs)

       
        self.mean_shape = sio.loadmat(osp.join(opts.p3d_cache_dir, 'uv',
                                               '{}_mean_shape.mat'.format(opts.p3d_class)))

        self.kp_uv = self.preprocess_to_find_kp_uv(self.kp3d, self.mean_shape['faces'],
                                                   self.mean_shape['verts'],
                                                   self.mean_shape['sphere_verts'])

        self.flip = opts.flip


def p3d_dataloader(opts, shuffle=True):
    dset = P3dDataset(opts)
    return DataLoader(
        dset,
        batch_size=opts.batch_size,
        shuffle=shuffle,
        num_workers=opts.n_data_workers,
        pin_memory=True, collate_fn=base_data.collate_fn)


class P3DTestDataset(Dataset):

    def __init__(self, opts, filter_key):
        self.filter_key = filter_key
        sdset = P3dDataset(opts)
        count = opts.number_pairs
        all_indices = [i for i in range(len(sdset))]
        rng = np.random.RandomState(len(sdset))
        pairs = zip(rng.choice(all_indices, count), rng.choice(all_indices, count))
        self.sdset = sdset
        self.tuples = pairs

    def __len__(self,):
        return len(self.tuples)

    def __getitem__(self, index):
        i1, i2 = self.tuples[index]
        # i1 = 100
        # i2 = 100
        b1 = self.sdset[i1]
        b2 = self.sdset[i2]
        # elem = {'pair' : [b1, b2]}
        if self.filter_key==1:
            return b1
        else:
            return b2


def p3d_test_pair_dataloader(opts, filter_key, shuffle=False):
    dset = P3DTestDataset(opts, filter_key)
    # dset = d_set_func(opts, filter_key=filter_key)
    return DataLoader(
        dset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=opts.n_data_workers,
        pin_memory=True, collate_fn=base_data.collate_fn)


class P3dPairDataset(Dataset):
    def __init__(self, opts):
        self.opts = opts
        self.dset = dset = P3dDataset(opts)
        all_indices = [i for i in range(len(dset))]
        rng = np.random.RandomState(len(dset))
        pairs = list(itertools.combinations(all_indices,2))
        rng.shuffle(pairs)
        self.tuples = pairs
        return

    def __len__(self, ):
        return len(self.tuples)

    def __getitem__(self, index):
        opts = self.opts
        i1, i2 = self.tuples[index]

        b1 = self.dset[i1]
        b2 = self.dset[i2]

        elem = {}

        elem['img1'] = b1['img']
        elem['img2'] = b2['img']
        elem['kp1'] = b1['kp']
        elem['kp2'] = b2['kp']
        elem['ind_kp1'] = np.clip(np.round((b1['kp']+1)*0.5*opts.img_size).astype(np.int) ,a_min=0, a_max=opts.img_size-1)
        elem['ind_kp2'] = np.clip(np.round((b2['kp']+1)*0.5*opts.img_size).astype(np.int) ,a_min=0, a_max=opts.img_size-1)

        common_vis = (b1['kp'][:,2]  + b2['kp'][:,2] > 1.5)
        elem['common_vis'] = np.array(common_vis).astype(np.float32)
        return elem


def p3d_pair_dataloader(opts, shuffle=True):
    dset = P3dPairDataset(opts)
    return DataLoader(dset,
        batch_size=opts.batch_size,
        shuffle=shuffle,
        num_workers=opts.n_data_workers,
        pin_memory=True, collate_fn=base_data.collate_fn)

