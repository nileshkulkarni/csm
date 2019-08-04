from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
import collections

import scipy.misc
import scipy.linalg
import scipy.io as sio
import scipy.ndimage.interpolation
from absl import flags
import cPickle as pkl
import torch
import multiprocessing
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import pdb
from datetime import datetime
import sys
from ..utils import cub_parse
import numpy as np
import pdb
import pymesh
import re
import scipy.misc
from ..utils import render_utils
from ..utils import image as image_utils
from ..utils import transformations
from ..nnutils import geom_utils
from . import base as base_data
import itertools

flags.DEFINE_string('cub_dir', '/nfs.yoda/nileshk/CorrespNet/datasets/cubs/', 'CUB Data Directory')
curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'cachedir')

flags.DEFINE_string('cub_cache_dir', osp.join(cache_path, 'cub'), 'CUB Data Directory')


class CubDataset(base_data.BaseDataset):

    def __init__(self, opts):
        super(CubDataset, self).__init__(opts,)
        self.data_dir = opts.cub_dir
        self.data_cache_dir = opts.cub_cache_dir
        self.opts = opts
        self.img_dir = osp.join(self.data_dir, 'images')
        self.anno_path = osp.join(self.data_cache_dir, 'data', '%s_cub_cleaned.mat' % opts.split)
        self.anno_sfm_path = osp.join(self.data_cache_dir, 'sfm', 'anno_%s.mat' % opts.split)
        self.anno_train_sfm_path = osp.join(self.data_cache_dir, 'sfm', 'anno_%s.mat' % 'train')
        self.jitter_frac = opts.jitter_frac
        self.padding_frac = opts.padding_frac
        self.img_size = opts.img_size
        if not osp.exists(self.anno_path):
            print('%s doesnt exist!' % self.anno_path)
            import ipdb
            ipdb.set_trace()

        # Load the annotation file.
        print('loading %s' % self.anno_path)
        self.anno = sio.loadmat(
            self.anno_path, struct_as_record=False, squeeze_me=True)['images']
        self.anno_sfm = sio.loadmat(
            self.anno_sfm_path, struct_as_record=False, squeeze_me=True)['sfm_anno']

        self.kp3d = sio.loadmat(self.anno_train_sfm_path, struct_as_record=False,
                                squeeze_me=True)['S'].transpose().copy()
        self.num_imgs = len(self.anno)
        print('%d images' % self.num_imgs)
        self.kp_perm = np.array([1, 2, 3, 4, 5, 6, 11, 12, 13, 10, 7, 8, 9, 14, 15]) - 1
        self.kp_names = ['Back', 'Beak', 'Belly', 'Breast', 'Crown', 'FHead', 'LEye',
                         'LLeg', 'LWing', 'Nape', 'REye', 'RLeg', 'RWing', 'Tail', 'Throat']
        self.mean_shape = sio.loadmat(osp.join(opts.cub_cache_dir, 'uv', 'mean_shape.mat'))
        self.kp_uv = self.preprocess_to_find_kp_uv(self.kp3d, self.mean_shape['faces'], self.mean_shape[
                                                   'verts'], self.mean_shape['sphere_verts'])
        self.flip = opts.flip
        return


class CubTestDataset(Dataset):

    def __init__(self, opts, filter_key):
        self.filter_key = filter_key
        sdset = CubDataset(opts)
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
        # i1 = 1452
        # i2 = 1269
        b1 = self.sdset[i1]
        b2 = self.sdset[i2]

        # elem = {'pair' : [b1, b2]}
        if self.filter_key==1:
            return b1
        else:
            return b2



def cub_dataloader(opts, shuffle=True):
    dset = CubDataset(opts)
    # dset = d_set_func(opts, filter_key=filter_key)
    return DataLoader(
        dset,
        batch_size=opts.batch_size,
        shuffle=shuffle,
        num_workers=opts.n_data_workers,
        pin_memory=True, collate_fn=base_data.collate_fn)


def cub_test_pair_dataloader(opts, filter_key, shuffle=False):
    dset = CubTestDataset(opts, filter_key)
    # dset = d_set_func(opts, filter_key=filter_key)
    return DataLoader(
        dset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=opts.n_data_workers,
        pin_memory=True, collate_fn=base_data.collate_fn)


def cub_dataset(opts):
    dset = CubDataset(opts)

    class DataIter():
        def __init__(self, dset, collate_fn):
            self.collate_fn = collate_fn
            self.dset = dset
            return

        def __len__(self,):
            return len(self.dset)

        def __getitem__(self, index):
            example = dset[index]
            return self.collate_fn([example])

    return DataIter(dset, base_data.collate_fn)


