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

imnet_class2sysnet = {'horse' : 'n02381460', 'zebra': 'n02391049' , 'bear':'n02131653', 'sheep': 'n10588074', 'cow': 'n01887787'}
flags.DEFINE_string('imnet_dir', '/nfs.yoda/nileshk/CorrespNet/datasets/Imagenet', 'PASCAL Data Directory')
flags.DEFINE_string('imnet_anno_path', osp.join(cache_path, 'imnet'), 'Directory where pascal annotations are saved')
flags.DEFINE_string('imnet_class', 'horse', 'Imagenet category name')
flags.DEFINE_string('imnet_cache_dir', osp.join(cache_path, 'imnet'), 'P3D Data Directory')

opts = flags.FLAGS

# -------------- Dataset ------------- #
# ------------------------------------ #


class ImnetDataset(base_data.BaseDataset):
    ''' 
    Imnet Data loader
    '''


    def __init__(self, opts,):
        super(ImnetDataset, self).__init__(opts,)
        sysnetId = imnet_class2sysnet[opts.imnet_class]
        self.img_dir = osp.join(opts.imnet_dir, 'ImageSets',sysnetId)
        self.data_cache_dir = opts.imnet_cache_dir
        imnet_cache_dir =  osp.join(opts.imnet_cache_dir, sysnetId)
        

        self.anno_path = osp.join(self.data_cache_dir, 'data', '{}_{}.mat'.format(sysnetId, opts.split))
        self.anno_sfm_path = osp.join(self.data_cache_dir, 'sfm', '{}_{}.mat'.format(sysnetId, opts.split))
        
        self.anno = sio.loadmat(
            self.anno_path, struct_as_record=False, squeeze_me=True)['images']
        self.anno_sfm = sio.loadmat(
            self.anno_sfm_path, struct_as_record=False, squeeze_me=True)['sfm_anno']

        self.kp_perm = np.linspace(0, 9, 10).astype(np.int)
        self.kp_names = ['lpsum' for _ in range(len(self.kp_perm))]
        self.kp_uv = np.random.uniform(0,1, (len(self.kp_perm), 2))
        opts.num_kps = len(self.kp_perm)
        self.num_imgs = len(self.anno)
        print('%d images' % self.num_imgs)
        self.flip = opts.flip
        return

def imnet_dataloader(opts, shuffle=True):
    dset = ImnetDataset(opts)
    return DataLoader(
        dset,
        batch_size=opts.batch_size,
        shuffle=shuffle,
        num_workers=opts.n_data_workers,
        pin_memory=True, collate_fn=base_data.collate_fn)


class ImnetTestDataset(Dataset):

    def __init__(self, opts, filter_key):
        self.filter_key = filter_key
        sdset = ImnetDataset(opts)
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
        b1 = self.sdset[i1]
        b2 = self.sdset[i2]
        # elem = {'pair' : [b1, b2]}
        if self.filter_key==1:
            return b1
        else:
            return b2


def imnet_test_pair_dataloader(opts, filter_key, shuffle=False):
    dset = ImnetTestDataset(opts, filter_key)
    # dset = d_set_func(opts, filter_key=filter_key)
    return DataLoader(
        dset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=opts.n_data_workers,
        pin_memory=True, collate_fn=base_data.collate_fn)


class ImnetPairDataset(Dataset):
    def __init__(self, opts):
        self.opts = opts
        self.dset = dset = ImnetDataset(opts)
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


def imnet_pair_dataloader(opts, shuffle=True):
    dset = ImnetPairDataset(opts)
    return DataLoader(dset,
        batch_size=opts.batch_size,
        shuffle=shuffle,
        num_workers=opts.n_data_workers,
        pin_memory=True, collate_fn=base_data.collate_fn)

