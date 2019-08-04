

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import sys
import os.path as osp
import os
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import json
import scipy.io as sio
import cPickle as pickle

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..','..', 'cachedir')

##Edit this path
imageNetDir = "/nfs.yoda/nileshk/CorrespNet/datasets/Imagenet"

imageNetCacheDir = osp.join(cache_path, 'imnet')

## Choose the apporpriate class.

# sysnetId = "n02391049" ## Zebra
sysnetId = "n02381460" ## horse
# sysnetId = "n10588074" ## Sheep
# sysnetId = "n01887787" ## Cow
# sysnetId = "n02131653" ## Bear
imgs_dir = osp.join(imageNetDir, 'ImageSets')
mask_dir = osp.join(imageNetDir, 'Masks')
split_name = 'val'


quality_anno_file = osp.join(imageNetCacheDir, sysnetId, '{}_quality.json'.format(sysnetId))
with open(quality_anno_file, 'r') as f:
    annotations = json.load(f)


imgIds = annotations.keys()
split_file = osp.join(imageNetCacheDir, sysnetId,  'split.pkl')

if osp.exists(split_file) and False:
    with open(split_file, 'rb') as f:
        splits = pickle.load(f)
else:
    splitRng = np.random.RandomState(100)
    splitRng.shuffle(imgIds)
    ## train = 80, val = 20
    num_imgs = len(imgIds)
    num_train = int(num_imgs*0.8)
    splits = {'train' : imgIds[0:num_train],
                'val' : imgIds[num_train:]
                }
    with open(split_file, 'wb') as f:
        pickle.dump(splits, f)

pdb.set_trace()
imgIds =  splits[split_name]
anno_file = osp.join(imageNetCacheDir,'data', "{}_{}.mat".format(sysnetId, split_name))
anno_sfm_file = osp.join(imageNetCacheDir,'sfm', "{}_{}.mat".format(sysnetId, split_name))


default_parts = np.random.uniform(0,1, (10,3))
default_parts[:,2] = np.maximum(np.sign(default_parts[:,2] -.5), 0*default_parts[:,2])

default_sfm = {'scale' : np.array([[52.0]]), 'trans' : np.array([[180., 210.]]), 'rot' : np.eye(3)}
# data_annotations = {}
# sfm_annotations = {}

data_annotations = []
sfm_annotations = []
# imgIds = imgIds[0:10]
counter = 0
for imgId in imgIds:
    img_anno = annotations[imgId]
    mask_file  = osp.join(mask_dir, sysnetId, "{}.mat".format(imgId))
    mask_data = sio.loadmat(mask_file)
    for obj_key in img_anno.keys():
        if obj_key == 'good':
            continue

        if not img_anno[obj_key]:
            continue
        data = {}
        data['rel_path'] = imgId
        data['bbox'] = {}
        obj_ind = int(obj_key)
        bbox = mask_data['boxes'][obj_ind]
        mask = mask_data['masks'][:,:, obj_ind]
        data['bbox'] = {'x1' : bbox[0],
                        'y1' : bbox[1],
                        'x2' : bbox[2],
                        'y2' : bbox[3],
                        }
        data['mask'] = mask
        parts = default_parts.copy()
        parts[:,0] = parts[:,0]* mask.shape[1]
        parts[:,1] = parts[:,1]* mask.shape[0]
        
        data['parts'] = {}
        data['parts']['T'] = parts
        data['imgId'] = imgId
        data['obj_key'] = obj_key
        data_annotations.append(data)
        sfm_annotations.append(default_sfm)
        
        # data_annotations[str(counter)] = data
        # sfm_annotations[str(counter)] = default_sfm
        counter += 1

pdb.set_trace()
sio.savemat(anno_file,{'images' : data_annotations})
sio.savemat(anno_sfm_file, {'sfm_anno' : sfm_annotations})
