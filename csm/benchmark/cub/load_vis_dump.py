from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
import scipy.io as sio
import scipy.misc
import pdb
from . import visutil

results_path = '/home/nileshk/CorrespNet/icn/nnutils/../cachedir/evaluation/feb22_birds_honest_no_vgg/val/epoch_200/vis_dump2.mat'
database = sio.loadmat(results_path)

mean_shape = database['mean_shape']
data = database['cub']

img = data[0,0]['img'][0,0]*255
img = img.astype(np.uint8)
pdb.set_trace()
scipy.misc.imsave(img[0],'img1.png')

pair_index=1

