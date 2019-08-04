#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""
'''
export PYTHONPATH=$PYTHONPATH:/home/nileshk/caffe2/detectron/
python instance_segment_2.py --cfg /home/nileshk/caffe2/detectron/configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml
  --output-dir /nfs.yoda/nileshk/CorrespNet/datasets/PASCAL3D+_release1.1/masks/car_imagenet --image-ext
JPEG --wts https://s3-us-west-2.amazonaws.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl /nfs.yoda/nileshk/CorrespNet/datasets/PASCAL3D+_release1.1/Images/car_imagenet/
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import numpy as np
import scipy.io as sio
from caffe2.python import workspace


import pycocotools.coco as coco
anno_file = '/home/nileshk/caffe2/cocoapi/data/annotations/instances_val2017.json'
coco_data = coco.COCO(anno_file)
coco_classes = coco_data.loadCats(coco_data.getCatIds())
coco_classes = [cat['name'] for cat in coco_classes]

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

def coco_to_pascal_name(category):
    if category == 'airplane':
        return 'aeroplane'
    if category == 'dining table':
        return 'diningtable'
    if category == 'motorcycle':
        return 'motorbike'
    if category == 'couch':
        return 'sofa'
    if category == 'tv':
        return 'tvmonitor'
    else:
        return category



def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--always-out',
        dest='out_when_no_box',
        help='output image even when no object is found',
        action='store_true'
    )
    parser.add_argument(
        '--output-ext',
        dest='output_ext',
        help='output image file format (default: pdf)',
        default='pdf',
        type=str
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='Threshold for visualizing detections',
        default=0.7,
        type=float
    )
    parser.add_argument(
        '--kp-thresh',
        dest='kp_thresh',
        help='Threshold for visualizing keypoints',
        default=2.0,
        type=float
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):
    logger = logging.getLogger(__name__)

    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)

    assert not cfg.MODEL.RPN_ONLY, \
        'RPN models are not supported'
    assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
        'Models that require precomputed proposals are not supported'

    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    for i, im_name in enumerate(im_list):
        #if i > 10:
        #    continue
        out_name = os.path.join(
            args.output_dir, '{}'.format(os.path.basename(im_name)[:-len(args.image_ext)] + 'mat')
        )
        
        im = cv2.imread(im_name)
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )
        boxes, segms, keyps, classes = vis_utils.convert_from_cls_format(cls_boxes, cls_segms, cls_keyps)
        if boxes is None:
            continue

        segms = vis_utils.mask_util.decode(segms)

        valid_inds = np.greater(boxes[:, 4], 0.5)
        boxes = boxes[valid_inds, :]
        segms = segms[:, :, valid_inds]
        classes = np.array(classes)[valid_inds]
        class_names = np.asarray(([coco_to_pascal_name(coco_classes[c-1]) for c in classes]), dtype='object')

        sio.savemat(out_name, {'masks': segms, 'boxes': boxes, 'classes': class_names});


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
