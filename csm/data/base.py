from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
import pdb
import scipy.misc
import scipy.linalg
import scipy.ndimage.interpolation
from absl import flags, app

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torchvision
from ..utils import image as image_utils
from ..utils import transformations, visutil
import pymesh
from ..nnutils import geom_utils
import cv2

flags.DEFINE_integer('img_size', 256, 'image size')
flags.DEFINE_integer('img_height', 320, 'image height')
flags.DEFINE_integer('img_width',  512, 'image width')
flags.DEFINE_enum('split', 'train', ['train', 'val', 'all', 'test'], 'eval split')
flags.DEFINE_float('padding_frac', 0.05, 'bbox is increased by this fraction of max_dim')
flags.DEFINE_float('jitter_frac', 0.05, 'bbox is jittered by this fraction of max_dim')
flags.DEFINE_boolean('flip', True, 'Allow flip bird left right')
flags.DEFINE_boolean('tight_crop', False, 'Use Tight crops')
flags.DEFINE_boolean('flip_train', True, 'Mirror Images while training')
flags.DEFINE_integer('number_pairs', 10000,
                     'N random pairs from the test to check if the correspondence transfers.')
flags.DEFINE_integer('num_kps', 12, 'Number of keypoints')

class BaseDataset(Dataset):

    def __init__(self, opts):
        # Child class should define/load:
        # self.kp_perm
        # self.img_dir
        # self.anno
        # self.anno_sfm
        self.opts = opts
        self.img_size = opts.img_size
        self.jitter_frac = opts.jitter_frac
        self.padding_frac = opts.padding_frac
        self.rngFlip = np.random.RandomState(0)
        self.flip_transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                        torchvision.transforms.RandomHorizontalFlip(1),
                                        torchvision.transforms.ToTensor()])

        return
    
    def preprocess_to_find_kp_uv(self, kp3d, faces, verts, verts_sphere, ):
        mesh = pymesh.form_mesh(verts, faces)
        dist, face_ind, closest_pts = pymesh.distance_to_mesh(mesh, kp3d)
        dist_to_verts = np.square(kp3d[:, None, :] - verts[None, :, :]).sum(-1)
        closest_pts = closest_pts / np.linalg.norm(closest_pts, axis=1, keepdims=1)
        min_inds = np.argmin(dist_to_verts, axis=1)
        kp_verts_sphere = verts_sphere[min_inds]
        kp_uv = geom_utils.convert_3d_to_uv_coordinates(closest_pts)
        return kp_uv

    def forward_random_pixel_samples(self, img, mask, count, margin):

        def sample_contrast(valid_indices, anchor, positive_count=10, negative_count=10):
            anchor_mask = 0 * mask
            ax_ind, ay_ind = anchor[1], anchor[0]
            valid_pos = np.meshgrid(np.arange(ay_ind - margin, ay_ind + margin, 1),
                                    np.arange(ax_ind - margin, ax_ind + margin, 1))
            valid_pos = np.stack(valid_pos, axis=-1)
            valid_pos = valid_pos.reshape(-1, 2)
            valid_pos = np.clip(valid_pos, 0, mask.shape[0] - 1)
            valid_pos = np.unique(valid_pos, axis=0)
            valid_pos = [v for v in valid_pos if mask[v[0], v[1]] > 0]
            # np.random.shuffle(valid_pos,)
            valid_pos = [valid_pos[i] for i in np.random.choice(np.arange(0, len(valid_pos), 1), positive_count)]
            valid_neg = np.array([valid_indices[i] for (i, v) in enumerate(
                valid_indices - anchor.reshape(1, 2)) if max(abs(v[0]), abs(v[1])) > 2 * margin])
            # np.random.shuffle(valid_neg,)
            valid_neg = [valid_neg[i] for i in np.random.choice(np.arange(0, len(valid_neg), 1), negative_count)]

            return valid_pos, valid_neg

        def sample_anchors(valid_indices, count):
            inds = np.random.choice(np.arange(0, len(valid_indices)), count, replace=False)
            return valid_indices[inds]

        valid_indices = np.where(mask)
        valid_indices = np.array(zip(*valid_indices))
        anchor_inds = sample_anchors(valid_indices, count)
        positive_samples = []
        negative_samples = []
        for aix in range(len(anchor_inds)):
            pos_samples, neg_samples = sample_contrast(valid_indices, anchor_inds[aix],)
            positive_samples.append(pos_samples)
            negative_samples.append(neg_samples)

        positive_samples = np.stack(positive_samples)
        negative_samples = np.stack(negative_samples)
        return anchor_inds, positive_samples, negative_samples

    def forward_img(self, index):

        data = self.anno[index]
        data_sfm = self.anno_sfm[index]
                # sfm_pose = (sfm_c, sfm_t, sfm_r)
        sfm_pose = [np.copy(data_sfm.scale), np.copy(data_sfm.trans), np.copy(data_sfm.rot)]

        sfm_rot = np.pad(sfm_pose[2], (0, 1), 'constant')
        sfm_rot[3, 3] = 1
        sfm_pose[2] = transformations.quaternion_from_matrix(sfm_rot, isprecise=True)

        img_path = osp.join(self.img_dir, str(data.rel_path))
        img = scipy.misc.imread(img_path) / 255.0
        # Some are grayscale:
        if len(img.shape) == 2:
            img = np.repeat(np.expand_dims(img, 2), 3, axis=2)
        mask = np.expand_dims(data.mask, 2)

        # Adjust to 0 indexing
        bbox = np.array(
            [data.bbox.x1, data.bbox.y1, data.bbox.x2, data.bbox.y2],
            float) - 1

        parts = data.parts.T.astype(float)
        kp = np.copy(parts)
        vis = kp[:, 2] > 0
        kp[vis, :2] -= 1
        kp_uv = self.kp_uv.copy()
        # Peturb bbox

        if self.opts.tight_crop:
            self.padding_frac = 0.0

        if self.opts.split == 'train':
            bbox = image_utils.peturb_bbox(
                bbox, pf=self.padding_frac, jf=self.jitter_frac)
        else:
            bbox = image_utils.peturb_bbox(
                bbox, pf=self.padding_frac, jf=0)
        if self.opts.tight_crop:
            bbox = bbox
        else: 
            bbox = image_utils.square_bbox(bbox)

        # crop image around bbox, translate kps
        img, mask, kp, sfm_pose = self.crop_image(img, mask, bbox, kp, vis, sfm_pose)


        # scale image, and mask. And scale kps.
        if self.opts.tight_crop:
            img, mask, kp, sfm_pose = self.scale_image_tight(img, mask, kp, vis, sfm_pose)
        else:
            img, mask, kp, sfm_pose = self.scale_image(img, mask, kp, vis, sfm_pose)

        
        # Mirror image on random.
        if self.opts.split == 'train':
            img, mask, kp, kp_uv, sfm_pose = self.mirror_image(img, mask, kp, kp_uv, sfm_pose)

        # Normalize kp to be [-1, 1]
        img_h, img_w = img.shape[:2]
        kp_norm, sfm_pose = self.normalize_kp(kp, sfm_pose, img_h, img_w)

        # Finally transpose the image to 3xHxW
        img = np.transpose(img, (2, 0, 1))

        return img, kp_norm, kp_uv, mask, sfm_pose

    def normalize_kp(self, kp, sfm_pose, img_h, img_w):
        vis = kp[:, 2, None] > 0
        new_kp = np.stack([2 * (kp[:, 0] / img_w) - 1,
                           2 * (kp[:, 1] / img_h) - 1,
                           kp[:, 2]]).T
        sfm_pose[0] *= (1.0 / img_w + 1.0 / img_h)
        sfm_pose[1][0] = 2.0 * (sfm_pose[1][0] / img_w) - 1
        sfm_pose[1][1] = 2.0 * (sfm_pose[1][1] / img_h) - 1
        new_kp = vis * new_kp

        return new_kp, sfm_pose

    def crop_image(self, img, mask, bbox, kp, vis, sfm_pose):
        # crop image and mask and translate kps
        img = image_utils.crop(img, bbox, bgval=1)
        mask = image_utils.crop(mask, bbox, bgval=0)
        kp[vis, 0] -= bbox[0]
        kp[vis, 1] -= bbox[1]
        
        kp[vis,0] = np.clip(kp[vis,0], a_min=0, a_max=bbox[2] -bbox[0])
        kp[vis,1] = np.clip(kp[vis,1], a_min=0, a_max=bbox[3] -bbox[1])
        
        sfm_pose[1][0] -= bbox[0]
        sfm_pose[1][1] -= bbox[1]

        return img, mask, kp, sfm_pose
    
    def scale_image_tight(self, img, mask, kp, vis, sfm_pose):
        # Scale image so largest bbox size is img_size
        bwidth = np.shape(img)[1]
        bheight = np.shape(img)[0]

        scale_x = self.img_size/bwidth
        scale_y = self.img_size/bheight

        # scale = self.img_size / float(max(bwidth, bheight))
        # pdb.set_trace()
        img_scale = cv2.resize(img, (self.img_size, self.img_size))
        # img_scale, _ = image_utils.resize_img(img, scale)
        # if img_scale.shape[0] != self.img_size:
        #     print('bad!')
        #     import ipdb; ipdb.set_trace()
        # mask_scale, _ = image_utils.resize_img(mask, scale)

        mask_scale = cv2.resize(mask, (self.img_size, self.img_size))
        
        kp[vis, 0:1] *= scale_x
        kp[vis, 1:2] *= scale_y
        sfm_pose[0] *= scale_x
        sfm_pose[1] *= scale_y

        return img_scale, mask_scale, kp, sfm_pose

    def scale_image(self, img, mask, kp, vis, sfm_pose):
        # Scale image so largest bbox size is img_size
        bwidth = np.shape(img)[0]
        bheight = np.shape(img)[1]
        scale = self.img_size / float(max(bwidth, bheight))
        img_scale, _ = image_utils.resize_img(img, scale)
        # if img_scale.shape[0] != self.img_size:
        #     print('bad!')
        #     import ipdb; ipdb.set_trace()
        mask_scale, _ = image_utils.resize_img(mask, scale)
        kp[vis, :2] *= scale
        sfm_pose[0] *= scale
        sfm_pose[1] *= scale

        return img_scale, mask_scale, kp, sfm_pose

    def mirror_image(self, img, mask, kp, kp_uv, sfm_pose):
        kp_perm = self.kp_perm
        if self.rngFlip.rand(1) > 0.5 and self.flip:
            # Need copy bc torch collate doesnt like neg strides
            img_flip = img[:, ::-1, :].copy()
            mask_flip = mask[:, ::-1].copy()

            # Flip kps.
            new_x = img.shape[1] - kp[:, 0] - 1
            kp_flip = np.hstack((new_x[:, None], kp[:, 1:]))
            kp_flip = kp_flip[kp_perm, :]
            kp_uv_flip = kp_uv[kp_perm, :]
            # Flip sfm_pose Rot.
            R = transformations.quaternion_matrix(sfm_pose[2])
            flip_R = np.diag([-1, 1, 1, 1]).dot(R.dot(np.diag([-1, 1, 1, 1])))
            sfm_pose[2] = transformations.quaternion_from_matrix(flip_R, isprecise=True)
            # Flip tx
            tx = img.shape[1] - sfm_pose[1][0] - 1
            sfm_pose[1][0] = tx
            return img_flip, mask_flip, kp_flip, kp_uv_flip, sfm_pose
        else:
            return img, mask, kp, kp_uv, sfm_pose

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, index):
        # if index == 1452:
        #     pdb.set_trace()

        img, kp, kp_uv, mask, sfm_pose = self.forward_img(index)
        anchor, positive_samples, negative_samples = self.forward_random_pixel_samples(img, mask, count=10, margin=10)
        sfm_pose[0].shape = 1
        elem = {
            'img': img,
            'kp': kp,
            'kp_uv': kp_uv,
            'mask': mask,
            'sfm_pose': np.concatenate(sfm_pose),  # scale (1), trans (2), quat(4)
            'inds': np.array([index]),
            # 'anchor': anchor,
            # 'pos_inds': positive_samples,
            # 'neg_inds': negative_samples,
        }
        if self.opts.flip_train:
            # flip_img = self.flip_transform((img.transpose(1,2,0)*255).astype(np.uint8)) 
            flip_img  = img[:, :, ::-1].copy()  
            elem['flip_img'] = flip_img
            # elem['flip_img'] = img[:,:,-1::-1].copy()
            # flip_mask = self.flip_transform((mask[None, :, :].transpose(1,2,0)*225).astype(np.uint8))
            flip_mask = mask[:, ::-1].copy()
            elem['flip_mask'] = flip_mask

        return elem





def collate_fn(batch):
    '''Globe data collater.

    Assumes each instance is a dict.
    Applies different collation rules for each field.

    Args:
        batch: List of loaded elements via Dataset.__getitem__
    '''
    collated_batch = {'empty': True}
    # iterate over keys
    # new_batch = []
    # for valid,t in batch:
    #     if valid:
    #         new_batch.append(t)
    #     else:
    #         'Print, found a empty in the batch'

    # # batch = [t for t in batch if t is not None]
    # # pdb.set_trace()
    # batch = new_batch
    if len(batch) > 0:
        for key in batch[0]:
            collated_batch[key] = default_collate([elem[key] for elem in batch])
        collated_batch['empty'] = False
    return collated_batch
