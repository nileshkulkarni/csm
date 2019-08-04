import math
import numpy as np
import torch
from . import transformations


## Input is torch.Tensor 
def quat_dist(pred, gt):
    rot_pred = transformations.quaternion_matrix(pred.numpy())
    rot_gt = transformations.quaternion_matrix(gt.numpy())
    rot_rel = np.matmul(rot_pred, np.transpose(rot_gt))
    quat_rel = transformations.quaternion_from_matrix(rot_rel, isprecise=True)
    angle = math.acos(np.clip(abs(quat_rel[0]), a_min=0.0, a_max=1.0))*360/math.pi
    return angle

## ... x 2 vector
def trans_error(gt_trans, pred_trans):
    return torch.norm(gt_trans - pred_trans, dim=-1).numpy()
    

def scale_error(gt_scale, pred_scale):
    return torch.abs(torch.log(gt_scale) - torch.log(pred_scale)).numpy()