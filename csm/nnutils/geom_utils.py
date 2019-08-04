"""
Code taken from https://github.com/akanazawa/cmr/blob/master/nnutils/geom_utils.py
Utils related to geometry like projection,,
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import pdb
from torch import nn
import numpy as np
import multiprocessing



'''
uv1 --> N x 2
uv2 --> M x 2
distance = N x M 
'''
def compute_distance_in_uv_sapce(uv1, uv2):
    uv1_3d = convert_uv_to_3d_coordinates(uv1)
    uv2_3d = convert_uv_to_3d_coordinates(uv2)
    pwd = ((uv1_3d[:,None,:] - uv2_3d[None, :,:])**2).sum(-1)
    return pwd

'''
B x H x W x 2
'''

def project_uv_to_3d(uv2points, uv_map):
    B = uv_map.size(0)
    H = uv_map.size(1)
    W = uv_map.size(2)
    uv_map_flatten = uv_map.view(-1, 2)
    points3d = uv2points.forward(uv_map_flatten)
    points3d = points3d.view(B, H*W, 3)
    return points3d

def project_3d_to_image(points3d, cam, offset_z):
    projected_points = orthographic_proj_withz(points3d, cam, offset_z)
    return projected_points

'''
Takes a uv coordinate between [0,1] and returns a 3d point on the sphere.
uv -- > [......, 2] shape
'''
def convert_uv_to_3d_coordinates(uv):
    phi = 2*np.pi*(uv[..., 0] - 0.5)
    theta = np.pi*(uv[...,1] - 0.5)

    if type(uv) == torch.Tensor:
        x = torch.cos(theta)*torch.cos(phi)
        y = torch.cos(theta)*torch.sin(phi)
        z = torch.sin(theta)
        points3d = torch.stack([x,y,z], dim=-1)
    else:
        x = np.cos(theta)*np.cos(phi)
        y = np.cos(theta)*np.sin(phi)
        z = np.sin(theta)
        points3d = np.stack([x,y,z], axis=-1)
    return points3d


'''
Takes a 3D point and returns an uv between [0,1]
3d ---> [......., 3] shape
'''
def convert_3d_to_uv_coordinates(points):
    # eps = 1E-8
    eps = 1E-4
    if type(points) == torch.Tensor:
        rad  = torch.clamp(torch.norm(points, p=2, dim=-1), min=eps)
        
        phi = torch.atan2(points[...,1], points[...,0])
        theta = torch.asin(torch.clamp(points[...,2]/rad, min=-1+eps, max=1-eps))
        u = 0.5 + phi/(2*np.pi)
        v = 0.5 + theta/np.pi
        return torch.stack([u,v],dim=-1)
    else:
        rad  = np.linalg.norm(points, axis=1)
        phi = np.arctan2(points[:,1], points[:,0])
        theta = np.arcsin(points[:,2]/rad)
        u = 0.5 + phi/(2*np.pi)
        v = 0.5 + theta/np.pi
        return np.stack([u,v],axis=1)


def sample_textures(texture_flow, images):
    """
    texture_flow: B x F x T x T x 2
    (In normalized coordinate [-1, 1])
    images: B x 3 x N x N

    output: B x F x T x T x 3
    """
    # Reshape into B x F x T*T x 2
    T = texture_flow.size(-2)
    F = texture_flow.size(1)
    flow_grid = texture_flow.view(-1, F, T * T, 2)
    # B x 3 x F x T*T
    samples = torch.nn.functional.grid_sample(images, flow_grid)
    # B x 3 x F x T x T
    samples = samples.view(-1, 3, F, T, T)
    # B x F x T x T x 3
    return samples.permute(0, 2, 3, 4, 1)


def orthographic_proj(X, cam):
    """
    X: B x N x 3
    cam: B x 7: [sc, tx, ty, quaternions]
    """
    quat = cam[:, -4:]
    X_rot = quat_rotate(X, quat)

    scale = cam[:, 0].contiguous().view(-1, 1, 1)
    trans = cam[:, 1:3].contiguous().view(cam.size(0), 1, -1)

    return scale * X_rot[:, :, :2] + trans


def orthographic_proj_withz(X, cam, offset_z=0.):
    """
    X: B x N x 3
    cam: B x 7: [sc, tx, ty, quaternions]
    Orth preserving the z.
    """
    quat = cam[:, -4:]
    X_rot = quat_rotate(X, quat)
    scale = cam[:, 0].contiguous().view(-1, 1, 1)
    trans = cam[:, 1:3].contiguous().view(cam.size(0), 1, -1)

    proj = scale * X_rot

    proj_xy = proj[:, :, :2] + trans
    proj_z = proj[:, :, 2, None] + offset_z
    
    return torch.cat((proj_xy, proj_z), 2)


def cross_product(qa, qb):
    """Cross product of va by vb.

    Args:
        qa: B X N X 3 vectors
        qb: B X N X 3 vectors
    Returns:
        q_mult: B X N X 3 vectors
    """
    qa_0 = qa[:, :, 0]
    qa_1 = qa[:, :, 1]
    qa_2 = qa[:, :, 2]

    qb_0 = qb[:, :, 0]
    qb_1 = qb[:, :, 1]
    qb_2 = qb[:, :, 2]

    # See https://en.wikipedia.org/wiki/Cross_product
    q_mult_0 = qa_1 * qb_2 - qa_2 * qb_1
    q_mult_1 = qa_2 * qb_0 - qa_0 * qb_2
    q_mult_2 = qa_0 * qb_1 - qa_1 * qb_0

    return torch.stack([q_mult_0, q_mult_1, q_mult_2], dim=-1)


def hamilton_product(qa, qb):
    """Multiply qa by qb.

    Args:
        qa: B X N X 4 quaternions
        qb: B X N X 4 quaternions
    Returns:
        q_mult: B X N X 4
    """
    qa_0 = qa[:, :, 0]
    qa_1 = qa[:, :, 1]
    qa_2 = qa[:, :, 2]
    qa_3 = qa[:, :, 3]

    qb_0 = qb[:, :, 0]
    qb_1 = qb[:, :, 1]
    qb_2 = qb[:, :, 2]
    qb_3 = qb[:, :, 3]

    # See https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
    q_mult_0 = qa_0 * qb_0 - qa_1 * qb_1 - qa_2 * qb_2 - qa_3 * qb_3
    q_mult_1 = qa_0 * qb_1 + qa_1 * qb_0 + qa_2 * qb_3 - qa_3 * qb_2
    q_mult_2 = qa_0 * qb_2 - qa_1 * qb_3 + qa_2 * qb_0 + qa_3 * qb_1
    q_mult_3 = qa_0 * qb_3 + qa_1 * qb_2 - qa_2 * qb_1 + qa_3 * qb_0

    return torch.stack([q_mult_0, q_mult_1, q_mult_2, q_mult_3], dim=-1)

def quat_conj(q):
    return torch.cat([q[:, :, [0]], -1 * q[:, :, 1:4]], dim=-1)

def quat2ang(q):
    ang  = 2*torch.acos(torch.clamp(q[:,:,0], min=-1 + 1E-6, max=1-1E-6))
    ang = ang.unsqueeze(-1)
    return ang


def quat_rotate(X, q):
    """Rotate points by quaternions.

    Args:
        X: B X N X 3 points
        q: B X 4 quaternions

    Returns:
        X_rot: B X N X 3 (rotated points)
    """
    # repeat q along 2nd dim
    ones_x = X[[0], :, :][:, :, [0]] * 0 + 1
    q = torch.unsqueeze(q, 1) * ones_x

    q_conj = torch.cat([q[:, :, [0]], -1 * q[:, :, 1:4]], dim=-1)
    X = torch.cat([X[:, :, [0]] * 0, X], dim=-1)

    X_rot = hamilton_product(q, hamilton_product(X, q_conj))
    return X_rot[:, :, 1:4]



'''
    point3d : B x N x 3
    camera = [phi_11, phi_12, phi_13, phi_21, phi_23, phi_33, t_x, t_y]
             B x 8
'''

def orthographic_proj_usingmatrix(points3d, scale, R, T, batch=True):
    if batch:
        projection = scale[:,None, None]*torch.bmm(points3d, R.permute(0, 2,1))[:, :, 0:2] +  T[:,None,:]
    else:
        projection = scale*torch.mm(points3d, R.permute(1,0))[:,0:2] + T[None,:]
    return projection


'''
    point3d : B x N x 3
    camera = [phi_11, phi_12, phi_13, t_x, phi_21, phi_23, phi_33, t_y]
             B x 8
'''


def affine_projection_withoutz(points3d, camera):
    camera = camera.view(camera.size(0), 2, 4)
    points3d = torch.cat([points3d, points3d[:, :, None, 1] * 0 + 1], dim=2)
    projection = torch.bmm(points3d, camera.permute(0, 2, 1))
    return projection
''' 
This should hopefully do all the operations on the GPU.
'''

'''
rotation -> B x 3 x 3
'''
def convert_rotation_to_quat(rotation):
    quats = []
    for rot in rotation:
        try:
            quat = torch.FloatTensor(transformations.quaternion_from_matrix(rot.data.cpu().numpy())).type(rot.type())
        except np.linalg.linalg.LinAlgError as e:
            pdb.set_trace()
            quat = np.zeros(4)
            quat[0] = 1

        quats.append(quat)
    return torch.stack(quats)


# class PnPSolver():

#     def __init__(self, tensor_type, device='cuda:0'):
#         self.Tensor = tensor_type
#         self.device = device
#         return

#     def solve_pnp(self, points2d, points3d):

#         return

class CameraSolver(nn.Module):

    def __init__(self, tensor_type, device='cuda:0', offset_z=5.0):
        super(CameraSolver, self).__init__()
        self.Tensor = tensor_type
        self.device = device
        self.offset_z = offset_z
        return

    '''
        x = S*R*X + T
        x = P*X + T
        There are 8 parameters and we solve using least squares optimization.
        Then we normalize to find the scale. And solve for the third row of R using cross product.
        points2d are normalized by the height of the image and the centered around the center pixel. 
        Use 10 points per ransac iteration.
        ## Create the A matrix, and b matrix and solve for the camera as a weak perspective projection
        ## You will have to use svd here.
        Can only run per example. Does not support batching
        
    # '''
    # @staticmethod
    # def compute_inverse_and_errors(points3d, points2d,ransac_samples=4):
    #     sample_indices = torch.clamp(torch.zeros(count).uniform_() * N, 0, N - 1).long()
    #     sample_points3d = points3d[sample_indices, :]
    #     sample_points2d = points2d[sample_indices, :]


    #     return
    def solve_camera(self, points3d, points2d, ransac_iteration=10, ransac_samples=100):

        assert len(points3d.size()) == 2, 'does not support batching'
        errors = []
        cams = []
        points3d = points3d.detach()
        points2d = points2d.detach()
        N = points3d.size(0)
        for i in range(ransac_iteration):
            try:
                sample_indices = self.sample_random_indices(ransac_samples, N)
                if len(torch.unique(sample_indices)) < 4:
                    continue
                sample_points3d = points3d[sample_indices, :]
                sample_points2d = points2d[sample_indices, :]
                A, b = self.computeAB(sample_points2d, sample_points3d)
                U, S, V = torch.svd(A, some=True)
                S = S*(S > 1E-3).float() + 1E-4*(S < 1E-3).float()
                psuedoSinv = S.pow(-1) * (S > 1E-3).float()
                Sinv = torch.diag(psuedoSinv)
                # Sinv = torch.diag((S + 1E-4).pow(-1))
                Ainv = torch.mm(V, torch.mm(Sinv, U.permute(1, 0)))
                camera_params = torch.mm(Ainv, b)
                scale, R, T = self.get_camera_matrix(camera_params)
                if torch.isnan(R).sum().item() > 0:
                    continue;

                error = self.compute_err(scale, R, T, points3d, points2d)
                cams.append((scale, R, T))
                errors.append(error)
                
        
            except RuntimeError as e: ## Sometimes the matrices are ill-conditioned. We ignore such examples
                continue

        if len(errors) < ransac_iteration/2:
            pdb.set_trace()
            return None, None, None ## The inputs are ill-conditioned

        assert len(errors) > 0,' You should have atleast one entry per image. Not all the samples should be ill-conditioned'
        errors = torch.stack(errors)  # ransac_iteration
        # cams = torch.stack(cams)  # ransac_iterations x 8

        _, ind = torch.min(errors, dim=0)
        cam_best = cams[ind]  # N x 8
        # Find the cam with the least error, and return it.
        return cam_best[0], cam_best[1], cam_best[2]

    '''
    Contains 11 parameters. Convert to rotation matrix.
    '''

    def get_camera_matrix(self, camera):
        # phimatrix = camera[0:9].view(3,3)
        # U, S,V = torch.svd(phimatrix)
        # R = torch.mm(U, V.permute(1,0))
        # T = camera[9:,0]
        # scale = (phimatrix/R).mean()
        # T = torch.cat([T, self.offset_z + T[None,0]*0])
        # return scale, R, T

        phimatrix = camera[0:6].view(2, 3)
        U, S, V = torch.svd(phimatrix)
        newS = torch.eye(U.size(1), V.size(1)).type(U.type())
        R = torch.mm(torch.mm(U,newS), V.permute(1,0))
        scale = (phimatrix / R).mean()
        R = torch.cat([R, torch.cross(R[0,:], R[1,:]).view(1,-1)], dim=0)
        T = camera[6:].view(-1)
        return scale, R, T

    '''
        points3d --> N x 3
    '''

    def computeAB(self, points2d, points3d):
        zeros = torch.zeros_like(points3d)
        zeros_one = zeros[:, None, 0]
        row = [points3d, zeros,  zeros_one + 1, zeros_one, zeros, points3d, zeros_one, zeros_one + 1]
        A = torch.cat(row, dim=1)
        A = A.view(-1, 8)
        B = points2d
        B = B.view(-1, 1)
        return A, B

    def sample_random_indices(self, count, N):
        sample_indices = torch.clamp(torch.zeros(count).uniform_() * N, 0, N - 1).long()
        return sample_indices.to(self.device)

    '''
    camera  4 x 2
    points3d N x 3
    points2d N x 2
    '''

    def compute_err(self, scale, R, T, points3d, points2d):
        projection = orthographic_proj_usingmatrix(points3d, scale, R, T, batch=False)
        error = (points2d - projection).pow(2).sum(1).mean()
        return error

'''
rotation is cuda.FloatTensor
quat cuda.FloatTensr
'''
from  ..utils import transformations

