"""
Mesh stuff.
https://github.com/akanazawa/cmr/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import meshzoo
from ..nnutils import geom_utils
import pdb

def create_sphere(n_subdivide=3):
    # 3 makes 642 verts, 1280 faces,
    # 4 makes 2562 verts, 5120 faces
    verts, faces = meshzoo.iso_sphere(n_subdivide)
    return verts, faces


def compute_vert2kp(verts, mean_shape):
    # verts: N x 3
    # mean_shape: 3 x K (K=15)
    #
    # computes vert2kp: K x N matrix by picking NN to each point in mean_shape.

    if mean_shape.shape[0] == 3:
        # Make it K x 3
        mean_shape = mean_shape.T
    num_kp = mean_shape.shape[1]

    nn_inds = [np.argmin(np.linalg.norm(verts - pt, axis=1)) for pt in mean_shape]

    dists = np.stack([np.linalg.norm(verts - verts[nn_ind], axis=1) for nn_ind in nn_inds])
    vert2kp = -.5*(dists)/.01
    return vert2kp


def get_spherical_coords(X):
    uv = geom_utils.convert_3d_to_uv_coordinates(X)
    uv = 2*uv -1
    return uv

def compute_uvsampler(verts, faces, tex_size=2):
    """
    For this mesh, pre-computes the UV coordinates for
    F x T x T points.
    Returns F x T x T x 2
    """
    alpha = np.arange(tex_size, dtype=np.float) / (tex_size-1)
    beta = np.arange(tex_size, dtype=np.float) / (tex_size-1)
    import itertools
    # Barycentric coordinate values
    coords = np.stack([p for p in itertools.product(*[alpha, beta])])
    # coords_sum = np.clip(np.sum(coords, axis=1, keepdims=True),a_min=1, a_max=2)
    # coords = coords/coords_sum
    # coords = np.concatenate([coords, 1 - np.sum(coords, axis=1).reshape(-1,1)], axis=1)
    vs = verts[faces]
    # Compute alpha, beta (this is the same order as NMR)
    v2 = vs[:, 2]
    v0v2 = vs[:, 0] - vs[:, 2]
    v1v2 = vs[:, 1] - vs[:, 2]
    # F x 3 x T**2
    samples = np.dstack([v0v2, v1v2]).dot(coords.T) + v2.reshape(-1, 3, 1)
    # F x T*2 x 3 points on the sphere 
    samples = np.transpose(samples, (0, 2, 1))
    # import pdb; pdb.set_trace()
    # Now convert these to uv.
    uv = get_spherical_coords(samples.reshape(-1, 3))
    # uv = uv.reshape(-1, len(coords), 2)
    uv = uv.reshape(-1, tex_size, tex_size, 2)
    return uv


def append_obj(mf_handle, vertices, faces):
    for vx in range(vertices.shape[0]):
        mf_handle.write('v {:f} {:f} {:f}\n'.format(vertices[vx, 0], vertices[vx, 1], vertices[vx, 2]))
    for fx in range(faces.shape[0]):
        mf_handle.write('f {:d} {:d} {:d}\n'.format(faces[fx, 0], faces[fx, 1], faces[fx, 2]))
    return


'''
Modifies the mesh to reduce the effect of the discontunitity 
'''
def modify_mesh(verts, faces, uv_verts):
    verts_new = verts*1
    need_duplication = (uv_verts[:,0] == 1)
    duplicate_vert_ids = np.where(need_duplication)
    duplicate_index = (uv_verts[:,0] == 1)*(len(verts)-1 + np.cumsum(need_duplication))
    
    duplicates = (uv_verts[need_duplication])
    duplicate_verts = verts[need_duplication]*1
    duplicate_uv_verts = uv_verts[need_duplication]*1
    duplicate_uv_verts[:,0] = 0

    uv_verts = np.concatenate([uv_verts, duplicate_uv_verts], axis=0)
    verts = np.concatenate([verts, duplicate_verts], axis=0)
    new_faces = faces * 1
    for fx, fv in enumerate(faces):
        v1 = uv_verts[fv[0]]
        v2 = uv_verts[fv[1]]
        v3 = uv_verts[fv[2]]
        if np.sum(need_duplication[fv]) > 0:
            face_sum = np.sum(uv_verts[fv][:,0])
            if face_sum < 2.5:
                print("{} , {} ".format( fx, fv))
                for vx, v in enumerate(uv_verts[fv]):
                    if need_duplication[fv[vx]]:
                        new_faces[fx][vx] = duplicate_index[fv[vx]]
    faces  = new_faces
    return verts, faces, uv_verts