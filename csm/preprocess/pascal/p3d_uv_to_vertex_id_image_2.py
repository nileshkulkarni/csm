
import scipy.io as sio
import numpy as np
import os.path as osp
import meshzoo
import pdb
import pymesh
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from ...nnutils import geom_utils


def create_sphere(n_subdivide=3):
    # 3 makes 642 verts, 1280 faces,
    # 4 makes 2562 verts, 5120 faces
    verts, faces = meshzoo.iso_sphere(n_subdivide)
    return verts, faces

def triangle_direction_intersection(tri, trg):
    '''
    Finds where an origin-centered ray going in direction trg intersects a triangle.
    Args:
        tri: 3 X 3 vertex locations. tri[0, :] is 0th vertex.
    Returns:
        alpha, beta, gamma
    '''
    p0 = np.copy(tri[0, :])
    # Don't normalize
    d1 = np.copy(tri[1, :]) - p0;
    d2 = np.copy(tri[2, :]) - p0;
    d = trg / np.linalg.norm(trg)

    mat = np.stack([d1, d2, d], axis=1)

    try:
      inv_mat = np.linalg.inv(mat)
    except np.linalg.LinAlgError:
      return False, 0

    # inv_mat = np.linalg.inv(mat)
    
    a_b_mg = -1*np.matmul(inv_mat, p0)
    is_valid = (a_b_mg[0] >= 0) and (a_b_mg[1] >= 0) and ((a_b_mg[0] + a_b_mg[1]) <= 1) and (a_b_mg[2] < 0)
    if is_valid:
        return True, -a_b_mg[2]*d
    else:
        return False, 0


def project_verts_on_mesh(verts, mesh_verts, mesh_faces):
    verts_out = np.copy(verts)
    for nv in range(verts.shape[0]):
        max_norm = 0
        vert = np.copy(verts_out[nv, :])
        for f in range(mesh_faces.shape[0]):
            face = mesh_faces[f]
            tri = mesh_verts[face, :]
            # is_v=True if it does intersect and returns the point
            is_v, pt = triangle_direction_intersection(tri, vert)
            # Take the furthest away intersection point
            if is_v and np.linalg.norm(pt) > max_norm:
                max_norm = np.linalg.norm(pt)
                verts_out[nv, :] = pt

    return verts_out


# '''
# Maps UV coordinates to a sphere of radius = 0.5
# '''
# def convert_uv_to_3d_coordinates(uv):
#     theta = np.pi*(2*uv[:, 0] - 1)
#     phi = np.pi*(0.5 - uv[:,1])
#     x = np.cos(phi) * np.cos(theta)
#     z = np.cos(phi) * np.sin(theta)
#     y = np.sin(phi)
#     point_3d = 1.0*np.stack([x, y, z], axis=-1)
#     return point_3d;

# def convert_3d_to_uv_coordinates(points):
#     u = 0.5 + np.arctan2(points[:,2], points[:,0])/(2*np.pi)
#     v = 0.5 - np.arcsin(points[:,1])/(np.pi)
#     return np.stack([u,v],axis=1)

def convert_to_barycentric_coordinates(faces, verts, face_inds, points):

    face_vertices = faces[face_inds]
    vertA = verts[face_vertices[:, 0]]
    vertB = verts[face_vertices[:, 1]]
    vertC = verts[face_vertices[:, 2]]

    AB = vertB - vertA
    AC  = vertC - vertA
    BC  = vertC - vertB

    AP = points - vertA
    BP = points - vertB
    CP = points - vertC

    areaBAC = np.linalg.norm(np.cross(AB, AC, axis=1), axis=1)
    areaBAP = np.linalg.norm(np.cross(AB, AP, axis=1), axis=1)
    areaCAP = np.linalg.norm(np.cross(AC, AP, axis=1), axis=1)
    areaCBP = np.linalg.norm(np.cross(BC, BP, axis=1), axis=1)  
    

    w = areaBAP/areaBAC
    v = areaCAP/areaBAC
    u = areaCBP/areaBAC

    barycentric_coordinates = np.stack([u, v, w], axis=1)
    
    # rePoints = vertA*barycentric_coordinates[:,None,0] + vertB*barycentric_coordinates[:,None,1] + vertC*barycentric_coordinates[:,None,2]
    return barycentric_coordinates #N x 3

'''
UV map size  (W, H)
'''
def map_shape_to_ico_sphere(uv_map_size=(1001, 1001)):
    p3d_cache_dir = '/home/nileshk/CorrespNet/icn/cachedir/shapenet/aeroplane/'
    # anno_sfm_path = osp.join(p3d_cache_dir, 'sfm', '{}_train.mat'.format(p3d_class))
    # anno_sfm = sio.loadmat(anno_sfm_path, struct_as_record=False, squeeze_me=True)
    # sfm_mean_shape = (np.transpose(anno_sfm['S']), anno_sfm['conv_tri']-1)
    #shapenet_mean_shape = pymesh.load_mesh(osp.join(p3d_cache_dir, 'marching_cubes_mean.obj'))
    shapenet_mean_shape = pymesh.load_mesh(osp.join(p3d_cache_dir, 'marching_cubes_mean.obj'))
    verts, faces = create_sphere(3)
    pdb.set_trace()
    verts_proj = project_verts_on_mesh(verts, shapenet_mean_shape.vertices, shapenet_mean_shape.faces)
    mesh_sphere = pymesh.form_mesh(verts, faces)
    mesh_shape = pymesh.form_mesh(verts_proj, faces)

    # pymesh.meshio.save_mesh('test_sphere.obj', mesh_sphere)
    pymesh.meshio.save_mesh('{}.obj'.format(p3d_class), mesh_shape)
    # uv = np.zeros((uv_map_size[1], uv_map_size[0], 2))
    # u_step = 1.0/1920
    # v_step = 1.0/960
    map_H = uv_map_size[1]
    map_W = uv_map_size[0]
    x = np.arange(0, 1 + 1.0/(map_W-1), 1.0/(map_W-1))
    y = np.arange(0, 1 + 1.0/(map_H-1), 1.0/(map_H-1))

    xx, yy = np.meshgrid(x, y, indexing='xy')

    map_uv = np.stack([xx, yy], axis=2)
    map_uv  = map_uv.reshape(-1, 2)
    map_3d = geom_utils.convert_uv_to_3d_coordinates(map_uv)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    dist , face_inds, closest_points = pymesh.distance_to_mesh(mesh_sphere, map_3d)
    # face_ind = face_inds.reshape(map_H, map_W,)
    
    barycentric_coordinates = convert_to_barycentric_coordinates(faces, verts, face_inds, map_3d)
    
    # ax.scatter(map_3d[:,0],map_3d[:,1], map_3d[:,2] ,'r')
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    uv_cords = geom_utils.convert_3d_to_uv_coordinates(verts)
    # plt.savefig('3d_uv_points.png')
    face_verts_ind = faces[face_inds]
    uv_vertsA =  uv_cords[face_verts_ind[:, 0]]
    uv_vertsB =  uv_cords[face_verts_ind[:, 1]]
    uv_vertsC =  uv_cords[face_verts_ind[:, 2]]
    
    new_uv = uv_vertsA*barycentric_coordinates[:,None,0] + uv_vertsB*barycentric_coordinates[:,None, 0] + uv_vertsC*barycentric_coordinates[:,None, 2]
    dist = dist.reshape(map_H, map_W)
    barycentric_coordinates = barycentric_coordinates.reshape(map_H, map_W, 3)
    face_inds = face_inds.reshape(map_H, map_W)
    map_3d = map_3d.reshape(map_H, map_W, -1)
    closest_points = closest_points.reshape(map_H, map_W, -1)
    map_uv = map_uv.reshape(map_H, map_W,2)
    stuff = {}
    stuff['sphere_verts'] = verts 
    stuff['verts'] = verts_proj 
    # stuff['verts'] = verts ## Deliberate change, restore later
    stuff['faces'] = faces
    stuff['uv_verts'] = uv_cords
    stuff['uv_map'] = map_uv
    stuff['bary_cord'] = barycentric_coordinates
    stuff['face_inds'] = face_inds

    return stuff



def checkpoint_inside_triangle(A, B, C, P):
    def sign(p1, p2, p3):
        return (p1[0] - p2[0])*(p2[1] - p3[1]) - (p2[0] - p3[0])*(p1[1] - p2[1])

    d1 = sign(P, A, B);
    d2 = sign(P, B, C);
    d3 = sign(P, C, A);

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0);
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0);

    return not (has_neg and has_pos)



def save_map_and_barycentric_to_mat(filename, stuff):
    sio.savemat(filename, stuff)
    return 

if __name__ == "__main__":
    p3d_class = 'aeroplane'
    out_file = osp.join('/nfs.yoda/nileshk/CorrespNet/cachedir/shapenet/{}/'.format(p3d_class), 'mean_shape.mat')
    stuff = map_shape_to_ico_sphere()
    save_map_and_barycentric_to_mat(out_file, stuff)

    # out_file = osp.join('/nfs.yoda/nileshk/CorrespNet/cachedir/cub/uv', 'mean_cmr_shape.mat')
    # stuff = map_cmr_mean_to_uv_image()
    # save_map_and_barycentric_to_mat(out_file, stuff)



