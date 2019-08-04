
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
'''
python -m icn.preprocess.parameterize.uv_to_vertex_map icn/cachedir/downloaded_models/aeroplane/aeroplane_mapping.mat icn/cachedir/downloaded_models/aeroplane/aeroplane_uv_map.mat
'''

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


def test_mapping(mapping, ):
    H, W = mapping['face_inds'].shape
    pdb.set_trace()
    for _ in range(100):
        h,w = np.random.choice(H), np.random.choice(W)
        point_3d = mapping['map_3d'][h,w]
        face_ind = mapping['face_inds'][h,w]
        vert_ind = mapping['faces'][face_ind]
        verts = mapping['sphere_verts'][vert_ind] # 3 x 3
        bc = mapping['bary_cord'][h,w] ## 
        recon_point = (bc.reshape(3,1)*verts).sum(0)
        error = np.linalg.norm(recon_point - point_3d)
        assert error < 0.01 , 'error in computing barycentric coordinates {}'.format(error)





'''
mapping is coming from the meshdeform function. UV map size  (W, H).
Mapping contains 
vshape, 
vsphere,
faces
'''
def map_shape_to_ico_sphere(mapping, uv_map_size=(1001, 1001), transform=None):
    vshape = mapping['vshape'].transpose(1,0).astype(np.float)
    vsphere = mapping['vsphere'].transpose(1,0).astype(np.float)
    faces = mapping['face'].transpose(1,0).astype(np.int)
    if transform is not None:
        vshape = np.dot(vshape, transform.transpose())

   
    mesh_sphere = pymesh.form_mesh(vsphere, faces)
    mesh_shape = pymesh.form_mesh(vshape, faces)

    pymesh.meshio.save_mesh('test_shape.obj', mesh_shape)
    pymesh.meshio.save_mesh('test_sphere.obj', mesh_sphere)
    # # pdb.set_trace()
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
    
    barycentric_coordinates = convert_to_barycentric_coordinates(faces, vsphere, face_inds, map_3d)
    
    # ax.scatter(map_3d[:,0],map_3d[:,1], map_3d[:,2] ,'r')
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    uv_cords = geom_utils.convert_3d_to_uv_coordinates(vsphere)

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
    stuff['map_3d'] = map_3d
    stuff['sphere_verts'] = vsphere 
    stuff['verts'] = vshape
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
    import sys
    if len(sys.argv) < 3:
        print('python uv_to_vertex_map.py <mapping.mat> <mean_shape_uv_map.mat>')
        sys.exit(0)
    mapping_file = sys.argv[1]
    out_file = sys.argv[2]
    transform = np.eye(3,3)
    if len(sys.argv) == 4:
        transform = sio.loadmat(sys.argv[3])['transform_shapenet2pascal']

    mapping = sio.loadmat(mapping_file)
    stuff = map_shape_to_ico_sphere(mapping, transform=transform)
    save_map_and_barycentric_to_mat(out_file, stuff)
    



