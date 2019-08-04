import pymesh
import scipy.misc
import math
from . import transformations
import numpy as np
import pdb
from ..nnutils import geom_utils
import torch
from ..renderer import renderer_utils
import tempfile
import shutil
import os
import os.path as osp

'''
UV map is H x W map of the image size.

'''
def project_uv_to_image(self, uv_map, texture_map):
    H = uv_map.size(0)
    W = uv_map.size(1)


'''
https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
'''
# Returns camera rotation and translation matrices from Blender.
# 
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates 
#         used in digital images)
#       - right-handed: positive z look-at direction

def campose_to_extrinsic(quat, location):
    # bcam stands for blender camera
    bcam2cv = np.array(
        [[1, 0,  0, 0],
         [0, -1, 0, 0],
         [0, 0, -1, 0],
         [0, 0, 0, 1]])

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    rotation = transformations.quaternion_matrix(quat)
    location = transformations.translation_matrix(location)
    # location, rotation = cam.matrix_world.decompose()[0:2]
    # R_world2bcam = rotation.to_matrix().transposed()
    # bcam2world = np.matmul(rotation, location)
    world2bcam = rotation.transpose()
    location_inv = location.copy()
    location_inv[:,0:3] = -1*location_inv[:,0:3]
    world2bcam = np.matmul(world2bcam, location)

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:     
    # T_world2bcam = -1 * np.matmul(R_world2bcam , location)

    # Build the coordinate transform matrix from world to computer vision camera
    world2cv = np.matmul(bcam2cv,world2bcam)
    # T_world2cv = np.matmul(R_bcam2cv,T_world2bcam)

    # # put into 3x4 matrix
    # RT = Matrix((
    #     R_world2cv[0][:] + (T_world2cv[0],),
    #     R_world2cv[1][:] + (T_world2cv[1],),
    #     R_world2cv[2][:] + (T_world2cv[2],)
    #      ))
    # RT = np.matmul(R_world2cv, location)
    RT = world2cv
    quat = transformations.quaternion_from_matrix(RT, isprecise=True)
    trans = transformations.translation_from_matrix(RT)
    return RT, quat, trans



'''
https://github.com/ShapeNet/RenderForCNN/blob/master/render_pipeline/render_model_views.py
'''

def camPosToQuaternion(cx, cy, cz):
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = cx / camDist
    cy = cy / camDist
    cz = cz / camDist
    axis = (-cz, 0, cx)
    angle = math.acos(cy)
    a = math.sqrt(2) / 2
    b = math.sqrt(2) / 2
    w1 = axis[0]
    w2 = axis[1]
    w3 = axis[2]
    c = math.cos(angle / 2)
    d = math.sin(angle / 2)
    q1 = a * c - b * d * w1
    q2 = b * c + a * d * w1
    q3 = a * d * w2 + b * d * w3
    q4 = -b * d * w2 + a * d * w3
    return (q1, q2, q3, q4)

def quaternionFromYawPitchRoll(yaw, pitch, roll):
    c1 = math.cos(yaw / 2.0)
    c2 = math.cos(pitch / 2.0)
    c3 = math.cos(roll / 2.0)    
    s1 = math.sin(yaw / 2.0)
    s2 = math.sin(pitch / 2.0)
    s3 = math.sin(roll / 2.0)    
    q1 = c1 * c2 * c3 + s1 * s2 * s3
    q2 = c1 * c2 * s3 - s1 * s2 * c3
    q3 = c1 * s2 * c3 + s1 * c2 * s3
    q4 = s1 * c2 * c3 - c1 * s2 * s3
    return (q1, q2, q3, q4)


def camPosToQuaternion(cx, cy, cz):
    q1a = 0
    q1b = 0
    q1c = math.sqrt(2) / 2
    q1d = math.sqrt(2) / 2
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = cx / camDist
    cy = cy / camDist
    cz = cz / camDist    
    t = math.sqrt(cx * cx + cy * cy) 
    tx = cx / t
    ty = cy / t
    yaw = math.acos(ty)
    if tx > 0:
        yaw = 2 * math.pi - yaw
    pitch = 0
    tmp = min(max(tx*cx + ty*cy, -1),1)
    #roll = math.acos(tx * cx + ty * cy)
    roll = math.acos(tmp)
    if cz < 0:
        roll = -roll    
    # print("%f %f %f" % (yaw, pitch, roll))
    q2a, q2b, q2c, q2d = quaternionFromYawPitchRoll(yaw, pitch, roll)    
    q1 = q1a * q2a - q1b * q2b - q1c * q2c - q1d * q2d
    q2 = q1b * q2a + q1a * q2b + q1d * q2c - q1c * q2d
    q3 = q1c * q2a - q1d * q2b + q1a * q2c + q1b * q2d
    q4 = q1d * q2a + q1c * q2b - q1b * q2c + q1a * q2d
    return (q1, q2, q3, q4)

def camRotQuaternion(cx, cy, cz, theta): 
    theta = theta / 180.0 * math.pi
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = -cx / camDist
    cy = -cy / camDist
    cz = -cz / camDist
    q1 = math.cos(theta * 0.5)
    q2 = -cx * math.sin(theta * 0.5)
    q3 = -cy * math.sin(theta * 0.5)
    q4 = -cz * math.sin(theta * 0.5)
    return (q1, q2, q3, q4)

def quaternionProduct(qx, qy): 
    a = qx[0]
    b = qx[1]
    c = qx[2]
    d = qx[3]
    e = qy[0]
    f = qy[1]
    g = qy[2]
    h = qy[3]
    q1 = a * e - b * f - c * g - d * h
    q2 = a * f + b * e + c * h - d * g
    q3 = a * g - b * h + c * e + d * f
    q4 = a * h + b * g - c * f + d * e    
    return (q1, q2, q3, q4)

def obj_centened_camera_pos(dist, azimuth_deg, elevation_deg):
    phi = float(elevation_deg) / 180 * math.pi
    theta = float(azimuth_deg) / 180 * math.pi
    x = (dist * math.cos(theta) * math.cos(phi))
    y = (dist * math.sin(theta) * math.cos(phi))
    z = (dist * math.sin(phi))
    return (x, y, z)

def campose_from_azi_el(azimuth_deg, elevation_deg, theta_deg, rho):
    cx, cy, cz = obj_centened_camera_pos(rho, azimuth_deg, elevation_deg)
    q1 = camPosToQuaternion(cx, cy, cz)
    q2 = camRotQuaternion(cx, cy, cz, theta_deg)
    q = quaternionProduct(q2, q1)
    return q, np.array([cx, cy, cz])

'''
uv_map : np.array H x W x 2
texture_map : np.array 3 x H x W

'''
def render_image_from_uv(mask, uv_map, texture_map):
    mask = mask.data.cpu().numpy()
    mask = mask.transpose(1,2,0)
    uv_map = uv_map.data.cpu().numpy()
    image = np.zeros(uv_map.shape)
    # pdb.set_trace()
    texture_img_size = texture_map.shape
    uv_map_ints = uv_map*0
    uv_map_ints[:,:,0] = uv_map[:,:,0] * texture_img_size[1]
    uv_map_ints[:,:,1] = uv_map[:,:,1] * texture_img_size[0]
    uv_map_ints = np.round(uv_map_ints).astype(np.int32)
    uv_map_ints = uv_map_ints.reshape(-1, 2)
    uv_map_ints[:,0] = np.clip(uv_map_ints[:,0], a_min=0, a_max = texture_img_size[1]-1)
    uv_map_ints[:,1] = np.clip(uv_map_ints[:,1], a_min=0, a_max = texture_img_size[0]-1)
    # pdb.set_trace()
    # uv_map_ints = convert_uv_map_to_texture_coordinates(uv_map, texture_map.size())
    # texture_map_flatten = texture_map.reshape(-1,)
    image = texture_map[uv_map_ints[:,1], uv_map_ints[:,0]]
    # pdb.set_trace() 
    H = uv_map.shape[0]
    W = uv_map.shape[1]
    image = image.reshape(H, W, 3)
    image = image * mask ## Texture map is uint8, with range 0, 255
    image = image.astype(np.uint8)
    scipy.misc.imsave('rendered.png', image)
    return image


def convert_uv_map_to_texture_coordinates(uv_map, texture_img_size):
    pdb.set_trace()
    uv_map_ints = uv_map*0
    uv_map_ints[:,:,0] = uv_map[:,:,0] * texture_img_size[1]
    uv_map_ints[:,:,1] = uv_map[:,:,1] * texture_img_size[0]
    uv_map_ints = np.round(uv_map_ints).astype(np.int32)
    uv_map_ints = uv_map_ints.reshape(-1, 2)
    uv_map_ints[:,0] = np.clip(uv_map_ints[:,0], a_min=0, a_max = texture_img_size[1]-1)
    uv_map_ints[:,1] = np.clip(uv_map_ints[:,1], a_min=0, a_max = texture_img_size[0]-1)
    return uv_map_ints
'''
points3d : np.array H x W x 3
'''
def render_image_using_uv_and_cam(mask, project_points, uv_map, texture_map,img_size):
    
    proj_xy = project_points
    mask = mask.data.cpu().numpy()
    mask = mask.transpose(1,2,0)
    proj_xy[:,:,0] = proj_xy[:,:,0]*img_size[1]/2 + img_size[1]/2
    proj_xy[:,:,1] = proj_xy[:,:,1]*img_size[0]/2 + img_size[0]/2

    proj_xy = proj_xy.long()
    ## Are proj_xy really unique? Becaue uv_map need not be unique.

    image = torch.zeros(mask.size(0), mask.size(1), 3).view(-1, 3)
    # uv_map_ints 
    uv_map_ints = convert_uv_map_to_texture_coordinates(uv_map, texture_map.size())
    new_uv_map = uv_map_ints.view(-1,2)
    image_indices = proj_xy[0] + img_size[0]*proj_xy[1]
    # image[image_indices, :] = texture_map.view(-1, 3)[uv_map_ints[1]*)]
    # scipy.misc.imsave('rendered.png', image)
    return image

'''
points3d : np.array N x 3
'''
def render_uvmap(mask, uv_map):
    mask = mask.data.cpu().numpy()
    mask = mask.transpose(1,2,0)
    uv_map = uv_map.data.cpu().numpy()
    uv_map = uv_map * mask
    uv_map = (uv_map * 255).astype(np.uint8)
    image_u = uv_map[:,:,0,None].repeat(1, axis=2)
    image_v = uv_map[:,:,1,None].repeat(1, axis=2)
    # scipy.misc.imsave('rendered.png', image)
    return image_u.repeat(3, axis=2), image_v.repeat(3, axis=2)


def render_model(render_dir, obj_file, offset_z, camera):
    if not osp.exists(render_dir):
        os.makedirs(render_dir)

    scale = camera[0]
    trans = camera[1:3]
    quat = camera[3:]
    renderer_utils.render_model_orthographiz(render_dir, obj_file, scale, trans, quat, offset_z)
    img = scipy.misc.imread(osp.join(render_dir, 'rendered.png'))
    img = img[:,:,0:3]
    img = img.astype(np.float32)
    return img/255.0