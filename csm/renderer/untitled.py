import os
import bpy
import sys
import math
import random
import numpy as np
import os.path as osp

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
    print("%f %f %f" % (yaw, pitch, roll))
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


def quat_conjugate(q):
    q_conj = np.array(q, copy=True)
    q_conj[1:] = -1*q_conj[1:]
    return q_conj

g_syn_light_num_lowbound = 0
g_syn_light_num_highbound = 6
g_syn_light_dist_lowbound = 8
g_syn_light_dist_highbound = 20
g_syn_light_azimuth_degree_lowbound = 0
g_syn_light_azimuth_degree_highbound = 360
g_syn_light_elevation_degree_lowbound = -90
g_syn_light_elevation_degree_highbound = 90
g_syn_light_energy_mean = 2
g_syn_light_energy_std = 2
g_syn_light_environment_energy_lowbound = 0
g_syn_light_environment_energy_highbound = 10

light_num_lowbound = g_syn_light_num_lowbound
light_num_highbound = g_syn_light_num_highbound
light_dist_lowbound = g_syn_light_dist_lowbound
light_dist_highbound = g_syn_light_dist_highbound

shape_file = osp.join('/nfs.yoda/nileshk/CorrespNet/datasets/globe_v3/model', 'model.obj')
7

bpy.context.scene.render.alpha_mode = 'TRANSPARENT'
#bpy.context.scene.render.use_shadows = False
#bpy.context.scene.render.use_raytrace = False
syn_images_folder = '/nfs.yoda/nileshk/CorrespNet/cachedir/rendering'
bpy.data.objects['Lamp'].data.energy = 0

camObj = bpy.data.objects['Camera']

bpy.ops.object.select_all(action='TOGGLE')
if 'Lamp' in list(bpy.data.objects.keys()):
    bpy.data.objects['Lamp'].select = True # remove default light

bpy.ops.object.delete()

bpy.context.scene.world.light_settings.use_environment_light = True
bpy.context.scene.world.light_settings.environment_energy = np.random.uniform(g_syn_light_environment_energy_lowbound, g_syn_light_environment_energy_highbound)
bpy.context.scene.world.light_settings.environment_color = 'PLAIN'
bpy.ops.object.select_by_type(type='LAMP')
bpy.ops.object.delete(use_global=False)
for i in range(random.randint(light_num_lowbound,light_num_highbound)):
    light_azimuth_deg = np.random.uniform(g_syn_light_azimuth_degree_lowbound, g_syn_light_azimuth_degree_highbound)
    light_elevation_deg  = np.random.uniform(g_syn_light_elevation_degree_lowbound, g_syn_light_elevation_degree_highbound)
    light_dist = np.random.uniform(light_dist_lowbound, light_dist_highbound)
    lx, ly, lz = obj_centened_camera_pos(light_dist, light_azimuth_deg, light_elevation_deg)
    bpy.ops.object.lamp_add(type='POINT', view_align = False, location=(lx, ly, lz))
    bpy.data.objects['Point'].data.energy = np.random.normal(g_syn_light_energy_mean, g_syn_light_energy_std)

light_azimuth_deg = np.random.uniform(g_syn_light_azimuth_degree_lowbound, g_syn_light_azimuth_degree_highbound)
light_elevation_deg  = np.random.uniform(g_syn_light_elevation_degree_lowbound, g_syn_light_elevation_degree_highbound)
light_dist = np.random.uniform(light_dist_lowbound, light_dist_highbound)
lx, ly, lz = obj_centened_camera_pos(light_dist, light_azimuth_deg, light_elevation_deg)
bpy.ops.object.lamp_add(type='POINT', view_align = False, location=(lx, ly, lz))
bpy.data.objects['Point'].data.energy = np.random.normal(g_syn_light_energy_mean, g_syn_light_energy_std)
camObj.location[0] = 0
camObj.location[1] = 0 
camObj.location[2] = 5
camObj.rotation_mode = 'QUATERNION'
camObj.rotation_quaternion[0] = 1
camObj.rotation_quaternion[1] = 0
camObj.rotation_quaternion[2] = 0
camObj.rotation_quaternion[3] = 0
syn_image_file = 'rendered.png'


rho=5
azimuth_deg =0
elevation_deg = 0
theta_deg = 0

cx, cy, cz = obj_centened_camera_pos(rho, azimuth_deg, elevation_deg)
q1 = camPosToQuaternion(cx, cy, cz)
q2 = camRotQuaternion(cx, cy, cz, theta_deg)
q = quaternionProduct(q2, q1)

camObj.location[0] = cx
camObj.location[1] = cy
camObj.location[2] = cz
camObj.rotation_mode = 'QUATERNION'
camObj.rotation_quaternion[0] = q[0]
camObj.rotation_quaternion[1] = q[1]
camObj.rotation_quaternion[2] = q[2]
camObj.rotation_quaternion[3] = q[3]


bpy.ops.render.render( write_still=True )
sys.path.append('/home/nileshk/CorrespNet/blender_utils/')
from . import transformations


def campose_to_extrinsic(quat, location):
    # bcam stands for blender camera
    bcam2cv = np.array( [[1, 0,  0, 0],         [0, -1, 0, 0],        [0, 0, -1, 0],        [0, 0, 0, 1]])
    rotation = transformations.quaternion_matrix(quat)
    location = transformations.translation_matrix(location)
    world2bcam = rotation.transpose()
    location_inv = location.copy()
    location_inv[:,0:3] = -1*location_inv[:,0:3]
    world2bcam = np.matmul(world2bcam, location)
    world2cv = np.matmul(bcam2cv,world2bcam)
    RT = world2cv
    quat = transformations.quaternion_from_matrix(RT, isprecise=True)
    trans = transformations.translation_from_matrix(RT)
    return RT, quat, trans


quat_conj = quat_conjugate(q)
neg_trans = -1*camObj.location


## apply this to the camera 

cam_rotation = quaternionProduct(quat_conj,camObj.rotation_quaternion)
camObj.rotation_quaternion[0] = cam_rotation[0]
camObj.rotation_quaternion[1] = cam_rotation[1]
camObj.rotation_quaternion[2] = cam_rotation[2]
camObj.rotation_quaternion[3] = cam_rotation[3]

camObj.location[0] = camObj.location[0] + neg_trans[0]
camObj.location[1] = camObj.location[1] + neg_trans[1]
camObj.location[2] = camObj.location[2] + neg_trans[2]



## apply this to the camera 

