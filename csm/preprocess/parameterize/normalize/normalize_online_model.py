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
'''
python  -m icn.preprocess.parameterize.normalize_online_model icn/cachedir/downloaded_models/aeroplane/aeroplane.obj .icn/cachedir/downloaded_models/aeroplane/aeroplane_norm.obj
'''

def writeObj(filename, model_obj):
    pymesh.meshio.save_mesh(filename, model_obj)
    return


def normalize_model(model_obj):

    verts = model_obj.vertices
    
    max_extent = np.max(verts.reshape(-1,3), axis=0)
    min_extent = np.min(verts.reshape(-1,3), axis=0)
    center = (max_extent - min_extent)/2
    scale = 0.90/np.max((max_extent - min_extent))
    center = center.reshape(1,3)
    verts = verts - center -min_extent.reshape(1,3)
    max_extent = np.max(verts.reshape(-1,3), axis=0)
    min_extent = np.min(verts.reshape(-1,3), axis=0)
    verts = verts * scale
    new_model = pymesh.form_mesh(verts, model_obj.faces)
    return new_model



if __name__ == "__main__":
    
    import sys
    inmodel_file = sys.argv[1]
    outmodel_file = sys.argv[2]
    if len(sys.argv) < 3    :
        print("python marching_cubes.py <in_file.obj> <out_file.obj>")

    model_obj = pymesh.load_mesh(inmodel_file)
    norm_model = normalize_model(model_obj)
    writeObj(outmodel_file, norm_model)
