import numpy as np
import os.path as osp
import os
import tempfile
import shutil
import pdb
# tmp_dirname = tempfile.mkdtemp(dir='/nfs.yoda/nileshk/CorrespNet/cachedir/rendering/', prefix='tmp_view_')
g_blender_executable_path = '/home/nileshk/softwares/blender-2.71/blender'
g_blank_blend_file_path = osp.join('/home/nileshk/CorrespNet/icn/', 'renderer/blank.blend')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
'''

tmp_dir to store outputs
quat -- np.array : (4,)
trans -- np.array : (2,)
scale -- float
offset_x -- float
'''


def render_model_orthographiz(tmp_dirname, model_file, scale, trans, quat, offset_z=5):
    viewparam_file = open(osp.join(tmp_dirname,'viewpoints.txt'), 'w')
    tmpfile_name = viewparam_file.name
    viewparam_file.write('{:2.2f}'.format(scale))
    for t in trans:
        viewparam_file.write(' {:2.2f}'.format(t))
    for q in quat:
        viewparam_file.write(' {:2.2f}'.format(q))
    viewparam_file.write('\n')
    viewparam_file.close()
    command = '{} {} --background --python {} -- {} {} {} {} > /dev/null 2>&1'.format(g_blender_executable_path, g_blank_blend_file_path, os.path.join(
        BASE_DIR, 'render_model_ortho_views.py'), model_file, viewparam_file.name, tmp_dirname, offset_z)
    # command = '{} {} --background --python {} -- {} {} {} {}'.format(g_blender_executable_path, g_blank_blend_file_path, os.path.join(
    #     BASE_DIR, 'render_model_ortho_viewspy'), model_file, viewparam_file.name, tmp_dirname, offset_z)
    os.system(command)
    # /home/nileshk/softwares/blender-2.71/blender blank.blend  --background --python render_model_ortho_views.py  /nfs.yoda/nileshk/CorrespNet/datasets/globe_v3/model/model.obj viewparams.txt . 3

# if __name__ == "__main__":

#     render_model_orthographiz(5)
#     shutil.rmtree(tmp_dirname)
