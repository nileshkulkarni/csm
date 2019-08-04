import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import os.path as osp
import platform
import sys
import scipy.io as sio

eval_set = 'val'
#net_name = 'dwr_shape_ft'
# epoch_num = 200
# iter_number = 10000

# suffix = '_True'
curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', '..', 'cachedir')

eval_dir =  osp.join(cache_path, 'evaluation')


# plot_mat_files = ['mar12_gt_quat_cars/val/epoch_150/','car_net_march1/val/epoch_200/','learn_cycle_baseline/val/epoch_0', 'feb22_honest_car_no_vgg_10_models/val/epoch_200/','de_cars_baseline_2000tps_wmask_unet/val/epoch_200/', 'pretrain_vgg16_cars/val/epoch_0']
# plot_mat_files = ['mar12_gt_quat_cars/val/epoch_150/','car_net_march1/val/epoch_200/','learn_cycle_baseline/val/epoch_0', 'csm_cars_high_depth/val/epoch_200/','de_cars_baseline_2000tps_wmask_unet/val/epoch_200/', 'pretrain_vgg16_cars/val/epoch_0']
if False:
    #plot_mat_files = ['csm_car_run_gt_pose_scaleLr0pt05_scaleBias0pt5/val/epoch_200/','car_net_march1/val/epoch_200/',
    #                  'learn_cycle_baseline/val/epoch_0', 'csm_car_run_az_ele_2/val/epoch_200/',
    #                  'de_cars_baseline_2000tps_wmask_unet/val/epoch_200/',
    #                  'pretrain_vgg16_cars/val/epoch_0']
    plot_mat_files = ['csm_car_run_gt_pose_cr_vis_july/val/epoch_200/','car_net_march1/val/epoch_200/',
                      'learn_cycle_baseline/val/epoch_0', 'csm_car_run_az_ele_2/val/epoch_200/',
                      'de_cars_baseline_2000tps_wmask_unet/val/epoch_200/',
                      'pretrain_vgg16_cars/val/epoch_0']
    line_type = ["--","--","--", "-", "-", "-"]
    plot_title = 'ABC'
    plot_file = '1.png'
    ap_strs = ['CSM w/ Pose', 'CMR', 'Zhou et. al' , 'CSM', 'DE', 'VGG Pretrain']
    # colors = ['blue', 'green', 'red','cyan','magenta','yellow']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',  'tab:cyan']
    plot_file = 'cars_pr.pdf'
    title = 'PASCAL Cars Keypoint Transfer PR'
if True:
    # plot_mat_files = ['mar12_gt_quat_birds/val/epoch_150/','bird_net_march1/val/epoch_200/', 'feb22_birds_honest_no_vgg/val/epoch_200/','de_baseline_2000tps_wmask_lst/val/epoch_200/', 'pretrain_vgg16_birds/val/epoch_0']
    #plot_mat_files = ['csm_bird_run_gt_pose_scaleLr0pt05_scaleBias0pt75/val/epoch_200/','bird_net_march1/val/epoch_200/', 'csm_bird_run_az_ele_warm_up2000_scaleLR0pt05_scaleBias0pt75/val/epoch_200/','de_baseline_2000tps_wmask_lst/val/epoch_200/', 'pretrain_vgg16_birds/val/epoch_0']
    plot_mat_files = ['csm_bird_run_gt_pose_cr_vis_july/val/epoch_200/','bird_net_march1/val/epoch_200/', 'csm_bird_run_az_ele_warm_up2000_scaleLR0pt05_scaleBias0pt75/val/epoch_200/','de_baseline_2000tps_wmask_lst/val/epoch_200/', 'pretrain_vgg16_birds/val/epoch_0']
    plot_title = 'ABC'
    plot_file = '1.png'
    line_type = ["--","--", "-", "-", "-"]
    ap_strs = ['CSM w/ Pose', 'CMR' , 'CSM', 'DE', 'VGG Pretrain']
    # colors = ['blue', 'green', 'cyan','magenta','yellow']
    colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:purple',  'tab:cyan']
    plot_file = 'birds_pr.pdf'
    title = 'CUBS-Birds Keypoint Transfer PR'



# plot_mat_files = ['mar12_gt_quat_cars/val/epoch_150/','car_net_march1/val/epoch_200/','learn_cycle_baseline/val/epoch_0']
# plot_title = 'ABC'
# plot_file = '1.png'
# # ap_strs = ['Ours', 'CMR', 'Reproject', 'Zhou et. al']
# ap_strs = ['CSM (ours)', 'CMR', 'Zhou et. al']
# plot_file = 'cars_pose.pdf'
# title = 'PASCAL3D+ Cars with Pose Annotation'


# plot_mat_files = ['feb22_honest_car_no_vgg_10_models/val/epoch_200/','de_cars_baseline_2000tps_wmask_unet/val/epoch_200/', 'pretrain_vgg16_cars/val/epoch_0']
# plot_title = 'ABC'
# plot_file = '1.png'
# ap_strs = ['CSM (ours)', 'DE', 'VGG Pretrain']
# plot_file = 'cars_no_pose.pdf'
# title = 'PASCAL3D+ Cars without Pose Annotation'

# plot_mat_files = ['mar12_gt_quat_birds/val/epoch_150/','bird_net_march1/val/epoch_200/']
# ap_strs = ['CSM (ours)', 'CMR']
# plot_file = 'birds_pose.pdf'
# title = 'CUBS-Birds with Pose Annotation'


# plot_mat_files = ['feb22_birds_honest_no_vgg/val/epoch_200/','de_baseline_2000tps_wmask_lst/val/epoch_200/', 'pretrain_vgg16_birds/val/epoch_0']
# plot_title = 'ABC'
# ap_strs = ['CSM (ours)', 'DE', 'VGG']
# plot_file = 'birds_no_pose.pdf'
# title = 'CUBS-Birds without Pose Annotation'

# plot_mat_files = ['mar12_gt_quat_cars/val/epoch_150/']

import pdb

def subplots(plt, Y_X, sz_y_sz_x=(10,10)):
  Y,X = Y_X
  sz_y,sz_x = sz_y_sz_x
  plt.rcParams['figure.figsize'] = (X*sz_x, Y*sz_y)
  fig, axes = plt.subplots(Y, X)
  plt.subplots_adjust(wspace=0.1, hspace=0.1)
  return fig, axes

def pr_plots():

  print('Saving plot to {}'.format(osp.abspath(plot_file)))
  # Plot 1 with AP for all, and minus other things one at a time.
  #with sns.axes_style("darkgrid"):
  with plt.style.context('fivethirtyeight'):
    fig, axes = subplots(plt, (1,1), (7,7))
    ax = axes
    legs = []
    for px, mat_file in enumerate(plot_mat_files):
      mat_file = osp.join(cache_path, 'evaluation', mat_file, 'pr_10000.mat')
      a = sio.loadmat(mat_file)
      # pdb.set_trace()
      # for i in np.arange(6, 12):
      # legs.append('{:4.1f} '.format(100*a['ap'][0,1]))
      # legs.append('{:s}: '.format(ap_strs[px]))
      legs.append('{:s}: {:4.1f} '.format(ap_strs[px], 100*a['ap'][0,1]))
      prec = np.array(a['prec'])[:,1]
      rec = np.array(a['rec'])[:,1]
      ax.plot(rec, prec, line_type[px], color=colors[px])

    ax.set_xlim([0, 0.6]); ax.set_ylim([0, 1.0]);
    ax.set_xlabel('Transfer Recall', fontsize=20)
    ax.set_ylabel('Transfer Precision', fontsize=20)
    ax.set_title('{}'.format(title), fontsize=20)
    ax.plot([0,0.6], [0,0], 'k-')
    ax.plot([0,0], [0,1.0], 'k-')
    l = ax.legend(legs, fontsize=23, bbox_to_anchor=(1,1), loc='upper right', framealpha=0.5, frameon=True)
    plt.tick_params(axis='both', which='major', labelsize=20)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
  pr_plots()
