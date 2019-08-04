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
net_name = sys.argv[1]
# epoch_num = 200
# iter_number = 10000

epoch_num = 200
iter_number = 10000

suffix = '_False'
# suffix = '_True'

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', '..', 'cachedir')
import pdb

def subplots(plt, Y_X, sz_y_sz_x=(10,10)):
  Y,X = Y_X
  sz_y,sz_x = sz_y_sz_x
  plt.rcParams['figure.figsize'] = (X*sz_x, Y*sz_y)
  fig, axes = plt.subplots(Y, X)
  plt.subplots_adjust(wspace=0.1, hspace=0.1)
  return fig, axes

def pr_plots(net_name, epoch_num, iter_number):
  dir_name = os.path.join(cache_path, 'evaluation')
  #json_file = os.path.join(dir_name, set_number, net_name, 'eval_set{}_0.json'.format(set_number))
  mat_file = os.path.join(dir_name, net_name, eval_set, 'epoch_{}'.format(epoch_num),'pr_{}.mat'.format(iter_number))
  a = sio.loadmat(mat_file)
  plot_dir = osp.join(dir_name, net_name,'plots','epoch_{}'.format(epoch_num))
  if not osp.exists(plot_dir):
    os.makedirs(plot_dir)

  plot_file = os.path.join(dir_name, net_name,'plots','epoch_{}'.format(epoch_num),'pr_plot_{}.pdf'.format(iter_number))

  print('Saving plot to {}'.format(osp.abspath(plot_file)))
  # Plot 1 with AP for all, and minus other things one at a time.
  #with sns.axes_style("darkgrid"):
  with plt.style.context('fivethirtyeight'):
    fig, axes = subplots(plt, (1,1), (7,7))
    ax = axes
    legs = []
    # for i in np.arange(6, 12):
    legs.append('{:s} {:4.1f} '.format(a['ap_str'][0], 100*a['ap'][0,1]))
    prec = np.array(a['prec'])[:,1]
    rec = np.array(a['rec'])[:,1]
    ax.plot(rec, prec, '-')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1]);
    ax.set_xlabel('Recall', fontsize=20)
    ax.set_ylabel('Precision', fontsize=20)
    ax.set_title('Precision Recall Plots on', fontsize=20)
    ax.plot([0,1], [0,0], 'k-')
    ax.plot([0,0], [0,1], 'k-')
    l = ax.legend(legs, fontsize=18, bbox_to_anchor=(0,0), loc='lower left', framealpha=0.5, frameon=True)
    plt.tick_params(axis='both', which='major', labelsize=20)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close(fig)
    print(plot_file)


if __name__ == '__main__':
  pr_plots(net_name, epoch_num, iter_number)
