import matplotlib
matplotlib.use('Agg')
import platform
import os.path as osp
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdb

'''
python icn/benchmark/cub/pck_plots.py 
'''
eval_set = 'val'
net_name = 'birds_gt_camera_thresh0pt5_t2_s100' ; epoch = 480
# net_name = 'birds_gt_camera_baseline' ; epoch = 0
# net_name = 'birds_gt_camera_baseline' ; epoch = 200

stat_file_name  = 'stats_m2_1000.json'
suffix = 'm2'
curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', '..', 'cachedir')
plots_dir = os.path.join(cache_path, 'evaluation', eval_set, 'plots')
cm = plt.get_cmap('jet')
keypoint_cmap = [cm(i * 17) for i in range(15)]

def subplots(plt, Y_X, sz_y_sz_x=(10, 10)):
    Y, X = Y_X
    sz_y, sz_x = sz_y_sz_x
    plt.rcParams['figure.figsize'] = (X*sz_x, Y*sz_y)
    fig, axes = plt.subplots(Y, X)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    return fig, axes


def pck_plots(net_name, set_number, eval_set):
    dir_name = os.path.join(cache_path, 'evaluation')
    json_file = os.path.join(dir_name, net_name, eval_set, "epoch_{}".format(epoch), stat_file_name)
    with open(json_file, 'rt') as f:
        stats = json.load(f)
    
    kpnames = stats['kp_names']
    interval = stats['interval']

    plot_file = os.path.join(dir_name, net_name, eval_set, "epoch_{}".format(epoch), 'pck_plot_{}.pdf'.format(suffix))

    print('Saving plot to {}'.format(osp.abspath(plot_file)))
    # Plot 1 with AP for all, and minus other things one at a time.
    # with sns.axes_style("darkgrid"):
    with plt.style.context('fivethirtyeight'):
        fig, axes = subplots(plt, (1, 1), (7, 7))
        ax = axes
        legs = []
        for kpx, kp_name in enumerate(kpnames):
            acc = np.array(stats['eval_params'][kp_name]['acc'])
            thresh = np.array(stats['eval_params'][kp_name]['thresh'])
          
            ax.plot(thresh, acc, '-', linewidth=2, color=keypoint_cmap[kpx])
            legs.append('{:s}'.format(kp_name))
        ax.set_xlim([0, 0.2])
        ax.set_ylim([0, 1.0])
        ax.set_xlabel('Error Threshold', fontsize=20)
        ax.set_ylabel('Accuracy', fontsize=20)
        ax.set_title('PCK Plots {}'.format(eval_set), fontsize=20)

        # l = ax.legend(legs, fontsize=10, bbox_to_anchor=(1, 1), loc='lower right', framealpha=0.5, frameon=True)
        l = ax.legend(legs, fontsize=10, bbox_to_anchor=(1, 1),  framealpha=0.5, frameon=True)

        ax.plot([0, 1], [0, 0], 'k-')
        ax.plot([0, 0], [0, 1], 'k-')
        plt.tick_params(axis='both', which='minor', labelsize=20)
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig(plot_file, bbox_inches='tight')
        plt.close(fig)

    return


if __name__ == '__main__':
    pck_plots(net_name, 0, eval_set)
