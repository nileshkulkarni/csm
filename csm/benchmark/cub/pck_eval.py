from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pprint
import pdb
from . import evaluate_pr
import scipy.io as sio
'''
intervals : Define thresholds to evaluate pck score 
kpnames : Keypoint names
bench_stats : stats
'''

def remove_nans(x):
    return x[~np.isnan(x)]


def pck_at_intervals(intervals, error):
    accuracy = []
    for interval in intervals:
        accuracy.append(float(np.round(np.mean(np.array(error)<interval),3)))
    return accuracy

def ck_at_interval(intervals, error):
    cks = []
    for interval in intervals:
        cks.append(np.array(error)<interval)
    return cks ## len(intervals) x error.shape


def benchmark_all_instances(intervals, kpnames, bench_stats, img_size):
    stats = {}
    plot_intervals  = [0.025*i for i in range(40)]
    kp_error_nan_mask = bench_stats['kps_err'][:, :, 1]*1
    pdb.set_trace()
    # valid_inds = 
    kp_error_nan_mask[kp_error_nan_mask < 0.5] = 'nan'
    bench_stats_kps_err  = bench_stats['kps_err'] / img_size
    mean_kp_error = bench_stats_kps_err[:, :, 0] * kp_error_nan_mask
    stats['mean_kp_err'] = [float(t) for t in np.round(np.nanmean(mean_kp_error, 0), 4)]
    stats['median_kp_err'] = [float(t) for t in np.round(np.nanmedian(mean_kp_error, 0), 4)]
    stats['std_kp_err'] = [float(t) for t in np.round(np.nanstd(mean_kp_error, 0), 4)]
    stats['data'] = {}
    stats['pck'] = {}
    stats['interval'] = intervals
    stats['kp_names'] = kpnames
    stats['eval_params'] = {}
    
    for kpx, kp_name in enumerate(kpnames):
        stats['data'][kp_name] = remove_nans(mean_kp_error[:,kpx])
        stats['data'][kp_name].sort()
        stats['data'][kp_name] = [float(t) for t in  stats['data'][kp_name]]
        stats['pck'][kp_name] = pck_at_intervals(intervals, stats['data'][kp_name])
        stats['eval_params'][kp_name] = {}
        stats['eval_params'][kp_name]['thresh'] = plot_intervals
        stats['eval_params'][kp_name]['acc'] = pck_at_intervals(plot_intervals, stats['data'][kp_name])

    return stats

def benchmark_all_instances_2(intervals, kpnames, bench_stats, img_size):
    stats = {}
    plot_intervals  = [0.025*i for i in range(40)]
    kp_error_nan_mask = bench_stats['kps_err'][:, :, 1]*1
   
    # valid_inds = 
    kp_error_nan_mask[kp_error_nan_mask < 0.5] = 'nan'
    bench_stats_kps_err  = bench_stats['kps_err'] / img_size
    mean_kp_error = bench_stats_kps_err[:, :, 0] * kp_error_nan_mask
    stats['mean_kp_err'] = [float(t) for t in np.round(np.nanmean(mean_kp_error, 0), 4)]
    stats['median_kp_err'] = [float(t) for t in np.round(np.nanmedian(mean_kp_error, 0), 4)]
    stats['std_kp_err'] = [float(t) for t in np.round(np.nanstd(mean_kp_error, 0), 4)]
    stats['data'] = {}
    stats['pck'] = {}
    stats['interval'] = intervals
    stats['kp_names'] = kpnames
    stats['eval_params'] = {}

    for kpx, kp_name in enumerate(kpnames):
        stats['data'][kp_name] = remove_nans(mean_kp_error[:,kpx])
        stats['data'][kp_name].sort()
        stats['data'][kp_name] = [float(t) for t in  stats['data'][kp_name]]
        stats['pck'][kp_name] = pck_at_intervals(intervals, stats['data'][kp_name])
        stats['eval_params'][kp_name] = {}
        stats['eval_params'][kp_name]['thresh'] = plot_intervals
        stats['eval_params'][kp_name]['acc'] = pck_at_intervals(plot_intervals, stats['data'][kp_name])

    samples = remove_nans(mean_kp_error.reshape(-1))
    stats['eval_params']['acc'] = pck_at_intervals(intervals, samples.tolist())
    return stats

def benchmark_vis_instances(intervals, dist_thresholds, kpnames, bench_stats, img_size):
    stats = {}
    stats['data'] = {}
    stats['eval_params'] ={}
    stats['pck'] ={}
    stats['interval'] = intervals
    bench_stats_kps_error = 1*bench_stats['kps_err']
    bench_stats_kps_error[:,:,0] = bench_stats_kps_error[:,:,0]/img_size
    ndata_points, nkps, _ = bench_stats['kps_err'].shape
    
    kps_vis1 =  bench_stats['kps1'][:,:,2] > 200 
    kps_vis2 =  bench_stats['kps2'][:,:,2] > 200
    stats['eval_params']['total'] = np.sum(kps_vis1, axis=0) + 1E-10
    for dx, dist_thresh in enumerate(dist_thresholds):
        stats['eval_params'][dx] = {}
        stats['eval_params'][dx]['correct'] = np.zeros((len(kpnames),len(intervals)))
        for kpx, kp_name in enumerate(kpnames):
            valid_inds = np.where(bench_stats_kps_error[:,kpx,2] < dist_thresh)[0].tolist()
            common_inds = np.where(bench_stats_kps_error[:, kpx, 1] > 0.5)[0].tolist()
            valid_inds = set(valid_inds)
            common_inds = set(common_inds)
            ck = ck_at_interval(intervals, bench_stats_kps_error[:,kpx,0])
            ck = np.stack(ck, axis=1)
            ex = np.array(list(common_inds & valid_inds))
            if len(ex) > 0:
                stats['eval_params'][dx]['correct'][kpx] += np.sum(ck[ex,:], axis=0)

            kps_vis1_ind = np.where(kps_vis1[:,kpx])[0]
            kps_vis2_ind = np.where(kps_vis2[:,kpx])[0]
            ex = np.array(list(set(kps_vis1_ind) - set(kps_vis2_ind))).astype(np.int)
            if len(ex) > 0:
                stats['eval_params'][dx]['correct'][kpx] += np.sum(bench_stats_kps_error[ex,kpx,2] > dist_thresh)
        stats['eval_params'][dx]['acc'] =  stats['eval_params'][dx]['correct'] / stats['eval_params']['total'].reshape(-1,1)
    return stats


def collate_all_instances(intervals, kp_names , bench_stats, img_size):
    bench_stats_kps_error  = bench_stats['kps_err']*1
    bench_stats_kps_error[:,:,0] = bench_stats_kps_error[:,:,0]/img_size
    prediction_error = []  # N x 1
    prediction_score = []  # N x 1 
    prediction_label = []  # N x len(intervals)
    gt_label = []

    kps_vis1 =  bench_stats['kps1'][:,:,2] > 200 
    kps_vis2 =  bench_stats['kps2'][:,:,2] > 200

    for kpx, kp_name in enumerate(kp_names):
        common_inds = np.where(bench_stats_kps_error[:, kpx, 1] > 0.5)[0].tolist()
        ck = ck_at_interval(intervals, bench_stats_kps_error[:,kpx,0])
        ck = np.stack(ck, axis=1)
        ex = np.array(list(common_inds))
        if len(ex) > 0:
            prediction_error.append(bench_stats_kps_error[ex,kpx,0])
            prediction_score.append(bench_stats_kps_error[ex,kpx,2])
            prediction_label.append(ck[ex,:]*1)
            gt_label.append(ck[ex,:]*0 + 1)

        kps_vis1_ind = np.where(kps_vis1[:,kpx])[0]
        kps_vis2_ind = np.where(kps_vis2[:,kpx])[0]
        ex = np.array(list(set(kps_vis1_ind) - set(kps_vis2_ind))).astype(np.int)
        if len(ex) > 0:
            prediction_error.append(bench_stats_kps_error[ex,kpx,0])
            prediction_score.append(bench_stats_kps_error[ex,kpx,2])
            prediction_label.append(ck[ex,:]*0)
            gt_label.append(ck[ex,:]*0)

    
    prediction_error = np.concatenate(prediction_error,axis=0)
    prediction_score = np.concatenate(prediction_score, axis=0)
    prediction_label = np.concatenate(prediction_label, axis=0)
    gt_label = np.concatenate(gt_label, axis=0)

    stats = {}
    stats['pred_label'] = prediction_label
    stats['gt_label'] = gt_label
    stats['score'] = prediction_score ## lower the score better it is.
    return stats



import os.path as osp
import json
kp_eval_thresholds = [0.05, 0.1, 0.2]
# kp_eval_thresholds = [0.05, 1.0]

def run_evaluation(bench_stats, n_iter, results_dir, img_size, kp_names, dist_thresholds ):
    json_file = osp.join(results_dir, 'stats_m1_{}.json'.format(n_iter))
    stats_m1 = benchmark_all_instances_2(kp_eval_thresholds, kp_names, bench_stats, img_size)
    stats = stats_m1
    print(' Method 1 | Keypoint | Median Err | Mean Err | STD Err')
    pprint.pprint(zip(stats['kp_names'], stats['median_kp_err'], stats['mean_kp_err'], stats['std_kp_err']))
    print('PCK Values')
    pprint.pprint(stats['interval'])
    pprint.pprint(stats['pck'])
    mean_pck = {}
    # pdb.set_trace()
    for i, thresh in enumerate(stats['interval']):
        mean_pck[thresh] = []
        for kp_name in kp_names:
            mean_pck[thresh].append(stats['pck'][kp_name][i])

    mean_pck = {k: np.mean(np.array(t)) for k, t in mean_pck.items()}
    pprint.pprint('Mean PCK  ')
    pprint.pprint(mean_pck)

    print('Instance Average **** ')
    pprint.pprint(stats['eval_params']['acc'])
    print('########################## ')
    
    with open(json_file, 'w') as f:
        json.dump(stats, f)

    stats_m1 = benchmark_vis_instances(
        kp_eval_thresholds, dist_thresholds, kp_names, bench_stats, img_size)
    stats = stats_m1

    mean_pck = {}
    # points_per_kp = {k: v for k, v in zip(kp_names, stats['eval_params'][0]['npoints'])}
    # points_per_thresh = np.sum(np.array(points_per_kp.values()))
    for dx, dthresh in enumerate(dist_thresholds):
        mean_pck[dx] = {}
        for i, thresh in enumerate(stats['interval']):
            mean_pck[dx][thresh] = []
            for kx, kp_name in enumerate(kp_names):
                mean_pck[dx][thresh].append(stats['eval_params'][dx]['acc'][kx, i])

        mean_pck[dx] = {k: np.round(np.mean(np.array(t)), 4) for k, t in mean_pck[dx].items()}

    # pdb.set_trace()
    print('***** Distance Thresholds ***** ')
    pprint.pprint('Mean PCK Acc')
    pprint.pprint(mean_pck)
    # pprint.pprint(points_per_kp)



    stats = collate_all_instances(kp_eval_thresholds, kp_names, bench_stats, img_size)
    pr_stats = evaluate_pr.inst_bench_evaluate(stats['pred_label'], stats['gt_label'], stats['score'])
    pr_mat_file = osp.join(results_dir, 'pr_{}.mat'.format(n_iter))

    sio.savemat(pr_mat_file, pr_stats)
    return stats_m1
