from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pprint
import pdb




def inst_bench_evaluate(pred_labels, gt_labels , scores, ):
    scores = scores + 100
    count = 100
    score_max = scores.max()
    score_min = scores.min()
    score_intervals = np.array(list(range(0,count))).astype(np.float)*(score_max-score_min)/100.0 + score_min
    score_ins  = np.linspace(1, len(scores)-1,100).astype(np.int)
    # pdb.set_trace()
    scores_sort = scores.copy()
    scores_sort.sort()
    # pdb.set_trace()
    sorted_indices = np.argsort(scores)  ## correct
    score_intervals = scores_sort[score_ins]
    ex_intervals = np.ceil(np.linspace(0, len(sorted_indices)-1,count)).astype(int)
    prec  = []
    rec = []
    total_positives = np.sum(gt_labels > 0.5, axis=0)
    
    exs = []
    # for tx, thresh in enumerate(score_intervals):
    #     ex = np.where(scores < thresh)[0]
    # pdb.set_trace()
    for tx in range(0, len(ex_intervals)):
        start_ind = 0
        end_ind = ex_intervals[tx]
        ex = sorted_indices[start_ind:end_ind].reshape(-1)
        
        # pdb.set_trace()

        pl = pred_labels[ex,:]
        gl = gt_labels[ex,:]
        # pdb.set_trace()
        if len(ex) > 0:
            true_positives = np.sum(pl*gl, axis=0)
            false_positives = np.sum((1-gl), axis=0)
        else:
            continue

        # precision =  true_positives / (true_positives + false_positives)  ## len(gl)

        precision =  true_positives / len(ex)  ## len(gl)

        # precision =  true_positives / np.sum(gl,axis=0)  ## sum(gl) only positives


        recall = true_positives/total_positives
        

        exs.append(len(ex)/len(scores))
        prec.append(precision)
        rec.append(recall)

  
    prec = np.stack(prec)
    exs = np.stack(exs)
    rec = np.stack(rec)
    # pdb.set_trace()
    ap_score = prec[0:-1,:]*(rec[1:,:] - rec[0:-1,:])
    ap_score = ap_score.sum(0)

    stats = {'prec': prec, 'rec':rec, 'ap': ap_score, 'ap_str':'APK'}
    return stats