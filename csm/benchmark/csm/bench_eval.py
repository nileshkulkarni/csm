
import numpy as np

def camera_benchmark(intervals, bench):
    stats = {}
    for key in intervals:
        stats[key] = {}
        stats[key]['mean'] = float(np.mean(bench[key]))
        stats[key]['median'] = float(np.median(bench[key]))

        errs = bench[key].copy()
        thresh = intervals[key]
        acc = []
        for t in thresh:
            acc.append(float(np.mean(errs < t)))
        stats[key]['acc'] = acc
    return stats