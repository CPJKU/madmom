#!/usr/bin/env python
# encoding: utf-8
"""
Script for calculating the tempo from beat ground truth annotations.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import numpy as np
import argparse
from scipy.cluster.vq import kmeans
from scipy.signal import argrelmax

from madmom.utils import search_files, strip_suffix
from madmom.evaluation.beats import calc_intervals


def main():
    """
    Simple tempo calculation tool.

    """
    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    The script calculates the tempo from beat ground truth annotations.

    """)
    # files used for evaluation
    p.add_argument('files', nargs='*',
                   help='files (or folder) to be corrected')
    p.add_argument('-o', dest='output', default=None,
                   help='output directory')
    p.add_argument('-c', dest='cluster', action='store_true', default=False,
                   help='build tempo cluster (instead of median tempo '
                        'calculation)')
    p.add_argument('--dev', action='store', default=0.2, type=float,
                   help='allowed tempo deviation for the clusters')
    # verbose
    p.add_argument('-v', dest='verbose', action='count',
                   help='increase verbosity level')
    # parse the arguments
    args = p.parse_args()
    # print the args
    if args.verbose >= 2:
        print args

    # correct all files
    for in_file in search_files(args.files, '.beats'):
        # calculate inter beat intervals
        beats = np.loadtxt(in_file)
        if beats.ndim > 1:
            beats = beats[:, 0]
        intervals = calc_intervals(beats)
        # convert to bpm
        bpm = 60. / np.median(intervals)
        # format output string
        output = '%.2f\n' % bpm
        if args.verbose:
            print in_file
            print ' median tempo     ', 60. / np.median(intervals)
        # build clusters for tempo calculation
        if args.cluster:
            # scale logarithmically
            log_intervals = np.log2(intervals)
            # build a histogram
            start = -2.5
            stop = 1
            hist = np.histogram(log_intervals, range=(start, stop),
                                bins=4. * (stop - start) / args.dev)
            bin_counts, bin_centers = hist
            # choose local maxima as starting centroids for k-means clustering
            bin_counts = hist[0].astype(float)
            # add small random numbers to always have a maximum
            bin_counts += 0.5 * np.random.random(len(bin_counts))
            bin_counts[bin_counts < 1] = 0
            # get the starting centroids
            centroids = hist[1][argrelmax(bin_counts)[0]]
            # perform clustering
            means, distortion = kmeans(log_intervals, centroids)
            # calculate the strength of the clusters
            centers = np.concatenate(([-5], means, [5]))
            edges = 0.5 * (centers[1:] + centers[:-1])
            cluster_counts = np.histogram(log_intervals, edges)[0]
            # TODO: normalize strengths with IBI!?
            strengths = cluster_counts / np.sum(cluster_counts).astype(float)
            cluster_order = np.argsort(cluster_counts)[::-1]
            # verbose information
            if args.verbose:
                print ' cluster tempi    ', 60. / 2. ** means[cluster_order]
                print ' cluster strengths', strengths[cluster_order]
            if args.verbose >= 2:
                print bin_counts
                for i in np.nonzero(bin_counts)[0]:
                    if bin_counts[i] == 0:
                        break
                    print ('  bin: %3d counts: %3d (tempo: %.1f)' %
                           (i, bin_counts[i], 60 / 2. ** bin_centers[i]))
            #  format output string
            output = '%.2f\n' % bpm

        # write the tempo file
        with open("%s.bpm" % strip_suffix(in_file, '.beats'), 'wb') as o:
            o.write(output)

if __name__ == '__main__':
    main()
