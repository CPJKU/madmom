#!/usr/bin/env python
# encoding: utf-8
"""
Script for calculating the tempo from beat ground truth annotations.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import numpy as np
import argparse

from madmom.utils import files, load_events, strip_ext
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
    p.add_argument('--meter', action='store_true', default=False,
                   help='also write .meter files')
    # verbose
    p.add_argument('-v', dest='verbose', action='count',
                   help='increase verbosity level')
    # parse the arguments
    args = p.parse_args()
    # print the args
    if args.verbose >= 2:
        print args

    # correct all files
    for in_file in files(args.files, '.beats'):
        if args.verbose:
            print in_file

        # create a histogram of inter beat intervals
        beats = np.loadtxt(in_file)[:, 0]
        intervals = calc_intervals(beats)

        # write the tempo file
        with open("%s.bpm" % strip_ext(in_file, '.beats'), 'wb') as o:
            # convert to bpm
            bpm = 60. / np.median(intervals)
            o.write('%.2f\n' % bpm)

if __name__ == '__main__':
    main()
