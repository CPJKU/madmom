#!/usr/bin/env python
# encoding: utf-8
"""
Script for calculating the length of the given audio files.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import argparse
import numpy as np
from madmom.audio.signal import Signal


def main():
    """
    Simple length calculation tool.

    """
    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    The script calculates the length of the given audio files.

    """)
    # files used for evaluation
    p.add_argument('files', nargs='*',
                   help='files (or folder) to be corrected')
    # verbose
    p.add_argument('-v', dest='verbose', action='count',
                   help='increase verbosity level')
    # parse the arguments
    args = p.parse_args()
    # print the args
    if args.verbose >= 2:
        print args

    lengths = []
    # get the length of all files
    for f in args.files:
        length = Signal(f).length
        lengths.append(length)
        if args.verbose:
            print "%s:\t%.2f" % (f, length)

    length = np.sum(lengths)
    h = 0
    m = 0
    s = length
    if length > 60:
        m = s / 60
        s %= 60
    if m > 60:
        h = m / 60
        m %= 60

    print len(lengths), "files"
    print "total length:  %6.2f s (%d h %d m %.2f s)" % (length, h, m, s)
    print "mean length:   %6.2f s" % np.mean(lengths)
    print "median length: %6.2f s" % np.median(lengths)

if __name__ == '__main__':
    main()
