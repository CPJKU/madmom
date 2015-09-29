#!/usr/bin/env python
# encoding: utf-8
"""
Script for matching detections against ground truth annotations.

"""

import numpy as np
import operator
import argparse

from madmom.evaluation.onsets import OnsetEvaluation


def parser():
    """
    Create a parser and parse the arguments.

    :return: the parsed arguments

    """
    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    The script matches a file with detections against multiple files or folder
    with annotations.

    This small tool can help to match annotations to audio files with
    completely unrelated names. Using any automated feature extraction method
    and then matching these detections against the ground-truth sometimes does
    the trick.

    """)
    # files used for evaluation
    p.add_argument('file', help='file to be matched')
    p.add_argument('files', nargs='*',
                   help='files (or folder) of possible matches')
    # verbose
    p.add_argument('-v', dest='verbose', action='count',
                   help='increase verbosity level')
    # parse the arguments
    args = p.parse_args()
    # print the args
    if args.verbose >= 2:
        print args
    # return
    return args


def main():
    """
    Simple annotation matching.

    """
    # parse arguments
    args = parser()

    # get the detections
    detections = np.loadtxt(args.file)
    # dict for storing evaluation scores
    evaluations = {}
    # evaluate against all annotation files
    for f in args.files:
        # load the annotations (use only the first column if more are present)
        annotations = np.loadtxt(f)[:, 0]
        # remove nans
        annotations = annotations[~ np.isnan(annotations)]
        # use OnsetEvaluation
        e = OnsetEvaluation(detections, annotations)
        # save the name of the files and the scores in a dict
        evaluations[f] = e.fmeasure
        # print stats for each file
        if args.verbose:
            print f
            e.to_string()
        # process the next annotation file
    # sort the evaluations by value
    sorted_scores = sorted(evaluations.iteritems(), key=operator.itemgetter(1))
    sorted_scores.reverse()
    # print the top 3 matches
    print 'possible matches for %s:' % args.file
    for i in range(3):
        print ' %.3f: %s' % (sorted_scores[i][1], sorted_scores[i][0])

if __name__ == '__main__':
    main()
