#!/usr/bin/env python
# encoding: utf-8
"""
This file contains onset evaluation functionality.

It is described in:

"Evaluating the Online Capabilities of Onset Detection Methods"
by Sebastian Böck, Florian Krebs and Markus Schedl
in Proceedings of the 13th International Society for Music Information
Retrieval Conference (ISMIR), 2012

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import numpy as np

from .simple import Evaluation, SumEvaluation, MeanEvaluation


# evaluation function for onset detection
def count_errors(detections, targets, window):
    """
    Count the true and false detections of the given detections and targets.

    :param detections: array with detected onsets [seconds]
    :param targets:    array with target onsets [seconds]
    :param window:     detection window [seconds]
    :return:           tuple of tp, fp, tn, fn numpy arrays

    tp: array with true positive detections
    fp: array with false positive detections
    tn: array with true negative detections (this one is empty!)
    fn: array with false negative detections

    Note: the true negative array is empty, because we are not interested in
          this class, since it is ~20 times as big as the onset class.

    """
    from .helpers import calc_absolute_errors
    # no detections
    if detections.size == 0:
        # all targets are FNs
        return np.empty(0), np.empty(0), np.empty(0), targets
    # for TP & FP, calc the absolute errors of detections wrt. targets
    errors = calc_absolute_errors(detections, targets)
    # true positive detections
    tp = detections[errors <= window]
    # the remaining detections are FP
    fp = detections[errors > window]
    # for FN, calc the absolute errors of targets wrt. detections
    errors = np.asarray(calc_absolute_errors(targets, detections))
    fn = targets[errors > window]
    # return the arrays
    return tp, fp, np.empty(0), fn


# default values
WINDOW = 0.025


# for onset evaluation with Presicion, Recall, F-measure use the Evaluation
# class and just define the evaluation function
class OnsetEvaluation(Evaluation):
    """
    Simple class for measuring Precision, Recall and F-measure.

    """
    def __init__(self, detections, targets, window=WINDOW):
        super(OnsetEvaluation, self).__init__(detections, targets, count_errors, window=window)


class SumOnsetEvaluation(SumEvaluation):
    pass


class MeanOnsetEvaluation(MeanEvaluation):
    pass


def parser():
    import argparse

    # define parser
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    If invoked without any parameters the script evaluates pairs of files with
    the targets (.onsets) and detection (.onsets.txt) as simple text files with
    one onset time-stamp per line.

    """)
    p.add_argument('files', metavar='files', nargs='+', help='path or files to be evaluated (list of files being filtered according to -d and -t arguments)')
    # extensions used for evaluation
    p.add_argument('-d', dest='detections', action='store', default='.onsets.txt', help='extensions of the detections [default: .onsets.txt]')
    p.add_argument('-t', dest='targets', action='store', default='.onsets', help='extensions of the targets [default: .onsets]')
    # parameters for evaluation
    p.add_argument('-w', dest='window', action='store', default=0.025, type=float, help='evaluation window (+/- the given size) [seconds, default=0.025]')
    p.add_argument('-c', dest='combine', action='store', default=0.03, type=float, help='combine target events within this range [seconds, default=0.03]')
    p.add_argument('--delay', action='store', default=0., type=float, help='add given delay to all detections [seconds]')
    p.add_argument('--tex', action='store_true', help='format errors for use in .tex files')
    # verbose
    p.add_argument('-v', dest='verbose', action='count', help='increase verbosity level')
    # parse the arguments
    args = p.parse_args()
    # print the args
    if args.verbose >= 2:
        print args
    # return
    return args


def main():
    from ..utils.helpers import files, load_events, combine_events

    # parse the arguments
    args = parser()

    # TODO: find a better way to determine the corresponding detection/target
    # files from a given list/path of files

    # filter target files
    tar_files = files(args.files, args.targets)
    # filter detection files
    det_files = files(args.files, args.detections)
    # must be the same number FIXME: find better solution which checks the names
    assert len(tar_files) == len(det_files), "different number of targets (%i) and detections (%i)" % (len(tar_files), len(det_files))

    # sum counter for all files
    sum_counter = SumOnsetEvaluation()
    mean_counter = MeanOnsetEvaluation()
    # evaluate all files
    for i in range(len(det_files)):
        detections = load_events(det_files[i])
        targets = load_events(tar_files[i])
        # combine the targets if needed
        if args.combine > 0:
            targets = combine_events(targets, args.combine)
        # shift the detections if needed
        if args.delay != 0:
            detections += args.delay
        # evaluate the onsets
        oe = OnsetEvaluation(detections, targets, args.window)
        # print stats for each file
        if args.verbose:
            print det_files[i]
            oe.print_errors(args.tex)
        # add to sum counter
        sum_counter += oe
        mean_counter += oe
    # print summary
    print 'sum for %i files; detection window %.1f ms (+- %.1f ms)' % (len(det_files), args.window * 2000, args.window * 1000)
    sum_counter.print_errors(args.tex)
    print 'mean for %i files; detection window %.1f ms (+- %.1f ms)' % (len(det_files), args.window * 2000, args.window * 1000)
    mean_counter.print_errors(args.tex)

if __name__ == '__main__':
    main()
