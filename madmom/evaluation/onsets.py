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

from . import calc_errors, Evaluation, SumEvaluation, MeanEvaluation


# evaluation function for onset detection
def onset_evaluation(detections, targets, window):
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

    Note: The true negative array is empty, because we are not interested in
          this class, since it is ~20 times as big as the onset class.

    """
    # sort the detections and targets
    det = sorted(detections.tolist())
    tar = sorted(targets.tolist())
    # cache variables
    det_length = len(detections)
    tar_length = len(targets)
    det_index = 0
    tar_index = 0
    # arrays for collecting the detections
    tp = []
    fp = []
    fn = []
    while det_index < det_length and tar_index < tar_length:
        # fetch the first detection
        d = det[det_index]
        # fetch the first target
        t = tar[tar_index]
        # compare them
        # FIXME: use < instead? some beat stuff uses < as well...
        if abs(d - t) <= window:
            # TP detection
            tp.append(d)
            # increase the detection and target index
            det_index += 1
            tar_index += 1
        elif d < t:
            # FP detection
            fp.append(d)
            # increase the detection index
            det_index += 1
            # do not increase the target index
        elif d > t:
            # we missed a target: FN
            fn.append(t)
            # do not increase the detection index
            # increase the target index
            tar_index += 1
    # the remaining detections are FP
    fp.extend(det[det_index:])
    # the remaining targets are FN
    fn.extend(tar[tar_index:])
    # transform them back to numpy arrays
    tp = np.asarray(tp)
    fp = np.asarray(fp)
    fn = np.asarray(fn)
    # check calculation
    assert tp.size + fp.size == detections.size, 'bad TP / FP calculation'
    assert tp.size + fn.size == targets.size, 'bad FN calculation'
    # return the arrays
    return tp, fp, np.zeros(0), fn


#def onset_evaluation(detections, targets, window):
#    """
#    Count the true and false detections of the given detections and targets.
#
#    :param detections: array with detected onsets [seconds]
#    :param targets:    array with target onsets [seconds]
#    :param window:     detection window [seconds]
#    :return:           tuple of tp, fp, tn, fn numpy arrays
#
#    tp: array with true positive detections
#    fp: array with false positive detections
#    tn: array with true negative detections (this one is empty!)
#    fn: array with false negative detections
#
#    Note: the true negative array is empty, because we are not interested in
#          this class, since it is ~20 times as big as the onset class.
#
#    """
#     FIXME: is there a numpy like way to achieve the same behavior as above
#     i.e. detections and targets can match only once?
#    from .helpers import calc_absolute_errors
#    # no detections
#    if detections.size == 0:
#        # all targets are FNs
#        return np.zeros(0), np.zeros(0), np.zeros(0), targets
#    # for TP & FP, calc the absolute errors of detections wrt. targets
#    errors = calc_absolute_errors(detections, targets)
#    # true positive detections
#    tp = detections[errors <= window]
#    # the remaining detections are FP
#    fp = detections[errors > window]
#    # for FN, calc the absolute errors of targets wrt. detections
#    errors = calc_absolute_errors(targets, detections)
#    fn = targets[errors > window]
#    # return the arrays
#    return tp, fp, np.zeros(0), fn


# default values
WINDOW = 0.025
COMBINE = 0.03


# for onset evaluation with Precision, Recall, F-measure use the Evaluation
# class and just define the evaluation function
class OnsetEvaluation(Evaluation):
    """
    Simple class for measuring Precision, Recall and F-measure.

    """
    def __init__(self, detections, targets, window=WINDOW):
        super(OnsetEvaluation, self).__init__()
        self.detections = detections
        self.targets = targets
        # evaluate
        numbers = onset_evaluation(detections, targets, window)
        self._tp, self._fp, self._tn, self._fn = numbers
        # init errors
        self._errors = None

    @property
    def errors(self):
        """
        Absolute errors of all true positive detections relative to the closest
        targets.

        """
        if self._errors is None:
            if self.num_tp == 0:
                # FIXME: what is the error in case of no TPs?
                self._errors = np.zeros(0)
            else:
                self._errors = calc_errors(self.tp, self.targets)
        return self._errors


def parser():
    """
    Create a parser and parse the arguments.

    :return: the parsed arguments

    """
    import argparse
    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    The script evaluates a file or folder with detections against a file or
    folder with targets. Extensions can be given to filter the detection and
    target file lists.
    """)
    # files used for evaluation
    p.add_argument('files', nargs='*',
                   help='files (or folder) to be evaluated')
    # extensions used for evaluation
    p.add_argument('-d', dest='det_ext', action='store', default='.onsets.txt',
                   help='extension of the detection files')
    p.add_argument('-t', dest='tar_ext', action='store', default='.onsets',
                   help='extension of the target files')
    # parameters for evaluation
    p.add_argument('-w', dest='window', action='store', type=float,
                   default=WINDOW,
                   help='evaluation window (+/- the given size) '
                        '[seconds, default=%(default).3f]')
    p.add_argument('-c', dest='combine', action='store', type=float,
                   default=COMBINE,
                   help='combine target events within this range '
                        '[seconds, default=%(default).3f]')
    p.add_argument('--delay', action='store', type=float, default=0.,
                   help='add given delay to all detections [seconds]')
    p.add_argument('--tex', action='store_true',
                   help='format errors for use is .tex files')
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
    Simple onset evaluation.

    """
    from ..utils import files, match_file, load_events, combine_events

    # parse arguments
    args = parser()

    # get detection and target files
    det_files = files(args.files, args.det_ext)
    tar_files = files(args.files, args.tar_ext)
    # quit if no files are found
    if len(det_files) == 0:
        print "no files to evaluate. exiting."
        exit()

    # sum and mean evaluation for all files
    sum_eval = SumEvaluation()
    mean_eval = MeanEvaluation()
    # evaluate all files
    for det_file in det_files:
        # get the detections file
        detections = load_events(det_file)
        # shift the detections if needed
        if args.delay != 0:
            detections += args.delay
        # get the matching target files
        matches = match_file(det_file, tar_files, args.det_ext, args.tar_ext)
        # quit if any file does not have a matching target file
        if len(matches) == 0:
            print " can't find a target file found for %s. exiting." % det_file
            exit()
        # do a mean evaluation with all matched target files
        me = MeanEvaluation()
        for tar_file in matches:
            # load the targets
            targets = load_events(tar_file)
            # combine the targets if needed
            if args.combine > 0:
                targets = combine_events(targets, args.combine)
            # add the OnsetEvaluation to mean evaluation
            me.append(OnsetEvaluation(detections, targets, window=args.window))
            # process the next target file
        # print stats for each file
        if args.verbose:
            print det_file
            print me.print_errors('  ', args.tex)
        # add the resulting sum counter
        sum_eval += me
        mean_eval.append(me)
        # process the next detection file
    # print summary
    print 'sum for %i files:' % (len(det_files))
    print sum_eval.print_errors('  ', args.tex)
    print 'mean for %i files:' % (len(det_files))
    print mean_eval.print_errors('  ', args.tex)

if __name__ == '__main__':
    main()
