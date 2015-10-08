#!/usr/bin/env python
# encoding: utf-8
"""
This file contains onset evaluation functionality.

It is described in:

"Evaluating the Online Capabilities of Onset Detection Methods"
Sebastian Böck, Florian Krebs and Markus Schedl.
Proceedings of the 13th International Society for Music Information Retrieval
Conference (ISMIR), 2012.

"""

import numpy as np

from . import (evaluation_io, calc_errors, Evaluation, SumEvaluation,
               MeanEvaluation)


# evaluation function for onset detection
# TODO: find a better name, this is misleading since it does not evaluate the
#       detections against the annotations per se
def onset_evaluation(detections, annotations, window):
    """
    Determine the true/false positive/negative detections.

    :param detections:  detected onsets [seconds, list or numpy array]
    :param annotations: annotated onsets [seconds, list or numpy array]
    :param window:      detection window [seconds, float]
    :return:            tuple of arrays (tp, fp, tn, fn)
                        tp: array with true positive detections
                        fp: array with false positive detections
                        tn: array with true negative detections
                        fn: array with false negative detections

    Note: The true negative list is empty, because we are not interested in
          this class, since it is ~20 times as big as the onset class.

    """
    # convert numpy array to lists if needed
    if isinstance(detections, np.ndarray):
        detections = detections.tolist()
    if isinstance(annotations, np.ndarray):
        annotations = annotations.tolist()
    # sort the detections and annotations
    det = sorted(detections)
    ann = sorted(annotations)
    # cache variables
    det_length = len(detections)
    ann_length = len(annotations)
    det_index = 0
    ann_index = 0
    # init TP, FP, TN and FN lists
    tp = []
    fp = []
    tn = []
    fn = []
    while det_index < det_length and ann_index < ann_length:
        # fetch the first detection
        d = det[det_index]
        # fetch the first annotation
        t = ann[ann_index]
        # compare them
        if abs(d - t) <= window:
            # TP detection
            tp.append(d)
            # increase the detection and annotation index
            det_index += 1
            ann_index += 1
        elif d < t:
            # FP detection
            fp.append(d)
            # increase the detection index
            det_index += 1
            # do not increase the annotation index
        elif d > t:
            # we missed a annotation: FN
            fn.append(t)
            # do not increase the detection index
            # increase the annotation index
            ann_index += 1
    # the remaining detections are FP
    fp.extend(det[det_index:])
    # the remaining annotations are FN
    fn.extend(ann[ann_index:])
    # check calculation
    assert len(tp) + len(fp) == len(detections), 'bad TP / FP calculation'
    assert len(tp) + len(fn) == len(annotations), 'bad FN calculation'
    # return the arrays
    return np.asarray(tp), np.asarray(fp), np.asarray(tn), np.asarray(fn)


# default values
WINDOW = 0.025
COMBINE = 0.03


# for onset evaluation with Precision, Recall, F-measure use the Evaluation
# class and just define the evaluation and error functions
class OnsetEvaluation(Evaluation):
    """
    Simple class for measuring Precision, Recall and F-measure.

    """

    def __init__(self, detections, annotations, window=WINDOW, **kwargs):
        """
        Evaluates onset detections against annotations.

        :param detections:  onset detections [list or numpy array]
        :param annotations: onset annotations [list or numpy array]
        :param window:      evaluation window [seconds, float]
        :param kwargs:      additional keywords are ignored

        """
        # convert the detections and annotations
        detections = np.asarray(sorted(detections), dtype=np.float)
        annotations = np.asarray(sorted(annotations), dtype=np.float)
        # evaluate
        numbers = onset_evaluation(detections, annotations, window)
        # tp, fp, tn, fn = numbers
        super(OnsetEvaluation, self).__init__(*numbers, **kwargs)
        # calculate errors
        self.errors = calc_errors(self.tp, annotations)

    @property
    def mean_error(self):
        """Mean of the errors."""
        if len(self.errors) == 0:
            return np.nan
        return np.mean(self.errors)

    @property
    def std_error(self):
        """Standard deviation of the errors."""
        if len(self.errors) == 0:
            return np.nan
        return np.std(self.errors)

    def tostring(self, **kwargs):
        """
        Format the evaluation metrics as a human readable string.

        :param kwargs: additional arguments will be ignored
        :return:       evaluation metrics formatted as a human readable string

        """
        ret = ''
        if self.name is not None:
            ret += '%s\n  ' % self.name
        ret += 'Onsets: %5d TP: %5d FP: %5d FN: %5d Precision: %.3f ' \
               'Recall: %.3f F-measure: %.3f mean: %5.1f ms std: %5.1f ms' % \
               (self.num_annotations, self.num_tp, self.num_fp, self.num_fn,
                self.precision, self.recall, self.fmeasure,
                self.mean_error * 1000., self.std_error * 1000.)
        return ret

    def __str__(self):
        return self.tostring()


class OnsetSumEvaluation(SumEvaluation, OnsetEvaluation):
    """
    Class for summing onset evaluations.

    """

    @property
    def errors(self):
        """Errors of the true positive detections wrt. the ground truth."""
        if len(self.eval_objects) == 0:
            # return empty array
            return np.zeros(0)
        return np.concatenate([e.errors for e in self.eval_objects])


class OnsetMeanEvaluation(MeanEvaluation, OnsetSumEvaluation):
    """
    Class for averaging onset evaluations.

    """

    @property
    def mean_error(self):
        """Mean of the errors."""
        return np.nanmean([e.mean_error for e in self.eval_objects])

    @property
    def std_error(self):
        """Standard deviation of the errors."""
        return np.nanmean([e.std_error for e in self.eval_objects])

    def tostring(self, **kwargs):
        """
        Format the evaluation metrics as a human readable string.

        :param kwargs: additional arguments will be ignored
        :return:       evaluation metrics formatted as a human readable string

        """
        # format with floats instead of integers
        ret = ''
        if self.name is not None:
            ret += '%s\n  ' % self.name
        ret += 'Onsets: %5.2f TP: %5.2f FP: %5.2f FN: %5.2f ' \
               'Precision: %.3f Recall: %.3f F-measure: %.3f ' \
               'mean: %5.1f ms std: %5.1f ms' % \
               (self.num_annotations, self.num_tp, self.num_fp, self.num_fn,
                self.precision, self.recall, self.fmeasure,
                self.mean_error * 1000., self.std_error * 1000.)
        return ret


def add_parser(parser):
    """
    Add an onset evaluation sub-parser to an existing parser.

    :param parser: existing argparse parser
    :return:       onset evaluation sub-parser and evaluation parameter group

    """
    import argparse
    # add beat evaluation sub-parser to the existing parser
    p = parser.add_parser(
        'onsets', help='onset evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''
    This program evaluates pairs of files containing the onset annotations and
    detections. Suffixes can be given to filter them from the list of files.

    Each line represents an onset and must have the following format:
    `onset_time`.

    Lines starting with # are treated as comments and are ignored.

    ''')
    # set defaults
    p.set_defaults(eval=OnsetEvaluation,
                   sum_eval=OnsetSumEvaluation,
                   mean_eval=OnsetMeanEvaluation)
    # file I/O
    evaluation_io(p, ann_suffix='.onsets', det_suffix='.onsets.txt')
    # evaluation parameters
    g = p.add_argument_group('onset evaluation arguments')
    g.add_argument('-w', dest='window', action='store', type=float,
                   default=WINDOW,
                   help='evaluation window (+/- the given size) '
                        '[seconds, default=%(default).3f]')
    g.add_argument('-c', dest='combine', action='store', type=float,
                   default=COMBINE,
                   help='combine annotation events within this range '
                        '[seconds, default=%(default).3f]')
    g.add_argument('--delay', action='store', type=float, default=0.,
                   help='add given delay to all detections [seconds]')
    # return the sub-parser and evaluation argument group
    return p, g
