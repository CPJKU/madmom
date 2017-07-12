# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains onset evaluation functionality described in [1]_:

References
----------
.. [1] Sebastian Böck, Florian Krebs and Markus Schedl,
       "Evaluating the Online Capabilities of Onset Detection Methods",
       Proceedings of the 13th International Society for Music Information
       Retrieval Conference (ISMIR), 2012.

"""

from __future__ import absolute_import, division, print_function

import numpy as np

from . import Evaluation, MeanEvaluation, SumEvaluation, evaluation_io
from madmom.io import load_onsets
from ..utils import combine_events

# default onset evaluation values
WINDOW = 0.025
COMBINE = 0.03


# onset evaluation function
def onset_evaluation(detections, annotations, window=WINDOW):
    """
    Determine the true/false positive/negative detections.

    Parameters
    ----------
    detections : numpy array
        Detected notes.
    annotations : numpy array
        Annotated ground truth notes.
    window : float, optional
        Evaluation window [seconds].

    Returns
    -------
    tp : numpy array, shape (num_tp,)
        True positive detections.
    fp : numpy array, shape (num_fp,)
        False positive detections.
    tn : numpy array, shape (0,)
        True negative detections (empty, see notes).
    fn : numpy array, shape (num_fn,)
        False negative detections.
    errors : numpy array, shape (num_tp,)
        Errors of the true positive detections wrt. the annotations.

    Notes
    -----
    The returned true negative array is empty, because we are not interested
    in this class, since it is magnitudes bigger than true positives array.

    """
    # make sure the arrays have the correct types and dimensions
    detections = np.asarray(detections, dtype=np.float)
    annotations = np.asarray(annotations, dtype=np.float)
    # TODO: right now, it only works with 1D arrays
    if detections.ndim > 1 or annotations.ndim > 1:
        raise NotImplementedError('please implement multi-dim support')

    # init TP, FP, FN and errors
    tp = np.zeros(0)
    fp = np.zeros(0)
    tn = np.zeros(0)  # we will not alter this array
    fn = np.zeros(0)
    errors = np.zeros(0)

    # if neither detections nor annotations are given
    if len(detections) == 0 and len(annotations) == 0:
        # return the arrays as is
        return tp, fp, tn, fn, errors
    # if only detections are given
    elif len(annotations) == 0:
        # all detections are FP
        return tp, detections, tn, fn, errors
    # if only annotations are given
    elif len(detections) == 0:
        # all annotations are FN
        return tp, fp, tn, annotations, errors

    # window must be greater than 0
    if float(window) <= 0:
        raise ValueError('window must be greater than 0')

    # sort the detections and annotations
    det = np.sort(detections)
    ann = np.sort(annotations)
    # cache variables
    det_length = len(detections)
    ann_length = len(annotations)
    det_index = 0
    ann_index = 0
    # iterate over all detections and annotations
    while det_index < det_length and ann_index < ann_length:
        # fetch the first detection
        d = det[det_index]
        # fetch the first annotation
        a = ann[ann_index]
        # compare them
        if abs(d - a) <= window:
            # TP detection
            tp = np.append(tp, d)
            # append the error to the array
            errors = np.append(errors, d - a)
            # increase the detection and annotation index
            det_index += 1
            ann_index += 1
        elif d < a:
            # FP detection
            fp = np.append(fp, d)
            # increase the detection index
            det_index += 1
            # do not increase the annotation index
        elif d > a:
            # we missed a annotation: FN
            fn = np.append(fn, a)
            # do not increase the detection index
            # increase the annotation index
            ann_index += 1
        else:
            # can't match detected with annotated onset
            raise AssertionError('can not match % with %', d, a)
    # the remaining detections are FP
    fp = np.append(fp, det[det_index:])
    # the remaining annotations are FN
    fn = np.append(fn, ann[ann_index:])
    # check calculations
    if len(tp) + len(fp) != len(detections):
        raise AssertionError('bad TP / FP calculation')
    if len(tp) + len(fn) != len(annotations):
        raise AssertionError('bad FN calculation')
    if len(tp) != len(errors):
        raise AssertionError('bad errors calculation')
    # convert to numpy arrays and return them
    return np.array(tp), np.array(fp), tn, np.array(fn), np.array(errors)


# for onset evaluation with Precision, Recall, F-measure use the Evaluation
# class and just define the evaluation and error functions
class OnsetEvaluation(Evaluation):
    """
    Evaluation class for measuring Precision, Recall and F-measure of onsets.

    Parameters
    ----------
    detections : str, list or numpy array
        Detected notes.
    annotations : str, list or numpy array
        Annotated ground truth notes.
    window : float, optional
        F-measure evaluation window [seconds]
    combine : float, optional
        Combine all annotated onsets within `combine` seconds.
    delay : float, optional
        Delay the detections `delay` seconds for evaluation.

    """

    def __init__(self, detections, annotations, window=WINDOW, combine=0,
                 delay=0, **kwargs):
        # load the onset detections and annotations
        detections = load_onsets(detections)
        annotations = load_onsets(annotations)
        # combine the annotations if needed
        if combine > 0:
            annotations = combine_events(annotations, combine)
        # shift the detections if needed
        if delay != 0:
            detections += delay
        # evaluate
        tp, fp, tn, fn, errors = onset_evaluation(detections, annotations,
                                                  window)
        # instantiate a Evaluation object
        super(OnsetEvaluation, self).__init__(tp, fp, tn, fn, **kwargs)
        # add the errors
        self.errors = errors

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

        Returns
        -------
        str
            Evaluation metrics formatted as a human readable string.

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
        if not self.eval_objects:
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

        Returns
        -------
        str
            Evaluation metrics formatted as a human readable string.

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

    Parameters
    ----------
    parser : argparse parser instance
        Existing argparse parser object.

    Returns
    -------
    sub_parser : argparse sub-parser instance
        Onset evaluation sub-parser.
    parser_group : argparse argument group
        Onset evaluation argument group.

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
