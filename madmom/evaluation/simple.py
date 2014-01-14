#!/usr/bin/env python
# encoding: utf-8
"""
This file contains basic evaluation functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import numpy as np

from .helpers import calc_errors


def calc_overlap(detections, targets, threshold=0.5):
    """
    Very simple overlap calculation based on two numpy array of the same shape.
    The arrays should be a quantized version of any event lists.

    :param detections: array with detections
    :param targets:    array with targets
    :param threshold:  binary decision threshold [default=0.5]

    """
    # detections and targets must have the same dimensions
    if detections.size != targets.size:
        raise ValueError("dimension mismatch")
    if detections.ndim > 1:
        # TODO: implement for multi-dimensional arrays
        raise NotImplementedError("please add multi-dimensional functionality")
    # threshold detections
    detections = (detections >= threshold)
    # threshold targets
    targets = targets >= threshold
    # calculate overlap
    tp = np.nonzero(detections * targets)[0]
    fp = np.nonzero(detections > targets)[0]
    tn = np.nonzero(-detections * -targets)[0]
    fn = np.nonzero(detections < targets)[0]
    if tp.size + tn.size + fp.size + fn.size != detections.size:
        raise AssertionError('bad overlap calculation')
    # return
    return tp, tn, fp, fn


class SimpleEvaluation(object):
    """
    Simple evaluation class for calculating Precision, Recall and F-measure
    based on the numbers of true/false positive/negative detections.

    Note: so far, this class is only suitable for a 1-class evaluation problem.

    """
    def __init__(self, num_tp=0, num_fp=0, num_tn=0, num_fn=0):
        """
        Creates a new SimpleEvaluation object instance.

        :param num_tp: number of true positive detections
        :param num_fp: number of false positive detections
        :param num_tn: number of true negative detections
        :param num_fn: number of false negative detections

        """
        # hidden variables, to be able to overwrite them in subclasses
        self._num_tp = num_tp
        self._num_fp = num_fp
        self._num_tn = num_tn
        self._num_fn = num_fn

    @property
    def num_tp(self):
        """Number of true positive detections."""
        return self._num_tp

    @property
    def num_fp(self):
        """Number of false positive detections."""
        return self._num_fp

    @property
    def num_tn(self):
        """Number of true negative detections."""
        return self._num_tn

    @property
    def num_fn(self):
        """Number of false negative detections."""
        return self._num_fn

    @property
    def precision(self):
        """Precision."""
        # correct / retrieved
        retrieved = float(self.num_tp + self.num_fp)
        # if there are no positive predictions, none of them are wrong
        if retrieved == 0:
            return 1.
        return self.num_tp / retrieved

    @property
    def recall(self):
        """Recall."""
        # correct / relevant
        relevant = float(self.num_tp + self.num_fn)
        # if there are no positive targets, we recalled all of them
        if relevant == 0:
            return 1.
        return self.num_tp / relevant

    @property
    def fmeasure(self):
        """F-measure."""
        numerator = 2. * self.precision * self.recall
        if numerator == 0:
            return 0.
        return numerator / (self.precision + self.recall)

    @property
    def accuracy(self):
        """Accuracy."""
        # acc: (TP + TN) / (TP + FP + TN + FN)
        numerator = float(self.num_tp + self.num_tn)
        if numerator == 0:
            return 0.
        return numerator / (self.num_fp + self.num_fn + self.num_tp +
                            self.num_tn)

    @property
    def mean_error(self):
        """Mean of the errors."""
        # FIXME: is returning 0 ok?
        return 0.

    @property
    def std_error(self):
        """Standard deviation of the errors."""
        # FIXME: is returning 0 ok?
        return 0.

    @property
    def errors(self):
        """Errors."""
        # FIXME: is returning an empty list ok?
        return np.empty(0)

    def print_errors(self, tex=False):
        """
        Print errors.

        :param tex: output format to be used in .tex files [default=False]

        """
        # print the errors
        targets = self.num_tp + self.num_fn
        tpr = self.recall
        fpr = (1 - self.precision)
        print '  targets: %5d correct: %5d fp: %4d fn: %4d p=%.3f r=%.3f '\
              'f=%.3f' % (targets, self.num_tp, self.num_fp, self.num_fn,
                          self.precision, self.recall, self.fmeasure)
        print '  tpr: %.1f%% fpr: %.1f%% acc: %.1f%% mean: %.1f ms std: '\
              '%.1f ms' % (tpr * 100., fpr * 100., self.accuracy * 100.,
                           self.mean_error * 1000., self.std_error * 1000.)
        if tex:
            print '& Precision & Recall & F-measure & True Positives & '\
                  'False Positives & Accuracy & Delay\\\\'
            print 'tex & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f %.1f\$\\pm\$'\
                  '%.1f\\,ms\\\\' % (self.precision, self.recall,
                                     self.fmeasure, tpr, fpr, self.accuracy,
                                     self.mean_error * 1000.,
                                     self.std_error * 1000.)


class SumEvaluation(SimpleEvaluation):
    """
    Simple evaluation class for summing true/false positive/(negative)
    detections and calculate Precision, Recall and F-measure.

    """
    # inherit from Evaluation class, since this is basically the same
    # this class just sums all the attributes and evaluates accordingly
    def __init__(self, other=None):
        super(SumEvaluation, self).__init__()
        self._num_tp = 0
        self._num_fp = 0
        self._num_tn = 0
        self._num_fn = 0
        self._errors = np.empty(0)
        # instance can be initialized with a Evaluation object
        if other:
            # add this object to self
            self += other

    # for adding an Evaluation object
    def __add__(self, other):
        #if isinstance(other, Evaluation):
        if issubclass(other.__class__, SimpleEvaluation):
            # extend
            self._num_tp += other.num_tp
            self._num_fp += other.num_fp
            self._num_tn += other.num_tn
            self._num_fn += other.num_fn
            self._errors = np.append(self._errors, other.errors)
            return self
        else:
            return NotImplemented

    @property
    def num_tp(self):
        """Number of true positive detections."""
        return self._num_tp

    @property
    def num_fp(self):
        """Number of false positive detections."""
        return self._num_fp

    @property
    def num_tn(self):
        """Number of true negative detections."""
        return self._num_tn

    @property
    def num_fn(self):
        """Number of false negative detections."""
        return self._num_fn

    @property
    def mean_error(self):
        """Mean of the errors."""
        if not self._errors.any():
            return 0
        return np.mean(self._errors)

    @property
    def std_error(self):
        """Standard deviation of the errors."""
        if not self._errors.any():
            return 0
        return np.std(self._errors)


class MeanEvaluation(SimpleEvaluation):
    """
    Simple evaluation class for averaging Precision, Recall and F-measure.

    """
    def __init__(self, other=None):
        """
        Creates a new MeanEvaluation object instance.

        """
        super(MeanEvaluation, self).__init__()
        # redefine most of the stuff
        self._precision = np.empty(0)
        self._recall = np.empty(0)
        self._fmeasure = np.empty(0)
        self._accuracy = np.empty(0)
        self._mean = np.empty(0)
        self._std = np.empty(0)
        self._errors = np.empty(0)
        self._num_tp = np.empty(0)
        self._num_fp = np.empty(0)
        self._num_tn = np.empty(0)
        self._num_fn = np.empty(0)
        self.num = 0
        # instance can be initialized with a Evaluation object
        if other:
            # add this object to self
            self += other

    # for adding a OnsetEvaluation object
    def __add__(self, other):
        """
        Appends the scores of another SimpleEvaluation object to the respective
        arrays.

        :param other: SimpleEvaluation object

        """
        if issubclass(other.__class__, SimpleEvaluation):
            self._precision = np.append(self._precision, other.precision)
            self._recall = np.append(self._recall, other.recall)
            self._fmeasure = np.append(self._fmeasure, other.fmeasure)
            self._accuracy = np.append(self._accuracy, other.accuracy)
            self._mean = np.append(self._mean, other.mean_error)
            self._std = np.append(self._std, other.std_error)
            self._errors = np.append(self._errors, other.errors)
            self._num_tp = np.append(self._num_tp, other.num_tp)
            self._num_fp = np.append(self._num_fp, other.num_fp)
            self._num_tn = np.append(self._num_tn, other.num_tn)
            self._num_fn = np.append(self._num_fn, other.num_fn)
            self.num += 1
            return self
        else:
            return NotImplemented

    @property
    def num_tp(self):
        """Number of true positive detections."""
        if self._num_tp.size == 0:
            return 0
        return np.mean(self._num_tp)

    @property
    def num_fp(self):
        """Number of false positive detections."""
        if self._num_fp.size == 0:
            return 0
        return np.mean(self._num_fp)

    @property
    def num_tn(self):
        """Number of true negative detections."""
        if self._num_tn.size == 0:
            return 0
        return np.mean(self._num_tn)

    @property
    def num_fn(self):
        """Number of false negative detections."""
        if self._num_fn.size == 0:
            return 0
        return np.mean(self._num_fn)

    @property
    def precision(self):
        """Precision."""
        if self._precision.size == 0:
            return 0
        return np.mean(self._precision)

    @property
    def recall(self):
        """Recall."""
        if self._recall.size == 0:
            return 0
        return np.mean(self._recall)

    @property
    def fmeasure(self):
        """F-measure."""
        if self._fmeasure.size == 0:
            return 0
        return np.mean(self._fmeasure)

    @property
    def accuracy(self):
        """Accuracy."""
        if self._accuracy.size == 0:
            return 0
        return np.mean(self._accuracy)

    @property
    def errors(self):
        """Errors."""
        if self._errors.size == 0:
            return 0
        return self._errors

    @property
    def mean_error(self):
        """Mean of the errors."""
        if self._mean.size == 0:
            return 0
        return np.mean(self._mean)

    @property
    def std_error(self):
        """Standard deviation of the errors."""
        if self._std.size == 0:
            return 0
        return np.mean(self._std)


# simple class for evaluation of Precision, Recall, F-measure
class Evaluation(SimpleEvaluation):
    """
    Evaluation class for measuring Precision, Recall and F-measure.

    """

    def __init__(self, detections, targets, eval_function, **kwargs):
        """
        Creates a new Evaluation object instance.

        :param detections:    sequence of estimated detections [seconds]
        :param targets:       sequence of ground truth targets [seconds]
        :param eval_function: evaluation function (see below)

        The evaluation function can be any function which returns a tuple of 4
        numpy arrays with the true/false positive and negative detections:
        ([true positive], [false positive], [true negative], [false negative])

        Note: All arrays can be multi-dimensional with events aligned on the
              first axis. Information in other columns/axes is not used.

        """
        # detections, targets and evaluation function
        self._detections = detections
        self._targets = targets
        self._eval_function = eval_function
        # save additional arguments and pass them to the evaluation function
        self._kwargs = kwargs
        # init some hidden variables as None, calculate them on demand
        self._tp = None
        self._fp = None
        self._tn = None
        self._fn = None
        self._errors = None

    def _calc_tp_fp_tn_fn(self):
        """Perform basic evaluation."""
        numbers = self._eval_function(self._detections, self._targets,
                                      **self._kwargs)
        self._tp, self._fp, self._tn, self._fn = numbers

    @property
    def tp(self):
        """True positive detections."""
        if self._tp is None:
            self._calc_tp_fp_tn_fn()
        return self._tp

    @property
    def num_tp(self):
        """Number of true positive detections."""
        return self.tp.shape[0]

    @property
    def fp(self):
        """False positive detections."""
        if self._fp is None:
            self._calc_tp_fp_tn_fn()
        return self._fp

    @property
    def num_fp(self):
        """Number of false positive detections."""
        return self.fp.shape[0]

    @property
    def tn(self):
        """True negative detections."""
        if self._tn is None:
            self._calc_tp_fp_tn_fn()
        return self._tn

    @property
    def num_tn(self):
        """Number of true negative detections."""
        return self.tn.shape[0]

    @property
    def fn(self):
        """False negative detections."""
        if self._fn is None:
            self._calc_tp_fp_tn_fn()
        return self._fn

    @property
    def num_fn(self):
        """Number of false negative detections."""
        return self.fn.shape[0]

    @property
    def errors(self):
        """
        Absolute errors of all true positive detections relative to the closest
        targets.

        """
        if self._errors is None:
            if self.num_tp == 0:
                # FIXME: what is the error in case of no TPs
                self._errors = np.empty(0)
            else:
                self._errors = calc_errors(self.tp, self._targets)
        return self._errors

    @property
    def mean_error(self):
        """
        Mean of the absolute errors of all true positive detections relative to
        the closest targets.

        """
        if not self.errors.any():
            return 0
        return np.mean(self.errors)

    @property
    def std_error(self):
        """
        Standard deviation of the absolute errors of all true positive
        detections relative to the closest targets.

        """
        if not self.errors.any():
            return 0
        return np.std(self.errors)


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
    p.add_argument('-d', dest='det_ext', action='store', default=None,
                   help='extension of the detection files')
    p.add_argument('-t', dest='tar_ext', action='store', default=None,
                   help='extension of the target files')
    # parameters for evaluation
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
    Simple evaluation.

    """
    from ..utils.helpers import files, match_file, load_events

    # parse arguments
    args = parser()

    # get detection and target files
    det_files = files(args.files, args.det_ext)
    tar_files = files(args.files, args.tar_ext)
    # quit if no files are found
    if len(det_files) == 0:
        print "no files to evaluate. exiting."
        exit()

    # sum and mean counter for all files
    sum_counter = SumEvaluation()
    mean_counter = MeanEvaluation()
    # evaluate all files
    for det_file in det_files:
        # get the detections file
        detections = load_events(det_file)
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
            # test with onsets (but use the beat detection window of 70ms)
            from .onsets import count_errors
            # add the Evaluation to mean evaluation
            me += Evaluation(detections, targets, count_errors, window=0.07)
            # process the next target file
        # print stats for each file
        if args.verbose:
            me.print_errors(args.tex)
        # add the resulting sum counter
        sum_counter += me
        mean_counter += me
        # process the next detection file
    # print summary
    print 'sum for %i files:' % (len(det_files))
    sum_counter.print_errors(args.tex)
    print 'mean for %i files:' % (len(det_files))
    mean_counter.print_errors(args.tex)

if __name__ == '__main__':
    main()
