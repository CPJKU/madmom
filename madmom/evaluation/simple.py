#!/usr/bin/env python
# encoding: utf-8
"""
This file contains basic evaluation functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import numpy as np


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
        self._errors = None  # None indicates not initialised

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
    def errors(self):
        """Errors."""
        if self._errors is None:
            return np.zeros(0)
        return self._errors

    @property
    def mean_error(self):
        """Mean of the errors."""
        if not self.errors.any():
            return 0
        return np.mean(self.errors)

    @property
    def std_error(self):
        """Standard deviation of the errors."""
        if not self.errors.any():
            return 0
        return np.std(self.errors)

    def print_errors(self, tex=False):
        """
        Print errors.

        :param tex: output format to be used in .tex files

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
            print 'tex & Precision & Recall & F-measure & True Positives & '\
                  'False Positives & Accuracy & Mean & Std.dev\\\\'
            print '%i targets & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & '\
                  '%.2f ms & %.2f ms\\\\' % (targets, self.precision,
                  self.recall, self.fmeasure, tpr, fpr, self.accuracy,
                  self.mean_error * 1000., self.std_error * 1000.)


# class for summing Evaluations
class SumEvaluation(SimpleEvaluation):
    """
    Simple evaluation class for summing Precision, Recall and F-measure.

    """
    def __init__(self):
        """
        Creates a new SumEvaluation object instance.

        """
        super(SumEvaluation, self).__init__()
        self._errors = np.zeros(0)

    # for adding two SimpleEvaluation objects
    def __add__(self, other):
        if isinstance(other, SimpleEvaluation):
            # increase the counters
            self._num_tp += other.num_tp
            self._num_fp += other.num_fp
            self._num_tn += other.num_tn
            self._num_fn += other.num_fn
            # extend the errors array
            self._errors = np.append(self.errors, other.errors)
            return self
        else:
            return NotImplemented


# class for averaging Evaluations
class MeanEvaluation(SimpleEvaluation):
    """
    Simple evaluation class for averaging Precision, Recall and F-measure.

    """
    def __init__(self):
        """
        Creates a new MeanEvaluation object instance.

        """
        super(MeanEvaluation, self).__init__()
        # redefine most of the stuff
        self._precision = np.zeros(0)
        self._recall = np.zeros(0)
        self._fmeasure = np.zeros(0)
        self._accuracy = np.zeros(0)
        self._mean = np.zeros(0)
        self._std = np.zeros(0)
        self._errors = np.zeros(0)
        self._num_tp = np.zeros(0)
        self._num_fp = np.zeros(0)
        self._num_tn = np.zeros(0)
        self._num_fn = np.zeros(0)
        self.num = 0

    # for adding another Evaluation object
    def __add__(self, other):
        """
        Appends the scores of another SimpleEvaluation object to the respective
        arrays.

        :param other: SimpleEvaluation object

        """
        if isinstance(other, SimpleEvaluation):
            # append the scores to an array so we can average later
            self._precision = np.append(self._precision, other.precision)
            self._recall = np.append(self._recall, other.recall)
            self._fmeasure = np.append(self._fmeasure, other.fmeasure)
            self._accuracy = np.append(self._accuracy, other.accuracy)
            self._mean = np.append(self._mean, other.mean_error)
            self._std = np.append(self._std, other.std_error)
            self._errors = np.append(self._errors, other.errors)
            # do the same with the raw numbers and errors
            self._num_tp = np.append(self._num_tp, other.num_tp)
            self._num_fp = np.append(self._num_fp, other.num_fp)
            self._num_tn = np.append(self._num_tn, other.num_tn)
            self._num_fn = np.append(self._num_fn, other.num_fn)
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


# class for evaluation of Precision, Recall, F-measure with arrays
class Evaluation(SimpleEvaluation):
    """
    Evaluation class for measuring Precision, Recall and F-measure based on
    numpy arrays with true/false positive/negative detections.

    """

    def __init__(self, tp=np.empty(0), fp=np.empty(0),
                 tn=np.empty(0), fn=np.empty(0)):
        """
        Creates a new Evaluation object instance.

        :param tp: numpy array with true positive detections [seconds]
        :param fp: numpy array with false positive detections [seconds]
        :param tn: numpy array with true negative detections [seconds]
        :param fn: numpy array with false negative detections [seconds]

        """
        super(Evaluation, self).__init__()
        # init some hidden variables as None, calculate them on demand
        self._tp = tp
        self._fp = fp
        self._tn = tn
        self._fn = fn

    @property
    def tp(self):
        return self._tp

    @property
    def num_tp(self):
        """Number of true positive detections."""
        return len(self._tp)

    @property
    def fp(self):
        return self._fp

    @property
    def num_fp(self):
        """Number of false positive detections."""
        return len(self._fp)

    @property
    def tn(self):
        return self._tn

    @property
    def num_tn(self):
        """Number of true negative detections."""
        return len(self._tn)

    @property
    def fn(self):
        return self._fn

    @property
    def num_fn(self):
        """Number of false negative detections."""
        return len(self._fn)
