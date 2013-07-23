#!/usr/bin/env python
# encoding: utf-8
"""
Copyright (c) 2012-2013 Sebastian Böck <sebastian.boeck@jku.at>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

import numpy as np

from helpers import absolute_errors


class SimpleEvaluation(object):
    """
    Simple evaluation class for calculating Precision, Recall and F-measure
    based on the numbers of true/false positive/(negative) detections.

    """
    def __init__(self, num_tp=0, num_fp=0, num_fn=0):
        """
        Creates a new SimpleEvaluation object instance.

        :param num_tp: number of true positive detections
        :param num_fp: number of false positive detections
        :param num_fn: number of false negative detections

        """
        # init some hidden variables as None, calculate them on demand
        # FIXME: invalidate (i.e. reset to None) if detections/targets change
        self.num_tp = num_tp
        self.num_fp = num_fp
        self.num_fn = num_fn

    @property
    def precision(self):
        """Precision."""
        try:
            return self.num_tp / float(self.num_tp + self.num_fp)
        except ZeroDivisionError:
            return 0.

    @property
    def recall(self):
        """Recall."""
        try:
            return self.num_tp / float(self.num_tp + self.num_fn)
        except ZeroDivisionError:
            return 0.

    @property
    def fmeasure(self):
        """F-measure."""
        try:
            return 2. * self.precision * self.recall / (self.precision + self.recall)
        except ZeroDivisionError:
            return 0.

    @property
    def accuracy(self):
        """Accuracy."""
        try:
            return self.num_tp / float(self.num_fp + self.num_fn + self.num_tp)
        except ZeroDivisionError:
            return 0.

    @property
    def mean_error(self):
        """Mean of the errors."""
        return 0

    @property
    def std_error(self):
        """Standard deviation of the errors."""
        return 0

    def print_errors(self, tex=False):
        """
        Print errors.

        :param tex: output format to be used in .tex files [default=False]

        """
        # print the errors
        print '  targets: %5d correct: %5d fp: %4d fn: %4d p=%.3f r=%.3f f=%.3f' % (self.num_tp + self.num_fn, self.num_tp, self.num_fp, self.num_fn, self.precision, self.recall, self.fmeasure)
        print '  tpr: %.1f%% fpr: %.1f%% acc: %.1f%% mean: %.1f ms std: %.1f ms' % (self.recall * 100., (1 - self.precision) * 100., self.accuracy * 100., self.mean_error * 1000., self.std_error * 1000.)
        if tex:
            print "%i events & Precision & Recall & F-measure & True Positves & False Positives & Accuracy & Delay\\\\" % (self.num)
            print "tex & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f %.1f\$\\pm\$%.1f\\,ms\\\\" % (self.precision, self.recall, self.fmeasure, self.true_positive_rate, self.false_positive_rate, self.accuracy, self.mean_error * 1000., self.std_error * 1000.)

    def __str__(self):
        return "%s p=%.3f r=%.3f f=%.3f" % (self.__class__, self.precision, self.recall, self.fmeasure)


class SumEvaluation(SimpleEvaluation):
    """
    Simple evaluation class for summing true/false positive/(negative)
    detections and calculate Precision, Recall and F-measure.

    """
    # inherit from Evaluation class, since this is basically the same
    # this class just sums all the attributes and evaluates accordingly
    def __init__(self, other=None):
        self.num_tp = 0
        self.num_fp = 0
        self.num_fn = 0
        self.__errors = []
        # instance can be initialized with an Evaluation object
        if isinstance(other, Evaluation):
            # add this object to self
            self += other

    # for adding an Evaluation object
    def __add__(self, other):
        #if isinstance(other, Evaluation):
        if issubclass(other.__class__, Evaluation):
            # extend
            self.num_tp += other.num_tp
            self.num_fp += other.num_fp
            self.num_fn += other.num_fn
            self.__errors.extend(other.errors)
            return self
        else:
            return NotImplemented

    @property
    def mean_error(self):
        """Mean of the errors."""
        return np.mean(self.__errors)

    @property
    def std_error(self):
        """Standard deviation of the errors."""
        return np.std(self.__errors)


class MeanEvaluation(SimpleEvaluation):
    """
    Simple class for averaging Precision, Recall and F-measure.

    """
    def __init__(self, other=None):
        """
        Creates a new MeanEvaluation object instance.

        """
        self.__precision = []
        self.__recall = []
        self.__fmeasure = []
        self.__accuracy = []
        self.__mean = []
        self.__std = []
        self.num = 0
        self.num_tp = 0
        self.num_fp = 0
        self.num_fn = 0
        # instance can be initialized with a Evaluation object
        if isinstance(other, Evaluation):
            # add this object to self
            self += other

    # for adding a OnsetEvaluation object
    def __add__(self, other):
        if isinstance(other, Evaluation):
            self.__precision.append(other.precision)
            self.__recall.append(other.recall)
            self.__fmeasure.append(other.fmeasure)
            self.__accuracy.append(other.accuracy)
            self.__mean.append(other.mean_error)
            self.__std.append(other.std_error)
            self.num += 1
            self.num_tp += other.num_tp
            self.num_fp += other.num_fp
            self.num_fn += other.num_fn
            return self
        else:
            return NotImplemented

    @property
    def precision(self):
        """Precision."""
        return np.mean(self.__precision)

    @property
    def recall(self):
        """Recall."""
        return np.mean(self.__recall)

    @property
    def fmeasure(self):
        """F-measure."""
        return np.mean(self.__fmeasure)

    @property
    def accuracy(self):
        """Accuracy."""
        return np.mean(self.__accuracy)

    @property
    def mean_error(self):
        """Mean of the errors."""
        return np.mean(self.__mean)

    @property
    def std_error(self):
        """Standard deviation of the errors."""
        return np.std(self.__std)


# simple class for evaluation of Presicion, Recall, F-measure
class Evaluation(SimpleEvaluation):
    """
    Simple evaluation class for measuring Precision, Recall and F-measure.

    """
    def __init__(self, detections, targets, eval_function, **kwargs):
        """
        Creates a new Evaluation object instance.

        :param detections: sequence of estimated beat times [seconds]
        :param targets: sequence of ground truth beat annotations [seconds]
        :param eval_function: evaluation function (see below)

        The evaluation function can be any function which returns a tuple of
        lists containing the true positive, false positive and false negative
        detections: ([true positives], [false positives], [false negatives])

        """
        self.detections = detections
        self.targets = targets
        self.eval_function = eval_function
        # save additional arguments and pass them to the evaluation function
        self.__kwargs = kwargs
        # init some hidden variables as None, calculate them on demand
        # FIXME: invalidate (i.e. reset to None) if detections/targets change
        # TODO: rename _ to __ again?
        self._tp = None
        self._fp = None
        self._fn = None

    def _calc_tp_fp_fn(self):
        """Perform basic evaluation."""
        self._tp, self._fp, self._fn, = self.eval_function(self.detections, self.targets, **self.__kwargs)

    @property
    def tp(self):
        """True positive detections."""
        if not self._tp:
            self._calc_tp_fp_fn()
        return self._tp

    @property
    def num_tp(self):
        """Number of true positive detections."""
        return len(self.tp)

    @property
    def fp(self):
        """False positive detections."""
        if not self._fp:
            self._calc_tp_fp_fn()
        return self._fp

    @property
    def num_fp(self):
        """Number of false positive detections."""
        return len(self.fp)

    @property
    def fn(self):
        """False negative detections."""
        if not self._fn:
            self._calc_tp_fp_fn()
        return self._fn

    @property
    def num_fn(self):
        """Number of false negative detections."""
        return len(self.fn)

    def absolute_errors(self):
        """
        Absolute errors of all true positive detections relative to the closest
        targets.

        """
        return absolute_errors(self.detections, self.targets)

    @property
    def mean_error(self):
        """
        Mean of the absolute errors of all true positive detections relative to
        the closest targets.

        """
        if not self.errors:
            return 0
        return np.mean(self.errors)

    @property
    def std_error(self):
        """
        Standard deviation of the absolute errors of all true positive
        detections relative to the clostest targets.

        """
        if not self.errors:
            return 0
        return np.std(self.errors)
