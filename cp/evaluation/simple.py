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


# helper functions to read/write events from files and combine these events
def load_events(filename):
    """
    Load a list of events from file.

    :param filename: name of the file
    :return: list of events

    """
    # array for events
    events = []
    # try to read in the onsets from the file
    with open(filename, 'rb') as f:
        # read in each line of the file
        for line in f:
            # append the event (1st column) to the list, ignore the rest
            events.append(float(line.split()[0]))
    # return
    return events


def write_events(events, filename):
    """
    Write the detected onsets to the given file.

    :param events: list of events [seconds]
    :param filename: output file name

    """
    with open(filename, 'w') as f:
        for e in events:
            f.write(str(e) + '\n')


def combine_events(events, delta):
    """
    Combine all events within a certain range.

    :param events: list of events [seconds]
    :param delta: combination length [seconds]
    :return: list of combined events

    """
    # sort the events
    events.sort()
    events_length = len(events)
    events_index = 0
    # array for combined events
    comb = []
    # iterate over all events
    while events_index < events_length - 1:
        # get the first event
        first = events[events_index]
        # always increase the events index
        events_index += 1
        # get the second event
        second = events[events_index]
        # combine the two events?
        if second - first <= delta:
            # two events within the combination window, combine them and replace
            # the second event in the original list with the mean of the events
            events[events_index] = (first + second) / 2.
        else:
            # the two events can not be combined,
            # store the first event in the new list
            comb.append(first)
    # always append the last element of the list
    comb.append(events[-1])
    # return the combined onsets
    return comb


def find_closest_match(detections, targets):
    """
    Find the closest matches for detections in targets.

    :param detections: sequence of events to be matched [seconds]
    :param targets: sequence of possible matches [seconds]
    :returns: a list of indices with closest matches

    """
    # solution found at: http://stackoverflow.com/questions/8914491/finding-the-nearest-value-and-return-the-index-of-array-in-python
    if not isinstance(detections, np.ndarray):
        detections = np.array(detections)
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)
    indices = targets.searchsorted(detections)
    indices = np.clip(indices, 1, len(targets) - 1)
    left = targets[indices - 1]
    right = targets[indices]
    indices -= detections - left < right - detections
    return indices


# simple class for evaluation of Presicion, Recall, F-measure if only the
# numbers of true/false positive/(negative) detections are given
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

    @property
    def errors(self):
        """Errors of the true positive detections relative to the targets."""
        # TODO: move this to a helper function? might be useful in other cases
        # calculate the deviation for all true positive detections
        indices = find_closest_match(self.tp, self.targets)
        # calc deviation
        errors = np.asarray(self.targets)[indices] - self.tp
        # return as python list, so that the "if not" test works without errors
        return errors.tolist()

    @property
    def mean_error(self):
        """Mean of the errors."""
        if not self.errors:
            return 0
        return np.mean(self.errors)

    @property
    def std_error(self):
        """Standard deviation of the errors."""
        if not self.errors:
            return 0
        return np.std(self.errors)





## simple class for evaluation of Presicion, Recall, F-measure
#class Evaluation(object):
#    """
#    Simple evaluation class for measuring Precision, Recall and F-measure.
#
#    """
#    def __init__(self, detections, targets, eval_function, **kwargs):
#        """
#        Creates a new Evaluation object instance.
#
#        :param detections: sequence of estimated beat times [seconds]
#        :param targets: sequence of ground truth beat annotations [seconds]
#        :param eval_function: evaluation function (see below)
#
#        The evaluation function can be any function which returns a tuple of
#        lists containing the true positive, false positive and false negative
#        detections: ([true positives], [false positives], [false negatives])
#
#        """
#        self.detections = detections
#        self.targets = targets
#        self.eval_function = eval_function
#        # save additional arguments and pass them to the evaluation function
#        self.__kwargs = kwargs
#        # init some hidden variables as None, calculate them on demand
#        # FIXME: invalidate (i.e. reset them to None) if the detections or
#        # targets change
#        self._tp = None
#        self._fp = None
#        self._fn = None
#
#    @property
#    def num_detections(self):
#        """Number of detections."""
#        return len(self.detections)
#
#    @property
#    def num_targets(self):
#        """Number of targets."""
#        return len(self.targets)
#
#    def _calc_tp_fp_fn(self):
#        """Perform basic evaluation."""
#        self._tp, self._fp, self._fn, = self.eval_function(self.detections, self.targets, **self.__kwargs)
#
#    @property
#    def tp(self):
#        """True positive detections."""
#        print type(self)
#        if not self._tp:
#            self._calc_tp_fp_fn()
#        return self._tp
#
#    @property
#    def num_tp(self):
#        """Number of true positive detections."""
#        return len(self.tp)
#
#    @property
#    def fp(self):
#        """False positive detections."""
#        if not self._fp:
#            self._calc_tp_fp_fn()
#        return self._fp
#
#    @property
#    def num_fp(self):
#        """Number of false positive detections."""
#        return len(self.fp)
#
#    @property
#    def fn(self):
#        """False negative detections."""
#        if not self._fn:
#            self._calc_tp_fp_fn()
#        return self._fn
#
#    @property
#    def num_fn(self):
#        """Number of false negative detections."""
#        return len(self.fn)
#
#    @property
#    def errors(self):
#        """errors of the true positive detections relative to the targets."""
#        # calculate the deviation for all true positive detections
#        return np.asarray(self.targets)[find_closest_match(self.tp, self.targets)] - self.tp
#
#    @property
#    def precision(self):
#        """Precision."""
#        try:
#            return self.num_tp / float(self.num_tp + self.num_fp)
#        except ZeroDivisionError:
#            return 0.
#
#    @property
#    def recall(self):
#        """Recall."""
#        try:
#            return self.num_tp / float(self.num_tp + self.num_fn)
#        except ZeroDivisionError:
#            return 0.
#
#    @property
#    def fmeasure(self):
#        """F-measure."""
#        try:
#            return 2. * self.precision * self.recall / (self.precision + self.recall)
#        except ZeroDivisionError:
#            return 0.
#
#    @property
#    def accuracy(self):
#        """Accuracy."""
#        try:
#            return self.num_tp / float(self.num_fp + self.num_fn + self.num_tp)
#        except ZeroDivisionError:
#            return 0.
#
#    @property
#    def true_positive_rate(self):
#        """True positive rate."""
#        try:
#            return self.num_tp / float(self.num_targets)
#        except ZeroDivisionError:
#            return 0.
#
#    @property
#    def false_positive_rate(self):
#        """False positive rate."""
#        try:
#            return self.num_fp / float(self.num_fp + self.num_tp)
#        except ZeroDivisionError:
#            return 0.
#
#    @property
#    def false_negative_rate(self):
#        """False negative rate."""
#        try:
#            return self.num_fn / float(self.num_fn + self.num_tp)
#        except ZeroDivisionError:
#            return 0.
#
#    def print_errors(self, tex=False):
#        """
#        Print errors.
#
#        :param tex: output format to be used in .tex files [default=False]
#
#        """
#        # print the errors
#        print '  targets: %5d correct: %5d fp: %4d fn: %4d p=%.3f r=%.3f f=%.3f' % (self.num_targets, self.num_tp, self.num_fp, self.num_fn, self.precision, self.recall, self.fmeasure)
#        print '  tp: %.1f%% fp: %.1f%% acc: %.1f%% mean: %.1f ms std: %.1f ms' % (self.true_positive_rate * 100., self.false_positive_rate * 100., self.accuracy * 100., np.mean(self.errors) * 1000., np.std(self.errors) * 1000.)
#        if tex:
#            print "%i events & Precision & Recall & F-measure & True Positves & False Positives & Accuracy & Delay\\\\" % (self.num)
#            print "tex & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f %.1f\$\\pm\$%.1f\\,ms\\\\" % (self.precision, self.recall, self.fmeasure, self.true_positive_rate, self.false_positive_rate, self.accuracy, np.mean(self.dev) * 1000., np.std(self.dev) * 1000.)
#
#    def __str__(self):
#        return "%s p=%.3f r=%.3f f=%.3f" % (self.__class__, self.precision, self.recall, self.fmeasure)
