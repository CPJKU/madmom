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

from helpers import calc_errors


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
    detections = detections >= threshold
    # threshold targets
    targets = targets >= threshold
    # calculate overlap
    tp = np.nonzero(detections * targets)[0]
    fp = np.nonzero(detections > targets)[0]
    tn = np.nonzero(-detections * -targets)[0]
    fn = np.nonzero(detections < targets)[0]
    assert tp.size + tn.size + fp.size + fn.size == detections.size, 'bad overlap calculation'
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
        # use hidden variables, because the properties get overridden in subclasses
        self.__num_tp = num_tp
        self.__num_fp = num_fp
        self.__num_tn = num_tn
        self.__num_fn = num_fn

    @property
    def num_tp(self):
        """Number of true positive detections."""
        return self.__num_tp

    @property
    def num_fp(self):
        """Number of false positive detections."""
        return self.__num_fp

    @property
    def num_tn(self):
        """Number of true negative detections."""
        return self.__num_tn

    @property
    def num_fn(self):
        """Number of false negative detections."""
        return self.__num_fn

    @property
    def precision(self):
        """Precision."""
        # correct / retrieved
        if self.num_tp == 0:
            # FIXME: why is this hack still needed? If not, we get a
            # RuntimeWarning: invalid value encountered in double_scalars
            return 0
        return self.num_tp / np.float64(self.num_tp + self.num_fp)

    @property
    def recall(self):
        """Recall."""
        # correct / relevant
        if self.num_tp == 0:
            # FIXME: why is this hack still needed? If not, we get a
            # RuntimeWarning: invalid value encountered in double_scalars
            return 0
        return self.num_tp / np.float64(self.num_tp + self.num_fn)

    @property
    def fmeasure(self):
        """F-measure."""
        numerator = 2 * self.precision * self.recall
        if numerator == 0:
            # FIXME: why is this hack still needed? If not, we get a
            # RuntimeWarning: invalid value encountered in double_scalars
            return 0
        return numerator / np.float64(self.precision + self.recall)

    @property
    def accuracy(self):
        """Accuracy."""
        # acc: (TP + TN) / (TP + FP + TN + FN)
        numerator = self.num_tp + self.num_tn
        if numerator == 0:
            # FIXME: why is this hack still needed? If not, we get a
            # RuntimeWarning: invalid value encountered in double_scalars
            return 0
        return numerator / np.float64(self.num_fp + self.num_fn + self.num_tp + self.num_tn)

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
        # FIXME: is returning an empty list?
        return np.empty(0)

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
        super(SumEvaluation, self).__init__()
        self.__num_tp = 0
        self.__num_fp = 0
        self.__num_tn = 0
        self.__num_fn = 0
        self.__errors = np.empty(0)
        # instance can be initialized with a Evaluation object
        if other:
            # add this object to self
            self += other

    # for adding an Evaluation object
    def __add__(self, other):
        #if isinstance(other, Evaluation):
        if issubclass(other.__class__, SimpleEvaluation):
            # extend
            self.__num_tp += other.num_tp
            self.__num_fp += other.num_fp
            self.__num_tn += other.num_tn
            self.__num_fn += other.num_fn
            self.__errors = np.append(self.__errors, other.errors)
            return self
        else:
            return NotImplemented

    @property
    def num_tp(self):
        """Number of true positive detections."""
        return self.__num_tp

    @property
    def num_fp(self):
        """Number of false positive detections."""
        return self.__num_fp

    @property
    def num_tn(self):
        """Number of true negative detections."""
        return self.__num_tn

    @property
    def num_fn(self):
        """Number of false negative detections."""
        return self.__num_fn

    @property
    def mean_error(self):
        """Mean of the errors."""
        if not self.__errors.any():
            return 0
        return np.mean(self.__errors)

    @property
    def std_error(self):
        """Standard deviation of the errors."""
        if not self.__errors.any():
            return 0
        return np.std(self.__errors)


class MeanEvaluation(SimpleEvaluation):
    """
    Simple class for averaging Precision, Recall and F-measure.

    """
    def __init__(self, other=None):
        """
        Creates a new MeanEvaluation object instance.

        """
        super(MeanEvaluation, self).__init__()
        # redefine most of the stuff
        self.__precision = np.empty(0)
        self.__recall = np.empty(0)
        self.__fmeasure = np.empty(0)
        self.__accuracy = np.empty(0)
        self.__mean = np.empty(0)
        self.__std = np.empty(0)
        self.__errors = np.empty(0)
        self.__num_tp = np.empty(0)
        self.__num_fp = np.empty(0)
        self.__num_tn = np.empty(0)
        self.__num_fn = np.empty(0)
        self.num = 0
        # instance can be initialized with a Evaluation object
        if other:
            # add this object to self
            self += other

    # for adding a OnsetEvaluation object
    def __add__(self, other):
        if issubclass(other.__class__, SimpleEvaluation):
            self.__precision = np.append(self.__precision, other.precision)
            self.__recall = np.append(self.__recall, other.recall)
            self.__fmeasure = np.append(self.__fmeasure, other.fmeasure)
            self.__accuracy = np.append(self.__accuracy, other.accuracy)
            self.__mean = np.append(self.__mean, other.mean_error)
            self.__std = np.append(self.__std, other.std_error)
            self.__errors = np.append(self.__errors, other.errors)
            self.__num_tp = np.append(self.__num_tp, other.num_tp)
            self.__num_fp = np.append(self.__num_fp, other.num_fp)
            self.__num_tn = np.append(self.__num_tn, other.num_tn)
            self.__num_fn = np.append(self.__num_fn, other.num_fn)
            self.num += 1
            return self
        else:
            return NotImplemented

    @property
    def num_tp(self):
        """Number of true positive detections."""
        return np.mean(self.__num_tp)

    @property
    def num_fp(self):
        """Number of false positive detections."""
        return np.mean(self.__num_fp)

    @property
    def num_tn(self):
        """Number of true negative detections."""
        return np.mean(self.__num_tn)

    @property
    def num_fn(self):
        """Number of false negative detections."""
        return np.mean(self.__num_fn)

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
    def errors(self):
        """Errors."""
        return self.__errors

    @property
    def mean_error(self):
        """Mean of the errors."""
        return np.mean(self.__mean)

    @property
    def std_error(self):
        """Standard deviation of the errors."""
        return np.mean(self.__std)


# simple class for evaluation of Presicion, Recall, F-measure
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
        numpy arrays containing the true/false positive and negative detections:
        ([true positive], [false positive], [true negative], [false negative])

        Note: All arrays can be multi-dimensional with events aligned on axis 0.
              Additional information in other columns/axes is not used.

        """
        # detections, targets and evaluation function
        self.__detections = detections
        self.__targets = targets
        self.__eval_function = eval_function
        # save additional arguments and pass them to the evaluation function
        self.__kwargs = kwargs
        # init some hidden variables as None, calculate them on demand
        self.__tp = None
        self.__fp = None
        self.__tn = None
        self.__fn = None
        self.__errors = None

    def _calc_tp_fp_tn_fn(self):
        """Perform basic evaluation."""
        self.__tp, self.__fp, self.__tn, self.__fn = self.__eval_function(self.__detections, self.__targets, **self.__kwargs)

    @property
    def tp(self):
        """True positive detections."""
        if self.__tp is None:
            self._calc_tp_fp_tn_fn()
        return self.__tp

    @property
    def num_tp(self):
        """Number of true positive detections."""
        return self.tp.shape[0]

    @property
    def fp(self):
        """False positive detections."""
        if self.__fp is None:
            self._calc_tp_fp_tn_fn()
        return self.__fp

    @property
    def num_fp(self):
        """Number of false positive detections."""
        return self.fp.shape[0]

    @property
    def tn(self):
        """True negative detections."""
        if self.__tn is None:
            self._calc_tp_fp_tn_fn()
        return self.__tn

    @property
    def num_tn(self):
        """Number of true negative detections."""
        return self.tn.shape[0]

    @property
    def fn(self):
        """False negative detections."""
        if self.__fn is None:
            self._calc_tp_fp_tn_fn()
        return self.__fn

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
        if self.__errors is None:
            if self.num_tp == 0:
                # FIXME: what is the error in case of no TPs
                self.__errors = np.empty(0)
            else:
                self.__errors = calc_errors(self.tp, self.__targets)
        return self.__errors

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
        detections relative to the clostest targets.

        """
        if not self.errors.any():
            return 0
        return np.std(self.errors)


def parser():
    import argparse
    # define parser
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    The script evaluates a file or folder with detections against a file or
    folder with targets. Extensions can be given to filter the detection and
    target file lists.

    """)
    # files used for evaluation
    p.add_argument('detections', help='file (or folder) with detections to be evaluated (files being filtered according to the -d argument)')
    p.add_argument('targets', nargs='*', help='file (or folder) with targets (files being filtered according to the -t argument)')
    # extensions used for evaluation
    p.add_argument('-d', dest='det_ext', action='store', default=None, help='extension of the detection files')
    p.add_argument('-t', dest='tar_ext', action='store', default=None, help='extension of the target files')
    # parameters for evaluation
    p.add_argument('--tex', action='store_true', help='format errors for use is .tex files')
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
    from helpers import match_files, load_events

    # parse arguments
    args = parser()
    # match detections to targets
    files = match_files(args.detections, args.targets, args.det_ext, args.tar_ext)

    # exit if no files were given
    if len(files) == 0:
        print 'no matching pairs found'
        exit(1)

    # sum and mean counter for all files
    sum_counter = SumEvaluation()
    mean_counter = MeanEvaluation()
    # evaluate all files
    for det_file, tar_file in files:
        if not tar_file:
            print 'no target file for %s found' % det_file
            exit(1)
        # get the detections file
        detections = load_events(det_file)
        # process all corresponding target files
        # if more than 1 files are found, do a mean evaluation over all of them
        me = MeanEvaluation()
        for f in tar_file:
            targets = load_events(f)
            # test with onsets (but use the beat detection window of 70ms)
            from onsets import count_errors
            e = Evaluation(detections, targets, count_errors, window=0.07)
#            # evaluate the detections
#            e = Evaluation(detections, targets, calc_overlap)
            # add to mean evaluation
            me += e
            # process the next target file
        # print stats for each file
        if args.verbose:
            if args.verbose >= 2:
                print det_file, tar_file
            else:
                print det_file
            me.print_errors(args.tex)
        # add the resulting sum counter
        sum_counter += me
        mean_counter += me
        # process the next detection file
    # print summary
    print 'sum for %i files:' % (len(files))
    sum_counter.print_errors(args.tex)
    print 'mean for %i files:' % (len(files))
    mean_counter.print_errors(args.tex)

if __name__ == '__main__':
    main()
