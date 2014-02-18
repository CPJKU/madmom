# encoding: utf-8
"""
Evaluation package.

All evaluation methods of this package can be used as scripts directly, if the
package is in $PYTHONPATH.

Example:

python -m madmom.evaluation.onsets /dir/to/be/evaluated

"""
import numpy as np


# evaluation helper functions
def find_closest_matches(detections, targets):
    """
    Find the closest matches for detections in targets.

    :param detections: sequence of events to be matched [seconds]
    :param targets:    sequence of possible matches [seconds]
    :returns:          a numpy array of indices with closest matches

    Note: the sequences must be ordered!

    """
    # if no targets are given
    if len(targets) == 0:
        # return a empty array
        return np.zeros(0, dtype=np.int)
        # FIXME: raise an error instead?
        #raise ValueError("at least one target must be given")
    # if only a single target is given
    if len(targets) == 1:
        # return an array as long as the detections with indices 0
        return np.zeros(len(detections), dtype=np.int)
    # solution found at: http://stackoverflow.com/questions/8914491/
    indices = targets.searchsorted(detections)
    indices = np.clip(indices, 1, len(targets) - 1)
    left = targets[indices - 1]
    right = targets[indices]
    indices -= detections - left < right - detections
    # return the indices of the closest matches
    return indices


def calc_errors(detections, targets, matches=None):
    """
    Calculates the errors of the detections relative to the closest targets.

    :param detections: sequence of events to be matched [seconds]
    :param targets:    sequence of possible matches [seconds]
    :param matches:    indices of the closest matches
    :returns:          a list of errors to closest matches [seconds]

    Note: the sequences must be ordered! To speed up the calculation, a list
          of pre-computed indices of the closest matches can be used.

    """
    # determine the closest targets
    if matches is None:
        matches = find_closest_matches(detections, targets)
    # calc error relative to those targets
    errors = detections - targets[matches]
    # return the errors
    return errors


def calc_absolute_errors(detections, targets, matches=None):
    """
    Calculate absolute errors of the detections relative to the closest
    targets.

    :param detections: sequence of events to be matched [seconds]
    :param targets:    sequence of possible matches [seconds]
    :param matches:    indices of the closest matches
    :returns:          a list of errors to closest matches [seconds]

    Note: the sequences must be ordered! To speed up the calculation, a list
          of pre-computed indices of the closest matches can be used.

    """
    # return the errors
    return np.abs(calc_errors(detections, targets, matches))


def calc_relative_errors(detections, targets, matches=None):
    """
    Relative errors of the detections to the closest targets.
    The absolute error is weighted by the absolute value of the target.

    :param detections: sequence of events to be matched [seconds]
    :param targets:    sequence of possible matches [seconds]
    :param matches:    indices of the closest matches
    :returns:          a list of relative errors to closest matches [seconds]

    Note: the sequences must be ordered! To speed up the calculation, a list of
          pre-computed indices of the closest matches can be used.

    """
    # determine the closest targets
    if matches is None:
        matches = find_closest_matches(detections, targets)
    # calculate the absolute errors
    errors = calc_errors(detections, targets, matches)
    # return the relative errors
    return np.abs(1 - (errors / targets[matches]))


# evaluation classes
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
        self._num_tp = int(num_tp)
        self._num_fp = int(num_fp)
        self._num_tn = int(num_tn)
        self._num_fn = int(num_fn)
        # define the errors as an (empty) array here
        # subclasses are required to redefine as needed
        self._errors = np.zeros(0, dtype=np.float)

    # for adding another SimpleEvaluation object, i.e. summing them
    def __iadd__(self, other):
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
            raise TypeError("Can't add %s to SimpleEvaluation." % type(other))

    # for adding two SimpleEvaluation objects
    def __add__(self, other):
        if isinstance(other, SimpleEvaluation):
            num_tp = self._num_tp + other.num_tp
            num_fp = self._num_fp + other.num_fp
            num_tn = self._num_tn + other.num_tn
            num_fn = self._num_fn + other.num_fn
            # create a new object we can return
            new = SimpleEvaluation(num_tp, num_fp, num_tn, num_fn)
            # modify the hidden variable directly
            # (needed for correct inheritance)
            new._errors = np.append(self.errors, other.errors)
            # return the newly created object
            return new
        else:
            raise TypeError("Can't add %s to SimpleEvaluation." % type(other))

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
        # 2pr / (p+r)
        numerator = 2. * self.precision * self.recall
        if numerator == 0:
            return 0.
        return numerator / (self.precision + self.recall)

    @property
    def accuracy(self):
        """Accuracy."""
        # acc: (TP + TN) / (TP + FP + TN + FN)
        denominator = self.num_fp + self.num_fn + self.num_tp + self.num_tn
        if denominator == 0:
            return 1.
        numerator = float(self.num_tp + self.num_tn)
        if numerator == 0:
            return 0.
        return numerator / denominator

    @property
    def errors(self):
        """Errors."""
        # if any errors are given, they have to be the same length as the true
        # positive detections
        if self._errors.any() and len(self._errors) != self._num_tp:
            raise AssertionError("length of the errors and number of true "
                                 "positive detections must match")
        return self._errors

    @property
    def mean_error(self):
        """Mean of the errors."""
        if not self.errors.any():
            return 0.
        return np.mean(self.errors)

    @property
    def std_error(self):
        """Standard deviation of the errors."""
        if not self.errors.any():
            return 0.
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
                                             self.recall, self.fmeasure,
                                             tpr, fpr, self.accuracy,
                                             self.mean_error * 1000.,
                                             self.std_error * 1000.)


# class for summing Evaluations
SumEvaluation = SimpleEvaluation


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

    # for adding another Evaluation object
    def append(self, other):
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
            # do the same with the raw numbers
            self._num_tp = np.append(self._num_tp, other.num_tp)
            self._num_fp = np.append(self._num_fp, other.num_fp)
            self._num_tn = np.append(self._num_tn, other.num_tn)
            self._num_fn = np.append(self._num_fn, other.num_fn)
        else:
            raise TypeError('can only append SimpleEvaluation (not "%s") to '
                            'MeanEvaluation' % type(other).__name__)

    @property
    def num_tp(self):
        """Number of true positive detections."""
        if self._num_tp.size == 0:
            return 0.
        return np.mean(self._num_tp)

    @property
    def num_fp(self):
        """Number of false positive detections."""
        if self._num_fp.size == 0:
            return 0.
        return np.mean(self._num_fp)

    @property
    def num_tn(self):
        """Number of true negative detections."""
        if self._num_tn.size == 0:
            return 0.
        return np.mean(self._num_tn)

    @property
    def num_fn(self):
        """Number of false negative detections."""
        if self._num_fn.size == 0:
            return 0.
        return np.mean(self._num_fn)

    @property
    def precision(self):
        """Precision."""
        if self._precision.size == 0:
            return 0.
        return np.mean(self._precision)

    @property
    def recall(self):
        """Recall."""
        if self._recall.size == 0:
            return 0.
        return np.mean(self._recall)

    @property
    def fmeasure(self):
        """F-measure."""
        if self._fmeasure.size == 0:
            return 0.
        return np.mean(self._fmeasure)

    @property
    def accuracy(self):
        """Accuracy."""
        if self._accuracy.size == 0:
            return 0.
        return np.mean(self._accuracy)

    @property
    def errors(self):
        """Errors."""
        if self._errors.size == 0:
            return 0.
        return self._errors

    @property
    def mean_error(self):
        """Mean of the errors."""
        if self._mean.size == 0:
            return 0.
        return np.mean(self._mean)

    @property
    def std_error(self):
        """Standard deviation of the errors."""
        if self._std.size == 0:
            return 0.
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
        self._tp = np.asarray(tp, dtype=np.float)
        self._fp = np.asarray(fp, dtype=np.float)
        self._tn = np.asarray(tn, dtype=np.float)
        self._fn = np.asarray(fn, dtype=np.float)

    # for adding another Evaluation object, i.e. summing them
    def __iadd__(self, other):
        if isinstance(other, Evaluation):
            # extend the arrays
            self._tp = np.append(self.tp, other.tp)
            self._fp = np.append(self.fp, other.fp)
            self._tn = np.append(self.tn, other.tn)
            self._fn = np.append(self.fn, other.fn)
            self._errors = np.append(self.errors, other.errors)
            return self
        else:
            raise TypeError("Can't add %s to Evaluation." % type(other))

    # for adding two Evaluation objects
    def __add__(self, other):
        if isinstance(other, Evaluation):
            # extend the arrays
            tp = np.append(self.tp, other.tp)
            fp = np.append(self.fp, other.fp)
            tn = np.append(self.tn, other.tn)
            fn = np.append(self.fn, other.fn)
            # create a new object we can return
            new = Evaluation(tp, fp, tn, fn)
            # modify the hidden variable directly
            # (needed for correct inheritance)
            new._errors = np.append(self.errors, other.errors)
            # return the newly created object
            return new
        else:
            raise TypeError("Can't add %s to Evaluation." % type(other))

    @property
    def tp(self):
        """True positive detections."""
        return self._tp

    @property
    def num_tp(self):
        """Number of true positive detections."""
        return len(self._tp)

    @property
    def fp(self):
        """False positive detections."""
        return self._fp

    @property
    def num_fp(self):
        """Number of false positive detections."""
        return len(self._fp)

    @property
    def tn(self):
        """True negative detections."""
        return self._tn

    @property
    def num_tn(self):
        """Number of true negative detections."""
        return len(self._tn)

    @property
    def fn(self):
        """False negative detections."""
        return self._fn

    @property
    def num_fn(self):
        """Number of false negative detections."""
        return len(self._fn)
