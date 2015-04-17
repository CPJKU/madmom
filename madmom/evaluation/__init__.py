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
def find_closest_matches(detections, annotations):
    """
    Find the closest annotation for each detection.

    :param detections:  numpy array with the detected events [float, seconds]
    :param annotations: numpy array with the annotated events [float, seconds]
    :return:            numpy array with indices of the closest matches [int]

    Note: The sequences must be ordered!

    """
    # if no detections or annotations are given
    if len(detections) == 0 or len(annotations) == 0:
        # return a empty array
        return np.zeros(0, dtype=np.int)
    # if only a single annotation is given
    if len(annotations) == 1:
        # return an array as long as the detections with indices 0
        return np.zeros(len(detections), dtype=np.int)
    # solution found at: http://stackoverflow.com/questions/8914491/
    indices = annotations.searchsorted(detections)
    indices = np.clip(indices, 1, len(annotations) - 1)
    left = annotations[indices - 1]
    right = annotations[indices]
    indices -= detections - left < right - detections
    # return the indices of the closest matches
    return indices


def calc_errors(detections, annotations, matches=None):
    """
    Errors of the detections relative to the closest annotations.

    :param detections:  numpy array with the detected events [float, seconds]
    :param annotations: numpy array with the annotated events [float, seconds]
    :param matches:     numpy array with indices of the closest events [int]
    :return:            numpy array with the errors [seconds]

    Note: The sequences must be ordered! To speed up the calculation, a list
          of pre-computed indices of the closest matches can be used.

    """
    # if no detections or annotations are given
    if len(detections) == 0 or len(annotations) == 0:
        # return a empty array
        return np.zeros(0, dtype=np.float)
    # determine the closest annotations
    if matches is None:
        matches = find_closest_matches(detections, annotations)
    # calc error relative to those annotations
    errors = detections - annotations[matches]
    # return the errors
    return errors


def calc_absolute_errors(detections, annotations, matches=None):
    """
    Absolute errors of the detections relative to the closest annotations.

    :param detections:  numpy array with the detected events [float, seconds]
    :param annotations: numpy array with the annotated events [float, seconds]
    :param matches:     numpy array with indices of the closest events [int]
    :return:            numpy array with the absolute errors [seconds]

    Note: The sequences must be ordered! To speed up the calculation, a list
          of pre-computed indices of the closest matches can be used.

    """
    # return the errors
    return np.abs(calc_errors(detections, annotations, matches))


def calc_relative_errors(detections, annotations, matches=None):
    """
    Relative errors of the detections to the closest annotations.

    :param detections:  numpy array with the detected events [float, seconds]
    :param annotations: numpy array with the annotated events [float, seconds]
    :param matches:     numpy array with indices of the closest events [int]
    :return:            numpy array with the relative errors [seconds]

    Note: The sequences must be ordered! To speed up the calculation, a list of
          pre-computed indices of the closest matches can be used.

    """
    # if no detections or annotations are given
    if len(detections) == 0 or len(annotations) == 0:
        # return a empty array
        return np.zeros(0, dtype=np.float)
    # determine the closest annotations
    if matches is None:
        matches = find_closest_matches(detections, annotations)
    # calculate the absolute errors
    errors = calc_errors(detections, annotations, matches)
    # return the relative errors
    return np.abs(1 - (errors / annotations[matches]))


# evaluation classes
class SimpleEvaluation(object):
    """
    Simple evaluation class for calculating Precision, Recall and F-measure
    based on the numbers of true/false positive/negative detections.

    Note: so far, this class is only suitable for a 1-class evaluation problem.

    """
    def __init__(self, num_tp=0, num_fp=0, num_tn=0, num_fn=0):
        """
        Creates a new SimpleEvaluation instance.

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
        # derived classes must redefine them accordingly
        self._errors = []

    # for adding another SimpleEvaluation object, i.e. summing them
    def __iadd__(self, other):
        if isinstance(other, SimpleEvaluation):
            # increase the counters
            self._num_tp += other.num_tp
            self._num_fp += other.num_fp
            self._num_tn += other.num_tn
            self._num_fn += other.num_fn
            # extend the errors array
            self._errors.extend(other.errors)
            # return the modified object
            return self
        else:
            raise TypeError('Can only add SimpleEvaluation or derived class to'
                            ' %s, not %s' % (type(self).__name__,
                                             type(other).__name__))

    # for adding two SimpleEvaluation objects
    def __add__(self, other):
        if isinstance(other, SimpleEvaluation):
            num_tp = self._num_tp + other.num_tp
            num_fp = self._num_fp + other.num_fp
            num_tn = self._num_tn + other.num_tn
            num_fn = self._num_fn + other.num_fn
            # create a new object
            new = SimpleEvaluation(num_tp, num_fp, num_tn, num_fn)
            # modify the hidden _errors variable directly
            # first copy the list and then extend it
            new._errors = list(self.errors)
            new._errors.extend(other.errors)
            # return the newly created object
            return new
        else:
            raise TypeError('Can only add SimpleEvaluation or derived class to'
                            ' %s, not %s' % (type(self).__name__,
                                             type(other).__name__))

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
    def num_annotations(self):
        """Number of annotations."""
        return self.num_tp + self.num_fn

    def __len__(self):
        # the length equals the number of annotations
        return self.num_annotations

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
        # if there are no positive annotations, we recalled all of them
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
        """
        Errors of the true positive detections relative to the corresponding
        annotations.

        """
        # if any errors are given, they have to be the same length as the true
        # positive detections
        # Note: access the hidden variable _errors and the property num_tp
        #       because different classes implement the latter differently
        if len(self._errors) > 0 and len(self._errors) != self.num_tp:
            raise AssertionError("length of the errors and number of true "
                                 "positive detections must match")
        return self._errors

    @property
    def mean_error(self):
        """Mean of the errors."""
        if len(self.errors) == 0:
            return 0.
        return np.mean(self.errors)

    @property
    def std_error(self):
        """Standard deviation of the errors."""
        if len(self.errors) == 0:
            return 0.
        return np.std(self.errors)

    def print_errors(self, indent='', tex=False, verbose=True):
        """
        Print errors.

        :param indent:  use the given string as indentation
        :param tex:     output format to be used in .tex files
        :param verbose: add true/false positive rates and mean/std of errors

        """
        # print the errors
        if tex:
            # tex formatting
            ret = 'tex & Precision & Recall & F-measure & Accuracy & Mean & ' \
                  'Std.dev\\\\\n %i annotations & %.3f & %.3f & %.3f & %.3f ' \
                  '& %.2f ms & %.2f ms\\\\' % \
                  (self.num_annotations, self.precision, self.recall,
                   self.fmeasure, self.accuracy, self.mean_error * 1000.,
                   self.std_error * 1000.)
        else:
            # normal formatting
            ret = '%sannotations: %5d correct: %5d fp: %5d fn: %5d p=%.3f ' \
                  'r=%.3f f=%.3f' % (indent, self.num_annotations, self.num_tp,
                                     self.num_fp, self.num_fn, self.precision,
                                     self.recall, self.fmeasure)
            if verbose:
                ret += ' acc: %.3f mean: %.1f ms std: %.1f ms' % \
                       (self.accuracy, self.mean_error * 1000.,
                        self.std_error * 1000.)
        # return
        return ret

    def __str__(self):
        return self.print_errors()


# class for summing Evaluations
SumEvaluation = SimpleEvaluation


# class for averaging Evaluations
class MeanEvaluation(SimpleEvaluation):
    """
    Simple evaluation class for averaging Precision, Recall and F-measure.

    """
    def __init__(self):
        """
        Creates a new MeanEvaluation instance.

        """
        super(MeanEvaluation, self).__init__()
        # redefine most of the stuff as arrays so we can average them
        self._num_tp = []
        self._num_fp = []
        self._num_tn = []
        self._num_fn = []
        self._precision = []
        self._recall = []
        self._fmeasure = []
        self._accuracy = []
        self._errors = []
        self._mean_errors = []
        self._std_errors = []

    def __len__(self):
        # just use the length of any of the arrays
        return len(self._num_tp)

    # for adding another Evaluation object
    def append(self, other):
        """
        Appends the scores of another SimpleEvaluation (or derived class)
        object to the respective arrays.

        :param other: SimpleEvaluation (or derived class) object

        """
        if isinstance(other, SimpleEvaluation):
            # append the numbers of any Evaluation object to the arrays
            self._num_tp.append(other.num_tp)
            self._num_fp.append(other.num_fp)
            self._num_tn.append(other.num_tn)
            self._num_fn.append(other.num_fn)
            self._precision.append(other.precision)
            self._recall.append(other.recall)
            self._fmeasure.append(other.fmeasure)
            self._accuracy.append(other.accuracy)
            # TODO: extend the errors list instead of appending, might lead to
            #       undesired effects with lists of tuples or lists of lists
            self._errors.extend(other.errors)
            self._mean_errors.append(other.mean_error)
            self._std_errors.append(other.std_error)
        else:
            raise TypeError('Can only append SimpleEvaluation or derived class'
                            ' to %s, not %s' % (type(self).__name__,
                                                type(other).__name__))

    @property
    def num_tp(self):
        """Number of true positive detections."""
        if len(self._num_tp) == 0:
            return 0.
        return np.mean(self._num_tp)

    @property
    def num_fp(self):
        """Number of false positive detections."""
        if len(self._num_fp) == 0:
            return 0.
        return np.mean(self._num_fp)

    @property
    def num_tn(self):
        """Number of true negative detections."""
        if len(self._num_tn) == 0:
            return 0.
        return np.mean(self._num_tn)

    @property
    def num_fn(self):
        """Number of false negative detections."""
        if len(self._num_fn) == 0:
            return 0.
        return np.mean(self._num_fn)

    @property
    def precision(self):
        """Precision."""
        if len(self._precision) == 0:
            return 0.
        return np.mean(self._precision)

    @property
    def recall(self):
        """Recall."""
        if len(self._recall) == 0:
            return 0.
        return np.mean(self._recall)

    @property
    def fmeasure(self):
        """F-measure."""
        if len(self._fmeasure) == 0:
            return 0.
        return np.mean(self._fmeasure)

    @property
    def accuracy(self):
        """Accuracy."""
        if len(self._accuracy) == 0:
            return 0.
        return np.mean(self._accuracy)

    @property
    def mean_error(self):
        """Mean of the errors."""
        if len(self._mean_errors) == 0:
            return 0.
        return np.mean(self._mean_errors)

    @property
    def std_error(self):
        """Standard deviation of the errors."""
        if len(self._std_errors) == 0:
            return 0.
        return np.mean(self._std_errors)

    def print_errors(self, indent='', verbose=True):
        """
        Print errors.

        :param indent:  use the given string as indentation
        :param verbose: add true/false positive rates and mean/std of errors

        """
        # use floats instead of integers for reporting
        ret = '%sannotations: %5.2f correct: %5.2f fp: %5.2f fn: %5.2f ' \
              'p=%.3f r=%.3f f=%.3f' % \
              (indent, self.num_annotations, self.num_tp, self.num_fp,
               self.num_fn, self.precision, self.recall, self.fmeasure)
        if verbose:
            ret += ' acc: %.3f mean: %.1f ms std: %.1f ms' % \
                   (self.accuracy, self.mean_error * 1000.,
                    self.std_error * 1000.)
        return ret


# class for evaluation of Precision, Recall, F-measure with lists
class Evaluation(SimpleEvaluation):
    """
    Evaluation class for measuring Precision, Recall and F-measure based on
    numpy arrays or lists with true/false positive/negative detections.

    """
    def __init__(self, tp=None, fp=None, tn=None, fn=None):
        """
        Creates a new Evaluation instance.

        :param tp: list with true positive detections [seconds]
        :param fp: list with false positive detections [seconds]
        :param tn: list with true negative detections [seconds]
        :param fn: list with false negative detections [seconds]

        """
        # set default values
        if tp is None:
            tp = []
        if fp is None:
            fp = []
        if tn is None:
            tn = []
        if fn is None:
            fn = []
        super(Evaluation, self).__init__()
        self._tp = list(tp)
        self._fp = list(fp)
        self._tn = list(tn)
        self._fn = list(fn)

    # for adding another Evaluation object, i.e. summing them
    def __iadd__(self, other):
        if isinstance(other, Evaluation):
            # extend the arrays
            self._tp.extend(other.tp)
            self._fp.extend(other.fp)
            self._tn.extend(other.tn)
            self._fn.extend(other.fn)
            self._errors.extend(other.errors)
            # return the modified object
            return self
        else:
            raise TypeError('Can only add Evaluation or derived class to %s, '
                            'not %s' % (type(self).__name__,
                                        type(other).__name__))

    # for adding two Evaluation objects
    def __add__(self, other):
        if isinstance(other, Evaluation):
            # copy the lists
            tp = list(self.tp)
            fp = list(self.fp)
            tn = list(self.tn)
            fn = list(self.fn)
            # and extend them
            tp.extend(other.tp)
            fp.extend(other.fp)
            tn.extend(other.tn)
            fn.extend(other.fn)
            # create a new object
            new = Evaluation(tp, fp, tn, fn)
            # modify the hidden _errors variable directly
            new._errors = list(self._errors)
            new._errors.extend(other.errors)
            # return the newly created object
            return new
        else:
            raise TypeError('Can only add Evaluation or derived class to %s, '
                            'not %s' % (type(self).__name__,
                                        type(other).__name__))

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


# class for evaluation of Precision, Recall, F-measure with 2D arrays
class MultiClassEvaluation(Evaluation):
    """
    Evaluation class for measuring Precision, Recall and F-measure based on
    2D numpy arrays with true/false positive/negative detections.

    """
    def print_errors(self, indent='', tex=False, verbose=True):
        """
        Print errors.

        :param indent:  use the given string as indentation
        :param tex:     output format to be used in .tex files
        :param verbose: add evaluation for individual classes

        """
        # print the errors
        annotations = self.num_tp + self.num_fn
        tpr = self.recall
        fpr = (1 - self.precision)
        ret = ''
        if tex:
            # tex formatting
            ret = 'tex & Precision & Recall & F-measure & True Positives & ' \
                  'False Positives & Accuracy & Mean & Std.dev\\\\\n %i ' \
                  'annotations & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & ' \
                  '%.2f ms & %.2f ms\\\\' % \
                  (annotations, self.precision, self.recall, self.fmeasure,
                   tpr, fpr, self.accuracy, self.mean_error * 1000.,
                   self.std_error * 1000.)
            # TODO: add individual class output
        else:
            if verbose:
                # print errors for all classes individually
                tp = np.asarray(self.tp)
                fp = np.asarray(self.fp)
                tn = np.asarray(self.tn)
                fn = np.asarray(self.fn)
                # extract all classes
                classes = []
                if tp.any():
                    np.append(classes, np.unique(tp[:, 1]))
                if fp.any():
                    np.append(classes, np.unique(fp[:, 1]))
                if tn.any():
                    np.append(classes, np.unique(tn[:, 1]))
                if fn.any():
                    np.append(classes, np.unique(fn[:, 1]))
                for cls in sorted(np.unique(classes)):
                    # extract the TP, FP, TN and FN of this class
                    tp_ = tp[tp[:, 1] == cls]
                    fp_ = fp[fp[:, 1] == cls]
                    tn_ = tn[tn[:, 1] == cls]
                    fn_ = fn[fn[:, 1] == cls]
                    # evaluate them
                    e = Evaluation(tp_, fp_, tn_, fn_)
                    # append to the output string
                    string = e.print_errors(indent * 2, verbose=False)
                    ret += '%s Class %s:\n%s\n' % (indent, cls, string)
            # normal formatting
            ret += '%sannotations: %5d correct: %5d fp: %4d fn: %4d p=%.3f ' \
                   'r=%.3f f=%.3f\n%stpr: %.1f%% fpr: %.1f%% acc: %.1f%% ' \
                   'mean: %.1f ms std: %.1f ms' % \
                   (indent, annotations, self.num_tp, self.num_fp, self.num_fn,
                    self.precision, self.recall, self.fmeasure, indent,
                    tpr * 100., fpr * 100., self.accuracy * 100.,
                    self.mean_error * 1000., self.std_error * 1000.)
        # return
        return ret


def evaluation_io(parser, ann_suffix, det_suffix, ann_dir=None, det_dir=None):
    """
    Add evaluation related arguments to an existing parser object.

    :param parser:     existing argparse parser object
    :param ann_suffix: suffix for the annotation files
    :param det_suffix: suffix for the detection files
    :param ann_dir:    use only annotations from this folder (+ sub-folders)
    :param det_dir:    use only detections from this folder (+ sub-folders)
    :return:           audio argument parser group object

    """
    parser.add_argument('files', nargs='*',
                        help='files (or folders) to be evaluated')
    # suffixes used for evaluation
    parser.add_argument('-a', dest='ann_suffix', action='store',
                        default=ann_suffix,
                        help='suffix of the annotation files '
                             '[default: %(default)s]')
    parser.add_argument('--ann_dir', action='store',
                        default=ann_dir,
                        help='search only this directory (recursively) for '
                             'annotation files [default: %(default)s]')
    parser.add_argument('-d', dest='det_suffix', action='store',
                        default=det_suffix,
                        help='suffix of the detection files '
                             '[default: %(default)s]')
    parser.add_argument('--det_dir', action='store',
                        default=det_dir,
                        help='search only this directory (recursively) for '
                             'detection files [default: %(default)s]')
    # output options
    g = parser.add_argument_group('formatting arguments')
    g.add_argument('--tex', action='store_true',
                   help='format output to be used in .tex files')
    # verbose
    parser.add_argument('-v', dest='verbose', action='count', default=0,
                        help='increase verbosity level')
    # return the parser
    return parser

# finally import the submodules
from . import onsets, beats, notes, tempo
