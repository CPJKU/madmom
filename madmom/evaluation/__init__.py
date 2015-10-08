# encoding: utf-8
"""
Evaluation package.

All evaluation methods of this package can be used as scripts directly, if the
package is in $PYTHONPATH.

Example:

python -m madmom.evaluation.onsets /dir/to/be/evaluated

"""
import abc
import re
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
    # make sure the arrays have the correct types
    detections = np.asarray(detections, dtype=np.float)
    annotations = np.asarray(annotations, dtype=np.float)
    # TODO: right now, it only works with 1D arrays
    if detections.ndim > 1 or annotations.ndim > 1:
        raise NotImplementedError('please implement multi-dim support')
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
    # make sure the arrays have the correct types
    detections = np.asarray(detections, dtype=np.float)
    annotations = np.asarray(annotations, dtype=np.float)
    if matches is not None:
        matches = np.asarray(matches, dtype=np.int)
    # TODO: right now, it only works with 1D arrays
    if detections.ndim > 1 or annotations.ndim > 1:
        raise NotImplementedError('please implement multi-dim support')
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
    # make sure the arrays have the correct types
    detections = np.asarray(detections, dtype=np.float)
    annotations = np.asarray(annotations, dtype=np.float)
    if matches is not None:
        matches = np.asarray(matches, dtype=np.int)
    # TODO: right now, it only works with 1D arrays
    if detections.ndim > 1 or annotations.ndim > 1:
        raise NotImplementedError('please implement multi-dim support')
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
    # make sure the arrays have the correct types
    detections = np.asarray(detections, dtype=np.float)
    annotations = np.asarray(annotations, dtype=np.float)
    if matches is not None:
        matches = np.asarray(matches, dtype=np.int)
    # TODO: right now, it only works with 1D arrays
    if detections.ndim > 1 or annotations.ndim > 1:
        raise NotImplementedError('please implement multi-dim support')
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


# abstract evaluation base class
class EvaluationABC(object):
    """
    Evaluation abstract base class.

    `METRIC_NAMES` is a list of tuples, containing the attribute's name and the
    corresponding label, e.g.:

    METRIC_NAMES = [
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('fmeasure', 'F-measure'),
    ]

    The attributes defined in `METRIC_NAMES` will be provided as an ordered
    dictionary as the `metrics` attribute of the

    """
    __metaclass__ = abc.ABCMeta

    name = None
    METRIC_NAMES = []
    FLOAT_FORMAT = '{:.3f}'

    @ property
    def metrics(self):
        """Metrics as a dictionary."""
        # TODO: use an ordered dict?
        from collections import OrderedDict
        metrics = OrderedDict()
        # metrics = {}
        for metric in [m[0] for m in self.METRIC_NAMES]:
            metrics[metric] = getattr(self, metric)
        return metrics

    @abc.abstractmethod
    def __len__(self):
        """Length of the evaluation object."""
        return

    def tostring(self, **kwargs):
        """
        Format the evaluation metrics as a human readable string.

        :param kwargs: additional keyword arguments
        :return:       evaluation metrics formatted as a human readable string

        Note: This is a fallback method formatting the 'metrics' dictionary in
              a human readable way. Classes implementing this abstract base
              class should provide a better suitable method.

        """
        import pprint
        return pprint.pformat(dict(self.metrics), indent=4)


# evaluation classes
class SimpleEvaluation(EvaluationABC):
    """
    Simple Precision, Recall, F-measure and Accuracy evaluation based on the
    numbers of true/false positive/negative detections.

    Note: This class is only suitable for a 1-class evaluation problem.

    """
    METRIC_NAMES = [
        ('num_tp', 'No. of true positives'),
        ('num_fp', 'No. of false positives'),
        ('num_tn', 'No. of true negatives'),
        ('num_fn', 'No. of false negatives'),
        ('num_annotations', 'No. Annotations'),
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('fmeasure', 'F-measure'),
        ('accuracy', 'Accuracy'),
    ]

    def __init__(self, num_tp=0, num_fp=0, num_tn=0, num_fn=0, name=None,
                 **kwargs):
        """
        Creates a new SimpleEvaluation instance.

        :param num_tp: number of true positive detections
        :param num_fp: number of false positive detections
        :param num_tn: number of true negative detections
        :param num_fn: number of false negative detections
        :param name:   name of the evaluation to be displayed
        :param kwargs: additional arguments will be ignored

        """
        # hidden variables, to be able to overwrite them in subclasses
        self._num_tp = int(num_tp)
        self._num_fp = int(num_fp)
        self._num_tn = int(num_tn)
        self._num_fn = int(num_fn)
        # name of the evaluation
        self.name = name

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

    def tostring(self, **kwargs):
        """
        Format the evaluation metrics as a human readable string.

        :param kwargs: additional arguments will be ignored
        :return:       evaluation metrics formatted as a human readable string

        """
        ret = ''
        if self.name is not None:
            ret += '%s\n  ' % self.name
        ret += 'Annotations: %5d TP: %5d FP: %5d FN: %5d ' \
               'Precision: %.3f Recall: %.3f F-measure: %.3f Acc: %.3f' % \
               (self.num_annotations, self.num_tp, self.num_fp, self.num_fn,
                self.precision, self.recall, self.fmeasure, self.accuracy)
        return ret

    def __str__(self):
        return self.tostring()


# evaluate Precision, Recall, F-measure and Accuracy with lists or numpy arrays
class Evaluation(SimpleEvaluation):
    """
    Evaluation class for measuring Precision, Recall and F-measure based on
    numpy arrays or lists with true/false positive/negative detections.

    """
    # METRIC_NAMES = [
    #     ('tp', 'True positives'),
    #     ('fp', 'False positives'),
    #     ('tn', 'True negatives'),
    #     ('fn', 'False negatives'),
    #     ('num_tp', 'No. of true positives'),
    #     ('num_fp', 'No. of false positives'),
    #     ('num_tn', 'No. of true negatives'),
    #     ('num_fn', 'No. of false negatives'),
    #     ('num_annotations', 'No. Annotations'),
    #     ('precision', 'Precision'),
    #     ('recall', 'Recall'),
    #     ('fmeasure', 'F-measure'),
    #     ('accuracy', 'Accuracy'),
    # ]

    def __init__(self, tp=None, fp=None, tn=None, fn=None, **kwargs):
        """
        Creates a new Evaluation instance.

        :param tp:     list/array with true positive detections
        :param fp:     list/array with false positive detections
        :param tn:     list/array with true negative detections
        :param fn:     list/array with false negative detections
        :param kwargs: keyword arguments passed to SimpleEvaluation()

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
        # convert everything to lists
        tp = list(tp)
        fp = list(fp)
        tn = list(tn)
        fn = list(fn)
        # and finally to numpy arrays
        super(Evaluation, self).__init__(**kwargs)
        self.tp = np.asarray(tp, dtype=np.float)
        self.fp = np.asarray(fp, dtype=np.float)
        self.tn = np.asarray(tn, dtype=np.float)
        self.fn = np.asarray(fn, dtype=np.float)

    @property
    def num_tp(self):
        """Number of true positive detections."""
        return len(self.tp)

    @property
    def num_fp(self):
        """Number of false positive detections."""
        return len(self.fp)

    @property
    def num_tn(self):
        """Number of true negative detections."""
        return len(self.tn)

    @property
    def num_fn(self):
        """Number of false negative detections."""
        return len(self.fn)


# class for evaluation of Precision, Recall, F-measure with 2D arrays
class MultiClassEvaluation(Evaluation):
    """
    Evaluation class for measuring Precision, Recall and F-measure based on
    2D numpy arrays with true/false positive/negative detections.

    """
    def __init__(self, tp=None, fp=None, tn=None, fn=None):
        """
        Creates a new Evaluation instance.

        :param tp: list of tuples or 2D array with true positive detections
        :param fp: list of tuples or 2D array with false positive detections
        :param tn: list of tuples or 2D array with true negative detections
        :param fn: list of tuples or 2D array with false negative detections

        Note: The second item of the tuples or the second column of the arrays
              denote the class the detection belongs to.

        """
        # set default values
        if tp is None:
            tp = np.zeros((0, 2))
        if fp is None:
            fp = np.zeros((0, 2))
        if tn is None:
            tn = np.zeros((0, 2))
        if fn is None:
            fn = np.zeros((0, 2))
        super(Evaluation, self).__init__()
        self.tp = np.asarray(tp, dtype=np.float)
        self.fp = np.asarray(fp, dtype=np.float)
        self.tn = np.asarray(tn, dtype=np.float)
        self.fn = np.asarray(fn, dtype=np.float)

    def tostring(self, verbose=True):
        """
        Format the evaluation metrics as a human readable string.

        :param verbose: add evaluation for individual classes
        :return:        evaluation metrics formatted as a human readable string

        """
        ret = ''

        if verbose:
            # extract all classes
            classes = []
            if self.tp.any():
                np.append(classes, np.unique(self.tp[:, 1]))
            if self.fp.any():
                np.append(classes, np.unique(self.fp[:, 1]))
            if self.tn.any():
                np.append(classes, np.unique(self.tn[:, 1]))
            if self.fn.any():
                np.append(classes, np.unique(self.fn[:, 1]))
            for cls in sorted(np.unique(classes)):
                # extract the TP, FP, TN and FN of this class
                tp = self.tp[self.tp[:, 1] == cls]
                fp = self.fp[self.fp[:, 1] == cls]
                tn = self.tn[self.tn[:, 1] == cls]
                fn = self.fn[self.fn[:, 1] == cls]
                # evaluate them
                e = Evaluation(tp, fp, tn, fn)
                # append to the output string
                string = e.tostring(verbose=False)
                ret += add_indent(string, 'Class %s:\n' % cls) + '\n'
        # normal formatting
        ret += 'Annotations: %5d TP: %5d FP: %4d FN: %4d ' \
               'Precision: %.3f Recall: %.3f F-measure: %.3f Acc: %.3f%' % \
               (self.num_annotations, self.num_tp, self.num_fp, self.num_fn,
                self.precision, self.recall, self.fmeasure, self.accuracy)
        # return
        return ret


# class for summing Evaluations
class SumEvaluation(SimpleEvaluation):
    """
    Simple class for summing evaluations.

    """

    def __init__(self, eval_objects, name=None):
        """
        Creates a new SumEvaluation instance.

        :param eval_objects: list of evaluation objects
        :param name:

        """
        # Note: we want to inherit the evaluation functions/properties, no need
        #       to call __super__
        if not isinstance(eval_objects, list):
            # wrap the given eval_object in a list
            eval_objects = [eval_objects]
        self.eval_objects = eval_objects
        self.name = name or 'sum for %d files' % len(self)

    def __len__(self):
        # just use the length of the evaluation objects
        return len(self.eval_objects)

    # redefine the counters (number of TP, FP, TN, FN & annotations)

    @property
    def num_tp(self):
        """Number of true positive detections."""
        return sum(e.num_tp for e in self.eval_objects)

    @property
    def num_fp(self):
        """Number of false positive detections."""
        return sum(e.num_fp for e in self.eval_objects)

    @property
    def num_tn(self):
        """Number of true negative detections."""
        return sum(e.num_tn for e in self.eval_objects)

    @property
    def num_fn(self):
        """Number of false negative detections."""
        return sum(e.num_fn for e in self.eval_objects)

    @property
    def num_annotations(self):
        """Number of annotations."""
        return sum(e.num_annotations for e in self.eval_objects)


# class for averaging Evaluations
class MeanEvaluation(SumEvaluation):
    """
    Simple class for averaging evaluation.

    """

    def __init__(self, eval_objects, name=None, **kwargs):
        """
        Creates a new MeanEvaluation instance.

        :param eval_objects: list of evaluation objects.
        :param name:

        """
        super(MeanEvaluation, self).__init__(eval_objects, **kwargs)
        self.name = name or 'mean for %d files' % len(self)

    # overwrite the properties to calculate the mean instead of the sum

    @property
    def num_tp(self):
        """Number of true positive detections."""
        if len(self.eval_objects) == 0:
            return 0.
        return np.nanmean([e.num_tp for e in self.eval_objects])

    @property
    def num_fp(self):
        """Number of false positive detections."""
        if len(self.eval_objects) == 0:
            return 0.
        return np.nanmean([e.num_fp for e in self.eval_objects])

    @property
    def num_tn(self):
        """Number of true negative detections."""
        if len(self.eval_objects) == 0:
            return 0.
        return np.nanmean([e.num_tn for e in self.eval_objects])

    @property
    def num_fn(self):
        """Number of false negative detections."""
        if len(self.eval_objects) == 0:
            return 0.
        return np.nanmean([e.num_fn for e in self.eval_objects])

    @property
    def num_annotations(self):
        """Number of annotations."""
        if len(self.eval_objects) == 0:
            return 0.
        return np.nanmean([e.num_annotations for e in self.eval_objects])

    @property
    def precision(self):
        """Precision."""
        return np.nanmean([e.precision for e in self.eval_objects])

    @property
    def recall(self):
        """Recall."""
        return np.nanmean([e.recall for e in self.eval_objects])

    @property
    def fmeasure(self):
        """F-measure."""
        return np.nanmean([e.fmeasure for e in self.eval_objects])

    @property
    def accuracy(self):
        """Accuracy."""
        return np.nanmean([e.accuracy for e in self.eval_objects])

    def tostring(self):
        """
        Format the evaluation metrics as a human readable string.

        :return: evaluation metrics formatted as a human readable string

        """
        # TODO: unify this with SimpleEvaluation but
        #       add option to provide field formatters (e.g. 3d or 5.2f)
        # format with floats instead of integers
        ret = 'Annotations: %5.2f TP: %5.2f FP: %5.2f FN: %5.2f' \
              'Precision: %.3f Recall: %.3f F-measure: %.3f Acc: %.3f' % \
              (self.num_annotations, self.num_tp, self.num_fp, self.num_fn,
               self.precision, self.recall, self.fmeasure, self.accuracy)
        return ret


def tostring(eval_objects, metric_names=None, float_format='{:.3f}'):
    """
    Format the given evaluation objects as human readable strings.

    :param eval_objects: evaluation objects
    :param metric_names: list of tuples defining the name of the property
                         corresponding to the metric, and the metric label
                         e.g. ('fp', 'False Positives')
    :param float_format: how to format the metrics
    :return:             human readable output of the evaluation objects

    Note: If no `metric_names` are given, they will be extracted from the first
          evaluation object.

    """
    return '\n'.join([e.tostring() for e in eval_objects])


def tocsv(eval_objects, metric_names=None, float_format='{:.3f}'):
    """
    Format the given evaluation objects as a CSV table.

    :param eval_objects: evaluation objects
    :param metric_names: list of tuples defining the name of the property
                         corresponding to the metric, and the metric label
                         e.g. ('fp', 'False Positives')
    :return:             CSV table representation of the evaluation objects

    Note: If no `metric_names` are given, they will be extracted from the first
          evaluation object.

    """
    if metric_names is None:
        # get the evaluation metrics from the first evaluation object
        metric_names = eval_objects[0].METRIC_NAMES
    metric_names, metric_labels = zip(*metric_names)
    # add header
    lines = ['Name,' + ','.join(metric_labels)]
    # TODO: use e.metrics dict?
    # add the evaluation objects
    for e in eval_objects:
        values = [float_format.format(getattr(e, mn)) for mn in metric_names]
        lines.append(e.name + ',' + ','.join(values))
    # return everything
    return '\n'.join(lines)


def totex(eval_objects, metric_names=None, float_format='{:.3f}'):
    """
    Format the given evaluation objects as a LaTeX table.

    :param eval_objects: evaluation objects
    :param metric_names: list of tuples defining the name of the property
                         corresponding to the metric, and the metric label
                         e.g. ('fp', 'False Positives')
    :return:             LaTeX table representation of the evaluation objects

    Note: If no `metric_names` are given, they will be extracted from the first
          evaluation object.
    """
    if metric_names is None:
        # get the evaluation metrics from the first evaluation object
        metric_names = eval_objects[0].METRIC_NAMES
    metric_names, metric_labels = zip(*metric_names)
    # add header
    lines = ['Name & ' + ' & '.join(metric_labels) + '\\\\']
    # TODO: use e.metrics dict
    # TODO: add a generic totable() function which accepts columns separator,
    #       newline stuff (e.g. tex \\\\) and others
    # add the evaluation objects
    for e in eval_objects:
        values = [float_format.format(getattr(e, mn)) for mn in metric_names]
        lines.append(e.name + ' & ' + ' & '.join(values) + '\\\\')
    # return everything
    return '\n'.join(lines)


def add_indent(lines, indent):
    """
    Adds an indent to a given string. The string may contain line breaks. In
    this case, the indent is used as is in the first line, while for the other
    line a white-space indent of the same length as the actual indent is used.

    :param lines:  string containing the lines to be indented
    :param indent: indent to use
    :return:       string containing the indented lines

    """
    # split the lines
    split_lines = lines.split('\n')
    # add indent to first line
    split_lines[0] = indent + split_lines[0]
    # create an indentation of the same length as indent parameter containing
    # only whitespaces
    whitespace_indent = re.sub('[^\s]', ' ', indent.split('\n')[-1])
    whitespace_indent = re.sub('\n', '', whitespace_indent)
    # add whitespace indent to other lines
    for i, eval_line in enumerate(split_lines[1:]):
        split_lines[i + 1] = whitespace_indent + eval_line
    # return the indented lines
    return '\n'.join(split_lines)


def evaluation_io(parser, ann_suffix, det_suffix, ann_dir=None, det_dir=None):
    """
    Add evaluation input/output related arguments to an existing parser object.

    :param parser:     existing argparse parser object
    :param ann_suffix: suffix for the annotation files
    :param det_suffix: suffix for the detection files
    :param ann_dir:    use only annotations from this folder (+ sub-folders)
    :param det_dir:    use only detections from this folder (+ sub-folders)
    :return:           evaluation output formatter argument group

    """
    parser.add_argument('files', nargs='*',
                        help='files (or folders) to be evaluated')
    # parser.add_argument('-o', STDOUT)
    # suffixes used for evaluation
    g = parser.add_argument_group('file/folder/suffix arguments')
    g.add_argument('-a', dest='ann_suffix', action='store', default=ann_suffix,
                   help='suffix of the annotation files '
                        '[default: %(default)s]')
    g.add_argument('--ann_dir', action='store', default=ann_dir,
                   help='search only this directory (recursively) for '
                        'annotation files [default: %(default)s]')
    g.add_argument('-d', dest='det_suffix', action='store', default=det_suffix,
                   help='suffix of the detection files [default: %(default)s]')
    g.add_argument('--det_dir', action='store', default=det_dir,
                   help='search only this directory (recursively) for '
                        'detection files [default: %(default)s]')
    # option to ignore non-existing detections
    g.add_argument('-i', '--ignore_non_existing', action='store_true',
                   help='ignore non-existing detections [default: raise a '
                        'warning and assume empty detections]')
    # verbose
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='increase verbosity level')
    # option to suppress warnings
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='suppress any warnings')
    # output format options
    g = parser.add_argument_group('formatting arguments')
    parser.set_defaults(output_formatter=tostring)
    formats = g.add_mutually_exclusive_group()
    formats.add_argument('--tex', dest='output_formatter',
                         action='store_const', const=totex,
                         help='format output to be used in .tex files')
    formats.add_argument('--csv', dest='output_formatter',
                         action='store_const', const=tocsv,
                         help='format output to be used in .csv files')
    # return the output formatting group so the caller can add more options
    return g


# finally import the submodules
from . import onsets, beats, notes, tempo, alignment
