# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This file contains global alignment evaluation functionality.

"""

from __future__ import absolute_import, division, print_function

import numpy as np

from . import EvaluationABC


# constants for the data format
_TIME = 0
_SCORE_POS = 1

# constants for missed events/notes
_MISSED_NOTE_VAL = np.NaN

# default settings
WINDOW = 0.25
HISTOGRAM_BINS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 1.]


class AlignmentFormatError(Exception):
    """
    Exception to be raised whenever an incorrect alignment format is given

    """

    def __init__(self, value=None):
        if value is None:
            value = 'Alignment has to be at least one row of two columns ' \
                    'representing time and score position.'
        self.value = value

    def __str__(self):
        return repr(self.value)


def load_alignment(values):
    """
    Load the alignment from given values or file.

    :param values: name of the file, file handle, list of numpy array
    :return:       2D numpy array time and score position columns

    """
    if values is None:
        # return 'empty' alignment
        return np.array([[0, -1]])

    if isinstance(values, (str, file)):
        values = np.loadtxt(values)

    values = np.atleast_2d(values)

    if values.shape[0] < 1 or values.shape[1] < 2 or len(values.shape) > 2:
        raise AlignmentFormatError()

    return values[:, :2]


def compute_event_alignment(alignment, ground_truth):
    """
    This function finds the alignment outputs corresponding to each ground
    truth alignment. In general, the alignment algorithm will output more
    alignment positions than events in the score, e.g. if it is designed to
    output the current alignment at constant intervals.

    :param alignment:    The score follower's resulting alignment.
                         2D NumPy array, first value is the time in seconds,
                         second value is the beat position.
    :param ground_truth: Ground truth of the aligned performance.
                         2D numpy array of similar. First value is the time in
                         seconds, second value is the beat position. It can
                         contain the alignment positions for each individual
                         note. In this case, the deviation for each note is
                         taken into account.
    :return:             2D numpy array of the same size as ground_truth, with
                         each row representing the alignment of the
                         corresponding ground truth element.

    """
    # find the spots where the alignment passes the score
    gt_pos = ground_truth[:, _SCORE_POS]
    al_pos = alignment[:, _SCORE_POS]

    # do not allow to move backwards
    for i in range(1, al_pos.shape[0]):
        al_pos[i] = max(al_pos[i - 1], al_pos[i])

    # find corresponding indices
    al_idxs = np.searchsorted(al_pos, gt_pos)

    # now, the number of indexes in the alignment should correspond
    # to the number of aligned positions in the ground truth
    assert len(al_idxs) == len(ground_truth)

    # first a dummy event at the very end of the alignment is added to be
    # able to process score events with were not reached by the tracker
    dummy = [[_MISSED_NOTE_VAL] * alignment.shape[1]]
    alignment = np.concatenate((alignment, dummy))

    return alignment[al_idxs]


def _attr_name(histogram_bin):
    """
    :return: attribute name for the histogram_bin

    """
    return 'below_{:.2f}'.format(histogram_bin).replace('.', '_')


def _label(histogram_bin):
    """
    :return: label for the histogram_bin

    """
    return '<{:.2f}'.format(histogram_bin)


def compute_metrics(event_alignment, ground_truth, window, err_hist_bins):
    """
    This function computes the evaluation metrics based on the paper
    "Evaluation of Real-Time Audio-to-Score Alignment" by Arshia Cont et al.
    plus an cumulative histogram of absolute errors

    :param event_alignment: sequence alignment as computed by the score
                            follower. 2D numpy array, where the first column is
                            the alignment time in seconds and the second column
                            the position in beats.
                            Needs to be the same length as `ground_truth`,
                            hence for each element in the ground truth the
                            corresponding alignment has to be available. You
                            can use the `compute_event_alignment()` function
                            to compute this.
    :param ground_truth:    ground truth of the aligned performance.
                            2D numpy array, first value is the time in seconds,
                            second value is the beat position. It can contain
                            the alignment positions for each individual note.
                            In this case, the deviation for each note is taken
                            into account.
    :param window:          tolerance window in seconds. Alignments off less
                            than this amount from the ground truth will be
                            considered correct.
    :param err_hist_bins:   list of error bounds for which the cumulative
                            histogram of absolute error will be computed (e.g.
                            [0.1, 0.3] will give the percentage of events
                            aligned with an error smaller than 0.1 and 0.3)
    :return:                dictionary containing (some) of the metrics
                            described in the paper mentioned above and the
                            error histogram

    """
    abs_error = np.abs(event_alignment[:, _TIME] - ground_truth[:, _TIME])
    missed = np.isnan(abs_error)
    aligned_error = np.ma.array(abs_error, mask=missed)

    with np.errstate(invalid='ignore'):
        # for some numpy versions the following prints a invalid value warning
        # although NaNs are masked - code still works.
        misaligned = aligned_error > window

    correctly_aligned_error = np.ma.array(aligned_error, mask=misaligned)
    pc_idx = float(correctly_aligned_error.mask[::-1].argmin())

    # we have to typecast everything to float, if we don't the variables will
    # be of type np.maskedarray
    results = {'miss_rate': float(missed.mean()),
               'misalign_rate': float(misaligned.mean()),
               'avg_imprecision': float(correctly_aligned_error.mean()),
               'stddev_imprecision': float(correctly_aligned_error.std()),
               'avg_error': float(aligned_error.mean()),
               'stddev_error': float(aligned_error.std()),
               'piece_completion': float(
                   1.0 - pc_idx / correctly_aligned_error.mask.shape[0])}

    # convert possibly masked values to NaN. A masked value can occur when
    # computing the mean or stddev of values that are all masked
    for k, v in results.iteritems():
        if v is np.ma.masked_singleton:
            results[k] = np.NaN

    # consider the case where EVERYTHING was missed or misaligned. the standard
    # computation fails then.
    if correctly_aligned_error.mask.all():
        results['piece_completion'] = 0.0

    err_hist, _ = np.histogram(aligned_error.compressed(),
                               bins=[-np.inf] + err_hist_bins + [np.inf])
    cum_hist = np.cumsum(err_hist.astype(float) / aligned_error.shape[0])

    # add the cumulative histogram value per value to the results
    for hb, p in zip(err_hist_bins, cum_hist):
        results[_attr_name(hb)] = p

    return results


class AlignmentEvaluation(EvaluationABC):
    """
    Alignment evaluation class for beat-level alignments. Beat-level aligners
    output beat positions for points in time, rather than computing a time step
    for each individual event in the score. The following metrics are
    available:

    miss_rate:
        Percentage of missed events (events that exist in the reference score,
        but are not reported).

    misalign_rate:
        Percentage of misaligned events (events with an alignment that is off
        by more than a defined threshold (window)).

    avg_imprecision:
        Average alignment error of non-misaligned events.

    stddev_imprecision:
        Standard deviation of alignment error of non-misaligned events.

    avg_error:
        Average alignment error.

    stddev_error:
        Standard deviation of alignment error.

    piece_completion:
        Percentage of events that was followed until the aligner hangs, i.e
        from where on there are only misaligned or missed events.

    below_{x}_{yy}:
        Percentage of events that are aligned with an
        error smaller than x.yy seconds

    """

    HISTOGRAM_METRICS = [(_attr_name(hb), _label(hb)) for hb in HISTOGRAM_BINS]

    METRIC_NAMES = [
        ('misalign_rate', 'Misalign Rate'),
        ('miss_rate', 'Miss Rate'),
        ('piece_completion', 'Piece Completion'),
        ('avg_imprecision', 'Avg. Imprecision'),
        ('stddev_imprecision', 'Std. Dev. of Imprecision'),
        ('avg_error', 'Avg. Error'),
        ('stddev_error', 'Std. Dev. of Error'),
    ] + HISTOGRAM_METRICS

    def __init__(self, alignment, ground_truth, window=WINDOW, name=None,
                 **kwargs):
        """
        Initializes the evaluation with the given data.

        :param alignment:     computed alignment. List of tuples, 2D numpy
                              array or similar. First value is the time in
                              seconds, second value is the beat position.
        :param ground_truth:  ground truth of the aligned file. List of tuples,
                              2D numpy array of similar. First value is the
                              time in seconds, second value is the beat
                              position. It can contain the alignment positions
                              for each individual event. In this case, the
                              deviation for each event is taken into account.
        :param window:        tolerance window in seconds. Alignments further
                              apart than this value will be considered as
                              errors.
        :param name:          name of the evaluation to be displayed
        :param kwargs:        additional arguments will be ignored

        """
        # pylint: disable=unused-argument

        alignment = load_alignment(alignment)
        ground_truth = load_alignment(ground_truth)

        self.name = name
        self.window = window

        self._length = len(ground_truth)

        # compute all the evaluation metrics
        metrics = compute_metrics(
            compute_event_alignment(alignment, ground_truth),
            ground_truth,
            self.window,
            HISTOGRAM_BINS
        )

        # MAGIC! This basically corresponds to doing
        # self.misalign_rate = metrics['misalign_rate']
        # for each metric
        for attr_name, _ in self.METRIC_NAMES:
            setattr(self, attr_name, metrics[attr_name])

    def __len__(self):
        """
        :return: number of ground truth events

        """
        return self._length

    def tostring(self, histogram=False, **kwargs):
        """
        Format the evaluation metrics as a human readable string.

        :param histogram: also output the error histogram [bool]
        :param kwargs:    additional arguments will be ignored
        :return:          evaluation metrics formatted as a human readable
                          string
        """
        ret = ''
        if self.name is not None:
            ret += '%s\n  ' % self.name

        ret += 'misalign-rate: %.3f miss-rate: %.3f piece-compl.: %.3f '\
               'avg-imprecision: %.3f stddev-imprecision %.3f '\
               'avg-error: %.3f stddev-error: %.3f' %\
               (self.misalign_rate, self.miss_rate, self.piece_completion,
                self.avg_imprecision, self.stddev_imprecision, self.avg_error,
                self.stddev_error)
        # also output the histogram
        if histogram:
            ret += '\n  '
            for attr_name, lbl in self.HISTOGRAM_METRICS:
                ret += '{}: {:.2f}  '.format(lbl, getattr(self, attr_name))
        # return everything
        return ret


def _combine_metrics(eval_objects, piecewise):
    """
    Combine the metrics of the given evaluation objects.

    :param eval_objects: evaluation objects
    :param piecewise:    if 'True' all evaluation objects are weighted the same
                         if 'False' the evaluation objects are weighted by the
                         number of their events
    :return:             combined metrics

    """
    if len(eval_objects) == 0:
        raise AssertionError('cannot handle empty eval_objects list yet')
    metrics = {}
    if piecewise:
        total_weight = len(eval_objects)
    else:
        total_weight = sum(len(e) for e in eval_objects)
    for e in eval_objects:
        for name, val in e.metrics.iteritems():
            if isinstance(val, np.ndarray) or not np.isnan(val):
                weight = 1.0 if piecewise else float(len(e))
                metrics[name] = \
                    metrics.get(name, 0.) + (weight / total_weight) * val
    # return combined metrics
    return metrics


class AlignmentSumEvaluation(AlignmentEvaluation):
    """
    Class for averaging alignment evaluation scores, considering the lengths
    of the aligned pieces. For a detailed description of the available metrics,
    refer to AlignmentEvaluation.

    """

    def __init__(self, eval_objects, name=None):
        """
        Creates a new MeanEvaluation instance.

        :param eval_objects: list of evaluation objects.
        :param name:         name to be displayed

        """
        self.name = name or 'piecewise mean for %d files' % len(eval_objects)
        self.window = eval_objects[0].window
        self._length = sum(len(e) for e in eval_objects)

        metrics = _combine_metrics(eval_objects, piecewise=False)
        for attr_name, _ in self.METRIC_NAMES:
            setattr(self, attr_name, metrics[attr_name])


class AlignmentMeanEvaluation(AlignmentEvaluation):
    """
    Class for averaging alignment evaluation scores, averaging piecewise (i.e.
    ignoring the lengths of the pieces). For a detailed description of the
    available metrics, refer to AlignmentEvaluation.

    """

    def __init__(self, eval_objects, name=None):
        self.name = name or 'mean for %d files' % len(eval_objects)
        self.window = eval_objects[0].window
        self._length = len(eval_objects)

        metrics = _combine_metrics(eval_objects, piecewise=True)
        for attr_name, _ in self.METRIC_NAMES:
            setattr(self, attr_name, metrics[attr_name])


def add_parser(parser):
    """
    Add an alignment evaluation sub-parser to an existing parser.

    :param parser: existing argparse parser
    :return:       alignment evaluation sub-parser and evaluation parameter
                   group

    """
    import argparse
    from . import evaluation_io

    p = parser.add_parser(
        'alignment', help='alignment evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''
    This script evaluates pairs of files containing the true and computed
    alignments of audio files. Suffixes can be given to filter them from the
    list of files.

    Each line represents an alignment point and must have the following format
    with values being separated by whitespace:
    `audio_time score_position`

    Note that this script enforces the alignment to go monotonically forward,
    meaning that if a event 'e' is aligned at time 't_e', the following events
    'ef' will be aligned at max(t_e, t_ef).

    Lines starting with # are treated as comments and are ignored.

    ''')
    p.set_defaults(eval=AlignmentEvaluation, sum_eval=AlignmentSumEvaluation,
                   mean_eval=AlignmentMeanEvaluation)

    # files used for evaluation
    _, f = evaluation_io(p, ann_suffix='.alignment',
                         det_suffix='.alignment.txt')

    # evaluation parameters
    g = p.add_argument_group('alignment evaluation arguments')
    g.add_argument('--window', type=float, default=WINDOW,
                   help='tolerance window for misaligned notes '
                        '[seconds, default: %(default)s]')

    # add histogram option to formatting group
    f.add_argument('--histogram', action='store_true',
                   help='also output the error histogram')
    # return the sub-parser and evaluation argument group
    return p, g
