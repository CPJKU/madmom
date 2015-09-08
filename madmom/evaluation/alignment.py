#!/usr/bin/env python
# encoding: utf-8
"""
This file contains global alignment evaluation functionality.

@author: Filip Korzeniowski <filip.korzeniowski@jku.at>

"""

import re
import warnings

import numpy as np

# constants for the data format
_TIME = 0
_SCORE_POS = 1

# constants for missed events/notes
_MISSED_NOTE_VAL = np.NaN

# default settings
TOLERANCE = 0.25
HISTOGRAM_BINS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 1.]


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


def compute_metrics(event_alignment, ground_truth, tolerance, err_hist_bins):
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
    :param tolerance:       tolerance window in seconds. Alignments off less
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
        misaligned = aligned_error > tolerance

    correctly_aligned_error = np.ma.array(aligned_error, mask=misaligned)
    pc_idx = float(correctly_aligned_error.mask[::-1].argmin())
    results = {'miss_rate': missed.mean(),
               'misalign_rate': misaligned.mean(),
               'avg_imprecision': correctly_aligned_error.mean(),
               'stddev_imprecision': correctly_aligned_error.std(),
               'avg_error': aligned_error.mean(),
               'stddev_error': aligned_error.std(),
               'piece_completion': (1.0 - pc_idx /
                                    correctly_aligned_error.mask.shape[0])}

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
    results['error_hist'] = np.cumsum(
        err_hist.astype(float) / aligned_error.shape[0])

    return results


class AlignmentEvaluation(object):
    """
    Alignment evaluation class for beat-level alignments.
    Beat-level aligners output beat positions for points in time,
    rather than computing a time step for each individual event in the
    score.
    """
    METRIC_NAMES = [
        ('misalign_rate', 'Misalign Rate'),
        ('miss_rate', 'Miss Rate'),
        ('piece_completion', 'Piece Completion'),
        ('avg_imprecision', 'Avg. Imprecision'),
        ('stddev_imprecision', 'Std. Dev. of Imprecision'),
        ('avg_error', 'Avg. Error'),
        ('stddev_error', 'Std. Dev. of Error'),
    ]

    def __init__(self, alignment, ground_truth,
                 tolerance=TOLERANCE, err_hist_bins=HISTOGRAM_BINS):
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
        :param tolerance:     tolerance window in seconds. Alignments further
                              apart than this value will be considered as
                              errors.
        :param err_hist_bins: error bounds for which the cumulative histogram
                              of absolute errors will be computed (e.g.
                              [0.1, 0.3] will give the percentage of events
                              aligned with an error smaller than 0.1 and 0.3)

        """

        self.alignment = alignment
        self.ground_truth = ground_truth
        self.tolerance = tolerance
        self.error_histogram_bins = err_hist_bins

        self._metrics = None
        self._saved_event_alignment = None

    def _event_alignment(self):
        """
        2d numpy array of event alignments corresponding to the elements
        present in the ground truth data.

        """
        if self._saved_event_alignment is None:
            self._saved_event_alignment = compute_event_alignment(
                self.alignment, self.ground_truth)
        return self._saved_event_alignment

    @property
    def metrics(self):
        """
        Most of the evaluation metrics presented in Cont's paper contained in
        a dictionary, plus an error histogram.

        """
        if self._metrics is None:
            self._metrics = compute_metrics(self._event_alignment(),
                                            self.ground_truth,
                                            self.tolerance,
                                            self.error_histogram_bins)
        return self._metrics

    @property
    def miss_rate(self):
        """
        Percentage of missed events (events that exist in the reference score,
        but are not reported.

        """
        return self.metrics['miss_rate']

    @property
    def misalign_rate(self):
        """
        Percentage of misaligned events (events with an alignment that is off
        by more than defined in the threshold).

        """
        return self.metrics['misalign_rate']

    @property
    def avg_imprecision(self):
        """Average alignment error of non-misaligned events."""
        return self.metrics['avg_imprecision']

    @property
    def stddev_imprecision(self):
        """Standard deviation of alignment error of non-misaligned events."""
        return self.metrics['stddev_imprecision']

    @property
    def avg_error(self):
        """Average alignment error."""
        return self.metrics['avg_error']

    @property
    def stddev_error(self):
        """Standard deviation of alignment error."""
        return self.metrics['stddev_error']

    @property
    def piece_completion(self):
        """
        Percentage of events that was followed until the aligner hangs, i.e
        from where on there are only misaligned or missed events.

        """
        return self.metrics['piece_completion']

    @property
    def error_histogram(self):
        """
        Cumulative histogram of absolute alignment error. For bounds see
        error_histogram_bins.
        """
        return self.metrics['error_hist']

    def print_errors(self, verbose=False):
        """
        Print errors.

        :param verbose: output error histogram
        """
        errs = 'misalign-rate: %.3f miss-rate: %.3f piece-compl.: %.3f '\
               'avg-imprecision: %.3f stddev-imprecision %.3f '\
               'avg-error: %.3f stddev-error: %.3f' %\
               (self.metrics['misalign_rate'],
                self.metrics['miss_rate'],
                self.metrics['piece_completion'],
                self.metrics['avg_imprecision'],
                self.metrics['stddev_imprecision'],
                self.metrics['avg_error'],
                self.metrics['stddev_error'])

        if verbose:
            # hacky way to create the format string. first, we
            # convert the bins to the desired string format
            bins_str = map('{:.2f}'.format, self.error_histogram_bins)
            # then, we join the stringified bins with formatting instructions
            # for the histogram values
            hist_str = '<' + ': {:.2f}  <'.join(bins_str) + ': {:.2f}'
            # then, we insert the histogram at the positions specified above
            hist_str = hist_str.format(*self.metrics['error_hist'])
            errs += '\n' + hist_str

        return errs


class MeanAlignmentEvaluation(AlignmentEvaluation):
    """
    Class for averaging alignment evaluation scores.

    """
    def __init__(self, piecewise=True):
        """
        :param piecewise:  average piecewise (each piece has the same weight)
        """
        self.piecewise = piecewise
        self.total_weight = 0.0
        self.evals = []
        self._metrics = None

        super(MeanAlignmentEvaluation, self).__init__(None, None)

    def __len__(self):
        """Number of averaged evaluations."""
        return len(self.evals)

    def append(self, other):
        """
        Add another evaluation to average.

        :param other: AlignmentEvaluation to add

        """
        weight = 1.0 if self.piecewise else len(other.ground_truth)
        self.total_weight += weight
        self.evals.append((weight, other.metrics))

        # invalidate any computed metrics
        self._metrics = None

    @property
    def metrics(self):
        """
        Alignment evaluation metrics averaged over all added individual
        evaluations.

        """
        if len(self) == 0:
            raise RuntimeError('Cannot compute mean evaluation on empty object')

        if self._metrics is None:
            self._metrics = {}

            for weight, sf_eval in self.evals:
                for name, val in sf_eval.iteritems():
                    if isinstance(val, np.ndarray) or not np.isnan(val):
                        self._metrics[name] = (self._metrics.get(name, 0.) +
                                               weight / self.total_weight * val)

        return self._metrics


def parse_args():
    """
    Create a parser and parse the arguments.

    :return: the parsed arguments

    """
    import argparse
    from . import evaluation_in, evaluation_out

    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    This script evaluates pairs of files containing the true and computed
    alignments of audio files. Suffixes can be given to filter them
    from the list of files.

    Each line represents an alignment point and must have the following format
    with values being separated by whitespace:
    `audio_time score_position`

    Note that this script enforces the alignment to go monotonically forward,
    meaning that if a event 'e' is aligned at time 't_e', the following events
    'ef' will be aligned at max(t_e, t_ef).

    Lines starting with # are treated as comments and are ignored.

    NOTE: Due tue implementation limitations, --tex activates the output of
          an error histogram. This will change in the future!
    """)

    evaluation_in(p, ann_suffix='.alignment', det_suffix='.aligned')
    out_opts = evaluation_out(p)
    out_opts.add_argument('--histogram', action='store_true',
                          help='Output error histogram [default: %(default)s]')

    g = p.add_argument_group('evaluation arguments')

    g.add_argument('--tolerance', type=float, default=TOLERANCE,
                   help='Tolerance threshold for misaligned notes '
                        '[seconds, default: %(default)s]')

    g.add_argument('--piecewise', action='store_true',
                   help='Combine metrics piecewise [default: %(default)s]')

    args = p.parse_args()
    # output the args
    if args.verbose >= 2:
        print args
    if args.quiet:
        warnings.filterwarnings("ignore")

    return p.parse_args()


def main():
    """
    Simple alignment evaluation.

    """
    import os
    from madmom.utils import search_files, match_file

    args = parse_args()

    # get ground truth and computed alignment files
    if args.det_dir is None:
        args.det_dir = args.files
    if args.ann_dir is None:
        args.ann_dir = args.files
    det_files = search_files(args.det_dir, args.det_suffix)
    ann_files = search_files(args.ann_dir, args.ann_suffix)
    # quit if no files are found
    if len(ann_files) == 0:
        print "no files to evaluate. exiting."
        exit()

    mean_eval = MeanAlignmentEvaluation(args.piecewise)
    eval_output = args.output_formatter(mean_eval.METRIC_NAMES)

    for ann_file in ann_files:
        ground_truth = np.loadtxt(ann_file)
        matches = match_file(ann_file, det_files,
                             args.ann_suffix, args.det_suffix)

        if len(matches) > 1:
            # exit if multiple detections were found
            raise SystemExit("multiple detections for %s found." % ann_file)
        elif len(matches) == 0:
            # output a warning if no detections were found
            warnings.warn(" can't find detections for %s." % ann_file)
            # but continue and assume no detections
            alignment = np.array([[0, -1]])

        else:
            # load the detections
            alignment = np.loadtxt(matches[0])

        e = AlignmentEvaluation(
            np.atleast_2d(alignment),
            np.atleast_2d(ground_truth),
            args.tolerance)

        if args.verbose:
            eval_output.add_eval(os.path.basename(ann_file), e,
                                 verbose=args.histogram)

        mean_eval.append(e)

    eval_output.add_eval('mean for %i file(s)' % len(mean_eval), mean_eval,
                         verbose=args.histogram)
    print eval_output


if __name__ == '__main__':
    main()
