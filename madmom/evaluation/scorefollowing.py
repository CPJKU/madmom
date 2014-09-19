#!/usr/bin/env python
# encoding: utf-8
"""
This file contains score following evaluation functionality.

@author: Filip Korzeniowski <filip.korzeniowski@jku.at>

"""

import numpy as np

# constants for the data format
_TIME = 0
_SCORE_POS = 1

# constants for missed events/notes
_MISSED_NOTE_VAL = np.NaN


def compute_event_alignment(alignment, ground_truth):
    """
    This function finds the alignment outputs corresponding to each ground
    truth alignment. In general, the score follower will output more alignment
    positions than notes in the score, e.g. if it is designed to output the
    current alignment at constant intervals.

    :param alignment:    The score follower's resulting alignment.
                         2D NumPy array, first value is the time in seconds,
                         second value is the beat position.
    :param ground_truth: Ground truth of the aligned performance.
                         2D numpy array of similar. First value is the time in
                         seconds, second value is the beat position. It can
                         contain the alignment positions for each individual
                         note. In this case, the deviation for each note is
                         taken into account.

    :return: 2D numpy array of the same size as ground_truth, with each
             row representing the alignment of the corresponding ground truth
             element.
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


def compute_cont_metrics(event_alignment, ground_truth, window):
    """
    This function computes the evaluation metrics based on the paper
    "Evaluation of Real-Time Audio-to-Score Alignment" by Arshia Cont et al.

    :param event_alignment: sequence alignment as computed by the score
        follower. 2D numpy array, where the first column is the
        alignment time in seconds and the second column the position in beats.
        Needs to be the same length as ground_truth, hence for each element in
        the ground truth the corresponding alignment has to be available. You
        can use the "compute_event_alignment" function to compute this.
    :param ground_truth: Ground truth of the aligned performance.
        2D numpy array, first value is the time in seconds, second value is the
        beat position. It can contain the alignment positions for each
        individual note. In this case, the deviation for each note is taken
        into account.
    :param window: Tolerance window in seconds. Alignments off less than this
        amount from the ground truth will be considered correct.

    :return: A dictionary containing (some) of the metrics described in the
        paper mentioned above.
    """
    abs_error = np.abs(event_alignment[:, _TIME] - ground_truth[:, _TIME])
    missed = np.isnan(abs_error)
    aligned_error = np.ma.array(abs_error, mask=missed)
    # for some numpy versions, the following prints a invalid value warning,
    # although NaNs are masked - code still works.
    misaligned = aligned_error > window
    correctly_aligned_error = np.ma.array(aligned_error, mask=misaligned)

    pc_idx = float(correctly_aligned_error.mask[::-1].argmin())
    results = {'miss_rate': float(missed.sum()) / len(ground_truth),
               'misalign_rate': float(misaligned.sum()) / len(ground_truth),
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

    return results


class ScoreFollowingEvaluation(object):
    """
    Score following evaluation class for beat-level score followers.
    Beat-level score followers output beat positions for points in time,
    rather than computing a time step for each individual note in the
    score.
    """
    _FIELDS_SORTED = ['misalign_rate', 'miss_rate', 'piece_completion',
                      'avg_imprecision', 'stddev_imprecision', 'avg_error',
                      'stddev_error']

    SORTED_FIELD_NAMES = ['Misalign Rate', 'Miss Rate', 'Piece Completion',
                          'Avg. Imprecision', 'Imprecision StdDev',
                          'Avg. Error', 'Error StdDev']

    def __init__(self, alignment, ground_truth, window=0.25):
        """
        Initializes the evaluation with the given data and window threshold.

        :param alignment: The score follower's resulting alignment.
                          List of tuples, 2d numpy array or similar. First
                          value is the time in seconds, second value is the
                          beat position.
        :param ground_truth: Ground truth of the aligned performance.
                          List of tuples, 2d numpy array of similar. First
                          value is the time in seconds, second value is the
                          beat position. It can contain the alignment
                          positions for each individual note. In this case,
                          the deviation for each note is taken into account.
        :param window: Tolerance window in seconds. Alignments further apart
                       than this value will be considered as errors.
        """

        self.alignment = alignment
        self.ground_truth = ground_truth
        self.tolerance = window

        self._cont_metrics = None
        self._event_alignment = None

    @property
    def event_alignment(self):
        """
        2d numpy array of event alignments corresponding to the elements
        present in the ground truth data.

        """
        if self._event_alignment is None:
            self._event_alignment = compute_event_alignment(self.alignment,
                                                            self.ground_truth)
        return self._event_alignment

    @property
    def cont_metrics(self):
        """
        Most of the evaluation metrics presented in Cont's paper contained in
        a dictionary.

        """
        if self._cont_metrics is None:
            self._cont_metrics = compute_cont_metrics(self.event_alignment,
                                                      self.ground_truth,
                                                      self.tolerance)
        return self._cont_metrics

    @property
    def cont_metrics_sorted(self):
        """
        Most of the evaluation metrics presented in Cont's paper contained in
        a list, sorted as defined in self._FIELDS_SORTED.

        """
        return [self.cont_metrics[f]
                for f in ScoreFollowingEvaluation._FIELDS_SORTED]

    @property
    def miss_rate(self):
        """
        Percentage of missed events (events that exist in the reference score,
        but are not reported.

        """
        return self.cont_metrics['miss_rate']

    @property
    def misalign_rate(self):
        """
        Percentage of misaligned events (events with an alignment that is off
        by more than defined in the threshold).

        """
        return self.cont_metrics['misalign_rate']

    @property
    def avg_imprecision(self):
        """Average alignment error of non-misaligned events."""
        return self.cont_metrics['avg_imprecision']

    @property
    def stddev_imprecision(self):
        """Standard deviation of alignment error of non-misaligned events."""
        return self.cont_metrics['stddev_imprecision']

    @property
    def avg_error(self):
        """Average alignment error."""
        return self.cont_metrics['avg_error']

    @property
    def stddev_error(self):
        """Standard deviation of alignment error."""
        return self.cont_metrics['stddev_error']

    @property
    def piece_completion(self):
        """
        Percentage of events that was followed until the aligner hangs, i.e
        from where on there are only misaligned or missed events.

        """
        return self.cont_metrics['piece_completion']

    def __str__(self):
        return 'Misalign rate: %f\n'\
               'Miss rate: %f\n'\
               'Piece completion: %f\n'\
               'Average imprecision: %f\n'\
               'Std.Dev. of imprecision: %f\n'\
               'Average error: %f\n'\
               'Std.Dev. of error: %f\n' %\
               (self.cont_metrics['misalign_rate'],
                self.cont_metrics['miss_rate'],
                self.cont_metrics['piece_completion'],
                self.cont_metrics['avg_imprecision'],
                self.cont_metrics['stddev_imprecision'],
                self.cont_metrics['avg_error'],
                self.cont_metrics['stddev_error'])

    def as_csv_row(self):
        """Format the metrics as comma separated values."""
        return ','.join([str(metric) for metric in self.cont_metrics_sorted])


def parse_arguments():
    """
    Create a parser and parse the arguments.

    :return: the parsed arguments

    """
    import argparse
    parser = argparse.ArgumentParser(description='Performs a numerical '
                                     ' analysis of a score/performance match')

    parser.add_argument('-gt', '--ground-truth', required=True,
                        help='Data-file containing the ground-truth alignment'
                             ' of the performance',
                        dest='ground_truth_filename')

    parser.add_argument('-s', '--segmentation', required=True,
                        help='Data-file containing the proposed alignment'
                             ' of the performance',
                        dest='segmentation_filename')

    parser.add_argument('-t', '--tolerance', type=int,
                        help='Tolerance threshold for misaligned notes [ms]',
                        dest='tolerance', default=300)

    parser.add_argument('-to', '--table-output', action='store_const',
                        const=True, default=False,
                        help='Enable output as a row with a space separator',
                        dest='table_output')

    return parser.parse_args()


def main():
    """
    Simple score following evaluation.

    """
    args = parse_arguments()

    ground_truth = np.loadtxt(args.ground_truth_filename)
    alignment = np.loadtxt(args.segmentation_filename)

    window = float(args.tolerance) / 1000
    sf_eval = ScoreFollowingEvaluation(alignment, ground_truth, window)

    if args.table_output:
        print sf_eval.as_csv_row()
    else:
        print sf_eval


if __name__ == '__main__':
    main()
