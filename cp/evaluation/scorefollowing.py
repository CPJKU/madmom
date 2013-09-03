#!/usr/bin/env python
# encoding: utf-8
"""
Copyright (c) 2013, Filip Korzeniowski <filip.korzeniowski@jku.at>
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
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np

# constants for the data format
_TIME = 0
_SCORE_POS = 1

# constants for missed events/notes
_MISSED_NOTE_TIME = np.NaN
_MISSED_NOTE_POS = np.NaN
_MISSED_NOTE_TEMPO = np.NaN


def compute_event_alignment(alignment, ground_truth):
    """
    This function finds the alignment outputs corresponding to each ground
    truth alignment. In general, the score follower will output more alignment
    positions than notes in the score, e.g. if it is designed to output the
    current alignment at constant intervals.

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

    :return: 2d numpy array of the same size as ground_truth, with each
             row representing the alignment of the corresponding ground truth
             element.
    """
    #TODO: Find out why this is set to this value, and find a more sensible
    #      way to deal with that
    DUMMY_TIME = 10000000000000

    # first a dummy event at the very end of the alignment is added to be
    # able to process score events with were not reached by the tracker
    dummy = [[DUMMY_TIME, _MISSED_NOTE_POS, _MISSED_NOTE_TEMPO]]
    alignment = np.concatenate((alignment, dummy))

    # find the spots where the alignment passes the score

    gt_pos = ground_truth[:, _SCORE_POS]
    al_pos = alignment[:, _SCORE_POS]

    # t will become a 2d-matrix with each cell indicating if the proposed
    # alignment (horizontal axis) is greater or equal the ground truth
    # alignment (vertical axis)
    t = ~(al_pos < gt_pos[:, np.newaxis])

    # now we get the indices of the 'true' cells
    t = t.nonzero()

    # the first array in the tuple t are the indices of the ground truth
    # alignment positions, the second one are the ones of the proposed
    # alignment positions.
    # we extract only the first ones, since the tracker can't move forward!
    al_idxs = t[1][np.nonzero(np.r_[1, np.diff(t[0])])]

    # now, the number of indexes in the alignment should correspond
    # to the number of aligned positions in the ground truth
    assert(len(al_idxs) == len(ground_truth))

    alignment = alignment[al_idxs]
    alignment[np.isnan(alignment[:, _SCORE_POS])] = _MISSED_NOTE_TIME

    return alignment


def compute_cont_metrics(event_alignment, ground_truth, window):
    """
    This function computes the evaluation metrics based on the paper
    "Evaluation of Real-Time Audio-to-Score Alignment" by Arshia Cont et al.

    :param event_alignment: sequence alignment as computed by the score
        follower. List of tuples, 2d numpy array or similar, where the first
        column is the alignment time in seconds and the second column the
        position in beats. Needs to be the same length as ground_truth, hence
        for each element in the ground truth the corresponding alignment has
        to be available. You can use the "compute_event_alignment" function
        to compute this.
    :param ground_truth: Ground truth of the aligned performance.
        List of tuples, 2d numpy array of similar. First value is the time in
        seconds, second value is the beat position. It can contain the
        alignment positions for each individual note. In this case, the
        deviation for each note is taken into account.
    :param window: Tolerance window in seconds. Alignments off less than this
        amount from the ground truth will be considered correct.

    :return: A dictionary containing (some) of the metrics described in the
        paper mentioned above.
    """

    abs_error = np.abs(event_alignment[:, _TIME] - ground_truth[:, _TIME])
    misaligned = abs_error > window
    missed = np.isnan(abs_error)

    # TODO: Check out why the following computation is so complicated and
    #       if diff and cumsum dont cancel each other out

    # consider the unlikely case that EVERYTHING was missed or misaligned
    mi_ma_series = np.diff((missed | misaligned).cumsum())
    if (mi_ma_series).all():
        completed_idx = -1
    else:
        completed_idx = np.flatnonzero(mi_ma_series - 1)[-1] + 1

    return {'miss_rate': float(missed.sum()) / len(ground_truth),
            'misalign_rate': float(misaligned.sum()) / len(ground_truth),
            'avg_imprecision': np.mean(abs_error[~(misaligned | missed)]),
            'stddev_imprecision': np.std(abs_error[~(misaligned | missed)]),
            'avg_error': np.mean(abs_error[~missed]),
            'stddev_error': np.std(abs_error[~missed]),
            'piece_completion': float(completed_idx + 1) / len(ground_truth)}


class ScoreFollowingEvaluation(object):
    """
    Score following evaluation class for beat-level score followers.
    Beat-level score followers output beat positions for points in time,
    rather than computing a timestep for each individual note in the
    score.
    """

    def __init__(self, alignment, ground_truth, window=0.25):
        """
        Inits the evaluation with the given data and window threshold.

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
            self._event_alignment = compute_event_alignment(self.alignment, self.ground_truth)
        return self._event_alignment

    @property
    def cont_metrics(self):
        """
        Most of the evaluation metrics presented in Cont's paper contained
        in a dictionary.
        """
        if self._cont_metrics is None:
            self._cont_metrics = compute_cont_metrics(self.event_alignment, self.ground_truth, self.tolerance)
        return self._cont_metrics

    @property
    def miss_rate(self):
        return self.cont_metrics['miss_rate']

    @property
    def misalign_rate(self):
        return self.cont_metrics['misalign_rate']

    @property
    def avg_imprecision(self):
        return self.cont_metrics['avg_imprecision']

    @property
    def stddev_imprecision(self):
        return self.cont_metrics['stddev_imprecision']

    @property
    def avg_error(self):
        return self.cont_metrics['avg_error']

    @property
    def stddev_error(self):
        return self.cont_metrics['stddev_error']

    @property
    def piece_completion(self):
        return self.cont_metrics['piece_completion']

    def print_eval(self, table_row=False):
        metrics = self.cont_metrics
        if table_row:
            fields = ['misalign_rate', 'miss_rate', 'piece_completion',
                      'avg_imprecision', 'stddev_imprecision',
                      'avg_error', 'stddev_error']

            print '%f %f %f %f %f %f %f' % tuple(metrics[f] for f in fields)

        else:
            print 'Misalign rate: %f' % metrics['misalign_rate']
            print 'Miss rate: %f' % metrics['miss_rate']
            print 'Piece completion: %f' % metrics['piece_completion']
            print 'Average imprecision: %f' % metrics['avg_imprecision']
            print 'Std. Dev. of imprecision: %f' % metrics['stddev_imprecision']
            print 'Average error: %f' % metrics['avg_error']
            print 'Std.Dev. of error: %f' % metrics['stddev_error']


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Performs a numerical analysis'
                                     ' of a score/performance match')

    parser.add_argument('-gt', '--ground-truth', required=True,
                        help='Data-file containing the ground-truth alignment'
                             ' of the performance',
                        dest='ground_truth_filename')

    parser.add_argument('-s', '--segmentation', required=True,
                        help='Data-file containing the proposed alignment'
                             ' of the performance',
                        dest='segmentation_filename')

    parser.add_argument('-t', '--tolerance', type=int,
                        help='Tolerance in ms - threshold for misaligned notes',
                        dest='tolerance', default=300)

    parser.add_argument('-to', '--table-output', action='store_const',
                        const=True, default=False,
                        help='Enable output as a row with a space seperator',
                        dest='table_output')

    # TODO: unclear, for what this option was used...
    # parser.add_argument('-co', '--condensed-output',
    #                     help='File where to save a condensed output of the '
    #                          'segmentation. Only alignment times of events in '
    #                          'the ground truth will be stored.',
    #                     dest='condensed_output_filename')

    return parser.parse_args()


def main():
    args = parse_arguments()

    ground_truth = np.loadtxt(args.ground_truth_filename)
    alignment = np.loadtxt(args.segmentation_filename)

    window = float(args.tolerance) / 1000
    sf_eval = ScoreFollowingEvaluation(alignment, ground_truth, window)

    # if args.condensed_output_filename is not None:
    #     np.savetxt(args.condensed_output_filename, sf_eval.event_alignment)

    sf_eval.print_eval(args.table_output)


if __name__ == '__main__':
    main()
