# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains tempo evaluation functionality.

"""

from __future__ import absolute_import, division, print_function

import warnings
import numpy as np

from . import EvaluationMixin, MeanEvaluation, evaluation_io
from madmom.io import load_tempo

# default tempo evaluation values
TOLERANCE = 0.04
DOUBLE = True
TRIPLE = True


# this evaluation function can evaluate multiple tempi simultaneously
def tempo_evaluation(detections, annotations, tolerance=TOLERANCE):
    """
    Calculate the tempo P-Score, at least one or both tempi correct.

    Parameters
    ----------
    detections : list of tuples or numpy array
        Detected tempi (rows, first column) and their relative strengths
        (second column).
    annotations : list or numpy array
        Annotated tempi (rows, first column) and their relative strengths
        (second column).
    tolerance : float, optional
        Evaluation tolerance (max. allowed deviation).

    Returns
    -------
    pscore : float
        P-Score.
    at_least_one : bool
        At least one tempo correctly identified.
    all : bool
        All tempi correctly identified.

    Notes
    -----
    All given detections are evaluated against all annotations according to the
    relative strengths given. If no strengths are given, evenly distributed
    strengths are assumed. If the strengths do not sum to 1, they will be
    normalized.

    References
    ----------
    .. [1] M. McKinney, D. Moelants, M. Davies and A. Klapuri,
           "Evaluation of audio beat tracking and music tempo extraction
           algorithms",
           Journal of New Music Research, vol. 36, no. 1, 2007.

    """
    # neither detections nor annotations are given
    if len(detections) == 0 and len(annotations) == 0:
        # perfect result
        return 1., True, True
    # either detections or annotations are empty
    if len(detections) == 0 or len(annotations) == 0:
        # worst result
        return 0., False, False
    # tolerance must be greater than 0
    if float(tolerance) <= 0:
        raise ValueError('tolerance must be greater than 0')
    # make sure the annotations and detections have a float dtype
    detections = np.asarray(detections, dtype=np.float)
    annotations = np.asarray(annotations, dtype=np.float)
    # extract the detected tempi, ignore the strengths
    if detections.ndim == 2:
        detections = detections[:, 0]
    # extract the annotated tempi and strengths
    strengths = []
    if annotations.ndim == 2:
        strengths = annotations[:, 1]
        annotations = annotations[:, 0]
    # strengths must sum up to 1
    strengths_sum = np.sum(strengths)
    if strengths_sum == 0:
        # uniformly distribute strengths
        warnings.warn('no annotated tempo strengths given, assuming a uniform '
                      'distribution')
        strengths = np.ones_like(annotations) / float(len(annotations))
    elif strengths_sum != 1:
        # normalize strengths
        warnings.warn('annotated tempo strengths do not sum to 1, normalizing')
        strengths /= float(strengths_sum)
    # test all detected tempi against all annotated tempi
    errors = np.abs(1 - (detections[:, np.newaxis] / annotations))
    # correctly identified annotation tempi
    correct = np.asarray(np.sum(errors <= tolerance, axis=0), np.bool)
    # the P-Score is the sum of the strengths of the correctly identified tempi
    pscore = np.sum(strengths[correct])
    # return the scores
    # TODO: also return the errors?
    return pscore, correct.any(), correct.all()


# basic tempo evaluation
class TempoEvaluation(EvaluationMixin):
    """
    Tempo evaluation class.

    Parameters
    ----------
    detections : str, list of tuples or numpy array
        Detected tempi (rows) and their strengths (columns).
        If a file name is given, load them from this file.
    annotations : str, list or numpy array
        Annotated ground truth tempi (rows) and their strengths (columns).
        If a file name is given, load them from this file.
    tolerance : float, optional
        Evaluation tolerance (max. allowed deviation).
    double : bool, optional
        Include double and half tempo variations.
    triple : bool, optional
        Include triple and third tempo variations.
    sort : bool, optional
        Sort the tempi by their strengths (descending order).
    max_len : bool, optional
        Evaluate at most `max_len` tempi.
    name : str, optional
        Name of the evaluation to be displayed.

    Notes
    -----
    For P-Score, the number of detected tempi will be limited to the number
    of annotations (if not further limited by `max_len`).
    For Accuracy 1 & 2 only one detected tempo is used. Depending on `sort`,
    this can be either the first or the strongest one.

    """
    METRIC_NAMES = [
        ('pscore', 'P-score'),
        ('any', 'one tempo correct'),
        ('all', 'both tempi correct'),
        ('acc1', 'Accuracy 1'),
        ('acc2', 'Accuracy 2')
    ]

    def __init__(self, detections, annotations, tolerance=TOLERANCE,
                 double=DOUBLE, triple=TRIPLE, sort=True, max_len=None,
                 name=None, **kwargs):
        # pylint: disable=unused-argument
        # load the tempo detections and annotations
        detections = load_tempo(detections, sort=sort, max_len=max_len)
        annotations = load_tempo(annotations, sort=sort, max_len=max_len)
        # TODO: truncate the detections to the length of the annotations?
        # evaluate P-score with all tempo annotations, but truncate the
        # detections to the same length
        ann = load_tempo(annotations, sort=sort)
        det = load_tempo(detections, sort=sort, max_len=(len(ann) or None))
        self.pscore, self.any, self.all = tempo_evaluation(det, ann, tolerance)
        # evaluate acc1 only with the strongest/first tempo
        # Note: the strengths are irrelevant or acc1 & acc2 calculation
        #       the accuracies correspond to either any or all tempi
        # TODO: allow a different max_len here?
        det = load_tempo(detections, sort=sort, max_len=1)
        ann = load_tempo(annotations, sort=sort, max_len=1)
        self.acc1 = tempo_evaluation(det, ann, tolerance)[1]
        # evaluate acc2 like acc1 but include double/half & triple/third tempi
        tempi = ann[:, 0].copy()
        ann = tempi.copy()
        if double:
            ann = np.hstack((ann, tempi * 2., tempi / 2.))
        if triple:
            ann = np.hstack((ann, tempi * 3., tempi / 3.))
        # accuracy doesn't need strengths, so we just add fake strengths
        ann = np.vstack((ann, np.ones_like(ann) / len(ann))).T
        self.acc2 = tempo_evaluation(det, ann, tolerance)[1]
        # save the name
        self.name = name

    def __len__(self):
        return 1

    def tostring(self, **kwargs):
        """
        Format the evaluation metrics as a human readable string.

        Returns
        -------
        str
            Evaluation metrics formatted as a human readable string.

        """
        # pylint: disable=unused-argument

        ret = ''
        if self.name is not None:
            ret += '%s\n  ' % self.name
        ret += 'pscore=%.3f (one tempo: %.3f, all tempi: %.3f) ' \
               'acc1=%.3f acc2=%.3f' % \
               (self.pscore, self.any, self.all, self.acc1, self.acc2)
        return ret

    def __str__(self):
        return self.tostring()


class TempoMeanEvaluation(MeanEvaluation):
    """
    Class for averaging tempo evaluation scores.

    """
    METRIC_NAMES = TempoEvaluation.METRIC_NAMES

    @property
    def pscore(self):
        """P-Score."""
        return np.nanmean([e.pscore for e in self.eval_objects])

    @property
    def any(self):
        """At least one tempo correct."""
        return np.nanmean([e.any for e in self.eval_objects])

    @property
    def all(self):
        """All tempi correct."""
        return np.nanmean([e.all for e in self.eval_objects])

    @property
    def acc1(self):
        """Accuracy 1."""
        return np.nanmean([e.acc1 for e in self.eval_objects])

    @property
    def acc2(self):
        """Accuracy 2."""
        return np.nanmean([e.acc2 for e in self.eval_objects])

    def tostring(self, **kwargs):
        """
        Format the evaluation metrics as a human readable string.

        Returns
        -------
        str
            Evaluation metrics formatted as a human readable string.

        """
        ret = ''
        if self.name is not None:
            ret += '%s\n  ' % self.name
        ret += 'pscore=%.3f (one tempo: %.3f, all tempi: %.3f) ' \
               'acc1=%.3f acc2=%.3f' % \
               (self.pscore, self.any, self.all, self.acc1, self.acc2)
        return ret

    def __str__(self):
        return self.tostring()


def add_parser(parser):
    """
    Add a tempo evaluation sub-parser to an existing parser.

    Parameters
    ----------
    parser : argparse parser instance
        Existing argparse parser object.

    Returns
    -------
    sub_parser : argparse sub-parser instance
        Tempo evaluation sub-parser.
    parser_group : argparse argument group
        Tempo evaluation argument group.

    """
    import argparse
    # add tempo evaluation sub-parser to the existing parser
    p = parser.add_parser(
        'tempo', help='tempo evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''
    This program evaluates pairs of files containing the tempo annotations and
    detections. Suffixes can be given to filter them from the list of files.

    A single line represents the tempi and their relative strength and must
    have the following format with values being separated by whitespace:
    `tempo_one tempo_two relative_strength`

    Lines starting with # are treated as comments and are ignored.

    For P-Score evaluation as many tempi detections are used as tempo
    annotations are given.

    For Accuracy 1 & 2 evaluation, only the strongest (if strengths are given)
    or the first tempo is used.

    ''')
    # set defaults
    p.set_defaults(eval=TempoEvaluation, mean_eval=TempoMeanEvaluation,
                   sum_eval=None)
    # file I/O
    evaluation_io(p, ann_suffix='.bpm', det_suffix='.bpm.txt')
    # evaluation parameters
    g = p.add_argument_group('tempo manipulation arguments')
    g.add_argument('--tolerance', type=float, action='store',
                   default=TOLERANCE,
                   help='tolerance for tempo detection '
                        '[default=%(default).3f]')
    g.add_argument('--no_double', dest='double', action='store_false',
                   help='do not include double/half tempo evaluation')
    g.add_argument('--no_triple', dest='triple', action='store_false',
                   help='do not include triple/third tempo evaluation')
    # how many and which of the tempi should be evaluated?
    g.add_argument('--no_sort', dest='sort', action='store_false',
                   help='do not sort the tempi by strength [default: sort '
                        'them by strength]')
    # TODO: add option to evaluate any other than the default number of tempi?
    # g.add_argument('--num', dest='max_len', action='store', type=int,
    #                help='evaluate NUM tempi [default: evaluate only the '
    #                     'first (after sorting them)]')
    # return the sub-parser and evaluation argument group
    return p, g
