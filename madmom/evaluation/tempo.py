#!/usr/bin/env python
# encoding: utf-8
"""
This file contains tempo evaluation functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import warnings
import numpy as np

from ..utils import open


def load_tempo(filename, split_value=1.):
    """
    Load tempo information from a file.

    :param filename:    name of the file
    :param split_value: values > split_value are interpreted as tempi in bpm,
                        values <= split_value are interpreted as strengths
    :return:            tuple with arrays containing the tempi and their
                        relative strengths (ordered by descending strength)

    Note: All tempi and strength information must be present in a single line.

    """
    # read in the tempi
    with open(filename, 'rb') as f:
        # init values
        values = np.zeros(0)
        # TODO: what to do if more information is in the file?
        #       right now we only keep the last line...
        for line in f:
            if line.startswith('#'):
                # ignore comments
                continue
            elif line:
                # non-empty line
                values = np.asarray(line.rstrip().split(None), dtype=float)
        # split the values according to their values into tempi and strengths
        # TODO: this is kind of hack-ish, find a better solution
        tempi = values[values > split_value]
        strengths = values[values <= split_value]
    # format the relative strengths
    if len(tempi) - len(strengths) == 1:
        # one relative strength is missing, add a calculated one
        strengths = np.append(strengths, 1. - np.sum(strengths))
    # tempi and strengths must have same length
    if len(strengths) > 0:
        if len(tempi) != len(strengths):
            raise ValueError('tempi and strengths must have same length')
    # order the tempi according to their strengths
    sort_idx = strengths.argsort()[::-1]
    # return
    return tempi[sort_idx], strengths[sort_idx]


# this evaluation function can evaluate multiple tempi simultaneously
def tempo_evaluation(detections, annotations, strengths, tolerance):
    """
    Calculate the tempo P-Score.

    :param detections:  array with (multiple) tempi [bpm]
    :param annotations: array with (multiple) tempi [bpm]
    :param strengths:   array with the relative strengths of the tempi
    :param tolerance:   evaluation tolerance
    :return:            p-score, at least one tempo correctly identified, all
                        tempi correctly identified (float, bool, bool)

    Note: All given detections are evaluated against all annotations according
          to the relative strengths given. If no strengths are given, evenly
          distributed strengths are assumed.

    "Evaluation of audio beat tracking and music tempo extraction algorithms"
    M. McKinney, D. Moelants, M. Davies and A. Klapuri
    Journal of New Music Research, vol. 36, no. 1, pp. 1–16, 2007.

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
    if tolerance <= 0:
        raise ValueError("tolerance must be greater than 0")
    # if no annotation strengths are given, distribute evenly
    if strengths is None:
        strengths = np.ones_like(annotations)
    if len(annotations) != len(strengths):
        raise ValueError("Annotations and strengths must have same length.")
    # strengths must sum up to 1
    strengths_sum = np.sum(strengths)
    if strengths_sum == 0:
        # create evenly distributed strengths
        strengths = np.ones_like(annotations) / float(len(annotations))
    elif strengths_sum != 1:
        # normalize strengths
        strengths /= float(strengths_sum)
    # test all detected tempi against all annotated tempi
    errors = np.abs(1 - (detections[:, np.newaxis] / annotations))
    # correctly identified annotation tempi
    correct = np.asarray(np.sum(errors <= tolerance, axis=0), np.bool)
    # the p-score is the sum of the strengths of the correctly identified tempi
    return np.sum(strengths[correct]), correct.any(), correct.all()


TOLERANCE = 0.04
DOUBLE = True
TRIPLE = True


# basic tempo evaluation
class TempoEvaluation(object):
    """
    Tempo evaluation class.

    """
    METRIC_NAMES = [
        ('pscore', 'P-score'),
        ('any', 'one tempo'),
        ('all', 'both tempi'),
        ('acc1', 'accuracy 1'),
        ('acc2', 'accuracy 2')
    ]

    def __init__(self, detections, annotations, strengths, tolerance=TOLERANCE,
                 double=DOUBLE, triple=TRIPLE):
        """
        Evaluate the given detection and annotation sequences.

        :param detections:  array with detected tempi [bpm]
        :param annotations: array with the annotated tempi [bpm]
        :param strengths:   array with the relative strengths of the tempi
        :param tolerance:   allowed tempo deviation
        :param double:      also evaluate double/half tempo [bool]
        :param triple:      also evaluate triple/third tempo [bool]

        """
        # convert the detections and annotations
        detections = np.asarray(detections, dtype=np.float)
        annotations = np.asarray(annotations, dtype=np.float)
        strengths = np.asarray(strengths, dtype=np.float)
        # evaluate
        results = tempo_evaluation(detections, annotations, strengths,
                                   tolerance)
        self.pscore, self.any, self.all = results
        self.acc1 = self.any
        # also evaluate with double / half and triple / third tempo
        ann = annotations.copy()
        if double:
            ann = np.hstack((ann, annotations * 2., annotations / 2.))
        if triple:
            ann = np.hstack((ann, annotations * 3., annotations / 3.))
        # we need to tile the strengths; in case of no strengths, divide by 1
        len_strengths = max(1, len(strengths))
        strengths = np.tile(strengths, len(ann) / len_strengths)
        self.acc2 = tempo_evaluation(detections, ann, strengths, tolerance)[1]

    def to_string(self):
        """
        Print errors.
        """
        return 'pscore=%.3f (one tempo: %.3f, all tempi: %.3f) ' \
               'acc1=%.3f acc2=%.3f' % (self.pscore, self.any,
                                        self.all, self.acc1, self.acc2)

    def __str__(self):
        return self.to_string()


class MeanTempoEvaluation(TempoEvaluation):
    """
    Class for averaging tempo evaluation scores.

    """
    # we just want to inherit the print_errors() function

    def __init__(self):
        """
        Class for averaging tempo evaluation scores.

        """
        # simple scores
        self._pscore = np.zeros(0)
        self._any = np.zeros(0, dtype=bool)
        self._all = np.zeros(0, dtype=bool)
        self._acc1 = np.zeros(0, dtype=bool)
        self._acc2 = np.zeros(0, dtype=bool)

    def __len__(self):
        # just use the length of any of the arrays
        return len(self._pscore)

    # for adding another TempoEvaluation object
    def append(self, other):
        """
        Appends the scores of another TempoEvaluation object to the respective
        arrays.

        :param other: TempoEvaluation object

        """
        if isinstance(other, TempoEvaluation):
            self._pscore = np.append(self._pscore, other.pscore)
            self._any = np.append(self._any, other.any)
            self._all = np.append(self._all, other.all)
            self._acc1 = np.append(self._acc1, other.acc1)
            self._acc2 = np.append(self._acc2, other.acc2)
        else:
            raise TypeError('Can only append TempoEvaluation to '
                            'MeanTempoEvaluation, not %s' %
                            type(other).__name__)

    @property
    def pscore(self):
        """P-Score."""
        if len(self._pscore) == 0:
            return 0.
        return np.mean(self._pscore)

    @property
    def any(self):
        """At least one tempo correct."""
        if len(self._any) == 0:
            return 0.
        return np.mean(self._any)

    @property
    def all(self):
        """All tempi correct."""
        if len(self._all) == 0:
            return 0.
        return np.mean(self._all)

    @property
    def acc1(self):
        """Accuracy 1."""
        if len(self._acc1) == 0:
            return 0.
        return np.mean(self._acc1)

    @property
    def acc2(self):
        """Accuracy 2."""
        if len(self._acc2) == 0:
            return 0.
        return np.mean(self._acc2)


def parser():
    """
    Create a parser and parse the arguments.

    :return: the parsed arguments

    """
    import argparse
    from . import evaluation_in, evaluation_out
    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    This script evaluates pairs of files containing the tempo annotations and
    detections. Suffixes can be given to filter them from the list of files.

    A single line represents the tempi and their relative strength and must
    have the following format with values being separated by tabs:
    `tempo_one tempo_two relative_strength`

    Lines starting with # are treated as comments and are ignored.

    """)
    # files used for evaluation
    evaluation_in(p, ann_suffix='.bpm', det_suffix='.bpm.txt')
    evaluation_out(p)
    # parameters for evaluation
    g = p.add_argument_group('evaluation arguments')
    g.add_argument('--tolerance', type=float, action='store',
                   default=TOLERANCE, help='tolerance for tempo detection '
                                           '[default=%(default).3f]')
    g.add_argument('--all', action='store_true', default=False,
                   help='evaluate all detections, even if only 1 annotation '
                        'is given')
    g.add_argument('--one', action='store_true', default=False,
                   help='evaluate only the first detection, even if multiple '
                        'annotation are given')
    g.add_argument('--no_double', dest='double', action='store_false',
                   default=DOUBLE,
                   help='do not include double/half tempo evaluation')
    g.add_argument('--no_triple', dest='triple', action='store_false',
                   default=TRIPLE,
                   help='do not include triple/third tempo evaluation')
    # parse the arguments
    args = p.parse_args()
    # print the args
    if args.verbose >= 2:
        print args
    if args.quiet:
        warnings.filterwarnings("ignore")
        # return
    return args


def main():
    """
    Simple tempo evaluation.

    """
    import os
    from madmom.utils import search_files, match_file

    # parse arguments
    args = parser()

    # get detection and annotation files
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

    # mean evaluation for all files
    mean_eval = MeanTempoEvaluation()
    eval_output = args.output_formatter(mean_eval.METRIC_NAMES)
    # evaluate all files
    for ann_file in ann_files:
        # load the annotations
        annotations, strengths = load_tempo(ann_file)
        # get the matching detection files
        matches = match_file(ann_file, det_files,
                             args.ann_suffix, args.det_suffix)
        if len(matches) > 1:
            # exit if multiple detections were found
            raise SystemExit("multiple detections for %s found." % ann_file)
        elif len(matches) == 0:
            # ignore non-existing detections
            if args.ignore_non_existing:
                continue
            # print a warning if no detections were found
            import warnings
            warnings.warn(" can't find detections for %s." % ann_file)
            # but continue and assume no detected tempo
            detections = np.zeros(0)
        else:
            # get the detections tempi (ignore the strengths)
            detections, _ = load_tempo(matches[0])
        # crop the detections to the length of the annotations
        # TODO: should this logic go into the TempoEvaluation class?
        if not args.all:
            detections = detections[:len(annotations)]
            strengths = strengths[:len(annotations)]
        if args.one:
            # only use the first (i.e. strongest) detection
            detections = [detections[0]]
        # add the Evaluation to mean evaluation
        e = TempoEvaluation(detections, annotations, strengths, args.tolerance,
                            double=args.double, triple=args.triple)
        # print stats for each file
        if args.verbose:
            eval_output.add_eval(os.path.basename(ann_file), e)
        # add this file's mean evaluation to the global evaluation
        mean_eval.append(e)
    # print summary
    eval_output.add_eval('mean for %i file(s)' % len(mean_eval), mean_eval)
    print eval_output


if __name__ == '__main__':
    main()
