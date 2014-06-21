#!/usr/bin/env python
# encoding: utf-8
"""
This file contains tempo evaluation functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import numpy as np

from madmom.utils import open


def load_tempo(filename, split_value=1.):
    """
    Load tempo information from a file.

    :param filename:    name of the file
    :param split_value: values > split_value are interpreted as tempi in bpm,
                        values <= split_value are interpreted as strengths
    :return:            tuple with arrays containing the tempi and
                        their relative strengths

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
        # return
    return tempi, strengths

# def average_tempi(tempi):
#     # implement the McKinney paper with merging multiple annotations
#     raise NotImplementedError


# this evaluation function can evaluate multiple tempi simultaneously
def tempo_evaluation(detections, annotations, strengths, tolerance):
    """
    Calculate the tempo P-Score.

    :param detections:  array with (multiple) tempi [bpm]
    :param annotations: array with (multiple) tempi [bpm]
    :param strengths:   array with the relative strengths of the tempi
    :param tolerance:   evaluation tolerance
    :returns:           p-score, at least one tempo correctly identified
                        (float, bool)

    Note: If no relative strengths are given, evenly distributed strengths
          are assumed.

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


TOLERANCE = 0.08


# basic tempo evaluation
class TempoEvaluation(object):
    """
    Tempo evaluation class.

    """

    def __init__(self, detections, annotations, strengths,
                 tolerance=TOLERANCE):
        """
        Evaluate the given detection and annotation sequences.

        :param detections:  array with detected tempi [bpm]
        :param annotations: array with the annotated tempi [bpm]
        :param strengths:   array with the relative strengths of the tempi
        :param tolerance:   allowed tempo deviation

        """
        # convert the detections and annotations
        detections = np.asarray(detections, dtype=np.float)
        annotations = np.asarray(annotations, dtype=np.float)
        strengths = np.asarray(strengths, dtype=np.float)
        # evaluate
        results = tempo_evaluation(detections, annotations, strengths,
                                   tolerance)
        self.pscore, self.any, self.all = results

    def print_errors(self, indent='', tex=False):
        """
        Print errors.

        :param indent: use the given string as indentation
        :param tex:    output format to be used in .tex files

        """
        if tex:
            # tex formatting
            ret = 'tex & P-Score & one tempo & both tempi\\\\\n& %.3f ' \
                  '& %.3f & %.3f\\\\' % (self.pscore, self.any, self.all)
        else:
            # normal formatting
            ret = '%spscore=%.3f (one tempo: %.3f, all tempi: %.3f)' % \
                  (indent, self.pscore, self.any, self.all)
        return ret


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


def parser():
    """
    Create a parser and parse the arguments.

    :return: the parsed arguments

    """
    import argparse
    from . import evaluation_io
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
    evaluation_io(p, ann_suffix='.bpm', det_suffix='.bpm.txt')
    # parameters for evaluation
    g = p.add_argument_group('evaluation arguments')
    g.add_argument('--tolerance', type=float, action='store',
                   default=TOLERANCE, help='tolerance for tempo detection '
                                           '[default=%(default).3f]')
    g.add_argument('--all', action='store_true', default=False,
                   help='evaluate all detections, even if only 1 annotation '
                        'is given')
    # parse the arguments
    args = p.parse_args()
    # print the args
    if args.verbose >= 2:
        print args
        # return
    return args


def main():
    """
    Simple tempo evaluation.

    """
    from ..utils import files, match_file

    # parse arguments
    args = parser()

    # get detection and annotation files
    det_files = files(args.files, args.det_suffix)
    ann_files = files(args.files, args.ann_suffix)
    # quit if no files are found
    if len(det_files) == 0:
        print "no files to evaluate. exiting."
        exit()

    # mean evaluation for all files
    mean_eval = MeanTempoEvaluation()
    # evaluate all files
    for det_file in det_files:
        # get the matching annotation files
        matches = match_file(det_file, ann_files, args.det_suffix,
                             args.ann_suffix)
        # quit if any file does not have a matching annotation file
        if len(matches) == 0:
            print " can't find a annotation file for %s. exiting." % det_file
            exit()
        # get the detections tempi (ignore the strengths)
        detections, _ = load_tempo(det_file)
        # do a mean evaluation with all matched annotation files
        # TODO: decide whether we want multiple annotations per file or
        #       multiple files and do a mean_evaluation on those
        me = MeanTempoEvaluation()
        for ann_file in matches:
            # load the annotations
            annotations, strengths = load_tempo(ann_file)
            # crop the detections to the length of the annotations
            if not args.all:
                detections = detections[:len(annotations)]
                strengths = strengths[:len(annotations)]
            # add the Evaluation to mean evaluation
            me.append(TempoEvaluation(detections, annotations, strengths,
                                      args.tolerance))
            # process the next annotation file
        # print stats for each file
        if args.verbose:
            print det_file
            print me.print_errors('  ', args.tex)
        # add this file's mean evaluation to the global evaluation
        mean_eval.append(me)
        # process the next detection file
    # print summary
    print 'mean for %i files:' % (len(det_files))
    print mean_eval.print_errors('  ', args.tex)

if __name__ == '__main__':
    main()
