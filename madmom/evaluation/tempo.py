#!/usr/bin/env python
# encoding: utf-8
"""
This file contains tempo evaluation functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import numpy as np


def load_tempo(filename):
    """
    Load tempo ground-truth from a file.

    :param filename: name of the file
    :return:         tempo

    Expected file format: tempo_1 [tempo_2 [rel_strength_1 [rel_strength_2]]]

    """
    return np.loadtxt(filename)


# def average_tempi(tempi):
#     # implement the McKinney Paper with merging multiple annotations
#     raise NotImplementedError


# this evaluation function can evaluate multiple tempi simultaneously
def tempo_evaluation(detections, annotations, tolerance, strengths=None):
    """
    Calculate the tempo P-Score.

    :param detections:  array with (multiple) tempi [bpm]
    :param annotations: array with (multiple) tempi [bpm]
    :param tolerance:   evaluation tolerance
    :param strengths:   array with the relative strengths of the tempi
    :returns:           p-score

    Note: If no relative strengths are given, an even distribution is assumed.

    """
    # no detections are given
    if detections.size == 0:
        return 0
    # no annotations are given
    if annotations.size == 0:
        raise TypeError("Target tempo must be given.")
    # if no annotation strengths are given, distribute evenly
    if strengths is None:
        strengths = np.ones_like(annotations)
    if annotations.size != strengths.size:
        raise ValueError("Tempo annotations and strengths must match in size.")
    # strengths must sum up to 1
    strengths_sum = np.sum(strengths)
    if strengths_sum == 0:
        # distribute evenly
        strengths = np.ones_like(annotations) / float(annotations.size)
    elif strengths_sum != 1:
        strengths /= float(strengths_sum)

    # test each detected tempi against all annotation tempi
    errors = np.abs(1 - (detections[:, np.newaxis] / annotations))
    # correctly identified annotation tempi
    correct = np.asarray(np.sum(errors < tolerance, axis=0), np.bool)
    # the p-score is the sum of the strengths of the correctly identified tempi
    return np.sum(strengths[correct])


TOLERANCE = 0.08


# basic tempo evaluation
class TempoEvaluation(object):
    """
    Tempo evaluation class.

    """

    def __init__(self, detections, annotations, tolerance=TOLERANCE):
        """
        Evaluate the given detection and annotation sequences.

        :param detections:  array with detected tempi [bpm]
        :param annotations: tuple with 2 numpy arrays (tempi, strength)
                            [bpm, floats] The first contains all tempi, the
                            second their relative strengths. If no strengths
                            are given, an even distribution is assumed.
        :param tolerance:   tolerance

        """
        # convert the detections and annotations
        detections = np.asarray(sorted(detections), dtype=np.float)
        annotations = np.asarray(sorted(annotations), dtype=np.float)
        # evaluate
        self.pscore = tempo_evaluation(detections, annotations, tolerance)

    def print_errors(self, indent='', tex=False):
        """
        Print errors.

        :param indent: use the given string as indentation
        :param tex:    output format to be used in .tex files

        """
        if tex:
            # tex formatting
            ret = 'tex & P-Score\\\\\n& %.3f\\\\' % self.pscore
        else:
            # normal formatting
            ret = '%spscore=%.3f' % (indent, self.pscore)
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

    # for adding another TempoEvaluation object
    def append(self, other):
        """
        Appends the scores of another TempoEvaluation object to the respective
        arrays.

        :param other: TempoEvaluation object

        """
        if isinstance(other, TempoEvaluation):
            self._pscore = np.append(self._pscore, other.pscore)
        else:
            raise TypeError('Can only append TempoEvaluation to '
                            'MeanTempoEvaluation, not %s' %
                            type(other).__name__)


def parser():
    """
    Create a parser and parse the arguments.

    :return: the parsed arguments

    """
    import argparse
    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    The script evaluates a file or folder with detections against a file or
    folder with annotations. ann

    """)
    # files used for evaluation
    # files used for evaluation
    p.add_argument('files', nargs='*',
                   help='files (or folder) to be evaluated')
    # extensions used for evaluation
    p.add_argument('-d', dest='det_suffix', action='store', default='.bpm.txt',
                   help='suffix of the detection files')
    p.add_argument('-t', dest='ann_suffix', action='store', default='.bpm',
                   help='suffix of the annotation files')
    # evaluation parameter
    p.add_argument('--tolerance', dest='tolerance', action='store',
                   default=TOLERANCE, help='tolerance for tempo detection')
    p.add_argument('--tex', action='store_true',
                   help='format errors for use is .tex files')
    # verbose
    p.add_argument('-v', dest='verbose', action='count',
                   help='increase verbosity level')
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
    from ..utils import files, match_file, load_events

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
        # get the detections file
        detections = load_events(det_file)
        # get the matching annotation files
        matches = match_file(det_file, ann_files, args.det_suffix,
                             args.ann_suffix)
        # quit if any file does not have a matching annotation file
        if len(matches) == 0:
            print " can't find a annotation file for %s. exiting." % det_file
            exit()
        # do a mean evaluation with all matched annotation files
        # TODO: decide whether we want multiple annotations per file or
        #       multiple files and do a mean_evaluation on those
        me = MeanTempoEvaluation()
        for ann_file in matches:
            # load the annotations
            annotations = load_events(ann_file)
            # add the Evaluation to mean evaluation
            me.append(TempoEvaluation(detections, annotations, args.tolerance))
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
