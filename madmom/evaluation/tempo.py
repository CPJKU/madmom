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
def pscore(detections, targets, tolerance, strengths=None):
    """
    Calculate the tempo P-Score.

    :param detections: array with (multiple) tempi [bpm]
    :param targets:    array with (multiple) tempi [bpm]
    :param tolerance:  evaluation tolerance
    :param strengths:  array with the relative strengths of the tempi [floats]
                       If no strength is given, an even distribution is assumed.
    :returns:          p-score

    """
    # no detections are given
    if detections.size == 0:
        return 0
    # no targets are given
    if targets.size == 0:
        raise TypeError("Target tempo must be given.")
    # if no target strengths are given, distribute evenly
    if strengths is None:
        strengths = np.ones_like(targets)
    if targets.size != strengths.size:
        raise ValueError("Tempo targets and strengths must match in size.")
    # strengths must sum up to 1
    strengths_sum = np.sum(strengths)
    if strengths_sum == 0:
        strengths = np.ones_like(targets) / float(targets.size)
    elif strengths_sum != 1:
        strengths /= float(strengths_sum)

    # test each detected tempi against all target tempi
    errors = np.abs(1 - (detections[:, np.newaxis] / targets))
    # correctly identified target tempi
    correct = np.asarray(np.sum(errors < tolerance, axis=0), np.bool)
    # the p-score is the sum of the strengths of the correctly identified tempi
    return np.sum(strengths[correct])


TOLERANCE = 0.08


# basic tempo evaluation
class TempoEvaluation(object):
    """
    Tempo evaluation class.

    """

    def __init__(self, detections, targets, tolerance=TOLERANCE):
        """
        Evaluate the given detection and target sequences.

        :param detections: array with detected tempi [bpm]
        :param targets:    tuple with 2 numpy arrays (tempi, strength) [bpm, floats]
                           The first contains all tempi, the second their relative
                           strengths. If no strength is given, an even distribution
                           is assumed.
        :param tolerance:  tolerance [default=0.08]

        """
        self.detections = detections
        self.targets = targets
        self.tolerance = tolerance
        # score
        self.__pscore = None

    @property
    def num(self):
        """Number of evaluated files."""
        return 1

    @property
    def pscore(self):
        """P-Score."""
        if self.__pscore is None:
            self.__pscore = pscore(self.detections, self.targets,
                                   self.tolerance)
        return self.__pscore

    def print_errors(self, tex=False):
        """
        Print errors.

        :param tex: output format to be used in .tex files [default=False]

        """
        # print the errors
        print "  pscore=%.3f (target tempi: %s detections: %s tolerance: %.1f \%)" % (self.pscore, self.targets, self.detections, self.tolerance * 100)
        if tex:
            print "%i events & P-Score\\\\" % self.num
            print "tex & %.3f \\\\" % self.pscore

    def __str__(self):
        return "%s pscore=%.3f" % (self.__class__, self.pscore)


class MeanTempoEvaluation(TempoEvaluation):
    """
    Class for averaging tempo evaluation scores.

    """

    def __init__(self, other=None):
        """
        MeanTempoEvaluation object can be either instantiated as an empty object
        or by passing in a TempoEvaluation object with the scores taken from
        that object.

        :param other: TempoEvaluation object

        """
        # simple scores
        self.__pscore = np.empty(0)
        # instance can be initialized with a Evaluation object
        if isinstance(other, TempoEvaluation):
            # add this object to self
            self += other

    # for adding another TempoEvaluation object
    def __add__(self, other):
        """
        Appends the scores of another TempoEvaluation object.

        :param other: TempoEvaluation object

        """
        if isinstance(other, TempoEvaluation):
            self.__pscore = np.append(self.__pscore, other.pscore)
        else:
            return NotImplemented

    @property
    def num(self):
        """Number of evaluated files."""
        return len(self.__pscore)

    @property
    def pscore(self):
        """P-Score."""
        return np.mean(self.__pscore)


def parser():
    """
    Create a parser and parse the arguments.

    :return: the parsed arguments

    """
    import argparse
    # define parser
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    The script evaluates a file or folder with detections against a file or
    folder with targets. Extensions can be given to filter the detection and
    target file lists.

    """)
    # files used for evaluation
    p.add_argument('detections',
                   help='file (or folder) with detections to be evaluated (files being filtered according to the -d argument)')
    p.add_argument('targets', nargs='*',
                   help='file (or folder) with targets (files being filtered according to the -t argument)')
    # extensions used for evaluation
    p.add_argument('-d', dest='det_ext', action='store', default='.bpm.txt',
                   help='extension of the detection files')
    p.add_argument('-t', dest='tar_ext', action='store', default='.bpm',
                   help='extension of the target files')
    p.add_argument('--tolerance', dest='tolerance', action='store',
                   default=TOLERANCE, help='tolerance for tempo detection')
    # parameters for evaluation
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
    from ..utils.helpers import files, match_file, load_events

    # parse arguments
    args = parser()

    # detection files
    det_files = files(args.files, args.det_ext)
    # quit if no files are found
    if len(det_files) == 0:
        print "no files to evaluate. exiting."
        exit()

    # sum and mean counter for all files
    mean_counter = MeanTempoEvaluation()
    # evaluate all files
    for det_file in det_files:
        # get the detections file
        detections = load_events(det_file)
        # get the matching target files
        tar_files = match_file(det_file, args.files, args.det_ext, args.tar_ext)
        if len(tar_files) == 0:
            print " can't find a target file found for %s. exiting." % det_file
            exit()
        # do a mean evaluation with all matched target files
        me = MeanTempoEvaluation()
        for tar_file in tar_files:
            # load the targets
            targets = load_events(tar_file)
            # add the Evaluation to mean evaluation
            me += TempoEvaluation(detections, targets, args.tolerance)
            # process the next target file
        # print stats for each file
        if args.verbose:
            me.print_errors(args.tex)
            # add the resulting sum counter
        mean_counter += me
        # process the next detection file
    # print summary
    print 'mean for %i files:' % (len(det_files))
    mean_counter.print_errors(args.tex)


if __name__ == '__main__':
    main()
