#!/usr/bin/env python
# encoding: utf-8
"""
Copyright (c) 2013, Sebastian Böck <sebastian.boeck@jku.at>
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

"""
This software serves as a Python implementation of the beat evaluation toolkit,
which can be downloaded from:
http://code.soundsoftware.ac.uk/projects/beat-evaluation/repository

The used measures are described in:

"Evaluation Methods for Musical Audio Beat Tracking Algorithms"
Matthew E. P. Davies, Norberto Degara, and Mark D. Plumbley
Technical Report C4DM-TR-09-06
Centre for Digital Music, Queen Mary University of London, 2009

Please note that this is a complete re-implementation, which took some other
design decisions. For example, the beat detections and targets are not quantized
before being evaluated with the F-measure, P-score and other metrics. Hence
these evaluation functions DO NOT report the exact same results/scores. This
approach was chosen, because it produces more accurate results.

Please send any comments, enhancements, errata, etc. to the main author.
"""

from helpers import load_events

import math
import numpy as np


class Score(object):
    """
    Simple class for aggregating scores.

    """
    def __init__(self, p_score=0, cemgil=0, cmlc=0, cmlt=0, amlc=0, amlt=0, information_gain=0, error_histogram=0):
        """
        Creates a new Score object instance.

        """
        # for simple events like onsets or beats
        self.p_score = p_score
        self.cemgil = cemgil
        self.cmlc = cmlc
        self.cmlt = cmlt
        self.amlc = amlc
        self.amlt = amlt
        self.information_gain = information_gain
        self.error_histogram = error_histogram
        self.global_information_gain = None
        self.__num = 0.

    # for adding 2 ScoreAggregators
    def __add__(self, other):
        if isinstance(other, Score):
            self.p_score += other.p_score
            self.cemgil += other.cemgil
            self.cmlc += other.cmlc
            self.cmlt += other.cmlt
            self.amlc += other.amlc
            self.amlt += other.amlt
            self.information_gain += other.information_gain
            self.error_histogram += other.error_histogram
            self.__num += 1
            return self
        else:
            return NotImplemented

    def average(self):
        """Average all scores."""
        # most scores are just averaged
        self.p_score /= self.__num
        self.cemgil /= self.__num
        self.cmlc /= self.__num
        self.cmlt /= self.__num
        self.amlc /= self.__num
        self.amlt /= self.__num
        self.information_gain /= self.__num
        # for calculation of the global information gain, re-calculate it on the
        # basis of the sum of all error histograms
        self.global_information_gain = calc_information_gain(self.error_histogram)

    def print_errors(self, tex=False):
        """
        Print errors.

        param: tex: output format to be used in .tex files [default=False]

        """
        # print the errors
        # report the scores always in the range 0..1, because of formatting
        if self.global_information_gain is None:
            print '  P-score: %.4f Cemgil: %.4f CMLc: %.4f CMLt: %.4f AMLc: %.4f AMLt: %.4f D: %.3f' % (self.p_score, self.cemgil, self.cmlc, self.cmlt, self.amlc, self.amlt, self.information_gain)
        else:
            print '  P-score: %.4f Cemgil: %.4f CMLc: %.4f CMLt: %.4f AMLc: %.4f AMLt: %.4f D: %.3f Dg: %.3f' % (self.p_score, self.cemgil, self.cmlc, self.cmlt, self.amlc, self.amlt, self.information_gain, self.global_information_gain)
        if tex:
            print "%i files & P-score & Cemgil & CMLc & CMLt & AMLc & AMLt & I_g\\\\" % (self.__num)
            print "tex & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f\\\\" % (self.p_score * 100, self.cemgil * 100, self.cmlc * 100, self.cmlt * 100, self.amlc * 100, self.amlt * 100, self.information_gain)


# helper functions
def find_closest_match(detections, targets):
    """
    Find the closest matches in targets to all detections.

    :param detections: sequence of events to be matched [seconds]
    :param targets: sequence of possible matches [seconds]
    :returns: a list of indices with closest matches

    """
    # solution found at: http://stackoverflow.com/questions/8914491/finding-the-nearest-value-and-return-the-index-of-array-in-python
    detections = np.array(detections)
    targets = np.array(targets)
    idx = detections.searchsorted(targets)
    idx = np.clip(idx, 1, len(detections) - 1)
    left = detections[idx - 1]
    right = detections[idx]
    idx -= targets - left < right - targets
    return idx


# evaluation functions
def p_score(detections, targets, window=0.2):
    """
    Calculate the P-Score accuracy.

    :param detections: sequence of estimated beat times [seconds]
    :param targets: sequence of ground truth beat annotations [seconds]
    :param window: tolerance window (fraction of the median beat interval) [default=0.2]
    :returns: p-score

    "Evaluation of audio beat tracking and music tempo extraction algorithms"
    M. F. McKinney, D. Moelants, M. E. P. Davies, and A. Klapuri
    Journal of New Music Research, vol. 36, no. 1, pp. 1–16, 2007.

    """
    # init p-score
    p_score = 0
    # no detections: score=0
    if not detections:
        return p_score
    # error window is the given fraction of the median beat interval
    window *= np.median(np.diff(targets))
    # evaluate
    det_length = len(detections)
    tar_length = len(targets)
    # start with the first detection and target
    det = 0
    tar = 0
    while det < det_length and tar < tar_length:
        # calculate the diff between first detection and target
        if abs(detections[det] - targets[tar]) < window:
            # correct detection
            p_score += 1
            # continue with the detection and target
            det += 1
            tar += 1
        elif detections[det] < targets[tar]:
            # detection is before target and outside tolerance window
            # continue with the next detection
            det += 1
        elif detections[det] > targets[tar]:
            # detection is after target and outside tolerance window
            # continue with the next target
            tar += 1
    # normalize by the max number of detections/targets
    p_score /= float(max(det_length, tar_length))
    # return p-score
    return p_score

# NOTE: this is not faster!
#def p_score(detections, targets, window=0.2):
#    """
#    Calculate the P-Score accuracy.
#
#    :param detections: sequence of estimated beat times [seconds]
#    :param targets: sequence of ground truth beat annotations [seconds]
#    :param window: error window (fraction of the median beat interval) [default=0.2]
#    :returns: p-score
#
#    "Evaluation of audio beat tracking and music tempo extraction algorithms"
#    M. F. McKinney, D. Moelants, M. E. P. Davies, and A. Klapuri
#    Journal of New Music Research, vol. 36, no. 1, pp. 1–16, 2007.
#
#    """
#    # error window is the given fraction of the median beat interval
#    window *= np.median(np.diff(targets))
#    # find closest targets to detections
#    closest = find_closest_match(targets, detections)
#    # init p-score as float
#    p_score = 0.
#    # evaluate
#    for det in range(len(detections)):
#        # get the closest target
#        tar = closest[det]
#        if abs(detections[det] - targets[tar]) <= window:
#            # correct detection
#            p_score += 1
#    # normalize by the max number of detections/targets
#    p_score /= max(len(targets), len(detections))
#    # return p-score
#    return p_score


def cemgil(detections, targets, sigma=0.04):
    """
    Calculate the Cemgil accuracy.

    :param detections: sequence of estimated beat times [seconds]
    :param targets: sequence of ground truth beat annotations [seconds]
    :param sigma: sigma for Gaussian window [default=0.04]
    :returns: beat tracking accuracy

    "On tempo tracking: Tempogram representation and Kalman filtering"
    A.T. Cemgil, B. Kappen, P. Desain, and H. Honing
    Journal Of New Music Research, vol. 28, no. 4, pp. 259–273, 2001

    """
    # beat accuracy is initially zero
    acc = 0
    # no detections
    if not detections:
        return acc
    # find closest detections to targets
    closest = find_closest_match(detections, targets)
    for tar in range(len(targets)):
        # calculate the difference between the target and its closets match
        diff = abs(detections[closest[tar]] - targets[tar])
        # determine the value on the Gaussian error function and add to the acc.
        acc += math.exp(-(diff ** 2.) / (2. * (sigma ** 2.)))
    # normalize by the mean of the number of detections and targets
    acc /= 0.5 * (len(targets) + len(detections))
    # return accuracy
    return acc


def continuity(detections, targets, tolerance=0.175):
    """
    Calculate cmlc, cmlt, amlc, amlt for the given detection and target sequences.

    :param detections: sequence of estimated beat times [seconds]
    :param targets: sequence of ground truth beat annotations [seconds]
    :param tolerance: tolerance window [default=0.175]
    :returns: cmlc, cmlt, amlc, amlt beat tracking accuracies

    cmlc: beat tracking accuracy, continuity required at the correct metrical level
    cmlt: beat tracking accuracy, continuity not required at the correct metrical level
    amlc: beat tracking accuracy, continuity required at allowed metrical levels
    amlt: beat tracking accuracy, continuity not required at allowed metrical levels

    "Techniques for the automated analysis of musical audio"
    S. Hainsworth
    Ph.D. dissertation, Department of Engineering, Cambridge University, 2004.

    "Analysis of the meter of acoustic musical signals"
    A. P. Klapuri, A. Eronen, and J. Astola
    IEEE Transactions on Audio, Speech and Language Processing, vol. 14, no. 1, pp. 342–355, 2006.

    """
    # needs at least 2 detections and targets to interpolate
    if min(len(targets), len(detections)) < 2:
        return 0, 0, 0, 0

    # Note: it does not extrapolate the last value, so skip it
    xold = np.arange(0, len(targets))
    xnew = np.arange(0, len(targets), 0.5)
    double_targets = np.interp(xnew, xold, targets)[:-1]

    # make different variants of annotations
    variations = []
    # double tempo
    variations.append(double_targets)
    # off-beats
    variations.append(double_targets[1::2])
    # half tempo odd beats (i.e. 1,3,1,3)
    variations.append(targets[::2])
    # half tempo even beats (i.e. 2,4,2,4)
    variations.append(targets[1::2])
    # TODO: include a third tempo as well? This might be needed for fast Waltz
    variations.append(targets[::3])

    # evaluate correct tempo
    cmlc, cmlt = cml(detections, targets, tolerance)
    # evaluate other metrical levels
    amlc = cmlc
    amlt = cmlt
    for tar in variations:
        # speed up calculation by skipping other metrical levels if the score
        # is higher than 0.5 already. We must have tested the correct metrical
        # level already
        if amlc > 0.5:
            continue
        # if other metrical levels achieve a higher accuracy, take these values
        c, t = cml(detections, tar, tolerance)
        amlc = max(amlc, c)
        amlt = max(amlt, t)

    # return a tuple
    return cmlc, cmlt, amlc, amlt


def cml(detections, targets, tolerance=0.175):
    """
    Calculate cmlc, cmlt for the given detection and target sequences.

    :param detections: sequence of estimated beat times [seconds]
    :param targets: sequence of ground truth beat annotations [seconds]
    :param tolerance: tolerance window [default=0.175]
    :returns: cmlc, cmlt

    cmlc: beat tracking accuracy, continuity required at the correct metrical level
    cmlt: beat tracking accuracy, continuity not required at the correct metrical level

    "Techniques for the automated analysis of musical audio"
    S. Hainsworth
    Ph.D. dissertation, Department of Engineering, Cambridge University, 2004.

    "Analysis of the meter of acoustic musical signals"
    A. P. Klapuri, A. Eronen, and J. Astola
    IEEE Transactions on Audio, Speech and Language Processing, vol. 14, no. 1, pp. 342–355, 2006.

    """
    # needs at least 2 detections and targets to calculate the intervals
    if min(len(targets), len(detections)) < 2:
        return 0, 0
    # list for collecting correct detections / intervals
    correct = []
    correct_interval = []
    # determine closest targets to detections
    closest = find_closest_match(targets, detections)
    # iterate over all detections
    for det in range(len(detections)):
        # look for nearest target to current detections
        tar = closest[det]
        # interval between this and the previous detection
        if det == 0:
            det_interval = detections[1] - detections[0]
        else:
            det_interval = detections[det] - detections[det - 1]
        # interval between this and the previous target
        if tar == 0:
            tar_interval = targets[1] - targets[0]
        else:
            tar_interval = targets[tar] - targets[tar - 1]
        # determine detection indices which are within the tolerance window
        if abs(detections[det] - targets[tar]) < tolerance * tar_interval:
            # add a fake beat if the first beat is correct
#            TODO: add these 2 lines if condition 2) is included
#            if det == 0:
#                correct.append(-1)
            correct.append(det)
        # determine intervals which are within the tolerance window
        if abs(1 - (det_interval / tar_interval)) < tolerance:
            # add a fake beat if the first beat is correct
#            TODO: add these 2 lines if condition 2) is included
#            if det == 0:
#                correct_interval.append(-1)
            correct_interval.append(det)
    # a detection is correct, if it fulfills 3 conditions:
    # 1) must match an annotation within a certain tolerance window
    # only detections which satisfy this condition are in the correct list
    # 2) same must be true for the previous detection / target combination
#   TODO: if condition 2) is included, uncomment the 4 lines above
#    correct = [c for c in correct if c - 1 in correct]
    # 3) the interval must not be apart more than the threshold
    correct = [c for c in correct if c in correct_interval]
    # split into groups of continuous sequences
    # solution to a similar problem found at:
    # http://stackoverflow.com/questions/10420464/group-list-of-ints-by-continuous-sequence
    from itertools import groupby, count
    cont_detections = [list(g) for _, g in groupby(correct, key=lambda n, c=count(): n - next(c))]
    # determine the longest continuous detection
    if cont_detections:
        cont = max([len(cont) for cont in cont_detections])
    else:
        cont = 0
    # maximal length of the given sequences
    length = float(max(len(detections), len(targets)))
    # accuracy for the longest continuous detections
    cmlc = cont / length
    # accuracy of all correct detections
    cmlt = len(correct) / length
    # return a tuple
    return cmlc, cmlt


def information_gain(detections, targets, bins=40):
    """
    Calculate information gain.

    :param detections: sequence of estimated beat times [seconds]
    :param targets: sequence of ground truth beat annotations [seconds]
    :param bins: number of histogram bins [default=40]
    :returns: infromation gain, beat error histogram


    "Measuring the performance of beat tracking algorithms algorithms using a beat error histogram"
    M. E. P. Davies, N. Degara and M. D. Plumbley
    IEEE Signal Processing Letters, vol. 18, vo. 3, 2011

    Note: even number of bins results in having a bin at the centre for all
          beats with an error close to 0 - which is desirable.


    """
    # in case of no detections
    if not detections or len(targets) < 2:
        # return information gain = 0 and a uniform beat error histogram
        return 0, np.ones(bins) * len(targets) / bins

    # create histogram bin borders
    # make the first and last bin just half as wide as the rest, thus the offset
    offset = 0.5 / bins
    # since np.histogram uses borders and not the centres as bins, one bin must
    # be added; + another, because the last bin is wraped around to the first
    # one later
    histogram_bins = np.linspace(-0.5 - offset, 0.5 + offset, bins + 2)

    # evaluate detections against targets
    errors = beat_errors(detections, targets)
    fwd_histogram = map_errors(errors, histogram_bins)
    fwd_ig = calc_information_gain(fwd_histogram)

    # in case of underdetection, the errors could be very small; thus evaluate
    # also the targets against the detections (i.e. simulate a lot of FPs)
    errors = beat_errors(targets, detections)
    bwd_histogram = map_errors(errors, histogram_bins)
    bwd_ig = calc_information_gain(bwd_histogram)

    # use the lower information gain
    if fwd_ig < bwd_ig:
        return fwd_ig, fwd_histogram
    else:
        return bwd_ig, bwd_histogram


# information gain helper functions
def beat_errors(detections, targets):
    """
    Calculate the relative errors of the given detection wrt. the targets.

    :param detections: sequence of estimated beat times [seconds]
    :param targets: sequence of ground truth beat annotations [seconds]
    :returns: relative beat errors

    """
    # array for relative detection errors
    # use a numpy array instead of a list, so we can do some math on it later
    errors = np.zeros(len(detections))
    # determine closest targets to detections
    closest = find_closest_match(targets, detections)
    # store the number of targets
    last_target = len(targets) - 1
    # iterate over all detections
    for det in range(len(detections)):
        # find closest target
        tar = closest[det]
        # difference to this target
        diff = detections[det] - targets[tar]
        # calculate the relative error for this target
        if tar == 0:
            # closet target is the first one or before the current beat
            # calculate the diff to the next target
            interval = targets[tar + 1] - targets[tar]
        elif tar == last_target:
            # closet target is the last one or after the current beat
            # calculate the diff to the previous target
            interval = targets[tar] - targets[tar - 1]
        else:
            # normal
            if diff > 0:
                # closet target is before the current beat
                interval = targets[tar + 1] - targets[tar]
            else:
                # closet target is after the current beat
                interval = targets[tar] - targets[tar - 1]
        # set the error in the array
        errors[det] = diff / interval
    # return the beat errors
    return errors


def map_errors(errors, bins):
    """
    Map the errors to the given histogram bins.

    :param errors: sequence of relative errors
    :param bins: histogram bins for mapping
    :returns: error histogram

    """
    # function [entropy,rawBinVals] = FindEntropy(beatError,hist_bins)
    #
    # map the relative beat errors to the range of -0.5..0.5
    mapped_errors = np.mod(errors + 0.5, -1) + 0.5
    # get bin counts for the given errors over the distribution
    error_histogram = np.histogram(mapped_errors, bins)[0].astype(np.float)
    # make the histogram circular by adding the last bin to the first one
    error_histogram[0] = error_histogram[0] + error_histogram[-1]
    # then remove the last bin
    error_histogram = error_histogram[:-1]
    # return error histogram
    return error_histogram


def calc_information_gain(error_histogram):
    """
    Calculate the information gain from the given error histogram.

    :param error_histogram: error histogram
    :returns: information gain

    """
    # make sure the bins heights sum to unity
    histogram = error_histogram / np.sum(error_histogram)
    # set 0 values to 1, to make entropy calculation well-behaved
    histogram[histogram == 0] = 1
    # calculate entropy
    entropy = - np.sum(histogram * np.log2(histogram))
    # return information gain
    return np.log2(len(error_histogram)) - entropy


def main():
    import os.path
    import argparse
    import glob
    import fnmatch

    # define parser
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    If invoked without any parameters the script evaluates pairs of files
    with the targets (.beats) and detection (.beats.txt) as simple text
    files with one beat timestamp per line.
    """)
    p.add_argument('files', metavar='files', nargs='+', help='path or files to be evaluated (list of files being filtered according to -d and -t arguments)')
    p.add_argument('-v', dest='verbose', action='store_true', help='be verbose')
    # extensions used for evaluation
    p.add_argument('-d', dest='detections', action='store', default='.beats.txt', help='extensions of the detections [default: .onsets.txt]')
    p.add_argument('-t', dest='targets', action='store', default='.beats', help='extensions of the targets [default: .onsets]')
    # parameters for evaluation
    p.add_argument('--window', action='store', default=0.2, type=float, help='evaluation window for P-score [default=0.2]')
    p.add_argument('--sigma', action='store', default=0.04, type=float, help='sigma for Cemgil accuracy [default=0.04]')
    p.add_argument('--tolerance', action='store', default=0.175, type=float, help='tolerance window for continuity accuracies [default=0.175]')
    p.add_argument('--bins', action='store', default=40, type=int, help='number of histogram bins for information gain [default=0.40]')
    p.add_argument('--skip', action='store', default=5., type=float, help='skip first N seconds for evaluation [default=5]')
    # output options
    p.add_argument('--tex', action='store_true', help='format errors for use is .tex files')
    # version
    p.add_argument('--version', action='version', version='%(prog)s 1.0 (2013-07-01)')
    # parse the arguments
    args = p.parse_args()

    # determine the files to process
    files = []
    for f in args.files:
        # check what we have (file/path)
        if os.path.isdir(f):
            # use all files in the given path
            files = glob.glob(f + '/*')
        else:
            # file was given, append to list
            files.append(f)
    # sort files
    files.sort()

    # TODO: find a better way to determine the corresponding detection/target files from a given list/path of files
    # filter target files
    tar_files = fnmatch.filter(files, "*%s" % args.targets)
    # filter detection files
    det_files = fnmatch.filter(files, "*%s" % args.detections)
    # must be the same number FIXME: find better solution which checks the names
    assert len(tar_files) == len(det_files), "different number of targets (%i) and detections (%i)" % (len(tar_files), len(det_files))

    # sum counter for all files
    avg_scores = Score()
    # evaluate all files
    for i in range(len(det_files)):
        # load the beat and annoation sequences
        detections = load_events(det_files[i])
        targets = load_events(tar_files[i])

        # remove beats and annotations that are within the first skip seconds (default=5)
        detections = filter(lambda a: a >= args.skip, detections)
        targets = filter(lambda a: a >= args.skip, targets)

        # evaluate
        score = Score()
        score.p_score = p_score(detections, targets, args.window)
        score.cemgil = cemgil(detections, targets, args.sigma)
        score.cmlc, score.cmlt, score.amlc, score.amlt = continuity(detections, targets, args.tolerance)
        score.information_gain, score.error_histogram = information_gain(detections, targets, args.bins)
        # print stats for each file
        if args.verbose:
            print det_files[i]
            score.print_errors(args.tex)
        # add to sum counter
        avg_scores += score
    # print summary
    print 'summary for %i files' % (len(det_files))
    avg_scores.average()
    avg_scores.print_errors(args.tex)

if __name__ == '__main__':
    main()
