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
before being evaluated with F-measure, P-score and other metrics. Hence these
evaluation functions DO NOT report the exact same results/scores. This approach
was chosen, because it is simpler and produces more accurate results.

Please send any comments, enhancements, errata, etc. to the main author.
"""

import math
import numpy as np

from simple import load_events, find_closest_match

from onsets import OnsetEvaluation


# evaluation function for beat detection
def pscore(detections, targets, tolerance):
    """
    Calculate the P-Score accuracy.

    :param detections: sequence of estimated beat times [seconds]
    :param targets: sequence of ground truth beat annotations [seconds]
    :param tolerance: tolerance window (fraction of the median beat interval)
    :returns: p-score

    "Evaluation of audio beat tracking and music tempo extraction algorithms"
    M. F. McKinney, D. Moelants, M. E. P. Davies, and A. Klapuri
    Journal of New Music Research, vol. 36, no. 1, pp. 1–16, 2007.

    """
    # init
    det_length = len(detections)
    tar_length = len(targets)
    p = 0
    # no detections: p-score=0
    # since we need an interval for the calculation of the score, at least two
    # targets must be given
    # TODO: which score should be returned otherwise
    if not detections or tar_length < 2:
        return p
    # error window is the given fraction of the median beat interval
    window = tolerance * np.median(np.diff(targets))
    # start with the first detection and target
    det = 0
    tar = 0
    while det < det_length and tar < tar_length:
        # calculate the diff between first detection and target
        if abs(detections[det] - targets[tar]) < window:
            # correct detection
            p += 1
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
    p /= float(max(det_length, tar_length))
    # return p-score
    return p


def cemgil(detections, targets, sigma):
    """
    Calculate the Cemgil accuracy.

    :param detections: sequence of estimated beat times [seconds]
    :param targets: sequence of ground truth beat annotations [seconds]
    :param sigma: sigma for Gaussian window
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
    closest = find_closest_match(targets, detections)
    for tar in range(len(targets)):
        # calculate the difference between the target and its closets match
        diff = abs(detections[closest[tar]] - targets[tar])
        # determine the value on the Gaussian error function and add to the acc.
        acc += math.exp(-(diff ** 2.) / (2. * (sigma ** 2.)))
    # normalize by the mean of the number of detections and targets
    acc /= 0.5 * (len(targets) + len(detections))
    # return accuracy
    return acc


def continuity(detections, targets, tempo_tolerance, phase_tolerance):
    """
    Calculate cmlc, cmlt, amlc, amlt for the given detection and target sequences.

    :param detections: sequence of estimated beat times [seconds]
    :param targets: sequence of ground truth beat annotations [seconds]
    :param tempo_tolerance: tempo tolerance window
    :param phase_tolerance: phase (interval) tolerance window
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
    cmlc, cmlt = cml(detections, targets, tempo_tolerance, phase_tolerance)
    # evaluate other metrical levels
    amlc = cmlc
    amlt = cmlt
    for targets_variation in variations:
        # speed up calculation by skipping other metrical levels if the score
        # is higher than 0.5 already. We must have tested the correct metrical
        # level already, otherwise the score would be lower.
        if amlc > 0.5:
            continue
        # if other metrical levels achieve a higher accuracy, take these values
        c, t = cml(detections, targets_variation, tempo_tolerance, phase_tolerance)
        amlc = max(amlc, c)
        amlt = max(amlt, t)

    # return a tuple
    return cmlc, cmlt, amlc, amlt


def cml(detections, targets, tempo_tolerance, phase_tolerance):
    """
    Calculate cmlc, cmlt for the given detection and target sequences.

    :param detections: sequence of estimated beat times [seconds]
    :param targets: sequence of ground truth beat annotations [seconds]
    :param tempo_tolerance: tempo tolerance window
    :param phase_tolerance: phase (interval) tolerance window
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
    # at least 2 detections and targets are needed to calculate the intervals
    if min(len(targets), len(detections)) < 2:
        return 0, 0
    # list for collecting correct detections / intervals
    correct = []
    correct_interval = []
    # determine closest targets to detections
    closest = find_closest_match(detections, targets)
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
        # determine detections which are within the tempo tolerance window
        if abs(detections[det] - targets[tar]) < tempo_tolerance * tar_interval:
            # add the detection to the correct list
            correct.append(det)
        # determine intervals which are within the phase tolerance window
        if abs(1 - (det_interval / tar_interval)) < phase_tolerance:
            # add the detection to the correct interval list
            correct_interval.append(det)
    # a detection is correct, if it fulfills 3 conditions:
    # 1) must match an annotation within a certain tolerance window
    # only detections which satisfy this condition are in the correct list
    # 2) same must be true for the previous detection / target combination
    # Note: Not enforced, since this condition is kind of pointless. Why not
    #       count a beat if it is correct only because the one before is not?
    #       Also, the original Matlab implementation does not enforce it.
    # correct = [c for c in correct if c - 1 in correct]
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


def information_gain(detections, targets, bins):
    """
    Calculate information gain.

    :param detections: sequence of estimated beat times [seconds]
    :param targets: sequence of ground truth beat annotations [seconds]
    :param bins: number of bins for the error histogram
    :returns: infromation gain, beat error histogram

    "Measuring the performance of beat tracking algorithms algorithms using a beat error histogram"
    M. E. P. Davies, N. Degara and M. D. Plumbley
    IEEE Signal Processing Letters, vol. 18, vo. 3, 2011

    """
    # in case of no detections
    if not detections or len(targets) < 2:
        # return information gain = 0 and a uniform beat error histogram
        return 0, np.ones(bins) * len(targets) / bins

    # only allow even number of bins
    if bins % 2 != 0:
        raise ValueError("Number of error histogram bins must be even")

    # create histogram bin borders
    # make the first and last bin just half as wide as the rest, thus the offset
    offset = 0.5 / bins
    # since np.histogram uses borders and not the centres as bins, one bin must
    # be added; + another, because the last bin is wraped around to the first
    # one later
    histogram_bins = np.linspace(-0.5 - offset, 0.5 + offset, bins + 2)

    # evaluate detections against targets
    fwd_histogram = error_histogram(detections, targets, histogram_bins)
    fwd_ig = calc_information_gain(fwd_histogram)

    # in case of underdetection, the errors could be very small; thus evaluate
    # also the targets against the detections (i.e. simulate a lot of FPs)
    bwd_histogram = error_histogram(targets, detections, histogram_bins)
    bwd_ig = calc_information_gain(bwd_histogram)

    # use the lower information gain
    if fwd_ig < bwd_ig:
        return fwd_ig, fwd_histogram
    else:
        return bwd_ig, bwd_histogram


# information gain helper functions
def error_histogram(detections, targets, bins):
    """
    Calculate the relative errors of the given detection wrt. the targets and
    map them to an error histogram with the given bins.

    :param detections: sequence of estimated beat times [seconds]
    :param targets: sequence of ground truth beat annotations [seconds]
    :param bins: histogram bins for mapping
    :returns: error histogram

    """
    # TODO: move calculation of relative detection errors to an extra function?
    # use a numpy array instead of a list, so we can do some math on it later
    errors = np.zeros(len(detections))
    # determine closest targets to detections
    closest = find_closest_match(detections, targets)
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
    # map the relative beat errors to the range of -0.5..0.5
    errors = np.mod(errors + 0.5, -1) + 0.5
    # get bin counts for the given errors over the distribution
    histogram = np.histogram(errors, bins)[0].astype(np.float)
    # make the histogram circular by adding the last bin to the first one
    histogram[0] = histogram[0] + histogram[-1]
    # then remove the last bin
    histogram = histogram[:-1]
    # return error histogram
    return histogram


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


# basic beat evaluation
class BeatEvaluation(OnsetEvaluation):
    """
    Beat evaluation class.

    """
    def __init__(self, detections, targets, window=0.07, tolerance=0.2, sigma=0.04, tempo_tolerance=0.175, phase_tolerance=0.175, bins=40):
        """
        Evaluate the given detection and target sequences.

        :param detections: sequence of estimated beat times [seconds]
        :param targets: sequence of ground truth beat annotations [seconds]
        :param window: F-measure evaluation window [seconds, default=0.07]
        :param tolerance: P-Score tolerance of median beat interval [default=0.2]
        :param sigma: sigma of Gaussian window for Cemgil accuracy [default=0.04]
        :param tempo_tolerance: tempo tolerance window for [AC]ML[ct] [default=0.175]
        :param phase_tolerance: phase (interval) tolerance window for [AC]ML[ct] [default=0.175]
        :param bins: number of bins for the error histogram

        """
        # set the window for precision, recall & fmeasure to 0.07
        super(BeatEvaluation, self).__init__(detections, targets, window)
        self.tolerance = tolerance
        self.sigma = sigma
        self.tempo_tolerance = tempo_tolerance
        self.phase_tolerance = phase_tolerance
        self.bins = bins
        # continuity scores
        self.__cmlc = None
        self.__cmlt = None
        self.__amlc = None
        self.__amlt = None
        # information gain stuff
        self.__information_gain = None
        self.__error_histogram = None

    @property
    def pscore(self):
        """P-Score."""
        return pscore(self.detections, self.targets, self.tolerance)

    @property
    def cemgil(self):
        """Cemgil accuracy."""
        return cemgil(self.detections, self.targets, self.sigma)

    def _calc_continuity(self):
        """Perform continuity evaluation."""
        # calculate scores
        self.__cmlc, self.__cmlt, self.__amlc, self.__amlt = continuity(self.detections, self.targets, self.tempo_tolerance, self.phase_tolerance)

    @property
    def cmlc(self):
        """CMLc."""
        if not self.__cmlc:
            self._calc_continuity()
        return self.__cmlc

    @property
    def cmlt(self):
        """CMLt."""
        if not self.__cmlt:
            self._calc_continuity()
        return self.__cmlt

    @property
    def amlc(self):
        """AMLc."""
        if not self.__amlc:
            self._calc_continuity()
        return self.__amlc

    @property
    def amlt(self):
        """AMLt."""
        if not self.__amlt:
            self._calc_continuity()
        return self.__amlt

    def _calc_information_gain(self):
        """Perform continuity evaluation."""
        # calculate score and error histogram
        self.__information_gain, self.__error_histogram = information_gain(self.detections, self.targets, self.bins)

    @property
    def information_gain(self):
        """Information gain."""
        if not self.__information_gain:
            self._calc_information_gain()
        return self.__information_gain

    @property
    def error_histogram(self):
        """Error histogram."""
        if self.__error_histogram is None:
            self._calc_information_gain()
        return self.__error_histogram

    def print_errors(self, tex=False):
        """
        Print errors.

        :param tex: output format to be used in .tex files [default=False]

        """
        # print the errors
        # report the scores always in the range 0..1, because of formatting
        try:
            # try to output the global information gain
            print '  F-measure: %.3f P-score: %.3f Cemgil: %.3f CMLc: %.3f CMLt: %.3f AMLc: %.3f AMLt: %.3f D: %.3f Dg: %.3f' % (self.fmeasure, self.pscore, self.cemgil, self.cmlc, self.cmlt, self.amlc, self.amlt, self.information_gain, self.global_information_gain)
        except AttributeError:
            # if this is not present, skip it
            print '  F-measure: %.3f P-score: %.3f Cemgil: %.3f CMLc: %.3f CMLt: %.3f AMLc: %.3f AMLt: %.3f D: %.3f' % (self.fmeasure, self.pscore, self.cemgil, self.cmlc, self.cmlt, self.amlc, self.amlt, self.information_gain)
#        if tex:
#            print "%i events & Precision & Recall & F-measure & True Positves & False Positives & Accuracy & Delay\\\\" % (self.num)
#            print "tex & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f %.1f\$\\pm\$%.1f\\,ms\\\\" % (self.precision, self.recall, self.f_measure, self.true_positive_rate, self.false_positive_rate, self.accuracy, np.mean(self.dev) * 1000., np.std(self.dev) * 1000.)


class MeanBeatEvaluation(BeatEvaluation):
    """
    Class for averaging beat evaluation scores.

    """
    def __init__(self, other=None):
        """
        MeanBeatEvaluation object can be either instanciated as an empty object
        or by passing in a BeatEvaluation object with the scores taken from that
        object.

        :param other: BeatEvaluation object

        """
        # simple scores
        self.__fmeasure = []
        self.__pscore = []
        self.__cemgil = []
        # continuity scores
        self.__cmlc = []
        self.__cmlt = []
        self.__amlc = []
        self.__amlt = []
        # information gain stuff
        self.__information_gain = []
        self.__error_histogram = None
        # instance can be initialized with a Evaluation object
        if isinstance(other, BeatEvaluation):
            # add this object to self
            self += other

    # for adding another BeatEvaluation object
    def __add__(self, other):
        """
        Apends the scores of another BeatEvaluation object to the repsective
        lists.

        :param other: BeatEvaluation object

        """
        if isinstance(other, BeatEvaluation):
            self.__fmeasure.append(other.fmeasure)
            self.__pscore.append(other.pscore)
            self.__cemgil.append(other.cemgil)
            self.__cmlc.append(other.cmlc)
            self.__cmlt.append(other.cmlt)
            self.__amlc.append(other.amlc)
            self.__amlt.append(other.amlt)
            self.__information_gain.append(other.information_gain)
            # the error histograms needs special treatment
            if self.__error_histogram is None:
                # if it is the first, just take this histogram
                self.__error_histogram = other.error_histogram
            else:
                # otherwise just add them
                self.__error_histogram += other.error_histogram
            return self
        else:
            return NotImplemented

    @property
    def fmeasure(self):
        """F-measure."""
        return np.mean(self.__fmeasure)

    @property
    def pscore(self):
        """P-Score."""
        return np.mean(self.__pscore)

    @property
    def cemgil(self):
        """Cemgil accuracy."""
        return np.mean(self.__cemgil)

    @property
    def cmlc(self):
        """CMLc."""
        return np.mean(self.__cmlc)

    @property
    def cmlt(self):
        """CMLt."""
        return np.mean(self.__cmlt)

    @property
    def amlc(self):
        """AMLc."""
        return np.mean(self.__amlc)

    @property
    def amlt(self):
        """AMLt."""
        return np.mean(self.__amlt)

    @property
    def information_gain(self):
        """Information gain."""
        return np.mean(self.__information_gain)

    @property
    def global_information_gain(self):
        """Global information gain."""
        return calc_information_gain(self.__error_histogram)

    @property
    def error_histogram(self):
        """Error histogram."""
        return self.__error_histogram


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
    # TODO: define an extra parser, which can be used for BeatEvaluation object instanciation?
    p.add_argument('--window', action='store', default=0.07, type=float, help='evaluation window for F-measure [seconds, default=0.07]')
    p.add_argument('--tolerance', action='store', default=0.2, type=float, help='evaluation tolerance for P-score [default=0.2]')
    p.add_argument('--sigma', action='store', default=0.04, type=float, help='sigma for Cemgil accuracy [default=0.04]')
    p.add_argument('--tempo_tolerance', action='store', default=0.175, type=float, help='tempo tolerance window for continuity accuracies [default=0.175]')
    p.add_argument('--phase_tolerance', action='store', default=0.175, type=float, help='phase tolerance window for continuity accuracies [default=0.175]')
    p.add_argument('--bins', action='store', default=40, type=int, help='number of histogram bins for information gain [default=0.40]')
    p.add_argument('--skip', action='store', default=5., type=float, help='skip first N seconds for evaluation [default=5]')
    # output options
    p.add_argument('--tex', action='store_true', help='format errors for use is .tex files')
    # version
    p.add_argument('--version', action='version', version='%(prog)s 0.1 (2013-07-17)')
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
    avg_scores = MeanBeatEvaluation()
    # evaluate all files
    for i in range(len(det_files)):
        # load the beat and annoation sequences
        detections = load_events(det_files[i])
        targets = load_events(tar_files[i])
        # remove beats and annotations that are within the first N seconds
        # FIXME: this definitely alters the results
        detections = filter(lambda a: a >= args.skip, detections)
        targets = filter(lambda a: a >= args.skip, targets)
        # evaluate
        score = BeatEvaluation(detections, targets, window=args.window, tolerance=args.tolerance, sigma=args.sigma, tempo_tolerance=args.tempo_tolerance, phase_tolerance=args.phase_tolerance, bins=args.bins)
        # print stats for each file
        if args.verbose:
            print det_files[i]
            score.print_errors(args.tex)
        # add to sum counter
        avg_scores += score
    # print summary
    print 'mean for %i files' % (len(det_files))
    avg_scores.print_errors(args.tex)

if __name__ == '__main__':
    main()
