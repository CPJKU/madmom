#!/usr/bin/env python
# encoding: utf-8
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
design decisions. For example, the beat detections and targets are not
quantized before being evaluated with F-measure, P-score and other metrics.
Hence these evaluation functions DO NOT report the exact same results/scores.
This approach was chosen, because it is simpler and produces more accurate
results.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import numpy as np

from .helpers import find_closest_matches, calc_errors, calc_absolute_errors
from .onsets import OnsetEvaluation


# helper functions for beat evaluation
def calc_intervals(events, fwd=False):
    """
    Calculate the intervals of all events to the previous / next event.

    :param events: sequence of events to be matched [seconds]
    :param fwd:    calculate the intervals to the next event [default=False]
    :returns:      the intervals [seconds]

    Note: the sequences must be ordered!

    """
    interval = np.zeros_like(events)
    if fwd:
        interval[:-1] = np.diff(events)
        # set the last interval to the same value as the second last
        interval[-1] = interval[-2]
    else:
        interval[1:] = np.diff(events)
        # set the first interval to the same value as the second
        interval[0] = interval[1]
    # return
    return interval


def find_closest_intervals(detections, targets, matches=None):
    """
    Find the closest target interval surrounding the detections.

    :param detections: sequence of events to be matched [seconds]
    :param targets:    sequence of possible matches [seconds]
    :param matches:    indices of the closest matches [default=None]
    :returns:          a list of closest target intervals [seconds]

    Note: the sequences must be ordered! To speed up the calculation, a list of
          pre-computed indices of the closest matches can be used.

    """
    # init array
    closest_interval = np.ones_like(detections)
    # init array for intervals
    # Note: if we combine the forward and backward intervals this is faster,
    # but we need expand the size accordingly
    intervals = np.zeros(len(targets) + 1)
    # intervals to previous target
    intervals[1:-1] = np.diff(targets)
    # interval from the first target to the left is the same as to the right
    intervals[0] = intervals[1]
    # interval from the last target to the right is the same as to the left
    intervals[-1] = intervals[-2]
    # Note: intervals to the next target are always those at the next index
    # determine the closest targets
    if matches is None:
        matches = find_closest_matches(detections, targets)
    # calculate the absolute errors
    errors = calc_errors(detections, targets, matches)
    # if the errors are positive, the detection is after the target
    # thus use the interval towards the next target
    closest_interval[errors > 0] = intervals[matches[errors > 0] + 1]
    # if the errors are 0 or negative, the detection is before the target or at
    # the same position; thus use the interval to previous target accordingly
    closest_interval[errors <= 0] = intervals[matches[errors <= 0]]
    # return the closest interval
    return closest_interval


def calc_relative_errors(detections, targets, matches=None):
    """
    Relative errors of the detections to the closest targets.
    The absolute error is weighted by the interval of two targets surrounding
    each detection.

    :param detections: sequence of events to be matched [seconds]
    :param targets:    sequence of possible matches [seconds]
    :param matches:    indices of the closest matches [default=None]
    :returns:          a list of relative errors to closest matches [seconds]

    Note: the sequences must be ordered! To speed up the calculation, a list of
          pre-computed indices of the closest matches can be used.

    """
    # determine the closest targets
    if matches is None:
        matches = find_closest_matches(detections, targets)
    # calculate the absolute errors
    errors = calc_errors(detections, targets, matches)
    # get the closest intervals
    intervals = find_closest_intervals(detections, targets, matches)
    # return the relative errors
    return errors / intervals


# evaluation functions for beat detection
def pscore(detections, targets, tolerance):
    """
    Calculate the P-Score accuracy.

    :param detections: sequence of estimated beat times [seconds]
    :param targets:    sequence of ground truth beat annotations [seconds]
    :param tolerance:  tolerance window (fraction of the median beat interval)
    :returns:          p-score

    "Evaluation of audio beat tracking and music tempo extraction algorithms"
    M. F. McKinney, D. Moelants, M. E. P. Davies, and A. Klapuri
    Journal of New Music Research, vol. 36, no. 1, pp. 1–16, 2007.

    """
    # since we need an interval for the calculation of the score, at least two
    # targets must be given
    # FIXME: what if only 1 target and detection are given; same with none?
    if detections.size == 0 or targets.size < 2:
        return 0
    # the error window is the given fraction of the median beat interval
    window = tolerance * np.median(np.diff(targets))
    # errors
    errors = calc_absolute_errors(detections, targets)
    # count the instances where the error is smaller or equal than the window
    p = detections[errors <= window].size
    # normalize by the max number of detections/targets
    p /= float(max(detections.size, targets.size))
    # return p-score
    return p


def cemgil(detections, targets, sigma):
    """
    Calculate the Cemgil accuracy.

    :param detections: sequence of estimated beat times [seconds]
    :param targets:    sequence of ground truth beat annotations [seconds]
    :param sigma:      sigma for Gaussian error function
    :returns:          beat tracking accuracy

    "On tempo tracking: Tempogram representation and Kalman filtering"
    A.T. Cemgil, B. Kappen, P. Desain, and H. Honing
    Journal Of New Music Research, vol. 28, no. 4, pp. 259–273, 2001

    """
    # beat accuracy is initially zero
    acc = 0
    # no detections
    if detections.size == 0:
        return acc
    # determine the absolute errors of the detections to the closest targets
    # Note: the original implementation searches for the closest matches of
    # detections to given targets. Since absolute errors > a usual beat
    # interval produce high errors (and thus in turn add negligible values to
    # the accuracy), it is safe to swap those two.
    errors = calc_absolute_errors(detections, targets)
    # apply a Gaussian error function with the given std. dev. on the errors
    acc = np.exp(-(errors ** 2.) / (2. * (sigma ** 2.)))
    # and sum up the accuracy
    acc = np.sum(acc)
    # normalized by the mean of the number of detections and targets
    acc /= 0.5 * (len(targets) + len(detections))
    # return accuracy
    return acc


# helper function for continuity calculation
def cml(detections, targets, tempo_tolerance, phase_tolerance):
    """
    Calculate cmlc, cmlt for the given detection and target sequences.

    :param detections:      sequence of estimated beat times [seconds]
    :param targets:         sequence of ground truth beat annotations [seconds]
    :param tempo_tolerance: tempo tolerance window
    :param phase_tolerance: phase (interval) tolerance window
    :returns:               cmlc, cmlt

    cmlc: tracking accuracy, continuity at the correct metrical level required
    cmlt: tracking accuracy, continuity at the correct metrical level not req.

    "Techniques for the automated analysis of musical audio"
    S. Hainsworth
    Ph.D. dissertation, Department of Engineering, Cambridge University, 2004.

    "Analysis of the meter of acoustic musical signals"
    A. P. Klapuri, A. Eronen, and J. Astola
    IEEE Transactions on Audio, Speech and Language Processing, vol. 14, no. 1,
    pp. 342–355, 2006.

    """
    # at least 2 detections and targets are needed to calculate the intervals
    if min(detections.size, targets.size) < 2:
        return 0, 0
    # determine closest targets to detections
    closest = find_closest_matches(detections, targets)
    # errors of the detections wrt. to the targets
    errors = calc_absolute_errors(detections, targets, closest)
    # detection intervals
    det_interval = calc_intervals(detections)
    # target intervals (get those intervals at the correct positions)
    tar_interval = calc_intervals(targets)[closest]
    # a detection is correct, if it fulfills 3 conditions:
    # 1) must match an annotation within a certain tolerance window
    correct = detections[errors < tempo_tolerance * tar_interval]
    # 2) same must be true for the previous detection / target combination
    # Note: Not enforced, since this condition is kind of pointless. Why not
    #       count a beat if it is correct only because the one before is not?
    #       Also, the original Matlab implementation does not enforce it.
    # 3) the interval must be within the phase tolerance
    correct_interval = detections[abs(1 - (det_interval / tar_interval)) <
                                  phase_tolerance]
    # now combine the conditions
    correct = np.intersect1d(correct, correct_interval)
    # convert on indices
    correct_idx = np.searchsorted(detections, correct)
    # add a fake start and end
    correct_idx = np.append(-5, correct_idx)
    correct_idx = np.append(correct_idx, detections.size + 5)
    # get continuous segment
    segments = np.nonzero(np.diff(correct_idx) != 1)[0]
    # determine the max length of those segment
    if segments.size == 0:
        # all detections are correct
        cont = detections.size
    elif segments.size == 1:
        # only one long segment
        cont = segments
    else:
        # multiple segments
        cont = np.max(np.diff(segments))
    # maximal length of the given sequences
    length = float(max(len(detections), len(targets)))
    # accuracy for the longest continuous detections
    cmlc = cont / length
    # accuracy of all correct detections
    cmlt = len(correct) / length
    # return a tuple
    return cmlc, cmlt


def continuity(detections, targets, tempo_tolerance, phase_tolerance):
    """
    Calculate cmlc, cmlt, amlc, amlt for the given detection and target
    sequences.

    :param detections:      sequence of estimated beat times [seconds]
    :param targets:         sequence of ground truth beat annotations [seconds]
    :param tempo_tolerance: tempo tolerance window
    :param phase_tolerance: phase (interval) tolerance window
    :returns:               cmlc, cmlt, amlc, amlt beat tracking accuracies

    cmlc: tracking accuracy, continuity at the correct metrical level required
    cmlt: tracking accuracy, continuity at the correct metrical level not req.
    amlc: tracking accuracy, continuity at allowed metrical levels required
    amlt: tracking accuracy, continuity at allowed metrical levels not req.

    "Techniques for the automated analysis of musical audio"
    S. Hainsworth
    Ph.D. dissertation, Department of Engineering, Cambridge University, 2004.

    "Analysis of the meter of acoustic musical signals"
    A. P. Klapuri, A. Eronen, and J. Astola
    IEEE Transactions on Audio, Speech and Language Processing, vol. 14, no. 1,
    pp. 342–355, 2006.

    """
    # needs at least 2 detections and targets to interpolate
    if min(len(targets), len(detections)) < 2:
        return 0, 0, 0, 0

    # create a target sequence with double tempo
    same = np.arange(0, len(targets))
    shifted = np.arange(0, len(targets), 0.5)
    # Note: it does not extrapolate the last value, so skip it
    double_targets = np.interp(shifted, same, targets)[:-1]

    # make different variants of annotations
    # double tempo, odd off-beats, even off-beats
    # half tempo odd beats (i.e. 1,3,1,3), half tempo even beats (i.e. 2,4,2,4)
    # TODO: include a third tempo as well? This might be needed for fast Waltz
    variations = [double_targets, double_targets[1::2], targets[::2],
                  targets[1::2], targets[::3]]

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
        # Note: do not use the cached values for the closest matches
        c, t = cml(detections, targets_variation, tempo_tolerance,
                   phase_tolerance)
        amlc = max(amlc, c)
        amlt = max(amlt, t)

    # return a tuple
    return cmlc, cmlt, amlc, amlt


def information_gain(detections, targets, bins):
    """
    Calculate information gain.

    :param detections: sequence of estimated beat times [seconds]
    :param targets:    sequence of ground truth beat annotations [seconds]
    :param bins:       number of bins for the error histogram
    :returns:          information gain, beat error histogram

    "Measuring the performance of beat tracking algorithms algorithms using a
    beat error histogram"
    M. E. P. Davies, N. Degara and M. D. Plumbley
    IEEE Signal Processing Letters, vol. 18, vo. 3, 2011

    """
    # in case of no detections
    if detections.size == 0 or targets.size < 2:
        # return information gain = 0 and a uniform beat error histogram
        return 0, np.ones(bins) * len(targets) / bins

    # only allow even number of bins
    if bins % 2 != 0:
        raise ValueError("Number of error histogram bins must be even")

    # create bins for the error histogram that cover the range from -0.5 to 0.5
    # make the first and last bin half as wide as the rest, so that the last
    # and the first can be added together later (make the histogram circular)

    # since np.histogram uses bin borders, we need to apply an offset
    offset = 0.5 / bins
    # and add another bin, because the last bin is wrapped around to the first
    # bin later
    histogram_bins = np.linspace(-0.5 - offset, 0.5 + offset, bins + 2)

    # evaluate detections against targets
    fwd_histogram = error_histogram(detections, targets, histogram_bins)
    fwd_ig = calc_information_gain(fwd_histogram)

    # in case of only few (but correct) detections, the errors could be small
    # thus evaluate also the targets against the detections, i.e. simulate a
    # lot of false positive detections. Do not use the cached matches!
    bwd_histogram = error_histogram(targets, detections, histogram_bins)
    bwd_ig = calc_information_gain(bwd_histogram)

    # only use the lower information gain
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
    :param targets:    sequence of ground truth beat annotations [seconds]
    :param bins:       histogram bins for mapping
    :returns:          error histogram

    """
    # get the relative errors of the detections to the targets
    errors = calc_relative_errors(detections, targets)
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
    :returns:               information gain

    """
    # copy the error_histogram, because it must not be altered
    histogram = np.copy(error_histogram)
    # all bins are 0, make a uniform distribution with values != 0
    if not histogram.any():
        # Note: this is needed, otherwise a histogram with all bins = 0 would
        # return the maximum possible information gain because the
        # normalization in the next step would fail
        histogram += 1
    # normalize the histogram
    histogram /= np.sum(histogram)
    # set 0 values to 1, to make entropy calculation well-behaved
    histogram[histogram == 0] = 1
    # calculate entropy
    entropy = - np.sum(histogram * np.log2(histogram))
    # return information gain
    return np.log2(histogram.size) - entropy


# default evaluation values
WINDOW = 0.07
TOLERANCE = 0.2
SIGMA = 0.04
TEMPO_TOLERANCE = 0.175
PHASE_TOLERANCE = 0.175
BINS = 40


# basic beat evaluation
class BeatEvaluation(OnsetEvaluation):
    # this class inherits from OnsetEvaluation the Precision, Recall, and
    # F-measure evaluation stuff; only the evaluation window is adjusted
    """
    Beat evaluation class.

    """
    def __init__(self, detections, targets, window=WINDOW, tolerance=TOLERANCE,
                 sigma=SIGMA, tempo_tolerance=TEMPO_TOLERANCE,
                 phase_tolerance=PHASE_TOLERANCE, bins=BINS):
        """
        Evaluate the given detection and target sequences.

        :param detections:      sequence of estimated beat times [seconds]
        :param targets:         sequence of ground truth beat annotations
                                [seconds]
        :param window:          F-measure evaluation window
                                [seconds, default=0.07]
        :param tolerance:       P-Score tolerance of median beat interval
                                [default=0.2]
        :param sigma:           sigma of Gaussian window for Cemgil accuracy
                                [default=0.04]
        :param tempo_tolerance: tempo tolerance window for [AC]ML[ct]
                                [default=0.175]
        :param phase_tolerance: phase (interval) tolerance window for
                                [AC]ML[ct] [default=0.175]
        :param bins:            number of bins for the error histogram
                                [default=40]

        """
        self.detections = detections
        self.targets = targets
        # set the window for precision, recall & fmeasure to 0.07
        super(BeatEvaluation, self).__init__(detections, targets, window)
        self.tolerance = tolerance
        self.sigma = sigma
        self.tempo_tolerance = tempo_tolerance
        self.phase_tolerance = phase_tolerance
        self.bins = bins
        # scores
        self._fmeasure = None
        self._pscore = None
        self._cemgil = None
        self._cmlc = None
        self._cmlt = None
        self._amlc = None
        self._amlt = None
        # information gain stuff
        self._information_gain = None
        self._error_histogram = None

    @property
    def num(self):
        """Number of evaluated files."""
        return 1

    @property
    def pscore(self):
        """P-Score."""
        if self._pscore is None:
            self._pscore = pscore(self.detections, self.targets,
                                  self.tolerance)
        return self._pscore

    @property
    def cemgil(self):
        """Cemgil accuracy."""
        if self._cemgil is None:
            self._cemgil = cemgil(self.detections, self.targets, self.sigma)
        return self._cemgil

    def _calc_continuity(self):
        """Perform continuity evaluation."""
        # calculate scores
        scores = continuity(self.detections, self.targets,
                            self.tempo_tolerance, self.phase_tolerance)
        self._cmlc, self._cmlt, self._amlc, self._amlt = scores

    @property
    def cmlc(self):
        """CMLc."""
        if self._cmlc is None:
            self._calc_continuity()
        return self._cmlc

    @property
    def cmlt(self):
        """CMLt."""
        if self._cmlt is None:
            self._calc_continuity()
        return self._cmlt

    @property
    def amlc(self):
        """AMLc."""
        if self._amlc is None:
            self._calc_continuity()
        return self._amlc

    @property
    def amlt(self):
        """AMLt."""
        if self._amlt is None:
            self._calc_continuity()
        return self._amlt

    def __information_gain(self):
        """Perform continuity evaluation."""
        # calculate score and error histogram
        ig_eh = information_gain(self.detections, self.targets, self.bins)
        self._information_gain, self._error_histogram = ig_eh

    @property
    def information_gain(self):
        """Information gain."""
        if self._information_gain is None:
            self.__information_gain()
        return self._information_gain

    @property
    def global_information_gain(self):
        """Global information gain."""
        # Note: if only 1 file is evaluated, it is the same as information gain
        return self.information_gain

    @property
    def error_histogram(self):
        """Error histogram."""
        if self._error_histogram is None:
            self.__information_gain()
        return self._error_histogram

    def print_errors(self, tex=False):
        """
        Print errors.

        :param tex: output format to be used in .tex files [default=False]

        """
        # report the scores always in the range 0..1, because of formatting
        print '  F-measure: %.3f P-score: %.3f Cemgil: %.3f CMLc: %.3f CMLt: '\
              '%.3f AMLc: %.3f AMLt: %.3f D: %.3f Dg: %.3f' % (self.fmeasure,
              self.pscore, self.cemgil, self.cmlc, self.cmlt, self.amlc,
              self.amlt, self.information_gain, self.global_information_gain)
        if tex:
            print 'tex & F-measure & P-score & Cemgil & CMLc & CMLt & AMLc & '\
                  'AMLt & D & Dg \\\\'
            print '%i file(s) & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & '\
                  '%.3f & %.3f & %.3f\\\\' % (self.num, self.fmeasure,
                  self.pscore, self.cemgil, self.cmlc, self.cmlt, self.amlc,
                  self.amlt, self.information_gain,
                  self.global_information_gain)


class MeanBeatEvaluation(BeatEvaluation):
    """
    Class for averaging beat evaluation scores.

    """

    def __init__(self, other=None):
        """
        MeanBeatEvaluation object can be either instantiated as an empty object
        or by passing in a BeatEvaluation object with the scores taken from
        that object.

        :param other: BeatEvaluation object

        """
        # simple scores
        self._fmeasure = np.empty(0)
        self._pscore = np.empty(0)
        self._cemgil = np.empty(0)
        # continuity scores
        self._cmlc = np.empty(0)
        self._cmlt = np.empty(0)
        self._amlc = np.empty(0)
        self._amlt = np.empty(0)
        # information gain stuff
        self._information_gain = np.empty(0)
        self._error_histogram = None
        # instance can be initialized with a Evaluation object
        if isinstance(other, BeatEvaluation):
            # add this object to self
            self += other

    # for adding another BeatEvaluation object
    def __add__(self, other):
        """
        Appends the scores of another BeatEvaluation object to the respective
        arrays.

        :param other: BeatEvaluation object

        """
        if isinstance(other, BeatEvaluation):
            self._fmeasure = np.append(self._fmeasure, other.fmeasure)
            self._pscore = np.append(self._pscore, other.pscore)
            self._cemgil = np.append(self._cemgil, other.cemgil)
            self._cmlc = np.append(self._cmlc, other.cmlc)
            self._cmlt = np.append(self._cmlt, other.cmlt)
            self._amlc = np.append(self._amlc, other.amlc)
            self._amlt = np.append(self._amlt, other.amlt)
            self._information_gain = np.append(self._information_gain,
                                               other.information_gain)
            # the error histograms needs special treatment
            if self._error_histogram is None:
                # if it is the first, just take this histogram
                self._error_histogram = other.error_histogram
            else:
                # otherwise just add them
                self._error_histogram += other.error_histogram
            return self
        else:
            return NotImplemented

    @property
    def num(self):
        """Number of evaluated files."""
        return len(self._fmeasure)

    @property
    def fmeasure(self):
        """F-measure."""
        return np.mean(self._fmeasure)

    @property
    def pscore(self):
        """P-Score."""
        return np.mean(self._pscore)

    @property
    def cemgil(self):
        """Cemgil accuracy."""
        return np.mean(self._cemgil)

    @property
    def cmlc(self):
        """CMLc."""
        return np.mean(self._cmlc)

    @property
    def cmlt(self):
        """CMLt."""
        return np.mean(self._cmlt)

    @property
    def amlc(self):
        """AMLc."""
        return np.mean(self._amlc)

    @property
    def amlt(self):
        """AMLt."""
        return np.mean(self._amlt)

    @property
    def information_gain(self):
        """Information gain."""
        return np.mean(self._information_gain)

    @property
    def global_information_gain(self):
        """Global information gain."""
        if self.error_histogram is None:
            return 0.
        return calc_information_gain(self.error_histogram)

    @property
    def error_histogram(self):
        """Error histogram."""
        return self._error_histogram


SKIP = 5.


def parser():
    """
    Create a parser and parse the arguments.

    :return: the parsed arguments

    """
    import argparse

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    If invoked without any parameters the script evaluates pairs of files
    with the targets (.beats) and detection (.beats.txt) as simple text
    files with one beat timestamp per line.
    """)
    p.add_argument('files', nargs='*',
                   help='files (or folder) to be evaluated')
    # extensions used for evaluation
    p.add_argument('-d', dest='det_ext', action='store', default='.beats.txt',
                   help='extensions of the detections [default: .beats.txt]')
    p.add_argument('-t', dest='tar_ext', action='store', default='.beats',
                   help='extensions of the targets [default: .beats]')
    # parameters for evaluation
    p.add_argument('--window', action='store', type=float, default=WINDOW,
                   help='evaluation window for F-measure [seconds, default=%f]'
                   % WINDOW)
    p.add_argument('--tolerance', action='store', type=float,
                   default=TOLERANCE,
                   help='evaluation tolerance for P-score [default=%f]'
                   % TOLERANCE)
    p.add_argument('--sigma', action='store', default=SIGMA, type=float,
                   help='sigma for Cemgil accuracy [default=%f]' % SIGMA)
    p.add_argument('--tempo_tolerance', action='store', type=float,
                   default=TEMPO_TOLERANCE,
                   help='tempo tolerance window for continuity accuracies '
                        '[default=%f]' % TEMPO_TOLERANCE)
    p.add_argument('--phase_tolerance', action='store', type=float,
                   default=PHASE_TOLERANCE,
                   help='phase tolerance window for continuity accuracies '
                        '[default=%f]' % PHASE_TOLERANCE)
    p.add_argument('--bins', action='store', type=int, default=BINS,
                   help='number of histogram bins for information gain '
                        '[default=%i]' % BINS)
    p.add_argument('--skip', action='store', type=float, default=SKIP,
                   help='skip first N seconds for evaluation [default=%f]'
                   % SKIP)
    # output options
    p.add_argument('--tex', action='store_true',
                   help='format errors for use in .tex files')
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
    Simple beat evaluation.

    """
    from ..utils.helpers import files, match_file, load_events

    # parse arguments
    args = parser()

    # get detection and target files
    det_files = files(args.files, args.det_ext)
    tar_files = files(args.files, args.tar_ext)
    # quit if no files are found
    if len(det_files) == 0:
        print "no files to evaluate. exiting."
        exit()

    # mean evaluation for all files
    mean_eval = MeanBeatEvaluation()
    # evaluate all files
    for det_file in det_files:
        # get the detections file
        detections = load_events(det_file)
        # get the matching target files
        matches = match_file(det_file, tar_files, args.det_ext, args.tar_ext)
        # quit if any file does not have a matching target file
        if len(matches) == 0:
            print " can't find a target file found for %s" % det_file
            exit()
        # do a mean evaluation with all matched target files
        me = MeanBeatEvaluation()
        for tar_file in matches:
            # load the targets
            targets = load_events(tar_file)
            # remove beats and annotations that are within the first N seconds
            if args.skip > 0:
                # FIXME: this definitely alters the results
                detections = detections[np.where(detections > args.skip)]
                targets = targets[np.where(targets > args.skip)]
            # add the BeatEvaluation this file's mean evaluation
            me += BeatEvaluation(detections, targets, window=args.window,
                                 tolerance=args.tolerance, sigma=args.sigma,
                                 tempo_tolerance=args.tempo_tolerance,
                                 phase_tolerance=args.phase_tolerance,
                                 bins=args.bins)
            # process the next target file
        # print stats for each file
        if args.verbose:
            print det_file
            me.print_errors(args.tex)
        # add this file's mean evaluation to the global evaluation
        mean_eval += me
        # process the next detection file
    # print summary
    print 'mean for %i files:' % (len(det_files))
    mean_eval.print_errors(args.tex)

if __name__ == '__main__':
    main()
