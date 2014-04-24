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
design decisions. For example, the beat detections and annotations are not
quantized before being evaluated with F-measure, P-score and other metrics.
Hence these evaluation functions DO NOT report the exact same results/scores.
This approach was chosen, because it is simpler and produces more accurate
results.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import warnings
import numpy as np

from . import find_closest_matches, calc_errors, calc_absolute_errors
from .onsets import OnsetEvaluation


# helper functions for beat evaluation
def calc_intervals(events, fwd=False):
    """
    Calculate the intervals of all events to the previous/next event.

    :param events: numpy array with the detected events [float, seconds]
    :param fwd:    calculate the intervals towards the next event [bool]
    :returns:      the intervals [seconds]

    Note: The sequences must be ordered!

    """
    # at least 2 events must be given to calculate an interval
    if len(events) < 2:
        return np.zeros(0, dtype=np.float)
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


def find_closest_intervals(detections, annotations, matches=None):
    """
    Find the closest annotated interval for each beat detection.

    :param detections:  numpy array with the detected beats [float, seconds]
    :param annotations: numpy array with the annotated beats [float, seconds]
    :param matches:     numpy array with indices of the closest beats [int]
    :returns:           numpy array with closest annotated intervals [seconds]

    Note: The sequences must be ordered! To speed up the calculation, a list of
          pre-computed indices of the closest matches can be used.

          The function does NOT test if each detection has a surrounding
          interval, it always returns the closest interval.

    """
    # at least 1 detection and 2 annotations must be given
    if len(detections) < 1 or len(annotations) < 2:
        return np.zeros(0, dtype=np.float)
    # init array
    closest_interval = np.ones_like(detections)
    # intervals
    # Note: it is faster if we combine the forward and backward intervals,
    #       but we need to take care of the sizes; intervals to the next
    #       annotation are always the same as those at the next index
    intervals = np.zeros(len(annotations) + 1)
    # intervals to previous annotation
    intervals[1:-1] = np.diff(annotations)
    # interval of the first annotation to the left is the same as to the right
    intervals[0] = intervals[1]
    # interval of the last annotation to the right is the same as to the left
    intervals[-1] = intervals[-2]
    # determine the closest annotations
    if matches is None:
        matches = find_closest_matches(detections, annotations)
    # calculate the absolute errors
    errors = calc_errors(detections, annotations, matches)
    # if the errors are positive, the detection is after the annotation
    # thus use the interval towards the next annotation
    closest_interval[errors > 0] = intervals[matches[errors > 0] + 1]
    # if the errors are 0 or negative, the detection is before the annotation
    # or at the same position; thus use the interval to previous annotation
    closest_interval[errors <= 0] = intervals[matches[errors <= 0]]
    # return the closest interval
    return closest_interval


def find_longest_continuous_segment(sequence_indices):
    """
    Find the longest consecutive segment in the given sequence_indices.

    :param sequence_indices: numpy array with the indices of the events [int]
    :returns:               length and start position of the longest continuous
                            segment [(int, int)]

    """
    # continuous segments hve consecutive indices, i.e. diffs =! 1 are
    # boundaries between continuous segments; add 1 to get the correct index
    boundaries = np.nonzero(np.diff(sequence_indices) != 1)[0] + 1
    # add a start (index 0) and stop (length of correct detections) to the
    # segment boundary indices
    boundaries = np.concatenate(([0], boundaries, [len(sequence_indices)]))
    # lengths of the individual segments
    segment_lengths = np.diff(boundaries)
    # return the length and start position of the longest continuous segment
    return np.max(segment_lengths), boundaries[np.argmax(segment_lengths)]


def calc_relative_errors(detections, annotations, matches=None):
    """
    Errors of the detections relative to the closest annotated interval.

    :param detections:  numpy array with the detected beats [float, seconds]
    :param annotations: numpy array with the annotated beats [float, seconds]
    :param matches:     numpy array with indices of the closest beats [int]
    :returns:           numpy array with errors relative to surrounding
                        annotated interval [seconds]

    Note: The sequences must be ordered! To speed up the calculation, a list of
          pre-computed indices of the closest matches can be used.

    """
    # at least 1 detection and 2 annotations must be given
    if len(detections) < 1 or len(annotations) < 2:
        return np.zeros(0, dtype=np.float)
    # determine the closest annotations
    if matches is None:
        matches = find_closest_matches(detections, annotations)
    # calculate the absolute errors
    errors = calc_errors(detections, annotations, matches)
    # get the closest intervals
    intervals = find_closest_intervals(detections, annotations, matches)
    # return the relative errors
    return errors / intervals


# evaluation functions for beat detection
def pscore(detections, annotations, tolerance):
    """
    Calculate the P-Score accuracy for the given detections and annotations.

    :param detections:  numpy array with the detected beats [float, seconds]
    :param annotations: numpy array with the annotated beats [float, seconds]
    :param tolerance:   tolerance window (fraction of the median beat interval)
    :returns:           p-score

    "Evaluation of audio beat tracking and music tempo extraction algorithms"
    M. McKinney, D. Moelants, M. Davies and A. Klapuri
    Journal of New Music Research, vol. 36, no. 1, pp. 1–16, 2007.

    """
    # neither detections nor annotations
    if len(detections) == 0 and len(annotations) == 0:
        return 1.
    # at least 1 detection and 2 annotations must be given
    if len(detections) < 1 or len(annotations) < 2:
        return 0.
    # tolerance must be greater than 0
    if tolerance <= 0:
        raise ValueError("tolerance must be greater than 0")
    # the error window is the given fraction of the median beat interval
    window = tolerance * np.median(np.diff(annotations))
    # errors
    errors = calc_absolute_errors(detections, annotations)
    # count the instances where the error is smaller or equal than the window
    p = len(detections[errors <= window])
    # normalize by the max number of detections/annotations
    p /= float(max(len(detections), len(annotations)))
    # return p-score
    return p


def cemgil(detections, annotations, sigma):
    """
    Calculate the Cemgil accuracy for the given detections and annotations.

    :param detections:  numpy array with the detected beats [float, seconds]
    :param annotations: numpy array with the annotated beats [float, seconds]
    :param sigma:       sigma for Gaussian error function [float]
    :returns:           beat tracking accuracy

    "On tempo tracking: Tempogram representation and Kalman filtering"
    A.T. Cemgil, B. Kappen, P. Desain, and H. Honing
    Journal Of New Music Research, vol. 28, no. 4, pp. 259–273, 2001

    """
    # neither detections nor annotations
    if len(detections) == 0 and len(annotations) == 0:
        return 1.
    # at least 1 detection and annotation must be given
    if len(detections) < 1 or len(annotations) < 1:
        return 0.
    # sigma must be greater than 0
    if sigma <= 0:
        raise ValueError("sigma must be greater than 0")
    # determine the abs. errors of the detections to the closest annotations
    # Note: the original implementation searches for the closest matches of
    #       detections given the annotations. Since absolute errors > a usual
    #       beat interval produce high errors (and thus in turn add negligible
    #       values to the accuracy), it is safe to swap those two.
    errors = calc_absolute_errors(detections, annotations)
    # apply a Gaussian error function with the given std. dev. on the errors
    acc = np.exp(-(errors ** 2.) / (2. * (sigma ** 2.)))
    # and sum up the accuracy
    acc = np.sum(acc)
    # normalized by the mean of the number of detections and annotations
    acc /= 0.5 * (len(annotations) + len(detections))
    # return accuracy
    return acc


def goto(detections, annotations, threshold, mu, sigma):
    """
    Calculate the Goto and Muraoka accuracy for the given detections and
    annotations.

    :param detections:  numpy array with the detected beats [float, seconds]
    :param annotations: numpy array with the annotated beats [float, seconds]
    :param threshold:  threshold [float]
    :param mu:         mu [float]
    :param sigma:      sigma for Gaussian error function [float]
    :returns:          beat tracking accuracy

    "Issues in evaluating beat tracking systems"
    M. Goto and Y. Muraoka
    Working Notes of the IJCAI-97 Workshop on Issues in AI and Music -
    Evaluation and Assessment, pp. 9–16, 1997

    """
    # neither detections nor annotations
    if len(detections) == 0 and len(annotations) == 0:
        return 1.
    # at least 1 detection and 2 annotations must be given
    if len(detections) < 1 or len(annotations) < 2:
        return 0.
    # get the indices of the closest detections to the annotations to determine
    # the longest continuous segment
    closest = find_closest_matches(annotations, detections)
    # keep only those which have abs(errors) <= threshold
    # Note: both the original paper and the Matlab implementation normalize by
    #       half a beat interval, thus our threshold is halved (same applies to
    #       sigma and mu)
    # errors of the detections relative to the surrounding annotation interval
    errors = calc_relative_errors(detections, annotations)
    # the absolute error must be smaller than the given threshold
    closest = closest[np.abs(errors[closest]) <= threshold]
    # get the length and start position of the longest continuous segment
    length, start = find_longest_continuous_segment(closest)
    # three conditions must be met to identify the segment as correct
    # 1) the length of the segment must be at least 1/4 of the total length
    # Note: the original paper requires that the first element must occur
    #       within the first 3/4 of the excerpt, but this was altered in the
    #       Matlab implementation to the above condition to be able to deal
    #       with audio with varying tempo
    if length < 0.25 * len(annotations):
        return 0.
    # errors of the longest segment
    segment_errors = errors[closest[start: start + length]]
    # 2) mean of the errors must not exceed mu
    if np.mean(np.abs(segment_errors)) > mu:
        return 0.
    # 3) std deviation of the errors must not exceed sigma
    # Note: contrary to the original paper and in line with the Matlab code,
    #       we calculate the std. deviation based on the raw errors and not on
    #       their absolute values.
    if np.std(segment_errors) > sigma:
        return 0.
    # otherwise return 1
    return 1.


def cml(detections, annotations, tempo_tolerance, phase_tolerance):
    """
    Helper function to calculate the cmlc and cmlt scores for the given
    detections and annotations.

    :param detections:      numpy array with the detected beats
                            [float, seconds]
    :param annotations:     numpy array with the annotated beats
                            [float, seconds]
    :param tempo_tolerance: tempo tolerance window [float]
    :param phase_tolerance: phase (interval) tolerance window [float]
    :returns:               cmlc, cmlt

    "Techniques for the automated analysis of musical audio"
    S. Hainsworth
    Ph.D. dissertation, Department of Engineering, Cambridge University, 2004.

    "Analysis of the meter of acoustic musical signals"
    A. P. Klapuri, A. Eronen, and J. Astola
    IEEE Transactions on Audio, Speech and Language Processing, vol. 14, no. 1,
    pp. 342–355, 2006.

    """
    # neither detections nor annotations
    if len(detections) == 0 and len(annotations) == 0:
        return 1.
    # at least 2 detections and annotations must be given
    if len(detections) < 2 or len(annotations) < 2:
        return 0., 0.
    # tolerances must be greater than 0
    if tempo_tolerance <= 0 or phase_tolerance <= 0:
        raise ValueError("tolerances must be greater than 0")

    # determine closest annotations to detections
    closest = find_closest_matches(detections, annotations)
    # errors of the detections wrt. to the annotations
    errors = calc_absolute_errors(detections, annotations, closest)
    # detection intervals
    det_interval = calc_intervals(detections)
    # annotation intervals (get those intervals at the correct positions)
    ann_interval = calc_intervals(annotations)[closest]
    # a detection is correct, if it fulfills 3 conditions:
    # 1) must match an annotation within a certain tolerance window
    correct_tempo = detections[errors <= ann_interval * tempo_tolerance]
    # 2) same must be true for the previous detection / annotation combination
    # Note: Not enforced, since this condition is kind of pointless. Why not
    #       count a correct beat just because its predecessor is not?
    #       Also, the original Matlab implementation does not enforce it.
    # 3) the interval must be within the phase tolerance
    correct_phase = detections[abs(1 - (det_interval / ann_interval)) <=
                               phase_tolerance]
    # combine the conditions
    correct = np.intersect1d(correct_tempo, correct_phase)
    # convert to indices
    correct_idx = np.searchsorted(detections, correct)
    # cmlc: longest continuous segment of detections normalized by the max.
    #       length of both sequences (detection and annotations)
    length = float(max(len(detections), len(annotations)))
    longest, _ = find_longest_continuous_segment(correct_idx)
    cmlc = longest / length
    # cmlt: same but for all detections (no need for continuity)
    cmlt = len(correct) / length
    # return a tuple
    return cmlc, cmlt


def continuity(detections, annotations, tempo_tolerance, phase_tolerance,
               double=True, triple=True):
    """
    Calculate the cmlc, cmlt, amlc and amlt scores for the given detections and
    annotations.

    :param detections:      numpy array with the detected beats
                            [float, seconds]
    :param annotations:     numpy array with the annotated beats
                            [float, seconds]
    :param tempo_tolerance: tempo tolerance window [float]
    :param phase_tolerance: phase (interval) tolerance window [float]
    :param double:          include 2x and 1/2x tempo variations
    :param triple:          include 3x and 1/3x tempo variations
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
    # at least 2 detection and annotations must be given
    if len(detections) < 2 or len(annotations) < 2:
        return 0., 0., 0., 0.

    # evaluate the correct tempo
    cmlc, cmlt = cml(detections, annotations, tempo_tolerance, phase_tolerance)
    amlc = cmlc
    amlt = cmlt
    # speed up calculation by skipping other metrical levels if the score is
    # higher than 0.5 already. We must have tested the correct metrical level
    # already, otherwise the cmlc score would be lower.
    if cmlc > 0.5:
        return cmlc, cmlt, amlc, amlt

    # create a annotation sequence with double tempo
    same = np.arange(0, len(annotations))
    shifted = np.arange(0, len(annotations), 0.5)
    double_annotations = np.interp(shifted, same, annotations)
    # np.interp does not extrapolate, so do this manually
    double_annotations[-1] += np.diff(double_annotations[:-1])[-1]
    # create different variants of the annotations:
    # same tempo, half a beat off
    variations = [double_annotations[1::2]]
    # double/half tempo variations
    if double:
        # double tempo
        variations.append(double_annotations)
        # half tempo odd beats (i.e. 1,3,1,3,..)
        variations.append(annotations[::2])
        # half tempo even beats (i.e. 2,4,2,4,..)
        variations.append(annotations[1::2])
    # triple/third tempo variations
    if triple:
        # create a annotation sequence with double tempo
        same = np.arange(0, len(annotations))
        shifted = np.arange(0, len(annotations), 1. / 3)
        triple_annotations = np.interp(shifted, same, annotations)
        # np.interp does not extrapolate, so do this manually
        extrapolated = np.diff(triple_annotations[:-2])[-2:] * np.arange(1, 3)
        triple_annotations[-2:] += extrapolated
        # triple tempo
        variations.append(triple_annotations)
        # third tempo 1st beat (1,4,3,2,..)
        variations.append(annotations[::3])
        # third tempo 2nd beat (2,1,4,3,..)
        variations.append(annotations[1::3])
        # third tempo 3rd beat (3,2,1,4,..)
        variations.append(annotations[2::3])
    # evaluate these metrical variants
    for variation in variations:
        # if other metrical levels achieve higher accuracies, take these values
        c, t = cml(detections, variation, tempo_tolerance, phase_tolerance)
        amlc = max(amlc, c)
        amlt = max(amlt, t)

    # return a tuple
    return cmlc, cmlt, amlc, amlt


def information_gain(detections, annotations, bins):
    """
    Calculate information gain for the given detections and annotations.

    :param detections:  numpy array with the detected beats [float, seconds]
    :param annotations: numpy array with the annotated beats [float, seconds]
    :param bins:        number of bins for the error histogram [int, even]
    :returns:           information gain, beat error histogram

    "Measuring the performance of beat tracking algorithms algorithms using a
    beat error histogram"
    M. E. P. Davies, N. Degara and M. D. Plumbley
    IEEE Signal Processing Letters, vol. 18, vo. 3, 2011

    Note: Since an error of 0 should map to the centre of a bin, only even
          number of bins are allowed.

    """
    # allow only even numbers and require at least 2 bins
    if bins % 2 != 0 or bins < 2:
        raise ValueError("Number of error histogram bins must be even and "
                         "greater than 0")

    # neither detections nor annotations
    if len(detections) == 0 and len(annotations) == 0:
        # return a max. information gain and an empty error histogram
        return np.log2(bins), np.zeros(bins)
    # at least 2 detections and annotations must be given
    if len(detections) < 2 or len(annotations) < 2:
        # return an information gain of 0 and a uniform beat error histogram
        # Note: Because we want flipped detections and annotations return the
        #       same uniform histogram, the maximum length of both the
        #       detections and annotations is chosen instead of just the length
        #       of the annotations as in the Matlab implementation.
        max_length = max(len(detections), len(annotations))
        return 0., np.ones(bins) * max_length / float(bins)

    # check if there are enough beat annotations for the number of bins
    if bins > len(annotations):
        warnings.warn("Not enough beat annotations (%d) for %d histogram bins."
                      % (len(annotations), bins))

    # create bins for the error histogram that cover the range from -0.5 to 0.5
    # make the first and last bin half as wide as the rest, so that the last
    # and the first bin can be added together (to make the histogram circular)

    # this is more or less accomplished automatically since np.histogram
    # accepts a sequence of bin edges instead of bin centres, but we need to
    # apply an offset and increase the number of bins by 1
    offset = 0.5 / bins
    # because the last bin is wrapped around to the first bin later on increase
    # the number of bins by a total of 2
    histogram_bins = np.linspace(-0.5 - offset, 0.5 + offset, bins + 2)

    # evaluate detections against annotations
    fwd_histogram = _error_histogram(detections, annotations, histogram_bins)
    fwd_ig = _information_gain(fwd_histogram)

    # in case of only few (but correct) detections, the errors could be small
    # thus evaluate also the annotations against the detections, i.e. simulate
    # a lot of false positive detections
    bwd_histogram = _error_histogram(annotations, detections, histogram_bins)
    bwd_ig = _information_gain(bwd_histogram)

    # only use the lower information gain
    if fwd_ig < bwd_ig:
        return fwd_ig, fwd_histogram
    else:
        return bwd_ig, bwd_histogram


def _error_histogram(detections, annotations, histogram_bins):
    """
    Helper function to calculate the relative errors of the given detections
    and annotations and map them to an error histogram with the given bins.

    :param detections:     numpy array with the detected beats [float, seconds]
    :param annotations:    numpy array with the annotated beats
                           [float, seconds]
    :param histogram_bins: sequence of histogram bin edges for mapping
    :returns:              error histogram

    Note: The returned error histogram is circular, i.e. it contains 1 bin less
          than indicated with the values of the last and first bin added and
          mapped to the first bin.

    """
    # get the relative errors of the detections to the annotations
    errors = calc_relative_errors(detections, annotations)
    # map the relative beat errors to the range of -0.5..0.5
    errors = np.mod(errors + 0.5, -1) + 0.5
    # get bin counts for the given errors over the distribution
    histogram = np.histogram(errors, histogram_bins)[0].astype(np.float)
    # make the histogram circular by adding the last bin to the first one
    histogram[0] = histogram[0] + histogram[-1]
    # then remove the last bin
    histogram = histogram[:-1]
    # return error histogram
    return histogram


def _information_gain(error_histogram):
    """
    Helper function to calculate the information gain from the given error
    histogram.

    :param error_histogram: error histogram
    :returns:               information gain

    """
    # copy the error_histogram, because it must not be altered
    histogram = np.copy(error_histogram)
    # if all bins are 0, make a uniform distribution with values != 0
    if not histogram.any():
        # Note: this is needed, otherwise a histogram with all bins = 0 would
        #       return the maximum possible information gain because the
        #       normalization in the next step would fail
        histogram += 1.
    # normalize the histogram
    histogram /= np.sum(histogram)
    # set 0 values to 1, to make entropy calculation well-behaved
    histogram[histogram == 0] = 1.
    # calculate entropy
    entropy = - np.sum(histogram * np.log2(histogram))
    # return information gain
    return np.log2(len(histogram)) - entropy


# default evaluation values
WINDOW = 0.07
TOLERANCE = 0.2
SIGMA = 0.04
GOTO_THRESHOLD = 0.175
GOTO_SIGMA = 0.1
GOTO_MU = 0.1
TEMPO_TOLERANCE = 0.175
PHASE_TOLERANCE = 0.175
DOUBLE = True
TRIPLE = True
BINS = 40


# beat evaluation class
class BeatEvaluation(OnsetEvaluation):
    # this class inherits from OnsetEvaluation the Precision, Recall, and
    # F-measure evaluation stuff but uses a different evaluation window
    """
    Beat evaluation class.

    """
    def __init__(self, detections, annotations, window=WINDOW,
                 tolerance=TOLERANCE, sigma=SIGMA,
                 goto_threshold=GOTO_THRESHOLD, goto_sigma=GOTO_SIGMA,
                 goto_mu=GOTO_MU, tempo_tolerance=TEMPO_TOLERANCE,
                 phase_tolerance=PHASE_TOLERANCE, double=DOUBLE,
                 triple=TRIPLE, bins=BINS):
        """
        Evaluate the given detections and annotations.

        :param detections:      sequence of estimated beat times [seconds]
        :param annotations:     sequence of ground truth beat annotations
                                [seconds]
        :param window:          F-measure evaluation window [seconds]
        :param tolerance:       P-Score tolerance of median beat interval
        :param sigma:           sigma of Gaussian window for Cemgil accuracy
        :param goto_threshold:  threshold for Goto error
        :param goto_sigma:      sigma for Goto error
        :param goto_mu:         mu for Goto error
        :param tempo_tolerance: tempo tolerance window for [AC]ML[ct]
        :param phase_tolerance: phase tolerance window for [AC]ML[ct]
        :param double:          include double/half tempo variations
        :param triple:          include triple/third tempo variations
        :param bins:            number of bins for the error histogram

        """
        # convert the detections and annotations
        detections = np.asarray(sorted(detections), dtype=np.float)
        annotations = np.asarray(sorted(annotations), dtype=np.float)
        # perform onset evaluation with the appropriate window
        super(BeatEvaluation, self).__init__(detections, annotations, window)
        # other scores
        self.pscore = pscore(detections, annotations, tolerance)
        self.cemgil = cemgil(detections, annotations, sigma)
        self.goto = goto(detections, annotations, goto_threshold, goto_sigma,
                         goto_mu)
        # continuity scores
        scores = continuity(detections, annotations, tempo_tolerance,
                            phase_tolerance, double, triple)
        self.cmlc, self.cmlt, self.amlc, self.amlt = scores
        # information gain stuff
        scores = information_gain(detections, annotations, bins)
        self.information_gain, self.error_histogram = scores

    @property
    def global_information_gain(self):
        """Global information gain."""
        # Note: if only 1 file is evaluated, it is the same as information gain
        return self.information_gain

    def print_errors(self, indent='', tex=False):
        """
        Print errors.

        :param indent: use the given string as indentation
        :param tex:    output format to be used in .tex files

        """
        # report the scores always in the range 0..1, because of formatting
        ret = ''
        if tex:
            # tex formatting
            ret += 'tex & F-measure & P-score & Cemgil & Goto & CMLc & CMLt &'\
                   ' AMLc & AMLt & D & Dg \\\\\n& %.3f & %.3f & %.3f & %.3f &'\
                   '%.3f & %.3f & %.3f & %.3f & %.3f & %.3f\\\\' %\
                   (self.fmeasure, self.pscore, self.cemgil, self.goto,
                    self.cmlc, self.cmlt, self.amlc, self.amlt,
                    self.information_gain, self.global_information_gain)
        else:
            # normal formatting
            ret += '%sF-measure: %.3f P-score: %.3f Cemgil: %.3f Goto: %.3f '\
                   'CMLc: %.3f CMLt: %.3f AMLc: %.3f AMLt: %.3f D: %.3f '\
                   'Dg: %.3f' %\
                   (indent, self.fmeasure, self.pscore, self.cemgil, self.goto,
                    self.cmlc, self.cmlt, self.amlc, self.amlt,
                    self.information_gain, self.global_information_gain)
        return ret

    def __str__(self):
        return self.print_errors()


class MeanBeatEvaluation(BeatEvaluation):
    """
    Class for averaging beat evaluation scores.

    """
    # we just want to inherit the print_errors() function
    def __init__(self):
        """
        Class for averaging beat evaluation scores.

        """
        # simple scores
        self._fmeasure = np.zeros(0)
        self._pscore = np.zeros(0)
        self._cemgil = np.zeros(0)
        self._goto = np.zeros(0)
        # continuity scores
        self._cmlc = np.zeros(0)
        self._cmlt = np.zeros(0)
        self._amlc = np.zeros(0)
        self._amlt = np.zeros(0)
        # information gain stuff
        self._information_gain = np.zeros(0)
        self._error_histogram = None

    # for adding another BeatEvaluation object
    def append(self, other):
        """
        Appends the scores of another BeatEvaluation object to the respective
        arrays.

        :param other: BeatEvaluation object

        """
        if isinstance(other, BeatEvaluation):
            # append the scores to the arrays
            self._fmeasure = np.append(self._fmeasure, other.fmeasure)
            self._pscore = np.append(self._pscore, other.pscore)
            self._cemgil = np.append(self._cemgil, other.cemgil)
            self._goto = np.append(self._goto, other.goto)
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
        else:
            raise TypeError('Can only append BeatEvaluation to '
                            'MeanBeatEvaluation, not %s' %
                            type(other).__name__)

    @property
    def fmeasure(self):
        """F-measure."""
        if len(self._fmeasure) == 0:
            return 0.
        return np.mean(self._fmeasure)

    @property
    def pscore(self):
        """P-Score."""
        if len(self._pscore) == 0:
            return 0.
        return np.mean(self._pscore)

    @property
    def cemgil(self):
        """Cemgil accuracy."""
        if len(self._cemgil) == 0:
            return 0.
        return np.mean(self._cemgil)

    @property
    def goto(self):
        """Goto accuracy."""
        if len(self._goto) == 0:
            return 0.
        return np.mean(self._goto)

    @property
    def cmlc(self):
        """CMLc."""
        if len(self._cmlc) == 0:
            return 0.
        return np.mean(self._cmlc)

    @property
    def cmlt(self):
        """CMLt."""
        if len(self._cmlt) == 0:
            return 0.
        return np.mean(self._cmlt)

    @property
    def amlc(self):
        """AMLc."""
        if len(self._amlc) == 0:
            return 0.
        return np.mean(self._amlc)

    @property
    def amlt(self):
        """AMLt."""
        if len(self._amlt) == 0:
            return 0.
        return np.mean(self._amlt)

    @property
    def information_gain(self):
        """Information gain."""
        if len(self._information_gain) == 0:
            return 0.
        return np.mean(self._information_gain)

    @property
    def global_information_gain(self):
        """Global information gain."""
        if self.error_histogram is None:
            return 0.
        return _information_gain(self.error_histogram)

    @property
    def error_histogram(self):
        """Error histogram."""
        return self._error_histogram


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
    with the annotations (.beats) and detection (.beats.txt) as simple text
    files with one beat timestamp per line. Suffixes can be given to filter
    the detection and annotation files.

    To maintain compatibility with the original Matlab implementation, use
    the arguments '--skip 5 --no_triple'.

    """)
    p.add_argument('files', nargs='*',
                   help='files (or folder) to be evaluated')
    # suffixes used for evaluation
    p.add_argument('-d', dest='det_suffix', action='store',
                   default='.beats.txt',
                   help='suffix of the detection files '
                        '[default: %(default)s]')
    p.add_argument('-t', dest='ann_suffix', action='store', default='.beats',
                   help='suffix of the annotation files '
                        '[default: %(default)s]')
    # parameters for evaluation
    g = p.add_argument_group('evaluation arguments')
    g.add_argument('--window', action='store', type=float, default=WINDOW,
                   help='evaluation window for F-measure '
                        '[seconds, default=%(default).3f]')
    g.add_argument('--tolerance', action='store', type=float,
                   default=TOLERANCE,
                   help='evaluation tolerance for P-score '
                        '[default=%(default).3f]')
    g.add_argument('--sigma', action='store', default=SIGMA, type=float,
                   help='sigma for Cemgil accuracy [default=%(default).3f]')
    g.add_argument('--goto_threshold', action='store', type=float,
                   default=GOTO_THRESHOLD,
                   help='threshold for Goto error [default=%(default).3f]')
    g.add_argument('--goto_sigma', action='store', type=float,
                   default=GOTO_SIGMA,
                   help='sigma for Goto error [default=%(default).3f]')
    g.add_argument('--goto_mu', action='store', type=float, default=GOTO_MU,
                   help='mu for Goto error [default=%(default).3f]')
    g.add_argument('--tempo_tolerance', action='store', type=float,
                   default=TEMPO_TOLERANCE,
                   help='tempo tolerance window for continuity accuracies '
                        '[default=%(default).3f]')
    g.add_argument('--phase_tolerance', action='store', type=float,
                   default=PHASE_TOLERANCE,
                   help='phase tolerance window for continuity accuracies '
                        '[default=%(default).3f]')
    g.add_argument('--no_double', dest='double', action='store_false',
                   default=DOUBLE,
                   help='do not include double/half tempo variations for AMLx')
    g.add_argument('--no_triple', dest='triple', action='store_false',
                   default=TRIPLE,
                   help='do not include triple/third tempo variations for '
                        'AMLx')
    g.add_argument('--bins', action='store', type=int, default=BINS,
                   help='number of histogram bins for information gain '
                        '[default=%(default)i]')
    g.add_argument('--skip', action='store', type=float, default=0,
                   help='skip first N seconds for evaluation '
                        '[default=%(default).3f]')
    # output options
    g = p.add_argument_group('formatting arguments')
    g.add_argument('--tex', action='store_true',
                   help='format errors for use in .tex files')
    # verbose
    p.add_argument('-v', dest='verbose', action='count',
                   help='increase verbosity level')
    p.add_argument('-s', dest='silent', action='store_true',
                   help='suppress warnings')
    # parse the arguments
    args = p.parse_args()
    # print the args
    if args.verbose >= 2:
        print args
    if args.silent:
        warnings.filterwarnings("ignore")
    # return
    return args


def main():
    """
    Simple beat evaluation.

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
    mean_eval = MeanBeatEvaluation()
    # evaluate all files
    for det_file in det_files:
        # load the detections
        detections = load_events(det_file)
        # get the matching annotation files
        matches = match_file(det_file, ann_files,
                             args.det_suffix, args.ann_suffix)
        # quit if any file does not have a matching annotation file
        if len(matches) == 0:
            print " can't find a annotation file found for %s" % det_file
            exit()
        # do a mean evaluation with all matched annotation files
        me = MeanBeatEvaluation()
        for ann_file in matches:
            # load the annotations
            annotations = load_events(ann_file)
            # remove beats and annotations that are within the first N seconds
            if args.skip > 0:
                # FIXME: this definitely alters the results
                start_idx = np.searchsorted(detections, args.skip, 'right')
                detections = detections[start_idx:]
                start_idx = np.searchsorted(annotations, args.skip, 'right')
                annotations = annotations[start_idx:]
            # add the BeatEvaluation this file's mean evaluation
            me.append(BeatEvaluation(detections, annotations,
                                     window=args.window,
                                     tolerance=args.tolerance,
                                     sigma=args.sigma,
                                     goto_threshold=args.goto_threshold,
                                     goto_sigma=args.goto_sigma,
                                     goto_mu=args.goto_mu,
                                     tempo_tolerance=args.tempo_tolerance,
                                     phase_tolerance=args.phase_tolerance,
                                     double=args.double, triple=args.triple,
                                     bins=args.bins))
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
