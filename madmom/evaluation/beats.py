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
quantised before being evaluated with F-measure, P-score and other metrics.
Hence these evaluation functions DO NOT report the exact same results/scores.
This approach was chosen, because it is simpler and produces more accurate
results.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import warnings
import numpy as np

from . import find_closest_matches, calc_errors, calc_absolute_errors
from .onsets import OnsetEvaluation


class BeatIntervalError(Exception):
    """
    Exception to be raised whenever an interval cannot be computed.

    """

    def __init__(self, value=None):
        if value is None:
            value = "At least two beats must be present to be able to " \
                    "calculate an interval."
        self.value = value

    def __str__(self):
        return repr(self.value)


# function for sequence variations generation
def variations(sequence, offbeat=False, double=False, half=False,
               triple=False, third=False):
    """
    Create variations of the given beat sequence.

    :param sequence: numpy array with the beat sequence [float, seconds]
    :param offbeat:  create offbeat sequence
    :param double:   create double tempo sequence
    :param half:     create half tempo sequences (includes offbeat version)
    :param triple:   create triple/third tempo sequence
    :param third:    create third tempo sequence (includes offbeat version)
    :return:         list with sequence variations

    """
    # create different variants of the annotations
    sequences = []
    # double/half and offbeat variation
    if double or offbeat:
        if len(sequence) == 0:
            # if we don't a sequence, there's nothing to interpolate
            double_sequence = []
        else:
            # create a sequence with double tempo
            same = np.arange(0, len(sequence))
            # request one item less, otherwise we would extrapolate
            shifted = np.arange(0, len(sequence), 0.5)[:-1]
            double_sequence = np.interp(shifted, same, sequence)
        # same tempo, half tempo off
        if offbeat:
            sequences.append(double_sequence[1::2])
        # double/half tempo variations
        if double:
            # double tempo
            sequences.append(double_sequence)
    if half:
        # half tempo odd beats (i.e. 1,3,1,3,..)
        sequences.append(sequence[0::2])
        # half tempo even beats (i.e. 2,4,2,4,..)
        sequences.append(sequence[1::2])
    # triple/third tempo variations
    if triple:
        if len(sequence) == 0:
            # if we don't a sequence, there's nothing to interpolate
            triple_sequence = []
        else:
            # create a annotation sequence with triple tempo
            same = np.arange(0, len(sequence))
            # request two items less, otherwise we would extrapolate
            shifted = np.arange(0, len(sequence), 1. / 3)[:-2]
            triple_sequence = np.interp(shifted, same, sequence)
        # triple tempo
        sequences.append(triple_sequence)
    if third:
        # third tempo 1st beat (1,4,3,2,..)
        sequences.append(sequence[0::3])
        # third tempo 2nd beat (2,1,4,3,..)
        sequences.append(sequence[1::3])
        # third tempo 3rd beat (3,2,1,4,..)
        sequences.append(sequence[2::3])
    # return
    return sequences


# helper functions for beat evaluation
def calc_intervals(events, fwd=False):
    """
    Calculate the intervals of all events to the previous/next event.

    :param events: numpy array with the detected events [float, seconds]
    :param fwd:    calculate the intervals towards the next event [bool]
    :return:       the intervals [seconds]

    Note: The sequences must be ordered!
          The first (last) interval will be set to the same value as the
          second (second to last) interval (when used in forward mode).

    """
    # at least 2 events must be given to calculate an interval
    if len(events) < 2:
        raise BeatIntervalError
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
    Find the closest annotated interval for each beat detection. For each
    detection the interval of the annotations surrounding this detection is
    returned.

    :param detections:  numpy array with the detected beats [float, seconds]
    :param annotations: numpy array with the annotated beats [float, seconds]
    :param matches:     numpy array with indices of the closest beats [int]
    :return:            numpy array with closest annotated intervals [seconds]

    Note: The sequences must be ordered! To speed up the calculation, a list of
          pre-computed indices of the closest matches can be used.

          The function does NOT test if each detection has a surrounding
          interval, it always returns the closest interval.

    """
    # if no detection are given, return an empty interval array
    if len(detections) == 0:
        return np.zeros(0, dtype=np.float)
    # at least annotations must be given
    if len(annotations) < 2:
        raise BeatIntervalError
    # make sure the annotations and detections have a float dtype
    detections = np.asarray(detections, dtype=np.float)
    annotations = np.asarray(annotations, dtype=np.float)
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
    :return:                 length and start position of the longest
                             continuous segment [(int, int)]

    """
    # continuous segments have consecutive indices, i.e. diffs =! 1 are
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
    :return:            numpy array with errors relative to surrounding
                        annotated interval [seconds]

    Note: The sequences must be ordered! To speed up the calculation, a list of
          pre-computed indices of the closest matches can be used.

    """
    # if no detection are given, return an empty interval array
    if len(detections) == 0:
        return np.zeros(0, dtype=np.float)
    # at least annotations must be given
    if len(annotations) < 2:
        raise BeatIntervalError
    # make sure the annotations and detections have a float dtype
    detections = np.asarray(detections, dtype=np.float)
    annotations = np.asarray(annotations, dtype=np.float)
    # determine the closest annotations
    if matches is None:
        matches = find_closest_matches(detections, annotations)
    # calculate the absolute errors
    errors = calc_errors(detections, annotations, matches)
    # get the closest intervals
    intervals = find_closest_intervals(detections, annotations, matches)
    # return the relative errors
    return errors / intervals


# default beat evaluation parameter values
FMEASURE_WINDOW = 0.07
PSCORE_TOLERANCE = 0.2
CEMGIL_SIGMA = 0.04
GOTO_THRESHOLD = 0.175
GOTO_SIGMA = 0.1
GOTO_MU = 0.1
CONTINUITY_TEMPO_TOLERANCE = 0.175
CONTINUITY_PHASE_TOLERANCE = 0.175
INFORMATION_GAIN_BINS = 40


# evaluation functions for beat detection
def pscore(detections, annotations, tolerance=PSCORE_TOLERANCE):
    """
    Calculate the P-score accuracy for the given detections and annotations.

    :param detections:  numpy array with the detected beats [float, seconds]
    :param annotations: numpy array with the annotated beats [float, seconds]
    :param tolerance:   tolerance window (fraction of the median beat interval)
    :return:            p-score

    The P-score is determined by taking the sum of the cross-correlation
    between two impulse trains, representing the detections and annotations
    allowing for a small window of 20% of the median annotated interval.

    "Evaluation of audio beat tracking and music tempo extraction algorithms"
    M. McKinney, D. Moelants, M. Davies and A. Klapuri
    Journal of New Music Research, vol. 36, no. 1, pp. 1–16, 2007.

    Note: Contrary to the original implementation which samples the two impulse
          trains with 100Hz, we do not quantise the annotations and detections
          but rather count all detections falling withing the defined tolerance
          window.

    """
    # neither detections nor annotations are given, perfect score
    if len(detections) == 0 and len(annotations) == 0:
        return 1.
    # either beat detections or annotations are empty, score 0
    if (len(detections) == 0) != (len(annotations) == 0):
        return 0.
    # at least 2 annotations must be given to calculate an interval
    if len(annotations) < 2:
        raise BeatIntervalError("At least 2 annotations are needed for"
                                "P-score.")

    # tolerance must be greater than 0
    if tolerance <= 0:
        raise ValueError("Tolerance must be greater than 0.")

    # make sure the annotations and detections have a float dtype
    detections = np.asarray(detections, dtype=np.float)
    annotations = np.asarray(annotations, dtype=np.float)

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


def cemgil(detections, annotations, sigma=CEMGIL_SIGMA):
    """
    Calculate the Cemgil accuracy for the given detections and annotations.

    :param detections:  numpy array with the detected beats [float, seconds]
    :param annotations: numpy array with the annotated beats [float, seconds]
    :param sigma:       sigma for Gaussian error function [float]
    :return:            beat tracking accuracy

    "On tempo tracking: Tempogram representation and Kalman filtering"
    A.T. Cemgil, B. Kappen, P. Desain, and H. Honing
    Journal Of New Music Research, vol. 28, no. 4, pp. 259–273, 2001

    """
    # neither detections nor annotations are given, perfect score
    if len(detections) == 0 and len(annotations) == 0:
        return 1.
    # either beat detections or annotations are empty, score 0
    if (len(detections) == 0) != (len(annotations) == 0):
        return 0.

    # sigma must be greater than 0
    if sigma <= 0:
        raise ValueError("Sigma must be greater than 0.")

    # make sure the annotations and detections have a float dtype
    detections = np.asarray(detections, dtype=np.float)
    annotations = np.asarray(annotations, dtype=np.float)

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


def goto(detections, annotations, threshold=GOTO_THRESHOLD, sigma=GOTO_SIGMA,
         mu=GOTO_MU):
    """
    Calculate the Goto and Muraoka accuracy for the given detections and
    annotations.

    :param detections:  numpy array with the detected beats [float, seconds]
    :param annotations: numpy array with the annotated beats [float, seconds]
    :param threshold:  threshold [float]
    :param sigma:      sigma for Gaussian error function [float]
    :param mu:         respective µ [float]
    :return:           beat tracking accuracy

    "Issues in evaluating beat tracking systems"
    M. Goto and Y. Muraoka
    Working Notes of the IJCAI-97 Workshop on Issues in AI and Music -
    Evaluation and Assessment, pp. 9–16, 1997

    """
    # neither detections nor annotations are given, perfect score
    if len(detections) == 0 and len(annotations) == 0:
        return 1.
    # either beat detections or annotations are empty, score 0
    if (len(detections) == 0) != (len(annotations) == 0):
        return 0.
    # at least 2 annotations must be given to calculate an interval
    if len(annotations) < 2:
        raise BeatIntervalError("At least 2 annotations are needed for Goto's "
                                "score.")

    # threshold, sigma and mu must be greater than 0
    if threshold < 0 or sigma < 0 or mu < 0:
        raise ValueError("Threshold, sigma and mu must be positive.")

    # make sure the annotations and detections have a float dtype
    detections = np.asarray(detections, dtype=np.float)
    annotations = np.asarray(annotations, dtype=np.float)

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


def cml(detections, annotations, phase_tolerance=CONTINUITY_PHASE_TOLERANCE,
        tempo_tolerance=CONTINUITY_TEMPO_TOLERANCE):
    """
    Calculate the cmlc and cmlt scores for the given detections and
    annotations.

    :param detections:      numpy array with the detected beats
                            [float, seconds]
    :param annotations:     numpy array with the annotated beats
                            [float, seconds]
    :param phase_tolerance: phase tolerance window [float]
    :param tempo_tolerance: tempo tolerance window [float]
    :return:                cmlc, cmlt

    "Techniques for the automated analysis of musical audio"
    S. Hainsworth
    Ph.D. dissertation, Department of Engineering, Cambridge University, 2004.

    "Analysis of the meter of acoustic musical signals"
    A. P. Klapuri, A. Eronen, and J. Astola
    IEEE Transactions on Audio, Speech and Language Processing, vol. 14, no. 1,
    pp. 342–355, 2006.

    """
    # neither detections nor annotations are given
    if len(detections) == 0 and len(annotations) == 0:
        return 1., 1.
    # either beat detections or annotations are empty, score 0
    if (len(detections) == 0) != (len(annotations) == 0):
        return 0., 0.
    # at least 2 annotations must be given to calculate an interval
    if len(annotations) < 2:
        raise BeatIntervalError("At least 2 annotations are needed for "
                                "continuity scores, %s given." % annotations)
    # TODO: remove this, see TODO below
    if len(detections) < 2:
        raise BeatIntervalError("At least 2 detections are needed for"
                                "continuity scores, %s given." % detections)

    # tolerances must be greater than 0
    if tempo_tolerance <= 0 or phase_tolerance <= 0:
        raise ValueError("Tempo and phase tolerances must be greater than 0")

    # make sure the annotations and detections have a float dtype
    detections = np.asarray(detections, dtype=np.float)
    annotations = np.asarray(annotations, dtype=np.float)

    # determine closest annotations to detections
    closest = find_closest_matches(detections, annotations)
    # errors of the detections wrt. to the annotations
    errors = calc_absolute_errors(detections, annotations, closest)
    # detection intervals
    det_interval = calc_intervals(detections)
    # annotation intervals (get those intervals at the correct positions)
    ann_interval = calc_intervals(annotations)[closest]
    # a detection is correct, if it fulfills 2 conditions:
    # 1) must match an annotation within a certain tolerance window, i.e. the
    #    phase must be correct
    correct_phase = detections[errors <= ann_interval * phase_tolerance]
    # Note: the initially cited technical report has an additional condition
    #       ii) on page 5 which requires the same condition to be true for the
    #       previous detection / annotation combination. We do not enforce
    #       this, since a) this condition is kind of pointless: why shouldn't
    #       we count a correct beat just because its predecessor is not? and
    #       b) the original Matlab implementation does not enforce it either
    # 2) the tempo, i.e. the intervals, must be within the tempo tolerance
    # TODO: as agreed with Matthew, this should only be enforced from the 2nd
    #       beat onwards.
    correct_tempo = detections[abs(1 - (det_interval / ann_interval)) <=
                               tempo_tolerance]
    # combine the conditions
    correct = np.intersect1d(correct_phase, correct_tempo)
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


def continuity(detections, annotations,
               phase_tolerance=CONTINUITY_PHASE_TOLERANCE,
               tempo_tolerance=CONTINUITY_TEMPO_TOLERANCE,
               offbeat=True, double=True, triple=True):
    """
    Calculate the cmlc, cmlt, amlc and amlt scores for the given detections and
    annotations.

    :param detections:      numpy array with the detected beats
                            [float, seconds]
    :param annotations:     numpy array with the annotated beats
                            [float, seconds]
    :param phase_tolerance: phase tolerance window [float]
    :param tempo_tolerance: tempo tolerance window [float]
    :param offbeat:         include offbeat variation
    :param double:          include 2x and 1/2x tempo variations
    :param triple:          include 3x and 1/3x tempo variations
    :return:                cmlc, cmlt, amlc, amlt beat tracking accuracies

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
    # neither detections nor annotations are given
    if len(detections) == 0 and len(annotations) == 0:
        return 1., 1., 1., 1.
    # either beat detections or annotations are empty, score 0
    if (len(detections) == 0) != (len(annotations) == 0):
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

    # create different variants of the annotations:
    # Note: double also includes half as does triple third, respectively
    sequences = variations(annotations, offbeat=offbeat, double=double,
                           half=double, triple=triple, third=triple)
    # evaluate these metrical variants
    for sequence in sequences:
        # if other metrical levels achieve higher accuracies, take these values
        try:
            # Note: catch the IntervalError here, because the beat variants
            #       could be too short for valid interval calculation;
            #       ok, since we already have valid values for amlc & amlt
            c, t = cml(detections, sequence, tempo_tolerance, phase_tolerance)
        except BeatIntervalError:
            c, t = np.nan, np.nan
        amlc = max(amlc, c)
        amlt = max(amlt, t)

    # return a tuple
    return cmlc, cmlt, amlc, amlt


def _histogram_bins(num_bins):
    """
    Helper function to generate the histogram bins used to calculate the error
    histogram of the information gain.

    :param num_bins: number of bins
    :return:         histogram bin edges

    Note: This functions returns the bin edges for a histogram with one more
          bin than the requested number of bins, because the fist and last bins
          are added together (to make the histogram circular) later on. Because
          of the same reason, the first and the last bin are only half as wide
          as the others.

    """
    # allow only even numbers and require at least 2 bins
    if num_bins % 2 != 0 or num_bins < 2:
        # Note: because of the implementation details of the histogram, the
        #       easiest way to make sure the an error of 0 is always mapped
        #       to the centre bin is to enforce an even number of bins
        raise ValueError("Number of error histogram bins must be even and "
                         "greater than 0")
    # since np.histogram accepts a sequence of bin edges we just increase the
    # number of bins by 1, but we need to apply offset
    offset = 0.5 / num_bins
    # because the histogram is made circular by adding the last bin to the
    # first one before being removed, increase the number of bins by 2
    return np.linspace(-0.5 - offset, 0.5 + offset, num_bins + 2)


def _error_histogram(detections, annotations, histogram_bins):
    """
    Helper function to calculate the relative errors of the given detections
    and annotations and map them to an histogram with the given bins edges.

    :param detections:     numpy array with the detected beats [float, seconds]
    :param annotations:    numpy array with the annotated beats
                           [float, seconds]
    :param histogram_bins: histogram bin edges for mapping
    :return:               error histogram

    Note: The returned error histogram is circular, i.e. it contains 1 bin less
          than a histogram built normally with the given histogram bin edges.
          The values of the last and first bin are summed and mapped to the
          first bin.

    """
    # get the relative errors of the detections to the annotations
    errors = calc_relative_errors(detections, annotations)
    # map the relative beat errors to the range of -0.5..0.5
    errors = np.mod(errors + 0.5, -1) + 0.5
    # get bin counts for the given errors over the distribution
    histogram = np.histogram(errors, histogram_bins)[0].astype(np.float)
    # make the histogram circular by adding the last bin to the first one
    histogram[0] += histogram[-1]
    # return the histogram without the last bin
    return histogram[:-1]


def _entropy(error_histogram):
    """
    Helper function to calculate the entropy of the given error histogram.

    :param error_histogram: error histogram
    :return:                entropy

    """
    # copy the error_histogram, because it must not be altered
    histogram = np.copy(error_histogram).astype(np.float)
    # normalize the histogram
    histogram /= np.sum(histogram)
    # set all 0 values to 1 to make entropy calculation well-behaved
    histogram[histogram == 0] = 1.
    # calculate entropy
    return - np.sum(histogram * np.log2(histogram))


def _information_gain(error_histogram):
    """
    Helper function to calculate the information gain of the given error
    histogram.

    :param error_histogram: error histogram
    :return:                information gain

    """
    # calculate the entropy of th error histogram
    if np.asarray(error_histogram).any():
        entropy = _entropy(error_histogram)
    else:
        # an empty error histogram has an entropy of 0
        entropy = 0.
    # return information gain
    return np.log2(len(error_histogram)) - entropy


def information_gain(detections, annotations, num_bins=INFORMATION_GAIN_BINS):
    """
    Calculate information gain for the given detections and annotations.

    :param detections:  numpy array with the detected beats [float, seconds]
    :param annotations: numpy array with the annotated beats [float, seconds]
    :param num_bins:    number of bins for the error histogram [int, even]
    :return:            information gain, beat error histogram

    "Measuring the performance of beat tracking algorithms algorithms using a
    beat error histogram"
    M. E. P. Davies, N. Degara and M. D. Plumbley
    IEEE Signal Processing Letters, vol. 18, vo. 3, 2011

    """
    # neither detections nor annotations are given, perfect score
    if len(detections) == 0 and len(annotations) == 0:
        # return a max. information gain and an empty error histogram
        return np.log2(num_bins), np.zeros(num_bins)
    # either beat detections or annotations are empty, score 0
    # Note: use "or" here since we test both the detections against the
    #       annotations and vice versa during the evaluation process
    if len(detections) == 0 or len(annotations) == 0:
        # return an information gain of 0 and a uniform beat error histogram
        # Note: because swapped detections and annotations should return the
        #       same uniform histogram, the maximum length of the detections
        #       and annotations is chosen (instead of just the length of the
        #       annotations as in the Matlab implementation).
        max_length = max(len(detections), len(annotations))
        return 0., np.ones(num_bins) * max_length / float(num_bins)

    # at least 2 annotations must be given to calculate an interval
    if len(detections) < 2 or len(annotations) < 2:
        raise BeatIntervalError("At least 2 annotations and 2 detections are"
                                "needed for Information gain.")

    # check if there are enough beat annotations for the number of bins
    if num_bins > len(annotations):
        warnings.warn("Not enough beat annotations (%d) for %d histogram bins."
                      % (len(annotations), num_bins))

    # create bins edges for the error histogram
    histogram_bins = _histogram_bins(num_bins)

    # evaluate detections against annotations
    fwd_histogram = _error_histogram(detections, annotations, histogram_bins)
    fwd_ig = _information_gain(fwd_histogram)
    # if only a few (but correct) beats are detected, the errors could be small
    # thus evaluate also the annotations against the detections, i.e. simulate
    # a lot of false positive detections
    bwd_histogram = _error_histogram(annotations, detections, histogram_bins)
    bwd_ig = _information_gain(bwd_histogram)

    # only use the lower information gain
    if fwd_ig < bwd_ig:
        return fwd_ig, fwd_histogram
    else:
        return bwd_ig, bwd_histogram


# beat evaluation class
class BeatEvaluation(OnsetEvaluation):
    # this class inherits from OnsetEvaluation the Precision, Recall, and
    # F-measure evaluation stuff but uses a different evaluation window
    """
    Beat evaluation class.

    """

    def __init__(self, detections, annotations,
                 fmeasure_window=FMEASURE_WINDOW,
                 pscore_tolerance=PSCORE_TOLERANCE,
                 cemgil_sigma=CEMGIL_SIGMA, goto_threshold=GOTO_THRESHOLD,
                 goto_sigma=GOTO_SIGMA, goto_mu=GOTO_MU,
                 continuity_phase_tolerance=CONTINUITY_PHASE_TOLERANCE,
                 continuity_tempo_tolerance=CONTINUITY_TEMPO_TOLERANCE,
                 offbeat=True, double=True, triple=True,
                 information_gain_bins=INFORMATION_GAIN_BINS):
        """
        Evaluate the given detections and annotations.

        :param detections:                 sequence of estimated beat times
                                           [seconds, float]
        :param annotations:                sequence of ground truth beat
                                           annotations [seconds, float]
        :param fmeasure_window:            F-measure evaluation window
                                           [seconds, float]
        :param pscore_tolerance:           P-score tolerance [fraction of
                                           median beat interval, float]
        :param cemgil_sigma:               sigma of Gaussian window for Cemgil
                                           accuracy [float]
        :param goto_threshold:             threshold for Goto error [float]
        :param goto_sigma:                 sigma for Goto error [float]
        :param goto_mu:                    mu for Goto error [float]
        :param continuity_phase_tolerance: continuity phase tolerance [float]
        :param continuity_tempo_tolerance: continuity tempo tolerance [float]
        :param offbeat:                    include offbeat variations
        :param double:                     include double/half tempo variations
        :param triple:                     include triple/third tempo
                                           variations
        :param information_gain_bins:      number of bins for the information
                                           gain error histogram

        """
        # convert the detections and annotations
        detections = np.asarray(sorted(detections), dtype=np.float)
        annotations = np.asarray(sorted(annotations), dtype=np.float)
        # perform onset evaluation with the appropriate fmeasure_window
        super(BeatEvaluation, self).__init__(detections, annotations,
                                             fmeasure_window)
        # other scores
        self.pscore = pscore(detections, annotations, pscore_tolerance)
        self.cemgil = cemgil(detections, annotations, cemgil_sigma)
        self.goto = goto(detections, annotations, goto_threshold,
                         goto_sigma, goto_mu)
        # continuity scores
        scores = continuity(detections, annotations,
                            continuity_tempo_tolerance,
                            continuity_phase_tolerance,
                            offbeat, double, triple)
        self.cmlc, self.cmlt, self.amlc, self.amlt = scores
        # information gain stuff
        scores = information_gain(detections, annotations,
                                  information_gain_bins)
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
                   ' %.3f & %.3f & %.3f & %.3f & %.3f & %.3f \\\\' %\
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

    def __len__(self):
        # just use the length of any of the arrays
        return len(self._fmeasure)

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
        """P-score."""
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
    from . import evaluation_io

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    This script evaluates pairs of files containing the beat annotations and
    detections. Suffixes can be given to filter them from the list of files.

    Each line represents a beat and must have the following format with values
    being separated by tabs [brackets indicate optional values]:
    `beat_time [bar.beat]`

    Lines starting with # are treated as comments and are ignored.

    To maintain compatibility with the original Matlab implementation, use
    the arguments '--skip 5 --no_triple'. Please note, that the results can
    still differ, because of the different implementation approach.

    """)
    # files used for evaluation
    evaluation_io(p, ann_suffix='.beats', det_suffix='.beats.txt')
    # parameters for sequence variants
    g = p.add_argument_group('sequence manipulation arguments')
    g.add_argument('--no_offbeat', dest='offbeat', action='store_false',
                   default=True,
                   help='do not include offbeat evaluation')
    g.add_argument('--no_double', dest='double', action='store_false',
                   default=True,
                   help='do not include double/half tempo evaluation')
    g.add_argument('--no_triple', dest='triple', action='store_false',
                   default=True,
                   help='do not include triple/third tempo evaluation')
    g.add_argument('--skip', action='store', type=float, default=0,
                   help='skip first N seconds for evaluation '
                        '[default=%(default).3f]')
    # parameters for evaluation
    g = p.add_argument_group('evaluation arguments')
    g.add_argument('--window', action='store', type=float,
                   default=FMEASURE_WINDOW,
                   help='evaluation window for F-measure '
                        '[seconds, default=%(default).3f]')
    g.add_argument('--tolerance', action='store', type=float,
                   default=PSCORE_TOLERANCE,
                   help='evaluation tolerance for P-score '
                        '[default=%(default).3f]')
    g.add_argument('--sigma', action='store', type=float, default=CEMGIL_SIGMA,
                   help='sigma for Cemgil accuracy [default=%(default).3f]')
    g.add_argument('--goto_threshold', action='store', type=float,
                   default=GOTO_THRESHOLD,
                   help='threshold for Goto error [default=%(default).3f]')
    g.add_argument('--goto_sigma', action='store', type=float,
                   default=GOTO_SIGMA,
                   help='sigma for Goto error [default=%(default).3f]')
    g.add_argument('--goto_mu', action='store', type=float, default=GOTO_MU,
                   help='mu for Goto error [default=%(default).3f]')
    g.add_argument('--phase_tolerance', action='store', type=float,
                   default=CONTINUITY_PHASE_TOLERANCE,
                   help='phase tolerance window for continuity accuracies '
                        '[default=%(default).3f]')
    g.add_argument('--tempo_tolerance', action='store', type=float,
                   default=CONTINUITY_TEMPO_TOLERANCE,
                   help='tempo tolerance window for continuity accuracies '
                        '[default=%(default).3f]')
    g.add_argument('--bins', action='store', type=int,
                   default=INFORMATION_GAIN_BINS,
                   help='number of histogram bins for information gain '
                        '[default=%(default)i]')
    # parse the arguments
    args = p.parse_args()
    # output the args
    if args.verbose >= 2:
        print args
    if args.quiet:
        warnings.filterwarnings("ignore")
    # return
    return args


def main():
    """
    Simple beat evaluation.

    """
    from madmom.utils import search_files, match_file, load_events

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
    mean_eval = MeanBeatEvaluation()
    # evaluate all files
    for ann_file in ann_files:
        # load the annotations
        annotations = load_events(ann_file)
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
            # output a warning if no detections were found
            warnings.warn(" can't find detections for %s." % ann_file)
            # but continue and assume no detections
            detections = np.zeros(0)
        else:
            # load the detections
            detections = load_events(matches[0])
        # remove beats and annotations that are within the first N seconds
        if args.skip > 0:
            # skipping the first few seconds alters the results
            start_idx = np.searchsorted(detections, args.skip, 'right')
            detections = detections[start_idx:]
            start_idx = np.searchsorted(annotations, args.skip, 'right')
            annotations = annotations[start_idx:]
        # evaluate
        e = BeatEvaluation(detections, annotations,
                           fmeasure_window=args.window,
                           pscore_tolerance=args.tolerance,
                           cemgil_sigma=args.sigma,
                           goto_threshold=args.goto_threshold,
                           goto_sigma=args.goto_sigma, goto_mu=args.goto_mu,
                           continuity_tempo_tolerance=args.tempo_tolerance,
                           continuity_phase_tolerance=args.phase_tolerance,
                           information_gain_bins=args.bins,
                           offbeat=args.offbeat, double=args.double,
                           triple=args.triple)
        # output stats for the file
        if args.verbose:
            print e.print_errors('%s\n  ' % ann_file)
        # add this file's evaluation to the global evaluation
        mean_eval.append(e)
    # output summary
    print mean_eval.print_errors('mean for %i file(s):\n  ' % len(mean_eval))


if __name__ == '__main__':
    main()
