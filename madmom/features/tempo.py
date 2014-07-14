#!/usr/bin/env python
# encoding: utf-8
"""
This file contains tempo related functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import numpy as np
from scipy.signal import argrelmax
from . import Activations, RNNEventDetection
from .beats import RNNBeatTracking


NO_TEMPO = np.nan


# helper functions
def smooth_signal(signal, smooth):
    """
    Smooth the given signal.

    :param signal: signal
    :param smooth: smoothing kernel [array or int]
    :returns:      smoothed signal

    """
    # init smoothing kernel
    kernel = None
    # size for the smoothing kernel is given
    if isinstance(smooth, int):
        if smooth > 1:
            kernel = np.hamming(smooth)
    # otherwise use the given smoothing kernel directly
    elif isinstance(smooth, np.ndarray):
        if len(smooth) > 1:
            kernel = smooth
    # check if a kernel is given
    if kernel is None:
        raise ValueError('can not smooth signal with %s' % smooth)
    # convolve with the kernel and return
    return np.convolve(signal, kernel, 'same')


def smooth_histogram(histogram, smooth):
    """
    Smooth the given histogram.

    :param histogram: histogram
    :param smooth:    smoothing kernel [array or int]
    :returns:         smoothed histogram

    """
    # smooth only the the histogram bins, not the corresponding delays
    return smooth_signal(histogram[0], smooth), histogram[1]


# interval detection
def interval_histogram_acf(activations, min_tau=1, max_tau=None):
    """
    Compute the interval histogram of the given activation function with via
    auto-correlation.

    :param activations: the activation function
    :param min_tau:     minimal delta for correlation function [frames]
    :param max_tau:     maximal delta for correlation function [frames]
    :returns:           histogram

    """
    # set the maximum delay
    if max_tau is None:
        max_tau = len(activations) - min_tau
    # test all possible delays
    taus = range(min_tau, max_tau + 1)
    bins = []
    # TODO: make this processing parallel or numpyfy if possible
    for tau in taus:
        bins.append(np.sum(np.abs(activations[tau:] * activations[0:-tau])))
    # return histogram
    return np.array(bins), np.array(taus)


def interval_histogram_comb(activations, alpha, min_tau=1, max_tau=None):
    """
    Compute the interval histogram of the given activation function via a
    bank of comb filters.

    :param activations: the activation function
    :param alpha:       scaling factor for the comb filter
    :param min_tau:     minimal delta for correlation function [frames]
    :param max_tau:     maximal delta for correlation function [frames]
    :returns:           histogram

    """
    # import comb filter
    from ..audio.filters import feed_backward_comb_filter
    from ..audio.filters import CombFilterbank
    # set the maximum delay
    if max_tau is None:
        max_tau = len(activations) - min_tau
    # get the range of taus
    taus = np.arange(min_tau, max_tau + 1)
    # apply a bank of comb filters
    cfb = CombFilterbank(activations, feed_backward_comb_filter, taus, alpha)
    # determine the tau with the highest value for each time step
    # sum up the maxima to yield the histogram bin values
    histogram_bins = np.sum(cfb * (cfb == np.max(cfb, axis=0)), axis=1)
    # return histogram
    return histogram_bins, taus


# helper functions
def dominant_interval(histogram, smooth=None):
    """
    Extract the dominant interval of the given histogram.

    :param histogram: histogram with interval distribution
    :param smooth:    smooth the histogram with the given kernel (size)
    :returns:         dominant interval

    """
    # smooth the histogram bins
    if smooth:
        histogram = smooth_histogram(histogram, smooth)
    # return the dominant interval
    return histogram[1][np.argmax(histogram[0])]


def log3(x):
    """
    Takes the logarithm to the basis of 3.

    :param x: array like
    :return:  logarithm to the basis of 3
    """
    return np.log(x) / np.log(3)


# extract the tempo from a histogram
def detect_tempo(histogram, fps, grouping_dev=0):
    # the histogram bins
    """
    Extract the tempo from the given histogram.

    :param histogram:    tempo histogram
    :param fps:          frames per second (needed for conversion to BPM)
    :param grouping_dev: allowed tempo deviation for grouping tempi
    :return:             tuple with (tempo1, tempo2, relative strength)

    """
    # the histogram of the IBIs
    bins = histogram[0]
    # convert the histogram bin delays to tempi in beats per minute
    tempi = 60.0 * fps / histogram[1]
    # to get the two dominant tempi, just keep the peaks
    # use 'wrap' mode to also get peaks at the borders
    peaks = argrelmax(bins, mode='wrap')[0]
    # we need more than 1 peak to report multiple tempi
    if len(peaks) == 0:
        # no peaks, no tempo
        return NO_TEMPO, NO_TEMPO, 0.
    elif len(peaks) == 1:
        # report only the strongest tempo
        return tempi[peaks[0]], NO_TEMPO, 1.
    else:
        # sort the peaks in descending order of bin heights
        sorted_peaks = peaks[np.argsort(bins[peaks])[::-1]]
        # group the corresponding tempi if needed
        if grouping_dev:
            # get the peak tempi
            t = tempi[sorted_peaks]
            # get the whole-numbered divisors from all tempo combinations
            c = t / t[:, np.newaxis]
            # group all double tempi
            # transform the fractions to the log2 space
            double_c = np.abs(np.log2(c))
            # and keep only those within a certain deviation
            double_c = np.round(double_c / grouping_dev) * grouping_dev
            double_c = double_c % 1 == 0
            # get the corresponding strengths
            double = bins[sorted_peaks][np.newaxis, :] * double_c
            # select the winning combination
            double_tempi = double[np.argmax(np.sum(double_c, axis=1))]
            # group all triple tempi
            # transform the fractions to the log3 space
            triple_c = np.abs(log3(c))
            # again, keep only those within a certain deviation
            triple_c = np.round(triple_c / grouping_dev) * grouping_dev
            triple_c = triple_c % 1 == 0
            # get the corresponding strengths
            triple = bins[sorted_peaks][np.newaxis, :] * triple_c
            # select the winning combination
            triple_tempi = triple[np.argmax(np.sum(triple_c, axis=1))]
            # combine the double and triple tempo combinations
            strengths = np.max((double_tempi, triple_tempi), axis=0)
            # re-sort the peaks
            sorted_peaks = sorted_peaks[np.argsort(strengths)[::-1]]
        # otherwise just return the 2 strongest tempi
        t1, t2 = tempi[sorted_peaks[:2]]
        # calculate the relative strength
        strength = bins[sorted_peaks[0]]
        strength /= np.sum(bins[sorted_peaks[:2]])
        return t1, t2, strength


class TempoEstimation(RNNBeatTracking):
    """
    Tempo Class.

    """
    # default values for tempo estimation
    METHOD = 'comb'
    MIN_BPM = 39
    MAX_BPM = 245
    HIST_SMOOTH = 5
    ACT_SMOOTH = 0.14
    GROUPING_DEV = 0
    ALPHA = 0.79

    def detect(self, method=METHOD, min_bpm=MIN_BPM, max_bpm=MAX_BPM,
               act_smooth=ACT_SMOOTH, hist_smooth=HIST_SMOOTH,
               grouping_dev=GROUPING_DEV, alpha=ALPHA, **kwargs):
        """
        Estimates the tempo of the signal.

        :param method:       either 'acf' or 'comb'.
        :param min_bpm:      minimum tempo to detect
        :param max_bpm:      maximum tempo to detect
        :param act_smooth:   smooth the activation function over N seconds
        :param hist_smooth:  smooth the activation function over N bins
        :param grouping_dev: allowed tempo deviation for grouping tempi
        :param alpha:        scaling factor for the comb filter
        :return:             tuple with the two most dominant tempi and the
                             relative strength of them

        Note: If the 'grouping_dev' is set to 0, the tempi are not grouped.
              The deviation is allowed delta in the log2 / log3 space.

        """
        # convert the arguments to frames
        min_tau = int(np.floor(60. * self.fps / max_bpm))
        max_tau = int(np.ceil(60. * self.fps / min_bpm))
        act_smooth = int(round(self.fps * act_smooth))
        # smooth the activations
        if act_smooth > 1:
            activations = smooth_signal(self.activations, act_smooth)
        else:
            activations = self.activations
        # generate a histogram of beat intervals
        if method == 'acf':
            histogram = interval_histogram_acf(activations, min_tau, max_tau)
        else:
            histogram = interval_histogram_comb(activations, alpha, min_tau,
                                                max_tau)
        # smooth the histogram
        if hist_smooth > 1:
            histogram = smooth_histogram(histogram, hist_smooth)
        # detect the tempi and return them
        self._detections = detect_tempo(histogram, self.fps, grouping_dev)
        return self._detections

    def write(self, filename, mirex=False):
        """
        Write the detections to a file.

        :param filename: output file name or file handle

        """
        from ..utils import open
        t1, t2, strength = self.detections
        # for MIREX, the lower tempo must be given first
        if mirex and t1 > t2:
            t1, t2, strength = t2, t1, 1. - strength
        # write to output
        with open(filename, 'wb') as f:
            f.write("%.2f\t%.2f\t%.2f\n" % (t1, t2, strength))

    @classmethod
    def add_arguments(cls, parser, nn_files=RNNBeatTracking.NN_FILES,
                      method=METHOD, min_bpm=MIN_BPM, max_bpm=MAX_BPM,
                      act_smooth=ACT_SMOOTH, hist_smooth=HIST_SMOOTH,
                      dev=GROUPING_DEV, alpha=ALPHA):
        """
        Add tempo estimation related arguments to an existing parser object.

        :param parser:      existing argparse parser object
        :param method:      either 'acf' or 'comb'.
        :param min_bpm:     minimum tempo [bpm]
        :param max_bpm:     maximum tempo [bpm]
        :param act_smooth:  smooth the activations over N seconds
        :param hist_smooth: smooth the tempo histogram over N bins
        :param dev:         allowed deviation of tempi when grouping them
        :return:            tempo argument parser group object

        """
        # add Activations parser
        Activations.add_arguments(parser)
        # add arguments from RNNEventDetection
        RNNEventDetection.add_arguments(parser, nn_files=nn_files)
        # add tempo estimation related options to the existing parser
        g = parser.add_argument_group('tempo estimation arguments')
        g.add_argument('--method', action='store', type=str, default=method,
                       help="which method to use ['acf' or 'comb', "
                            "default=%(default)s]")
        g.add_argument('--min_bpm', action='store', type=float,
                       default=min_bpm, help='minimum tempo [bpm, '
                       ' default=%(default).2f]')
        g.add_argument('--max_bpm', action='store', type=float,
                       default=max_bpm, help='maximum tempo [bpm, '
                       ' default=%(default).2f]')
        g.add_argument('--act_smooth', action='store', type=float,
                       default=act_smooth,
                       help='smooth the activations over N seconds '
                            '[default=%(default)d]')
        g.add_argument('--hist_smooth', action='store', type=int,
                       default=hist_smooth,
                       help='smooth the tempo histogram over N bins '
                            '[default=%(default)d]')
        g.add_argument('--dev', action='store', type=float, default=dev,
                       help='maximum allowed tempo deviation when grouping '
                       ' tempi [default=%(default).2f]')
        g.add_argument('--alpha', action='store', type=float, default=alpha,
                       help='alpha for comb filter tempo estimation '
                       '[default=%(default).2f]')
        # return the argument group so it can be modified if needed
        return g
