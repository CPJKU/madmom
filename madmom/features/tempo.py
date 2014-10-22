#!/usr/bin/env python
# encoding: utf-8
"""
This file contains tempo related functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import numpy as np
from scipy.signal import argrelmax

from . import Activations, RNNEventDetection, smooth_signal
from .beats import RNNBeatTracking


NO_TEMPO = np.nan


# helper functions
def smooth_histogram(histogram, smooth):
    """
    Smooth the given histogram.

    :param histogram: histogram
    :param smooth:    smoothing kernel [numpy array or int]
    :return:          smoothed histogram

    Note: If 'smooth' is an integer, a Hamming window of that length will be
          used as a smoothing kernel.

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
    :return:            histogram

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
    :return:            histogram

    """
    # import comb filter
    from madmom.audio.filters import CombFilterbank
    # set the maximum delay
    if max_tau is None:
        max_tau = len(activations) - min_tau
    # get the range of taus
    taus = np.arange(min_tau, max_tau + 1)
    # apply a bank of comb filters
    cfb = CombFilterbank(activations, 'backward', taus, alpha)
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
    :return:          dominant interval

    """
    # smooth the histogram bins
    if smooth:
        histogram = smooth_histogram(histogram, smooth)
    # return the dominant interval
    return histogram[1][np.argmax(histogram[0])]


# extract the tempo from a histogram
def detect_tempo(histogram, fps):
    """
    Extract the tempo from the given histogram.

    :param histogram: tempo histogram
    :param fps:       frames per second (needed for conversion to BPM)
    :return:          numpy array with the dominant tempi (first column)
                      and their relative strengths (second column)

    """
    # histogram of IBIs
    bins = histogram[0]
    # convert the histogram bin delays to tempi in beats per minute
    tempi = 60.0 * fps / histogram[1]
    # to get the two dominant tempi, just keep the peaks
    # use 'wrap' mode to also get peaks at the borders
    peaks = argrelmax(bins, mode='wrap')[0]
    # we need more than 1 peak to report multiple tempi
    if len(peaks) == 0:
        # a flat histogram has no peaks, use the center bin
        if len(bins):
            return np.asarray([tempi[len(bins) / 2], 1.])
        # otherwise: no peaks, no tempo
        return np.asarray([NO_TEMPO, 0.])
    elif len(peaks) == 1:
        # report only the strongest tempo
        return np.asarray([tempi[peaks[0]], 1.])
    else:
        # sort the peaks in descending order of bin heights
        sorted_peaks = peaks[np.argsort(bins[peaks])[::-1]]
        # normalize their strengths
        strengths = bins[sorted_peaks]
        strengths /= np.sum(strengths)
        # return the tempi and their normalized strengths
        return np.asarray(zip(tempi[sorted_peaks], strengths))


class TempoEstimation(RNNBeatTracking):
    """
    Tempo Estimation class.

    """
    # default values for tempo estimation
    METHOD = 'comb'
    MIN_BPM = 40
    MAX_BPM = 250
    HIST_SMOOTH = 7
    ACT_SMOOTH = 0.14
    ALPHA = 0.79

    def detect(self, method=METHOD, min_bpm=MIN_BPM, max_bpm=MAX_BPM,
               act_smooth=ACT_SMOOTH, hist_smooth=HIST_SMOOTH, alpha=ALPHA):
        """
        Estimates the tempo of the signal.

        :param method:      either 'acf' or 'comb'.
        :param min_bpm:     minimum tempo to detect
        :param max_bpm:     maximum tempo to detect
        :param act_smooth:  smooth the activation function over N seconds
        :param hist_smooth: smooth the activation function over N bins
        :param alpha:       scaling factor for the comb filter
        :return:            numpy array with the dominant tempi (first column)
                            and their relative strengths (second column)

        """
        # convert the arguments to frames
        min_tau = int(np.floor(60. * self.fps / max_bpm))
        max_tau = int(np.ceil(60. * self.fps / min_bpm))
        act_smooth = int(round(self.fps * act_smooth))
        # smooth the activations
        activations = smooth_signal(self.activations, act_smooth)
        # generate a histogram of beat intervals
        if method == 'acf':
            histogram = interval_histogram_acf(activations, min_tau, max_tau)
        elif method == 'comb':
            histogram = interval_histogram_comb(activations, alpha, min_tau,
                                                max_tau)
        else:
            raise ValueError('tempo estimation method unknown')
        # smooth the histogram
        histogram = smooth_histogram(histogram, hist_smooth)
        # detect the tempi and return them
        self._detections = detect_tempo(histogram, self.fps)
        return self._detections

    def write(self, filename, mirex=False):
        """
        Write the two most dominant tempi and the relative strength to a file.

        :param filename: output file name or file handle
        :param mirex:    report the lower tempo first (as required by MIREX)

        """
        from madmom.utils import open
        # default values
        t1, t2, strength = 0., 0., 1.
        # only one tempo was detected
        if len(self.detections) == 1:
            t1 = self.detections[0][0]
            # generate a fake second tempo
            if t1 > 120:
                t2 = t1 / 2.
            else:
                t2 = t1 * 2.
        # consider only the two strongest tempi and strengths
        elif len(self.detections) > 1:
            t1, t2 = self.detections[:2, 0]
            strength = self.detections[0, 1] / sum(self.detections[:2, 1])
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
                      alpha=ALPHA):
        """
        Add tempo estimation related arguments to an existing parser object.

        :param parser:      existing argparse parser object
        :param nn_files:    list with files of NN models
        :param method:      either 'acf' or 'comb'.
        :param min_bpm:     minimum tempo [bpm]
        :param max_bpm:     maximum tempo [bpm]
        :param act_smooth:  smooth the activations over N seconds
        :param hist_smooth: smooth the tempo histogram over N bins
        :param alpha:       scaling factor of the comb filter
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
        g.add_argument('--alpha', action='store', type=float, default=alpha,
                       help='alpha for comb filter tempo estimation '
                       '[default=%(default).2f]')
        # return the argument group so it can be modified if needed
        return g
