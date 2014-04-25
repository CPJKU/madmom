#!/usr/bin/env python
# encoding: utf-8
"""
This file contains tempo related functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import numpy as np
from scipy.signal import argrelmax

from . import Event


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
    # smooth only the the histogram bins, not the
    return smooth_signal(histogram[0], smooth), histogram[1]


# interval detection
def interval_histogram(activations, threshold=0, smooth=None, min_tau=1,
                       max_tau=None):
    """
    Compute the interval histogram of the given activation function.

    :param activations: the activation function
    :param threshold:   threshold for the activation function before
                        auto-correlation
    :param smooth:      kernel (size) for smoothing the activation function
                        before auto-correlating it. [array or int]
    :param min_tau:     minimal delta for correlation function [frames]
    :param max_tau:     maximal delta for correlation function [frames]
    :returns:           histogram

    """
    # smooth activations
    if smooth:
        activations = smooth_signal(activations, smooth)
    # threshold function if needed
    if threshold > 0:
        activations[activations < threshold] = 0
    # set the maximum delay
    if max_tau is None:
        max_tau = len(activations) - min_tau
    # test all possible delays
    taus = range(min_tau, max_tau)
    bins = []
    # TODO: make this processing parallel or numpyfy if possible
    for tau in taus:
        bins.append(np.sum(np.abs(activations[tau:] * activations[0:-tau])))
    # return histogram
    return np.array(bins), np.array(taus)


def dominant_interval(histogram, smooth=None):
    """
    Extract the dominant interval of the given histogram.

    :param histogram: histogram with interval distribution
    :param smooth:    smooth the histogram with the kernel
    :returns:         dominant interval

    """
    # smooth the histogram bins
    if smooth:
        histogram = smooth_histogram(histogram, smooth)
    # return the dominant interval
    return histogram[1][np.argmax(histogram[0])]


# default values for tempo estimation
THRESHOLD = 0
SMOOTH = 0.09
MIN_BPM = 40
MAX_BPM = 240


class Tempo(Event):
    """
    Tempo Class.

    """
    def __init__(self, activations, fps, sep=''):
        """
        Creates a new Tempo instance with the given activations.
        The activations can be read in from file.

        :param activations: array with the beat activations or a file (handle)
        :param fps:         frame rate of the activations
        :param sep:         separator if activations are read from file

        """
        super(Tempo, self).__init__(activations, fps, sep)

    def detect(self, threshold=THRESHOLD, smooth=SMOOTH, min_bpm=MIN_BPM,
               max_bpm=MAX_BPM, mirex=False):
        """
        Detect the tempo on basis of the given beat activation function.

        :param threshold: threshold for peak-picking
        :param smooth:    smooth the activation function over N seconds
        :param min_bpm:   minimum tempo used for beat tracking
        :param max_bpm:   maximum tempo used for beat tracking
        :param mirex:     always output the lower tempo first
        :returns:         tuple with the two most dominant tempi and the
                          relative weight of them

        """
        # convert the arguments to frames
        smooth = int(round(self.fps * smooth))
        min_tau = int(np.floor(60. * self.fps / max_bpm))
        max_tau = int(np.ceil(60. * self.fps / min_bpm))
        # generate a histogram of beat intervals
        histogram = interval_histogram(self.activations, threshold,
                                       smooth=smooth, min_tau=min_tau,
                                       max_tau=max_tau)
        # smooth the histogram again
        if smooth:
            histogram = smooth_histogram(histogram, smooth)
        # the histogram bins
        bins = histogram[0]
        # convert the histogram bin delays to tempi in beats per minute
        tempi = 60.0 * self.fps / histogram[1]
        # to get the two dominant tempi, just keep the peaks
        # use 'wrap' mode to also get peaks at the borders
        peaks = argrelmax(bins, mode='wrap')[0]
        # get the weights of the peaks to sort them in descending order
        strengths = bins[peaks]
        sorted_peaks = peaks[np.argsort(strengths)[::-1]]
        # we need more than 1 peak to report multiple tempi
        if len(sorted_peaks) < 2:
            # return tempi[sorted_peaks[0]], np.nan, 1.
            raise AssertionError('this should not happen!')
        # get the 2 strongest tempi
        t1, t2 = tempi[sorted_peaks[:2]]
        # calculate the relative strength
        strength = bins[sorted_peaks[0]]
        strength /= np.sum(bins[sorted_peaks[:2]])
        # return the tempi + the relative strength
        if mirex and t1 > t2:
            # for MIREX, the lower tempo must be given first
            return t2, t1, 1. - strength
        return t1, t2, strength
