#!/usr/bin/env python
# encoding: utf-8
"""
This file contains all beat tracking related functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import sys
import numpy as np

from . import Event
from scipy.signal import argrelmax


## TODO: implement some simple algorithms
#class SpectralBeatTracking(object):
#    """
#    The SpectralBeatTracking class implements some common beat tracking
#    algorithms.
#
#    """
#    def __init__(self, spectrogram):
#        """
#        Creates a new SpectralBeatTracking instance.
#
#        :param spectrogram: the spectrogram object on which the detections
#                            functions operate
#
#        """
#        # import
#        from ..audio.spectrogram import Spectrogram
#        # check spectrogram type
#        if isinstance(spectrogram, Spectrogram):
#            # already the right format
#            self.s = spectrogram
#        else:
#            # assume a file name, try to instantiate a Spectrogram object
#            self.s = Spectrogram(spectrogram)
#
#    # beat tracking algorithms
#    def acf(self):
#        """Auto correlation beat tracking."""
#        # TODO: include code
#        raise NotImplementedError
#
#    def multiple_agents(self):
#        """Multiple agents based tracker."""
#        # TODO: include code
#        raise NotImplementedError


# interval (tempo) detection
def detect_dominant_interval(activations, threshold=0, smooth=None, min_tau=1,
                             max_tau=None):
    """
    Extract the dominant interval of the given activation function.

    :param activations: the onset activation function
    :param threshold:   threshold for the activation function before
                        auto-correlation
    :param smooth:      smooth the activation function with the kernel
    :param min_tau:     minimal delta for correlation function [frames]
    :param max_tau:     maximal delta for correlation function [frames]

    """
    # smooth activations
    kernel = None
    if isinstance(smooth, int):
        # size for the smoothing kernel is given
        if smooth > 1:
            kernel = np.hamming(smooth)
    elif isinstance(smooth, np.ndarray):
        # otherwise use the given smooth kernel directly
        if smooth.size > 1:
            kernel = smooth
    if kernel is not None:
        # convolve with the kernel
        activations = np.convolve(activations, kernel, 'same')

    # threshold function if needed
    if threshold > 0:
        activations[activations < threshold] = 0

    # test all possible intervals
    taus = range(min_tau, max_tau)
    sums = []
    # TODO: make this processing parallel or numpyfy if possible
    for tau in taus:
        sums.append(np.sum(np.abs(activations[tau:] * activations[0:-tau])))

    # return dominant interval
    interval = np.argmax(sums) + min_tau
    return interval


def interval_histogram(activations, threshold=0, smooth=None, min_tau=1,
                       max_tau=None):
    """
    Compute the interval histogram of the given activation function.

    :param activations: the onset activation function
    :param threshold:   threshold for the activation function before
                        auto-correlation
    :param smooth:      smooth the activation function with the kernel
    :param min_tau:     minimal delta for correlation function [frames]
    :param max_tau:     maximal delta for correlation function [frames]
    :returns:           histogram

    """
    # smooth activations
    kernel = None
    if isinstance(smooth, int):
        # size for the smoothing kernel is given
        if smooth > 1:
            kernel = np.hamming(smooth)
    elif isinstance(smooth, np.ndarray):
        # otherwise use the given smooth kernel directly
        if smooth.size > 1:
            kernel = smooth
    if kernel is not None:
        # convolve with the kernel
        activations = np.convolve(activations, kernel, 'same')

    # threshold function if needed
    if threshold > 0:
        activations[activations < threshold] = 0

    if max_tau is None:
        max_tau = len(activations) - min_tau

    # test all possible intervals
    taus = range(min_tau, max_tau)
    sums = []
    # TODO: make this processing parallel or numpyfy if possible
    for tau in taus:
        sums.append(np.sum(np.abs(activations[tau:] * activations[0:-tau])))

    # return histogram
    return np.array(sums), np.array(taus)


def dominant_interval(histogram, smooth=None):
    """
    Extract the dominant interval of the given histogram.

    :param histogram: histogram with interval distribution
    :param smooth:    smooth the histogram with the kernel
    :returns:         dominant interval

    """
    # smooth histogram
    kernel = None
    if isinstance(smooth, int):
        # size for the smoothing kernel is given
        if smooth > 1:
            kernel = np.hamming(smooth)
    elif isinstance(smooth, np.ndarray):
        # otherwise use the given smooth kernel directly
        if smooth.size > 1:
            kernel = smooth
    if kernel is not None:
        # convolve with the kernel
        values = np.convolve(histogram[0], kernel, 'same')
    else:
        values = histogram[0]
    # return the dominant interval
    return histogram[1][np.argmax(values)]


# TODO: unify with dominant interval
def detect_tempo(histogram, fps, smooth=None):
    """
    Extract the dominant interval of the given histogram.

    :param histogram: histogram with interval distribution
    :param fps:       frame rate of the original beat activations
    :param smooth:    smooth the histogram with the kernel
    :returns:         dominant interval

    """
    # smooth histogram
    kernel = None
    if isinstance(smooth, int):
        # size for the smoothing kernel is given
        if smooth > 1:
            kernel = np.hamming(smooth)
    elif isinstance(smooth, np.ndarray):
        # otherwise use the given smooth kernel directly
        if smooth.size > 1:
            kernel = smooth
    if kernel is not None:
        # convolve with the kernel
        values = np.convolve(histogram[0], kernel, 'same')
    else:
        values = histogram[0]
    tempi = 60.0 * fps / histogram[1]
    # to get the two dominant tempi, just keep the peaks
    # use 'wrap' mode to also get peaks at the borders
    peaks = argrelmax(values, mode='wrap')[0]
    # get the weights of the peaks to sort them in descending order
    strengths = values[peaks]
    sorted_peaks = peaks[np.argsort(strengths)[::-1]]
    # if we have more than 2 peaks, we can report multiple tempi
    if len(sorted_peaks) > 1:
        # get the 2 strongest tempi
        t1, t2 = tempi[sorted_peaks[:2]]
        # calculate the relative strength
        strength = values[sorted_peaks[0]]
        strength /= np.sum(values[sorted_peaks[:2]])
        # return the tempi + the relative strength
        return t1, t2, strength
    else:
        # return just the strongest tempo
        return tempi[sorted_peaks[0]], np.nan, 1.


def detect_beats(activations, interval, look_aside=0.2):
    """
    Detects the beats in the given activation function.

    :param activations: array with beat activations
    :param interval:    look for the next beat each N frames
    :param look_aside:  look this fraction of the interval to the side to
                        detect the beats

    Note: A Hamming window of 2*look_aside*interval is applied for smoothing.

    """
    # TODO: make this faster!
    sys.setrecursionlimit(len(activations))
    # look for which starting beat the sum gets maximized
    sums = np.zeros(interval)
    positions = []
    # always look at least 1 frame to each side
    frames_look_aside = max(1, int(interval * look_aside))
    win = np.hamming(2 * frames_look_aside)
    for i in range(interval):
        # TODO: threads?
        def recursive(pos):
            """
            Recursively detect the next beat.

            :param pos: start at this position
            :return:    the next beat position

            """
            # detect the nearest beat around the actual position
            start = pos - frames_look_aside
            end = pos + frames_look_aside
            if start < 0:
                # pad with zeros
                act = np.append(np.zeros(-start), activations[0:end])
            elif end > len(activations):
                # append zeros accordingly
                zeros = np.zeros(end - len(activations))
                act = np.append(activations[start:], zeros)
            else:
                act = activations[start:end]
            # apply a filtering window to prefer beats closer to the centre
            act = np.multiply(act, win)
            # search max
            if np.argmax(act) > 0:
                # maximum found, take that position
                pos = np.argmax(act) + start
            # add the found position
            positions.append(pos)
            # add the activation at that position
            sums[i] += activations[pos]
            # go to the next beat, until end is reached
            if pos + interval < len(activations):
                recursive(pos + interval)
            else:
                return
        # start at initial position
        recursive(i)
    # take the winning start position
    pos = np.argmax(sums)
    # and calc the beats for this start position
    positions = []
    recursive(pos)
    # return indices (as floats, since they get converted to seconds later on)
    return np.array(positions, dtype=np.float)


# default values for beat tracking
THRESHOLD = 0
SMOOTH = 0.09
MIN_BPM = 60
MAX_BPM = 240
LOOK_ASIDE = 0.2
LOOK_AHEAD = 4
DELAY = 0


class Beat(Event):
    """
    Beat Class.

    """
    def __init__(self, activations, fps, online=False, sep=''):
        """
        Creates a new Beat instance with the given activations.
        The activations can be read in from file.

        :param activations: array with the beat activations or a file (handle)
        :param fps:         frame rate of the activations
        :param online:      work in online mode (i.e. use only past
                            information)
        :param sep:         separator if activations are read from file

        """
        if online:
            raise NotImplementedError('online mode not implemented (yet)')
        # inherit most stuff from the base class
        super(Beat, self).__init__(activations, fps, sep)

    def detect(self, threshold=THRESHOLD, delay=DELAY, smooth=SMOOTH,
               min_bpm=MIN_BPM, max_bpm=MAX_BPM, look_aside=LOOK_ASIDE):
        """
        Detect the beats with a simple auto-correlation method.

        :param threshold:  threshold for peak-picking
        :param delay:      report onsets N seconds delayed
        :param smooth:     smooth the activation function over N seconds
        :param min_bpm:    minimum tempo used for beat tracking
        :param max_bpm:    maximum tempo used for beat tracking
        :param look_aside: look this fraction of a beat interval to the side

        First the global tempo is estimated and then the beats are aligned
        according to:

        "Enhanced Beat Tracking with Context-Aware Neural Networks"
        Sebastian Böck and Markus Schedl
        Proceedings of the 14th International Conference on Digital Audio
        Effects (DAFx-11), Paris, France, September 2011

        """
        # convert timing information to frames and set default values
        # TODO: use at least 1 frame if any of these values are > 0?
        smooth = int(round(self.fps * smooth))
        min_tau = int(np.floor(60. * self.fps / max_bpm))
        max_tau = int(np.ceil(60. * self.fps / min_bpm))
        # detect the dominant interval
        interval = detect_dominant_interval(self.activations, threshold,
                                            smooth, min_tau, max_tau)
        # detect beats based on this interval (function returns int indices)
        detections = detect_beats(self.activations, interval, look_aside)
        # convert detected beats to a list of timestamps
        detections = detections.astype(np.float) / self.fps
        # shift if necessary
        if delay != 0:
            detections += delay
        # remove beats with negative times
        self.detections = detections[np.searchsorted(detections, 0):]
        # also return the detections
        return self.detections

    def track(self, threshold=THRESHOLD, delay=DELAY, smooth=SMOOTH,
              min_bpm=MIN_BPM, max_bpm=MAX_BPM, look_aside=LOOK_ASIDE,
              look_ahead=LOOK_AHEAD):
        """
        Track the beats with a simple auto-correlation method.

        :param threshold:  threshold for peak-picking
        :param delay:      report onsets N seconds delayed
        :param smooth:     smooth the activation function over N seconds
        :param min_bpm:    minimum tempo used for beat tracking
        :param max_bpm:    maximum tempo used for beat tracking
        :param look_aside: look this fraction of a beat interval to the side
        :param look_ahead: look N seconds ahead (and back) to determine the
                           tempo

        First local tempo (in a range +- look_ahead seconds around the actual
        position) is estimated and then the next beat is tracked accordingly.
        Then the same procedure is repeated from this new position.

        "Enhanced Beat Tracking with Context-Aware Neural Networks"
        Sebastian Böck and Markus Schedl
        Proceedings of the 14th International Conference on Digital Audio
        Effects (DAFx-11), Paris, France, September 2011

        """
        # convert timing information to frames and set default values
        # TODO: use at least 1 frame if any of these values are > 0?
        smooth = int(round(self.fps * smooth))
        min_tau = int(np.floor(60. * self.fps / max_bpm))
        max_tau = int(np.ceil(60. * self.fps / min_bpm))
        look_ahead_frames = int(look_ahead * self.fps)

        # detect the beats
        detections = []
        pos = 0
        # TODO: make this _much_ faster!
        while pos < len(self.activations):
            # look N frames around the actual position
            start = pos - look_ahead_frames
            end = pos + look_ahead_frames
            if start < 0:
                # pad with zeros
                act = np.append(np.zeros(-start), self.activations[0:end])
            elif end > len(self.activations):
                # append zeros accordingly
                zeros = np.zeros(end - len(self.activations))
                act = np.append(self.activations[start:], zeros)
            else:
                act = self.activations[start:end]
            # detect the dominant interval
            interval = detect_dominant_interval(act, threshold, smooth,
                                                min_tau, max_tau)
            # add the offset (i.e. the new detected start position)
            positions = np.array(detect_beats(act, interval, look_aside))
            # correct the beat positions
            positions += start
            # search the closest beat to the predicted beat position
            pos = positions[(np.abs(positions - pos)).argmin()]
            # append to the beats
            detections.append(pos)
            pos += interval
        # convert detected beats to a list of timestamps
        detections = np.array(detections) / float(self.fps)
        # shift if necessary
        if delay != 0:
            detections += delay
        # remove beats with negative times
        self.detections = detections[np.searchsorted(detections, 0):]
        # also return the detections
        return self.detections

    # TODO: make an extra Tempo class
    def tempo(self, threshold=THRESHOLD, smooth=SMOOTH, min_bpm=MIN_BPM,
              max_bpm=MAX_BPM, mirex=False):
        """
        Detect the tempo on basis of the beat activation function.

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
        hist = interval_histogram(self.activations, threshold, smooth=smooth,
                                  min_tau=min_tau, max_tau=max_tau)
        t1, t2, weight = detect_tempo(hist, self.fps, smooth=None)
        # for MIREX, the lower tempo must be given first
        if mirex and t1 > t2:
            return t2, t1, 1 - weight
        return t1, t2, weight
