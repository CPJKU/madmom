#!/usr/bin/env python
# encoding: utf-8
"""
This file contains all beat tracking related functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import sys
import numpy as np

from . import Event
from .tempo import (smooth_signal, interval_histogram_acf, dominant_interval,
                    MIN_BPM, MAX_BPM)


# wrapper function for detecting the dominant interval
def detect_dominant_interval(activations, act_smooth=None, hist_smooth=None,
                             min_tau=1, max_tau=None):
    """
    Compute the dominant interval of the given activation function.

    :param activations: the activation function
    :param act_smooth:  kernel (size) for smoothing the activation function
    :param hist_smooth: kernel (size) for smoothing the interval histogram
    :param min_tau:     minimal delay for histogram building [frames]
    :param max_tau:     maximal delay for histogram building [frames]
    :returns:           dominant interval

    """
    # smooth activations
    if act_smooth > 1:
        activations = smooth_signal(activations, act_smooth)
    # create a interval histogram
    h = interval_histogram_acf(activations, min_tau, max_tau)
    # get the dominant interval and return it
    return dominant_interval(h, smooth=hist_smooth)


# detect the beats based on the given dominant interval
def detect_beats(activations, interval, look_aside=0.2):
    """
    Detects the beats in the given activation function.

    :param activations: array with beat activations
    :param interval:    look for the next beat each N frames
    :param look_aside:  look this fraction of the interval to the side to
                        detect the beats

    "Enhanced Beat Tracking with Context-Aware Neural Networks"
    Sebastian Böck and Markus Schedl
    Proceedings of the 14th International Conference on Digital Audio
    Effects (DAFx-11), Paris, France, September 2011

    Note: A Hamming window of 2*look_aside*interval is applied for smoothing

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
        def recursive(position):
            """
            Recursively detect the next beat.

            :param position: start at this position
            :return:    the next beat position

            """
            # detect the nearest beat around the actual position
            start = position - frames_look_aside
            end = position + frames_look_aside
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
                position = np.argmax(act) + start
            # add the found position
            positions.append(position)
            # add the activation at that position
            sums[i] += activations[position]
            # go to the next beat, until end is reached
            if position + interval < len(activations):
                recursive(position + interval)
            else:
                return
        # start at initial position
        recursive(i)
    # take the winning start position
    start_position = np.argmax(sums)
    # and calc the beats for this start position
    positions = []
    recursive(start_position)
    # return indices (as floats, since they get converted to seconds later on)
    return np.array(positions, dtype=np.float)


# default values for beat tracking
SMOOTH = 0.09
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

    def detect(self, smooth=SMOOTH, min_bpm=MIN_BPM, max_bpm=MAX_BPM,
               look_aside=LOOK_ASIDE):
        """
        Detect the beats with a simple auto-correlation method.

        :param smooth: smooth the activation function over N seconds
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
        interval = detect_dominant_interval(self.activations,
                                            act_smooth=smooth,
                                            hist_smooth=None,
                                            min_tau=min_tau, max_tau=max_tau)
        # detect beats based on this interval
        detections = detect_beats(self.activations, interval, look_aside)
        # convert detected beats to a list of timestamps
        detections = detections.astype(np.float) / self.fps
        # remove beats with negative times
        self.detections = detections[np.searchsorted(detections, 0):]
        # also return the detections
        return self.detections

    def track(self, smooth=SMOOTH, min_bpm=MIN_BPM, max_bpm=MAX_BPM,
              look_aside=LOOK_ASIDE, look_ahead=LOOK_AHEAD):
        """
        Track the beats with a simple auto-correlation method.

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
            interval = detect_dominant_interval(act, act_smooth=smooth,
                                                hist_smooth=None,
                                                min_tau=min_tau,
                                                max_tau=max_tau)
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
        # remove beats with negative times
        self.detections = detections[np.searchsorted(detections, 0):]
        # also return the detections
        return self.detections
