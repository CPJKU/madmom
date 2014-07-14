#!/usr/bin/env python
# encoding: utf-8
"""
This file contains all beat tracking related functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import sys
import os
import glob
import numpy as np

from .tempo import (smooth_signal, interval_histogram_acf, dominant_interval,
                    TempoEstimator)

from . import EventDetection, RNNEventDetection


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


class BeatDetector(RNNEventDetection):
    # set the path to saved neural networks and generate lists of NN files
    NN_PATH = '%s/../ml/data' % (os.path.dirname(__file__))
    NN_FILES = glob.glob("%s/beats_blstm*npz" % NN_PATH)

    # default values for beat detection
    SMOOTH = 0.09
    LOOK_ASIDE = 0.2

    def __init__(self, data, nn_files=NN_FILES, smooth=SMOOTH,
                 min_bpm=TempoEstimator.MIN_BPM,
                 max_bpm=TempoEstimator.MAX_BPM, look_aside=LOOK_ASIDE,
                 **kwargs):
        """
        Detect the beats with a simple auto-correlation method.

        :param data:       Signal, activations or filename.
                           See EventDetection class for more details.
        :param nn_files:   list of files that define the RNN
        :param smooth:     smooth the activation function over N seconds
        :param min_bpm:    minimum tempo used for beat tracking
        :param max_bpm:    maximum tempo used for beat tracking
        :param look_aside: look this fraction of a beat interval to the side

        For more parameters see the parent classes.

        First the global tempo is estimated and then the beats are aligned
        according to:

        "Enhanced Beat Tracking with Context-Aware Neural Networks"
        Sebastian Böck and Markus Schedl
        Proceedings of the 14th International Conference on Digital Audio
        Effects (DAFx-11), Paris, France, September 2011

        """

        super(BeatDetector, self).__init__(data, nn_files=nn_files, **kwargs)

        self.smooth = int(round(self.fps * smooth))
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.look_aside = look_aside

    def detect(self):
        """ See EventDetection class """
        # convert timing information to frames and set default values
        # TODO: use at least 1 frame if any of these values are > 0?
        min_tau = int(np.floor(60. * self.fps / self.max_bpm))
        max_tau = int(np.ceil(60. * self.fps / self.min_bpm))
        # detect the dominant interval
        acts = self.activations
        interval = detect_dominant_interval(acts, act_smooth=self.smooth,
                                            hist_smooth=None,
                                            min_tau=min_tau, max_tau=max_tau)
        # detect beats based on this interval
        detections = detect_beats(acts, interval, self.look_aside)
        # convert detected beats to a list of timestamps
        detections = np.array(detections) / float(self.fps)
        # remove beats with negative times & return
        return detections[np.searchsorted(detections, 0):]

    @classmethod
    def add_arguments(cls, parser, smooth=SMOOTH, look_aside=LOOK_ASIDE,
                      nn_files=NN_FILES, **kwargs):
        """
        Add BeatDetector related arguments to an existing parser object.

        :param parser:     existing argparse parser object
        :param smooth:     smooth the beat activations over N seconds
        :param look_aside: fraction of beat interval to look aside for the
                           strongest beat
        :param nn_files: list of NN files
        :return:           beat argument parser group object

        """
        # add EventDetection parser
        EventDetection.add_arguments(parser)
        # add arguments from RNNEventDetection
        RNNEventDetection.add_arguments(parser, nn_files=nn_files)

        # add beat detection related options to the existing parser
        g = parser.add_argument_group('beat detection arguments')
        g.add_argument('--smooth', action='store', type=float, default=smooth,
                       help='smooth the beat activations over N seconds '
                       '[default=%(default).2f]')
        # make switchable (useful for including the beat stuff for tempo
        if look_aside is not None:
            g.add_argument('--look_aside', action='store', type=float,
                           default=look_aside, help='look this fraction of '
                           'the beat interval around the beat to get the '
                           'strongest one [default=%(default).2f]')
        # return the argument group so it can be modified if needed
        return g


class BeatTracker(RNNEventDetection):
    # set the path to saved neural networks and generate lists of NN files
    NN_PATH = '%s/../ml/data' % (os.path.dirname(__file__))
    NN_FILES = glob.glob("%s/beats_blstm*npz" % NN_PATH)

    # default values for beat tracking
    SMOOTH = 0.09
    LOOK_ASIDE = 0.2
    LOOK_AHEAD = 4

    def __init__(self, data, nn_files=NN_FILES):
                 # , smooth=SMOOTH,
                 # min_bpm=TempoEstimator.MIN_BPM,
                 # max_bpm=TempoEstimator.MAX_BPM, look_aside=LOOK_ASIDE,
                 # look_ahead=LOOK_AHEAD, **kwargs):
        """
        Track the beats with a simple auto-correlation method.

        :param data:       Signal, activations or filename.
                           See EventDetection class for more details.
        :param nn_files:   list of files that define the RNN
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

        super(BeatTracker, self).__init__(data, nn_files=nn_files, **kwargs)

        self.smooth = int(round(self.fps * smooth))
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.look_aside = look_aside
        self.look_ahead = look_ahead

    def detect(self):
        """ See EventDetection class """
        # convert timing information to frames and set default values
        # TODO: use at least 1 frame if any of these values are > 0?
        min_tau = int(np.floor(60. * self.fps / self.max_bpm))
        max_tau = int(np.ceil(60. * self.fps / self.min_bpm))
        look_ahead_frames = int(self.look_ahead * self.fps)

        activations = self.activations

        # detect the beats
        detections = []
        pos = 0
        # TODO: make this _much_ faster!
        while pos < len(activations):
            # look N frames around the actual position
            start = pos - look_ahead_frames
            end = pos + look_ahead_frames
            if start < 0:
                # pad with zeros
                act = np.append(np.zeros(-start), activations[0:end])
            elif end > len(activations):
                # append zeros accordingly
                zeros = np.zeros(end - len(activations))
                act = np.append(activations[start:], zeros)
            else:
                act = activations[start:end]
            # detect the dominant interval
            interval = detect_dominant_interval(act, act_smooth=self.smooth,
                                                hist_smooth=None,
                                                min_tau=min_tau,
                                                max_tau=max_tau)
            # add the offset (i.e. the new detected start position)
            positions = np.array(detect_beats(act, interval, self.look_aside))
            # correct the beat positions
            positions += start
            # search the closest beat to the predicted beat position
            pos = positions[(np.abs(positions - pos)).argmin()]
            # append to the beats
            detections.append(pos)
            pos += interval

        # convert detected beats to a list of timestamps
        detections = np.array(detections) / float(self.fps)
        # remove beats with negative times & return
        return detections[np.searchsorted(detections, 0):]

    @classmethod
    def add_arguments(cls, parser, smooth=SMOOTH, look_aside=LOOK_ASIDE,
                      look_ahead=LOOK_AHEAD, nn_files=NN_FILES, **kwargs):
        """
        Add BeatTracker related arguments to an existing parser object.

        :param parser:     existing argparse parser object
        :param smooth:     smooth the beat activations over N seconds
        :param look_aside: fraction of beat interval to look aside for the
                           strongest beat
        :param look_ahead: seconds to look ahead in order to estimate the local
                           tempo
        :param nn_files:   list of files that define the RNN
        :return:           beat argument parser group object

        """
        # add EventDetection parser
        EventDetection.add_arguments(parser)
        # add arguments from RNNEventDetection
        RNNEventDetection.add_arguments(parser, nn_files=nn_files)

        # add beat detection related options to the existing parser
        g = parser.add_argument_group('beat detection arguments')
        g.add_argument('--smooth', action='store', type=float, default=smooth,
                    help='smooth the beat activations over N seconds '
                         '[default=%(default).2f]')

        # make switchable (useful for including the beat stuff for tempo
        if look_aside is not None:
            g.add_argument('--look_aside', action='store', type=float,
                        default=look_aside,
                        help='look this fraction of the beat interval around '
                             'the beat to get the strongest one '
                             '[default=%(default).2f]')

        if look_ahead is not None:
            g.add_argument('--look_ahead', action='store', type=float,
                        default=look_ahead,
                        help='look this many seconds ahead to estimate the '
                             'local tempo [default=%(default).2f]')
        # return the argument group so it can be modified if needed
        return g
