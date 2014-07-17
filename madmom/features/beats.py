#!/usr/bin/env python
# encoding: utf-8
"""
This file contains all beat tracking related functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""
import os
import glob
import sys
import numpy as np

from . import Activations, RNNEventDetection


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
    import warnings
    warnings.warn('This function will be removed soon! Please update your '
                  'code to work without this function.')
    from .tempo import smooth_signal, interval_histogram_acf, dominant_interval
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


class RNNBeatTracking(RNNEventDetection):
    # set the path to saved neural networks and generate lists of NN files
    # TODO: where should the NN_FILES get defined?
    NN_PATH = '%s/../ml/data' % (os.path.dirname(__file__))
    NN_FILES = glob.glob("%s/beats_blstm*npz" % NN_PATH)
    # default values for beat detection
    SMOOTH = 0.09
    LOOK_ASIDE = 0.2
    LOOK_AHEAD = 4
    MIN_BPM = 40
    MAX_BPM = 240

    def __init__(self, data, nn_files=NN_FILES, **kwargs):
        """
        Use RNNs to compute the beat activation function and then align the
        beats according to the previously determined global tempo.

        :param data:      Signal, activations or file. See EventDetection class
        :param nn_files:  list of files that define the RNN

        """
        super(RNNBeatTracking, self).__init__(data, nn_files, **kwargs)

    def pre_process(self):
        """
        Pre-process the signal to obtain a data representation suitable for RNN
        processing.

        :return:            pre-processed data

        """
        spr = super(RNNBeatTracking, self)
        spr.pre_process(frame_sizes=[1024, 2048, 4096], bands_per_octave=3,
                        mul=1, ratio=0.25)
        # return data
        return self._data

    def detect(self, smooth=SMOOTH, min_bpm=MIN_BPM, max_bpm=MAX_BPM,
               look_aside=LOOK_ASIDE, look_ahead=LOOK_AHEAD):
        """
        Track the beats with a simple auto-correlation method.

        :param smooth:     smooth the activation function over N seconds
        :param min_bpm:    minimum tempo used for beat tracking
        :param max_bpm:    maximum tempo used for beat tracking
        :param look_aside: look this fraction of a beat interval to the side
        :param look_ahead: look N seconds ahead (and back) to determine the
                           tempo
        :return:           detected beat positions

        Note: If `look_ahead` is undefined, a constant tempo throughout the
              whole piece is assumed.
              If `look_ahead` is set, the local tempo (in a range +/-
              look_ahead seconds around the actual position) is estimated and
              then the next beat is tracked accordingly. This procedure is
              repeated from the new position to the end of the piece.

        "Enhanced Beat Tracking with Context-Aware Neural Networks"
        Sebastian Böck and Markus Schedl
        Proceedings of the 14th International Conference on Digital Audio
        Effects (DAFx-11), Paris, France, September 2011

        """
        # convert timing information to frames and set default values
        # TODO: use at least 1 frame if any of these values are > 0?
        min_tau = int(np.floor(60. * self.fps / max_bpm))
        max_tau = int(np.ceil(60. * self.fps / min_bpm))

        # if look_ahead is not defined, assume a global tempo
        if look_ahead is None:
            # detect the dominant interval (i.e. global tempo)
            interval = detect_dominant_interval(self.activations,
                                                act_smooth=smooth,
                                                hist_smooth=None,
                                                min_tau=min_tau,
                                                max_tau=max_tau)
            # detect beats based on this interval
            detections = detect_beats(self.activations, interval, look_aside)

        else:
            # allow varying tempo
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
        # remove beats with negative times and save them to detections
        self._detections = detections[np.searchsorted(detections, 0):]
        # also return the detections
        return self._detections

    @classmethod
    def add_arguments(cls, parser, nn_files=NN_FILES, smooth=SMOOTH,
                      min_bpm=MIN_BPM, max_bpm=MAX_BPM, look_aside=LOOK_ASIDE,
                      look_ahead=LOOK_AHEAD):
        """
        Add BeatDetector related arguments to an existing parser object.

        :param parser:     existing argparse parser object
        :param nn_files:   list with files of NN models
        :param smooth:     smooth the beat activations over N seconds
        :param min_bpm:    minimum tempo used for beat tracking
        :param max_bpm:    maximum tempo used for beat tracking
        :param look_aside: fraction of beat interval to look aside for the
                           strongest beat
        :param look_ahead: seconds to look ahead in order to estimate the local
                           tempo and align the next beat
        :return:           beat argument parser group object

        """
        # add Activations parser
        Activations.add_arguments(parser)
        # add arguments from RNNEventDetection
        RNNEventDetection.add_arguments(parser, nn_files=nn_files)
        # add beat detection related options to the existing parser
        g = parser.add_argument_group('beat detection arguments')
        g.add_argument('--smooth', action='store', type=float, default=smooth,
                       help='smooth the beat activations over N seconds '
                       '[default=%(default).2f]')
        # TODO: refactor this stuff to use the TempoEstimation functionality
        g.add_argument('--min_bpm', action='store', type=float,
                       default=min_bpm, help='minimum tempo [bpm, '
                       ' default=%(default).2f]')
        g.add_argument('--max_bpm', action='store', type=float,
                       default=max_bpm, help='maximum tempo [bpm, '
                       ' default=%(default).2f]')
        # make switchable (useful for including the beat stuff for tempo
        if look_aside is not None:
            g.add_argument('--look_aside', action='store', type=float,
                           default=look_aside,
                           help='look this fraction of the beat interval '
                                'around the beat to get the strongest one '
                                '[default=%(default).2f]')
        if look_ahead is not None:
            g.add_argument('--look_ahead', action='store', type=float,
                           default=look_ahead,
                           help='look this many seconds ahead to estimate the '
                                'local tempo [default=%(default).2f]')
        # return the argument group so it can be modified if needed
        return g


class CRFBeatDetection(RNNBeatTracking):
    """Conditional Random Field Beat Detection"""

    MIN_BPM = 20
    MAX_BPM = 240
    SMOOTH = 0.09
    INTERVAL_SIGMA = 0.18
    FACTORS = [0.5, 0.67, 1.0, 1.5, 2.0]

    try:
        from crf_viterbi import viterbi
    except ImportError:
        import warnings
        warnings.warn("CRFBeatDetection only works if you build the viterbi "
                      "code with cython!")

    def __init__(self, data, nn_files=RNNBeatTracking.NN_FILES, **kwargs):
        """
        Use RNNs to compute the beat activation function and then align the
        beats according to the previously determined global tempo using
        a conditional random field model.

        :param data:      Signal, activations or file. See EventDetection class
        :param nn_files:  list of files that define the RNN

        """
        super(CRFBeatDetection, self).__init__(data, nn_files, **kwargs)

    @staticmethod
    def initial_distribution(n_states, dominant_interval):
        """
        Compute the initial distribution.

        :param n_states:          number of states in the model
        :param dominant_interval: dominant interval of the piece [frames]
        :return:                  initial distribution of the model

        """
        init_dist = np.ones(n_states, dtype=np.float32) / dominant_interval
        init_dist[dominant_interval:] = 0
        return init_dist

    @staticmethod
    def transition_distribution(dominant_interval, interval_sigma):
        """
        Compute the transition distribution between beats.

        :param dominant_interval: dominant interval of the piece [frames]
        :param interval_sigma:    allowed deviation from the dominant interval
                                  per beat
        :return:                  transition distribution between beats

        """
        from scipy.stats import norm

        move_range = np.arange(dominant_interval * 2, dtype=np.float)
        # to avoid floating point hell due to np.log2(0)
        move_range[0] = 0.1

        trans_dist = norm.pdf(np.log2(move_range),
                              loc=np.log2(dominant_interval),
                              scale=interval_sigma)
        trans_dist /= trans_dist.sum()
        return trans_dist.astype(np.float32)

    @staticmethod
    def normalisation_factors(activations, transition_distribution):
        """
        Compute normalisation factors for model.

        :param activations:             activations of the piece
        :param transition_distribution: transition distribution of the model
        :return:                        normalisation factors for model

        """
        from scipy.ndimage.filters import correlate1d
        return correlate1d(activations, transition_distribution,
                           mode='constant', cval=0,
                           origin=-int(transition_distribution.shape[0] / 2))

    @staticmethod
    def best_sequence(activations, smooth, dominant_interval, interval_sigma):
        """
        Extract the best beat sequence for a piece.

        :param activations:       activations
        :param dominant_interval: dominant interval of the piece.
        :param interval_sigma:    allowed deviation from the dominant interval
                                  per beat
        :return:                  tuple with extracted beat positions [frames]
                                  and log probability of beat sequence
        """
        # shortcut!
        crb = CRFBeatDetection

        init = crb.initial_distribution(activations.shape[0],
                                        dominant_interval)
        trans = crb.transition_distribution(dominant_interval, interval_sigma)
        norm_fact = crb.normalisation_factors(activations, trans)

        return crb.viterbi(init, trans, norm_fact, activations,
                           dominant_interval)

    def detect(self, smooth=SMOOTH, min_bpm=MIN_BPM, max_bpm=MAX_BPM,
               interval_sigma=INTERVAL_SIGMA, factors=FACTORS):
        """
        Detect the beats with a conditional random field method based on
        neural network activations and a tempo estimation using
        auto-correlation.

        :param smooth:         smooth the activations over N seconds
        :param min_bpm:        minimum tempo used for beat tracking
        :param max_bpm:        maximum tempo used for beat tracking
        :param interval_sigma: allowed deviation from the dominant interval per
                               beat
        :param factors:        factors of the dominant interval to try
        :return:               detected beat positions

        """
        # convert timing information to frames and set default values
        min_interval = int(np.floor(60. * self.fps / max_bpm))
        max_interval = int(np.ceil(60. * self.fps / min_bpm))
        # detect the dominant interval
        # TODO: refactor this to use new feature.tempo functionality
        #       and add ability to handle multiple tempi
        dominant_interval = detect_dominant_interval(self.activations,
                                                     act_smooth=smooth,
                                                     min_tau=min_interval,
                                                     max_tau=max_interval)
        # create variations of the dominant interval to check
        possible_intervals = [int(dominant_interval * f) for f in factors]
        # remove all intervals ouside the allowed range
        possible_intervals = [i for i in possible_intervals
                              if max_interval >= i >= min_interval]
        # get the best beat sequences for all intervals
        results = [CRFBeatDetection.best_sequence(self.activations, smooth,
                                                  interval, interval_sigma)
                   for interval in possible_intervals]
        # normalise their probabilities
        normalised_seq_probabilities = np.array([r[1] / r[0].shape[0]
                                                 for r in results])
        # pick the best one
        best_seq = results[normalised_seq_probabilities.argmax()][0]
        # save the detected beats
        self._detections = best_seq.astype(np.float) / self.fps
        # and return them
        return self._detections

    @classmethod
    def add_arguments(cls, parser, nn_files=RNNBeatTracking.NN_FILES,
                      interval_sigma=INTERVAL_SIGMA, smooth=SMOOTH,
                      min_bpm=MIN_BPM, max_bpm=MAX_BPM, factors=FACTORS):
        """
        Add CRFBeatDetection related arguments to an existing parser object.

        :param parser:         existing argparse parser object
        :param nn_files:       list with files of NN models
        :param interval_sigma: allowed deviation from the dominant interval per
                               beat
        :param smooth:         smooth the beat activations over N seconds
        :param min_bpm:        minimum tempo used for beat tracking
        :param max_bpm:        maximum tempo used for beat tracking
        :param factors:        factors of the dominant interval to try
        :return:               beat argument parser group object
        """
        # add RNNBeatTracking arguments
        g = RNNBeatTracking.add_arguments(parser, nn_files=nn_files,
                                          smooth=smooth, min_bpm=min_bpm,
                                          max_bpm=max_bpm, look_ahead=None,
                                          look_aside=None)
        # add CRF related arguments
        g.add_argument('--interval_sigma', action='store', type=float,
                       default=interval_sigma,
                       help='allowed deviation from the dominant interval '
                            '[default=%(default).2f]')
        from ..utils import OverrideDefaultListAction
        g.add_argument('--factor', '-f', action=OverrideDefaultListAction,
                       type=float, default=factors, dest='factors',
                       help='factors of dominant interval to try. '
                            'multiple factors can be given, one factor per '
                            'argument. [default=%(default)s]')
        return g
