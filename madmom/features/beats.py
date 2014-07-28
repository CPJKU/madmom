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
    :return:            dominant interval

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

    Note: A Hamming window of 2*look_aside*interval is applied around the
          position where the beat is expected to prefer beats closer to the
          centre.

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
            :return:         the next beat position

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
    """
    Class for tracking beats with a recurrent neural network (RNN).

    """
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
        beats according to the previously determined tempo.

        :param data:      Signal, activations or file. See EventDetection class
        :param nn_files:  list of files that define the RNN

        "Enhanced Beat Tracking with Context-Aware Neural Networks"
        Sebastian Böck and Markus Schedl
        Proceedings of the 14th International Conference on Digital Audio
        Effects (DAFx-11), Paris, France, September 2011

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
                        mul=1, ratio=0.5)
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
        # only keep beats with a bigger inter beat interval than that of the
        # maximum allowed tempo
        # self._detections = np.append(detections[0],
        #                              detections[1:][np.diff(detections)
        #                                             > (60. / max_bpm)])
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
    """
    Conditional Random Field Beat Detection.

    """
    MIN_BPM = 20
    MAX_BPM = 240
    SMOOTH = 0.09
    INTERVAL_SIGMA = 0.18
    FACTORS = [0.5, 0.67, 1.0, 1.5, 2.0]

    try:
        from viterbi import crf_viterbi
    except ImportError:
        import warnings
        warnings.warn('CRFBeatDetection only works if you build the viterbi '
                      'code with cython!')

    def __init__(self, data, nn_files=RNNBeatTracking.NN_FILES, **kwargs):
        """
        Use RNNs to compute the beat activation function and then align the
        beats according to the previously determined global tempo using
        a conditional random field model.

        :param data:      Signal, activations or file. See EventDetection class
        :param nn_files:  list of files that define the RNN

        "Probabilistic extraction of beat positions from a beat activation
         function"
        Filip Korzeniowski, Sebastian Böck and Gerhard Widmer
        In Proceedings of the 15th International Society for Music Information
        Retrieval Conference (ISMIR 2014), Taipeh, Taiwan, November 2014.

        """
        super(CRFBeatDetection, self).__init__(data, nn_files, **kwargs)

    @staticmethod
    def initial_distribution(num_states, dominant_interval):
        """
        Compute the initial distribution.

        :param num_states:        number of states in the model
        :param dominant_interval: dominant interval of the piece [frames]
        :return:                  initial distribution of the model

        """
        init_dist = np.ones(num_states, dtype=np.float32) / dominant_interval
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
        move_range[0] = 0.000001

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

    @classmethod
    def best_sequence(cls, activations, dominant_interval, interval_sigma):
        """
        Extract the best beat sequence for a piece.

        :param activations:       activations
        :param dominant_interval: dominant interval of the piece.
        :param interval_sigma:    allowed deviation from the dominant interval
                                  per beat
        :return:                  tuple with extracted beat positions [frames]
                                  and log probability of beat sequence
        """
        init = cls.initial_distribution(activations.shape[0],
                                        dominant_interval)
        trans = cls.transition_distribution(dominant_interval, interval_sigma)
        norm_fact = cls.normalisation_factors(activations, trans)

        return cls.crf_viterbi(init, trans, norm_fact, activations,
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
        # remove all intervals outside the allowed range
        possible_intervals = [i for i in possible_intervals
                              if max_interval >= i >= min_interval]
        # get the best beat sequences for all intervals
        results = [CRFBeatDetection.best_sequence(self.activations, interval,
                                                  interval_sigma)
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


class MMBeatTracking(RNNBeatTracking):
    """
    Class for tracking beats with a recurrent neural network (RNN) while
    choosing the most appropriate model automatically and infer the beats with
    a dynamic Bayesian network.

    """
    # set the path to saved neural networks and generate lists of NN files
    NN_REF_FILES = glob.glob("%s/beats_ref_blstm*npz" %
                             RNNBeatTracking.NN_PATH)
    # default values for beat detection
    NUM_BEAT_STATES = 640
    TEMPO_CHANGE_PROBABILITY = 0.002
    OBSERVATION_LAMBDA = 16
    MIN_BPM = 56
    MAX_BPM = 215
    CORRECT = True

    try:
        from viterbi import mm_viterbi
    except ImportError:
        import warnings
        warnings.warn('MMBeatTracking only works if you build the viterbi '
                      'code with cython!')

    def __init__(self, data, nn_files=RNNBeatTracking.NN_FILES,
                 nn_ref_files=NN_REF_FILES, **kwargs):
        """
        Use RNNs to compute the beat activation function and then align the
        beats according to the previously determined global tempo.

        :param data:        Signal, activations or file.
        :param nn_files:    list of files that define the RNN
        :param ref_nn_file: list of files that define the reference NN model

        "A multi-model approach to beat tracking considering heterogeneous
         music styles"
        Sebastian Böck, Florian Krebs and Gerhard Widmer
        Proceedings of the 15th International Society for Music Information
        Retrieval Conference (ISMIR 2014), Taipeh, Taiwan, November 2014

        """
        super(RNNBeatTracking, self).__init__(data, nn_files, **kwargs)
        self.nn_ref_files = nn_ref_files
        self._states = None

    def process(self):
        """
        Computes the predictions on the data with the RNN models defined/given
        and save the predictions of the most suitable model as activations.

        :return: most suitable RNN activation (prediction)

        """
        from ..ml.rnn import process_rnn
        # append the nn_files to the list of reference model(s)
        nn_files = self.nn_ref_files + self.nn_files
        # compute the predictions with RNNs, do not average them
        predictions = process_rnn(self.data, nn_files, self.num_threads,
                                  average=False)
        # get the reference predictions
        num_ref_files = len(self.nn_ref_files)
        if num_ref_files > 1:
            # if we have multiple reference networks, average their predictions
            reference_prediction = (sum(predictions[:num_ref_files]) /
                                    num_ref_files)
        elif num_ref_files == 1:
            # if only 1 reference network was given, use the first prediction
            reference_prediction = predictions[0]
        else:
            # just average all predictions to simulate a reference network
            reference_prediction = sum(predictions) / len(nn_files)
        # init the error with the max. possible value (i.e. prediction length)
        best_error = len(reference_prediction)
        # compare the (remaining) predictions with the reference prediction
        for prediction in predictions[num_ref_files:]:
            # calculate the squared error w.r.t. the reference prediction
            error = np.sum((prediction - reference_prediction) ** 2.)
            # chose the best activation
            if error < best_error:
                best_prediction = prediction
                best_error = error
        # save the best prediction as activations
        self._activations = Activations(best_prediction.ravel(), self.fps)
        # and return them
        return self._activations

    def detect(self, num_beat_states=NUM_BEAT_STATES,
               tempo_change_probability=TEMPO_CHANGE_PROBABILITY,
               observation_lambda=OBSERVATION_LAMBDA, min_bpm=MIN_BPM,
               max_bpm=MAX_BPM, correct=CORRECT,
               num_threads=RNNEventDetection.NUM_THREADS):
        """
        Track the beats with a dynamic Bayesian network.

        :param num_beat_states:          number of cells for one beat period
        :param tempo_change_probability: probability of a tempo change from
                                         one observation to the next one
        :param observation_lambda:       split one beat period into N parts,
                                         the first representing beat states
                                         and the remaining non-beat states
        :param min_bpm:                  minimum tempo used for beat tracking
        :param max_bpm:                  maximum tempo used for beat tracking
        :param correct:                  correct the beat positions
        :param num_threads:              number of parallel threads
        :return:                         detected beat positions

        """
        # convert timing information to the tempo space
        max_tau = int(np.ceil(max_bpm * num_beat_states / (60. * self.fps)))
        min_tau = int(np.floor(min_bpm * num_beat_states / (60. * self.fps)))
        # infer the state space sequence
        path = self.mm_viterbi(self.activations,
                               num_beat_states=num_beat_states,
                               tempo_change_probability=
                               tempo_change_probability,
                               min_tau=min_tau, max_tau=max_tau,
                               observation_lambda=observation_lambda,
                               num_threads=num_threads)
        # determine the frame indices with the smallest beat states
        states = path % num_beat_states
        self._states = states

        # correct the beat positions
        if correct:
            detections = []
            # for each detection determine the "beat range", i.e. states <=
            # num_beat_states / observation_lambda and choose the frame with
            # the highest activations value
            beat_range = states < num_beat_states / observation_lambda
            # get all change points between True and False
            idx = np.nonzero(np.diff(beat_range))[0] + 1
            # if the first frame is in the beat range, prepend a 0
            if beat_range[0]:
                idx = np.r_[0, idx]
            # if the last frame is in the beat range, append the length of the
            # array
            if beat_range[-1]:
                idx = np.r_[idx, beat_range.size]
            # iterate over all regions
            for left, right in idx.reshape((-1, 2)):
                detections.append(np.argmax(self.activations[left:right]) +
                                  left)
            detections = np.asarray(detections, np.float)
        else:
            # just take the frames with the smallest beat state values
            from scipy.signal import argrelmin
            detections = argrelmin(states, mode='wrap')[0]
            # recheck if they are within the "beat range", i.e. the
            # beat states < number of beat states / observation lambda
            detections = detections[states[detections] <
                                    num_beat_states / observation_lambda]
            # Note: interpolation and alignment of the beats to be at state 0
            #       does not improve results over this simple method
        # convert them to a list of timestamps
        self._detections = detections / float(self.fps)
        # also return the detections
        return self._detections

    @property
    def states(self):
        """States of the dynamic Bayesian network."""
        if self._states is None:
            self.detect()
        return self._states

    @classmethod
    def add_arguments(cls, parser, nn_files=RNNBeatTracking.NN_FILES,
                      nn_ref_files=NN_REF_FILES,
                      num_beat_states=NUM_BEAT_STATES,
                      tempo_change_probability=TEMPO_CHANGE_PROBABILITY,
                      observation_lambda=OBSERVATION_LAMBDA, min_bpm=MIN_BPM,
                      max_bpm=MAX_BPM, correct=CORRECT):
        """
        Add MMBeatTracking related arguments to an existing parser object.

        :param parser:                   existing argparse parser object
        :param nn_files:                 list with files of NN models
        :param nn_ref_files:             reference NN model
        :param num_beat_states:          number of cells for one beat period
        :param tempo_change_probability: probability of a tempo change between
                                         two adjacent observations
        :param observation_lambda:       split one beat period into N parts,
                                         the first representing beat states
                                         and the remaining non-beat states
        :param min_bpm:                  minimum tempo used for beat tracking
        :param max_bpm:                  maximum tempo used for beat tracking
        :param correct:                  correct the beat positions
        :return:                         beat argument parser group object

        """
        # add Activations parser
        Activations.add_arguments(parser)
        # add arguments from RNNEventDetection
        g = RNNEventDetection.add_arguments(parser, nn_files=nn_files)
        g.add_argument('--nn_ref_files', action='append', type=str,
                       default=nn_ref_files,
                       help='compare the predictions to these pre-trained '
                            'neural networks (multiple files can be given, '
                            'one file per argument) and choose the most '
                            'suitable one accordingly (i.e. the one with the '
                            'least deviation form these reference models).')
        # add beat detection related options to the existing parser
        g = parser.add_argument_group('beat detection arguments')
        g.add_argument('--num_beat_states', action='store', type=int,
                       default=num_beat_states,
                       help='number of beat states for one beat period '
                            '[default=%(default)i]')
        g.add_argument('--tempo_change_probability', action='store',
                       type=float, default=tempo_change_probability,
                       help='probability of a tempo between two adjacent '
                            'observations [default=%(default).4f]')
        g.add_argument('--observation_lambda', action='store', type=int,
                       default=observation_lambda,
                       help='split one beat period into N parts, the first '
                            'representing beat states and the remaining '
                            'non-beat states [default=%(default)i]')
        g.add_argument('--min_bpm', action='store', type=float,
                       default=min_bpm, help='minimum tempo [bpm, '
                       ' default=%(default).2f]')
        g.add_argument('--max_bpm', action='store', type=float,
                       default=max_bpm, help='maximum tempo [bpm, '
                       ' default=%(default).2f]')
        if correct:
            g.add_argument('--no_correct', dest='correct',
                           action='store_false', default=correct,
                           help='do not correct the beat positions')
        else:
            g.add_argument('--correct', dest='correct',
                           action='store_true', default=correct,
                           help='correct the beat positions')
        # return the argument group so it can be modified if needed
        return g
