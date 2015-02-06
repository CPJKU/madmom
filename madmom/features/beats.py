#!/usr/bin/env python
# encoding: utf-8
"""
This file contains all beat tracking related functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import sys
import glob
import numpy as np

from madmom import MODELS_PATH, IOProcessor
from madmom.audio.signal import SignalProcessor, smooth as smooth_signal
from madmom.audio.spectrogram import StackSpectrogramProcessor
from madmom.ml.rnn import RNNProcessor
from madmom.utils import write_events
from madmom.features.tempo import TempoEstimationProcessor


# detect the beats based on the given dominant interval
def detect_beats(activations, interval, look_aside=0.2):
    """
    Detects the beats in the given activation function.

    :param activations: array with beat activations
    :param interval:    look for the next beat each N frames
    :param look_aside:  look this fraction of the interval to the side to
                        detect the beats

    Note: A Hamming window of 2 * `look_aside` * `interval` is applied around
         the position where the beat is expected to prefer beats closer to the
         centre.

    "Enhanced Beat Tracking with Context-Aware Neural Networks"
    Sebastian Böck and Markus Schedl
    Proceedings of the 14th International Conference on Digital Audio
    Effects (DAFx-11), 2011

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
            act_ = np.multiply(act, win)
            # search max
            if np.argmax(act_) > 0:
                # maximum found, take that position
                position = np.argmax(act_) + start
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
    # return indices
    return np.array(positions)


# classes for the detection of the beat inside a beat activation function
class BeatTrackingProcessor(IOProcessor):
    """
    Class for tracking beats with a simple tempo estimation and beat aligning.

    """
    LOOK_ASIDE = 0.2
    LOOK_AHEAD = None
    # tempo defaults
    TEMPO_METHOD = 'comb'
    MIN_BPM = 40
    MAX_BPM = 240
    ACT_SMOOTH = 0.09
    HIST_SMOOTH = 7
    ALPHA = 0.79

    def __init__(self, look_aside=LOOK_ASIDE, look_ahead=LOOK_AHEAD, **kwargs):
        """
        Track the beats according to the previously determined (global) tempo
        by simply aligning them around the estimated position.

        :param look_aside:   look this fraction of a beat interval to each side
                             of the assumed next beat position to look for the
                             most likely position of the next beat
        :param look_ahead:   look N seconds in both directions to determine the
                             local tempo and align the beats accordingly

        If `look_ahead` is not set, a constant tempo throughout the whole piece
        is assumed. If `look_ahead` is set, the local tempo (in a range +/-
        look_ahead seconds around the actual position) is estimated and then
        the next beat is tracked accordingly. This procedure is repeated from
        the new position to the end of the piece.

        "Enhanced Beat Tracking with Context-Aware Neural Networks"
        Sebastian Böck and Markus Schedl
        Proceedings of the 14th International Conference on Digital Audio
        Effects (DAFx), 2011

        Instead of the auto-correlation based method for tempo estimation, it
        uses a comb filter per default. The behaviour can be controlled with
        the `tempo_method` parameter.

        """
        # make this an IOProcessor by defining input and output processings
        super(BeatTrackingProcessor, self).__init__(self.detect, write_events)
        # save variables
        self.look_aside = look_aside
        self.look_ahead = look_ahead
        # get fps from kwargs
        self.fps = kwargs.get('fps', None)
        # tempo estimator
        self.tempo_estimator = TempoEstimationProcessor(**kwargs)

    def detect(self, activations):
        """
        Detect the beats in the given activation function.

        :param activations: beat activation function
        :return:            detected beat positions

        """
        # smooth activations
        act_smooth = int(self.fps * self.tempo_estimator.act_smooth)
        activations = smooth_signal(activations, act_smooth)
        # TODO: refactor interval stuff to use TempoEstimationProcessor
        # if look_ahead is not defined, assume a global tempo
        if self.look_ahead is None:
            # create a interval histogram
            histogram = self.tempo_estimator.interval_histogram(activations)
            # get the dominant interval
            interval = self.tempo_estimator.dominant_interval(histogram)
            # detect beats based on this interval
            detections = detect_beats(activations, interval, self.look_aside)
        else:
            # allow varying tempo
            look_ahead_frames = int(self.look_ahead * self.fps)
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
                # create a interval histogram
                histogram = self.tempo_estimator.interval_histogram(act)
                # get the dominant interval
                interval = self.tempo_estimator.dominant_interval(histogram)
                # add the offset (i.e. the new detected start position)
                positions = detect_beats(act, interval, self.look_aside)
                # correct the beat positions
                positions += start
                # search the closest beat to the predicted beat position
                pos = positions[(np.abs(positions - pos)).argmin()]
                # append to the beats
                detections.append(pos)
                pos += interval

        # convert detected beats to a list of timestamps
        detections = np.array(detections) / float(self.fps)
        # remove beats with negative times and return them
        return detections[np.searchsorted(detections, 0):]
        # only return beats with a bigger inter beat interval than that of the
        # maximum allowed tempo
        # return np.append(detections[0], detections[1:][np.diff(detections) >
        #                                                (60. / max_bpm)])

    @classmethod
    def add_arguments(cls, parser, look_aside=LOOK_ASIDE,
                      look_ahead=LOOK_AHEAD):
        """
        Add beat tracking related arguments to an existing parser.

        :param parser:      existing argparse parser
        :param look_aside:  look this fraction of a beat interval to each side
                            of the assumed next beat position to look for the
                            most likely position of the next beat
        :param look_ahead:  look N seconds in both directions to determine the
                            local tempo and align the beats accordingly
        :return:            beat argument parser group

        Parameters are included in the group only if they are not 'None'.

        """
        # add beat detection related options to the existing parser
        g = parser.add_argument_group('beat detection arguments')
        # TODO: unify look_aside with CRFBeatDetection's interval_sigma
        if look_aside is not None:
            g.add_argument('--look_aside', action='store', type=float,
                           default=look_aside,
                           help='look this fraction of a beat interval to '
                                'each side of the assumed next beat position '
                                'to look for the most likely position of the '
                                'next beat [default=%(default).2f]')
        if look_ahead is not None:
            g.add_argument('--look_ahead', action='store', type=float,
                           default=look_ahead,
                           help='look this many seconds in both directions '
                                'to determine the local tempo and align the '
                                'beats accordingly [default=%(default).2f]')
        # return the argument group so it can be modified if needed
        return g


# TODO: refactor the whole CRF Viterbi stuff as a .pyx class including the
#       initial_distribution and all other functionality, but omit the factors,
#       they should get replaced by the output of the comb filter stuff
def _process_crf(process_tuple):
    """
    Extract the best beat sequence for a piece.

    :param process_tuple: tuple with (activations, dominant_interval, allowed
                          deviation from the dominant interval per beat)
    :return:              tuple with extracted beat positions [frames]
                          and log probability of beat sequence

    """
    # activations, dominant_interval, interval_sigma = process_tuple
    return CRFBeatDetectionProcessor.best_sequence(*process_tuple)


class CRFBeatDetectionProcessor(BeatTrackingProcessor):
    """
    Conditional Random Field Beat Detection.

    """
    INTERVAL_SIGMA = 0.18
    FACTORS = [0.5, 0.67, 1.0, 1.5, 2.0]
    # tempo defaults
    TEMPO_METHOD = 'acf'
    MIN_BPM = 20
    MAX_BPM = 240
    ACT_SMOOTH = 0.09
    HIST_SMOOTH = 7
    ALPHA = 0.79

    try:
        from .viterbi import crf_viterbi
    except ImportError:
        import warnings
        warnings.warn('CRFBeatDetection only works if you build the viterbi '
                      'module with cython!')

    def __init__(self, interval_sigma=INTERVAL_SIGMA, factors=FACTORS,
                 **kwargs):
        """
        Track the beats according to the previously determined global tempo
        using a conditional random field model.

        :param interval_sigma: allowed deviation from the dominant beat
                               interval per beat
        :param factors:        factors of the dominant interval to try

        "Probabilistic extraction of beat positions from a beat activation
         function"
        Filip Korzeniowski, Sebastian Böck and Gerhard Widmer
        In Proceedings of the 15th International Society for Music Information
        Retrieval Conference (ISMIR), 2014.

        """
        super(CRFBeatDetectionProcessor, self).__init__(**kwargs)
        # save variables
        self.interval_sigma = interval_sigma
        self.factors = factors
        # get fps and num_frames from kwargs
        self.fps = kwargs.get('fps', None)
        self.num_threads = kwargs.get('num_threads', None)
        # TODO: implement comb filter stuff and remove this...
        self.tempo_estimator.method = 'acf'

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

    def detect(self, activations):
        """
        Detect the beats in the given activation function.

        :param activations: beat activation function
        :return:            detected beat positions

        """
        import itertools as it
        # convert timing information to frames and set default values
        act_smooth = int(self.fps * self.tempo_estimator.act_smooth)
        # smooth activations
        activations = smooth_signal(activations, act_smooth)
        # TODO: refactor interval stuff to use TempoEstimationProcessor
        # create a interval histogram
        histogram = self.tempo_estimator.interval_histogram(activations)
        # get the dominant interval
        interval = self.tempo_estimator.dominant_interval(histogram)

        # TODO: use the tempi returned by the TempoEstimationProcessor instead
        # create variations of the dominant interval to check
        possible_intervals = [int(interval * f) for f in self.factors]
        # remove all intervals outside the allowed range
        possible_intervals = [i for i in possible_intervals
                              if self.tempo_estimator.max_interval >= i >=
                              self.tempo_estimator.min_interval]
        # sort the intervals
        possible_intervals.sort()
        # put the greatest first so that it get processed first
        possible_intervals.reverse()

        # init a pool of workers (if needed)
        map_ = map
        if min(len(self.factors), max(1, self.num_threads)) != 1:
            import multiprocessing as mp
            map_ = mp.Pool(self.num_threads).map

        # compute the beat sequences (in parallel)
        # since the cython code uses memory views, we need to make sure that
        # the activations are C-contiguous and of C-type float (np.float32)
        contiguous_act = np.ascontiguousarray(activations, dtype=np.float32)
        results = map_(_process_crf, it.izip(it.repeat(contiguous_act),
                                             possible_intervals,
                                             it.repeat(self.interval_sigma)))

        # normalize their probabilities
        normalized_seq_probabilities = np.array([r[1] / r[0].shape[0]
                                                 for r in results])
        # pick the best one
        best_seq = results[normalized_seq_probabilities.argmax()][0]
        # convert the detected beat positions to seconds and return them
        return best_seq.astype(np.float) / self.fps

    @classmethod
    def add_arguments(cls, parser, interval_sigma=INTERVAL_SIGMA,
                      factors=FACTORS):
        """
        Add CRFBeatDetection related arguments to an existing parser.

        :param parser:         existing argparse parser
        :param interval_sigma: allowed deviation from the dominant interval per
                               beat
        :param factors:        factors of the dominant interval to try
        :return:               beat argument parser group

        """
        # add CRF related arguments
        g = parser.add_argument_group('conditional random field arguments')
        g.add_argument('--interval_sigma', action='store', type=float,
                       default=interval_sigma,
                       help='allowed deviation from the dominant interval '
                            '[default=%(default).2f]')
        from madmom.utils import OverrideDefaultListAction
        g.add_argument('-f', '--factor', action=OverrideDefaultListAction,
                       type=float, default=factors, dest='factors',
                       help='factors of dominant interval to try. '
                            'multiple factors can be given, one factor per '
                            'argument. [default=%(default)s]')
        return g


class DBNBeatTrackingProcessor(IOProcessor):
    """
    Beat tracking with RNNs and a DBN.

    """
    # some default values
    CORRECT = True
    NUM_BEAT_STATES = 1280
    NUM_TEMPO_STATES = None
    TEMPO_CHANGE_PROBABILITY = 0.008
    OBSERVATION_LAMBDA = 16
    NORM_OBSERVATIONS = False
    MIN_BPM = 50
    MAX_BPM = 215

    try:
        from .dbn import (BeatTrackingDynamicBayesianNetwork as DBN,
                          BeatTrackingTransitionModel as TM,
                          NNBeatTrackingObservationModel as OM)
    except ImportError:
        import warnings
        warnings.warn('MMBeatTracking only works if you build the dbn '
                      'module with cython!')

    def __init__(self, correct=CORRECT, num_beat_states=NUM_BEAT_STATES,
                 num_tempo_states=NUM_TEMPO_STATES,
                 tempo_change_probability=TEMPO_CHANGE_PROBABILITY,
                 min_bpm=MIN_BPM, max_bpm=MAX_BPM,
                 observation_lambda=OBSERVATION_LAMBDA,
                 norm_observations=NORM_OBSERVATIONS, **kwargs):
        """
        Track the beats with a dynamic Bayesian network.

        :param correct:                  correct the beats (i.e. align them
                                         to the nearest peak of the beat
                                         activation function)

        Parameters for the transition model:

        :param num_beat_states:          number of states for one beat period
        :param num_tempo_states:         number of tempo states (if set, limit
                                         the number of states and use a log
                                         spacing, otherwise a linear spacing)
        :param tempo_change_probability: probability of a tempo change between
                                         two adjacent observations
        :param min_bpm:                  minimum tempo used for beat tracking
        :param max_bpm:                  maximum tempo used for beat tracking

        Parameters for the observation model:

        :param observation_lambda:       split one beat period into N parts,
                                         the first representing beat states
                                         and the remaining non-beat states
        :param norm_observations:        normalize the observations

        "A multi-model approach to beat tracking considering heterogeneous
         music styles"
        Sebastian Böck, Florian Krebs and Gerhard Widmer
        Proceedings of the 15th International Society for Music Information
        Retrieval Conference (ISMIR), 2014

        """
        # make this an IOProcessor by defining input and output processings
        super(DBNBeatTrackingProcessor, self).__init__(self.detect,
                                                       write_events)
        self.correct = correct
        self.num_beat_states = num_beat_states
        self.num_tempo_states = num_tempo_states
        self.tempo_change_probability = tempo_change_probability
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.observation_lambda = observation_lambda
        self.norm_observations = norm_observations
        # get fps and num_frames from kwargs
        self.fps = kwargs.get('fps', None)
        self.num_threads = kwargs.get('num_threads', None)

    def detect(self, activations):
        """
        Detect the beats in the given activation function.

        :param activations: beat activation function
        :return:            detected beat positions

        """
        # convert timing information to tempo space
        max_tempo = self.max_bpm * self.num_beat_states / (60. * self.fps)
        min_tempo = self.min_bpm * self.num_beat_states / (60. * self.fps)
        if self.num_tempo_states is None:
            # do not limit the number of tempo states, use a linear spacing
            tempo_states = np.arange(np.round(min_tempo),
                                     np.round(max_tempo) + 1)
        else:
            # limit the number of tempo states, thus use a quasi log spacing
            tempo_states = np.logspace(np.log2(min_tempo),
                                       np.log2(max_tempo),
                                       self.num_tempo_states, base=2)
        # quantize to integer tempo states
        tempo_states = np.unique(np.round(tempo_states).astype(np.int))
        # transition model
        tm = self.TM(num_beat_states=self.num_beat_states,
                     tempo_states=tempo_states,
                     tempo_change_probability=self.tempo_change_probability)
        # observation model
        om = self.OM(activations, num_states=tm.num_states,
                     num_beat_states=tm.num_beat_states,
                     observation_lambda=self.observation_lambda,
                     norm_observations=self.norm_observations)
        # init the DBN
        dbn = self.DBN(transition_model=tm, observation_model=om,
                       num_threads=self.num_threads, correct=self.correct)
        # convert the detected beats to seconds and return them
        return dbn.beats / float(self.fps)

    @classmethod
    def add_arguments(cls, parser, num_beat_states=NUM_BEAT_STATES,
                      num_tempo_states=NUM_TEMPO_STATES, min_bpm=MIN_BPM,
                      max_bpm=MAX_BPM,
                      tempo_change_probability=TEMPO_CHANGE_PROBABILITY,
                      observation_lambda=OBSERVATION_LAMBDA,
                      norm_observations=NORM_OBSERVATIONS, correct=CORRECT):
        """
        Add DBN related arguments to an existing parser object.

        :param parser: existing argparse parser object

        Parameters for the transition model:

        :param num_beat_states:          number of states for one beat period
        :param num_tempo_states:         number of tempo states (if set, limit
                                         the number of states and use a log
                                         spacing, otherwise a linear spacing)
        :param min_bpm:                  minimum tempo used for beat tracking
        :param max_bpm:                  maximum tempo used for beat tracking
        :param tempo_change_probability: probability of a tempo change between
                                         two adjacent observations

        Parameters for the observation model:

        :param observation_lambda: split one beat period into N parts, the
                                   first representing beat states and the
                                   remaining non-beat states
        :param norm_observations:  normalize the observations

        Post-processing parameters:

        :param correct: correct the beat positions

        :return: beat argument parser group object

        """
        # add DBN parser group
        g = parser.add_argument_group('dynamic Bayesian Network arguments')
        if correct:
            g.add_argument('--no_correct', dest='correct',
                           action='store_false', default=correct,
                           help='do not correct the beat positions')
        else:
            g.add_argument('--correct', dest='correct',
                           action='store_true', default=correct,
                           help='correct the beat positions')
        # add a transition parameters
        g.add_argument('--num_beat_states', action='store', type=int,
                       default=num_beat_states,
                       help='number of beat states for one beat period '
                            '[default=%(default)i]')
        g.add_argument('--num_tempo_states', action='store', type=int,
                       default=num_tempo_states,
                       help='limit the number of tempo states; if set, align '
                            'them with a log spacing, otherwise linearly '
                            '[default=None]')
        g.add_argument('--min_bpm', action='store', type=float,
                       default=min_bpm,
                       help='minimum tempo [bpm, default=%(default).2f]')
        g.add_argument('--max_bpm', action='store', type=float,
                       default=max_bpm,
                       help='maximum tempo [bpm,  default=%(default).2f]')
        g.add_argument('--tempo_change_probability', action='store',
                       type=float, default=tempo_change_probability,
                       help='probability of a tempo between two adjacent '
                            'observations [default=%(default).4f]')
        # observation model stuff
        g.add_argument('--observation_lambda', action='store', type=int,
                       default=observation_lambda,
                       help='split one beat period into N parts, the first '
                            'representing beat states and the remaining '
                            'non-beat states [default=%(default)i]')
        if norm_observations:
            g.add_argument('--no_norm_obs', dest='norm_observations',
                           action='store_false', default=norm_observations,
                           help='do not normalize the observations of the DBN')
        else:
            g.add_argument('--norm_obs', dest='norm_observations',
                           action='store_true', default=norm_observations,
                           help='normalize the observations of the DBN')
        # return the argument group so it can be modified if needed
        return g


class RNNBeatProcessor(IOProcessor):
    """
    Class for tracking beats with a recurrent neural network (RNN).

    """
    NN_FILES = glob.glob("%s/beats_blstm_[1-8].npz" % MODELS_PATH)

    def __init__(self, beat_method='dbn', nn_files=NN_FILES, **kwargs):
        """
        Use (multiple) RNNs to predict a beat activation function.

        :param nn_files: list of RNN model files

        "Enhanced Beat Tracking with Context-Aware Neural Networks"
        Sebastian Böck and Markus Schedl
        Proceedings of the 14th International Conference on Digital Audio
        Effects (DAFx), 2011

        The individual beat tracking methods are described in the publications
        mentioned in the respective methods.

        """
        # FIXME: remove this hack of setting fps here
        kwargs['fps'] = 100
        # input processor chain
        sig = SignalProcessor(mono=True, **kwargs)

        self.num_threads = kwargs.get('num_threads', None)
        # TODO: this information should be stored in the nn_files
        stack = StackSpectrogramProcessor(frame_sizes=[1024, 2048, 4096],
                                          online=False, bands=3,
                                          norm_filters=True, mul=1, add=1,
                                          diff_ratio=0.5, **kwargs)
        rnn = RNNProcessor(nn_files=nn_files, **kwargs)
        in_processor = [sig, stack, rnn]
        # output processor
        self.method = getattr(self, beat_method)
        # sequentially process everything
        super(RNNBeatProcessor, self).__init__(in_processor, self.method)
        self.nn_files = nn_files
        self._kwargs = kwargs

    # define the available trackers
    def dbn(self, data, output):
        """
        Track the beats with a dynamic Bayesian network.

        :param data:
        :param output:
        :return:

        "A multi-model approach to beat tracking considering heterogeneous
         music styles"
        Sebastian Böck, Florian Krebs and Gerhard Widmer
        Proceedings of the 15th International Society for Music Information
        Retrieval Conference (ISMIR), 2014

        """
        return DBNBeatTrackingProcessor(**self._kwargs).process(data, output)

    def crf(self, data, output):
        """
        Track the beats with a conditional random field.

        :param data:
        :param output:
        :return:

        "Probabilistic extraction of beat positions from a beat activation
         function"
        Filip Korzeniowski, Sebastian Böck and Gerhard Widmer
        In Proceedings of the 15th International Society for Music Information
        Retrieval Conference (ISMIR), 2014.

        """
        return CRFBeatDetectionProcessor(**self._kwargs).process(data, output)

    def detect(self, data, output):
        """
        Detect the beats by simply aligning them at positions corresponding to
        peaks in the beat activation function with a previously determined
        interval (i.e. global tempo).

        :param data:
        :param output:
        :return:

        "Enhanced Beat Tracking with Context-Aware Neural Networks"
        Sebastian Böck and Markus Schedl
        Proceedings of the 14th International Conference on Digital Audio
        Effects (DAFx), 2011

        """
        return BeatTrackingProcessor(**self._kwargs)(data, output)

    def track(self, data, output):
        """
        Track the beats by simply aligning them at positions corresponding to
        peaks in the beat activation function with a previously determined
        interval (i.e. global tempo).

        :param data:
        :param output:
        :return:

        "Enhanced Beat Tracking with Context-Aware Neural Networks"
        Sebastian Böck and Markus Schedl
        Proceedings of the 14th International Conference on Digital Audio
        Effects (DAFx), 2011

        """
        return BeatTrackingProcessor(**self._kwargs).process(data, output)

    @classmethod
    def add_arguments(cls, parser, nn_files=NN_FILES):
        """
        Add RNN beat tracking related arguments to an existing parser.

        :param parser:   existing argparse parser
        :param nn_files: list of RNN model files

        """
        # add signal processing arguments
        SignalProcessor.add_arguments(parser, norm=False, att=0)
        # add rnn processing arguments
        RNNProcessor.add_arguments(parser, nn_files=nn_files)

    # aliases for other argument parsers
    add_tempo_arguments = TempoEstimationProcessor.add_arguments
    add_detect_arguments = BeatTrackingProcessor.add_arguments
    add_dbn_arguments = DBNBeatTrackingProcessor.add_arguments
    add_crf_arguments = CRFBeatDetectionProcessor.add_arguments


# TODO: should we inherit from RNNBeatProcessor?
class MultiModelRNNBeatProcessor(IOProcessor):
    """
    Multi-model beat tracking with RNNs.

    """
    NN_FILES = RNNBeatProcessor.NN_FILES
    NN_REF_FILES = None

    def __init__(self, nn_files=NN_FILES, nn_ref_files=NN_REF_FILES, **kwargs):
        """
        Use multiple RNNs to compute beat activation functions and then choose
        the most appropriate one automatically by comparing them to a reference
        model.

        :param nn_files:    list of files that define the RNN
        :param ref_nn_file: list of files that define the reference NN model

        :param args:        additional arguments passed to DBNBeatTracking()
        :param kwargs:      additional arguments passed to DBNBeatTracking()

        "A multi-model approach to beat tracking considering heterogeneous
         music styles"
        Sebastian Böck, Florian Krebs and Gerhard Widmer
        Proceedings of the 15th International Society for Music Information
        Retrieval Conference (ISMIR), 2014

        """
        self.nn_files = nn_files
        self.nn_ref_files = nn_ref_files
        # signal handling processor
        sig = SignalProcessor(mono=True, **kwargs)
        # parallel specs + stacking processor
        # TODO: this information should be stored in the nn_files
        stack = StackSpectrogramProcessor(frame_sizes=[1024, 2048, 4096],
                                          online=False, bands=3,
                                          norm_filters=True, mul=1, add=1,
                                          diff_ratio=0.5, **kwargs)
        # multiple RNN processor (without averaging the predictions)
        if nn_ref_files is not None:
            nn_files += nn_ref_files
        rnn = RNNProcessor(nn_files=nn_files, average=False, **kwargs)
        # sequentially process everything
        seq = [sig, stack, rnn, self.multi_model_selector]
        super(MultiModelRNNBeatProcessor, self).__init__(seq)

    def multi_model_selector(self, predictions):
        """
        Selects the most appropriate predictions form the list of predictions.

        :param predictions: list with predictions (beat activation functions)
        :return:            most suitable prediction

        Note: the reference beat activation function must be given first

        """
        # get the reference predictions

        if self.nn_ref_files is not None:
            num_ref_files = len(self.nn_ref_files)
        else:
            num_ref_files = 0
        # determine the reference prediction
        if num_ref_files > 1:
            # average the reference predictions
            reference_prediction = (sum(predictions[:num_ref_files]) /
                                    num_ref_files)
        elif num_ref_files == 1:
            # use the only given reference prediction
            reference_prediction = predictions[0]
        else:
            # just average all predictions to simulate a reference network
            reference_prediction = sum(predictions) / len(self.nn_files)
        # init the error with the max. possible value (i.e. prediction length)
        best_error = len(reference_prediction)
        # init the best_prediction with an empty array
        best_prediction = np.empty(0)
        # compare the (remaining) predictions with the reference prediction
        for prediction in predictions[num_ref_files:]:
            # calculate the squared error w.r.t. the reference prediction
            error = np.sum((prediction - reference_prediction) ** 2.)
            # chose the best activation
            if error < best_error:
                best_prediction = prediction
                best_error = error
        # return the best prediction
        return best_prediction.ravel()

    @classmethod
    def add_arguments(cls, parser, nn_files=NN_FILES,
                      nn_ref_files=NN_REF_FILES, **kwargs):
        """
        Add MMBeatTracking related arguments to an existing parser object.

        :param parser:       existing argparse parser object
        :param nn_files:     list of files that define the RNN
        :param nn_ref_files: list with files of reference NN model(s)
        :param kwargs:       additional arguments passed to
                             DBNBeatTracking.add_dbn_arguments()
        :return:             Multi-model DBN beat tracking parser group object

        """
        # add signal processing arguments
        SignalProcessor.add_arguments(parser, norm=False, att=0)
        # add rnn processing arguments
        g = RNNProcessor.add_arguments(parser, nn_files=nn_files)
        # add option for the reference files
        g.add_argument('--nn_ref_files', action='append', type=str,
                       default=nn_ref_files,
                       help='Compare the predictions to these pre-trained '
                            'neural networks (multiple files can be given, '
                            'one file per argument) and choose the most '
                            'suitable one accordingly (i.e. the one with the '
                            'least deviation form the reference model). '
                            'If multiple reference files are given, the '
                            'predictions of the networks are averaged first.')
        # return the argument group so it can be modified if needed
        return g
