#!/usr/bin/env python
# encoding: utf-8
"""
This file contains all beat tracking related functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""
import glob
import sys
import numpy as np

from madmom import MODELS_PATH
from . import Activations, RNNEventDetection
from madmom.audio.signal import smooth as smooth_signal


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
    Effects (DAFx-11), 2011

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
    # return indices
    return np.array(positions)


class RNNBeatTracking(RNNEventDetection):
    """
    Class for tracking beats with a recurrent neural network (RNN).

    """
    # define the NN files
    NN_FILES = glob.glob("%s/beats_blstm*npz" % MODELS_PATH)
    # default values for beat detection
    # TODO: refactor this to use TempoEstimation functionality
    ACT_SMOOTH = 0.09
    LOOK_ASIDE = 0.2
    LOOK_AHEAD = 10
    MIN_BPM = 40
    MAX_BPM = 240
    ALPHA = 0.79
    HIST_SMOOTH = 7

    def __init__(self, signal, nn_files=NN_FILES, *args, **kwargs):
        """
        Use RNNs to compute the beat activation function and then align the
        beats according to the previously determined tempo.

        :param signal:   Signal instance or file name or file handle
        :param nn_files: list of files that define the RNN

        :param args:     additional arguments passed to RNNEventDetection()
        :param kwargs:   additional arguments passed to RNNEventDetection()

        "Enhanced Beat Tracking with Context-Aware Neural Networks"
        Sebastian Böck and Markus Schedl
        Proceedings of the 14th International Conference on Digital Audio
        Effects (DAFx-11), 2011

        """
        super(RNNBeatTracking, self).__init__(signal, nn_files, *args,
                                              **kwargs)

    def pre_process(self):
        """
        Pre-process the signal to obtain a data representation suitable for RNN
        processing.

        :return: pre-processed data

        """
        spr = super(RNNBeatTracking, self)
        spr.pre_process(frame_sizes=[1024, 2048, 4096], bands_per_octave=3,
                        mul=1, ratio=0.5)
        # return data
        return self._data

    def detect(self, min_bpm=MIN_BPM, max_bpm=MAX_BPM, act_smooth=ACT_SMOOTH,
               hist_smooth=HIST_SMOOTH, alpha=ALPHA, look_aside=LOOK_ASIDE,
               look_ahead=LOOK_AHEAD):
        """
        Track the beats by first estimating the (local) tempo and then align
        the beats accordingly.

        :param min_bpm:     minimum tempo used for beat tracking
        :param max_bpm:     maximum tempo used for beat tracking
        :param act_smooth:  smooth the beat activation function over N seconds
        :param hist_smooth: smooth the tempo histogram over N bins
        :param alpha:       scaling factor of the comb filter
        :param look_aside:  look this fraction of a beat interval to each side
                            of the assumed next beat position to look for the
                            most likely position of the next beat
        :param look_ahead:  look N seconds in both directions to determine the
                            local tempo and align the beats accordingly
        :return:            detected beat positions

        Note: If `look_ahead` is undefined, a constant tempo throughout the
              whole piece is assumed.
              If `look_ahead` is set, the local tempo (in a range +/-
              look_ahead seconds around the actual position) is estimated and
              then the next beat is tracked accordingly. This procedure is
              repeated from the new position to the end of the piece.

        """
        from .tempo import dominant_interval, interval_histogram_comb
        # convert timing information to frames and set default values
        min_tau = int(np.floor(60. * self.fps / max_bpm))
        max_tau = int(np.ceil(60. * self.fps / min_bpm))

        # smooth activations
        act_smooth = int(self.fps * act_smooth)
        activations = smooth_signal(self.activations, act_smooth)

        # TODO: refactor interval stuff to use TempoEstimation functionality
        # if look_ahead is not defined, assume a global tempo
        if look_ahead is None:
            # create a interval histogram
            histogram = interval_histogram_comb(activations, alpha, min_tau,
                                                max_tau)
            # get the dominant interval
            interval = dominant_interval(histogram, smooth=hist_smooth)
            # detect beats based on this interval
            detections = detect_beats(activations, interval, look_aside)
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
                    act = np.append(np.zeros(-start), activations[0:end])
                elif end > len(activations):
                    # append zeros accordingly
                    zeros = np.zeros(end - len(activations))
                    act = np.append(activations[start:], zeros)
                else:
                    act = activations[start:end]
                # create a interval histogram
                histogram = interval_histogram_comb(act, alpha, min_tau,
                                                    max_tau)
                # get the dominant interval
                interval = dominant_interval(histogram, smooth=hist_smooth)
                # add the offset (i.e. the new detected start position)
                positions = detect_beats(act, interval, look_aside)
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
    def add_arguments(cls, parser, nn_files=NN_FILES, min_bpm=MIN_BPM,
                      max_bpm=MAX_BPM, act_smooth=ACT_SMOOTH,
                      hist_smooth=HIST_SMOOTH, alpha=ALPHA,
                      look_aside=LOOK_ASIDE, look_ahead=LOOK_AHEAD):
        """
        Add beat tracking related arguments to an existing parser object.

        :param parser:      existing argparse parser object
        :param nn_files:    list with files of NN models
        :param min_bpm:     minimum tempo used for beat tracking
        :param max_bpm:     maximum tempo used for beat tracking
        :param act_smooth:  smooth the beat activations over N seconds
        :param hist_smooth: smooth the tempo histogram over N bins
        :param alpha:       scaling factor of the comb filter
        :param look_aside:  look this fraction of a beat interval to each side
                            of the assumed next beat position to look for the
                            most likely position of the next beat
        :param look_ahead:  look N seconds in both directions to determine the
                            local tempo and align the beats accordingly
        :return:            beat argument parser group object

        """
        # add Activations parser
        Activations.add_arguments(parser)
        # add arguments from RNNEventDetection
        RNNEventDetection.add_arguments(parser, nn_files=nn_files)
        # add beat detection related options to the existing parser
        g = parser.add_argument_group('beat detection arguments')
        # TODO: refactor this stuff to use the TempoEstimation functionality
        g.add_argument('--min_bpm', action='store', type=float,
                       default=min_bpm, help='minimum tempo [bpm, '
                       ' default=%(default).2f]')
        g.add_argument('--max_bpm', action='store', type=float,
                       default=max_bpm, help='maximum tempo [bpm, '
                       ' default=%(default).2f]')
        g.add_argument('--act_smooth', action='store', type=float,
                       default=act_smooth,
                       help='smooth the beat activations over N seconds '
                            '[default=%(default).2f]')
        # make switchable (useful for including the beat stuff for tempo)
        if hist_smooth is not None:
            g.add_argument('--hist_smooth', action='store', type=int,
                           default=hist_smooth,
                           help='smooth the tempo histogram over N bins '
                                '[default=%(default)d]')
        if alpha is not None:
            g.add_argument('--alpha', action='store', type=float,
                           default=alpha,
                           help='alpha for comb filter tempo estimation '
                                '[default=%(default).2f]')
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
#       initial_distribution and all other functionality, but omit the factors
def _process_crf(data):
    """
    Extract the best beat sequence for a piece.

    :param data: tuple with (activations, dominant_interval, allowed
                             deviation from the dominant interval per beat)
    :return:     tuple with extracted beat positions [frames]
                 and log probability of beat sequence

    """
    activations, dominant_interval, interval_sigma = data
    return CRFBeatDetection.best_sequence(activations, dominant_interval,
                                          interval_sigma)


class CRFBeatDetection(RNNBeatTracking):
    """
    Conditional Random Field Beat Detection.

    """
    MIN_BPM = 20
    MAX_BPM = 240
    ACT_SMOOTH = 0.09
    INTERVAL_SIGMA = 0.18
    FACTORS = [0.5, 0.67, 1.0, 1.5, 2.0]

    try:
        from .viterbi import crf_viterbi
    except ImportError:
        import warnings
        warnings.warn('CRFBeatDetection only works if you build the viterbi '
                      'module with cython!')

    def __init__(self, signal, nn_files=RNNBeatTracking.NN_FILES, *args,
                 **kwargs):
        """
        Use RNNs to compute the beat activation function and then align the
        beats according to the previously determined global tempo using
        a conditional random field model.

        :param signal:   Signal instance or file name or file handle
        :param nn_files: list of files that define the RNN

        :param args:     additional arguments passed to RNNBeatTracking()
        :param kwargs:   additional arguments passed to RNNBeatTracking()

        "Probabilistic extraction of beat positions from a beat activation
         function"
        Filip Korzeniowski, Sebastian Böck and Gerhard Widmer
        In Proceedings of the 15th International Society for Music Information
        Retrieval Conference (ISMIR), 2014.

        """
        super(CRFBeatDetection, self).__init__(signal, nn_files, *args,
                                               **kwargs)

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

    def detect(self, act_smooth=ACT_SMOOTH, min_bpm=MIN_BPM, max_bpm=MAX_BPM,
               interval_sigma=INTERVAL_SIGMA, factors=FACTORS):
        """
        Detect the beats with a conditional random field method based on
        neural network activations and a tempo estimation using
        auto-correlation.

        :param act_smooth:     smooth the beat activations over N seconds
        :param min_bpm:        minimum tempo used for beat tracking
        :param max_bpm:        maximum tempo used for beat tracking
        :param interval_sigma: allowed deviation from the dominant interval per
                               beat
        :param factors:        factors of the dominant interval to try
        :return:               detected beat positions

        """
        import itertools as it
        from .tempo import interval_histogram_acf, dominant_interval
        # convert timing information to frames and set default values
        min_interval = int(np.floor(60. * self.fps / max_bpm))
        max_interval = int(np.ceil(60. * self.fps / min_bpm))

        # smooth activations
        act_smooth = int(self.fps * act_smooth)
        activations = smooth_signal(self.activations, act_smooth)

        # create a interval histogram
        # TODO: refactor this to use the new TempoEstimation functionality
        #       directly or the functionality inherited from RNNBeatTracking
        hist = interval_histogram_acf(activations, min_interval, max_interval)
        # get the dominant interval
        interval = dominant_interval(hist, smooth=None)
        # create variations of the dominant interval to check
        possible_intervals = [int(interval * f) for f in factors]
        # remove all intervals outside the allowed range
        possible_intervals = [i for i in possible_intervals
                              if max_interval >= i >= min_interval]
        # sort the intervals
        possible_intervals.sort()
        # put the greatest first so that it get processed first
        possible_intervals.reverse()

        # init a pool of workers (if needed)
        map_ = map
        if min(len(factors), max(1, self.num_threads)) != 1:
            import multiprocessing as mp
            map_ = mp.Pool(self.num_threads).map

        # compute the beat sequences (in parallel)
        # since the cython code uses memory views, we need to make sure that
        # the activations are c-contiguous
        c_contiguous_act = np.ascontiguousarray(self.activations)
        results = map_(_process_crf, it.izip(it.repeat(c_contiguous_act),
                                             possible_intervals,
                                             it.repeat(interval_sigma)))

        # normalize their probabilities
        normalized_seq_probabilities = np.array([r[1] / r[0].shape[0]
                                                 for r in results])
        # pick the best one
        best_seq = results[normalized_seq_probabilities.argmax()][0]
        # save the detected beats
        self._detections = best_seq.astype(np.float) / self.fps
        # and return them
        return self._detections

    @classmethod
    def add_arguments(cls, parser, nn_files=RNNBeatTracking.NN_FILES,
                      interval_sigma=INTERVAL_SIGMA, act_smooth=ACT_SMOOTH,
                      min_bpm=MIN_BPM, max_bpm=MAX_BPM, factors=FACTORS):
        """
        Add CRFBeatDetection related arguments to an existing parser object.

        :param parser:         existing argparse parser object
        :param nn_files:       list with files of NN models
        :param interval_sigma: allowed deviation from the dominant interval per
                               beat
        :param act_smooth:     smooth the beat activations over N seconds
        :param min_bpm:        minimum tempo used for beat tracking
        :param max_bpm:        maximum tempo used for beat tracking
        :param factors:        factors of the dominant interval to try
        :return:               beat argument parser group object
        """
        # add RNNBeatTracking arguments
        g = RNNBeatTracking.add_arguments(parser, nn_files=nn_files,
                                          min_bpm=min_bpm, max_bpm=max_bpm,
                                          act_smooth=act_smooth,
                                          hist_smooth=None, alpha=None,
                                          look_ahead=None, look_aside=None)
        # add CRF related arguments
        g.add_argument('--interval_sigma', action='store', type=float,
                       default=interval_sigma,
                       help='allowed deviation from the dominant interval '
                            '[default=%(default).2f]')
        from madmom.utils import OverrideDefaultListAction
        g.add_argument('--factor', '-f', action=OverrideDefaultListAction,
                       type=float, default=factors, dest='factors',
                       help='factors of dominant interval to try. '
                            'multiple factors can be given, one factor per '
                            'argument. [default=%(default)s]')
        return g


class DBNBeatTracking(RNNBeatTracking):
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

    def __init__(self, data, nn_files=RNNBeatTracking.NN_FILES, *args,
                 **kwargs):
        """
        Use multiple RNNs to compute beat activation functions and then choose
        the most appropriate one automatically by comparing them to a reference
        model and finally infer the beats with a dynamic Bayesian network.

        :param signal:      Signal instance or file name or file handle
        :param nn_files:    list of files that define the RNN

        :param args:        additional arguments passed to RNNBeatTracking()
        :param kwargs:      additional arguments passed to RNNBeatTracking()

        "A multi-model approach to beat tracking considering heterogeneous
         music styles"
        Sebastian Böck, Florian Krebs and Gerhard Widmer
        Proceedings of the 15th International Society for Music Information
        Retrieval Conference (ISMIR), 2014

        It does not use the multi-model (Section 2.2.) and selection stage
        (Section 2.3), i.e. this version corresponds to the pure DBN version
        of the algorithm for which results are given in Table 2.

        """
        super(DBNBeatTracking, self).__init__(data, nn_files, *args, **kwargs)

    def detect(self, correct=CORRECT, num_beat_states=NUM_BEAT_STATES,
               num_tempo_states=NUM_TEMPO_STATES,
               tempo_change_probability=TEMPO_CHANGE_PROBABILITY,
               min_bpm=MIN_BPM, max_bpm=MAX_BPM,
               observation_lambda=OBSERVATION_LAMBDA,
               norm_observations=NORM_OBSERVATIONS):
        """
        Track the beats with a dynamic Bayesian network.

        :param correct:                  correct the beats

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

        :return:                         detected beat positions

        """
        # convert timing information to tempo space
        max_tempo = max_bpm * num_beat_states / (60. * self.fps)
        min_tempo = min_bpm * num_beat_states / (60. * self.fps)
        if num_tempo_states is None:
            # do not limit the number of tempo states, use a linear spacing
            tempo_states = np.arange(np.round(min_tempo),
                                     np.round(max_tempo) + 1)
        else:
            # limit the number of tempo states, thus use a quasi log spacing
            tempo_states = np.logspace(np.log2(min_tempo),
                                       np.log2(max_tempo),
                                       num_tempo_states, base=2)
        # quantize to integer tempo states
        tempo_states = np.unique(np.round(tempo_states).astype(np.int))
        # transition model
        tm = self.TM(num_beat_states=num_beat_states,
                     tempo_states=tempo_states,
                     tempo_change_probability=tempo_change_probability)
        # observation model
        om = self.OM(self.activations,
                     num_states=tm.num_states,
                     num_beat_states=tm.num_beat_states,
                     observation_lambda=observation_lambda,
                     norm_observations=norm_observations)
        # init the DBN
        dbn = self.DBN(transition_model=tm, observation_model=om,
                       num_threads=self.num_threads, correct=correct)
        # convert the detected beats to a list of timestamps
        self._detections = dbn.beats / float(self.fps)
        # also return the detections
        return self._detections

    @classmethod
    def add_dbn_arguments(cls, parser, num_beat_states=NUM_BEAT_STATES,
                          num_tempo_states=NUM_TEMPO_STATES,
                          min_bpm=MIN_BPM, max_bpm=MAX_BPM,
                          tempo_change_probability=TEMPO_CHANGE_PROBABILITY,
                          observation_lambda=OBSERVATION_LAMBDA,
                          norm_observations=NORM_OBSERVATIONS,
                          correct=CORRECT):
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

    @classmethod
    def add_arguments(cls, parser, nn_files=RNNBeatTracking.NN_FILES):
        """
        Add DBNBeatTracking related arguments to an existing parser object.

        :param parser:   existing argparse parser object
        :param nn_files: list with files of NN models

        :return:         DBN beat tracking parser group object

        """
        # add Activations parser
        Activations.add_arguments(parser)
        # add arguments from RNNEventDetection
        RNNEventDetection.add_arguments(parser, nn_files=nn_files)
        # add DBN parser stuff
        g = cls.add_dbn_arguments(parser)
        # return the argument group so it can be modified if needed
        return g


class MMBeatTracking(DBNBeatTracking):
    """
    Multi-model beat tracking with RNNs and a DBN.

    """
    # define the reference model files
    NN_REF_FILES = glob.glob("%s/beats_ref_blstm*npz" % MODELS_PATH)

    def __init__(self, data, nn_files=DBNBeatTracking.NN_FILES,
                 nn_ref_files=NN_REF_FILES, *args, **kwargs):
        """
        Use multiple RNNs to compute beat activation functions and then choose
        the most appropriate one automatically by comparing them to a reference
        model and finally infer the beats with a dynamic Bayesian network.

        :param signal:      Signal instance or file name or file handle
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
        super(MMBeatTracking, self).__init__(data, nn_files, *args, **kwargs)
        self.nn_ref_files = nn_ref_files

    def process(self):
        """
        Computes the predictions on the data with the RNN models defined/given
        and save the predictions of the most suitable model as activations.

        :return: most suitable RNN activation function (prediction)

        """
        from madmom.ml.rnn import process_rnn
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
        # save the best prediction as activations
        self._activations = Activations(best_prediction.ravel(), self.fps)
        # and return them
        return self._activations

    @classmethod
    def add_arguments(cls, parser, nn_files=RNNBeatTracking.NN_FILES,
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
        # add Activations parser
        Activations.add_arguments(parser)
        # add arguments from RNNEventDetection
        g = RNNEventDetection.add_arguments(parser, nn_files=nn_files)
        g.add_argument('--nn_ref_files', action='append', type=str,
                       default=nn_ref_files,
                       help='Compare the predictions to these pre-trained '
                            'neural networks (multiple files can be given, '
                            'one file per argument) and choose the most '
                            'suitable one accordingly (i.e. the one with the '
                            'least deviation form the reference model). '
                            'If multiple reference files are given, the '
                            'predictions of the networks are averaged first.')
        # add DBN stuff
        g = DBNBeatTracking.add_dbn_arguments(parser, **kwargs)
        # return the argument group so it can be modified if needed
        return g
