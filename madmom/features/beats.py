#!/usr/bin/env python
# encoding: utf-8
"""
This file contains all beat tracking related functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import sys
import glob
import numpy as np

from madmom import MODELS_PATH, Processor, IOProcessor, SequentialProcessor
from madmom.audio.signal import (SignalProcessor, FramedSignalProcessor,
                                 smooth as smooth_signal)
from madmom.audio.spectrogram import (SpectrogramProcessor,
                                      StackSpectrogramProcessor,
                                      MultiBandSpectrogramProcessor)
from madmom.ml.rnn import RNNProcessor, average_predictions
from madmom.utils import write_events
from madmom.features import ActivationsProcessor


# classes for obtaining beat activation functions from (multiple) RNNs
class MultiModelSelector(Processor):
    """
    Class for selecting the most suitable model (i.e. the predictions thereof)
    from a multiple models (i.e. the predictions thereof).

    """

    def __init__(self, num_ref_predictions, **kwargs):
        """
        Use multiple RNNs to compute beat activation functions and then choose
        the most appropriate one automatically by comparing them to a reference
        model.

        :param num_ref_predictions: number of reference predictions

        "A multi-model approach to beat tracking considering heterogeneous
         music styles"
        Sebastian Böck, Florian Krebs and Gerhard Widmer
        Proceedings of the 15th International Society for Music Information
        Retrieval Conference (ISMIR), 2014

        """
        self.num_ref_predictions = num_ref_predictions

    def process(self, predictions):
        """
        Selects the most appropriate predictions form the list of predictions.

        :param predictions: list with predictions (beat activation functions)
        :return:            most suitable prediction

        Note: the reference beat activation function must be the first ones in
              the list of given predictions

        """
        # TODO: right now we only have 1D predictions, what to do with
        #       multi-dim?
        num_refs = self.num_ref_predictions
        # determine the reference prediction
        if num_refs in (None, 0):
            # just average all predictions to simulate a reference network
            reference = average_predictions(predictions)
        elif num_refs > 0:
            # average the reference predictions
            reference = average_predictions(predictions[:num_refs])
        else:
            raise ValueError('`num_ref_predictions` must be positive or None, '
                             '%s given' % num_refs)
        # init the error with the max. possible value (i.e. prediction length)
        best_error = len(reference)
        # init the best_prediction with an empty array
        best_prediction = np.empty(0)
        # compare the (remaining) predictions with the reference prediction
        for prediction in predictions[num_refs:]:
            # calculate the squared error w.r.t. the reference prediction
            error = np.sum((prediction - reference) ** 2.)
            # chose the best activation
            if error < best_error:
                best_prediction = prediction
                best_error = error
        # return the best prediction
        return best_prediction.ravel()


class RNNBeatProcessing(SequentialProcessor):
    """
    Class for tracking beats with a recurrent neural network (RNN).

    """
    NN_FILES = glob.glob("%s/beats_blstm_[1-8].npz" % MODELS_PATH)
    NN_REF_FILES = None

    def __init__(self, nn_files=NN_FILES, nn_ref_files=NN_REF_FILES, **kwargs):
        """
        Use (multiple) RNNs to predict a beat activation function.

        :param nn_files:    list of RNN model files
        :param ref_nn_file: list of files that define the reference NN model

        "Enhanced Beat Tracking with Context-Aware Neural Networks"
        Sebastian Böck and Markus Schedl
        Proceedings of the 14th International Conference on Digital Audio
        Effects (DAFx), 2011

        If `nn_ref_files` are set, the most appropriate model is chosen
        according to the method described in:

        "A multi-model approach to beat tracking considering heterogeneous
         music styles"
        Sebastian Böck, Florian Krebs and Gerhard Widmer
        Proceedings of the 15th International Society for Music Information
        Retrieval Conference (ISMIR), 2014

        If `nn_ref_files` are the same as `ref_files`, the averaged predictions
        of the `ref_files` are used as a reference.


        """
        # FIXME: remove this hack of setting fps here
        #        all information should be stored in the nn_files or in a
        #        pickled Processor (including information about spectrograms,
        #        mul, add & diff_ratio and so on)
        kwargs['fps'] = self.fps = 100
        # define processing chain
        sig = SignalProcessor(num_channels=1, sample_rate=44100, **kwargs)
        stack = StackSpectrogramProcessor(frame_sizes=[1024, 2048, 4096],
                                          online=False, bands=3,
                                          norm_filters=True, log=True, mul=1,
                                          add=1, diff_ratio=0.5, **kwargs)
        if nn_ref_files is not None:
            if nn_ref_files == nn_files:
                # if we don't have nn_ref_files given or they are the same as
                # the nn_files, set num_ref_predictions to 0
                num_ref_predictions = 0
            else:
                # set the number of reference files according to the length
                num_ref_predictions = len(nn_ref_files)
                # redefine the list of files to be tested
                nn_files = nn_ref_files + nn_files
            # define the selector
            selector = MultiModelSelector(num_ref_predictions)
        else:
            # use simple averaging
            selector = average_predictions
        rnn = RNNProcessor(nn_files=nn_files, **kwargs)
        # sequentially process everything
        super(RNNBeatProcessing, self).__init__([sig, stack, rnn, selector])

    @classmethod
    def add_arguments(cls, parser, nn_files=NN_FILES,
                      nn_ref_files=NN_REF_FILES):
        """
        Add RNN beat tracking related arguments to an existing parser.

        :param parser:       existing argparse parser
        :param nn_files:     list of files that define the RNN
        :param nn_ref_files: list with files of reference NN model(s)
        :return:             RNN beat tracking parser group

        """
        # add signal processing arguments
        SignalProcessor.add_arguments(parser, norm=False, att=0)
        # add rnn processing arguments
        g = RNNProcessor.add_arguments(parser, nn_files=nn_files)
        # add option for the reference files
        if nn_ref_files is not None:
            g.add_argument('--nn_ref_files', action='append', type=str,
                           default=nn_ref_files,
                           help='Compare the predictions to these pre-trained '
                                'neural networks (multiple files can be'
                                'given, one file per argument) and choose the '
                                'most suitable one accordingly (i.e. the one '
                                'with the least deviation form the reference '
                                'model). If multiple reference files are'
                                'given, the predictions of the networks are '
                                'averaged first.')
        # return the argument group so it can be modified if needed
        return g


# function for detecting the beats based on the given dominant interval
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


# classes for detecting/tracking of beat inside a beat activation function
class BeatTracking(Processor):
    """
    Class for tracking beats with a simple tempo estimation and beat aligning.

    """
    LOOK_ASIDE = 0.2
    LOOK_AHEAD = 10
    # tempo defaults
    TEMPO_METHOD = 'comb'
    MIN_BPM = 40
    MAX_BPM = 240
    ACT_SMOOTH = 0.09
    HIST_SMOOTH = 7
    ALPHA = 0.79

    def __init__(self, look_aside=LOOK_ASIDE, look_ahead=LOOK_AHEAD, fps=None,
                 **kwargs):
        """
        Track the beats according to the previously determined (local) tempo
        by simply aligning them around the estimated position.

        :param look_aside: look this fraction of a beat interval to each side
                           of the assumed next beat position to look for the
                           most likely position of the next beat
        :param look_ahead: look N seconds in both directions to determine the
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
        # import the TempoEstimation here otherwise we have a loop
        from madmom.features.tempo import TempoEstimation
        # save variables
        self.look_aside = look_aside
        self.look_ahead = look_ahead
        self.fps = fps
        # tempo estimator
        self.tempo_estimator = TempoEstimation(fps=fps, **kwargs)

    def process(self, activations):
        """
        Detect the beats in the given activation function.

        :param activations: beat activation function
        :return:            detected beat positions

        """
        # smooth activations
        act_smooth = int(self.fps * self.tempo_estimator.act_smooth)
        activations = smooth_signal(activations, act_smooth)
        # TODO: refactor interval stuff to use TempoEstimation
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

        :param parser:     existing argparse parser
        :param look_aside: look this fraction of a beat interval to each side
                           of the assumed next beat position to look for the
                           most likely position of the next beat
        :param look_ahead: look N seconds in both directions to determine the
                           local tempo and align the beats accordingly
        :return:           beat argument parser group

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

    @classmethod
    def add_tempo_arguments(cls, parser, method=TEMPO_METHOD, min_bpm=MIN_BPM,
                            max_bpm=MAX_BPM, act_smooth=ACT_SMOOTH,
                            hist_smooth=HIST_SMOOTH, alpha=ALPHA):
        """
        Add tempo arguments to an existing parser.

        :param parser:      existing argparse parser
        :param method:      tempo estimation method ['comb', 'acf']
        :param min_bpm:     minimum tempo [bpm]
        :param max_bpm:     maximum tempo [bpm]
        :param act_smooth:  smooth the activations over N seconds
        :param hist_smooth: smooth the tempo histogram over N bins
        :param alpha:       scaling factor of the comb filter
        :return:            tempo argument parser group

        """
        # TODO: import the TempoEstimation here otherwise we have a
        #       loop. This is super ugly, but right now I can't think of a
        #       better solution...
        from madmom.features.tempo import TempoEstimation as tempo
        return tempo.add_arguments(parser, method=method, min_bpm=min_bpm,
                                   max_bpm=max_bpm, act_smooth=act_smooth,
                                   hist_smooth=hist_smooth, alpha=alpha)


class BeatDetection(BeatTracking):
    """
    Class for detecting beats with a simple tempo estimation and beat aligning.

    """
    LOOK_ASIDE = 0.2

    def __init__(self, look_aside=LOOK_ASIDE, fps=None, **kwargs):
        """
        Detect the beats according to the previously determined global tempo
        by simply aligning them around the estimated position.

        :param look_aside: look this fraction of a beat interval to each side
                           of the assumed next beat position to look for the
                           most likely position of the next beat

        "Enhanced Beat Tracking with Context-Aware Neural Networks"
        Sebastian Böck and Markus Schedl
        Proceedings of the 14th International Conference on Digital Audio
        Effects (DAFx), 2011

        Instead of the auto-correlation based method for tempo estimation, it
        uses a comb filter per default. The behaviour can be controlled with
        the `tempo_method` parameter.

        """
        super(BeatDetection, self).__init__(look_aside=look_aside,
                                            look_ahead=None, fps=fps, **kwargs)


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
    return CRFBeatDetection.best_sequence(*process_tuple)


class CRFBeatDetection(BeatTracking):
    """
    Conditional Random Field Beat Detection.

    """
    INTERVAL_SIGMA = 0.18
    FACTORS = [0.5, 0.67, 1.0, 1.5, 2.0]
    # tempo defaults
    MIN_BPM = 20
    MAX_BPM = 240
    ACT_SMOOTH = 0.09
    HIST_SMOOTH = 7

    try:
        from .viterbi import crf_viterbi
    except ImportError:
        import warnings
        warnings.warn('CRFBeatDetection only works if you build the viterbi '
                      'module with cython!')

    def __init__(self, interval_sigma=INTERVAL_SIGMA, factors=FACTORS, **kwargs):
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
        super(CRFBeatDetection, self).__init__(**kwargs)
        # save variables
        self.interval_sigma = interval_sigma
        self.factors = factors
        # get num_threads from kwargs
        num_threads = min(len(self.factors), kwargs.get('num_threads', 1))
        # TODO: implement comb filter stuff and remove this...
        self.tempo_estimator.method = 'acf'
        # init a pool of workers (if needed)
        self.map = map
        if num_threads != 1:
            import multiprocessing as mp
            self.map = mp.Pool(num_threads).map

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

    def process(self, activations):
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
        # TODO: refactor interval stuff to use TempoEstimation
        # create a interval histogram
        histogram = self.tempo_estimator.interval_histogram(activations)
        # get the dominant interval
        interval = self.tempo_estimator.dominant_interval(histogram)

        # TODO: use the tempi returned by the TempoEstimation instead
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

        # since the cython code uses memory views, we need to make sure that
        # the activations are C-contiguous and of C-type float (np.float32)
        contiguous_act = np.ascontiguousarray(activations, dtype=np.float32)
        results = self.map(_process_crf,
                           it.izip(it.repeat(contiguous_act),
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

    @classmethod
    def add_tempo_arguments(cls, parser, min_bpm=MIN_BPM, max_bpm=MAX_BPM,
                            act_smooth=ACT_SMOOTH, hist_smooth=HIST_SMOOTH):
        """
        Add tempo related arguments to an existing parser.

        :param parser:      existing argparse parser
        :param min_bpm:     minimum tempo [bpm]
        :param max_bpm:     maximum tempo [bpm]
        :param act_smooth:  smooth the activations over N seconds
        :param hist_smooth: smooth the tempo histogram over N bins
        :return:            tempo argument parser group

        """
        # TODO: import the TempoEstimation here otherwise we have a
        #       loop. This is super ugly, but right now I can't think of a
        #       better solution...
        from madmom.features.tempo import TempoEstimation as tempo
        tempo.add_arguments(parser, method=None, min_bpm=min_bpm,
                            max_bpm=max_bpm, act_smooth=act_smooth,
                            hist_smooth=hist_smooth, alpha=None)


class DBNBeatTracking(Processor):
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
        from .dbn import (DynamicBayesianNetwork as DBN,
                          BeatTrackingTransitionModel as TM,
                          BeatTrackingObservationModel as OM)
    except ImportError:
        import warnings
        warnings.warn('MMBeatTracking only works if you build the dbn '
                      'module with cython!')

    def __init__(self, correct=CORRECT, num_beat_states=NUM_BEAT_STATES,
                 num_tempo_states=NUM_TEMPO_STATES,
                 tempo_change_probability=TEMPO_CHANGE_PROBABILITY,
                 min_bpm=MIN_BPM, max_bpm=MAX_BPM,
                 observation_lambda=OBSERVATION_LAMBDA,
                 norm_observations=NORM_OBSERVATIONS, fps=None,
                 num_threads=None, **kwargs):
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
        self.fps = fps
        # convert timing information to tempo space
        max_tempo = max_bpm * num_beat_states / (60. * fps)
        min_tempo = min_bpm * num_beat_states / (60. * fps)
        if num_tempo_states is None:
            # do not limit the number of tempo states, use a linear spacing
            tempo_states = np.arange(np.round(min_tempo),
                                     np.round(max_tempo) + 1)
        else:
            # limit the number of tempo states, thus use a quasi log spacing
            tempo_states = np.logspace(np.log2(min_tempo),
                                       np.log2(max_tempo),
                                       num_tempo_states, base=2)
        # transition model
        self.tm = self.TM(num_beat_states, tempo_states, tempo_change_probability)
        # observation model
        self.om = self.OM(self.tm, observation_lambda, norm_observations)
        # instantiate a DBN
        self.dbn = self.DBN(self.tm, self.om, None, num_threads)
        # save variables
        self.correct = correct

    def process(self, activations):
        """
        Detect the beats in the given activation function.

        :param activations: beat activation function
        :return:            detected beat positions

        """
        # first compute the observation model's log_densities
        self.om.compute_densities(activations)
        # then get the best state path by calling the viterbi algorithm
        path, _ = self.dbn.viterbi()
        # correct the beat positions if needed
        if self.correct:
            beats = []
            # for each detection determine the "beat range", i.e. states where
            # the pointers of the observation model are 0
            beat_range = self.om.pointers[path]
            # get all change points between True and False
            idx = np.nonzero(np.diff(beat_range))[0] + 1
            # if the first frame is in the beat range, add a change at frame 0
            if not beat_range[0]:
                idx = np.r_[0, idx]
            # if the last frame is in the beat range, append the length of the
            # array
            if not beat_range[-1]:
                idx = np.r_[idx, beat_range.size]
            # iterate over all regions
            for left, right in idx.reshape((-1, 2)):
                # pick the frame with the highest observation_model value
                beats.append(np.argmax(activations[left:right]) + left)
            beats = np.asarray(beats)
        else:
            # just take the frames with the smallest beat state values
            from scipy.signal import argrelmin
            beats = argrelmin(self.tm.position(path),
                              mode='wrap')[0]
            # recheck if they are within the "beat range", i.e. the pointers
            # of the observation model for that state must be 0
            # Note: interpolation and alignment of the beats to be at state 0
            #       does not improve results over this simple method
            beats = beats[self.om.pointers[path[beats]] == 0]
        # convert the detected beats to seconds
        return beats / float(self.fps)

    @classmethod
    def add_arguments(cls, parser, num_beat_states=NUM_BEAT_STATES,
                      num_tempo_states=NUM_TEMPO_STATES, min_bpm=MIN_BPM,
                      max_bpm=MAX_BPM,
                      tempo_change_probability=TEMPO_CHANGE_PROBABILITY,
                      observation_lambda=OBSERVATION_LAMBDA,
                      norm_observations=NORM_OBSERVATIONS, correct=CORRECT):
        """
        Add DBN related arguments to an existing parser.

        :param parser:                   existing argparse parser

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

        :param observation_lambda:       split one beat period into N parts,
                                         the first representing beat states and
                                         the remaining non-beat states
        :param norm_observations:        normalize the observations

        Post-processing parameters:

        :param correct:                  correct the beat positions

        :return:                         beat argument parser group

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
                            'observation_model [default=%(default).4f]')
        # observation model stuff
        g.add_argument('--observation_lambda', action='store', type=int,
                       default=observation_lambda,
                       help='split one beat period into N parts, the first '
                            'representing beat states and the remaining '
                            'non-beat states [default=%(default)i]')
        if norm_observations:
            g.add_argument('--no_norm_obs', dest='norm_observations',
                           action='store_false', default=norm_observations,
                           help='do not normalize the observation_model of the DBN')
        else:
            g.add_argument('--norm_obs', dest='norm_observations',
                           action='store_true', default=norm_observations,
                           help='normalize the observation_model of the DBN')
        # return the argument group so it can be modified if needed
        return g


# TODO: split the classes similar to madmom.features.onsets?
# meta class for tracking beats with RNNs and any post-processing method
class RNNBeatTracking(IOProcessor):
    """
    Class for detecting/tracking beats with recurrent neural networks (RNN)
    and different post-processing methods.

    """
    NN_FILES = RNNBeatProcessing.NN_FILES

    def __init__(self, beat_method='DBNBeatTracking', multi_model=False,
                 nn_files=NN_FILES, load=False, save=False, **kwargs):
        """
        Detecting/tracking beats with multiple recurrent neural networks (RNN)
        and different post-processing methods.

        :param beat_method: method for tracking the beats
        :param multi_model: use a multi-model approach to select the most
                            suitable RNN model
        :param nn_files:    list of NN model files
        :param load:        load the NN beat activations from file
        :param save:        save the NN beat activations to file

        """
        # set the reference model files
        nn_ref_files = nn_files if multi_model else None
        # TODO: remove this fps hack!
        kwargs['fps'] = 100
        # set input processor
        if load:
            in_processor = ActivationsProcessor(mode='r', **kwargs)
        else:
            in_processor = RNNBeatProcessing(nn_files, nn_ref_files, **kwargs)
        # set output processor
        if save:
            out_processor = ActivationsProcessor(mode='w', **kwargs)
        else:
            out_processor = [globals()[beat_method](**kwargs), write_events]
        # make this an IOProcessor by defining input and output processors
        super(RNNBeatTracking, self).__init__(in_processor, out_processor)

    # add aliases to argument parsers
    add_activation_arguments = ActivationsProcessor.add_arguments
    add_rnn_arguments = RNNBeatProcessing.add_arguments


# downbeat tracking stuff
class DownbeatTracking(Processor):
    """
    Class for tracking beats and downbeats with a Hidden Markov Model (HMM)
    according to

    """
    MIN_BPM = [55, 60]
    MAX_BPM = [205, 225]
    NUM_BAR_STATES = [1200, 1600]
    NUM_BEATS = [3, 4]
    TEMPO_CHANGE_PROBABILITY = [0.02, 0.02]
    NORM_OBSERVATIONS = False
    GMM_FILE = glob.glob("%s/downbeat_ismir2013.pkl" % MODELS_PATH)[0]

    try:
        from .dbn import (DynamicBayesianNetwork as DBN,
                          DownBeatTrackingTransitionModel as TM,
                          GMMDownBeatTrackingObservationModel as OM)
    except ImportError:
        import warnings
        warnings.warn('Downbeat tracking only works if you build the dbn '
                      'module with cython!')

    def __init__(self, gmm_file=GMM_FILE, num_bar_states=NUM_BAR_STATES,
                 num_beats=NUM_BEATS, min_bpm=MIN_BPM, max_bpm=MAX_BPM,
                 tempo_change_probability=TEMPO_CHANGE_PROBABILITY,
                 norm_observations=NORM_OBSERVATIONS,
                 downbeats=False, fps=None, **kwargs):
        """

        :param gmm_file:                 load the fitted GMMs from this file

        Parameters for the transition model:

        :param num_bar_states:           number of states for one bar period
        :param num_beats:                number of beats in one bar period
        :param min_bpm:                  minimum tempo used for beat tracking
        :param max_bpm:                  maximum tempo used for beat tracking
        :param tempo_change_probability: probability of a tempo change between
                                         two adjacent observations

        Parameters for the observation model:

        :param norm_observations:        normalise the observations

        Other parameters:

        :param downbeats:                report only the downbeats (default:
                                         beats and the respective position)

        "Rhythmic Pattern Modeling for Beat and Downbeat Tracking in Musical
         Audio"
        Florian Krebs, Sebastian Böck and Gerhard Widmer
        Proceedings of the 15th International Society for Music Information
        Retrieval Conference (ISMIR), 2013

        """
        # check if all lists have the same length
        if not (len(num_bar_states) == len(num_beats) == len(min_bpm) ==
                    len(max_bpm) == len(tempo_change_probability)):
            raise ValueError("'num_bar_states', 'num_beats', 'min_bpm', "
                             "'max_bpm' and 'tempo_change_probability' must "
                             "have the same length")
        self.fps = fps
        self.num_beats = num_beats
        self.downbeats = downbeats
        import cPickle
        with open(gmm_file, 'r') as f:
            # load the fitted GMMs
            gmms = cPickle.load(f)
        # convert timing information to tempo space for each pattern
        tempo_states = []
        for pattern in range(len(num_bar_states)):
            max_tempo = int(np.ceil(max_bpm[pattern] * num_bar_states[pattern]
                                    / (60. * num_beats[pattern] * self.fps)))
            min_tempo = int(np.floor(min_bpm[pattern] * num_bar_states[pattern]
                                     / (60. * num_beats[pattern] * self.fps)))
            tempo_states.append(np.arange(min_tempo, max_tempo))
        # transition model
        self.tm = self.TM(num_bar_states, tempo_states,
                          tempo_change_probability)
        # observation model
        self.om = self.OM(gmms, self.tm, norm_observations)
        # instantiate a DBN
        self.dbn = self.DBN(self.tm, self.om, None, None)

    def process(self, activations):
        """
        Detect the beats in the given activation function.

        :param activations: beat activation function
        :return:            detected beat positions

        """
        # first compute the observation model's log_densities
        self.om.compute_densities(activations)
        # then get the best state path by calling the viterbi algorithm
        path, _ = self.dbn.viterbi()
        # get the corresponding pattern (use only the first state, since it
        # doesn't change throughout the sequence)
        pattern = self.tm.pattern(path[0])
        # the position inside the pattern
        position = self.tm.position(path)
        # beat position (= weighted by number of beats in bar)
        beat_counter = (position * self.num_beats[pattern]).astype(int)
        # transitions are the points where the beat counters change
        beat_positions = np.nonzero(np.diff(beat_counter))[0] + 1
        # the beat numbers are the counters + 1 at the transition points
        beat_numbers = beat_counter[beat_positions] + 1
        # convert the detected beats to a list of timestamps
        beats = np.asarray(beat_positions) / float(self.fps)
        # return the downbeats or beats and their beat number
        if self.downbeats:
            return beats[beat_numbers == 1]
        else:
            return zip(beats, beat_numbers)

    @classmethod
    def add_arguments(cls, parser, num_bar_states=NUM_BAR_STATES,
                      min_bpm=MIN_BPM, max_bpm=MAX_BPM,
                      tempo_change_probability=TEMPO_CHANGE_PROBABILITY,
                      norm_observations=NORM_OBSERVATIONS):
        """
        Add DBN related arguments to an existing parser.

        :param parser:                   existing argparse parser

        Parameters for the transition model:

        Each of the following arguments expect a list with as many values as
        rhythmic patterns.

        :param num_bar_states:           number of states for one bar period
        :param min_bpm:                  minimum tempo used for beat tracking
        :param max_bpm:                  maximum tempo used for beat tracking
        :param tempo_change_probability: probability of a tempo change between
                                         two adjacent observations

        Parameters for the observation model:

        :param norm_observations:        normalize the observations

        :return:                         downbeat argument parser group

        """
        # add DBN parser group
        g = parser.add_argument_group('dynamic Bayesian Network arguments')
        from madmom.utils import OverrideDefaultListAction
        # TODO: make parsing nicer!
        g.add_argument('--num_bar_states', action=OverrideDefaultListAction,
                       type=str, default=num_bar_states,
                       help='number of states for one bar period (one value '
                            'must be given for each pattern) [default=%s]' %
                            num_bar_states)
        g.add_argument('--min_bpm', action=OverrideDefaultListAction,
                       type=float, default=min_bpm,
                       help='minimum tempo (one value must be given for each '
                            'pattern) [bpm, default=%s]' % min_bpm)
        g.add_argument('--max_bpm', action=OverrideDefaultListAction,
                       type=float, default=max_bpm,
                       help='maximum tempo (one value must be given for each '
                            'pattern) [bpm, default=%s]' % max_bpm)
        g.add_argument('--tempo_change_probability',
                       action=OverrideDefaultListAction, type=float,
                       default=tempo_change_probability,
                       help='probability of a tempo between two adjacent '
                            'observations (one value must be given for each '
                            'pattern) [default=%s]' % tempo_change_probability)
        # observation model stuff
        if norm_observations:
            g.add_argument('--no_norm_obs', dest='norm_observations',
                           action='store_false', default=norm_observations,
                           help='do not normalize the observations of the DBN')
        else:
            g.add_argument('--norm_obs', dest='norm_observations',
                           action='store_true', default=norm_observations,
                           help='normalize the observations of the DBN')
        # add output format stuff
        g = parser.add_argument_group('output arguments')
        g.add_argument('--downbeats', action='store_true', default=False,
                       help='output only the downbeats')
        # return the argument group so it can be modified if needed
        return g


# TODO: split the classes similar to madmom.features.beats?
class SpectralBeatTracking(IOProcessor):
    """
    The SpectralBeatTracking class implements (down-)beat tracking based on the
    magnitude spectrogram.

    """

    def __init__(self, downbeats=False, load=False, save=False, **kwargs):
        """
        Creates a new SpectralBeatTracking instance.

        """
        from madmom.features.notes import write_notes as write_beats
        # define input and output processors
        sig = SignalProcessor(mono=True, **kwargs)
        frames = FramedSignalProcessor(**kwargs)
        spec = MultiBandSpectrogramProcessor(diff=True, **kwargs)
        in_processor = [sig, frames, spec]
        if downbeats:
            write_beats = write_events
        out_processor = [DownbeatTracking(downbeats=downbeats, **kwargs),
                         write_beats]
        # swap in/out processors if needed
        if load:
            in_processor = ActivationsProcessor(mode='r', **kwargs)
        if save:
            out_processor = ActivationsProcessor(mode='w', **kwargs)
        # make this an IOProcessor by defining input and output processors
        super(SpectralBeatTracking, self).__init__(in_processor, out_processor)

    @classmethod
    def add_arguments(cls, parser, beat_method=None):
        """
        Add spectral beat tracking arguments to an existing parser.

        :param parser:      existing argparse parser
        :param beat_method: default ODF method
        :return:            spectral onset detection argument parser group

        """
        # add onset detection method arguments to the existing parser
        g = parser.add_argument_group('spectral beat tracking arguments')
        if beat_method is not None:
            g.add_argument('-o', '--odf', dest='beat_method',
                           default=beat_method, choices=cls.METHODS,
                           help='use this onset detection function '
                                '[default=%(default)s]')
        # return the argument group so it can be modified if needed
        return g

    # add aliases to other argument parsers
    add_activations_arguments = ActivationsProcessor.add_arguments
    add_signal_arguments = SignalProcessor.add_arguments
    add_framing_arguments = FramedSignalProcessor.add_arguments
    add_filter_arguments = SpectrogramProcessor.add_filter_arguments
    add_log_arguments = SpectrogramProcessor.add_log_arguments
    add_diff_arguments = SpectrogramProcessor.add_diff_arguments
    add_multi_band_arguments = \
        MultiBandSpectrogramProcessor.add_multi_band_arguments
