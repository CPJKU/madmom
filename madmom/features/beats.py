#!/usr/bin/env python
# encoding: utf-8
"""
This file contains all beat tracking related functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import sys
import glob

import numpy as np

from madmom import MODELS_PATH
from madmom.processors import Processor, SequentialProcessor
from madmom.audio.signal import SignalProcessor, smooth as smooth_signal
from madmom.audio.spectrogram import (LogarithmicFilteredSpectrogramProcessor,
                                      StackedSpectrogramProcessor)
from madmom.ml.rnn import RNNProcessor, average_predictions


# classes for obtaining beat activation functions from (multiple) RNNs
class MultiModelSelectionProcessor(Processor):
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


class RNNBeatProcessor(SequentialProcessor):
    """
    Class for predicting beats with a recurrent neural network (RNN).

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

        If `nn_ref_files` is 'True' or identical to `ref_files`, the averaged
        predictions of the `ref_files` are used as the reference the individual
        predictions are compared to.

        """
        # FIXME: remove this hack of setting fps and the other stuff here!
        #        all information should be stored in the nn_files or in a
        #        pickled Processor (including information about spectrograms,
        #        mul, add & diff_ratio and so on)
        kwargs['fps'] = self.fps = 100
        # processing chain
        sig = SignalProcessor(num_channels=1, sample_rate=44100, **kwargs)
        # we need to define which specs should be stacked
        spec = LogarithmicFilteredSpectrogramProcessor(num_bands=3,
                                                       norm_filters=True,
                                                       mul=1, add=1)
        # stack specs with the given frame sizes
        stack = StackedSpectrogramProcessor(frame_size=[1024, 2048, 4096],
                                            spectrogram=spec, stack_diffs=True,
                                            diff_ratio=0.5,
                                            positive_diffs=True, **kwargs)
        if nn_ref_files is not None:
            if nn_ref_files is True:
                # FIXME: this is kind of hackish, but being able to simply set
                #        the nn_ref_files to 'True' makes the arguments stuff
                #        for the MMBeatTracker much simpler
                nn_ref_files = nn_files
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
            selector = MultiModelSelectionProcessor(num_ref_predictions)
        else:
            # use simple averaging
            selector = average_predictions
        rnn = RNNProcessor(nn_files=nn_files, **kwargs)
        # sequentially process everything
        super(RNNBeatProcessor, self).__init__([sig, stack, rnn, selector])

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
        # add RNN processing arguments
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
    # always look at least 1 frame to each side
    frames_look_aside = max(1, int(interval * look_aside))
    win = np.hamming(2 * frames_look_aside)

    # list to be filled with beat positions from inside the recursive function
    positions = []

    def recursive(position):
        """
        Recursively detect the next beat.

        :param position: start at this position

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
        # go to the next beat, until end is reached
        if position + interval < len(activations):
            recursive(position + interval)
        else:
            return

    # calculate the beats for each start position (up to the interval length)
    sums = np.zeros(interval)
    for i in range(interval):
        positions = []
        # detect the beats for this start position
        recursive(i)
        # calculate the sum of the activations at the beat positions
        sums[i] = np.sum(activations[positions])
    # take the winning start position
    start_position = np.argmax(sums)
    # and calc the beats for this start position
    positions = []
    recursive(start_position)
    # return indices
    return np.array(positions)


# classes for detecting/tracking of beat inside a beat activation function
class BeatTrackingProcessor(Processor):
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
        from madmom.features.tempo import TempoEstimationProcessor
        # save variables
        self.look_aside = look_aside
        self.look_ahead = look_ahead
        self.fps = fps
        # tempo estimator
        self.tempo_estimator = TempoEstimationProcessor(fps=fps, **kwargs)

    def process(self, activations):
        """
        Detect the beats in the given activation function.

        :param activations: beat activation function
        :return:            detected beat positions [seconds]

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
        from madmom.features.tempo import TempoEstimationProcessor as Tempo
        return Tempo.add_arguments(parser, method=method, min_bpm=min_bpm,
                                   max_bpm=max_bpm, act_smooth=act_smooth,
                                   hist_smooth=hist_smooth, alpha=alpha)


class BeatDetectionProcessor(BeatTrackingProcessor):
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
        super(BeatDetectionProcessor, self).__init__(look_aside=look_aside,
                                                     look_ahead=None, fps=fps,
                                                     **kwargs)


def _process_crf(process_tuple):
    """
    Extract the best beat sequence for a piece.

    This proxy function is necessary if we want to process different intervals
    in parallel using the multiprocessing module.

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
    USE_FACTORS = False
    FACTORS = [0.5, 0.67, 1.0, 1.5, 2.0]
    NUM_INTERVALS = 5
    # tempo defaults
    MIN_BPM = 20
    MAX_BPM = 240
    ACT_SMOOTH = 0.09
    HIST_SMOOTH = 7

    try:
        from .beats_crf import best_sequence
    except ImportError:
        import warnings
        warnings.warn('CRFBeatDetection only works if you build the viterbi '
                      'module with cython!')
        # else, use dummy function
        best_sequence = lambda x: np.array([]), -np.inf

    def __init__(self, interval_sigma=INTERVAL_SIGMA, use_factors=USE_FACTORS,
                 num_intervals=NUM_INTERVALS, factors=FACTORS, **kwargs):
        """
        Track the beats according to the previously determined global tempo
        using a conditional random field model.

        :param interval_sigma: allowed deviation from the dominant beat
                               interval per beat [float]
        :param use_factors:    use dominant interval multiplied by factors
                               instead of intervals estimated by
                               tempo estimator.
        :param num_intervals:  max number of estimated intervals to try. [int]
        :param factors:        factors of the dominant interval to try
                               [list of floats]

        This method is based on the following work with some improvements:

        "Probabilistic extraction of beat positions from a beat activation
         function"
        Filip Korzeniowski, Sebastian Böck and Gerhard Widmer
        In Proceedings of the 15th International Society for Music Information
        Retrieval Conference (ISMIR), 2014.

        """
        super(CRFBeatDetectionProcessor, self).__init__(**kwargs)
        # save variables
        self.interval_sigma = interval_sigma
        self.use_factors = use_factors
        self.num_intervals = num_intervals
        self.factors = factors

        # get num_threads from kwargs
        num_threads = min(len(factors) if use_factors else num_intervals,
                          kwargs.get('num_threads', 1))
        # init a pool of workers (if needed)
        self.map = map
        if num_threads != 1:
            import multiprocessing as mp
            self.map = mp.Pool(num_threads).map

    def process(self, activations):
        """
        Detect the beats in the given activation function.

        :param activations: beat activation function
        :return:            detected beat positions [seconds]

        """
        import itertools as it
        # estimate the tempo
        tempi = self.tempo_estimator.process(activations)
        intervals = self.fps * 60. / tempi[:, 0]

        # compute possible intervals
        if self.use_factors:
            # use the dominant interval with different factors
            possible_intervals = [int(intervals[0] * f) for f in self.factors]
            possible_intervals = [i for i in possible_intervals if
                                  self.tempo_estimator.max_interval >= i >=
                                  self.tempo_estimator.min_interval]
        else:
            # take the top n intervals from the tempo estimator
            possible_intervals = intervals[:self.num_intervals]

        # sort and start from the greatest interval
        possible_intervals.sort()
        possible_intervals = [int(i) for i in possible_intervals[::-1]]

        # smooth activations
        act_smooth = int(self.fps * self.tempo_estimator.act_smooth)
        activations = smooth_signal(activations, act_smooth)

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
                      use_factors=USE_FACTORS, num_intervals=NUM_INTERVALS,
                      factors=FACTORS):
        """
        Add CRFBeatDetection related arguments to an existing parser.

        :param parser:         existing argparse parser
        :param interval_sigma: allowed deviation from the dominant interval per
                               beat
        :param use_factors:    use dominant interval multiplied by factors
                               instead of intervals estimated by
                               tempo estimator.
        :param num_intervals:  max number of estimated intervals to try. [int]
        :param factors:        factors of the dominant interval to try
                               [list of floats]
        :return:               beat argument parser group

        """
        from madmom.utils import OverrideDefaultListAction
        # add CRF related arguments
        g = parser.add_argument_group('conditional random field arguments')
        g.add_argument('--interval_sigma', action='store', type=float,
                       default=interval_sigma,
                       help='allowed deviation from the dominant interval '
                            '[default=%(default).2f]')

        g.add_argument('--use_factors', action='store_true',
                       default=use_factors,
                       help='use dominant interval multiplied with factors '
                            'instead of multiple estimated intervals '
                            '[default=%(default)s]')

        g.add_argument('--num_intervals', action='store', type=int,
                       default=num_intervals, dest='num_intervals',
                       help='number of estimated intervals to try '
                            '[default=%(default)s]')
        g.add_argument('-f', '--factors', action=OverrideDefaultListAction,
                       default=factors, type=float, sep=',',
                       help='(comma separated) list with factors of dominant '
                            'interval to try [default=%(default)s]')
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
        from madmom.features.tempo import TempoEstimationProcessor as tempo
        tempo.add_arguments(parser, method=None, min_bpm=min_bpm,
                            max_bpm=max_bpm, act_smooth=act_smooth,
                            hist_smooth=hist_smooth, alpha=None)


# function for converting min & max tempo ranges to beat states
def beat_states(min_bpm, max_bpm, fps, num_tempo_states=None):
    """
    Convert the timing information to beat states usable for transition models
    of a Hidden Markov Model.

    :param min_bpm:          minimum tempo to model one cycle [float]
    :param max_bpm:          maximum tempo to model one cycle [float]
    :param fps:              frame rate (frames per second) [float]
    :param num_tempo_states: number of tempo states [int] (if set, limit the
                             number of states and use a log spacing, otherwise
                             a linear spacing)
    :return:                 numpy array with beat states

    """
    # convert timing information to beat space
    min_interval = 60. * fps / max_bpm
    max_interval = 60. * fps / min_bpm
    if num_tempo_states is None:
        # do not limit the number of tempo states, use a linear spacing
        states = np.arange(np.round(min_interval), np.round(max_interval) + 1)
    else:
        # limit the number of tempo states, thus use a log spacing
        states = np.logspace(np.log2(min_interval), np.log2(max_interval),
                             num_tempo_states, base=2)
    # quantize to integer tempo states
    return np.unique(np.round(states).astype(np.int))


# class for beat tracking
class DBNBeatTrackingProcessor(Processor):
    """
    Beat tracking with RNNs and a dynamic Bayesian network (DBN).

    """
    CORRECT = True
    NUM_TEMPO_STATES = None
    TRANSITION_LAMBDA = 100
    OBSERVATION_LAMBDA = 16
    NORM_OBSERVATIONS = False
    MIN_BPM = 55
    MAX_BPM = 215

    def __init__(self, correct=CORRECT, min_bpm=MIN_BPM, max_bpm=MAX_BPM,
                 num_tempo_states=NUM_TEMPO_STATES,
                 transition_lambda=TRANSITION_LAMBDA,
                 observation_lambda=OBSERVATION_LAMBDA,
                 norm_observations=NORM_OBSERVATIONS, fps=None, **kwargs):
        """
        Track the beats with a dynamic Bayesian network (DBN) approximated
        by a Hidden Markov Model (HMM).

        :param correct:            correct the beats (i.e. align them
                                   to the nearest peak of the beat
                                   activation function)

        Parameters for the transition model:

        :param min_bpm:            minimum tempo used for beat tracking
        :param max_bpm:            maximum tempo used for beat tracking
        :param num_tempo_states:   number of tempo states (if set, limit the
                                   number of states and use a log spacing,
                                   otherwise a linear spacing)
        :param transition_lambda:  lambda for the exponential tempo change
                                   distribution (higher values prefer a
                                   constant tempo over a tempo change from
                                   one beat to the next one)

        Parameters for the observation model:

        :param observation_lambda: split one beat period into N parts, the
                                   first representing beat states and the
                                   remaining non-beat states
        :param norm_observations:  normalize the observations

        "A multi-model approach to beat tracking considering heterogeneous
         music styles"
        Sebastian Böck, Florian Krebs and Gerhard Widmer
        Proceedings of the 15th International Society for Music Information
        Retrieval Conference (ISMIR), 2014

        Instead of the originally proposed transition model for the DBN, the
        following is used:

        "An efficient state space model for joint tempo and meter tracking"
        Florian Krebs, Sebastian Böck and Gerhard Widmer
        Proceedings of the 16th International Society for Music Information
        Retrieval Conference (ISMIR), 2015.

        """

        from madmom.ml.hmm import HiddenMarkovModel as Hmm
        from .beats_hmm import (BeatTrackingTransitionModel as Tm,
                                BeatTrackingObservationModel as Om)

        # convert timing information to beat space
        beat_space = beat_states(min_bpm, max_bpm, fps, num_tempo_states)
        # transition model
        self.tm = Tm(beat_space, transition_lambda)
        # observation model
        self.om = Om(self.tm, observation_lambda, norm_observations)
        # instantiate a HMM
        self.hmm = Hmm(self.tm, self.om, None)
        # save variables
        self.fps = fps
        self.correct = correct

    def process(self, activations):
        """
        Detect the beats in the given activation function.

        :param activations: beat activation function
        :return:            detected beat positions [seconds]

        """
        # get the best state path by calling the viterbi algorithm
        path, _ = self.hmm.viterbi(activations)
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
                # pick the frame with the highest activations value
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
    def add_arguments(cls, parser, min_bpm=MIN_BPM, max_bpm=MAX_BPM,
                      num_tempo_states=NUM_TEMPO_STATES,
                      transition_lambda=TRANSITION_LAMBDA,
                      observation_lambda=OBSERVATION_LAMBDA,
                      norm_observations=NORM_OBSERVATIONS, correct=CORRECT):
        """
        Add HMM related arguments to an existing parser object.

        :param parser: existing argparse parser object

        Parameters for the transition model:

        :param min_bpm:            minimum tempo used for beat tracking
        :param max_bpm:            maximum tempo used for beat tracking
        :param num_tempo_states:   number of tempo states (if set, limit the
                                   number of states and use a log spacing,
                                   otherwise a linear spacing)
        :param transition_lambda:  lambda for the exponential tempo change
                                   distribution (higher values prefer a
                                   constant tempo over a tempo change from
                                   one beat to the next one)

        Parameters for the observation model:

        :param observation_lambda: split one beat period into N parts, the
                                   first representing beat states and the
                                   remaining non-beat states
        :param norm_observations:  normalize the observations

        Post-processing parameters:

        :param correct:            correct the beat positions

        :return:                   beat argument parser group

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
        g.add_argument('--min_bpm', action='store', type=float,
                       default=min_bpm,
                       help='minimum tempo [bpm, default=%(default).2f]')
        g.add_argument('--max_bpm', action='store', type=float,
                       default=max_bpm,
                       help='maximum tempo [bpm,  default=%(default).2f]')
        g.add_argument('--num_tempo_states', action='store', type=int,
                       default=num_tempo_states,
                       help='limit the number of tempo states; if set, align '
                            'them with a log spacing, otherwise linearly')
        g.add_argument('--transition_lambda', action='store',
                       type=float, default=transition_lambda,
                       help='lambda of the tempo transition distribution; '
                            'higher values prefer a constant tempo over a '
                            'tempo change from one beat to the next one '
                            '[default=%(default).1f]')
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


# class for beat tracking
class DownbeatTrackingProcessor(Processor):
    """
    Beat and downbeat tracking with a dynamic Bayesian network (DBN).

    """
    MIN_BPM = [55, 60]
    MAX_BPM = [205, 225]
    NUM_TEMPO_STATES = [None, None]
    TRANSITION_LAMBDA = [100, 100]
    NUM_BEATS = [3, 4]
    NORM_OBSERVATIONS = False
    GMM_FILE = glob.glob("%s/downbeat_ismir2013.pkl" % MODELS_PATH)[0]

    def __init__(self, gmm_file=GMM_FILE, min_bpm=MIN_BPM, max_bpm=MAX_BPM,
                 num_tempo_states=NUM_TEMPO_STATES,
                 transition_lambda=TRANSITION_LAMBDA, num_beats=NUM_BEATS,
                 norm_observations=NORM_OBSERVATIONS, downbeats=False,
                 fps=None, **kwargs):
        """

        Track the beats and downbeats with a Dynamic Bayesian Network (DBN)
        approximated by a Hidden Markov Model (HMM).

        :param gmm_file:          load the fitted GMMs from this file

        Parameters for the transition model:

        Each of the following arguments expect a list with as many items as
        rhythmic patterns.

        :param min_bpm:           list with minimum tempi used for tracking
        :param max_bpm:           list with maximum tempi used for tracking
        :param num_tempo_states:  list with number of tempo states (if set,
                                  limit the number of states and use a log
                                  spacing, otherwise a linear spacing). If a
                                  single value is given, the same value is
                                  assumed for all patterns.
        :param transition_lambda: (list with) lambda(s) for the exponential
                                  tempo change distribution (higher values
                                  prefer a constant tempo over a tempo change
                                  from one beat to the next one). If a single
                                  value is given, the same value is assumed
                                  for all patterns.
        :param num_beats:         list with number of beats per bar

        Parameters for the observation model:

        :param norm_observations: normalise the observations

        Other parameters:

        :param downbeats:         report only the downbeats (default: beats
                                  and the respective position)

        "Rhythmic Pattern Modeling for Beat and Downbeat Tracking in Musical
         Audio"
        Florian Krebs, Sebastian Böck and Gerhard Widmer
        Proceedings of the 15th International Society for Music Information
        Retrieval Conference (ISMIR), 2013

        Instead of the originally proposed transition model for the DBN, the
        following is used:

        "An efficient state space model for joint tempo and meter tracking"
        Florian Krebs, Sebastian Böck and Gerhard Widmer
        Proceedings of the 16th International Society for Music Information
        Retrieval Conference (ISMIR), 2015.

        """

        from madmom.ml.hmm import HiddenMarkovModel as Hmm
        from .beats_hmm import (DownBeatTrackingTransitionModel as Tm,
                                GMMDownBeatTrackingObservationModel as Om)

        # expand num_tempo_states and transition_lambda to lists if needed
        if not isinstance(num_tempo_states, list):
            num_tempo_states = [num_tempo_states] * len(num_tempo_states)
        if not isinstance(transition_lambda, list):
            transition_lambda = [transition_lambda] * len(beat_states)
        # check if all lists have the same length
        if not (len(min_bpm) == len(max_bpm) == len(num_tempo_states) ==
                len(transition_lambda) == len(num_beats)):
            raise ValueError("'min_bpm', 'max_bpm', 'num_tempo_states', "
                             "'transition_lambda' and 'num_beats' must have "
                             "the same length")
        self.fps = fps
        self.num_beats = num_beats
        self.downbeats = downbeats
        import cPickle
        with open(gmm_file, 'r') as f:
            # load the fitted GMMs
            gmms = cPickle.load(f)
        # convert timing information to tempo space for each pattern
        beat_space = []
        for pattern in range(len(num_tempo_states)):
            # convert timing information to beat space
            # Note: we multiply the fps with the number of beats in this
            #       pattern, since a complete cycle is N times that long
            beat_space.append(beat_states(min_bpm[pattern], max_bpm[pattern],
                                          fps * num_beats[pattern],
                                          num_tempo_states[pattern]))
        # transition model
        self.tm = Tm(beat_space, transition_lambda)
        # observation model
        self.om = Om(gmms, self.tm, norm_observations)
        # instantiate a HMM
        self.hmm = Hmm(self.tm, self.om, None)

    def process(self, activations):
        """
        Detect the beats in the given activation function.

        :param activations: beat activation function
        :return:            detected beat positions [seconds]

        """
        # get the best state path by calling the viterbi algorithm
        path, _ = self.hmm.viterbi(activations)
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
    def add_arguments(cls, parser, min_bpm=MIN_BPM, max_bpm=MAX_BPM,
                      num_tempo_states=NUM_TEMPO_STATES,
                      transition_lambda=TRANSITION_LAMBDA,
                      num_beats=NUM_BEATS,
                      norm_observations=NORM_OBSERVATIONS):
        """
        Add HMM related arguments to an existing parser.

        :param parser:            existing argparse parser

        Parameters for the transition model:

        Each of the following arguments expect a list with as many items as
        rhythmic patterns.

        :param min_bpm:           list with minimum tempi used for tracking
        :param max_bpm:           list with maximum tempi used for tracking
        :param num_tempo_states:  list with number of tempo states (if set,
                                  limit the number of states and use a log
                                  spacing, otherwise a linear spacing)
        :param transition_lambda: list with lambdas for the exponential tempo
                                  change distribution (higher values prefer a
                                  constant tempo over a tempo change from one
                                  bar to the next one)
        :param num_beats:         list with number of beats per bar

        Parameters for the observation model:

        :param norm_observations: normalize the observations

        :return:                  downbeat argument parser group

        """
        from madmom.utils import OverrideDefaultListAction
        # add HMM parser group
        g = parser.add_argument_group('dynamic Bayesian Network arguments')
        g.add_argument('--min_bpm', action=OverrideDefaultListAction,
                       default=min_bpm, type=float, sep=',',
                       help='minimum tempo (comma separated list with one '
                            'value per pattern) [bpm, default=%(default)s]')
        g.add_argument('--max_bpm', action=OverrideDefaultListAction,
                       default=max_bpm, type=float, sep=',',
                       help='maximum tempo (comma separated list with one '
                            'value per pattern) [bpm, default=%(default)s]')
        g.add_argument('--num_tempo_states', action=OverrideDefaultListAction,
                       default=num_tempo_states, type=int, sep=',',
                       help='limit the number of tempo states; if set, align '
                            'them with a log spacing, otherwise linearly '
                            '(comma separated list with one value per pattern)'
                            ' [default=%(default)s]')
        g.add_argument('--transition_lambda', action=OverrideDefaultListAction,
                       default=transition_lambda, type=float, sep=',',
                       help='lambda of the tempo transition distribution; '
                            'higher values prefer a constant tempo over a '
                            'tempo change from one bar to the next one (comma '
                            'separated list with one value per pattern) '
                            '[default=%(default)s]')
        if num_beats is not None:
            g.add_argument('--num_beats', action=OverrideDefaultListAction,
                           default=num_beats, type=int, sep=',',
                           help='number of beats per par (comma separated '
                                'list with one value per pattern) '
                                '[default=%(default)s]')
        # observation model stuff
        if norm_observations:
            g.add_argument('--no_norm_obs', dest='norm_observations',
                           action='store_false', default=norm_observations,
                           help='do not normalize the observations of the HMM')
        else:
            g.add_argument('--norm_obs', dest='norm_observations',
                           action='store_true', default=norm_observations,
                           help='normalize the observations of the HMM')
        # add output format stuff
        g = parser.add_argument_group('output arguments')
        g.add_argument('--downbeats', action='store_true', default=False,
                       help='output only the downbeats')
        # return the argument group so it can be modified if needed
        return g
