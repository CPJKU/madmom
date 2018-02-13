# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains beat tracking related functionality.

"""

from __future__ import absolute_import, division, print_function

import sys

import numpy as np

from ..audio.signal import signal_frame, smooth as smooth_signal
from ..ml.nn import average_predictions
from ..processors import (OnlineProcessor, ParallelProcessor, Processor,
                          SequentialProcessor)


# classes for tracking (down-)beats with RNNs
class RNNBeatProcessor(SequentialProcessor):
    """
    Processor to get a beat activation function from multiple RNNs.

    Parameters
    ----------
    post_processor : Processor, optional
        Post-processor, default is to average the predictions.
    online : bool, optional
        Use signal processing parameters and RNN models suitable for online
        mode.
    nn_files : list, optional
        List with trained RNN model files. Per default ('None'), an ensemble
        of networks will be used.

    References
    ----------
    .. [1] Sebastian Böck and Markus Schedl,
           "Enhanced Beat Tracking with Context-Aware Neural Networks",
           Proceedings of the 14th International Conference on Digital Audio
           Effects (DAFx), 2011.

    Examples
    --------
    Create a RNNBeatProcessor and pass a file through the processor.
    The returned 1d array represents the probability of a beat at each frame,
    sampled at 100 frames per second.

    >>> proc = RNNBeatProcessor()
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.beats.RNNBeatProcessor object at 0x...>
    >>> proc('tests/data/audio/sample.wav')  # doctest: +ELLIPSIS
    array([0.00479, 0.00603, 0.00927, 0.01419, ... 0.02725], dtype=float32)

    For online processing, `online` must be set to 'True'. If processing power
    is limited, fewer number of RNN models can be defined via `nn_files`. The
    audio signal is then processed frame by frame.

    >>> from madmom.models import BEATS_LSTM
    >>> proc = RNNBeatProcessor(online=True, nn_files=[BEATS_LSTM[0]])
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.beats.RNNBeatProcessor object at 0x...>
    >>> proc('tests/data/audio/sample.wav')  # doctest: +ELLIPSIS
    array([0.03887, 0.02619, 0.00747, 0.00218, ... 0.04825], dtype=float32)

    """

    def __init__(self, post_processor=average_predictions, online=False,
                 nn_files=None, **kwargs):
        # pylint: disable=unused-argument
        from ..audio.signal import SignalProcessor, FramedSignalProcessor
        from ..audio.stft import ShortTimeFourierTransformProcessor
        from ..audio.spectrogram import (
            FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
            SpectrogramDifferenceProcessor)
        from ..ml.nn import NeuralNetworkEnsemble
        from ..models import BEATS_LSTM, BEATS_BLSTM
        # choose the appropriate models and set frame sizes accordingly
        if online:
            if nn_files is None:
                nn_files = BEATS_LSTM
            frame_sizes = [2048]
            num_bands = 12
        else:
            if nn_files is None:
                nn_files = BEATS_BLSTM
            frame_sizes = [1024, 2048, 4096]
            num_bands = 6
        # define pre-processing chain
        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        # process the multi-resolution spec & diff in parallel
        multi = ParallelProcessor([])
        for frame_size in frame_sizes:
            frames = FramedSignalProcessor(frame_size=frame_size, **kwargs)
            stft = ShortTimeFourierTransformProcessor()  # caching FFT window
            filt = FilteredSpectrogramProcessor(num_bands=num_bands, fmin=30,
                                                fmax=17000, norm_filters=True)
            spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
            diff = SpectrogramDifferenceProcessor(
                diff_ratio=0.5, positive_diffs=True, stack_diffs=np.hstack)
            # process each frame size with spec and diff sequentially
            multi.append(SequentialProcessor((frames, stft, filt, spec, diff)))
        # stack the features and processes everything sequentially
        pre_processor = SequentialProcessor((sig, multi, np.hstack))
        # process the pre-processed signal with a NN ensemble and the given
        # post_processor
        nn = NeuralNetworkEnsemble.load(nn_files,
                                        ensemble_fn=post_processor, **kwargs)
        # instantiate a SequentialProcessor
        super(RNNBeatProcessor, self).__init__((pre_processor, nn))


# class for selecting a certain beat activation functions from (multiple) NNs
class MultiModelSelectionProcessor(Processor):
    """
    Processor for selecting the most suitable model (i.e. the predictions
    thereof) from a multiple models/predictions.

    Parameters
    ----------
    num_ref_predictions : int
        Number of reference predictions (see below).

    Notes
    -----
    This processor selects the most suitable prediction from multiple models by
    comparing them to the predictions of a reference model. The one with the
    smallest mean squared error is chosen.

    If `num_ref_predictions` is 0 or None, an averaged prediction is computed
    from the given predictions and used as reference.

    References
    ----------
    .. [1] Sebastian Böck, Florian Krebs and Gerhard Widmer,
           "A Multi-Model Approach to Beat Tracking Considering Heterogeneous
           Music Styles",
           Proceedings of the 15th International Society for Music Information
           Retrieval Conference (ISMIR), 2014.

    Examples
    --------
    The MultiModelSelectionProcessor takes a list of model predictions as it's
    call argument. Thus, `ppost_processor` of `RNNBeatProcessor` hast to be set
    to 'None' in order to get the predictions of all models.

    >>> proc = RNNBeatProcessor(post_processor=None)
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.beats.RNNBeatProcessor object at 0x...>

    When passing a file through the processor, a list with predictions, one for
    each model tested, is returned.

    >>> predictions = proc('tests/data/audio/sample.wav')
    >>> predictions  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [array([0.00535, 0.00774, ..., 0.02343, 0.04931], dtype=float32),
     array([0.0022 , 0.00282, ..., 0.00825, 0.0152 ], dtype=float32),
     ...,
     array([0.005  , 0.0052 , ..., 0.00472, 0.01524], dtype=float32),
     array([0.00319, 0.0044 , ..., 0.0081 , 0.01498], dtype=float32)]

    We can feed these predictions to the MultiModelSelectionProcessor.
    Since we do not have a dedicated reference prediction (which had to be the
    first element of the list and `num_ref_predictions` set to 1), we simply
    set `num_ref_predictions` to 'None'. MultiModelSelectionProcessor averages
    all predictions to obtain a reference prediction it compares all others to.

    >>> mm_proc = MultiModelSelectionProcessor(num_ref_predictions=None)
    >>> mm_proc(predictions)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    array([0.00759, 0.00901, ..., 0.00843, 0.01834], dtype=float32)

    """

    def __init__(self, num_ref_predictions, **kwargs):
        # pylint: disable=unused-argument

        self.num_ref_predictions = num_ref_predictions

    def process(self, predictions, **kwargs):
        """
        Selects the most appropriate predictions form the list of predictions.

        Parameters
        ----------
        predictions : list
            Predictions (beat activation functions) of multiple models.

        Returns
        -------
        numpy array
            Most suitable prediction.

        Notes
        -----
        The reference beat activation function must be the first one in the
        list of given predictions.

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


# function for detecting the beats based on the given dominant interval
def detect_beats(activations, interval, look_aside=0.2):
    """
    Detects the beats in the given activation function as in [1]_.

    Parameters
    ----------
    activations : numpy array
        Beat activations.
    interval : int
        Look for the next beat each `interval` frames.
    look_aside : float
        Look this fraction of the `interval` to each side to detect the beats.

    Returns
    -------
    numpy array
        Beat positions [frames].

    Notes
    -----
    A Hamming window of 2 * `look_aside` * `interval` is applied around the
    position where the beat is expected to prefer beats closer to the centre.

    References
    ----------
    .. [1] Sebastian Böck and Markus Schedl,
           "Enhanced Beat Tracking with Context-Aware Neural Networks",
           Proceedings of the 14th International Conference on Digital Audio
           Effects (DAFx), 2011.

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

        Parameters
        ----------
        position : int
            Start at this position.

        """
        # detect the nearest beat around the actual position
        act = signal_frame(activations, position, frames_look_aside * 2, 1)
        # apply a filtering window to prefer beats closer to the centre
        act = np.multiply(act, win)
        # search max
        if np.argmax(act) > 0:
            # maximum found, take that position
            position = np.argmax(act) + position - frames_look_aside
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
    Track the beats according to previously determined (local) tempo by
    iteratively aligning them around the estimated position [1]_.

    Parameters
    ----------
    look_aside : float, optional
        Look this fraction of the estimated beat interval to each side of the
        assumed next beat position to look for the most likely position of the
        next beat.
    look_ahead : float, optional
        Look `look_ahead` seconds in both directions to determine the local
        tempo and align the beats accordingly.
    tempo_estimator : :class:`TempoEstimationProcessor`, optional
        Use this processor to estimate the (local) tempo. If 'None' a default
        tempo estimator will be created and used.
    fps : float, optional
        Frames per second.
    kwargs : dict, optional
        Keyword arguments passed to
        :class:`madmom.features.tempo.TempoEstimationProcessor` if no
        `tempo_estimator` was given.

    Notes
    -----
    If `look_ahead` is not set, a constant tempo throughout the whole piece
    is assumed. If `look_ahead` is set, the local tempo (in a range +/-
    `look_ahead` seconds around the actual position) is estimated and then
    the next beat is tracked accordingly. This procedure is repeated from
    the new position to the end of the piece.

    Instead of the auto-correlation based method for tempo estimation proposed
    in [1]_, it uses a comb filter based method [2]_ per default. The behaviour
    can be controlled with the `tempo_method` parameter.

    References
    ----------
    .. [1] Sebastian Böck and Markus Schedl,
           "Enhanced Beat Tracking with Context-Aware Neural Networks",
           Proceedings of the 14th International Conference on Digital Audio
           Effects (DAFx), 2011.
    .. [2] Sebastian Böck, Florian Krebs and Gerhard Widmer,
           "Accurate Tempo Estimation based on Recurrent Neural Networks and
           Resonating Comb Filters",
           Proceedings of the 16th International Society for Music Information
           Retrieval Conference (ISMIR), 2015.

    Examples
    --------
    Create a BeatTrackingProcessor. The returned array represents the positions
    of the beats in seconds, thus the expected sampling rate has to be given.

    >>> proc = BeatTrackingProcessor(fps=100)
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.beats.BeatTrackingProcessor object at 0x...>

    Call this BeatTrackingProcessor with the beat activation function returned
    by RNNBeatProcessor to obtain the beat positions.

    >>> act = RNNBeatProcessor()('tests/data/audio/sample.wav')
    >>> proc(act)
    array([0.11, 0.45, 0.79, 1.13, 1.47, 1.81, 2.15, 2.49])

    """
    LOOK_ASIDE = 0.2
    LOOK_AHEAD = 10.

    def __init__(self, look_aside=LOOK_ASIDE, look_ahead=LOOK_AHEAD, fps=None,
                 tempo_estimator=None, **kwargs):
        # save variables
        self.look_aside = look_aside
        self.look_ahead = look_ahead
        self.fps = fps
        # tempo estimator
        if tempo_estimator is None:
            # import the TempoEstimation here otherwise we have a loop
            from .tempo import TempoEstimationProcessor
            # create default tempo estimator
            tempo_estimator = TempoEstimationProcessor(fps=fps, **kwargs)
        self.tempo_estimator = tempo_estimator

    def process(self, activations, **kwargs):
        """
        Detect the beats in the given activation function.

        Parameters
        ----------
        activations : numpy array
            Beat activation function.

        Returns
        -------
        beats : numpy array
            Detected beat positions [seconds].

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
                act = signal_frame(activations, pos, look_ahead_frames * 2, 1)
                # create a interval histogram
                histogram = self.tempo_estimator.interval_histogram(act)
                # get the dominant interval
                interval = self.tempo_estimator.dominant_interval(histogram)
                # add the offset (i.e. the new detected start position)
                positions = detect_beats(act, interval, self.look_aside)
                # correct the beat positions
                positions += pos - look_ahead_frames
                # remove all positions < already detected beats + min_interval
                next_pos = (detections[-1] + self.tempo_estimator.min_interval
                            if detections else 0)
                positions = positions[positions >= next_pos]
                # search the closest beat to the predicted beat position
                pos = positions[(np.abs(positions - pos)).argmin()]
                # append to the beats
                detections.append(pos)
                pos += interval

        # convert detected beats to a list of timestamps
        detections = np.array(detections) / float(self.fps)
        # remove beats with negative times and return them
        return detections[np.searchsorted(detections, 0):]

    @staticmethod
    def add_arguments(parser, look_aside=LOOK_ASIDE,
                      look_ahead=LOOK_AHEAD):
        """
        Add beat tracking related arguments to an existing parser.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        look_aside : float, optional
            Look this fraction of the estimated beat interval to each side of
            the assumed next beat position to look for the most likely position
            of the next beat.
        look_ahead : float, optional
            Look `look_ahead` seconds in both directions to determine the local
            tempo and align the beats accordingly.

        Returns
        -------
        parser_group : argparse argument group
            Beat tracking argument parser group.

        Notes
        -----
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


class BeatDetectionProcessor(BeatTrackingProcessor):
    """
    Class for detecting beats according to the previously determined global
    tempo by iteratively aligning them around the estimated position [1]_.

    Parameters
    ----------
    look_aside : float
        Look this fraction of the estimated beat interval to each side of the
        assumed next beat position to look for the most likely position of the
        next beat.
    fps : float, optional
        Frames per second.

    Notes
    -----
    A constant tempo throughout the whole piece is assumed.

    Instead of the auto-correlation based method for tempo estimation proposed
    in [1]_, it uses a comb filter based method [2]_ per default. The behaviour
    can be controlled with the `tempo_method` parameter.

    See Also
    --------
    :class:`BeatTrackingProcessor`

    References
    ----------
    .. [1] Sebastian Böck and Markus Schedl,
           "Enhanced Beat Tracking with Context-Aware Neural Networks",
           Proceedings of the 14th International Conference on Digital Audio
           Effects (DAFx), 2011.
    .. [2] Sebastian Böck, Florian Krebs and Gerhard Widmer,
           "Accurate Tempo Estimation based on Recurrent Neural Networks and
           Resonating Comb Filters",
           Proceedings of the 16th International Society for Music Information
           Retrieval Conference (ISMIR), 2015.

    Examples
    --------
    Create a BeatDetectionProcessor. The returned array represents the
    positions of the beats in seconds, thus the expected sampling rate has to
    be given.

    >>> proc = BeatDetectionProcessor(fps=100)
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.beats.BeatDetectionProcessor object at 0x...>

    Call this BeatDetectionProcessor with the beat activation function returned
    by RNNBeatProcessor to obtain the beat positions.

    >>> act = RNNBeatProcessor()('tests/data/audio/sample.wav')
    >>> proc(act)
    array([0.11, 0.45, 0.79, 1.13, 1.47, 1.81, 2.15, 2.49])

    """
    LOOK_ASIDE = 0.2

    def __init__(self, look_aside=LOOK_ASIDE, fps=None, **kwargs):
        super(BeatDetectionProcessor, self).__init__(look_aside=look_aside,
                                                     look_ahead=None, fps=fps,
                                                     **kwargs)


def _process_crf(process_tuple):
    """
    Extract the best beat sequence for a piece.

    This proxy function is necessary to process different intervals in parallel
    using the multiprocessing module.

    Parameters
    ----------
    process_tuple : tuple
        Tuple with (activations, dominant_interval, allowed deviation from the
        dominant interval per beat).

    Returns
    -------
    beats : numpy array
        Extracted beat positions [frames].
    log_prob : float
        Log probability of the beat sequence.

    """
    # pylint: disable=no-name-in-module
    from .beats_crf import best_sequence
    # activations, dominant_interval, interval_sigma = process_tuple
    return best_sequence(*process_tuple)


class CRFBeatDetectionProcessor(BeatTrackingProcessor):
    """
    Conditional Random Field Beat Detection.

    Tracks the beats according to the previously determined global tempo using
    a conditional random field (CRF) model.

    Parameters
    ----------
    interval_sigma : float, optional
        Allowed deviation from the dominant beat interval per beat.
    use_factors : bool, optional
        Use dominant interval multiplied by factors instead of intervals
        estimated by tempo estimator.
    num_intervals : int, optional
        Maximum number of estimated intervals to try.
    factors : list or numpy array, optional
        Factors of the dominant interval to try.

    References
    ----------
    .. [1] Filip Korzeniowski, Sebastian Böck and Gerhard Widmer,
           "Probabilistic Extraction of Beat Positions from a Beat Activation
           Function",
           Proceedings of the 15th International Society for Music Information
           Retrieval Conference (ISMIR), 2014.

    Examples
    --------
    Create a CRFBeatDetectionProcessor. The returned array represents the
    positions of the beats in seconds, thus the expected sampling rate has to
    be given.

    >>> proc = CRFBeatDetectionProcessor(fps=100)
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.beats.CRFBeatDetectionProcessor object at 0x...>

    Call this BeatDetectionProcessor with the beat activation function returned
    by RNNBeatProcessor to obtain the beat positions.

    >>> act = RNNBeatProcessor()('tests/data/audio/sample.wav')
    >>> proc(act)
    array([0.09, 0.79, 1.49])

    """
    INTERVAL_SIGMA = 0.18
    USE_FACTORS = False
    FACTORS = np.array([0.5, 0.67, 1.0, 1.5, 2.0])
    NUM_INTERVALS = 5
    # tempo defaults
    MIN_BPM = 20
    MAX_BPM = 240
    ACT_SMOOTH = 0.09
    HIST_SMOOTH = 7

    def __init__(self, interval_sigma=INTERVAL_SIGMA, use_factors=USE_FACTORS,
                 num_intervals=NUM_INTERVALS, factors=FACTORS, **kwargs):
        super(CRFBeatDetectionProcessor, self).__init__(**kwargs)
        # save parameters
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

    def process(self, activations, **kwargs):
        """
        Detect the beats in the given activation function.

        Parameters
        ----------
        activations : numpy array
            Beat activation function.

        Returns
        -------
        numpy array
            Detected beat positions [seconds].

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
        results = list(self.map(
            _process_crf, zip(it.repeat(contiguous_act), possible_intervals,
                              it.repeat(self.interval_sigma))))

        # normalize their probabilities
        normalized_seq_probabilities = np.array([r[1] / r[0].shape[0]
                                                 for r in results])
        # pick the best one
        best_seq = results[normalized_seq_probabilities.argmax()][0]

        # convert the detected beat positions to seconds and return them
        return best_seq.astype(np.float) / self.fps

    @staticmethod
    def add_arguments(parser, interval_sigma=INTERVAL_SIGMA,
                      use_factors=USE_FACTORS, num_intervals=NUM_INTERVALS,
                      factors=FACTORS):
        """
        Add CRFBeatDetection related arguments to an existing parser.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        interval_sigma : float, optional
            allowed deviation from the dominant beat interval per beat
        use_factors : bool, optional
            use dominant interval multiplied by factors instead of intervals
            estimated by tempo estimator
        num_intervals : int, optional
            max number of estimated intervals to try
        factors : list or numpy array, optional
            factors of the dominant interval to try

        Returns
        -------
        parser_group : argparse argument group
            CRF beat tracking argument parser group.

        """
        # pylint: disable=arguments-differ
        from ..utils import OverrideDefaultListAction
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
        g.add_argument('--factors', action=OverrideDefaultListAction,
                       default=factors, type=float, sep=',',
                       help='(comma separated) list with factors of dominant '
                            'interval to try [default=%(default)s]')
        return g


class DBNBeatTrackingProcessor(OnlineProcessor):
    """
    Beat tracking with RNNs and a dynamic Bayesian network (DBN) approximated
    by a Hidden Markov Model (HMM).

    Parameters
    ----------
    min_bpm : float, optional
        Minimum tempo used for beat tracking [bpm].
    max_bpm : float, optional
        Maximum tempo used for beat tracking [bpm].
    num_tempi : int, optional
        Number of tempi to model; if set, limit the number of tempi and use a
        log spacing, otherwise a linear spacing.
    transition_lambda : float, optional
        Lambda for the exponential tempo change distribution (higher values
        prefer a constant tempo from one beat to the next one).
    observation_lambda : int, optional
        Split one beat period into `observation_lambda` parts, the first
        representing beat states and the remaining non-beat states.
    threshold : float, optional
        Threshold the observations before Viterbi decoding.
    correct : bool, optional
        Correct the beats (i.e. align them to the nearest peak of the beat
        activation function).
    fps : float, optional
        Frames per second.
    online : bool, optional
        Use the forward algorithm (instead of Viterbi) to decode the beats.

    Notes
    -----
    Instead of the originally proposed state space and transition model for
    the DBN [1]_, the more efficient version proposed in [2]_ is used.

    References
    ----------
    .. [1] Sebastian Böck, Florian Krebs and Gerhard Widmer,
           "A Multi-Model Approach to Beat Tracking Considering Heterogeneous
           Music Styles",
           Proceedings of the 15th International Society for Music Information
           Retrieval Conference (ISMIR), 2014.
    .. [2] Florian Krebs, Sebastian Böck and Gerhard Widmer,
           "An Efficient State Space Model for Joint Tempo and Meter Tracking",
           Proceedings of the 16th International Society for Music Information
           Retrieval Conference (ISMIR), 2015.

    Examples
    --------
    Create a DBNBeatTrackingProcessor. The returned array represents the
    positions of the beats in seconds, thus the expected sampling rate has to
    be given.

    >>> proc = DBNBeatTrackingProcessor(fps=100)
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.beats.DBNBeatTrackingProcessor object at 0x...>

    Call this DBNBeatTrackingProcessor with the beat activation function
    returned by RNNBeatProcessor to obtain the beat positions.

    >>> act = RNNBeatProcessor()('tests/data/audio/sample.wav')
    >>> proc(act)
    array([0.1 , 0.45, 0.8 , 1.12, 1.48, 1.8 , 2.15, 2.49])

    """
    MIN_BPM = 55.
    MAX_BPM = 215.
    NUM_TEMPI = None
    TRANSITION_LAMBDA = 100
    OBSERVATION_LAMBDA = 16
    THRESHOLD = 0
    CORRECT = True

    def __init__(self, min_bpm=MIN_BPM, max_bpm=MAX_BPM, num_tempi=NUM_TEMPI,
                 transition_lambda=TRANSITION_LAMBDA,
                 observation_lambda=OBSERVATION_LAMBDA, correct=CORRECT,
                 threshold=THRESHOLD, fps=None, online=False, **kwargs):
        # pylint: disable=unused-argument
        # pylint: disable=no-name-in-module
        from .beats_hmm import (BeatStateSpace, BeatTransitionModel,
                                RNNBeatTrackingObservationModel)
        from ..ml.hmm import HiddenMarkovModel
        # convert timing information to construct a beat state space
        min_interval = 60. * fps / max_bpm
        max_interval = 60. * fps / min_bpm
        self.st = BeatStateSpace(min_interval, max_interval, num_tempi)
        # transition model
        self.tm = BeatTransitionModel(self.st, transition_lambda)
        # observation model
        self.om = RNNBeatTrackingObservationModel(self.st, observation_lambda)
        # instantiate a HMM
        self.hmm = HiddenMarkovModel(self.tm, self.om, None)
        # save variables
        self.correct = correct
        self.threshold = threshold
        self.fps = fps
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        # kepp state in online mode
        self.online = online
        # TODO: refactor the visualisation stuff
        if self.online:
            self.visualize = kwargs.get('verbose', False)
            self.counter = 0
            self.beat_counter = 0
            self.strength = 0
            self.last_beat = 0
            self.tempo = 0

    def reset(self):
        """Reset the DBNBeatTrackingProcessor."""
        # pylint: disable=attribute-defined-outside-init
        # reset the HMM
        self.hmm.reset()
        # reset other variables
        self.counter = 0
        self.beat_counter = 0
        self.strength = 0
        self.last_beat = 0
        self.tempo = 0

    def process_offline(self, activations, **kwargs):
        """
        Detect the beats in the given activation function with Viterbi
        decoding.

        Parameters
        ----------
        activations : numpy array
            Beat activation function.

        Returns
        -------
        beats : numpy array
            Detected beat positions [seconds].

        """
        # init the beats to return and the offset
        beats = np.empty(0, dtype=np.int)
        first = 0
        # use only the activations > threshold
        if self.threshold:
            idx = np.nonzero(activations >= self.threshold)[0]
            if idx.any():
                first = max(first, np.min(idx))
                last = min(len(activations), np.max(idx) + 1)
            else:
                last = first
            activations = activations[first:last]
        # return the beats if no activations given / remain after thresholding
        if not activations.any():
            return beats
        # get the best state path by calling the viterbi algorithm
        path, _ = self.hmm.viterbi(activations)
        # correct the beat positions if needed
        if self.correct:
            # for each detection determine the "beat range", i.e. states where
            # the pointers of the observation model are 1
            beat_range = self.om.pointers[path]
            # get all change points between True and False
            idx = np.nonzero(np.diff(beat_range))[0] + 1
            # if the first frame is in the beat range, add a change at frame 0
            if beat_range[0]:
                idx = np.r_[0, idx]
            # if the last frame is in the beat range, append the length of the
            # array
            if beat_range[-1]:
                idx = np.r_[idx, beat_range.size]
            # iterate over all regions
            if idx.any():
                for left, right in idx.reshape((-1, 2)):
                    # pick the frame with the highest activations value
                    peak = np.argmax(activations[left:right]) + left
                    beats = np.hstack((beats, peak))
        else:
            # just take the frames with the smallest beat state values
            from scipy.signal import argrelmin
            beats = argrelmin(self.st.state_positions[path], mode='wrap')[0]
            # recheck if they are within the "beat range", i.e. the pointers
            # of the observation model for that state must be 1
            # Note: interpolation and alignment of the beats to be at state 0
            #       does not improve results over this simple method
            beats = beats[self.om.pointers[path[beats]] == 1]
        # convert the detected beats to seconds and return them
        return (beats + first) / float(self.fps)

    def process_online(self, activations, reset=True, **kwargs):
        """
        Detect the beats in the given activation function with the forward
        algorithm.

        Parameters
        ----------
        activations : numpy array
            Beat activation for a single frame.
        reset : bool, optional
            Reset the DBNBeatTrackingProcessor to its initial state before
            processing.

        Returns
        -------
        beats : numpy array
            Detected beat position [seconds].

        """
        # reset to initial state
        if reset:
            self.reset()
        # use forward path to get best state
        fwd = self.hmm.forward(activations, reset=reset)
        # choose the best state for each step
        states = np.argmax(fwd, axis=1)
        # decide which time steps are beats
        beats = self.om.pointers[states] == 1
        # the positions inside the beats
        positions = self.st.state_positions[states]
        # visualisation stuff (only when called frame by frame)
        if self.visualize and len(activations) == 1:
            beat_length = 80
            display = [' '] * beat_length
            display[int(positions * beat_length)] = '*'
            # activation strength indicator
            strength_length = 10
            self.strength = int(max(self.strength, activations * 10))
            display.append('| ')
            display.extend(['*'] * self.strength)
            display.extend([' '] * (strength_length - self.strength))
            # reduce the displayed strength every couple of frames
            if self.counter % 5 == 0:
                self.strength -= 1
            # beat indicator
            if beats:
                self.beat_counter = 3
            if self.beat_counter > 0:
                display.append('| X ')
            else:
                display.append('|   ')
            self.beat_counter -= 1
            # display tempo
            display.append('| %5.1f | ' % self.tempo)
            sys.stderr.write('\r%s' % ''.join(display))
            sys.stderr.flush()
        # forward path often reports multiple beats close together, thus report
        # only beats more than the minimum interval apart
        beats_ = []
        for frame in np.nonzero(beats)[0]:
            cur_beat = (frame + self.counter) / float(self.fps)
            next_beat = self.last_beat + 60. / self.max_bpm
            # FIXME: this skips the first beat, but maybe this has a positive
            #        effect on the overall beat tracking accuracy
            if cur_beat >= next_beat:
                # update tempo
                self.tempo = 60. / (cur_beat - self.last_beat)
                # update last beat
                self.last_beat = cur_beat
                # append to beats
                beats_.append(cur_beat)
        # increase counter
        self.counter += len(activations)
        # return beat(s)
        return np.array(beats_)

    process_forward = process_online

    process_viterbi = process_offline

    @staticmethod
    def add_arguments(parser, min_bpm=MIN_BPM, max_bpm=MAX_BPM,
                      num_tempi=NUM_TEMPI, transition_lambda=TRANSITION_LAMBDA,
                      observation_lambda=OBSERVATION_LAMBDA,
                      threshold=THRESHOLD, correct=CORRECT):
        """
        Add DBN related arguments to an existing parser object.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        min_bpm : float, optional
            Minimum tempo used for beat tracking [bpm].
        max_bpm : float, optional
            Maximum tempo used for beat tracking [bpm].
        num_tempi : int, optional
            Number of tempi to model; if set, limit the number of tempi and use
            a log spacing, otherwise a linear spacing.
        transition_lambda : float, optional
            Lambda for the exponential tempo change distribution (higher values
            prefer a constant tempo over a tempo change from one beat to the
            next one).
        observation_lambda : float, optional
            Split one beat period into `observation_lambda` parts, the first
            representing beat states and the remaining non-beat states.
        threshold : float, optional
            Threshold the observations before Viterbi decoding.
        correct : bool, optional
            Correct the beats (i.e. align them to the nearest peak of the beat
            activation function).

        Returns
        -------
        parser_group : argparse argument group
            DBN beat tracking argument parser group

        """
        # pylint: disable=arguments-differ
        # add DBN parser group
        g = parser.add_argument_group('dynamic Bayesian Network arguments')
        # add a transition parameters
        g.add_argument('--min_bpm', action='store', type=float,
                       default=min_bpm,
                       help='minimum tempo [bpm, default=%(default).2f]')
        g.add_argument('--max_bpm', action='store', type=float,
                       default=max_bpm,
                       help='maximum tempo [bpm,  default=%(default).2f]')
        g.add_argument('--num_tempi', action='store', type=int,
                       default=num_tempi,
                       help='limit the number of tempi; if set, align the '
                            'tempi with a log spacing, otherwise linearly')
        g.add_argument('--transition_lambda', action='store', type=float,
                       default=transition_lambda,
                       help='lambda of the tempo transition distribution; '
                            'higher values prefer a constant tempo over a '
                            'tempo change from one beat to the next one '
                            '[default=%(default).1f]')
        # observation model stuff
        g.add_argument('--observation_lambda', action='store', type=float,
                       default=observation_lambda,
                       help='split one beat period into N parts, the first '
                            'representing beat states and the remaining '
                            'non-beat states [default=%(default)i]')
        g.add_argument('-t', dest='threshold', action='store', type=float,
                       default=threshold,
                       help='threshold the observations before Viterbi '
                            'decoding [default=%(default).2f]')
        # option to correct the beat positions
        if correct:
            g.add_argument('--no_correct', dest='correct',
                           action='store_false', default=correct,
                           help='do not correct the beat positions (i.e. do '
                                'not align them to the nearest peak of the '
                                'beat activation function)')
        else:
            g.add_argument('--correct', dest='correct',
                           action='store_true', default=correct,
                           help='correct the beat positions (i.e. align them '
                                'to the nearest peak of the beat activation'
                                'function)')
        # return the argument group so it can be modified if needed
        return g
