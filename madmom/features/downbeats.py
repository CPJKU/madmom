# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains downbeat tracking related functionality.

"""

from __future__ import absolute_import, division, print_function

import numpy as np

from madmom.processors import Processor, SequentialProcessor, ParallelProcessor


class RNNDownBeatProcessor(SequentialProcessor):
    """
    Processor to get a joint beat and downbeat activation function from
    multiple RNNs.

    References
    ----------
    .. [1] Sebastian Böck, Florian Krebs and Gerhard Widmer,
           "Joint Beat and Downbeat Tracking with Recurrent Neural Networks"
           Proceedings of the 17th International Society for Music Information
           Retrieval Conference (ISMIR), 2016.

    Examples
    --------
    Create a RNNDownBeatProcessor and pass a file through the processor.
    The returned 2d array represents the probabilities at each frame, sampled
    at 100 frames per second. The columns represent 'beat' and 'downbeat'.

    >>> proc = RNNDownBeatProcessor()
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.downbeats.RNNDownBeatProcessor object at 0x...>
    >>> proc('tests/data/audio/sample.wav')
    ... # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    array([[ 0.00011, 0.00037],
           [ 0.00008, 0.00043],
           ...,
           [ 0.00791, 0.00169],
           [ 0.03425, 0.00494]], dtype=float32)

    """

    def __init__(self, **kwargs):
        # pylint: disable=unused-argument
        from functools import partial
        from ..audio.signal import SignalProcessor, FramedSignalProcessor
        from ..audio.stft import ShortTimeFourierTransformProcessor
        from ..audio.spectrogram import (
            FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
            SpectrogramDifferenceProcessor)
        from ..ml.nn import NeuralNetworkEnsemble
        from ..models import DOWNBEATS_BLSTM

        # define pre-processing chain
        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        # process the multi-resolution spec & diff in parallel
        multi = ParallelProcessor([])
        frame_sizes = [1024, 2048, 4096]
        num_bands = [3, 6, 12]
        for frame_size, num_bands in zip(frame_sizes, num_bands):
            frames = FramedSignalProcessor(frame_size=frame_size, fps=100)
            stft = ShortTimeFourierTransformProcessor()  # caching FFT window
            filt = FilteredSpectrogramProcessor(
                num_bands=num_bands, fmin=30, fmax=17000, norm_filters=True)
            spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
            diff = SpectrogramDifferenceProcessor(
                diff_ratio=0.5, positive_diffs=True, stack_diffs=np.hstack)
            # process each frame size with spec and diff sequentially
            multi.append(SequentialProcessor((frames, stft, filt, spec, diff)))
        # stack the features and processes everything sequentially
        pre_processor = SequentialProcessor((sig, multi, np.hstack))
        # process the pre-processed signal with a NN ensemble
        nn = NeuralNetworkEnsemble.load(DOWNBEATS_BLSTM, **kwargs)
        # use only the beat & downbeat (i.e. remove non-beat) activations
        act = partial(np.delete, obj=0, axis=1)
        # instantiate a SequentialProcessor
        super(RNNDownBeatProcessor, self).__init__((pre_processor, nn, act))


def _process_dbn(process_tuple):
    """
    Extract the best path through the state space in an observation sequence.

    This proxy function is necessary to process different sequences in parallel
    using the multiprocessing module.

    Parameters
    ----------
    process_tuple : tuple
        Tuple with (HMM, observations).

    Returns
    -------
    path : numpy array
        Best path through the state space.
    log_prob : float
        Log probability of the path.

    """
    # pylint: disable=no-name-in-module
    return process_tuple[0].viterbi(process_tuple[1])


class DBNDownBeatTrackingProcessor(Processor):
    """
    Downbeat tracking with RNNs and a dynamic Bayesian network (DBN)
    approximated by a Hidden Markov Model (HMM).

    Parameters
    ----------
    beats_per_bar : int or list
        Number of beats per bar to be modeled. Can be either a single number
        or a list or array with bar lengths (in beats).
    min_bpm : float or list, optional
        Minimum tempo used for beat tracking [bpm]. If a list is given, each
        item corresponds to the number of beats per bar at the same position.
    max_bpm : float or list, optional
        Maximum tempo used for beat tracking [bpm]. If a list is given, each
        item corresponds to the number of beats per bar at the same position.
    num_tempi : int or list, optional
        Number of tempi to model; if set, limit the number of tempi and use a
        log spacing, otherwise a linear spacing. If a list is given, each
        item corresponds to the number of beats per bar at the same position.
    transition_lambda : float or list, optional
        Lambda for the exponential tempo change distribution (higher values
        prefer a constant tempo from one beat to the next one).  If a list is
        given, each item corresponds to the number of beats per bar at the
        same position.
    observation_lambda : int, optional
        Split one (down-)beat period into `observation_lambda` parts, the first
        representing (down-)beat states and the remaining non-beat states.
    threshold : float, optional
        Threshold the RNN (down-)beat activations before Viterbi decoding.
    correct : bool, optional
        Correct the beats (i.e. align them to the nearest peak of the
        (down-)beat activation function).
    downbeats : bool, optional
        Report downbeats only, not all beats and their position inside the bar.
    fps : float, optional
        Frames per second.

    References
    ----------
    .. [1] Sebastian Böck, Florian Krebs and Gerhard Widmer,
           "Joint Beat and Downbeat Tracking with Recurrent Neural Networks"
           Proceedings of the 17th International Society for Music Information
           Retrieval Conference (ISMIR), 2016.

    Examples
    --------
    Create a DBNDownBeatTrackingProcessor. The returned array represents the
    positions of the beats and their position inside the bar. The position is
    given in seconds, thus the expected sampling rate is needed. The position
    inside the bar follows the natural counting and starts at 1.

    The number of beats per bar which should be modelled must be given, all
    other parameters (e.g. tempo range) are optional but must have the same
    length as `beats_per_bar`, i.e. must be given for each bar length.

    >>> proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.downbeats.DBNDownBeatTrackingProcessor object at 0x...>

    Call this DBNDownBeatTrackingProcessor with the beat activation function
    returned by RNNDownBeatProcessor to obtain the beat positions.

    >>> act = RNNDownBeatProcessor()('tests/data/audio/sample.wav')
    >>> proc(act)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    array([[ 0.09, 1. ],
           [ 0.45, 2. ],
           ...,
           [ 2.14, 3. ],
           [ 2.49, 4. ]])

    """

    MIN_BPM = 55.
    MAX_BPM = 215.
    NUM_TEMPI = 60
    TRANSITION_LAMBDA = 100
    OBSERVATION_LAMBDA = 16
    THRESHOLD = 0.05
    CORRECT = True

    def __init__(self, beats_per_bar, min_bpm=MIN_BPM, max_bpm=MAX_BPM,
                 num_tempi=NUM_TEMPI, transition_lambda=TRANSITION_LAMBDA,
                 observation_lambda=OBSERVATION_LAMBDA, threshold=THRESHOLD,
                 correct=CORRECT, downbeats=False, fps=None, **kwargs):
        # pylint: disable=unused-argument
        # pylint: disable=no-name-in-module

        from madmom.ml.hmm import HiddenMarkovModel as Hmm
        from .beats_hmm import (BarStateSpace, BarTransitionModel,
                                RNNDownBeatTrackingObservationModel)

        # expand arguments to arrays
        beats_per_bar = np.array(beats_per_bar, ndmin=1)
        min_bpm = np.array(min_bpm, ndmin=1)
        max_bpm = np.array(max_bpm, ndmin=1)
        num_tempi = np.array(num_tempi, ndmin=1)
        transition_lambda = np.array(transition_lambda, ndmin=1)
        # make sure the other arguments are long enough by repeating them
        # TODO: check if they are of length 1?
        if len(min_bpm) != len(beats_per_bar):
            min_bpm = np.repeat(min_bpm, len(beats_per_bar))
        if len(max_bpm) != len(beats_per_bar):
            max_bpm = np.repeat(max_bpm, len(beats_per_bar))
        if len(num_tempi) != len(beats_per_bar):
            num_tempi = np.repeat(num_tempi, len(beats_per_bar))
        if len(transition_lambda) != len(beats_per_bar):
            transition_lambda = np.repeat(transition_lambda,
                                          len(beats_per_bar))
        if not (len(min_bpm) == len(max_bpm) == len(num_tempi) ==
                len(beats_per_bar) == len(transition_lambda)):
            raise ValueError('`min_bpm`, `max_bpm`, `num_tempi`, `num_beats` '
                             'and `transition_lambda` must all have the same '
                             'length.')
        # get num_threads from kwargs
        num_threads = min(len(beats_per_bar), kwargs.get('num_threads', 1))
        # init a pool of workers (if needed)
        self.map = map
        if num_threads != 1:
            import multiprocessing as mp
            self.map = mp.Pool(num_threads).map
        # convert timing information to construct a beat state space
        min_interval = 60. * fps / max_bpm
        max_interval = 60. * fps / min_bpm
        # model the different bar lengths
        self.hmms = []
        for b, beats in enumerate(beats_per_bar):
            st = BarStateSpace(beats, min_interval[b], max_interval[b],
                               num_tempi[b])
            tm = BarTransitionModel(st, transition_lambda[b])
            om = RNNDownBeatTrackingObservationModel(st, observation_lambda)
            self.hmms.append(Hmm(tm, om))
        # save variables
        self.beats_per_bar = beats_per_bar
        self.threshold = threshold
        self.correct = correct
        self.downbeats = downbeats
        self.fps = fps

    def process(self, activations, **kwargs):
        """
        Detect the (down-)beats in the given activation function.

        Parameters
        ----------
        activations : numpy array, shape (num_frames, 2)
            Activation function with probabilities corresponding to beats
            and downbeats given in the first and second column, respectively.

        Returns
        -------
        beats : numpy array, shape (num_beats, 2)
            Detected (down-)beat positions [seconds] and beat numbers.

        """
        import itertools as it
        # use only the activations > threshold (init offset to be added later)
        first = 0
        if self.threshold:
            idx = np.nonzero(activations >= self.threshold)[0]
            if idx.any():
                first = max(first, np.min(idx))
                last = min(len(activations), np.max(idx) + 1)
            else:
                last = first
            activations = activations[first:last]
        # return no beats if no activations given / remain after thresholding
        if not activations.any():
            return np.empty((0, 2))
        # (parallel) decoding of the activations with HMM
        results = list(self.map(_process_dbn, zip(self.hmms,
                                                  it.repeat(activations))))
        # choose the best HMM (highest log probability)
        best = np.argmax(np.asarray(results)[:, 1])
        # the best path through the state space
        path, _ = results[best]
        # the state space and observation model of the best HMM
        st = self.hmms[best].transition_model.state_space
        om = self.hmms[best].observation_model
        # the positions inside the pattern (0..num_beats)
        positions = st.state_positions[path]
        # corresponding beats (add 1 for natural counting)
        beat_numbers = positions.astype(int) + 1
        if self.correct:
            beats = np.empty(0, dtype=np.int)
            # for each detection determine the "beat range", i.e. states where
            # the pointers of the observation model are >= 1
            beat_range = om.pointers[path] >= 1
            # get all change points between True and False (cast to int before)
            idx = np.nonzero(np.diff(beat_range.astype(np.int)))[0] + 1
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
                    # Note: we look for both beats and down-beat activations;
                    #       since np.argmax works on the flattened array, we
                    #       need to divide by 2
                    peak = np.argmax(activations[left:right]) // 2 + left
                    beats = np.hstack((beats, peak))
        else:
            # transitions are the points where the beat numbers change
            # FIXME: we might miss the first or last beat!
            #        we could calculate the interval towards the beginning/end
            #        to decide whether to include these points
            beats = np.nonzero(np.diff(beat_numbers))[0] + 1
        # convert the detected beats to seconds and add the beat numbers
        if beats.any():
            beats = np.vstack(((beats + first) / float(self.fps),
                               beat_numbers[beats])).T
        else:
            beats = np.empty((0, 2))
        # return the downbeats or beats and their beat number
        if self.downbeats:
            return beats[beats[:, 1] == 1][:, 0]
        else:
            return beats

    @staticmethod
    def add_arguments(parser, beats_per_bar, min_bpm=MIN_BPM, max_bpm=MAX_BPM,
                      num_tempi=NUM_TEMPI, transition_lambda=TRANSITION_LAMBDA,
                      observation_lambda=OBSERVATION_LAMBDA,
                      threshold=THRESHOLD, correct=CORRECT):
        """
        Add DBN downbeat tracking related arguments to an existing parser
        object.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        beats_per_bar : int or list, optional
            Number of beats per bar to be modeled. Can be either a single
            number or a list with bar lengths (in beats).
        min_bpm : float or list, optional
            Minimum tempo used for beat tracking [bpm]. If a list is given,
            each item corresponds to the number of beats per bar at the same
            position.
        max_bpm : float or list, optional
            Maximum tempo used for beat tracking [bpm]. If a list is given,
            each item corresponds to the number of beats per bar at the same
            position.
        num_tempi : int or list, optional
            Number of tempi to model; if set, limit the number of tempi and use
            a log spacing, otherwise a linear spacing. If a list is given,
            each item corresponds to the number of beats per bar at the same
            position.
        transition_lambda : float or list, optional
            Lambda for the exponential tempo change distribution (higher values
            prefer a constant tempo over a tempo change from one beat to the
            next one). If a list is given, each item corresponds to the number
            of beats per bar at the same position.
        observation_lambda : float, optional
            Split one (down-)beat period into `observation_lambda` parts, the
            first representing (down-)beat states and the remaining non-beat
            states.
        threshold : float, optional
            Threshold the RNN (down-)beat activations before Viterbi decoding.
        correct : bool, optional
            Correct the beats (i.e. align them to the nearest peak of the
            (down-)beat activation function).

        Returns
        -------
        parser_group : argparse argument group
            DBN downbeat tracking argument parser group

        """
        # pylint: disable=arguments-differ
        from madmom.utils import OverrideDefaultListAction

        # add DBN parser group
        g = parser.add_argument_group('dynamic Bayesian Network arguments')
        # add a transition parameters
        g.add_argument('--beats_per_bar', action=OverrideDefaultListAction,
                       default=beats_per_bar, type=int, sep=',',
                       help='number of beats per bar to be modeled (comma '
                            'separated list of bar length in beats) '
                            '[default=%(default)s]')
        g.add_argument('--min_bpm', action=OverrideDefaultListAction,
                       default=min_bpm, type=float, sep=',',
                       help='minimum tempo (comma separated list with one '
                            'value per bar length) [bpm, default=%(default)s]')
        g.add_argument('--max_bpm', action=OverrideDefaultListAction,
                       default=max_bpm, type=float, sep=',',
                       help='maximum tempo (comma separated list with one '
                            'value per bar length) [bpm, default=%(default)s]')
        g.add_argument('--num_tempi', action=OverrideDefaultListAction,
                       default=num_tempi, type=int, sep=',',
                       help='limit the number of tempi; if set, align the '
                            'tempi with log spacings, otherwise linearly '
                            '(comma separated list with one value per bar '
                            'length) [default=%(default)s]')
        g.add_argument('--transition_lambda',
                       action=OverrideDefaultListAction,
                       default=transition_lambda, type=float, sep=',',
                       help='lambda of the tempo transition distribution; '
                            'higher values prefer a constant tempo over a '
                            'tempo change from one beat to the next one ('
                            'comma separated list with one value per bar '
                            'length) [default=%(default)s]')
        # observation model stuff
        g.add_argument('--observation_lambda', action='store', type=float,
                       default=observation_lambda,
                       help='split one (down-)beat period into N parts, the '
                            'first representing beat states and the remaining '
                            'non-beat states [default=%(default)i]')
        g.add_argument('-t', dest='threshold', action='store', type=float,
                       default=threshold,
                       help='threshold the observations before Viterbi '
                            'decoding [default=%(default).2f]')
        # option to correct the beat positions
        if correct is True:
            g.add_argument('--no_correct', dest='correct',
                           action='store_false', default=correct,
                           help='do not correct the (down-)beat positions '
                                '(i.e. do not align them to the nearest peak '
                                'of the (down-)beat activation function)')
        elif correct is False:
            g.add_argument('--correct', dest='correct',
                           action='store_true', default=correct,
                           help='correct the (down-)beat positions (i.e. '
                                'align them to the nearest peak of the '
                                '(down-)beat  activation function)')
        # add output format stuff
        g = parser.add_argument_group('output arguments')
        g.add_argument('--downbeats', action='store_true', default=False,
                       help='output only the downbeats')
        # return the argument group so it can be modified if needed
        return g


class PatternTrackingProcessor(Processor):
    """
    Pattern tracking with a dynamic Bayesian network (DBN) approximated by a
    Hidden Markov Model (HMM).

    Parameters
    ----------
    pattern_files : list
        List of files with the patterns (including the fitted GMMs and
        information about the number of beats).
    min_bpm : list, optional
        Minimum tempi used for pattern tracking [bpm].
    max_bpm : list, optional
        Maximum tempi used for pattern tracking [bpm].
    num_tempi : int or list, optional
        Number of tempi to model; if set, limit the number of tempi and use a
        log spacings, otherwise a linear spacings.
    transition_lambda : float or list, optional
        Lambdas for the exponential tempo change distributions (higher values
        prefer constant tempi from one beat to the next one).
    downbeats : bool, optional
        Report only the downbeats instead of the beats and the respective
        position inside the bar.
    fps : float, optional
        Frames per second.

    Notes
    -----
    `min_bpm`, `max_bpm`, `num_tempo_states`, and `transition_lambda` must
    contain as many items as rhythmic patterns are modeled (i.e. length of
    `pattern_files`).
    If a single value is given for `num_tempo_states` and `transition_lambda`,
    this value is used for all rhythmic patterns.

    Instead of the originally proposed state space and transition model for
    the DBN [1]_, the more efficient version proposed in [2]_ is used.

    References
    ----------
    .. [1] Florian Krebs, Sebastian Böck and Gerhard Widmer,
           "Rhythmic Pattern Modeling for Beat and Downbeat Tracking in Musical
           Audio",
           Proceedings of the 15th International Society for Music Information
           Retrieval Conference (ISMIR), 2013.
    .. [2] Florian Krebs, Sebastian Böck and Gerhard Widmer,
           "An Efficient State Space Model for Joint Tempo and Meter Tracking",
           Proceedings of the 16th International Society for Music Information
           Retrieval Conference (ISMIR), 2015.

    Examples
    --------
    Create a PatternTrackingProcessor from the given pattern files. These
    pattern files include fitted GMMs for the observation model of the HMM.
    The returned array represents the positions of the beats and their position
    inside the bar. The position is given in seconds, thus the expected
    sampling rate is needed. The position inside the bar follows the natural
    counting and starts at 1.

    >>> from madmom.models import PATTERNS_BALLROOM
    >>> proc = PatternTrackingProcessor(PATTERNS_BALLROOM, fps=50)
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.downbeats.PatternTrackingProcessor object at 0x...>

    Call this PatternTrackingProcessor with a multi-band spectrogram to obtain
    the beat and downbeat positions. The parameters of the spectrogram have to
    correspond to those used to fit the GMMs.

    >>> from madmom.audio.spectrogram import LogarithmicSpectrogramProcessor, \
SpectrogramDifferenceProcessor, MultiBandSpectrogramProcessor
    >>> from madmom.processors import SequentialProcessor
    >>> log = LogarithmicSpectrogramProcessor()
    >>> diff = SpectrogramDifferenceProcessor(positive_diffs=True)
    >>> mb = MultiBandSpectrogramProcessor(crossover_frequencies=[270])
    >>> pre_proc = SequentialProcessor([log, diff, mb])

    >>> act = pre_proc('tests/data/audio/sample.wav')
    >>> proc(act)  # doctest: +ELLIPSIS
    array([[ 0.82,  4.  ],
           [ 1.78,  1.  ],
           ...,
           [ 3.7 ,  3.  ],
           [ 4.66,  4.  ]])
    """
    # TODO: this should not be lists (lists are mutable!)
    MIN_BPM = [55, 60]
    MAX_BPM = [205, 225]
    NUM_TEMPI = [None, None]
    # TODO: make this parametric
    # Note: if lambda is given as a list, the individual values represent the
    #       lambdas for each transition into the beat at this index position
    TRANSITION_LAMBDA = [100, 100]

    def __init__(self, pattern_files, min_bpm=MIN_BPM, max_bpm=MAX_BPM,
                 num_tempi=NUM_TEMPI, transition_lambda=TRANSITION_LAMBDA,
                 downbeats=False, fps=None, **kwargs):
        # pylint: disable=unused-argument
        # pylint: disable=no-name-in-module

        import pickle
        from .beats_hmm import (BarTransitionModel, BarStateSpace,
                                MultiPatternStateSpace,
                                MultiPatternTransitionModel,
                                GMMPatternTrackingObservationModel)
        from ..ml.hmm import HiddenMarkovModel

        # expand num_tempi and transition_lambda to lists if needed
        if not isinstance(num_tempi, list):
            num_tempi = [num_tempi] * len(pattern_files)
        if not isinstance(transition_lambda, list):
            transition_lambda = [transition_lambda] * len(pattern_files)
        # check if all lists have the same length
        if not (len(min_bpm) == len(max_bpm) == len(num_tempi) ==
                len(transition_lambda) == len(pattern_files)):
            raise ValueError('`min_bpm`, `max_bpm`, `num_tempi` and '
                             '`transition_lambda` must have the same length '
                             'as number of patterns.')
        # save some variables
        self.downbeats = downbeats
        self.fps = fps
        self.num_beats = []
        # convert timing information to construct a state space
        min_interval = 60. * self.fps / np.asarray(max_bpm)
        max_interval = 60. * self.fps / np.asarray(min_bpm)
        # collect beat/bar state spaces, transition models, and GMMs
        state_spaces = []
        transition_models = []
        gmms = []
        # check that at least one pattern is given
        if not pattern_files:
            raise ValueError('at least one rhythmical pattern must be given.')
        # load the patterns
        for p, pattern_file in enumerate(pattern_files):
            with open(pattern_file, 'rb') as f:
                # Python 2 and 3 behave differently
                # TODO: use some other format to save the GMMs (.npz, .hdf5)
                try:
                    # Python 3
                    pattern = pickle.load(f, encoding='latin1')
                except TypeError:
                    # Python 2 doesn't have/need the encoding
                    pattern = pickle.load(f)
            # get the fitted GMMs and number of beats
            gmms.append(pattern['gmms'])
            num_beats = pattern['num_beats']
            self.num_beats.append(num_beats)
            # model each rhythmic pattern as a bar
            state_space = BarStateSpace(num_beats, min_interval[p],
                                        max_interval[p], num_tempi[p])
            transition_model = BarTransitionModel(state_space,
                                                  transition_lambda[p])
            state_spaces.append(state_space)
            transition_models.append(transition_model)
        # create multi pattern state space, transition and observation model
        self.st = MultiPatternStateSpace(state_spaces)
        self.tm = MultiPatternTransitionModel(transition_models)
        self.om = GMMPatternTrackingObservationModel(gmms, self.st)
        # instantiate a HMM
        self.hmm = HiddenMarkovModel(self.tm, self.om, None)

    def process(self, features, **kwargs):
        """
        Detect the (down-)beats given the features.

        Parameters
        ----------
        features : numpy array
            Multi-band spectral features.

        Returns
        -------
        beats : numpy array, shape (num_beats, 2)
            Detected (down-)beat positions [seconds] and beat numbers.

        """
        # get the best state path by calling the viterbi algorithm
        path, _ = self.hmm.viterbi(features)
        # the positions inside the pattern (0..num_beats)
        positions = self.st.state_positions[path]
        # corresponding beats (add 1 for natural counting)
        beat_numbers = positions.astype(int) + 1
        # transitions are the points where the beat numbers change
        # FIXME: we might miss the first or last beat!
        #        we could calculate the interval towards the beginning/end to
        #        decide whether to include these points
        beat_positions = np.nonzero(np.diff(beat_numbers))[0] + 1
        # stack the beat positions (converted to seconds) and beat numbers
        beats = np.vstack((beat_positions / float(self.fps),
                           beat_numbers[beat_positions])).T
        # return the downbeats or beats and their beat number
        if self.downbeats:
            return beats[beats[:, 1] == 1][:, 0]
        else:
            return beats

    @staticmethod
    def add_arguments(parser, pattern_files=None, min_bpm=MIN_BPM,
                      max_bpm=MAX_BPM, num_tempi=NUM_TEMPI,
                      transition_lambda=TRANSITION_LAMBDA):
        """
        Add DBN related arguments for pattern tracking to an existing parser
        object.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        pattern_files : list
            Load the patterns from these files.
        min_bpm : list, optional
            Minimum tempi used for beat tracking [bpm].
        max_bpm : list, optional
            Maximum tempi used for beat tracking [bpm].
        num_tempi : int or list, optional
            Number of tempi to model; if set, limit the number of states and
            use log spacings, otherwise a linear spacings.
        transition_lambda : float or list, optional
            Lambdas for the exponential tempo change distribution (higher
            values prefer constant tempi from one beat to the next one).

        Returns
        -------
        parser_group : argparse argument group
            Pattern tracking argument parser group

        Notes
        -----
        `pattern_files`, `min_bpm`, `max_bpm`, `num_tempi`, and
        `transition_lambda` must have the same number of items.

        """
        from ..utils import OverrideDefaultListAction
        # add GMM options
        if pattern_files is not None:
            g = parser.add_argument_group('GMM arguments')
            g.add_argument('--pattern_files', action=OverrideDefaultListAction,
                           default=pattern_files,
                           help='load the patterns (with the fitted GMMs) '
                                'from these files (comma separated list)')
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
        g.add_argument('--num_tempi', action=OverrideDefaultListAction,
                       default=num_tempi, type=int, sep=',',
                       help='limit the number of tempi; if set, align the '
                            'tempi with log spacings, otherwise linearly '
                            '(comma separated list with one value per pattern)'
                            ' [default=%(default)s]')
        g.add_argument('--transition_lambda', action=OverrideDefaultListAction,
                       default=transition_lambda, type=float, sep=',',
                       help='lambda of the tempo transition distribution; '
                            'higher values prefer a constant tempo over a '
                            'tempo change from one bar to the next one (comma '
                            'separated list with one value per pattern) '
                            '[default=%(default)s]')
        # add output format stuff
        g = parser.add_argument_group('output arguments')
        g.add_argument('--downbeats', action='store_true', default=False,
                       help='output only the downbeats')
        # return the argument group so it can be modified if needed
        return g
