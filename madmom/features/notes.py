# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains note transcription related functionality.

Notes are stored as numpy arrays with the following column definition:

'note_time' 'MIDI_note' ['duration' ['MIDI_velocity']]

"""

from __future__ import absolute_import, division, print_function

import numpy as np

from .onsets import OnsetPeakPickingProcessor, peak_picking
from ..processors import ParallelProcessor, Processor, SequentialProcessor
from ..utils import combine_events


# class for detecting notes with a RNN
class RNNPianoNoteProcessor(SequentialProcessor):
    """
    Processor to get a (piano) note onset activation function from a RNN.

    References
    ----------

    .. [1] Sebastian Böck and Markus Schedl,
           "Polyphonic Piano Note Transcription with Recurrent Neural Networks"
           Proceedings of the 37th International Conference on Acoustics,
           Speech and Signal Processing (ICASSP), 2012.

    Examples
    --------
    Create a RNNPianoNoteProcessor and pass a file through the processor to
    obtain a note onset activation function (sampled with 100 frames per
    second).

    >>> proc = RNNPianoNoteProcessor()
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.notes.RNNPianoNoteProcessor object at 0x...>
    >>> act = proc('tests/data/audio/sample.wav')
    >>> act.shape
    (281, 88)
    >>> act  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    array([[-0.00014,  0.0002 , ..., -0.     ,  0.     ],
           [ 0.00008,  0.0001 , ...,  0.00006, -0.00001],
           ...,
           [-0.00005, -0.00011, ...,  0.00005, -0.00001],
           [-0.00017,  0.00002, ...,  0.00009, -0.00009]], dtype=float32)

    """

    def __init__(self, **kwargs):
        # pylint: disable=unused-argument
        from ..audio.signal import SignalProcessor, FramedSignalProcessor
        from ..audio.stft import ShortTimeFourierTransformProcessor
        from ..audio.filters import FilterbankProcessor, LogarithmicFilterbank
        from ..audio.spectrogram import (ScalingProcessor,
                                         SpectrogramDifferenceProcessor)
        from ..models import NOTES_BRNN
        from ..ml.nn import NeuralNetwork
        # set parameters
        kwargs['sample_rate'] = 44100
        kwargs['num_channels'] = 1
        kwargs['fps'] = 100
        # define pre-processing chain
        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        # process the multi-resolution spec & diff in parallel
        multi = ParallelProcessor([])
        for frame_size, diff_frame in zip([1024, 2048, 4096], [1, 1, 2]):
            frames = FramedSignalProcessor(frame_size=frame_size, **kwargs)
            filt = FilterbankProcessor(LogarithmicFilterbank,
                                       num_bands=12, fmin=30, fmax=17000,
                                       norm_filters=True,
                                       frame_size=frame_size, **kwargs)
            stft = ShortTimeFourierTransformProcessor(filterbank=filt)
            log = ScalingProcessor(scaling_fn=np.log10, mul=5, add=1)
            diff = SpectrogramDifferenceProcessor(diff_frames=diff_frame,
                                                  positive_diffs=True,
                                                  stack_diffs=np.hstack)
            # process each frame size with spec and diff sequentially
            multi.append(SequentialProcessor((frames, stft, log, diff)))
        # stack the features and processes everything sequentially
        pre_processor = SequentialProcessor((sig, multi, np.hstack))

        # process the pre-processed signal with a NN
        nn = NeuralNetwork.load(NOTES_BRNN[0])

        # instantiate a SequentialProcessor
        super(RNNPianoNoteProcessor, self).__init__((pre_processor, nn))


class NoteOnsetPeakPickingProcessor(OnsetPeakPickingProcessor):
    """
    This class implements the note onset peak-picking functionality.

    Parameters
    ----------
    threshold : float
        Threshold for peak-picking.
    smooth : float, optional
        Smooth the activation function over `smooth` seconds.
    pre_avg : float, optional
        Use `pre_avg` seconds past information for moving average.
    post_avg : float, optional
        Use `post_avg` seconds future information for moving average.
    pre_max : float, optional
        Use `pre_max` seconds past information for moving maximum.
    post_max : float, optional
        Use `post_max` seconds future information for moving maximum.
    combine : float, optional
        Only report one note per pitch within `combine` seconds.
    delay : float, optional
        Report the detected notes `delay` seconds delayed.
    online : bool, optional
        Use online peak-picking, i.e. no future information.
    fps : float, optional
        Frames per second used for conversion of timings.

    Returns
    -------
    notes : numpy array
        Detected notes [seconds, pitch].

    Notes
    -----
    If no moving average is needed (e.g. the activations are independent of
    the signal's level as for neural network activations), `pre_avg` and
    `post_avg` should be set to 0.
    For peak picking of local maxima, set `pre_max` >= 1. / `fps` and
    `post_max` >= 1. / `fps`.
    For online peak picking, all `post_` parameters are set to 0.

    Examples
    --------
    Create a NoteOnsetPeakPickingProcessor. The returned array represents the
    positions of the onsets in seconds, thus the frame rate has to be given.
    To obtain piano MIDI note numbers, the pitch offset must be set to 21.

    >>> proc = NoteOnsetPeakPickingProcessor(fps=100, pitch_offset=21)
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.notes.NoteOnsetPeakPickingProcessor object at 0x...>

    Call this NoteOnsetPeakPickingProcessor with the note activations from an
    RNNPianoNoteProcessor.

    >>> act = RNNPianoNoteProcessor()('tests/data/audio/stereo_sample.wav')
    >>> proc(act)  # doctest: +ELLIPSIS
    array([[ 0.14, 72.  ],
           [ 1.56, 41.  ],
           [ 3.37, 75.  ]])

    """
    THRESHOLD = 0.5  # binary threshold
    SMOOTH = 0.
    PRE_AVG = 0.
    POST_AVG = 0.
    PRE_MAX = 0.
    POST_MAX = 0.
    COMBINE = 0.03
    DELAY = 0.

    def __init__(self, threshold=THRESHOLD, smooth=SMOOTH, pre_avg=PRE_AVG,
                 post_avg=POST_AVG, pre_max=PRE_MAX, post_max=POST_MAX,
                 combine=COMBINE, delay=DELAY, fps=None, pitch_offset=0,
                 **kwargs):
        # pylint: disable=unused-argument
        super(NoteOnsetPeakPickingProcessor, self).__init__(
            threshold=threshold, smooth=smooth, pre_avg=pre_avg,
            post_avg=post_avg, pre_max=pre_max, post_max=post_max,
            combine=combine, delay=delay, fps=fps)
        self.pitch_offset = pitch_offset

    def process(self, activations, **kwargs):
        """
        Detect the notes in the given activation function.

        Parameters
        ----------
        activations : numpy array
            Note activation function.

        Returns
        -------
        onsets : numpy array
            Detected notes [seconds, pitches].

        """
        # convert timing information to frames and set default values
        # TODO: use at least 1 frame if any of these values are > 0?
        timings = np.array([self.smooth, self.pre_avg, self.post_avg,
                            self.pre_max, self.post_max]) * self.fps
        timings = np.round(timings).astype(int)
        # detect the peaks (function returns int indices)
        onsets, pitches = peak_picking(activations, self.threshold, *timings)
        # if no note onsets are detected, return empty array
        if not onsets.any():
            return np.empty((0, 2))
        # convert onset timing and apply pitch offset
        onsets = onsets.astype(np.float) / self.fps
        pitches += self.pitch_offset
        # shift if necessary
        if self.delay:
            onsets += self.delay
        # combine notes
        if self.combine > 0:
            notes = []
            # iterate over each detected note pitch separately
            for pitch in np.unique(pitches):
                # get all onsets for this pitch
                onsets_ = onsets[pitches == pitch]
                # combine onsets
                onsets_ = combine_events(onsets_, self.combine, 'left')
                # zip onsets and pitches and add them to list of detections
                notes.extend(list(zip(onsets_, [pitch] * len(onsets_))))
        else:
            # just zip all detected notes
            notes = list(zip(onsets, pitches))
        # sort the detections and return as numpy array
        return np.array(sorted(notes))


class NotePeakPickingProcessor(NoteOnsetPeakPickingProcessor):
    """
    Deprecated as of version 0.17. Will be removed in version 0.19. Use
    :class:`NoteOnsetPeakPickingProcessor` instead and set `fps` and
    `pitch_offset` accordingly.

    """

    def __init__(self, fps=100, pitch_offset=21, **kwargs):
        # pylint: disable=unused-argument
        super(NotePeakPickingProcessor, self).__init__(
            fps=fps, pitch_offset=pitch_offset, **kwargs)


def _cnn_pad(data):
    """Pad the data by repeating the first and last frame 5 times."""
    pad_start = np.repeat(data[:1], 5, axis=0)
    pad_stop = np.repeat(data[-1:], 5, axis=0)
    return np.concatenate((pad_start, data, pad_stop))


class CNNPianoNoteProcessor(SequentialProcessor):
    """
    Processor to get piano note activations from a CNN in a multi-task fashion
    which simultaneously detects onsets and intermediate note features.

    References
    ----------

    .. [1] Rainer Kelz, Sebastian Böck and Gerhard Widmer,
           "Deep Polyphonic ADSR Piano Note Transcription",
           Proceedings of the 44th International Conference on Acoustics,
           Speech and Signal Processing (ICASSP), 2019.

    Examples
    --------
    Create a CNNPianoNoteProcessor and pass a file through the processor
    to obtain a note activation function (sampled with 50 frames per second).

    >>> proc = CNNPianoNoteProcessor()
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.notes.CNNPianoNoteProcessor object at 0x...>
    >>> act = proc('tests/data/audio/stereo_sample.wav')

    The activations are returned as a 3-dimensional array, the first axis
    representing time, the second the MIDI notes, and the third dimension
    contains the (sounding) note and onset activations (first and second value,
    respectively).

    >>> act.shape
    (208, 88, 3)

    Sounding notes,

    >>> act[..., 0]  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    array([[0., 0., ..., 0., 0.],
           [0., 0., ..., 0., 0.],
           ...,
           [0., 0., ..., 0., 0.],
           [0., 0., ..., 0., 0.]], dtype=float32)

    and onset activations.

    >>> act[..., 1]  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    array([[0.     , 0.00001, ..., 0.     , 0.     ],
           [0.     , 0.00002, ..., 0.     , 0.     ],
           ...,
           [0.     , 0.     , ..., 0.     , 0.     ],
           [0.     , 0.     , ..., 0.     , 0.     ]], dtype=float32)

    """

    def __init__(self, **kwargs):
        from ..audio.signal import SignalProcessor, FramedSignalProcessor
        from ..audio.stft import ShortTimeFourierTransformProcessor
        from ..audio.spectrogram import (FilteredSpectrogramProcessor,
                                         LogarithmicSpectrogramProcessor)
        from ..models import NOTES_CNN
        from ..ml.nn import NeuralNetworkEnsemble
        # define pre-processing chain
        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        frames = FramedSignalProcessor(frame_size=4096, fps=50)
        stft = ShortTimeFourierTransformProcessor()  # caching FFT window
        filt = FilteredSpectrogramProcessor(num_bands=24, fmin=30, fmax=10000)
        spec = LogarithmicSpectrogramProcessor(add=1)
        # pre-processes everything sequentially
        pre_processor = SequentialProcessor(
            (sig, frames, stft, filt, spec, _cnn_pad))
        # process the pre-processed signal with a NN
        nn = NeuralNetworkEnsemble.load(NOTES_CNN)
        # instantiate a SequentialProcessor
        super(CNNPianoNoteProcessor, self).__init__((pre_processor, nn))


class ADSRNoteTrackingProcessor(Processor):
    """
    Track the notes with an HMM based on a model of attack, decay, sustain,
    release (ADSR) envelopes.

    Parameters
    ----------
    onset_prob : float, optional
        Transition probability to enter an onset state.
    note_prob : float, optional
        Transition probability to enter a sounding note state.
    offset_prob : float, optional
        Transition probability to enter an offset state.
    attack_length : float, optional
        Minimum required attack (i.e. onset activation required) length.
    decay_length : float, optional
        Minimum required decay (i.e. note activation required) length.
    release_length : float, optional
        Minimum required release (i.e. note activation required) length.
    complete : bool, optional
        Require notes to transition all states (i.e. discard incomplete notes).
    onset_threshold : float, optional
        Require notes to have an onset activation greater or equal this
        threshold.
    note_threshold : float, optional
        Require notes to have a note activation greater equal this threshold.
    fps : float, optional
        Frames per second.
    pitch_offset : int, optional
        Pitch offset for the detected notes.

    References
    ----------

    .. [1] Rainer Kelz, Sebastian Böck and Gerhard Widmer,
           "Deep Polyphonic ADSR Piano Note Transcription",
           Proceedings of the 44th International Conference on Acoustics,
           Speech and Signal Processing (ICASSP), 2019.

    Examples
    --------
    Create a CNNPianoNoteProcessor and pass a file through the processor
    to obtain a note activation function (sampled with 50 frames per second).

    >>> proc = CNNPianoNoteProcessor()
    >>> act = proc('tests/data/audio/stereo_sample.wav')

    Track the notes by means on ADSR note tracking:
    >>> adsr = ADSRNoteTrackingProcessor()
    >>> adsr(act)  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.12, 72. , 1.44],
           [ 1.54, 41. , 1.84],
           [ 2.5 , 77. , 1.  ],
           [ 2.52, 65. , 0.96],
           [ 2.54, 60. , 0.82],
           [ 2.58, 56. , 0.82],
           [ 3.34, 75. , 0.82],
           [ 3.42, 43. , 0.74]])
    """

    def __init__(self, onset_prob=0.8, note_prob=0.8, offset_prob=0.5,
                 attack_length=0.04, decay_length=0.04, release_length=0.02,
                 complete=True, onset_threshold=0.5, note_threshold=0.5,
                 fps=50, pitch_offset=21, **kwargs):
        from .notes_hmm import (ADSRStateSpace, ADSRTransitionModel,
                                ADSRObservationModel)
        from ..ml.hmm import HiddenMarkovModel
        # state space
        self.st = ADSRStateSpace(attack_length=int(attack_length * fps),
                                 decay_length=int(decay_length * fps),
                                 release_length=int(release_length * fps))
        # transition model
        self.tm = ADSRTransitionModel(self.st, onset_prob=onset_prob,
                                      note_prob=note_prob,
                                      offset_prob=offset_prob)
        # observation model
        self.om = ADSRObservationModel(self.st)
        # instantiate a HMM
        self.hmm = HiddenMarkovModel(self.tm, self.om, None)
        # save variables
        self.complete = complete
        self.onset_threshold = onset_threshold
        self.note_threshold = note_threshold
        self.pitch_offset = pitch_offset
        self.fps = fps

    def process(self, activations, **kwargs):
        """
        Detect the notes in the given activation function.

        Parameters
        ----------
        activations : numpy array
            Combined note and onset activation function.

        Returns
        -------
        notes : numpy array
            Detected notes [seconds, pitches, duration].

        """
        notes = []
        note_path = np.arange(self.st.attack, self.st.release)
        # process each pitch individually
        for pitch in range(activations.shape[1]):
            # decode activations for this pitch with HMM
            with np.errstate(divide='ignore'):
                # ignore warnings when taking the log of 0
                path, _ = self.hmm.viterbi(activations[:, pitch, :])
            # extract HMM note segments
            segments = np.logical_and(path > self.st.attack,
                                      path < self.st.release)
            # extract start and end positions (transition points)
            idx = np.nonzero(np.diff(segments.astype(np.int)))[0]
            # add end if needed
            if len(idx) % 2 != 0:
                idx = np.append(idx, [len(activations)])
            # all sounding frames
            frames = activations[:, pitch, 0]
            # all frames with onset activations
            onsets = activations[:, pitch, 1]
            # iterate over all segments to decide which to keep
            for onset, offset in idx.reshape((-1, 2)):
                # extract note segment
                segment = path[onset:offset]
                # discard segment which do not contain the complete note path
                if self.complete and np.setdiff1d(note_path, segment).any():
                    continue
                # discard segments without a real note
                if frames[onset:offset].max() < self.note_threshold:
                    continue
                # discard segments without a real onset
                if onsets[onset:offset].max() < self.onset_threshold:
                    continue
                # append segment as note
                notes.append([onset / self.fps, pitch + self.pitch_offset,
                              (offset - onset) / self.fps])
        # if no note notes are detected, return empty array
        if len(notes) == 0:
            return np.empty((0, 3))
        # sort the notes, convert timing information and return them
        return np.array(sorted(notes), ndmin=2)
