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
from ..processors import ParallelProcessor, SequentialProcessor
from ..utils import combine_events


# class for detecting notes with a RNN
class RNNPianoNoteProcessor(SequentialProcessor):
    """
    Processor to get a (piano) note activation function from a RNN.

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
        from ..audio.spectrogram import (
            FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
            SpectrogramDifferenceProcessor)
        from ..models import NOTES_BRNN
        from ..ml.nn import NeuralNetwork

        # define pre-processing chain
        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        # process the multi-resolution spec & diff in parallel
        multi = ParallelProcessor([])
        for frame_size in [1024, 2048, 4096]:
            frames = FramedSignalProcessor(frame_size=frame_size, fps=100)
            stft = ShortTimeFourierTransformProcessor()  # caching FFT window
            filt = FilteredSpectrogramProcessor(
                num_bands=12, fmin=30, fmax=17000, norm_filters=True)
            spec = LogarithmicSpectrogramProcessor(mul=5, add=1)
            diff = SpectrogramDifferenceProcessor(
                diff_ratio=0.5, positive_diffs=True, stack_diffs=np.hstack)
            # process each frame size with spec and diff sequentially
            multi.append(SequentialProcessor((frames, stft, filt, spec, diff)))
        # stack the features and processes everything sequentially
        pre_processor = SequentialProcessor((sig, multi, np.hstack))

        # process the pre-processed signal with a NN
        nn = NeuralNetwork.load(NOTES_BRNN[0])

        # instantiate a SequentialProcessor
        super(RNNPianoNoteProcessor, self).__init__((pre_processor, nn))


class NotePeakPickingProcessor(OnsetPeakPickingProcessor):
    """
    This class implements the note peak-picking functionality.

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
    Create a PeakPickingProcessor. The returned array represents the positions
    of the onsets in seconds, thus the expected sampling rate has to be given.

    >>> proc = NotePeakPickingProcessor(fps=100)
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.notes.NotePeakPickingProcessor object at 0x...>

    Call this NotePeakPickingProcessor with the note activations from an
    RNNPianoNoteProcessor.

    >>> act = RNNPianoNoteProcessor()('tests/data/audio/stereo_sample.wav')
    >>> proc(act)  # doctest: +ELLIPSIS
    array([ 0.09,  0.29,  0.45,  ...,  2.34,  2.49,  2.67])

    """
    FPS = 100
    THRESHOLD = 0.5  # binary threshold
    SMOOTH = 0.
    PRE_AVG = 0.
    POST_AVG = 0.
    PRE_MAX = 0.
    POST_MAX = 0.
    COMBINE = 0.03
    DELAY = 0.
    ONLINE = False

    def __init__(self, threshold=THRESHOLD, smooth=SMOOTH, pre_avg=PRE_AVG,
                 post_avg=POST_AVG, pre_max=PRE_MAX, post_max=POST_MAX,
                 combine=COMBINE, delay=DELAY, online=ONLINE, fps=FPS,
                 **kwargs):
        # pylint: disable=unused-argument
        super(NotePeakPickingProcessor, self).__init__(
            threshold=threshold, smooth=smooth, pre_avg=pre_avg,
            post_avg=post_avg, pre_max=pre_max, post_max=post_max,
            combine=combine, delay=delay, online=online, fps=fps)

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
        notes = peak_picking(activations, self.threshold, *timings)
        # split onsets and pitches
        onsets = notes[0].astype(np.float) / self.fps
        pitches = notes[1] + 21
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
        return np.asarray(sorted(notes))
