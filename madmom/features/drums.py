from ..processors import SequentialProcessor
from .notes import NotePeakPickingProcessor, RNNPianoNoteProcessor

import numpy as np


def _crnn_drum_processor_pad(data):
    """
    Pad the data by repeating the first and last frame 8 times.

    Parameters
    ----------
    numpy array

    Returns
    -------
    padded data

    """
    pad_start = np.repeat(data[:1], 8, axis=0)
    pad_stop = np.repeat(data[-1:], 8, axis=0)
    return np.concatenate((pad_start, data, pad_stop))


def _crnn_drum_processor_zero_pad(tup):
    """
    stacks 0's between the input data

    Parameters
    ----------
    tup : tuple of 2 numpy arrays

    Returns
    -------
    the stacked input with 0' in between

    """
    data1, data2 = tup
    return np.hstack((data1, np.zeros((data1.shape[0], 1)), data2))


class CRNNDrumProcessor(SequentialProcessor):
    """

    """

    def __init__(self, **kwargs):
        from ..audio.spectrogram import (
            LogarithmicFilteredSpectrogramProcessor,
            SpectrogramDifferenceProcessor)
        from ..audio.filters import LogarithmicFilterbank
        from ..audio.signal import SignalProcessor, FramedSignalProcessor
        from ..audio.stft import ShortTimeFourierTransformProcessor
        from ..ml.nn import NeuralNetwork
        # TODO: parse models, add to folder, relative import
        from madmom.workspace.LSTM import DRUM_CRNN

        # choose the appropriate models
        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        frames = FramedSignalProcessor(frame_size=2048, fps=100)
        stft = ShortTimeFourierTransformProcessor()  # caching FFT window
        spec = LogarithmicFilteredSpectrogramProcessor(
            num_channels=1, sample_rate=44100,
            filterbank=LogarithmicFilterbank, frame_size=2048,
            fps=100, num_bands=12,
            fmin=20, fmax=20000,
            norm_filters=True)
        diff = SpectrogramDifferenceProcessor(
            diff_ratio=0.5, positive_diffs=True,
            stack_diffs=_crnn_drum_processor_zero_pad)
        pad = _crnn_drum_processor_pad

        pre_processor = SequentialProcessor(
            (sig, frames, stft, spec, diff, pad))

        nn = NeuralNetwork.load(DRUM_CRNN[0])

        # instantiate a SequentialProcessor
        super(CRNNDrumProcessor, self).__init__((pre_processor, nn))


class DrumPeakPickingProcessor(NotePeakPickingProcessor):
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
    Create a NotePeakPickingProcessor. The returned array represents the note
    positions in seconds, thus the expected sampling rate has to be given.

    >>> proc = DrumPeakPickingProcessor(fps=100)
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.drums.DrumPeakPickingProcessor object at 0x...>

    Call this NotePeakPickingProcessor with the note activations from an
    RNNPianoNoteProcessor.

    >>> act = RNNPianoNoteProcessor()('tests/data/audio/stereo_sample.wav')
    >>> proc(act)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    array([[  0.14,  51.  ],
           [  1.56,  20.  ],
           [  3.37,  54.  ]])

    """

    # TODO: Change docstring
    FPS = 100
    THRESHOLD = 0.15  # binary threshold
    SMOOTH = 0.
    PRE_AVG = 0.10
    POST_AVG = 0.01
    PRE_MAX = 0.02
    POST_MAX = 0.01
    COMBINE = 0.02
    DELAY = 0.
    ONLINE = False
    pitch_offset = 0
