# encoding: utf-8
# pylint: disable=invalid-name
"""
This module contains drum transcription related functionality.

"""

from __future__ import absolute_import, division, print_function

import numpy as np

from .notes import NotePeakPickingProcessor
from ..processors import SequentialProcessor


def _crnn_drum_processor_pad(data):
    """
    Pad the data by repeating the first and last frame 8 times.

    Parameters
    ----------
    data: numpy array
        Input data.

    Returns
    -------
    numpy array
        Padded data.

    """
    pad_start = np.repeat(data[:1], 4, axis=0)
    pad_stop = np.repeat(data[-1:], 4, axis=0)
    return np.concatenate((pad_start, data, pad_stop))


def _crnn_drum_processor_stack(data):
    """
    Stacks a row of zeros between the spctrogram and the differences.

    Parameters
    ----------
    data : tuple
        Two numpy arrays (spectrogram, differences).

    Returns
    -------
    numpy array
        Stacked input with 0's in between.

    """
    return np.hstack((data[0], np.zeros((data[0].shape[0], 1)), data[1]))


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
        from ..ml.nn import NeuralNetworkEnsemble
        from ..models import DRUMS_CRNN
        # signal processing chain
        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        frames = FramedSignalProcessor(frame_size=2048, fps=100)
        stft = ShortTimeFourierTransformProcessor()  # caching FFT window
        spec = LogarithmicFilteredSpectrogramProcessor(
            num_channels=1, sample_rate=44100,
            filterbank=LogarithmicFilterbank, frame_size=2048, fps=100,
            num_bands=12, fmin=20, fmax=20000, norm_filters=True)
        diff = SpectrogramDifferenceProcessor(
            diff_ratio=0.5, positive_diffs=True,
            stack_diffs=_crnn_drum_processor_stack)
        # process input data
        pre_processor = SequentialProcessor(
            (sig, frames, stft, spec, diff, _crnn_drum_processor_pad))
        # process with a NN
        nn = NeuralNetworkEnsemble.load(DRUMS_CRNN)
        # instantiate a SequentialProcessor
        super(CRNNDrumProcessor, self).__init__((pre_processor, nn))


class DrumPeakPickingProcessor(NotePeakPickingProcessor):
    """
    This class implements the drum peak-picking functionality.

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
        Only report one drum hit per instrument within `combine` seconds.
    delay : float, optional
        Report the detected drums `delay` seconds delayed.
    online : bool, optional
        Use online peak-picking, i.e. no future information.
    fps : float, optional
        Frames per second used for conversion of timings.

    Returns
    -------
    drums : numpy array
        Detected drums [seconds, pitch].

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
    Create a DrumPeakPickingProcessor. The returned array represents the note
    positions in seconds, thus the expected sampling rate has to be given.

    >>> proc = DrumPeakPickingProcessor(fps=100)
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.drums.DrumPeakPickingProcessor object at 0x...>

    Call this DrumPeakPickingProcessor with the drum activations from a
    CRNNDrumProcessor.

    >>> act = CRNNDrumProcessor()('tests/data/audio/stereo_sample.wav')
    >>> proc(act)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    array([[0.13, 0.],
          [0.13, 2.],
          [0.48, 2.],
          [0.65, 0.],
          [0.8, 0.],
          [1.16, 0.],
          [1.16, 2.],
          [1.52, 0.],
          [1.66, 1.],
          [1.84, 0.],
          [1.84, 2.],
          [2.18, 1.],
          [2.7, 0.]])

    """

    pitch_offset = 0
