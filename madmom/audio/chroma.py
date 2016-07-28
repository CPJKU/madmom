# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains chroma related functionality.

"""

from __future__ import absolute_import, division, print_function

import numpy as np

from madmom.processors import SequentialProcessor


def _dcp_flatten(fs):
    """Flatten spectrograms for DeepChromaProcessor. Needs to be outside
       of the class in order to be picklable for multiprocessing.
    """
    return np.concatenate(fs).reshape(len(fs), -1)


class DeepChromaProcessor(SequentialProcessor):
    """
    Compute chroma vectors from an audio file using a deep neural network
    that focuses on harmonically relevant spectral content.

    Parameters
    ----------
    fmin : int, optional
        Minimum frequency of the filterbank [Hz].
    fmax : float, optional
        Maximum frequency of the filterbank [Hz].
    unique_filters : bool, optional
        Indicate if the filterbank should contain only unique filters, i.e.
        remove duplicate filters resulting from insufficient resolution at
        low frequencies.
    models : list of filenames, optional
        List of model filenames.

    Notes
    -----
    Provided model files must be compatible with the processing pipeline and
    the values of `fmin`, `fmax`, and `unique_filters`. The
    general use case for the `models` parameter is to use a specific
    model instead of an ensemble of all models.

    The models shipped with madmom differ slightly from those presented in the
    paper (less hidden units, narrower frequency band for spectrogram), but
    achieve similar results.

    References
    ----------
    .. [1] Filip Korzeniowski and Gerhard Widmer,
           "Feature Learning for Chord Recognition: The Deep Chroma Extractor",
           Proceedings of the 17th International Society for Music Information
           Retrieval Conference (ISMIR), 2016.

    Examples
    --------
    Extract a chroma vector using the deep chroma extractor:

    >>> dcp = DeepChromaProcessor()
    >>> chroma = dcp('tests/data/audio/sample2.wav')
    >>> chroma  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    array([[ 0.01317,  0.00721,  ...,  0.00546,  0.00943],
           [ 0.36809,  0.01314,  ...,  0.02213,  0.01838],
           ...,
           [ 0.1534 ,  0.06475,  ...,  0.00896,  0.05789],
           [ 0.17513,  0.0729 ,  ...,  0.00945,  0.06913]], dtype=float32)
    >>> chroma.shape
    (41, 12)

    """

    def __init__(self, fmin=65, fmax=2100, unique_filters=True, models=None,
                 **kwargs):
        from ..models import CHROMA_DNN
        from ..audio.signal import SignalProcessor, FramedSignalProcessor
        from ..audio.spectrogram import LogarithmicFilteredSpectrogramProcessor
        from madmom.ml.nn import NeuralNetworkEnsemble

        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        frames = FramedSignalProcessor(frame_size=8192, fps=10)
        spec = LogarithmicFilteredSpectrogramProcessor(
            num_bands=24, fmin=fmin, fmax=fmax, unique_filters=unique_filters)
        spec_frames = FramedSignalProcessor(frame_size=15, hop_size=1)

        nn = NeuralNetworkEnsemble.load(models or CHROMA_DNN)

        super(DeepChromaProcessor, self).__init__([
            sig, frames, spec, spec_frames, _dcp_flatten, nn
        ])
