# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains key recognition related functionality.

"""

import numpy as np

from ..processors import SequentialProcessor


KEY_LABELS = ['A major', 'Bb major', 'B major', 'C major', 'Db major',
              'D major', 'Eb major', 'E major', 'F major', 'F# major',
              'G major', 'Ab major', 'A minor', 'Bb minor', 'B minor',
              'C minor', 'C# minor', 'D minor', 'D# minor', 'E minor',
              'F minor', 'F# minor', 'G minor', 'G# minor']


def key_prediction_to_label(prediction):
    """
    Convert key class id to a human-readable key name.

    Parameters
    ----------
    prediction : numpy array
        Array containing the probabilities of each key class.

    Returns
    -------
    str
        Human-readable key name.

    """
    prediction = np.atleast_2d(prediction)
    return KEY_LABELS[prediction[0].argmax()]


class CNNKeyRecognitionProcessor(SequentialProcessor):
    """
    Recognise the global key of a musical piece using a Convolutional Neural
    Network as described in [1]_.

    Parameters
    ----------
    nn_files : list, optional
        List with trained CNN model files. Per default ('None'), an ensemble
        of networks will be used.
    single_net : bool, optional
        Use only a single CNN for prediction. This speeds up processing, but
        slightly worsens the results.

    References
    ----------
    .. [1] Filip Korzeniowski and Gerhard Widmer,
           "End-to-End Musical Key Estimation Using a Convolutional Neural
           Network", In Proceedings of the 25th European Signal Processing
           Conference (EUSIPCO), Kos, Greece, 2017.

    Examples
    --------
    Create a CNNKeyRecognitionProcessor and pass a file through it.
    The returned array represents the probability of each key class.

    >>> proc = CNNKeyRecognitionProcessor()
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.key.CNNKeyRecognitionProcessor object at 0x...>
    >>> proc('tests/data/audio/sample.wav')  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.     ,  0.     ,  0.00001,  0.00012,  0.     ,  0.     ,
             0.00151,  0.     ,  0.     ,  0.     ,  0.00003,  0.81958,
             0.     ,  0.     ,  0.     ,  0.01747,  0.     ,  0.     ,
             0.00001,  0.     ,  0.00006,  0.     ,  0.00001,  0.16119]],
          dtype=float32)

    """

    def __init__(self, nn_files=None, **kwargs):
        from ..audio.signal import SignalProcessor, FramedSignalProcessor
        from ..audio.stft import ShortTimeFourierTransformProcessor
        from ..audio.spectrogram import LogarithmicFilteredSpectrogramProcessor
        from ..ml.nn import NeuralNetworkEnsemble
        from ..models import KEY_CNN

        # spectrogram computation
        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        frames = FramedSignalProcessor(frame_size=8192, fps=5)
        stft = ShortTimeFourierTransformProcessor()  # caching FFT window
        spec = LogarithmicFilteredSpectrogramProcessor(
            num_bands=24, fmin=65, fmax=2100, unique_filters=True
        )

        # neural network
        nn_files = nn_files or KEY_CNN
        nn = NeuralNetworkEnsemble.load(nn_files)

        # create processing pipeline
        super(CNNKeyRecognitionProcessor, self).__init__([
            sig, frames, stft, spec, nn
        ])
