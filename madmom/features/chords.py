# encoding: utf-8
"""
This module contains chord recognition related functionality.

"""
from __future__ import absolute_import, division, print_function

from functools import partial

import numpy as np

from ..io import SEGMENT_DTYPE
from ..processors import SequentialProcessor


def majmin_targets_to_chord_labels(targets, fps):
    """
    Converts a series of major/minor chord targets to human readable chord
    labels. Targets are assumed to be spaced equidistant in time as defined
    by the `fps` parameter (each target represents one 'frame').

    Ids 0-11 encode major chords starting with root 'A', 12-23 minor chords.
    Id 24 represents 'N', the no-chord class.

    Parameters
    ----------
    targets : iterable
        Iterable containing chord class ids.
    fps : float
        Frames per second. Consecutive class

    Returns
    -------
    chord labels : list
        List of tuples of the form (start time, end time, chord label)

    """
    # create a map of semitone index to semitone name (e.g. 0 -> A, 1 -> A#)
    pitch_class_to_label = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F',
                            'F#', 'G', 'G#']

    def pred_to_cl(pred):
        """
        Map a class id to a chord label.
        0..11 major chords, 12..23 minor chords, 24 no chord
        """
        if pred == 24:
            return 'N'
        return '{}:{}'.format(pitch_class_to_label[pred % 12],
                              'maj' if pred < 12 else 'min')

    # get labels per frame
    spf = 1. / fps
    labels = [(i * spf, pred_to_cl(p)) for i, p in enumerate(targets)]

    # join same consecutive predictions
    prev_label = (None, None)
    uniq_labels = []

    for label in labels:
        if label[1] != prev_label[1]:
            uniq_labels.append(label)
            prev_label = label

    # end time of last label is one frame duration after
    # the last prediction time
    start_times, chord_labels = zip(*uniq_labels)
    end_times = start_times[1:] + (labels[-1][0] + spf,)

    return np.array(list(zip(start_times, end_times, chord_labels)),
                    dtype=SEGMENT_DTYPE)


class DeepChromaChordRecognitionProcessor(SequentialProcessor):
    """
    Recognise major and minor chords from deep chroma vectors [1]_ using a
    Conditional Random Field.

    Parameters
    ----------
    model : str
        File containing the CRF model. If None, use the model supplied with
        madmom.
    fps : float
        Frames per second. Must correspond to the fps of the incoming
        activations and the model.

    References
    ----------
    .. [1] Filip Korzeniowski and Gerhard Widmer,
           "Feature Learning for Chord Recognition: The Deep Chroma Extractor",
           Proceedings of the 17th International Society for Music Information
           Retrieval Conference (ISMIR), 2016.

    Examples
    --------
    To recognise chords in an audio file using the
    DeepChromaChordRecognitionProcessor you first need to create a
    madmom.audio.chroma.DeepChromaProcessor to extract the appropriate chroma
    vectors.

    >>> from madmom.audio.chroma import DeepChromaProcessor
    >>> dcp = DeepChromaProcessor()
    >>> dcp  # doctest: +ELLIPSIS
    <madmom.audio.chroma.DeepChromaProcessor object at ...>

    Then, create the DeepChromaChordRecognitionProcessor to decode a chord
    sequence from the extracted chromas:

    >>> decode = DeepChromaChordRecognitionProcessor()
    >>> decode  # doctest: +ELLIPSIS
    <madmom.features.chords.DeepChromaChordRecognitionProcessor object at ...>

    To transcribe the chords, you can either manually call the processors
    one after another,

    >>> chroma = dcp('tests/data/audio/sample2.wav')
    >>> decode(chroma)
    ... # doctest: +NORMALIZE_WHITESPACE +NORMALIZE_ARRAYS
    array([(0. , 1.6, 'F:maj'), (1.6, 2.5, 'A:maj'), (2.5, 4.1, 'D:maj')],
          dtype=[('start', '<f8'), ('end', '<f8'), ('label', 'O')])

    or create a `SequentialProcessor` that connects them:

    >>> from madmom.processors import SequentialProcessor
    >>> chordrec = SequentialProcessor([dcp, decode])
    >>> chordrec('tests/data/audio/sample2.wav')
    ... # doctest: +NORMALIZE_WHITESPACE +NORMALIZE_ARRAYS
    array([(0. , 1.6, 'F:maj'), (1.6, 2.5, 'A:maj'), (2.5, 4.1, 'D:maj')],
          dtype=[('start', '<f8'), ('end', '<f8'), ('label', 'O')])
    """

    def __init__(self, model=None, fps=10, **kwargs):
        from ..ml.crf import ConditionalRandomField
        from ..models import CHORDS_DCCRF
        crf = ConditionalRandomField.load(model or CHORDS_DCCRF[0])
        lbl = partial(majmin_targets_to_chord_labels, fps=fps)
        super(DeepChromaChordRecognitionProcessor, self).__init__((crf, lbl))


# functions necessary for CNNChordFeatureProcessor - they need to
# be outside of the class so the processor stays picklable
def _cnncfp_pad(data):
    """Pad the input"""
    pad_data = np.zeros((11, 113))
    return np.vstack([pad_data, data, pad_data])


def _cnncfp_superframes(data):
    """Segment input into superframes"""
    from ..utils import segment_axis
    return segment_axis(data, 3, 1, axis=0)


def _cnncfp_avg(data):
    """Global average pool"""
    return data.mean((1, 2))


class CNNChordFeatureProcessor(SequentialProcessor):
    """
    Extract learned features for chord recognition, as described in [1]_.

    References
    ----------
    .. [1] Filip Korzeniowski and Gerhard Widmer,
           "A Fully Convolutional Deep Auditory Model for Musical Chord
           Recognition",
           Proceedings of IEEE International Workshop on Machine Learning for
           Signal Processing (MLSP), 2016.

    Examples
    --------
    >>> proc = CNNChordFeatureProcessor()
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.chords.CNNChordFeatureProcessor object at 0x...>
    >>> features = proc('tests/data/audio/sample2.wav')
    >>> features.shape
    (41, 128)
    >>> features # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    array([[0.05798, 0.     , ..., 0.02757, 0.014  ],
           [0.06604, 0.     , ..., 0.02898, 0.00886],
           ...,
           [0.00655, 0.1166 , ..., 0.00651, 0.     ],
           [0.01476, 0.11185, ..., 0.00287, 0.     ]])
    """

    def __init__(self, **kwargs):
        from ..audio.signal import SignalProcessor, FramedSignalProcessor
        from ..audio.stft import ShortTimeFourierTransformProcessor
        from ..audio.spectrogram import LogarithmicFilteredSpectrogramProcessor
        from ..ml.nn import NeuralNetwork
        from ..models import CHORDS_CNN_FEAT

        # spectrogram computation
        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        frames = FramedSignalProcessor(frame_size=8192, fps=10)
        stft = ShortTimeFourierTransformProcessor()  # caching FFT window
        spec = LogarithmicFilteredSpectrogramProcessor(
            num_bands=24, fmin=60, fmax=2600, unique_filters=True
        )

        # padding, neural network and global average pooling
        pad = _cnncfp_pad
        nn = NeuralNetwork.load(CHORDS_CNN_FEAT[0])
        superframes = _cnncfp_superframes
        avg = _cnncfp_avg

        # create processing pipeline
        super(CNNChordFeatureProcessor, self).__init__([
            sig, frames, stft, spec, pad, nn, superframes, avg
        ])


class CRFChordRecognitionProcessor(SequentialProcessor):
    """
    Recognise major and minor chords from learned features extracted by
    a convolutional neural network, as described in [1]_.

    Parameters
    ----------
    model : str
        File containing the CRF model. If None, use the model supplied with
        madmom.
    fps : float
        Frames per second. Must correspond to the fps of the incoming
        activations and the model.

    References
    ----------
    .. [1] Filip Korzeniowski and Gerhard Widmer,
           "A Fully Convolutional Deep Auditory Model for Musical Chord
           Recognition",
           Proceedings of IEEE International Workshop on Machine Learning for
           Signal Processing (MLSP), 2016.

    Examples
    --------
    To recognise chords using the CRFChordRecognitionProcessor, you first need
    to extract features using the CNNChordFeatureProcessor.

    >>> featproc = CNNChordFeatureProcessor()
    >>> featproc  # doctest: +ELLIPSIS
    <madmom.features.chords.CNNChordFeatureProcessor object at 0x...>

    Then, create the CRFChordRecognitionProcessor to decode a chord sequence
    from the extracted features:

    >>> decode = CRFChordRecognitionProcessor()
    >>> decode  # doctest: +ELLIPSIS
    <madmom.features.chords.CRFChordRecognitionProcessor object at 0x...>

    To transcribe the chords, you can either manually call the processors
    one after another,

    >>> feats = featproc('tests/data/audio/sample2.wav')
    >>> decode(feats)
    ... # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS +IGNORE_UNICODE
    ... # doctest: +NORMALIZE_ARRAYS
    array([(0. , 0.2, 'N'), (0.2, 1.6, 'F:maj'),
           (1.6, 2.4..., 'A:maj'), (2.4..., 4.1, 'D:min')],
          dtype=[('start', '<f8'), ('end', '<f8'), ('label', 'O')])

    or create a `madmom.processors.SequentialProcessor` that connects them:

    >>> from madmom.processors import SequentialProcessor
    >>> chordrec = SequentialProcessor([featproc, decode])
    >>> chordrec('tests/data/audio/sample2.wav')
    ... # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS +IGNORE_UNICODE
    ... # doctest: +NORMALIZE_ARRAYS
    array([(0. , 0.2, 'N'), (0.2, 1.6, 'F:maj'),
           (1.6, 2.4..., 'A:maj'), (2.4..., 4.1, 'D:min')],
          dtype=[('start', '<f8'), ('end', '<f8'), ('label', 'O')])
    """
    def __init__(self, model=None, fps=10, **kwargs):
        from ..ml.crf import ConditionalRandomField
        from ..models import CHORDS_CFCRF
        crf = ConditionalRandomField.load(model or CHORDS_CFCRF[0])
        lbl = partial(majmin_targets_to_chord_labels, fps=fps)
        super(CRFChordRecognitionProcessor, self).__init__((crf, lbl))
