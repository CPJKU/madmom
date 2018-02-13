# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains chroma related functionality.

"""

from __future__ import absolute_import, division, print_function

import warnings
import numpy as np

from madmom.audio.spectrogram import (Spectrogram, FilteredSpectrogram,
                                      SemitoneBandpassSpectrogram)
from madmom.audio.filters import (A4, Filterbank,
                                  PitchClassProfileFilterbank as PCP,
                                  HarmonicPitchClassProfileFilterbank as HPCP)
from madmom.processors import SequentialProcessor, Processor


# inherit from FilteredSpectrogram, since this class is closest related
class PitchClassProfile(FilteredSpectrogram):
    """
    Simple class for extracting pitch class profiles (PCP), i.e. chroma
    vectors from a spectrogram.

    Parameters
    ----------
    spectrogram : :class:`.audio.spectrogram.Spectrogram` instance
        :class:`.audio.spectrogram.Spectrogram` instance.
    filterbank : :class:`.audio.filters.Filterbank` class or instance
        :class:`.audio.filters.Filterbank` class or instance.
    num_classes : int, optional
        Number of pitch classes.
    fmin : float, optional
        Minimum frequency of the PCP filterbank [Hz].
    fmax : float, optional
        Maximum frequency of the PCP filterbank [Hz].
    fref : float, optional
        Reference frequency for the first PCP bin [Hz].
    kwargs : dict, optional
        If no :class:`.audio.spectrogram.Spectrogram` instance was given,
        one is instantiated with these additional keyword arguments.

    Notes
    -----
    If `fref` is 'None', the reference frequency is estimated from the given
    spectrogram.

    References
    ----------
    .. [1] T. Fujishima,
           "Realtime chord recognition of musical sound: a system using Common
           Lisp Music",
           Proceedings of the International Computer Music Conference (ICMC),
           1999.

    """
    # pylint: disable=super-on-old-class
    # pylint: disable=super-init-not-called
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, spectrogram, filterbank=PCP, num_classes=PCP.CLASSES,
                 fmin=PCP.FMIN, fmax=PCP.FMAX, fref=A4, **kwargs):
        # this method is for documentation purposes only
        pass

    def __new__(cls, spectrogram, filterbank=PCP, num_classes=PCP.CLASSES,
                fmin=PCP.FMIN, fmax=PCP.FMAX, fref=A4, **kwargs):
        # check spectrogram type
        if not isinstance(spectrogram, Spectrogram):
            spectrogram = Spectrogram(spectrogram, **kwargs)
        # spectrogram should not be filtered
        if hasattr(spectrogram, 'filterbank'):
            warnings.warn('Spectrogram should not be filtered.',
                          RuntimeWarning)
        # reference frequency for the filterbank
        if fref is None:
            fref = spectrogram.tuning_frequency()

        # set filterbank
        if issubclass(filterbank, Filterbank):
            filterbank = filterbank(spectrogram.bin_frequencies,
                                    num_classes=num_classes, fmin=fmin,
                                    fmax=fmax, fref=fref)
        if not isinstance(filterbank, Filterbank):
            raise ValueError('not a Filterbank type or instance: %s' %
                             filterbank)
        # filter the spectrogram
        data = np.dot(spectrogram, filterbank)
        # cast as PitchClassProfile
        obj = np.asarray(data).view(cls)
        # save additional attributes
        obj.filterbank = filterbank
        obj.spectrogram = spectrogram
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here, also needed for views
        self.filterbank = getattr(obj, 'filterbank', None)
        self.spectrogram = getattr(obj, 'spectrogram', None)


class HarmonicPitchClassProfile(PitchClassProfile):
    """
    Class for extracting harmonic pitch class profiles (HPCP) from a
    spectrogram.

    Parameters
    ----------
    spectrogram : :class:`.audio.spectrogram.Spectrogram` instance
        :class:`.audio.spectrogram.Spectrogram` instance.
    filterbank : :class:`.audio.filters.Filterbank` class or instance
        Filterbank class or instance.
    num_classes : int, optional
        Number of harmonic pitch classes.
    fmin : float, optional
        Minimum frequency of the HPCP filterbank [Hz].
    fmax : float, optional
        Maximum frequency of the HPCP filterbank [Hz].
    fref : float, optional
        Reference frequency for the first HPCP bin [Hz].
    window : int, optional
        Length of the weighting window [bins].
    kwargs : dict, optional
        If no :class:`.audio.spectrogram.Spectrogram` instance was given,
        one is instantiated with these additional keyword arguments.

    Notes
    -----
    If `fref` is 'None', the reference frequency is estimated from the given
    spectrogram.

    References
    ----------
    .. [1] Emilia Gómez,
           "Tonal Description of Music Audio Signals",
           PhD thesis, Universitat Pompeu Fabra, Barcelona, Spain, 2006.

    """
    # pylint: disable=super-on-old-class
    # pylint: disable=super-init-not-called
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, spectrogram, filterbank=HPCP, num_classes=HPCP.CLASSES,
                 fmin=HPCP.FMIN, fmax=HPCP.FMAX, fref=A4, window=HPCP.WINDOW,
                 **kwargs):
        # this method is for documentation purposes only
        pass

    def __new__(cls, spectrogram, filterbank=HPCP, num_classes=HPCP.CLASSES,
                fmin=HPCP.FMIN, fmax=HPCP.FMAX, fref=A4, window=HPCP.WINDOW,
                **kwargs):
        # check spectrogram type
        if not isinstance(spectrogram, Spectrogram):
            spectrogram = Spectrogram(spectrogram, **kwargs)
        # spectrogram should not be filtered
        if hasattr(spectrogram, 'filterbank'):
            warnings.warn('Spectrogram should not be filtered.',
                          RuntimeWarning)
        # reference frequency for the filterbank
        if fref is None:
            fref = spectrogram.tuning_frequency()

        # set filterbank
        if issubclass(filterbank, Filterbank):
            filterbank = filterbank(spectrogram.bin_frequencies,
                                    num_classes=num_classes, fmin=fmin,
                                    fmax=fmax, fref=fref, window=window)
        if not isinstance(filterbank, Filterbank):
            raise ValueError('not a Filterbank type or instance: %s' %
                             filterbank)
        # filter the spectrogram
        data = np.dot(spectrogram, filterbank)
        # cast as PitchClassProfile
        obj = np.asarray(data).view(cls)
        # save additional attributes
        obj.filterbank = filterbank
        obj.spectrogram = spectrogram
        # return the object
        return obj


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
    array([[0.01317, 0.00721, ..., 0.00546, 0.00943],
           [0.36809, 0.01314, ..., 0.02213, 0.01838],
           ...,
           [0.1534 , 0.06475, ..., 0.00896, 0.05789],
           [0.17513, 0.0729 , ..., 0.00945, 0.06913]], dtype=float32)
    >>> chroma.shape
    (41, 12)

    """

    def __init__(self, fmin=65, fmax=2100, unique_filters=True, models=None,
                 **kwargs):
        from ..models import CHROMA_DNN
        from ..audio.signal import SignalProcessor, FramedSignalProcessor
        from ..audio.stft import ShortTimeFourierTransformProcessor
        from ..audio.spectrogram import LogarithmicFilteredSpectrogramProcessor
        from madmom.ml.nn import NeuralNetworkEnsemble
        # signal pre-processing
        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        frames = FramedSignalProcessor(frame_size=8192, fps=10)
        stft = ShortTimeFourierTransformProcessor()  # caching FFT window
        spec = LogarithmicFilteredSpectrogramProcessor(
            num_bands=24, fmin=fmin, fmax=fmax, unique_filters=unique_filters)
        # split the spectrogram into overlapping frames
        spec_signal = SignalProcessor(sample_rate=10)
        spec_frames = FramedSignalProcessor(frame_size=15, hop_size=1, fps=10)
        # predict chroma bins with a DNN
        nn = NeuralNetworkEnsemble.load(models or CHROMA_DNN, **kwargs)
        # instantiate a SequentialProcessor
        super(DeepChromaProcessor, self).__init__([
            sig, frames, stft, spec, spec_signal, spec_frames, _dcp_flatten, nn
        ])


# Compressed Log Pitch (CLP) chroma stuff
CLP_FPS = 50
CLP_FMIN = 27.5
CLP_FMAX = 4200.
CLP_COMPRESSION_FACTOR = 100
CLP_NORM = True
CLP_THRESHOLD = 0.001


class CLPChroma(np.ndarray):
    """
    Compressed Log Pitch (CLP) chroma as proposed in [1]_ and [2]_.

    Parameters
    ----------
    data : str, Signal, or SemitoneBandpassSpectrogram
        Input data.
    fps : int, optional
        Desired frame rate of the signal [Hz].
    fmin : float, optional
        Lowest frequency of the spectrogram [Hz].
    fmax : float, optional
        Highest frequency of the spectrogram [Hz].
    compression_factor : float, optional
        Factor for compression of the energy.
    norm : bool, optional
        Normalize the energy of each frame to one (divide by the L2 norm).
    threshold : float, optional
        If the energy of a frame is below a threshold, the energy is equally
        distributed among all chroma bins.

    Notes
    -----
    The resulting chromagrams differ slightly from those obtained by the
    MATLAB chroma toolbox [2]_ because of different resampling and filter
    methods.

    References
    ----------
    .. [1] Meinard Müller,
           "Information retrieval for music and motion", Springer, 2007.

    .. [2] Meinard Müller and Sebastian Ewert,
           "Chroma Toolbox: MATLAB Implementations for Extracting Variants of
           Chroma-Based Audio Features",
           Proceedings of the International Conference on Music Information
           Retrieval (ISMIR), 2011.

    """

    def __init__(self, data, fps=CLP_FPS, fmin=CLP_FMIN, fmax=CLP_FMAX,
                 compression_factor=CLP_COMPRESSION_FACTOR, norm=CLP_NORM,
                 threshold=CLP_THRESHOLD, **kwargs):
        # this method is for documentation purposes only
        pass

    def __new__(cls, data, fps=CLP_FPS, fmin=CLP_FMIN, fmax=CLP_FMAX,
                compression_factor=CLP_COMPRESSION_FACTOR, norm=CLP_NORM,
                threshold=CLP_THRESHOLD, **kwargs):
        from madmom.audio.filters import hz2midi
        # check input type
        if not isinstance(data, SemitoneBandpassSpectrogram):
            # compute SemitoneBandpassSpectrogram
            data = SemitoneBandpassSpectrogram(data, fps=fps, fmin=fmin,
                                               fmax=fmax)
        # apply log compression
        log_pitch_energy = np.log10(data * compression_factor + 1)
        # compute chroma by adding up bins that correspond to the same
        # pitch class
        obj = np.zeros((log_pitch_energy.shape[0], 12)).view(cls)
        midi_min = int(np.round(hz2midi(data.bin_frequencies[0])))
        for p in range(log_pitch_energy.shape[1]):
            # make sure that p maps to the correct bin_label (midi_min=12
            # corresponds to a C and therefore chroma_idx=0)
            chroma_idx = np.mod(midi_min + p, 12)
            obj[:, chroma_idx] += log_pitch_energy[:, p]
        obj.bin_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G',
                          'G#', 'A', 'A#', 'B']
        obj.fps = fps
        if norm:
            # normalise the vectors according to the l2 norm
            mean_energy = np.sqrt((obj ** 2).sum(axis=1))
            idx_below_threshold = np.where(mean_energy < threshold)
            obj /= mean_energy[:, np.newaxis]
            obj[idx_below_threshold, :] = np.ones((1, 12)) / np.sqrt(12)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here
        self.fps = getattr(obj, 'fps', None)
        self.bin_labels = getattr(obj, 'bin_labels', None)


class CLPChromaProcessor(Processor):
    """
    Compressed Log Pitch (CLP) Chroma Processor.

    Parameters
    ----------
    fps : int, optional
        Desired frame rate of the signal [Hz].
    fmin : float, optional
        Lowest frequency of the spectrogram [Hz].
    fmax : float, optional
        Highest frequency of the spectrogram [Hz].
    compression_factor : float, optional
        Factor for compression of the energy.
    norm : bool, optional
        Normalize the energy of each frame to one (divide by the L2 norm).
    threshold : float, optional
        If the energy of a frame is below a threshold, the energy is equally
        distributed among all chroma bins.

    """

    def __init__(self, fps=CLP_FPS, fmin=CLP_FMIN, fmax=CLP_FMAX,
                 compression_factor=CLP_COMPRESSION_FACTOR, norm=CLP_NORM,
                 threshold=CLP_THRESHOLD, **kwargs):
        # pylint: disable=unused-argument
        self.fps = fps
        self.fmin = fmin
        self.fmax = fmax
        self.compression_factor = compression_factor
        self.norm = norm
        self.threshold = threshold

    def process(self, data, **kwargs):
        """
        Create a CLPChroma from the given data.

        Parameters
        ----------
        data : Signal instance or filename
            Data to be processed.

        Returns
        -------
        clp : :class:`CLPChroma` instance
            CLPChroma.

        """
        # update arguments passed to CLPChroma
        args = dict(fps=self.fps, fmin=self.fmin, fmax=self.fmax,
                    compression_factor=self.compression_factor,
                    norm=self.norm, threshold=self.threshold)
        args.update(kwargs)
        # instantiate a CLPChroma
        return CLPChroma(data, **args)
