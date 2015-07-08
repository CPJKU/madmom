#!/usr/bin/env python
# encoding: utf-8
"""
This file contains spectrogram related functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import numpy as np
import scipy.fftpack as fft

from madmom.processors import Processor, SequentialProcessor, ParallelProcessor
from .filters import (LogarithmicFilterbank, BANDS, FMIN, FMAX, A4,
                      NORM_FILTERS, DUPLICATE_FILTERS)


def fft_frequencies(num_fft_bins, sample_rate):
    """
    Frequencies of the FFT bins.

    :param num_fft_bins: number of FFT bins (i.e. half the FFT length)
    :param sample_rate:  sample rate of the signal
    :return:             frequencies of the FFT bins

    """
    return np.fft.fftfreq(num_fft_bins * 2, 1. / sample_rate)[:num_fft_bins]


# functions
def dft(signal, window, fft_size, circular_shift=False):
    """
    Calculates the discrete Fourier transform (DFT) of the given signal.

    :param signal:         discrete signal [1D numpy array]
    :param window:         window function [1D numpy array]
    :param fft_size:       use this size for FFT [int, should be a power of 2]
    :param circular_shift: circular shift for correct phase [bool]
    :return:               the complex DFT of the signal

    """
    # multiply the signal frame with the window function
    signal = np.multiply(signal, window)
    # only shift and perform complex DFT if needed
    if circular_shift:
        # circular shift the signal (needed for correct phase)
        shift = len(window) >> 1
        signal = np.concatenate((signal[shift:], signal[:shift]))
    # perform DFT and return the signal
    return fft.fft(signal, fft_size)[:fft_size >> 1]


def stft(frames, window, fft_size=None, circular_shift=False):
    """
    Calculates the complex Short-Time Fourier Transform (STFT) of the given
    framed signal.

    :param frames:         framed signal [2D numpy array or iterable]
    :param window:         window function [1D numpy array]
    :param fft_size:       use this size for FFT [int, should be a power of 2]
    :param circular_shift: circular shift for correct phase [bool]
    :return:               the complex STFT of the signal

    Note: The window is centered around the current sample and the total length
          of the STFT is calculated such that the last frame still covers some
          signal.

    """
    # number of FFT bins
    if fft_size is None:
        fft_size = len(window)
    num_fft_bins = fft_size >> 1
    # init STFT matrix
    stft = np.empty((len(frames), num_fft_bins), np.complex64)
    for f, frame in enumerate(frames):
        # perform DFT
        stft[f] = dft(frame, window, fft_size, circular_shift)
    # return STFT
    return stft


def spec(stft):
    """
    Returns the magnitudes of the complex Short Time Fourier Transform of a
    signal.

    :param stft: complex STFT of a signal
    :return:     magnitude spectrogram

    """
    return np.abs(stft)


def phase(stft):
    """
    Returns the phase of the complex Short Time Fourier Transform of a signal.

    :param stft: complex STFT of a signal
    :return:     phase

    """
    return np.angle(stft)


def local_group_delay(phase):
    """
    Returns the local group delay of the phase Short Time Fourier Transform of
    a signal.

    :param phase: phase of the STFT of a signal
    :return:      local group delay

    """
    # unwrap phase
    unwrapped_phase = np.unwrap(phase)
    # local group delay is the derivative over frequency
    unwrapped_phase[:, :-1] -= unwrapped_phase[:, 1:]
    # set the highest frequency to 0
    unwrapped_phase[:, -1] = 0
    # return the local group delay
    return unwrapped_phase

# alias
lgd = local_group_delay


# some functions working on magnitude spectra
def adaptive_whitening(spec, floor=0.5, relaxation=10):
    """
    Return an adaptively whitened version of the magnitude spectrogram.

    :param spec:       magnitude spectrogram [numpy array]
    :param floor:      floor coefficient [float]
    :param relaxation: relaxation time [frames]
    :return:           the whitened magnitude spectrogram

    "Adaptive Whitening For Improved Real-time Audio Onset Detection"
    Dan Stowell and Mark Plumbley
    Proceedings of the International Computer Music Conference (ICMC), 2007

    """
    raise NotImplementedError("check if adaptive_whitening returns meaningful "
                              "results")
    relaxation = 10.0 ** (-6. * relaxation)
    p = np.zeros_like(spec)
    # iterate over all frames
    for f in range(len(spec)):
        if f > 0:
            p[f] = np.maximum(spec[f], floor, relaxation * p[f - 1])
        else:
            p[f] = np.maximum(spec[f], floor)
    # return the whitened spectrogram
    return spec / p


def statistical_spectrum_descriptors(spec):
    """
    Statistical Spectrum Descriptors of the STFT.

    :param spec: magnitude spectrogram [numpy array]
    :return:     statistical spectrum descriptors of the spectrogram

    "Evaluation of Feature Extractors and Psycho-acoustic Transformations
     for Music Genre Classification."
    Thomas Lidy and Andreas Rauber
    Proceedings of the 6th International Conference on Music Information
    Retrieval (ISMIR), 2005

    """
    from scipy.stats import skew, kurtosis
    return {'mean': np.mean(spec, axis=0),
            'median': np.median(spec, axis=0),
            'variance': np.var(spec, axis=0),
            'skewness': skew(spec, axis=0),
            'kurtosis': kurtosis(spec, axis=0),
            'min': np.min(spec, axis=0),
            'max': np.max(spec, axis=0)}


def tuning_frequency(spec, bin_frequencies, num_hist_bins=15, fref=A4):
    """
    Determines the tuning frequency of the audio signal based on the given
    (peak) magnitude spectrogram.

    :param spec:            (peak) magnitude spectrogram [numpy array]
    :param bin_frequencies: frequencies of the spectrogram bins [numpy array]
    :param num_hist_bins:   number of histogram bins
    :param fref:            reference tuning frequency [Hz]
    :return:                tuning frequency

    """
    raise NotImplementedError("check if tuning_frequency returns meaningful "
                              "results")
    # TODO: make this function accept just a signal?
    # interval of spectral bins from the reference frequency in semitones
    semitone_int = 12. * np.log2(bin_frequencies / fref)
    # deviation from the next semitone
    semitone_dev = semitone_int - np.round(semitone_int)
    # build a histogram
    hist = np.histogram(semitone_dev * spec,
                        bins=num_hist_bins, range=(-0.5, 0.5))
    # deviation of the bins (calculate the bin centres)
    dev_bins = (hist[1][:-1] + hist[1][1:]) / 2.
    # dominant deviation
    dev = num_hist_bins * dev_bins[np.argmax(hist[0])]
    # calculate the tuning frequency
    return fref * 2. ** (dev / 12.)


# mixin for some basic properties of all classes
class _PropertyMixin(object):

    @property
    def num_frames(self):
        """Number of frames."""
        return len(self)

    @property
    def num_bins(self):
        """Number of bins."""
        return self.shape[1]


# short-time Fourier transform classes
class ShortTimeFourierTransform(_PropertyMixin, np.ndarray):
    """
    ShortTimeFourierTransform class.

    """

    def __new__(cls, frames, window=np.hanning, fft_size=None,
                circular_shift=False, **kwargs):
        """
        Creates a new ShortTimeFourierTransform instance from the given
        FramedSignal.

        :param frames:         FramedSignal instance (or anything a
                               FramedSignal can be instantiated from)

        FFT parameters:

        :param window:         window function [numpy ufunc or numpy array]
        :param fft_size:       use this size for the FFT [int, power of 2]
        :param circular_shift: circular shift the signal before performing the
                               FFT; needed for correct phase

        If no FramedSignal instance was given, one is instantiated and these
        arguments are passed:

        :param args:           arguments passed to FramedSignal
        :param kwargs:         keyword arguments passed to FramedSignal

        """
        from .signal import FramedSignal

        # take the FramedSignal from the given STFT
        if isinstance(frames, ShortTimeFourierTransform):
            # already a STFT
            frames = frames.frames
        # instantiate a FramedSignal if needed
        if not isinstance(frames, FramedSignal):
            frames = FramedSignal(frames, **kwargs)

        # check if the Signal is mono
        if frames.signal.num_channels > 1:
            raise ValueError('please implement multi-channel support')

        # determine which window to use
        if hasattr(window, '__call__'):
            # if only function is given, use the size to the audio frame size
            window = window(frames.frame_size)
            # # multi-channel window
            # if frames.signal.num_channels > 1:
            #     window = np.tile(window[:, np.newaxis],
            #                      frames.signal.num_channels)
        elif isinstance(window, np.ndarray):
            # otherwise use the given window directly
            if len(window) != frames.frame_size:
                raise ValueError('Window size must be equal to frame size.')
        else:
            # other types are not supported
            raise TypeError("Invalid window type.")
        # window used for FFT
        try:
            # the audio signal is not scaled, scale the window accordingly
            max_range = np.iinfo(frames.signal.dtype).max
            fft_window = window / max_range
        except ValueError:
            fft_window = window
        # circular shift the window for correct phase
        if circular_shift:
            fft_shift = len(fft_window) >> 1

        # FFT size to use
        if fft_size is None:
            fft_size = len(window)
        # number of FFT bins to store
        fft_bins = fft_size >> 1

        # create an empty object
        data = np.empty((frames.num_frames, fft_bins), np.complex64)
        # iterate over all frames
        for f, frame in enumerate(frames):
            # multiply the signal frame with the window function
            signal = np.multiply(frame, fft_window)
            # only shift and perform complex DFT if needed
            if circular_shift:
                # circular shift the signal (needed for correct phase)
                signal = np.concatenate((signal[fft_shift:],
                                         signal[:fft_shift]))
            # perform DFT and return the signal
            data[f] = fft.fft(signal, fft_size, axis=0)[:fft_bins]

        # cast as ShortTimeFourierTransform
        obj = np.asarray(data).view(cls)
        # save the other parameters
        obj.frames = frames
        obj.window = window
        obj.fft_window = fft_window
        obj.fft_size = fft_size
        obj.circular_shift = circular_shift
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here, also needed for views
        self.frames = getattr(obj, 'frames', None)
        self.window = getattr(obj, 'window', np.hanning)
        self.fft_window = getattr(obj, 'fft_window', None)
        self.fft_size = getattr(obj, 'fft_size', None)
        self.circular_shift = getattr(obj, 'circular_shift', False)

    def __reduce__(self):
        # needed for correct pickling
        # source: http://stackoverflow.com/questions/26598109/
        # get the parent's __reduce__ tuple
        pickled_state = super(ShortTimeFourierTransform, self).__reduce__()
        # create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.window, self.fft_window,
                                        self.fft_size, self.circular_shift)
        # return a tuple that replaces the parent's __reduce__ tuple
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):
        # needed for correct un-pickling
        # set the attributes
        self.window = state[-4]
        self.fft_window = state[-3]
        self.fft_size = state[-2]
        self.circular_shift = state[-1]
        # call the parent's __setstate__ with the other tuple elements
        super(ShortTimeFourierTransform, self).__setstate__(state[0:-4])

    @property
    def bin_freqs(self):
        """Frequencies of the FFT bins."""
        return fft_frequencies(self.num_bins, self.frames.signal.sample_rate)

    def spec(self, **kwargs):
        """Magnitude spectrogram"""
        return Spectrogram(self, **kwargs)

    def phase(self, **kwargs):
        """Phase spectrogram"""
        return Phase(self, **kwargs)


STFT = ShortTimeFourierTransform


class ShortTimeFourierTransformProcessor(Processor):
    """
    ShortTimeFourierTransformProcessor class.

    """

    def __init__(self, window=np.hanning, fft_size=None, circular_shift=False,
                 **kwargs):
        """
        Creates a new ShortTimeFourierTransformProcessor instance.

        :param window:         window function [numpy ufunc or numpy array]
        :param fft_size:       use this size for the FFT [int, power of 2]
        :param circular_shift: circular shift the signal before performing the
                               FFT; needed for correct phase

        """
        self.window = window
        self.fft_size = fft_size
        self.circular_shift = circular_shift

    def process(self, data, **kwargs):
        """
        Perform FFT on a framed signal and return the STFT.

        :param data:   data to be processed
        :param kwargs: keyword arguments passed to ShortTimeFourierTransform
        :return:       ShortTimeFourierTransform instance

        """
        # instantiate a STFT
        return ShortTimeFourierTransform(data, window=self.window,
                                         fft_size=self.fft_size,
                                         circular_shift=self.circular_shift,
                                         **kwargs)

    @classmethod
    def add_arguments(cls, parser, window=None, fft_size=None):
        """
        Add STFT related arguments to an existing parser.

        :param parser:   existing argparse parser
        :param window:   window function
        :param fft_size: use this size for FFT [int, should be a power of 2]
        :return:         STFT argument parser group

        Parameters are included in the group only if they are not 'None'.

        """
        # add filterbank related options to the existing parser
        g = parser.add_argument_group('short-time Fourier transform arguments')
        if window is not None:
            g.add_argument('--window', dest='window',
                           action='store', default=window,
                           help='window function to use for FFT')
        if fft_size is not None:
            g.add_argument('--fft_size', action='store', type=int,
                           default=fft_size,
                           help='use this size for FFT (should be a power of '
                                '2) [default=%(default)i]')
        # return the group
        return g


STFTProcessor = ShortTimeFourierTransformProcessor


# phase of STFT
class Phase(np.ndarray):
    """
    Phase class.

    """

    def __new__(cls, stft, **kwargs):
        """
        Creates a new Phase instance from the given ShortTimeFourierTransform.

        :param stft:   ShortTimeFourierTransform instance (or anything a
                       ShortTimeFourierTransform can be instantiated from)

        If no ShortTimeFourierTransform instance was given, one is instantiated
        and these arguments are passed:

        :param args:   arguments passed to ShortTimeFourierTransform
        :param kwargs: keyword arguments passed to ShortTimeFourierTransform

        """

        # take the STFT
        if isinstance(stft, Phase):
            stft = stft.stft
        # instantiate a ShortTimeFourierTransform object if needed
        if not isinstance(stft, ShortTimeFourierTransform):
            stft = ShortTimeFourierTransform(stft, circular_shift=True,
                                             **kwargs)
        # TODO: just recalculate with circular_shift set?
        if not stft.circular_shift:
            import warnings
            warnings.warn("`circular_shift` of the STFT must be set to 'True' "
                          "for correct phase")
        # take the abs of the stft
        data = np.angle(stft)
        # cast as Spectrogram
        obj = np.asarray(data).view(cls)
        # save additional attributes
        obj.stft = stft
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here, also needed for views
        self.stft = getattr(obj, 'stft', None)

    @property
    def bin_freqs(self):
        """Frequencies of the FFT bins."""
        return fft_frequencies(self.num_bins,
                               self.stft.frames.signal.sample_rate)

    def local_group_delay(self, **kwargs):
        """Local group delay."""
        return LocalGroupDelay(self, **kwargs)


# local group delay of STFT
class LocalGroupDelay(_PropertyMixin, Phase):
    """
    Phase class.

    """

    def __new__(cls, phase, **kwargs):
        """
        Creates a new LocalGroupDelay instance from the given
        ShortTimeFourierTransform.

        :param stft:   ShortTimeFourierTransform instance (or anything a
                       ShortTimeFourierTransform can be instantiated from)

        If no ShortTimeFourierTransform instance was given, one is instantiated
        and these arguments are passed:

        :param args:   arguments passed to ShortTimeFourierTransform
        :param kwargs: keyword arguments passed to ShortTimeFourierTransform

        """
        #
        if not isinstance(stft, Phase):
            # try to instantiate a ShortTimeFourierTransform object
            phase = Phase(phase, circular_shift=True, **kwargs)
        if not phase.stft.circular_shift:
            import warnings
            warnings.warn("`circular_shift` of the STFT must be set to 'True' "
                          "for correct local group delay")
        # unwrap phase
        data = np.unwrap(phase)
        # local group delay is the derivative over frequency
        data[:, :-1] -= data[:, 1:]
        # set the highest frequency to 0
        data[:, -1] = 0
        # cast as Spectrogram
        obj = np.asarray(data).view(cls)
        # save additional attributes
        obj.phase = phase
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here, also needed for views
        self.phase = getattr(obj, 'phase', None)


# magnitude spectrogram of STFT
class Spectrogram(_PropertyMixin, np.ndarray):
    """
    Spectrogram class.

    """

    def __new__(cls, stft, **kwargs):
        """
        Creates a new Spectrogram instance from the given
        ShortTimeFourierTransform.

        :param stft:   ShortTimeFourierTransform instance (or anything a
                       ShortTimeFourierTransform can be instantiated from)

        If no ShortTimeFourierTransform instance was given, one is instantiated
        and these arguments are passed:

        :param args:   arguments passed to ShortTimeFourierTransform
        :param kwargs: keyword arguments passed to ShortTimeFourierTransform

        """
        if isinstance(stft, Spectrogram):
            # already a Spectrogram
            data = stft
        elif isinstance(stft, ShortTimeFourierTransform):
            # take the abs of the STFT
            data = np.abs(stft)
        else:
            # try to instantiate a ShortTimeFourierTransform
            stft = ShortTimeFourierTransform(stft, **kwargs)
            # take the abs of the STFT
            data = np.abs(stft)
        # cast as Spectrogram
        obj = np.asarray(data).view(cls)
        # save additional attributes
        obj.stft = stft
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here, also needed for views
        self.stft = getattr(obj, 'stft', None)

    @property
    def bin_freqs(self):
        """Frequencies of the FFT bins."""
        return fft_frequencies(self.num_bins,
                               self.stft.frames.signal.sample_rate)

    def diff(self, **kwargs):
        """Difference of spectrogram."""
        return SpectrogramDifference(self, **kwargs)

    def filter(self, **kwargs):
        """Filtered spectrogram."""
        return FilteredSpectrogram(self, **kwargs)

    def log(self, **kwargs):
        """Logarithmic spectrogram."""
        return LogarithmicSpectrogram(self, **kwargs)


class SpectrogramProcessor(Processor):
    """
    SpectrogramProcessor class.

    """

    def process(self, data, **kwargs):
        """
        Create a Spectrogram from the given data.

        :param data: data to be processed
        :return:     Spectrogram instance

        """
        return Spectrogram(data, **kwargs)


# filtered spectrogram stuff
FILTERBANK = LogarithmicFilterbank


class FilteredSpectrogram(Spectrogram):
    """
    FilteredSpectrogram class.

    """

    # we just want to inherit some properties from Spectrogram
    def __new__(cls, spectrogram, filterbank=FILTERBANK, bands=BANDS,
                fmin=FMIN, fmax=FMAX, fref=A4, norm_filters=NORM_FILTERS,
                duplicate_filters=DUPLICATE_FILTERS, block_size=2048,
                **kwargs):
        """
        Creates a new FilteredSpectrogram instance from the given Spectrogram.

        :param spectrogram:       Spectrogram instance (or anything a
                                  Spectrogram can be instantiated from)

        Filterbank parameters:

        :param filterbank:        Filterbank type or instance [Filterbank]

        If a Filterbank type is given rather than a Filterbank instance, one
        will be created with the given type and these parameters:

        :param bands:             number of filter bands (per octave, depending
                                  on the type of the filterbank) [int]
        :param fmin:              the minimum frequency [Hz, float]
        :param fmax:              the maximum frequency [Hz, float]
        :param fref:              tuning frequency [Hz, float]
        :param norm_filters:      normalize the filter to area 1 [bool]
        :param duplicate_filters: keep duplicate filters [bool]

        Other filtering options:

        :param block_size:        perform filtering in blocks of this size
                                  [int, power of 2]

        If no Spectrogram instance was given, one is instantiated and
        these arguments are passed:

        :param args:              arguments passed to Spectrogram
        :param kwargs:            keyword arguments passed to Spectrogram

        """
        from .filters import Filterbank
        # instantiate a Spectrogram if needed
        if not isinstance(spectrogram, Spectrogram):
            # try to instantiate a Spectrogram object
            spectrogram = Spectrogram(spectrogram, **kwargs)

        # instantiate a Filterbank if needed
        if issubclass(filterbank, Filterbank):
            # create a filterbank of the given type
            filterbank = filterbank(spectrogram.bin_freqs, bands=bands,
                                    fmin=fmin, fmax=fmax, fref=fref,
                                    norm_filters=norm_filters,
                                    duplicate_filters=duplicate_filters)
        if not isinstance(filterbank, Filterbank):
            raise ValueError('not a Filterbank type or instance: %s' %
                             filterbank)

        # TODO: reactivate this or move this whole block/batch processing to
        #       the processors?
        # # init the return matrix
        # num_frames = spectrogram.num_frames
        # data = np.empty((num_frames, filterbank.num_bands), np.float32)
        # # process in blocks of this size
        # if block_size is None:
        #     block_size = spectrogram.num_frames
        # # iterate over the STFT in blocks of the given size
        # for b, start in enumerate(range(0, num_frames, block_size)):
        #     # determine stop index of the block
        #     stop = min(start + block_size, num_frames)
        #     # get the block
        #     block = spectrogram[start: stop]
        #     # determine the position inside the data to be returned
        #     start = b * block_size
        #     stop = start + len(block)
        #     # filter it and put it in the return spectrogram
        #     data[start: stop] = np.dot(block, filterbank)

        # filter the spectrogram
        data = np.dot(spectrogram, filterbank)
        # cast as FilteredSpectrogram
        obj = np.asarray(data).view(cls)
        # save additional attributes
        obj.stft = spectrogram.stft
        obj.filterbank = filterbank
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here, also needed for views
        self.frames = getattr(obj, 'frames', None)
        self.filterbank = getattr(obj, 'filterbank', None)

    def __reduce__(self):
        # needed for correct pickling
        # source: http://stackoverflow.com/questions/26598109/
        # get the parent's __reduce__ tuple
        pickled_state = super(FilteredSpectrogram, self).__reduce__()
        # create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.filterbank,)
        # return a tuple that replaces the parent's __reduce__ tuple
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):
        # needed for correct un-pickling
        # set the attributes
        self.filterbank = state[-1]
        # call the parent's __setstate__ with the other tuple elements
        super(FilteredSpectrogram, self).__setstate__(state[0:-1])

    @property
    def bin_freqs(self):
        """Frequencies of the spectrogram bins."""
        # overwrite with the filterbank center frequencies
        return self.filterbank.center_frequencies


class FilteredSpectrogramProcessor(Processor):
    """
    FilteredSpectrogramProcessor class.

    """

    def __init__(self, filterbank=FILTERBANK, bands=BANDS, fmin=FMIN,
                 fmax=FMAX, fref=A4, norm_filters=NORM_FILTERS,
                 duplicate_filters=DUPLICATE_FILTERS, **kwargs):
        """
        Creates a new FilteredSpectrogramProcessor instance.

        Magnitude spectrogram filtering parameters:

        :param filterbank:        filter the magnitude spectrogram with a
                                  filterbank of this type [None or Filterbank]
        :param bands:             use N bands (per octave) [int]
        :param fmin:              minimum frequency of the filterbank [float]
        :param fmax:              maximum frequency of the filterbank [float]
        :param fref:              tuning frequency [Hz, float]
        :param norm_filters:      normalize the filter to area 1 [bool]
        :param duplicate_filters: keep duplicate filters resulting from
                                  insufficient resolution of low frequencies

        """
        self.filterbank = filterbank
        self.bands = bands
        self.fmin = fmin
        self.fmax = fmax
        self.fref = fref
        self.norm_filters = norm_filters
        self.duplicate_filters = duplicate_filters

    def process(self, data, **kwargs):
        """
        Perform filtering of a spectrogram.

        :param data:       data to be processed
        :param block_size: perform processing in blocks of this size [int]
        :return:           Spectrogram instance

        Note: If `block_size` is 'None', all data is processed in a single
              chunk, if set its value should be a power of 2.

        """
        # instantiate a FilteredSpectrogram and return it
        return FilteredSpectrogram(data, filterbank=self.filterbank,
                                   bands=self.bands, fmin=self.fmin,
                                   fmax=self.fmax, fref=self.fref,
                                   norm_filters=self.norm_filters,
                                   duplicate_filters=self.duplicate_filters,
                                   **kwargs)

    @classmethod
    def add_arguments(cls, parser, filterbank=FILTERBANK, bands=BANDS,
                      fmin=FMIN, fmax=FMAX, norm_filters=NORM_FILTERS,
                      duplicate_filters=DUPLICATE_FILTERS):
        """
        Add spectrogram filtering related arguments to an existing parser.

        :param parser:            existing argparse parser
        :param filterbank:        filter the magnitude spectrogram with a
                                  logarithmic filterbank [Filterbank or bool]
        :param bands:             use N bands per octave [int]
        :param fmin:              minimum frequency of the filterbank [float]
        :param fmax:              maximum frequency of the filterbank [float]
        :param norm_filters:      normalize the filter to area 1 [bool]
        :param duplicate_filters: keep duplicate filters resulting from
                                  insufficient resolution of low frequencies
                                  [bool]
        :return:                  spectrogram filtering argument parser group

        Parameters are included in the group only if they are not 'None'.

        """
        # add filterbank related options to the existing parser
        g = parser.add_argument_group('spectrogram filtering arguments')
        if filterbank is not None:
            # TODO: add literal values
            if filterbank:
                g.add_argument('--no_filter', dest='filterbank',
                               action='store_false',
                               default=filterbank,
                               help='do not filter the spectrogram with a '
                                    'filterbank [default=True]')
            else:
                g.add_argument('--filter', action='store_true', default=None,
                               help='filter the spectrogram with a '
                                    'logarithmically spaced filterbank '
                                    '[default=False]')
        if bands is not None:
            g.add_argument('--bands', action='store', type=int,
                           default=bands,
                           help='use a filterbank with N bands (per octave) '
                                '[default=%(default)i]')
        if fmin is not None:
            g.add_argument('--fmin', action='store', type=float,
                           default=fmin,
                           help='minimum frequency of the filterbank '
                                '[Hz, default=%(default).1f]')
        if fmax is not None:
            g.add_argument('--fmax', action='store', type=float,
                           default=fmax,
                           help='maximum frequency of the filterbank '
                                '[Hz, default=%(default).1f]')
        if norm_filters is not None:
            if norm_filters:
                g.add_argument('--no_norm_filters', dest='norm_filters',
                               action='store_false', default=norm_filters,
                               help='do not normalize the filter to area 1 '
                                    '[default=True]')
            else:
                g.add_argument('--norm_filters', dest='norm_filters',
                               action='store_true', default=norm_filters,
                               help='normalize the filter to area 1 '
                                    '[default=False]')
        if duplicate_filters is not None:
            if duplicate_filters:
                g.add_argument('--no_duplicate_filters',
                               dest='duplicate_filters',
                               action='store_false', default=duplicate_filters,
                               help='do not keep duplicate filters resulting '
                                    'from insufficient resolution of low '
                                    'frequencies [default=True]')
            else:
                g.add_argument('--duplicate_filters', dest='duplicate_filters',
                               action='store_true', default=duplicate_filters,
                               help='keep duplicate filters resulting from '
                                    'insufficient resolution of low '
                                    'frequencies [default=False]')
        # return the group
        return g


# logarithmic spectrogram stuff
LOG = True
MUL = 1.
ADD = 1.


class LogarithmicSpectrogram(Spectrogram):
    """
    LogarithmicSpectrogram class.

    """

    # we just want to inherit some properties from Spectrogram
    def __new__(cls, spectrogram, mul=MUL, add=ADD, **kwargs):
        """
        Creates a new LogarithmicSpectrogram instance from the given
        Spectrogram.

        :param spectrogram: Spectrogram instance (or anything a Spectrogram
                            can be instantiated from)

        Logarithmic magnitude parameters:

        :param mul:         multiply the magnitude spectrogram with this factor
                            before taking the logarithm [float]
        :param add:         add this value before taking the logarithm of the
                            magnitudes [float]

        If no Spectrogram instance was given, one is instantiated and these
        arguments are passed:

        :param args:   arguments passed to Spectrogram
        :param kwargs: keyword arguments passed to Spectrogram

        """
        # instantiate a Spectrogram if needed
        if not isinstance(spectrogram, Spectrogram):
            # try to instantiate a Spectrogram object
            spectrogram = Spectrogram(spectrogram, **kwargs)

        # filter the spectrogram
        data = np.log10(mul * spectrogram + add)
        # cast as FilteredSpectrogram
        obj = np.asarray(data).view(cls)
        # save additional attributes
        obj.stft = spectrogram.stft
        obj.mul = mul
        obj.add = add
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here, also needed for views
        self.stft = getattr(obj, 'stft', None)
        self.mul = getattr(obj, 'mul', MUL)
        self.add = getattr(obj, 'add', ADD)

    def __reduce__(self):
        # needed for correct pickling
        # source: http://stackoverflow.com/questions/26598109/
        # get the parent's __reduce__ tuple
        pickled_state = super(LogarithmicSpectrogram, self).__reduce__()
        # create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.mul, self.add,)
        # return a tuple that replaces the parent's __reduce__ tuple
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):
        # needed for correct un-pickling
        # set the attributes
        self.mul = state[-2]
        self.add = state[-1]
        # call the parent's __setstate__ with the other tuple elements
        super(LogarithmicSpectrogram, self).__setstate__(state[0:-2])


class LogarithmicSpectrogramProcessor(Processor):
    """
    Logarithmic Spectrogram Processor class.

    """

    def __init__(self, mul=MUL, add=ADD, **kwargs):
        """
        Creates a new LogarithmicSpectrogramProcessor instance.

        Magnitude spectrogram scaling parameters:

        :param mul: multiply the spectrogram with this factor before taking
                    the logarithm of the magnitudes [float]
        :param add: add this value before taking the logarithm of the
                    magnitudes [float]

        """
        self.mul = mul
        self.add = add

    def process(self, data, **kwargs):
        """
        Perform logarithmic scaling of a spectrogram.

        :param data: data to be processed
        :return:     LogarithmicSpectrogram instance

        """
        # instantiate a LogarithmicSpectrogram
        return LogarithmicSpectrogram(data, mul=self.mul, add=self.add,
                                      **kwargs)

    @classmethod
    def add_arguments(cls, parser, log=None, mul=None, add=None):
        """
        Add logarithmic spectrogram scaling related arguments to an existing
        parser.

        :param parser: existing argparse parser
        :param log:    take the logarithm of the magnitude [bool]
        :param mul:    multiply the spectrogram with this factor before
                       taking the logarithm of the magnitudes [float]
        :param add:    add this value before taking the logarithm of the
                       magnitudes [float]
        :return:       logarithmic spectrogram scaling argument parser group

        Parameters are included in the group only if they are not 'None'.

        """
        # add log related options to the existing parser
        g = parser.add_argument_group('logarithmic magnitude arguments')
        if log is not None:
            if log:
                g.add_argument('--no_log', dest='log',
                               action='store_false', default=log,
                               help='linear magnitudes [default=logarithmic]')
            else:
                g.add_argument('--log', action='store_true',
                               default=-log,
                               help='logarithmic magnitudes [default=linear]')
        if mul is not None:
            g.add_argument('--mul', action='store', type=float,
                           default=mul, help='multiplier (before taking '
                           'the log) [default=%(default)i]')
        if add is not None:
            g.add_argument('--add', action='store', type=float,
                           default=add, help='value added (before taking '
                           'the log) [default=%(default)i]')
        # return the groups
        return g


# logarithmic filtered spectrogram class
class LogarithmicFilteredSpectrogram(LogarithmicSpectrogram):
    """
    LogarithmicFilteredSpectrogram class.

    """

    def __new__(cls, spectrogram, **kwargs):
        """
        Creates a new LogarithmicFilteredSpectrogram instance of the given
        FilteredSpectrogram.

        :param spectrogram: FilteredSpectrogram instance (or anything a
                            FilteredSpectrogram can be instantiated from)

        If no FilteredSpectrogram instance was given, one is instantiated and
        logarithmically scaled afterwards. These arguments are passed:

        :param args:        arguments passed to FilteredSpectrogram and
                            LogarithmicSpectrogram
        :param kwargs:      keyword arguments passed to FilteredSpectrogram and
                            LogarithmicSpectrogram

        """
        # instantiate a FilteredSpectrogram if needed
        if not isinstance(spectrogram, FilteredSpectrogram):
            spectrogram = FilteredSpectrogram(spectrogram, **kwargs)
        # take the logarithm
        data = LogarithmicSpectrogram(spectrogram, **kwargs)
        # cast as LogarithmicFilteredSpectrogram
        obj = np.asarray(data).view(cls)
        # save additional attributes
        obj.stft = spectrogram.stft
        obj.filterbank = spectrogram.filterbank
        obj.mul = data.mul
        obj.add = data.add
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here, also needed for views
        self.stft = getattr(obj, 'stft', None)
        self.filterbank = getattr(obj, 'filterbank', None)
        self.mul = getattr(obj, 'mul', MUL)
        self.add = getattr(obj, 'add', ADD)

    def __reduce__(self):
        # get the parent's __reduce__ tuple
        pickled_state = super(LogarithmicFilteredSpectrogram, self).__reduce__()
        # create our own tuple to pass to __setstate__
        # Note: we only need to save the filterbank, mul & add are handled by
        #       the parent's class
        new_state = pickled_state[2] + (self.filterbank,)
        # return a tuple that replaces the parent's __reduce__ tuple
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):
        # Note: we only need to set the filterbank, mul & add are handled by
        #       the parent's class
        self.filterbank = state[-1]
        # call the parent's __setstate__ with the other tuple elements
        super(LogarithmicFilteredSpectrogram, self).__setstate__(state[0:-1])

    @property
    def bin_freqs(self):
        """Frequencies of the spectrogram bins."""
        # overwrite with the filterbank center frequencies
        return self.filterbank.center_frequencies


# spectrogram difference stuff
DIFF_RATIO = 0.5
DIFF_FRAMES = None
DIFF_MAX_BINS = None
POSITIVE_DIFFS = False


class SpectrogramDifference(Spectrogram):
    """
    SpectrogramDifference class.

    """

    # we just want to inherit some properties from Spectrogram
    def __new__(cls, spectrogram, diff_ratio=DIFF_RATIO,
                diff_frames=DIFF_FRAMES, diff_max_bins=DIFF_MAX_BINS,
                positive_diffs=POSITIVE_DIFFS, **kwargs):
        """
        Creates a new SpectrogramDifference instance from the given
        spectrogram.

        :param spectrogram:       Spectrogram instance (or anything a
                                  Spectrogram can be instantiated from)

        Difference parameters:

        :param diff_ratio:        calculate the difference to the frame at
                                  which the window used for the STFT yields
                                  this ratio of the maximum height [float]
        :param diff_frames:       calculate the difference to the N-th previous
                                  frame (if set, this overrides the value
                                  calculated from the `diff_ratio`) [int]
        :param diff_max_bins:     apply a maximum filter with this width (in
                                  bins in frequency dimension) [int]
        :param positive_diffs:    keep only the positive differences, i.e. set
                                  all diff values < 0 to 0. [bool]

        If no Spectrogram instance was given, one is instantiated and these
        arguments are passed:

        :param args:              arguments passed to Spectrogram
        :param kwargs:            keyword arguments passed to Spectrogram

        Note: The SuperFlux algorithm uses a maximum filtered spectrogram with
              3 `max_bins` together with a 24 band logarithmic filterbank to
              calculate the difference spectrogram.

        """
        # instantiate a Spectrogram if needed
        if not isinstance(spectrogram, Spectrogram):
            # try to instantiate a Spectrogram object
            spectrogram = Spectrogram(spectrogram, **kwargs)

        # calculate the number of diff frames to use
        if not diff_frames:
            # calculate the number of diff_frames on basis of the diff_ratio
            # get the first sample with a higher magnitude than given ratio
            window = spectrogram.stft.window
            sample = np.argmax(window > diff_ratio * max(window))
            diff_samples = len(spectrogram.stft.window) / 2 - sample
            # convert to frames
            hop_size = spectrogram.stft.frames.hop_size
            diff_frames = int(round(diff_samples / hop_size))
        # always set the minimum to 1
        if diff_frames < 1:
            diff_frames = 1

        # init matrix
        diff = np.zeros_like(spectrogram)

        # apply a maximum filter to diff_spec if needed
        if diff_max_bins > 1:
            from scipy.ndimage.filters import maximum_filter
            # widen the spectrogram in frequency dimension
            diff_spec = maximum_filter(spectrogram, size=[1, diff_max_bins])
        else:
            diff_spec = spectrogram
        # calculate the diff
        diff[diff_frames:] = (spectrogram[diff_frames:] -
                              diff_spec[: -diff_frames])
        # positive differences only?
        if positive_diffs:
            np.maximum(diff, 0, diff)

        # cast as FilteredSpectrogram
        obj = np.asarray(diff).view(cls)
        obj.stft = spectrogram.stft
        obj.diff_ratio = diff_ratio
        obj.diff_frames = diff_frames
        obj.diff_max_bins = diff_max_bins
        obj.positive_diffs = positive_diffs
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here, also needed for views
        self.stft = getattr(obj, 'stft', None)
        self.diff_ratio = getattr(obj, 'diff_ratio', 0.5)
        self.diff_frames = getattr(obj, 'diff_frames', None)
        self.diff_max_bins = getattr(obj, 'diff_max_bins', None)
        self.positive_diffs = getattr(obj, 'positive_diffs', False)

    def __reduce__(self):
        # get the parent's __reduce__ tuple
        pickled_state = super(SpectrogramDifference, self).__reduce__()
        # create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.diff_ratio, self.diff_frames,
                                        self.diif_max_bins,
                                        self.positive_diffs)
        # return a tuple that replaces the parent's __reduce__ tuple
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):
        # set the attributes
        self.diff_ratio = state[-4]
        self.diff_frames = state[-3]
        self.diff_max_bins = state[-2]
        self.positive_diffs = state[-1]
        # call the parent's __setstate__ with the other tuple elements
        super(SpectrogramDifference, self).__setstate__(state[0:-4])

    def positive_diff(self):
        """Positive diff."""
        return np.maximum(self, 0)


class SpectrogramDifferenceProcessor(Processor):
    """
    Difference Spectrogram Processor class.

    """

    def __init__(self, diff_ratio=DIFF_RATIO, diff_frames=DIFF_FRAMES,
                 diff_max_bins=DIFF_MAX_BINS, positive_diffs=POSITIVE_DIFFS,
                 **kwargs):
        """
        Spectrogram difference parameters:

        :param diff_ratio:        calculate the difference to the frame at
                                  which the window used for the STFT yields
                                  this ratio of the maximum height [float]
        :param diff_frames:       calculate the difference to the N-th previous
                                  frame [int] (if set, this overrides the value
                                  calculated from the `diff_ratio`)
        :param diff_max_bins:     apply a maximum filter with this width (in
                                  bins in frequency dimension) [int]
        :param positive_diffs:    keep only the positive differences, i.e. set
                                  all diff values < 0 to 0

        """
        self.diff_ratio = diff_ratio
        self.diff_frames = diff_frames
        self.diff_max_bins = diff_max_bins
        self.positive_diffs = positive_diffs

    def process(self, data, **kwargs):
        """
        Perform a temporal difference calculation on the given data.

        :param data: data to calculate the difference on
        :return:     temporal diff of the data

        """
        # instantiate a SpectrogramDifference and return it
        return SpectrogramDifference(data, diff_ratio=self.diff_ratio,
                                     diff_frames=self.diff_frames,
                                     diff_max_bins=self.diff_max_bins,
                                     positive_diffs=self.positive_diffs,
                                     **kwargs)

    @classmethod
    def add_arguments(cls, parser, diff_ratio=None, diff_frames=None,
                      diff_max_bins=None, positive_diffs=None):
        """
        Add spectrogram difference related arguments to an existing parser.

        :param parser:            existing argparse parser
        :param diff_ratio:        calculate the difference to the frame at
                                  which the window used for the STFT yields
                                  this ratio of the maximum height [float]
        :param diff_frames:       calculate the difference to the N-th previous
                                  frame [int] (if set, this overrides the value
                                  calculated from the `diff_ratio`)
        :param diff_max_bins:     apply a maximum filter with this width (in
                                  bins in frequency dimension) [int]
        :param positive_diffs:    keep only the positive differences, i.e. set
                                  all diff values < 0 to 0
        :return:                  spectrogram difference argument parser group

        Parameters are included in the group only if they are not 'None'.
        Only the `diff_frames` parameter behaves differently, it is included
        if either the `diff_ratio` is set or a value != 'None' is given.

        """
        # add diff related options to the existing parser
        g = parser.add_argument_group('spectrogram difference arguments')
        if diff_ratio is not None:
            g.add_argument('--diff_ratio', action='store', type=float,
                           default=diff_ratio,
                           help='calculate the difference to the frame at '
                                'which the window of the STFT have this ratio '
                                'of the maximum height '
                                '[default=%(default).1f]')
        if diff_ratio is not None or diff_frames:
            g.add_argument('--diff_frames', action='store', type=int,
                           default=diff_frames,
                           help='calculate the difference to the N-th previous'
                                ' frame (this overrides the value calculated '
                                'with `diff_ratio`) [default=%(default)s]')
        if positive_diffs is not None:
            if positive_diffs:
                g.add_argument('--all_diffs', dest='positive_diffs',
                               action='store_false', default=positive_diffs,
                               help='keep both positive and negative diffs')
            else:
                g.add_argument('--positive_diffs', action='store_true',
                               default=-positive_diffs,
                               help='keep only positive and diffs')
        # add maximum filter related options to the existing parser
        if diff_max_bins is not None:
            g.add_argument('--max_bins', action='store', type=int,
                           dest='diff_max_bins', default=diff_max_bins,
                           help='apply a maximum filter with this width (in '
                                'frequency bins) [default=%(default)d]')

        # return the group
        return g


class SuperFluxProcessor(SequentialProcessor):
    """
    Spectrogram processor which sets the default values suitable for the
    SuperFlux algorithm.

    """

    def __init__(self, **kwargs):
        """
        Creates a new SuperFluxProcessor instance.

        """
        # set the default values (can be overwritten if set)
        # we need an un-normalized LogarithmicFilterbank with 24 bands
        filterbank = kwargs.pop('filterbank', FILTERBANK)
        bands = kwargs.pop('bands', 24)
        norm_filters = kwargs.pop('norm_filters', False)
        # we want max filtered diffs
        diff_ratio = kwargs.pop('diff_ratio', 0.5)
        diff_max_bins = kwargs.pop('diff_max_bins', 3)
        # processing chain
        stft = ShortTimeFourierTransformProcessor(**kwargs)
        spec = FilteredSpectrogramProcessor(filterbank=filterbank, bands=bands,
                                            norm_filters=norm_filters,
                                            **kwargs)
        lfs = LogarithmicSpectrogramProcessor(**kwargs)
        diff = SpectrogramDifferenceProcessor(diff_ratio=diff_ratio,
                                              diff_max_bins=diff_max_bins,
                                              **kwargs)
        # sequentially process everything
        super(SuperFluxProcessor, self).__init__([stft, spec, lfs, diff])


class MultiBandSpectrogram(FilteredSpectrogram):
    """
    MultiBandSpectrogram class.

    """

    def __init__(self, spectrogram, crossover_frequencies, norm_bands=False,
                 **kwargs):
        """
        Creates a new MultiBandSpectrogram instance from the given
        Spectrogram.

        :param spectrogram:           Spectrogram instance (or anything a
                                      FilteredSpectrogram be instantiated from)

        Multi-band parameters:

        :param crossover_frequencies: list of crossover frequencies at which
                                      the spectrogram is split into bands
        :param norm_bands:            normalize the bands [bool]

        If no Spectrogram instance was given, a FilteredSpectrogram is
        instantiated and these arguments are passed:

        :param args:                  arguments passed to FilteredSpectrogram
        :param kwargs:                keyword arguments passed to
                                      FilteredSpectrogram

        """
        from .filters import Filterbank
        # instantiate a FilteredSpectrogram if needed
        if not isinstance(spectrogram, Spectrogram):
            spectrogram = FilteredSpectrogram(spectrogram, **kwargs)
        # TODO: move this to filterbank and make it accept a list of bin
        #       frequencies (data.bin_freqs) to generate a rectangular filter

        # create a filterbank
        fb = np.zeros((spectrogram.num_bins, len(crossover_frequencies) + 1))
        # get the closest spectrogram bins to the requested crossover bins
        freq_distance = (spectrogram.bin_freqs -
                         np.asarray(crossover_frequencies)[:, np.newaxis])
        crossover_bins = np.argmin(np.abs(freq_distance), axis=1)
        # prepend index 0 and append length of the filterbank
        crossover_bins = np.r_[0, crossover_bins, len(fb)]
        # map the spectrogram bins to the filterbank bands
        for i in range(fb.shape[1]):
            fb[crossover_bins[i]:crossover_bins[i + 1], i] = 1
        # normalize the filterbank
        if norm_bands:
            fb /= np.sum(fb, axis=0)
        # wrap it as a Filterbank
        fb = Filterbank(fb, spectrogram.bin_freqs)
        # instantiate a FilteredSpectrogram with this filterbank
        super(MultiBandSpectrogram, self).__init__(spectrogram, filterbank=fb)
        # save the arguments
        self.crossover_frequencies = crossover_frequencies
        self.norm_bands = norm_bands


class MultiBandSpectrogramProcessor(Processor):
    """
    Spectrogram processor which combines the differences of a log filtered
    spectrogram into multiple bands.

    """

    def __init__(self, crossover_frequencies, norm_bands=False, **kwargs):
        """

        :param crossover_frequencies: list of crossover frequencies at which
                                      the spectrogram is split into bands
        :param norm_bands:            normalize the bands [bool]

        """
        self.crossover_frequencies = crossover_frequencies
        self.norm_bands = norm_bands

    def process(self, data, **kwargs):
        """
        Return the a multi-band representation of the given spectrogram.

        :param data: spectrogram to be processed [Spectrogram]
        :return:     Spectrogram instance

        """
        # instantiate a MultiBandSpectrogram
        return MultiBandSpectrogram(
            data, crossover_frequencies=self.crossover_frequencies,
            norm_bands=self.norm_bands, **kwargs)

    @classmethod
    def add_arguments(cls, parser, crossover_frequencies=None,
                      norm_bands=None):
        """
        Add multi-band spectrogram related arguments to an existing parser.

        :param parser:                existing argparse parser
        :param crossover_frequencies: list with crossover frequencies
        :param norm_bands:            normalize the bands
        :return:                      multi-band argument parser group

        Parameters are included in the group only if they are not 'None'.

        """
        # add filterbank related options to the existing parser
        g = parser.add_argument_group('multi-band spectrogram arguments')
        if crossover_frequencies is not None:
            from madmom.utils import OverrideDefaultListAction
            g.add_argument('--crossover_frequencies', type=float, sep=',',
                           action=OverrideDefaultListAction,
                           default=crossover_frequencies,
                           help='(comma separated) list with crossover '
                                'frequencies [Hz, default=%(default)s]')
        if norm_bands is not None:
            if norm_bands:
                g.add_argument('--no_norm_bands', dest='norm_bands',
                               action='store_false', default=norm_bands,
                               help='no not normalize the bands')
            else:
                g.add_argument('--norm_bands', action='store_true',
                               default=-norm_bands,
                               help='normalize the bands')
        # return the group
        return g


class StackSpectrogramProcessor(Processor):
    """
    Class to stack multiple spectrograms (and their differences) in a certain
    dimension.

    """
    def __init__(self, frame_size, fps=100, online=False,
                 filterbank=FILTERBANK, bands=BANDS, fmin=FMIN, fmax=FMAX,
                 norm_filters=NORM_FILTERS,
                 duplicate_filters=DUPLICATE_FILTERS, log=LOG, mul=MUL,
                 add=ADD, diff_ratio=DIFF_RATIO, stack=np.hstack,
                 stack_diffs=False, **kwargs):
        """
        Creates a new StackSpectrogramProcessor instance.

        Multiple magnitude spectra (with different FFT sizes) are filtered and
        logarithmically scaled before being stacked together with their first
        order differences.

        Framing parameters:

        :param frame_size:        include spectrogram with these frame sizes
                                  [list of int]
        :param fps:               frames per second [float]
        :param online:            online mode [bool]

        Filterbank parameters:

        :param filterbank:        filter the magnitude spectrogram with a
                                  filterbank of this type [None or Filterbank]
        :param bands:             use N bands per octave [int]
        :param fmin:              minimum frequency of the filterbank [float]
        :param fmax:              maximum frequency of the filterbank [float]
        :param norm_filters:      normalize the filter to area 1 [bool]
        :param duplicate_filters: keep duplicate filters resulting from
                                  insufficient resolution of low frequencies

        Logarithmic magnitude parameters:

        :param mul:               multiply the spectrogram with this factor
                                  before taking the logarithm of the magnitudes
                                  [float]
        :param add:               add this value before taking the logarithm of
                                  the magnitudes [float]

        Difference parameters:

        :param diff_ratio:        calculate the difference to the frame at
                                  which the window used for the STFT yields
                                  this ratio of the maximum height [float]

        Stacking parameters:

        :param stack:             stacking function for stacking the
                                  spectrograms (and their differences)
                                  - 'np.vstack' stack multiple spectrograms
                                    vertically, i.e. stack in time dimension
                                  - 'np.hstack' stack multiple spectrograms
                                    horizontally, i.e. stack in the frequency
                                    dimension
                                  - 'np.dstack' stacks them in depth, i.e.
                                    returns them as a 3D representation
                                  Additionally, the literal values {'time',
                                  'freq' | 'frequency', 'depth'} are supported
        :param stack_diffs:       also stack the differences [bool]

        Note: To be able to stack filtered spectrograms in depth (i.e. use
              'np.dstack' as a stacking function), `duplicate_filters` must be
              set to 'True', otherwise they differ in dimensionality.

        """
        from .signal import FramedSignalProcessor
        # stacking parameters
        if stack == 'time':
            stack = np.vstack
        elif stack in ('freq', 'frequency'):
            stack = np.hstack
        elif stack == 'depth':
            stack = np.dstack
        self.stack = stack
        self.stack_diffs = stack_diffs
        # set the duplicate filters
        if self.stack == np.dstack and filterbank is not None:
            if not duplicate_filters:
                import warnings
                warnings.warn("Set 'duplicate_filters' to 'True', otherwise "
                              "the spectrograms can not be stacked in depth.")
            duplicate_filters = True

        # use the same spectrogram for all frame sizes
        sp = SpectrogramProcessor(filterbank=filterbank,
                                  bands=bands, fmin=fmin, fmax=fmax,
                                  norm_filters=norm_filters, log=log,
                                  mul=mul, add=add, diff_ratio=diff_ratio,
                                  duplicate_filters=duplicate_filters,
                                  **kwargs)
        # multiple framing & spectrogram processors
        processor = []
        for frame_size_ in frame_size:
            fs = FramedSignalProcessor(frame_size=frame_size_, fps=fps,
                                       online=online, **kwargs)
            processor.append(SequentialProcessor([fs, sp]))
        # process all specs in parallel
        # FIXME: this does not work with more than 1 threads!
        self.processor = ParallelProcessor(processor, num_threads=1)

    def process(self, data, **kwargs):
        """
        Stack the magnitudes spectrograms (and their differences).

        :param data: Signal instance [Signal]
        :return:     stacked specs (and diffs)

        """
        # process everything
        specs = self.processor.process(data, **kwargs)
        # stack everything (a list of Spectrogram instances was returned)
        stack = []
        for s in specs:
            # always append the spectrogram
            stack.append(s.spec)
            # and the differences only if needed
            if self.stack_diffs:
                stack.append(s.diff)
        # stack them in the given direction and return them
        return self.stack(stack)

    @classmethod
    def add_arguments(cls, parser, stack='freq', stack_diffs=False):
        """
        Add stacking related arguments to an existing parser.

        :param parser:      existing argparse parser
        :param stack:       stacking direction {'time', 'freq', 'depth'}
        :param stack_diffs: also stack the differences [bool]
        :return:            stacking argument parser group


        """
        # add diff related options to the existing parser
        g = parser.add_argument_group('stacking arguments')
        if stack is not None:
            g.add_argument('--stack', action='store', type=str,
                           default=stack,
                           help="stacking direction {'time', 'freq', 'depth'} "
                                "[default=%(default)s]")
        if stack_diffs is not None:
            if stack_diffs:
                g.add_argument('--no_stack_diffs', dest='stack_diffs',
                               action='store_false', default=stack_diffs,
                               help='no not stack the differences of the '
                                    'spectrograms')
            else:
                g.add_argument('--stack_diffs', action='store_true',
                               default=-stack_diffs,
                               help='in addition to the spectrograms, also '
                                    'stack their differences')
        # return the group
        return g
