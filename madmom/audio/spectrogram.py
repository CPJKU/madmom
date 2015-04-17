#!/usr/bin/env python
# encoding: utf-8
"""
This file contains spectrogram related functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import numpy as np
import scipy.fftpack as fft

from madmom import Processor, SequentialProcessor, ParallelProcessor

# filter defaults
from madmom.audio.filters import (LogarithmicFilterbank, BANDS, FMIN, FMAX,
                                  A4, NORM_FILTERS, DUPLICATE_FILTERS)
FILTERBANK = LogarithmicFilterbank
# log defaults
LOG = True
MUL = 1.
ADD = 1.
# diff defaults
DIFF_RATIO = 0.5
DIFF_FRAMES = None
DIFF_MAX_BINS = 1
POSITIVE_DIFF = True


def fft_frequencies(num_fft_bins, sample_rate):
    """
    Frequencies of the FFT bins.

    :param num_fft_bins: number of FFT bins (i.e. half the FFT length)
    :param sample_rate:  sample rate of the signal
    :return:             frequencies of the FFT bins

    """
    return np.fft.fftfreq(num_fft_bins * 2, 1. / sample_rate)[:num_fft_bins]


# functions
def dft(signal, window, fft_size=None, correct_phase=False):
    """
    Calculates the discrete Fourier transform (DFT) of the given signal.

    :param signal:        discrete signal [1D numpy array]
    :param window:        window function [1D numpy array]
    :param fft_size:      use given size for FFT [int, should be a power of 2]
    :param correct_phase: circular shift for correct phase [bool]
    :return:              the complex DFT of the signal

    """
    # multiply the signal frame with the window function
    signal = np.multiply(signal, window)
    # only shift and perform complex DFT if needed
    if correct_phase:
        # circular shift the signal (needed for correct phase)
        shift = len(window) >> 1
        signal = np.concatenate((signal[shift:], signal[:shift]))
    # perform DFT and return the signal
    return fft.fft(signal, fft_size)[:fft_size >> 1]


def stft(signal, window, hop_size, fft_size=None, correct_phase=False):
    """
    Calculates the complex Short-Time Fourier Transform (STFT) of the given
    signal.

    :param signal:        signal [1D numpy array]
    :param window:        window function [1D numpy array]
    :param hop_size:      hop size between adjacent frames [float]
    :param fft_size:      use given size for FFT [int, should be a power of 2]
    :param correct_phase: circular shift for correct phase [bool]
    :return:              the complex STFT of the signal

    Note: the window is centered around the current sample and the total length
          of the STFT is calculated such that the last frame still covers some
          signal.

    """
    from .signal import FramedSignal
    # slice the signal into frames
    frames = FramedSignal(signal, len(window), hop_size)
    # number of FFT bins
    if fft_size is None:
        num_fft_bins = len(window) >> 1
    else:
        num_fft_bins = fft_size >> 1
    # init STFT matrix
    stft = np.empty((len(frames), num_fft_bins), np.complex)
    for f, frame in enumerate(frames):
        # perform DFT
        stft[f] = dft(frame, window, fft_size, correct_phase)
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


# Spectrogram class
class Spectrogram(object):
    """
    Spectrogram Class.

    """
    def __init__(self, frames, window=np.hanning, fft_size=None,
                 block_size=2048, filterbank=None, bands=BANDS,
                 fmin=FMIN, fmax=FMAX, fref=A4, norm_filters=NORM_FILTERS,
                 duplicate_filters=DUPLICATE_FILTERS, log=False, mul=MUL,
                 add=ADD, diff_ratio=DIFF_RATIO, diff_frames=DIFF_FRAMES,
                 diff_max_bins=DIFF_MAX_BINS, positive_diff=POSITIVE_DIFF,
                 **kwargs):
        """
        Creates a new Spectrogram instance of the given audio.

        :param frames:            FramedSignal instance (or anything a
                                  FramedSignal can be instantiated from)

        FFT parameters:

        :param window:            window function [numpy ufunc or numpy array]
        :param fft_size:          use this size for FFT [int, power of 2]
        :param block_size:        perform some operations (e.g. filtering) in
                                  blocks of this size [int, power of 2]

        Filterbank parameters:

        :param filterbank:        filterbank type [Filterbank]
        :param bands:             number of filter bands (per octave, depending
                                  on the type of the filterbank)
        :param fmin:              the minimum frequency [Hz]
        :param fmax:              the maximum frequency [Hz]
        :param fref:              tuning frequency [Hz]
        :param norm_filters:      normalize the filter to area 1 [bool]
        :param duplicate_filters: keep duplicate filters [bool]

        Logarithmic magnitude parameters:

        :param log:               scale the magnitude spectrogram
                                  logarithmically [bool]
        :param mul:               multiply the magnitude spectrogram with this
                                  factor before taking the logarithm [float]
        :param add:               add this value before taking the logarithm
                                  of the magnitudes [float]

        Difference parameters:

        :param diff_ratio:        calculate the difference to the frame at
                                  which the window used for the STFT yields
                                  this ratio of the maximum height [float]
        :param diff_frames:       calculate the difference to the N-th previous
                                  frame [int] (if set, this overrides the value
                                  calculated from the `diff_ratio`)
        :param diff_max_bins:     apply a maximum filter with this width (in
                                  bins in frequency dimension) before
                                  calculating the diff; (e.g. for the
                                  difference spectrogram of the SuperFlux
                                  algorithm 3 `max_bins` are used together
                                  with a 24 band logarithmic filterbank)
        :param positive_diff:     keep only the positive differences,
                                  i.e. set all diff values < 0 to 0.

        If no FramedSignal instance was given, one is instantiated and these
        arguments are passed:

        :param args:              arguments passed to FramedSignal
        :param kwargs:            keyword arguments passed to FramedSignal

        Note: `fft_size` and `block_size` should be a power of 2.

        """
        from .signal import FramedSignal

        # framed signal stuff
        if isinstance(frames, FramedSignal):
            # already a FramedSignal
            self.frames = frames
        else:
            # try to instantiate a FramedSignal object
            self.frames = FramedSignal(frames, **kwargs)

        # FFT stuff
        # determine which window to use
        if hasattr(window, '__call__'):
            # if only function is given, use the size to the audio frame size
            self.window = window(self.frames.frame_size)
        elif isinstance(window, np.ndarray):
            # otherwise use the given window directly
            self.window = window
            if self.window.size != self.frames.frame_size:
                raise ValueError('Window must have the same size as the audio '
                                 'frames.')
        else:
            # other types are not supported
            raise TypeError("Invalid window type.")
        # window used for DFT
        try:
            # the audio signal is not scaled, scale the window accordingly
            max_range = np.iinfo(self.frames.signal.dtype).max
            self.fft_window = self.window / max_range
        except ValueError:
            self.fft_window = self.window
        # DFT size
        if fft_size is None:
            self.fft_size = self.window.size
        else:
            self.fft_size = fft_size
        # perform some calculations (e.g. filtering) in blocks of that size
        self.block_size = block_size

        # filterbank stuff
        # TODO: add option to automatically calculate `fref`
        if filterbank is not None:
            # create a filterbank of the given type
            filterbank = filterbank(self.fft_freqs, bands=bands, fmin=fmin,
                                    fmax=fmax, fref=fref,
                                    norm_filters=norm_filters,
                                    duplicate_filters=duplicate_filters)
        # save the filterbank so it gets used when calculating the STFT
        self.filterbank = filterbank

        # log stuff
        self.log = log
        self.mul = mul
        self.add = add

        # diff stuff
        # calculate the number of diff frames to use
        if not diff_frames:
            # calculate on basis of the diff_ratio
            # get the first sample with a higher magnitude than given ratio
            sample = np.argmax(self.window > diff_ratio * max(self.window))
            diff_samples = self.window.size / 2 - sample
            # convert to frames
            diff_frames = int(round(diff_samples / self.frames.hop_size))
        # always set the minimum to 1
        if diff_frames < 1:
            diff_frames = 1
        self.num_diff_frames = diff_frames
        # bins for maximum filter
        self.diff_max_bins = diff_max_bins
        # keep only the positive differences?
        self.positive_diff = positive_diff

        # init hidden variables
        self._spec = None
        self._stft = None
        self._phase = None
        self._lgd = None
        self._diff = None

    @property
    def num_frames(self):
        """Number of frames."""
        return len(self.frames)

    @property
    def num_fft_bins(self):
        """Number of FFT bins."""
        return self.fft_size >> 1

    @property
    def num_bins(self):
        """Number of bins of the spectrogram."""
        if self.filterbank is None:
            return self.num_fft_bins
        else:
            return self.filterbank.shape[1]

    @property
    def fft_freqs(self):
        """Frequencies of the FFT bins."""
        return fft_frequencies(self.num_fft_bins,
                               self.frames.signal.sample_rate)

    @property
    def bin_freqs(self):
        """Frequencies of the spectrogram bins."""
        if self.filterbank is None:
            return self.fft_freqs
        else:
            return self.filterbank.center_frequencies

    def compute_stft(self, complex_stft=False, block_size=None):
        """
        This is a memory saving method to batch-compute different spectrograms.

        :param complex_stft: save the complex_stft STFT to the "stft" attribute
        :param block_size:   perform some operations (e.g. filtering) in blocks
                             of this size [int, should be a power of 2]

        """
        # cache variables
        num_frames = self.num_frames
        fft_size = self.fft_size
        fft_window = self.fft_window
        fft_shift = len(fft_window) >> 1
        num_fft_bins = self.num_fft_bins
        num_bins = self.num_bins

        # init STFT matrix
        if complex_stft:
            self._stft = np.empty((num_frames, num_fft_bins), np.complex64)
        # init spectrogram matrix
        self._spec = np.empty((num_frames, num_bins), np.float32)

        # process in blocks
        if self.filterbank is not None:
            if block_size is None:
                block_size = self.block_size
            if not block_size or block_size > num_frames:
                block_size = num_frames
            # init a matrix of that size
            block = np.zeros([block_size, num_fft_bins])

        # calculate DFT for all frames
        for f, frame in enumerate(self.frames):
            # multiply the signal frame with the window function
            signal = np.multiply(frame, fft_window)
            # only shift and perform complex DFT if needed
            if complex_stft:
                # circular shift the signal (needed for correct phase)
                signal = np.concatenate((signal[fft_shift:],
                                         signal[:fft_shift]))
            # perform DFT and return the signal
            dft_signal = fft.fft(signal, fft_size)[:num_fft_bins]

            # save the complex STFT
            if complex_stft:
                self._stft[f] = dft_signal

            # is block wise processing needed?
            if self.filterbank is None:
                # no filtering needed, thus no block wise processing needed
                self._spec[f] = np.abs(dft_signal)
            else:
                # filter the magnitude spectrogram in blocks
                block[f % block_size] = np.abs(dft_signal)
                # if the end of a block or end of the signal is reached
                end_of_block = (f + 1) % block_size == 0
                end_of_signal = (f + 1) == num_frames
                if end_of_block or end_of_signal:
                    start = f // block_size * block_size
                    self._spec[start:f + 1] = np.dot(block[:f % block_size + 1],
                                                     self.filterbank)

        # take the logarithm of the magnitude spectrogram if needed (inplace)
        if self.log:
            np.log10(self.mul * self._spec + self.add, out=self._spec)

    @property
    def stft(self):
        """Short Time Fourier Transform of the signal."""
        # TODO: this is highly inefficient if other properties depending on the
        #       STFT were accessed previously; better call compute_stft() with
        #       appropriate parameters.
        # compute STFT if needed
        if self._stft is None:
            self.compute_stft(complex_stft=True)
        return self._stft

    @property
    def phase(self):
        """Phase of the STFT."""
        # compute phase if needed
        if self._phase is None:
            # TODO: this also stores the STFT, which might not be needed
            self._phase = phase(self.stft).astype(np.float32)
        # return phase
        return self._phase

    @property
    def lgd(self):
        """Local group delay of the STFT."""
        # compute the local group delay if needed
        if self._lgd is None:
            # TODO: this also stores the phase, which might not be needed
            self._lgd = local_group_delay(self.phase).astype(np.float32)
        # return lgd
        return self._lgd

    @property
    def spec(self):
        """Magnitude spectrogram of the STFT."""
        # compute spec if needed
        if self._spec is None:
            # check if STFT was computed already
            if self._stft is not None:
                # use it
                self._spec = np.abs(self._stft)
                # filter if needed
                if self.filterbank is not None:
                    self._spec = np.dot(self._spec, self.filterbank)
                # take the logarithm
                if self.log:
                    self._spec = np.log10(self.mul * self._spec + self.add)
            else:
                # compute the spec
                self.compute_stft()
        # return spec
        return self._spec

    # alias
    magnitude = spec

    @property
    def diff(self):
        """Differences of the magnitude spectrogram."""
        if self._diff is None:
            # init array
            self._diff = np.zeros_like(self.spec)
            # apply a maximum filter if needed
            if self.diff_max_bins > 1:
                from scipy.ndimage.filters import maximum_filter
                # widen the spectrogram in frequency dimension by `max_bins`
                max_spec = maximum_filter(self.spec,
                                          size=[1, self.diff_max_bins])
            else:
                max_spec = self.spec
            # calculate the diff
            df = self.num_diff_frames
            self._diff[df:] = self.spec[df:] - max_spec[:-df]
            # positive differences only?
            if self.positive_diff:
                np.maximum(self._diff, 0, self._diff)
        # return diff
        return self._diff


# Spectrogram Processor class
class SpectrogramProcessor(Processor):
    """
    Spectrogram Processor Class.

    """

    def __init__(self, filterbank=FILTERBANK, bands=BANDS, fmin=FMIN, fmax=FMAX,
                 norm_filters=NORM_FILTERS, log=LOG, mul=MUL, add=ADD,
                 diff_ratio=DIFF_RATIO, diff_frames=DIFF_FRAMES,
                 diff_max_bins=DIFF_MAX_BINS, **kwargs):
        """
        Creates a new SpectrogramProcessor instance.

        Magnitude spectrogram filtering parameters:

        :param filterbank:    filter the magnitude spectrogram with a
                              filterbank of this type [None or Filterbank]
        :param bands:         use N bands (per octave) [int]
        :param fmin:          minimum frequency of the filterbank [float]
        :param fmax:          maximum frequency of the filterbank [float]
        :param norm_filters:  normalize the filter to area 1 [bool]

        Magnitude spectrogram scaling parameters:

        :param log:           take the logarithm of the magnitude [bool]
        :param mul:           multiply the spectrogram with this factor before
                              taking the logarithm of the magnitudes [float]
        :param add:           add this value before taking the logarithm of
                              the magnitudes [float]

        Magnitude spectrogram difference parameters:

        :param diff_ratio:    calculate the difference to the frame at which
                              the window used for the STFT yields this ratio
                              of the maximum height [float]
        :param diff_frames:   calculate the difference to the N-th previous
                              frame [int] (if set, this overrides the value
                              calculated from the `diff_ratio`)
        :param diff_max_bins: apply a maximum filter with this width (in bins
                              in frequency dimension) before calculating the
                              diff; (e.g. for the difference spectrogram of
                              the SuperFlux algorithm 3 `max_bins` are used
                              together with a 24 band logarithmic filterbank)

        Note: `filterbank` also accepts the boolean values 'False' and 'True'
              which get translated to no filterbank or a filterbank of the
              default type ('LogarithmicFilterbank').

        """
        # filterbank stuff
        # TODO: add literal values?
        if filterbank is True:
            filterbank = LogarithmicFilterbank
        elif filterbank is False:
            filterbank = None
        self.filterbank = filterbank
        self.bands = bands
        self.fmin = fmin
        self.fmax = fmax
        self.norm_filters = norm_filters
        # log stuff
        self.log = log
        self.mul = mul
        self.add = add
        # diff stuff
        self.diff_ratio = diff_ratio
        self.diff_frames = diff_frames
        self.diff_max_bins = diff_max_bins

    def process(self, data):
        """
        Perform FFT on a framed signal and return the spectrogram.

        :param data: frames to be processed [FramedSignal]
        :return:     Spectrogram instance

        """
        # instantiate a Spectrogram
        return Spectrogram(data, filterbank=self.filterbank, bands=self.bands,
                           fmin=self.fmin, fmax=self.fmax,
                           norm_filters=self.norm_filters,
                           log=self.log, mul=self.mul, add=self.add,
                           diff_ratio=self.diff_ratio,
                           diff_frames=self.diff_frames,
                           diff_max_bins=self.diff_max_bins)

    @classmethod
    def add_fft_arguments(cls, parser, window=None, fft_size=None):
        """
        Add spectrogram related arguments to an existing parser.

        :param parser:      existing argparse parser
        :param window:      window function
        :param fft_size:    use this size for FFT [int, should be a power of 2]
        :return:            spectrogram argument parser group

        Parameters are included in the group only if they are not 'None'.

        """
        # add filterbank related options to the existing parser
        g = parser.add_argument_group('spectrogram arguments')
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

    @classmethod
    def add_filter_arguments(cls, parser, filterbank=FILTERBANK, bands=BANDS,
                             fmin=FMIN, fmax=FMAX, norm_filters=NORM_FILTERS):
        """
        Add spectrogram filtering related arguments to an existing parser.

        :param parser:       existing argparse parser
        :param filterbank:   filter the magnitude spectrogram with a
                             logarithmic filterbank [bool]
        :param bands:        use N bands per octave [int]
        :param fmin:         minimum frequency of the filterbank [float]
        :param fmax:         maximum frequency of the filterbank [float]
        :param norm_filters: normalize the filter to area 1 [bool]
        :return:             spectrogram filtering argument parser group

        Parameters are included in the group only if they are not 'None'.

        """
        # add filterbank related options to the existing parser
        g = parser.add_argument_group('spectrogram filtering arguments')
        if filterbank is not None:
            # TODO: add literal values
            if filterbank:
                g.add_argument('--no_filter', dest='filterbank',
                               action='store_false',
                               default=LogarithmicFilterbank,
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
        # return the group
        return g

    @classmethod
    def add_log_arguments(cls, parser, log=None, mul=None, add=None):
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

    @classmethod
    def add_diff_arguments(cls, parser, diff_ratio=None,
                           diff_frames=None, diff_max_bins=None):
        """
        Add spectrogram difference related arguments to an existing parser.

        :param parser:        existing argparse parser
        :param diff_ratio:    calculate the difference to the frame at which
                              the window of the STFT have this ratio of the
                              maximum height [float]
        :param diff_frames:   calculate the difference to the N-th previous
                              frame [int] (if set, this overrides the value
                              calculated from the `diff_ratio`)
        :param diff_max_bins: apply a maximum filter with this width (in bins
                              in frequency dimension) before calculating the
                              diff; (e.g. for the difference spectrogram of
                              the SuperFlux algorithm 3 `max_bins` are used
                              together with a 24 band logarithmic filterbank)
        :return:              spectrogram difference argument parser group

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
                                'of the maximum height [default=%(default).1f]')
        if diff_ratio is not None or diff_frames:
            g.add_argument('--diff_frames', action='store', type=int,
                           default=diff_frames,
                           help='calculate the difference to the N-th previous '
                                'frame (this overrides the value calculated '
                                'with `diff_ratio`) [default=%(default)s]')
        if diff_max_bins is not None:
            g.add_argument('--diff_max_bins', action='store', type=int,
                           default=diff_max_bins,
                           help='apply a maximum filter with this width '
                                '(in frequency bins) before calculating the '
                                'diff [default=%(default)d]')
        # return the group
        return g


class SuperFluxProcessor(SpectrogramProcessor):
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
        filterbank = kwargs.pop('filterbank', LogarithmicFilterbank)
        bands = kwargs.pop('bands', 24)
        norm_filters = kwargs.pop('norm_filters', False)
        # log magnitudes
        log = kwargs.pop('log', True)
        # we want max filtered diffs
        diff_ratio = kwargs.pop('diff_ratio', 0.5)
        diff_max_bins = kwargs.pop('diff_max_bins', 3)
        # instantiate SpectrogramProcessor
        super(SuperFluxProcessor, self).__init__(
            filterbank=filterbank, bands=bands, norm_filters=norm_filters,
            log=log, diff_ratio=diff_ratio, diff_max_bins=diff_max_bins,
            **kwargs)


class MultiBandSpectrogramProcessor(SpectrogramProcessor):
    """
    Spectrogram processor which combines the differences of a log filtered
    spectrogram into multiple bands.

    """

    def __init__(self, crossover_frequencies, norm_bands=False, diff=True,
                 **kwargs):
        """

        :param crossover_frequencies: list of crossover frequencies at which
                                      the spectrogram is split into bands
        :param norm_bands:            normalize the bands [bool]
        :param diff:                  use the differences of the magnitude
                                      spectrogram [bool] (see below)

        The combination of the spectrogram magnitudes or differences into
        multiple bands can be controlled with the `diff` parameter. It can
        have these values:
            - 'False':    do not use the differences, combine the spectrogram
            - 'True':     combine only the differences, not the magnitudes

        """
        # instantiate SpectrogramProcessor
        super(MultiBandSpectrogramProcessor, self).__init__(**kwargs)
        self.crossover_frequencies = crossover_frequencies
        self.norm_bands = norm_bands
        self.diff = diff

    def process(self, data):
        """
        Perform FFT on a framed signal and return the a multi-band
        representation of either the magnitude spectrogram or the differences
        thereof.

        :param data: frames to be processed [FramedSignal]
        :return:     Spectrogram instance

        """
        # instantiate a Spectrogram
        data = Spectrogram(data, filterbank=self.filterbank, bands=self.bands,
                           fmin=self.fmin, fmax=self.fmax,
                           norm_filters=self.norm_filters,
                           log=self.log, mul=self.mul, add=self.add,
                           diff_ratio=self.diff_ratio,
                           diff_frames=self.diff_frames,
                           diff_max_bins=self.diff_max_bins)
        # TODO: move this to filterbank and make it accept a list of bin
        #       frequencies (data.bin_freqs) to generate a rectangular filter
        # create an empty filterbank
        fb = np.zeros((data.num_bins, len(self.crossover_frequencies) + 1))
        # get the closest cross over bins
        freq_distance = (data.bin_freqs -
                         np.asarray(self.crossover_frequencies)[:, np.newaxis])
        crossover_bins = np.argmin(np.abs(freq_distance), axis=1)
        # prepend index 0 and append length of the filterbank
        crossover_bins = np.r_[0, crossover_bins, len(fb)]
        # map the spectrogram bins to the filterbank bands
        for i in range(fb.shape[1]):
            fb[crossover_bins[i]:crossover_bins[i + 1], i] = 1
        # normalize it
        if self.norm_bands:
            fb /= np.sum(fb, axis=0)
        # filter the spec or the diff a second time
        # TODO: should this class always act on just the differences? filtering
        #       the spec a second time is not the most meaningful thing...
        if self.diff:
            return np.dot(data.diff, fb)
        else:
            return np.dot(data.spec, fb)

    @classmethod
    def add_multi_band_arguments(cls, parser, crossover_frequencies=None,
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
            from madmom.utils import OverrideDefaultTypedListAction
            g.add_argument('--crossover_frequencies', list_type=float,
                           action=OverrideDefaultTypedListAction,
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
    Stack spec & diff.

    """
    def __init__(self, frame_sizes, fps=100, online=False,
                 filterbank=FILTERBANK, bands=BANDS, fmin=FMIN, fmax=FMAX,
                 norm_filters=NORM_FILTERS, log=LOG, mul=MUL, add=ADD,
                 diff_ratio=DIFF_RATIO, **kwargs):
        """
        Creates a new StackSpectrogramProcessor instance.

        Multiple magnitude spectra (with different FFT sizes) are filtered and
        logarithmically scaled before being stacked together with their first
        order differences.

        Framing parameters:

        :param frame_sizes:   include spectrogram with these frame sizes
                              [list of int]
        :param fps:           frames per second [float]
        :param online:        online mode [bool]

        Filterbank parameters:

        :param filterbank:    filter the magnitude spectrogram with a
                              filterbank of this type [None or Filterbank]
        :param bands:         use N bands per octave [int]
        :param fmin:          minimum frequency of the filterbank [float]
        :param fmax:          maximum frequency of the filterbank [float]
        :param norm_filters:  normalize the filter to area 1 [bool]

        Logarithmic magnitude parameters:

        :param mul:           multiply the spectrogram with this factor before
                              taking the logarithm of the magnitudes [float]
        :param add:           add this value before taking the logarithm of
                              the magnitudes [float]

        Difference parameters:

        :param diff_ratio:    calculate the difference to the frame at which
                              the window used for the STFT yields this ratio
                              of the maximum height [float]

        """
        from .signal import FramedSignalProcessor
        # use the same spec for all frame sizes
        sp = SpectrogramProcessor(filterbank=filterbank,
                                  bands=bands, fmin=fmin, fmax=fmax,
                                  norm_filters=norm_filters, log=log,
                                  mul=mul, add=add, diff_ratio=diff_ratio,
                                  **kwargs)
        # multiple framing & spec processors
        processor = []
        for frame_size in frame_sizes:
            fs = FramedSignalProcessor(frame_size=frame_size, fps=fps,
                                       online=online, **kwargs)
            processor.append(SequentialProcessor([fs, sp]))
        # process all specs in parallel
        # FIXME: this does not work with more than 1 threads!
        self.processor = ParallelProcessor(processor, num_threads=1)

    def process(self, data):
        """
        Stack the spectrograms and stack their magnitudes and differences.

        :param data: Signal instance [Signal]
        :return:     stacked specs and diffs

        """
        # process everything
        specs = self.processor.process(data)
        # stack everything (a list of Spectrogram instances was returned)
        stack = []
        for s in specs:
            stack.extend([s.spec, s.diff])
        return np.hstack(stack)
