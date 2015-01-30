#!/usr/bin/env python
# encoding: utf-8
"""
This file contains spectrogram related functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import numpy as np
import scipy.fftpack as fft

from madmom import Processor
from .filters import fft_frequencies, A4


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
    :param hop_size:      hop_size between adjacent frames [float]
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
    stft = np.empty([len(frames), num_fft_bins], np.complex)
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
    raise Warning("check if adaptive_whitening returns meaningful results")
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
    T. Lidy and A. Rauber
    Proceedings of the 6th International Conference on Music Information
    Retrieval (ISMIR 2005), London, UK, September 2005

    """
    from scipy.stats import skew, kurtosis
    return {'mean': np.mean(spec, axis=0),
            'median': np.median(spec, axis=0),
            'variance': np.var(spec, axis=0),
            'skewness': skew(spec, axis=0),
            'kurtosis': kurtosis(spec, axis=0),
            'min': np.min(spec, axis=0),
            'max': np.max(spec, axis=0)}


def tuning_frequency(spec, sample_rate, num_hist_bins=15, fref=A4):
    """
    Determines the tuning frequency based on the given (peak) magnitude
    spectrogram.

    :param spec:          (peak) magnitude spectrogram [numpy array]
    :param sample_rate:   sample rate of the audio file [Hz]
    :param num_hist_bins: number of histogram bins
    :param fref:          reference tuning frequency [Hz]
    :return:              tuning frequency

    """
    # frequencies of the bins
    bin_freqs = fft_frequencies(spec.shape[1], sample_rate)
    # interval of spectral bins from the reference frequency in semitones
    semitone_int = 12. * np.log2(bin_freqs / fref)
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
    # default values
    filterbank = None
    log = None

    def __init__(self, frames, window=np.hanning, norm_window=False,
                 fft_size=None, *args, **kwargs):
        """
        Creates a new Spectrogram instance of the given audio.

        :param frames:      FramedSignal instance (or anything a FramedSignal
                            can be instantiated from)

        FFT parameters:

        :param window:      window function
        :param norm_window: set area of window function to 1 [bool]
        :param fft_size:    use this size for FFT [int, should be a power of 2]

        If no FramedSignal instance was given, one is instantiated and these
        arguments are passed:

        :param args:        arguments passed to FramedSignal
        :param kwargs:      keyword arguments passed to FramedSignal


        """
        from .signal import FramedSignal
        # audio signal stuff
        if isinstance(frames, FramedSignal):
            # already a FramedSignal
            self.frames = frames
        else:
            # try to instantiate a FramedSignal object
            self.frames = FramedSignal(frames, *args, **kwargs)

        # determine window to use
        if hasattr(window, '__call__'):
            # if only function is given, use the size to the audio frame size
            self.window = window(self.frames.frame_size)
        elif isinstance(window, np.ndarray):
            # otherwise use the given window directly
            self.window = window
        else:
            # other types are not supported
            raise TypeError("Invalid window type.")
        # normalize the window if needed
        if norm_window:
            self.window /= np.sum(self.window)
        # window used for DFT
        try:
            # the audio signal is not scaled, scale the window accordingly
            max_value = np.iinfo(self.frames.signal.dtype).max
            self.fft_window = self.window / max_value
        except ValueError:
            self.fft_window = self.window

        # parameters used for the DFT
        if fft_size is None:
            self.fft_size = self.window.size
        else:
            self.fft_size = fft_size

        # init matrices
        self._spec = None
        self._stft = None
        self._phase = None
        self._lgd = None

    @property
    def num_frames(self):
        """Number of frames."""
        return len(self.frames)

    @property
    def fft_freqs(self):
        """Frequencies of the FFT bins."""
        return fft_frequencies(self.num_fft_bins, self.frames.sample_rate)

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

    def compute_stft(self, complex_stft=False):
        """
        This is a memory saving method to batch-compute different spectrograms.

        :param complex_stft: save the complex_stft STFT to the "stft"
        attribute

        """
        # init STFT matrix
        if complex_stft:
            self._stft = np.empty([self.num_frames, self.num_fft_bins],
                                  dtype=np.complex64)
        # init spectrogram matrix
        self._spec = np.empty([self.num_frames, self.num_fft_bins],
                              dtype=np.float32)

        # calculate DFT for all frames
        for f, frame in enumerate(self.frames):
            # perform DFT
            _dft = dft(frame, self.fft_window, self.fft_size, complex_stft)
            # save the complex STFT
            if complex_stft:
                self._stft[f] = _dft
            # save the magnitude spec
            self._spec[f] = np.abs(_dft)

        # filter the magnitude spectrogram if needed
        if self.filterbank is not None:
            self._spec = np.dot(self._spec, self.filterbank)

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
            self.compute_stft(raw_stft=True)
        return self._stft

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
                if self._filterbank is not None:
                    self._spec = np.dot(self._spec, self._filterbank)
                # take the logarithm
                if self._log:
                    self._spec = np.log10(self._mul * self._spec + self._add)
            else:
                # compute the spec
                self.compute_stft()
        # return spec
        return self._spec

    # alias
    magnitude = spec

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


class FilteredSpectrogram(Spectrogram):
    """
    FilteredSpectrogram is a subclass of Spectrogram which filters the
    magnitude spectrogram based on the given filterbank.

    """
    from .filters import (LogarithmicFilterbank, FMIN, FMAX, NORM_FILTERS,
                          A4, DUPLICATE_FILTERS)

    def __init__(self, frames, filterbank=LogarithmicFilterbank,
                 bands=LogarithmicFilterbank.BANDS_PER_OCTAVE,
                 fmin=FMIN, fmax=FMAX, fref=A4, norm_filters=NORM_FILTERS,
                 duplicate_filters=DUPLICATE_FILTERS, *args, **kwargs):
        """
        Creates a new FilteredSpectrogram instance.

        :param frames:            FramedSignal instance (or anything a
                                  FramedSignal can be instantiated from)

        This class creates a filterbank with these parameters and passes it to
        Spectrogram.

        :param filterbank:        filterbank type
        :param bands:             number of filter bands (per octave, depending
                                  on the type of the filterbank)
        :param fmin:              the minimum frequency [Hz]
        :param fmax:              the maximum frequency [Hz]
        :param fref:              tuning frequency [Hz]
        :param norm_filters:      normalize the filter to area 1 [bool]
        :param duplicate_filters: keep duplicate filters [bool]

        Additional arguments passed to Spectrogram:

        :param args:              arguments passed to Spectrogram
        :param kwargs:            keyword arguments passed to Spectrogram

        Note: if the `filterbank` parameter is set to 'None' no filterbank will
              be used, i.e. the STFT bins remain linearly spaced and unaltered.


        """
        # instantiate a Spectrogram with the additional arguments
        super(FilteredSpectrogram, self).__init__(frames, *args, **kwargs)
        # TODO: add option to automatically calculate `fref`
        # create a filterbank
        if filterbank is not None:
            fb = filterbank(self.num_fft_bins, self.frames.sample_rate,
                            bands=bands, fmin=fmin, fmax=fmax, fref=fref,
                            norm_filters=norm_filters,
                            duplicate_filters=duplicate_filters)
            # save the filterbank so it gets used when calculating the STFT
            self.filterbank = fb

# aliases
FiltSpec = FilteredSpectrogram


class LogarithmicFilteredSpectrogram(FilteredSpectrogram):
    """
    LogarithmicFilteredSpectrogram is a subclass of FilteredSpectrogram which
    filters the magnitude spectrogram based on the given filterbank and
    converts it to a logarithmic scale.

    """
    def __init__(self, frames, log=True, mul=1, add=0, *args, **kwargs):
        """
        Creates a new LogarithmicFilteredSpectrogram instance.

        :param frames: FramedSignal instance (or anything a FramedSignal can
                       be instantiated from)
        This class sets these parameters for logarithmically scaled magnitudes
        and passes them to FilteredSpectrogram.

        :param mul:    multiply the magnitude spectrogram with given value
        :param add:    add the given value to the magnitude spectrogram

        Additional arguments passed to FilteredSpectrogram:

        :param args:   arguments passed to FilteredSpectrogram
        :param kwargs: keyword arguments passed to FilteredSpectrogram

        Note: if the `log` parameter is set to 'False' or 'None', the magnitude
              spectrogram will remain in a linear scale.

        """
        # create a FilteredSpectrogram object
        super(LogarithmicFilteredSpectrogram, self).__init__(frames, *args,
                                                             **kwargs)
        # save the log parameters so they get used when calculating the STFT
        self.log = log
        self.mul = mul
        self.add = add

# aliases
LogFiltSpec = LogarithmicFilteredSpectrogram


# Diff of the spectrogram
class SpectrogramDifference(object):
    """
    Class for calculating the difference of a magnitude spectrogram.

    """
    def __init__(self, spectrogram, ratio=0.5, diff_frames=None,
                 diff_type=None, *args, **kwargs):
        """
        Creates a new Spectrogram instance of the given audio.

        :param spectrogram: Spectrogram instance (or anything a Spectrogram can
                            be instantiated from)

        Diff parameters:

        :param ratio:       calculate the difference to the frame at which the
                            window of the STFT have this ratio of the maximum
                            height [float]
        :param diff_frames: calculate the difference to the N-th previous frame
                            [int] (if set, this overrides the value calculated
                            from the ratio)

        If no Spectrogram instance was given, one is instantiated and these
        arguments are passed:

        :param args:        arguments passed to Spectrogram
        :param kwargs:      keyword arguments passed to Spectrogram

        """
        # spectrogram handling
        if isinstance(spectrogram, Spectrogram):
            # already a signal
            self.spectrogram = Spectrogram
        else:
            # try to instantiate a SignalProcessor
            self.spectrogram = Spectrogram(spectrogram, *args, **kwargs)

        # init variables
        self._diff = None
        # calculate the number of diff frames to use
        self.ratio = ratio
        if not diff_frames:
            # calculate on basis of the ratio
            # use the window and hop size of the Spectrogram and FramedSignal
            window = self.spectrogram.window
            hop_size = self.spectrogram.frames.hop_size
            # get the first sample with a higher magnitude than given ratio
            sample = np.argmax(window > self.ratio * max(window))
            diff_samples = window.size / 2 - sample
            # convert to frames
            diff_frames = int(round(diff_samples / hop_size))
        # always set the minimum to 1
        if diff_frames < 1:
            diff_frames = 1
        self.num_diff_frames = diff_frames
        # type of diff
        self.diff_type = diff_type

    @property
    def diff(self):
        """Differences of the magnitude spectrogram."""
        if self._diff is None:
            # cache Spectrogram reference
            s = self.spectrogram
            # init array
            self._diff = np.empty_like(s.spec)
            # calculate the diff
            df = self.num_diff_frames
            self._diff[df:] = s.spec[df:] - s.spec[:-df]
            # fill the first frames with the values of the spec
            self._diff[:df] = s.spec[:df]
            # calculate diff (backwards)
            for i in range(df - 1, -1, -1):
                # get the correct frame of the FramedSignal
                frame = s.frames[i - df]
                # perform DFT
                frame_ = np.abs(dft(frame, s._fft_window, s.fft_size))
                # filter with a filterbank?
                if s.filterbank is not None:
                    frame_ = np.dot(frame_, s.filterbank)
                # subtract everything from the existing value
                self._diff[i] = frame_
        # return diff
        return self._diff

    @property
    def pos_diff(self):
        """Positive differences of the magnitude spectrogram."""
        # return only the positive elements of the diff
        return np.maximum(self.diff, 0)


# Spectrogram Processor class
class SpectrogramProcessor(Processor):
    """
    Spectrogram Class.

    """
    # filter defaults
    FILTERBANK = False
    BANDS = 6
    FMIN = 30
    FMAX = 17000
    # log defaults
    LOG = False
    MUL = 1
    ADD = 0
    # diff defaults
    DIFF = False
    RATIO = 0.5
    DIFF_FRAMES = None
    DIFF_TYPE = 'normal'

    def __init__(self, filterbank=FILTERBANK, bands=BANDS, fmin=FMIN,
                 fmax=FMAX, log=LOG, mul=MUL, add=ADD, diff=DIFF, ratio=RATIO,
                 diff_frames=DIFF_FRAMES, diff_type=DIFF_TYPE):
        """
        Creates a new SpectrogramProcessor instance.

        Magnitude spectrogram filtering parameters:

        :param filterbank:  filter the magnitude spectrogram with a filterbank
                            of this type [None or Filterbank]
        :param bands:       use N bands per octave [int]
        :param fmin:        minimum frequency of the filterbank [float]
        :param fmax:        maximum frequency of the filterbank [float]

        Magnitude spectrogram scaling parameters:

        :param log:         take the logarithm of the magnitude [bool]
        :param mul:         multiply the spectrogram with this factor before
                            taking the logarithm of the magnitudes [float]
        :param add:         add this value before taking the logarithm of the
                            magnitudes [float]

        Magnitude spectrogram difference parameters:

        :param diff:        use the differences of the magnitude spectrogram
                            {None, False, True, 'stack'} (see below)
        :param ratio:       calculate the difference to the frame which window
                            overlaps to this ratio [float]
        :param diff_frames: calculate the difference to the N-th previous frame
                            [int] (if set, this overrides the value calculated
                            from the ratio)
        :param diff_type:   type of the differences {'normal', 'superflux'}

        The spectrogram difference can be controlled with the following
        parameters. The `diff` parameter can have these values:
          - None, False:    do not use the differences of the spectrogram
          - True:           use only the differences, not the spectrogram
          - 'stack':        stack the differences on top of the spectrogram
        The `diff_type` parameter can have these values:
          - None, 'normal': use the normal difference
          - 'superflux':    use a maximum filtered difference (see SuperFlux)


        """
        # how to alter the magnitude spectrogram
        self.filterbank = filterbank
        self.bands = bands
        self.fmin = fmin
        self.fmax = fmax
        self.log = log
        self.mul = mul
        self.add = add
        self.diff = diff
        self.ratio = ratio
        self.diff_frames = diff_frames
        self.diff_type = diff_type

    def process(self, data):
        """
        Compute the (magnitude, complex, phase, lgd) spectrograms of the data.

        This is a memory saving method to batch-compute different spectrograms.

        :param data:   data [2D numpy array or FramedSignal]
                       if any other data is given, the method tries to
                       instantiate a FramedSignal form it and passes
                       additional (keyword) arguments to FramedSignal()
        :return:       Spectrogram instance

        """
        # always try to instantiate a LogarithmicFilteredSpectrogram, since
        # this can also ignore the filter and log parameters
        data = LogarithmicFilteredSpectrogram(
            data, filterbank=self.filterbank, bands=self.bands, fmin=self.fmin,
            fmax=self.fmax, log=self.log, mul=self.mul, add=self.add)
        if self.diff:
            # use the diff, so instantiate a SpectrogramDifference
            data = SpectrogramDifference(data, ratio=self.ratio,
                                         diff_frames=self.diff_frames,
                                         diff_type=self.diff_type)
            if self.diff == 'stack':
                # return stacked spec and diff
                return np.hstack((data.spectrogram.spec, data.pos_diff))
            else:
                # return only the diff
                return data.pos_diff
        else:
            # return only the spec
            return data.spec

    @staticmethod
    def add_arguments(parser, filter=None, log=None, mul=MUL, add=ADD,
                      ratio=RATIO,
                      diff_frames=DIFF_FRAMES):
        """
        Add spectrogram related arguments to an existing parser object.

        :param parser:      existing argparse parser object
        :param log:         include logarithm options (adds a switch to negate)
        :param mul:         multiply the magnitude spectrogram with given value
        :param add:         add the given value to the magnitude spectrogram
        :param ratio:       calculate the difference to the frame which window
                            overlaps to this ratio
        :param diff_frames: calculate the difference to the N-th previous frame
        :return:            spectrogram argument parser group object

        Parameters are included in the group only if they are not 'None'.

        """
        # TODO: add norm_window & fft_size
        # add spec related options to the existing parser
        g = parser.add_argument_group('spectrogram arguments')
        g.add_argument('--ratio', action='store', type=float, default=ratio,
                       help='window magnitude ratio to calc number of diff '
                       'frames [default=%(default).1f]')
        g.add_argument('--diff_frames', action='store', type=int,
                       default=diff_frames, help='number of diff frames '
                       '(if set, this overrides the value calculated from '
                       'the ratio)')
        # add log related options to the existing parser if needed
        l = None
        if log is not None:
            l = parser.add_argument_group('logarithmic magnitude arguments')
            if log:
                l.add_argument('--no_log', dest='log',
                               action='store_false', default=log,
                               help='no logarithmic magnitude '
                               '[default=logarithmic]')
            else:
                l.add_argument('--log', action='store_true',
                               default=-log, help='logarithmic '
                               'magnitude [default=linear]')
            if mul is not None:
                l.add_argument('--mul', action='store', type=float,
                               default=mul, help='multiplier (before taking '
                               ' the log) [default=%(default)i]')
            if add is not None:
                l.add_argument('--add', action='store', type=float,
                               default=add, help='value added (before taking '
                               'the log) [default=%(default)i]')
        # return the groups
        return g, l
