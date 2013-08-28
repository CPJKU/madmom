#!/usr/bin/env python
# encoding: utf-8
"""
Copyright (c) 2012-2013 Sebastian Böck <sebastian.boeck@jku.at>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import scipy.fftpack as fft


def stft(signal, window, hop_size, online=False, phase=False, fft_size=None):
    """
    Calculates the Short-Time-Fourier-Transform of the given signal.

    :param signal:   the discrete signal
    :param window:   window function
    :param hop_size: the hop size in samples between adjacent frames
    :param online:   only use past information of signal [default=False]
    :param phase:    circular shift for correct phase [default=False]
    :param fft_size: use given size for FFT [default=size of window]
    :returns:        the complex STFT of the signal

    Note: in offline mode, the window function is centered around the current
    position; whereas in online mode, the window is always positioned left to
    the current position.

    """
    from audio import signal_frame

    # if the signal is not scaled, scale the window function accordingly
    try:
        window = window[:] / np.iinfo(signal.dtype).max
    except ValueError:
        window = window[:]
    # size of window
    window_size = window.size
    # number of samples
    samples = np.shape(signal)[0]
    # number of frames
    frames = int(np.ceil(samples / hop_size))
    # size of FFT
    if fft_size is None:
        fft_size = window_size
    # number of resulting FFT bins
    fft_bins = fft_size >> 1
    # init stft matrix
    stft = np.empty([frames, fft_bins], np.complex)
    # perform STFT
    for frame in range(frames):
        # get the right portion of the signal
        fft_signal = signal_frame(signal, frame, window_size, hop_size, online)
        # multiply the signal with the window function
        fft_signal = np.multiply(fft_signal, window)
        # only shift and perform complex DFT if needed
        if phase:
            # circular shift the signal (needed for correct phase)
            #fft_signal = fft.fftshift(fft_signal)  # slower!
            fft_signal = np.append(fft_signal[window_size / 2:], fft_signal[:window_size / 2])
        # perform DFT
        stft[frame] = fft.fft(fft_signal, fft_size)[:fft_bins]
        # next frame
    # return
    return stft


def strided_stft(signal, window, hop_size, phase=True):
    """
    Calculates the Short-Time-Fourier-Transform of the given signal.

    :param signal:   the discrete signal
    :param window:   window function
    :param hop_size: the hop size in samples between adjacent frames
    :param phase:    circular shift for correct phase [default=False]
    :returns:        the complex STFT of the signal

    Note: This function is here only for completeness.
          It is faster only in rare circumstances.
          Also, seeking to the right position is only working properly, if
          integer hop_sizes are used.

    """
    from audio import strided_frames

    # init variables
    ffts = window.size >> 1
    # get a strided version of the signal
    fft_signal = strided_frames(signal, window.size, hop_size)
    # circular shift the signal
    if phase:
        fft_signal = fft.fftshift(fft_signal)
    # apply window function
    fft_signal *= window
    # perform the FFT
    return fft.fft(fft_signal)[:, :ffts]


# Spectrogram defaults
FILTERBANK = None
LOG = False         # default: linear spectrogram
MUL = 1
ADD = 1
STFT = False
PHASE = False
LGD = False
NORM_WINDOW = False
FFT_SIZE = None
RATIO = 0.5
DIFF_FRAMES = None


class Spectrogram(object):
    """
    Spectrogram Class.

    """
    def __init__(self, audio, window=np.hanning, filterbank=FILTERBANK,
                 log=LOG, mul=MUL, add=ADD, stft=STFT, phase=PHASE, lgd=LGD,
                 norm_window=NORM_WINDOW, fft_size=FFT_SIZE,
                 ratio=RATIO, diff_frames=DIFF_FRAMES):
        """
        Creates a new Spectrogram object instance of the given audio.

        :param signal:   a FramedAudio object; or file name or tuple (signal, samplerate)
        :param window:   window function [default=Hann window]

        Magnitude spectrogram manipulation parameters:

        :param filterbank: filterbank used for dimensionality reduction of the
                           magnitude spectrogram [default=None]

        :param log: take the logarithm of the magnitudes [default=False]
        :param mul: multiplier before taking the logarithm of the magnitudes [default=1]
        :param add: add this value before taking the logarithm of the magnitudes [default=0]

        Additional computations:

        :param stft:  save the raw complex STFT [default=False]
        :param phase: include phase information [default=False]
        :param lgd:   include local group delay information [default=False]

        FFT parameters:

        :param norm_window: set area of window function to 1 [default=False]
        :param fft_size:    use this size for FFT [default=size of window]

        Diff parameters:

        :param ratio:       calculate the difference to the frame which window overlaps to this ratio [default=0.5]
        :param diff_frames: calculate the difference to the N-th previous frame [default=None]
                            If set, this overrides the value calculated from the ratio.

        Note: including phase and/or local group delay information slows down
              calculation considerably (phase: x2; lgd: x3)!

        """
        from audio import FramedAudio
        # audio signal stuff
        if issubclass(audio.__class__, FramedAudio):
            # already the right format
            self.audio = audio
        elif issubclass(audio.__class__, tuple):
            # assume a tuple (signal, sample rate)
            self.audio = FramedAudio(audio[0], audio[1])
        else:
            # assume a file name, try to instantiate a Wav object
            # TODO: make an intelligent class which can handle a lot of different file types automatically
            from wav import Wav
            self.audio = Wav(audio)

        # window stuff
        if hasattr(window, '__call__'):
            # if only function is given, use the size to the audio frame size
            self.window = window(self.audio.frame_size)
        elif isinstance(window, np.ndarray):
            # otherwise use the given window directly
            self.window = window
        else:
            # other types are not supported
            raise TypeError("Invalid window type.")
        if norm_window:
            # normalize the window if needed
            self.window /= np.sum(self.window)

        # magnitude spectrogram (+ others)
        self.__spec = None
        self.__stft = None
        self.__phase = None
        self.__lgd = None
        # TODO: does this attribute belong to this class?
        self.__diff = None

        # additional calculations
        # Note: the naming might be a bit confusing but is short
        self._stft = stft
        self._phase = phase
        self._lgd = lgd

        # parameters used for the DFT
        if fft_size is None:
            self.fft_size = self.window.size
        else:
            self.fft_size = fft_size

        # parameters for magnitude spectrogram processing
        self.__filterbank = filterbank
        self.__log = log
        self.__mul = mul
        self.__add = add

        # diff parameters
        self.ratio = ratio
        self.__diff_frames = diff_frames

    @property
    def filterbank(self):
        return self.__filterbank

    @filterbank.setter
    def filterbank(self, filterbank):
        # set filterbank
        self.__filterbank = filterbank
        # invalidate the magnitude spectrogram
        self.__spec = None

    @property
    def log(self):
        return self.__log

    @log.setter
    def log(self, log):
        # set logarithm attribute
        self.__log = log
        # invalidate the magnitude spectrogram
        self.__spec = None

    @property
    def mul(self):
        return self.__mul

    @mul.setter
    def mul(self, mul):
        # set multiplication factor
        self.__mul = mul
        # invalidate the magnitude spectrogram
        self.__spec = None

    @property
    def add(self):
        return self.__add

    @add.setter
    def add(self, add):
        # set addition for logarithm
        self.__add = add
        # invalidate the magnitude spectrogram
        self.__spec = None

    def _fft_window(self):
        """
        Returns the window function to be used for FFT.
        The window is scaled automatically, depending on whether the audio
        is scaled to the range of -1..+1 or present in signed int16.

        """
        # if the audio signal is not scaled, scale the window function accordingly
        # copy the window, and leave self.window untouched
        try:
            return np.copy(self.window) / np.iinfo(self.audio.signal.dtype).max
        except ValueError:
            return np.copy(self.window)

    def compute_stft(self, index=None, filterbank=None, log=None, mul=None, add=None, stft=None, phase=None, lgd=None, fft_size=None):
        """
        This is a memory saving method to batch-compute different spectrograms.

        :param index:      slice for which the computation should be perfomed
        :param filterbank: filterbank used for dimensionality reduction of the
                           magnitude spectrogram
        :param log:        take the logarithm of the magnitudes
        :param mul:        multiplier before taking the logarithm of the magnitudes
        :param add:        add this value before taking the logarithm of the magnitudes
        :param stft:       save the raw complex STFT to the "stft" attribute
        :param phase:      save the phase of the STFT to the "phase" attribute
        :param lgd:        save the local group delay of the STFT to the "lgd" attribute

        """
        # determine the number of time frames needed
        # a slice is given
        if isinstance(index, slice):
            frames = (index.stop + 1 - index.start) / index.step
        # a single index is given
        elif isinstance(index, int):
            frames = 1
        # all frames
        else:
            index = slice(None)
            frames = self.audio.num_frames

        # filterbank
        if filterbank is not None:
            self.filterbank = filterbank
        # logarithm
        if log is not None:
            self.log = log
        if mul is not None:
            self.mul = mul
        if add is not None:
            self.add = add

        # determine the number of frequency bins needed for the magnitude spectrogram
        if isinstance(self.filterbank, np.ndarray):
            # the second dimension is the number of desired bins
            bins = np.shape(self.filterbank)[1]
        else:
            bins = self.fft_bins

        # init spectrogram matrix
        self.__spec = np.empty([frames, bins], np.float)
        # init STFT matrix
        if stft:
            self._stft = True
        if self._stft:
            self.__stft = np.empty([frames, self.fft_bins], np.complex)
        # init phase matrix
        if phase:
            self._phase = True
        if self._phase:
            self.__phase = np.empty([frames, self.fft_bins], np.float)
        # init local group delay matrix
        if lgd:
            self._lgd = True
        if self._lgd:
            self.__lgd = np.empty([frames, self.fft_bins], np.float)

        # get a scaled windowing function
        window = self._fft_window()

        # FFT size to use
        if fft_size is None:
            fft_size = self.fft_size
        if fft_size is None:
            fft_size = self.window.size

        # TODO: use yield instead of the index counting stuff and cache the results
        # maybe similar to the solution proposed in:
        # http://stackoverflow.com/questions/2322642/index-and-slice-a-generator-in-python
        frame_index = 0
        for frame in self.audio[index]:
            # multiply the signal with the window function
            signal = np.multiply(frame, window)
            # only shift and perform complex DFT if needed
            if self._phase or self._lgd:
                # circular shift the signal (needed for correct phase)
                #signal = fft.fftshift(signal)  # slower!
                centre = self.window.size / 2
                signal = np.append(signal[centre:], signal[:centre])
            # perform DFT
            dft = fft.fft(signal, fft_size)[:self.fft_bins]

            # save raw stft
            if self._stft:
                self.__stft[frame_index] = dft
            # phase / lgd
            if self._phase or self._lgd:
                angle = np.angle(dft)
            if self._phase:
                self.__phase[frame_index] = angle
            if self._lgd:
                # unwrap phase over frequency axis
                unwrapped_phase = np.unwrap(angle, axis=1)
                # local group delay is the derivative over frequency
                self.__lgd[frame_index, :-1] = unwrapped_phase[:-1] - unwrapped_phase[1:]

            # magnitude spectrogram
            spec = np.abs(dft)
            # filter with a given filterbank
            if self.filterbank is not None:
                spec = np.dot(spec, self.filterbank)
            # take the logarithm if needed
            if self.log:
                spec = np.log10(self.mul * spec + self.add)
            self.__spec[frame_index] = spec

            # increase index counter for next frame
            frame_index += 1

    @property
    def stft(self):
        """Short Time Fourier Transform of the signal."""
        # TODO: this is highly inefficient, if more properties are accessed
        # better call compute_stft() only once with appropriate parameters.
        if self.__stft is None:
            self.compute_stft(stft=True)
        return self.__stft

    @property
    def spec(self):
        """Magnitude spectrogram of the STFT."""
        # TODO: this is highly inefficient, if more properties are accessed
        # better call compute_stft() only once with appropriate parameters.
        if self.__spec is None:
            # check if STFT was computed already
            if self.__stft is not None:
                # use it
                self.__spec = np.abs(self.__stft)
                # filter if needed
                if self.filterbank is not None:
                    self.__spec = np.dot(self.__spec, self.filterbank)
                # take the logarithm
                if self.log:
                    self.__spec = np.log10(self.mul * self.__spec + self.add)
            else:
                # compute the spec
                self.compute_stft()
        return self.__spec

    @property
    def num_diff_frames(self):
        """Number of frames used for difference calculation of the magnitude spectrogram."""
        if self.__diff_frames is None:
            # calculate on basis of the ratio
            # get the first sample with a higher magnitude than given ratio
            sample = np.argmax(self.window > self.ratio * max(self.window))
            diff_samples = self.window.size / 2 - sample
            # convert to frames
            diff_frames = int(round(diff_samples / self.hop_size))
            # set the minimum to 1
            if diff_frames < 1:
                diff_frames = 1
            # return
            return diff_frames
        else:
            # return the set value
            return self.__diff_frames

    @num_diff_frames.setter
    def num_diff_frames(self, diff_frames):
        """
        Set the number of diff frames.

        :param diff_frames: number of frames used for difference calculation

        Note: if set to None, the number is calculated dynamically on the ratio
              to which extend the windows overlap.

        """
        self.__diff_frames = diff_frames

    @property
    def diff(self):
        """Differences of the magnitude spectrogram."""
        if self.__diff is None:
            # init array
            self.__diff = np.zeros_like(self.spec)
            # calculate the diff
            self.__diff[self.num_diff_frames:] = self.spec[self.num_diff_frames:] - self.spec[:-self.num_diff_frames]
            # TODO: make the filling of the first diff_frames frames work properly
        return self.__diff

    @property
    def pos_diff(self):
        """Positive differences of the magnitude spectrogram."""
        return self.diff * (self.diff > 0)

    @property
    def phase(self):
        """Phase of the STFT."""
        # TODO: this is highly inefficient, if more properties are accessed
        # better call compute_stft() only once with appropriate parameters.
        if self.__phase is None:
            # check if STFT was computed already
            if self.__stft is not None:
                # use it
                self.__phase = np.angle(self.__stft)
            else:
                # compute the phase
                self.compute_stft(phase=True)
        return self.__phase

    @property
    def lgd(self):
        """Local group delay of the STFT."""
        # TODO: this is highly inefficient, if more properties are accessed
        # better call compute_stft() only once with appropriate parameters.
        if self.__lgd is None:
            # if the STFT was computed already, but not the phase
            if self.__stft is not None and self.__phase is None:
                # save the phase as well
                # FIXME: this uses unneeded memory, if only STFT and LGD are of
                # interest, but not the phase (very rare case only)
                self.__phase = np.angle(self.__stft)
            # check if phase was computed already
            if self.__phase is not None:
                # FIXME: remove duplicate code
                # unwrap phase over frequency axis
                unwrapped_phase = np.unwrap(self.__phase, axis=1)
                # local group delay is the derivative over frequency
                self.__lgd[:, :-1] = unwrapped_phase[:, -1] - unwrapped_phase[:, 1:]
            else:
                # compute the local group delay
                self.compute_stft(lgd=True)
        return self.__lgd

    @property
    def num_frames(self):
        """Number of frames."""
        return np.shape(self.spec)[0]

    # TODO: remove this?
    @property
    def hop_size(self):
        """Hop-size between two adjacent frames."""
        return self.audio.hop_size

    # TODO: remove this?
    @property
    def overlap_factor(self):
        """Overlap factor of two adjacent frames."""
        return self.audio.overlap_factor

    @property
    def fft_bins(self):
        """Number of FFT bins."""
        return self.fft_size >> 1

    @property
    def bins(self):
        """Number of bins of the spectrogram."""
        return np.shape(self.spec)[1]

    @property
    def mapping(self):
        """Conversion factor for mapping frequencies in Hz to spectrogram bins."""
        return self.audio.samplerate / 2.0 / self.fft_bins

    @property
    def fft_freqs(self):
        """List of frequencies corresponding to the spectrogram bins."""
        return np.fft.fftfreq(self.window.size)[:self.fft_bins] * self.audio.samplerate

    def aw(self, floor=0.5, relaxation=10):
        """
        Perform adaptive whitening on the magnitude spectrogram.

        :param floor:      floor coefficient [default=0.5]
        :param relaxation: relaxation time [frames, default=10]

        "Adaptive Whitening For Improved Real-time Audio Onset Detection"
        Dan Stowell and Mark Plumbley
        Proceedings of the International Computer Music Conference (ICMC), 2007

        """
        mem_coeff = 10.0 ** (-6. * relaxation / self.fps)
        P = np.zeros_like(self.spec)
        # iterate over all frames
        for f in range(self.frames):
            if f > 0:
                P[f] = np.maximum(self.spec[f], floor, mem_coeff * P[f - 1])
            else:
                P[f] = np.maximum(self.spec[f], floor)
        # adjust spec
        self.spec /= P


class FilteredSpectrogram(Spectrogram):
    """
    FilteredSpectrogram is a subclass of Spectrogram which filters the
    magnitude spectrogram based on the given filterbank.

    """
    def __init__(self, *args, **kwargs):
        """
        Creates a new FilteredSpectrogram object instance.

        :param filterbank: filterbank for dimensionality reduction

        If no filterbank is given, one with the following parameters is created
        automatically.

        :param bands_per_octave: number of filter bands per octave [default=12]
        :param fmin:             the minimum frequency [Hz, default=27]
        :param fmax:             the maximum frequency [Hz, default=17000]
        :param norm:             normalize the area of the filter to 1 [default=True]
        :param a4:               tuning frequency of A4 [Hz, default=440]

        """
        import filterbank
        # fetch the arguments special to the filterbank creation (or set defaults)
        fb = kwargs.pop('filterbank', None)
        bands_per_octave = kwargs.pop('bands_per_octave', filterbank.BANDS_PER_OCTAVE)
        fmin = kwargs.pop('fmin', filterbank.FMIN)
        fmax = kwargs.pop('fmax', filterbank.FMAX)
        norm = kwargs.pop('norm', filterbank.NORM_FILTER)
        # create Spectrogram object
        super(FilteredSpectrogram, self).__init__(*args, **kwargs)
        # if no filterbank was given, create one
        if fb is None:
            fb = filterbank.LogarithmicFilter(fft_bins=self.fft_bins, fs=self.audio.samplerate, bands_per_octave=bands_per_octave, fmin=fmin, fmax=fmax, norm=norm)
        # save the filterbank, so it gets used when the magnitude spectrogram gets computed
        self.filterbank = fb

# aliases
FiltSpec = FilteredSpectrogram
FS = FilteredSpectrogram


class LogarithmicFilteredSpectrogram(FilteredSpectrogram):
    """
    LogarithmicFilteredSpectrogram is a subclass of FilteredSpectrogram which
    filters the magnitude spectrogram based on the given filterbank and converts
    it to a logarithmic (magnitude) scale.

    """
    def __init__(self, *args, **kwargs):
        """
        Creates a new LogarithmicFilteredSpectrogram object instance.

        The magnitudes of the filtered spectrogram are then converted to a
        logarithmic scale.

        :param mul: multiply the magnitude spectrogram with given value [default=1]
        :param add: add the given value to the magnitude spectrogram [default=1]

        """
        # fetch the arguments special to the logarithmic magnitude (or set defaults)
        mul = kwargs.pop('mul', MUL)
        add = kwargs.pop('add', ADD)
        # create Spectrogram object
        super(LogarithmicFilteredSpectrogram, self).__init__(*args, **kwargs)
        # set the parameters, so they get used when the magnitude spectrogram gets computed
        self.log = True
        self.mul = mul
        self.add = add

# aliases
LogFiltSpec = LogarithmicFilteredSpectrogram
LFS = LogarithmicFilteredSpectrogram
