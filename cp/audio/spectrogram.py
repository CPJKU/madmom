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

    :param signal: the discrete signal
    :param window: window function
    :param hop_size: the hop size in samples between adjacent frames
    :param online: only use past information of signal [default=False]
    :param phase: circular shift for correct phase [default=False]
    :param fft_size: use given size for FFT [default=size of window]
    :returns: the complex STFT of the signal

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

    :param signal: the discrete signal
    :param window: window function
    :param hop_size: the hop size in samples between adjacent frames
    :param phase: circular shift for correct phase [default=False]
    :returns: the complex STFT of the signal

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


class Spectrogram(object):
    """
    Spectrogram Class.

    """
    def __init__(self, audio, window=np.hanning(2048), hop_size=441., online=False, phase=False, lgd=False, norm_window=False, fft_size=None):
        """
        Creates a new Spectrogram object instance and performs a STFT on the given audio.

        :param signal: a FramedAudio object (or file name)
        :param window: window function [default=Hann window with 2048 samples]
        :param hop_size: progress N samples between adjacent frames [default=441.0]
        :param online: work in online mode [default=False]
        :param phase: include phase information [default=False]
        :param lgd: include local group delay information [default=False]
        :param norm_window: set area of window function to 1 [default=False]
        :param fft_size: use this size for FFT [default=size of window]

        Note: including phase and/or local group delay information slows down
        calculation considerably (phase: x2; lgd: x3)!

        """
        from audio import FramedAudio
        from wav import Wav

        # window stuff
        if isinstance(window, int):
            # if a window size is given, create a Hann window with that size
            window = np.hanning(window)
        elif isinstance(window, np.ndarray):
            # otherwise use the window directly
            self.window = window
        else:
            # other types are not supported
            raise TypeError("Invalid window type.")
        if norm_window:
            # normalize the window if needed
            self.window /= np.sum(self.window)

        # signal stuff
        if issubclass(audio.__class__, FramedAudio):
            # already the right format
            self.audio = audio
        else:
            # assume a file name, try to instantiate a Wav object with the given parameters
            # TODO: make an intelligent class which handles a lot of different file types
            self.audio = Wav(audio, frame_size=window.size, hop_size=hop_size, online=online)

        # FFT size to use
        if fft_size is None:
            fft_size = self.window.size
        self.fft_size = fft_size

        # init STFT matrix
        self.stft = np.empty([self.audio.num_frames, self.fft_bins], np.complex)
        # if the audio signal is not scaled, scale the window function accordingly
        # copy the window, and leave self.window untouched
        try:
            window = np.copy(self.window) / np.iinfo(self.audio.signal.dtype).max
        except ValueError:
            window = np.copy(self.window)

        # calculate STFT
        # TODO: use yield instead of the index counting stuff
        index = 0
        for frame in self.audio:
            # multiply the signal with the window function
            signal = np.multiply(frame, window)
            # only shift and perform complex DFT if needed
            if phase or lgd:
                # circular shift the signal (needed for correct phase)
                #signal = fft.fftshift(signal)  # slower!
                centre = self.window.size / 2
                signal = np.append(signal[centre:], signal[:centre])
            # perform DFT
            self.stft[index] = fft.fft(signal, fft_size)[:self.fft_bins]
            # increase index for next frame
            index += 1

        # magnitude spectrogram
        self.spec = np.abs(self.stft)

        # phase
        if phase or lgd:
            # init array
            self.phase = np.angle(self.stft)

        # local group delay
        if lgd:
            # init array
            self.lgd = np.zeros_like(self.phase)
            # unwrap phase over frequency axis
            unwrapped_phase = np.unwrap(self.phase, axis=1)
            # local group delay is the derivative over frequency
            self.lgd[:, :-1] = unwrapped_phase[:, :-1] - unwrapped_phase[:, 1:]
        # TODO: set self.phase and self.lgd to None otherwise?

    @property
    def num_frames(self):
        """Number of frames."""
        return self.audio.num_frames

    @property
    def hop_size(self):
        """Hop-size between two adjacent frames."""
        return self.audio.hop_size

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

        :param floor: floor coefficient [default=0.5]
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

    def log(self, mul=5, add=1):
        """
        Takes the logarithm of the magnitude spectrogram.

        :param mul: multiply the magnitude spectrogram with given value [default=5]
        :param add: add the given value to the magnitude spectrogram [default=1]

        """
        assert add > 0, 'a positive value must be added before taking the logarithm'
        self.spec = np.log10(mul * self.spec + add)

    def filter(self, filterbank=None):
        """
        Filter the magnitude spectrogram with the given filterbank.

        :param filterbank: filterbank for dimensionality reduction

        """
        # TODO: should the default filter stuff be included here? It is handy
        # to just call .filter() without having to create a filterbank first.
        if filterbank is None:
            from filterbank import CQFilter
            # construct a standard filterbank
            filterbank = CQFilter(fft_bins=self.fft_bins, fs=self.audio.samplerate).filterbank
        self.spec = np.dot(self.spec, filterbank)


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
        :param fmin: the minimum frequency [Hz, default=27]
        :param fmax: the maximum frequency [Hz, default=17000]
        :param norm: normalize the area of the filter to 1 [default=True]
        :param a4: tuning frequency of A4 [Hz, default=440]

        """
        # fetch the arguments special to the filterbank creation (or set defaults)
        fb = kwargs.pop('filterbank', None)
        bands_per_octave = kwargs.pop('bands', 12)
        fmin = kwargs.pop('fmin', 27)
        fmax = kwargs.pop('fmax', 17000)
        norm = kwargs.pop('norm', True)
        # create Spectrogram object
        super(FilteredSpectrogram, self).__init__(*args, **kwargs)
        # create a filterbank if needed
        if fb is None:
            from filterbank import CQFilter
            # construct a standard filterbank
            fb = CQFilter(fft_bins=self.fft_bins, fs=self.audio.samplerate, bands_per_octave=bands_per_octave, fmin=fmin, fmax=fmax, norm=norm).filterbank
        # TODO: use super.filter(fb) ?
        self.spec = np.dot(self.spec, fb)

# alias
FS = FilteredSpectrogram


class LogarithmicFilteredSpectrogram(FilteredSpectrogram):
    """
    LogarithmicFilteredSpectrogram is a subclass of FilteredSpectrogram which
    filters the magnitude spectrogram based on the given filterbank and converts
    it to a logarithmic scale.

    """
    def __init__(self, *args, **kwargs):
        """
        Creates a new LogarithmicFilteredSpectrogram object instance.

        The magnitudes of the filtered spectrogram are then converted to a
        logarithmic scale.

        :param mul: multiply the magnitude spectrogram with given value [default=5]
        :param add: add the given value to the magnitude spectrogram [default=1]

        """
        # fetch the arguments special to the logarithmic magnitude (or set defaults)
        mul = kwargs.pop('mul', 5)
        add = kwargs.pop('add', 1)
        # create Spectrogram object
        super(LogarithmicFilteredSpectrogram, self).__init__(*args, **kwargs)
        # take the logarithm
        self.log(mul, add)

# alias
LFS = LogarithmicFilteredSpectrogram
