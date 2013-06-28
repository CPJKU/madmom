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
    :param fft_size: use given size for FFT [default=size of window]
    :returns: the (complex) STFT of the signal

    Note: in offline mode, the window function is centered around the current
    position; whereas in online mode, the window is always positioned left to
    the current position.

    """
    # init variables
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
    # init spec matrix
    stft = np.empty([frames, fft_bins], np.complex)
    # perform STFT
    for frame in range(frames):
        # seek to the correct position in the audio signal
        if online:
            # step back a complete window size and move forward 1 hop size
            # so that the current position is at the end of the window
            #seek = int((frame + 1) * hop_size - window_size)
            # step back a complete window size
            # the current position is the right edge of the window
            seek = int(frame * hop_size - window_size)
        else:
            # step back half of the window size
            # the current position is the center of the window
            seek = int(frame * hop_size - window_size / 2.)
        # read in the right portion of the audio
        if seek >= samples:
            # end of file reached
            break
        elif seek + window_size >= samples:
            # end behind the actual audio end, append zeros accordingly
            zeros = np.zeros(seek + window_size - samples)
            fft_signal = np.append(signal[seek:], zeros)
        elif seek < 0:
            # start before the actual audio start, pad zeros accordingly
            zeros = np.zeros(-seek)
            fft_signal = np.append(zeros, signal[0:seek + window_size])
        else:
            # normal read operation
            fft_signal = signal[seek:seek + window_size]
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


def stft_strided(signal, window, hop_size, phase=True):
    """
    This function is only for completeness. It is faster in rare circumstances.

    Please note that the seeking to the right position is not always working
    properly, i.e. only for integer hop_sizes.

    """
    # init variables
    samples = np.shape(signal)[0]
    ffts = window.size >> 1
    # FIXME: does not perform the seeking the proper way
    as_strided = np.lib.stride_tricks.as_strided
    if phase:
        return fft.fft(fft.fftshift(as_strided(signal, (samples, window.size), (signal.strides[0], signal.strides[0]))[::hop_size] * window))[:, :ffts]
    else:
        return fft.fft(as_strided(signal, (samples, window.size), (signal.strides[0], signal.strides[0]))[::hop_size] * window)[:, :ffts]


def spec(stft):
    """
    Calculates the magnitude spectrogram of the Short-Time-Fourier-Transform.

    :param stft: the STFT of the signal
    :returns: the magnitude spectrogram of the STFT

    """
    return np.abs(stft)


def diff(spec, frames, pos=True):
    """
    Calculates the difference on the magnitude spectrogram.

    :param spec: the magnitude spectrogram
    :param frames: number of frames to calculate the difference to
    :param pos: only keep positive values [default=True]
    :returns: the difference spectrogram

    """
    diff = np.zeros_like(spec)
    # calculate the diff
    diff[frames:] = spec[frames:] - spec[:-frames]
    if pos:
        diff = diff * (diff > 0)
    return diff


def phase(stft):
    """
    Calculates the phase of the Short-Time-Fourier-Transform.

    :param stft: the STFT of the signal
    :returns: the phase of the STFT

    """
    return np.angle(stft)


def lgd(phase):
    """
    Calculates the local group delay of the phase spectrogram.

    :param phase: the phase spectrogram
    :returns: the local group delay

    """
    # init array
    lgd = np.zeros_like(phase)
    # unwrap phase over frequency axis
    unwrapped_phase = np.unwrap(phase, axis=1)
    # local group delay is the derivative over frequency
    lgd[:, :-1] = unwrapped_phase[:, :-1] - unwrapped_phase[:, 1:]
    # return
    return lgd


class Spectrogram(object):
    """
    Spectrogram Class.

    """
    def __init__(self, wav, window=np.hanning(2048), fps=100, online=False, phase=False, norm_window=False, fft_size=None):
        """
        Creates a new Spectrogram object instance and performs a STFT on the given audio.

        :param wav: a Wav object
        :param window: window function [default=Hann window with 2048 samples]
        :param fps: is the desired frame rate [frames per second, default=100]
        :param online: work in online mode [default=False]
        :param phase: include phase information [default=False]
        :param norm_window: set area of window function to 1 [default=False]
        :param fft_size: use given size for FFT [default=size of window]

        Note: including the phase slows down calculation.

        """
        # init variables
        self.wav = wav
        self.window = window
        self.fps = fps
        self.online = online
        if fft_size is None:
            fft_size = window.size
        self.fft_size = fft_size

        # normalize the window
        if norm_window:
            self.window /= np.sum(self.window)

        # perform STFT
        self.stft = stft(self.wav.audio, self.window, self.hop_size, online, phase, fft_size)
        # magnitude spectrogram
        self.spec = np.abs(self.stft)
        # phase
        if phase:
            # init array
            self.phase = phase(self.stft)
            # local group delay
            self.lgd = lgd(self.phase)
            # estimate local group delay
        else:
            self.phase = None

    @property
    def frames(self):
        """Number of frames."""
        # use ceil to not truncate the signal
        return int(np.ceil(self.wav.samples / self.hop_size))

    @property
    def hop_size(self):
        """Hop-size between two adjacent frames."""
        # use floats to make seeking work properly
        return float(self.wav.samplerate) / float(self.fps)

    @property
    def overlap(self):
        """Overlap factor of two adjacent frames."""
        return 1.0 - self.hop_size / self.window.size

    @property
    def ffts(self):
        """Number of FFT bins."""
        return self.fft_size >> 1

    @property
    def bins(self):
        """Number of bins of the spectrogram."""
        return np.shape(self.spec)[1]

    @property
    def mapping(self):
        """Conversion factor for mapping frequencies in Hz to spectrogram bins."""
        return self.wav.samplerate / 2.0 / self.ffts

    @property
    def fftfreqs(self):
        """List of frequencies corresponding to the spectrogram bins."""
        return np.fft.fftfreq(self.window.size)[:self.ffts] * self.wav.samplerate

    def aw(self, floor=0.5, relaxation=10):
        """
        Perform adaptive whitening on the magnitude spectrogram.

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

    def filter(self, filterbank):
        """
        Filter the magnitude spectrogram.

        :param filterbank: filterbank for dimensionality reduction

        """
        # TODO: should the filter stuff be included here?
        from filterbank import CQFilter
        if filterbank is None:
            # construct a standard filterbank
            filterbank = CQFilter(ffts=self.ffts, fs=self.wav.samplerate).filterbank
        self.spec = np.dot(self.spec, filterbank)
