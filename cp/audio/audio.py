#!/usr/bin/env python
# encoding: utf-8
"""
Copyright (c) 2013 Sebastian Böck <sebastian.boeck@jku.at>
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


def root_mean_square(signal):
    """
    Computes the root mean square of the signal. This can be used as
    a measurement of power.

    :param signal: the audio signal
    :returns: root mean square of the signal

    """
    # make sure the signal is a numpy array
    if not isinstance(signal, np.ndarray):
        raise TypeError("Invalid type for audio signal.")
    # Note: type conversion needed because of integer overflows
    if signal.dtype != np.float:
        signal = signal.astype(np.float)
    # return
    return np.sqrt(np.dot(signal, signal) / signal.size)


def sound_pressure_level(signal, p_ref=1.0):
    """
    Computes the sound pressure level of a signal.

    :param signal: the audio signal
    :param signal: reference sound pressure level [default=1.0]
    :returns: sound pressure level of the signal

    From en.wikipedia.org/wiki/Sound_pressure:
    Sound pressure level (SPL) or sound level is a logarithmic measure of the
    effective sound pressure of a sound relative to a reference value.
    It is measured in decibels (dB) above a standard reference level.

    """
    # compute the RMS
    rms = root_mean_square(signal)
    # compute the SPL
    if rms == 0:
        # return the largest possible negative number
        return -np.finfo(float).max
    else:
        # normal SPL computation
        return 20.0 * np.log10(rms / p_ref)


def signal_frame(signal, index, frame_size, hop_size, online):
    """
    This function returns frame[index] of the signal.

    :param signal: the audio signal
    :param frame_size: size of one frame
    :param hop_size: progress N samples between adjacent frames
    :param online: use only past information
    :returns: the single frame of the audio signal

    """
    # make sure the signal is a numpy array
    if not isinstance(signal, np.ndarray):
        raise TypeError("Invalid type for audio signal.")
    # length of the signal
    samples = np.shape(signal)[0]
    # seek to the correct position in the audio signal
    seek = int(index * hop_size)
    if seek < 0:
        # seek before the start of signal
        raise IndexError("seek before start of signal")
    if seek > samples:
        # seek after end of signal
        raise IndexError("seek after end of signal")
    # depending on online/offline mode position the moving window
    if online:
        # the current position is the right edge of the frame
        # step back a complete frame size
        start = seek - frame_size
        stop = seek
    else:
        # the current position is the center of the frame
        # step back half of the window size
        start = seek - frame_size / 2
        stop = seek + frame_size / 2
    # return the right portion of the signal
    if start < 0:
        # start before the actual signal start, pad zeros accordingly
        zeros = np.zeros(-start, dtype=signal.dtype)
        return np.append(zeros, signal[0:stop])
    elif stop > samples:
        # end behind the actual signal end, append zeros accordingly
        zeros = np.zeros(stop - samples, dtype=signal.dtype)
        return np.append(signal[start:], zeros)
    else:
        # normal read operation
        return signal[start:stop]


def strided_frames(signal, frame_size, hop_size):
    """
    Returns a 2D representation of the signal with overlapping frames.

    :param signal: the discrete signal
    :param frame_size: size of each frame
    :param hop_size: the hop size in samples between adjacent frames
    :returns: the framed audio signal

    Note: This function is here only for completeness.
          It is faster only in rare circumstances.
          Also, seeking to the right position is only working properly, if
          integer hop_sizes are used.

    """
    # init variables
    samples = np.shape(signal)[0]
    # FIXME: does not perform the seeking the right way (only int working properly)
    as_strided = np.lib.stride_tricks.as_strided
    # return the strided array
    return as_strided(signal, (samples, frame_size), (signal.strides[0], signal.strides[0]))[::hop_size]


class Audio(object):
    """
    Audio is a very simple class which just stores the signal and the samplerate
    and provides some basic methods for signal processing.

    """
    def __init__(self, signal, samplerate):
        """
        Creates a new Audio object instance.

        :param signal: the audio signal [numpy array]
        :param samplerate: samplerate of the signal

        """
        if not isinstance(signal, np.ndarray):
            # make sure the signal is a numpy array
            raise TypeError("Invalid type for audio signal.")
        self.signal = signal
        self.samplerate = samplerate

    @property
    def num_samples(self):
        """Number of samples."""
        return np.shape(self.signal)[0]

    @property
    def num_channels(self):
        """Number of channels."""
        try:
            # multi channel files
            return np.shape(self.signal)[1]
        except IndexError:
            # catch mono files
            return 1

    @property
    def length(self):
        """Length of signal in seconds."""
        return float(self.num_samples) / float(self.samplerate)

    # downmix to mono
    def downmix(self):
        """Down-mix the audio signal to mono."""
        if self.num_channels > 1:
            self.signal = np.mean(self.signal, -1)

    # normalize the signal
    def normalize(self):
        """Normalise the audio signal."""
        self.signal = self.signal.astype(float) / np.max(self.signal)

    # truncate
    def truncate(self, offset=None, length=None):
        """
        Truncate the audio signal permanently.

        :param offset: given in seconds
        :param length: given in seconds

        """
        # truncate the beginning
        if offset != None:
            # check offset
            if offset <= 0:
                raise ValueError("offset must be positive")
            if offset * self.samplerate > self.num_samples:
                raise ValueError("offset must be < length of signal")
            self.signal = self.signal[(offset * self.samplerate):]
        # truncate the end
        if length != None:
            # check length
            if length <= 0:
                raise ValueError("a positive value must given")
            if length * self.samplerate > self.num_samples:
                raise ValueError("length must be < length of signal")
            self.signal = self.signal[:length * self.samplerate + 1]

    # downsample
    def downsample(self, factor=2):
        """
        Downsamples the audio signal by the given factor.

        :param factor: down-sampling factor [default=2]

        """
        from scipy.signal import decimate
        self.signal = np.hstack(decimate(self.signal, factor))
        self.samplerate /= factor

    # trim zeros
    def trim(self):
        """
        Trim leading and trailing zeros of the audio signal permanently.

        """
        self.signal = np.trim_zeros(self.signal, 'fb')

    # TODO: make this nicer!
    def __str__(self):
        return "%s length: %i samples (%.2f seconds) samplerate: %i" % (self.__class__, self.num_samples, self.length, self.samplerate)


class FramedAudio(Audio):
    """
    FramedAudio splits an audio signal into frames and makes them iterable.

    """
    def __init__(self, signal, samplerate, frame_size=2048, hop_size=441, online=False):
        """
        Creates a new FramedAudio object instance.

        :param signal: the audio signal [numpy array]
        :param samplerate: samplerate of the signal

        :param frame_size: size of one frame [default=2048]
        :param hop_size: progress N samples between adjacent frames [default=441]
        :param online: use only past information [default=False]

        Note: the FramedAudio class is implemented as an iterator. It splits the
        signal automatically into frames (of frame_size length) and progresses
        hop_size samples (can be float, with normal rounding applied) between
        frames.

        In offline mode, the frame is centered around the current position;
        whereas in online mode, the frame is always positioned left to the
        current position.

        """
        # instantiate a Audio object
        super(FramedAudio, self).__init__(signal, samplerate)
        # arguments for splitting the signal into frames
        self.frame_size = frame_size
        self.hop_size = float(hop_size)
        self.online = online

    # make the Object iterable
    def __getitem__(self, index):
        """
        This makes the FramedAudio class an iterable object.

        The signal is split into frames (of length frame_size) automatically.
        Two frames are located hop_size samples apart. hop_size can be float,
        normal rounding applies.

        Note: index -1 refers NOT to the last frame, but to the frame directly
        left of frame 0. Although this is contrary to common behavior, being
        able to access these frames is important, because if the frames overlap
        frame -1 contains parts of the audio signal of frame 0.

        """
        # a slice is given
        if isinstance(index, slice):
            # return the frames given by the slice
            return [self[i] for i in xrange(*index.indices(len(self)))]
        # a single index is given
        elif isinstance(index, int):
            # TODO: use this code if normal indexing behavior is needed
            if index < 0:
                index += self.num_frames
            return signal_frame(self.signal, index, self.frame_size, self.hop_size, self.online)
        # other index types are invalid
        else:
            raise TypeError("Invalid argument type.")

    # len() should return the number of frames, since it iterates over frames
    def __len__(self):
        return self.num_frames

    @property
    def num_frames(self):
        """Number of frames."""
        # TODO: add a complete frame_size, to cover the whole audio signal?
        # modifications to signal_frame() needed
        #return int(np.ceil(((np.shape(self.signal)[0]) + self.frame_size) / self.hop_size))
        return int(np.ceil((np.shape(self.signal)[0]) / self.hop_size))

    @property
    def fps(self):
        """Frames per second."""
        return float(self.wav.samplerate) / float(self.hop_size)

    @property
    def overlap_factor(self):
        """Overlap factor of two adjacent frames."""
        return 1.0 - self.hop_size / self.window.size

    # TODO: make this nicer!
    def __str__(self):
        return "%s length: %i samples (%.2f seconds) samplerate: %i frames: %i (%i num_samples %.1f hopsize)" % (self.__class__, self.num_samples, self.length, self.samplerate, self.frames, self.frame_size, self.hop_size)
