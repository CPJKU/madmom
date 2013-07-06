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


class Wav(object):
    """
    Wav Class is a wrapper around scipy.io.wavfile.

    """

    def __init__(self, filename):
        """
        Creates a new Wav object instance.

        :param filename: name of the .wav file or file handle

        """
        # init variables
        self.filename = filename        # the name of the file
        self.audio = None               # the real audio (unscaled)
        self.samplerate = None          # with that samplerate
        # if a filename is given, read in audio
        if filename:
            self.load(filename)

    # load audio
    def load(self, filename=None):
        """
        Load audio data from file.

        :param filename: the file to load

        """
        from scipy.io import wavfile
        # overwrite file
        if filename is not None:
            # TODO: overwrite filename or just read in the new file?
            self.filename = filename
        # read in the audio
        # TODO: setting the samples attribute here has some speed improvements
        # but is less flexible than defining it as a property as below.
        self.samplerate, self.audio = wavfile.read(self.filename)
        # self.samples = np.shape(self.audio)[0]

    @property
    def samples(self):
        """Number of samples."""
        return np.shape(self.audio)[0]

    @property
    def channels(self):
        """Number of channels."""
        try:
            # multi channel files
            return np.shape(self.audio)[1]
        except IndexError:
            # catch mono files
            return 1

    @property
    def length(self):
        """Length of audio in seconds."""
        return float(self.samples) / float(self.samplerate)

    # attenuate the signal
    def attenuate(self, attenuation):
        """
        Attenuate the audio signal.

        :param attenuation: attenuation level [dB]

        """
        if attenuation <= 0:
            raise ValueError("a positive attenuation level must be given")
        self.audio /= np.power(np.sqrt(10.), attenuation / 10.)

    # downmix to mono
    def downmix(self):
        """Down-mix the audio signal to mono."""
        if self.channels > 1:
            self.audio = np.mean(self.audio, -1)

    # normalize the audio
    def normalize(self):
        """Normalise the audio signal."""
        self.audio /= np.max(self.audio)

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
            if offset * self.samplerate > self.samples:
                raise ValueError("offset must be < length of audio")
            self.audio = self.audio[(offset * self.samplerate):]
        # truncate the end
        if length != None:
            # check length
            if length <= 0:
                raise ValueError("a positive value must given")
            if length * self.samplerate > self.samples:
                raise ValueError("length must be < length of audio")
            self.audio = self.audio[:length * self.samplerate + 1]

    # downsample
    def downsample(self, factor=2):
        """
        Down-samples the audio signal by the given factor.

        :param factor: down-sampling factor [default=2]

        """
        from scipy.signal import decimate
        self.audio = np.hstack(decimate(self.audio, factor))
        self.samplerate /= factor

    # trim zeros
    def trim(self):
        """
        Trim leading and trailing zeros of the audio signal permanently.

        """
        self.audio = np.trim_zeros(self.audio, 'fb')


class SplittedWav(Wav):
    """
    SplittedWav Class is an iterable extension of the Wav Class.

    """

    def __init__(self, wav, frame_size=2048, hop_size=441.0, online=False):
        """
        Creates a new SplittedWav object instance or makes an existing Wav
        object iterable.

        :param wav: the existing Wav object (or any filename)

        :param frame_size: size of one frame [default=2048]
        :param hop_size: progress N samples between adjacent frames [default=441.0]
        :param online: use only past information [default=False]

        Note: the Wav class is implemented as an iterator. It splits the audio
        automatically into frames (of frame_size length) and progresses hop_size
        samples (can be float, with normal rounding applied) between frames.

        In offline mode, the frame is centered around the current position;
        whereas in online mode, the frame is always positioned left to the
        current position.

        """
        # arguments for splitting the audio into frames
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.online = online
        # filename or object given?
        if isinstance(wav, Wav):
            # a Wav object was given, make it iterable
            # usually the copying does not use additional memory
            super(SplittedWav, self).__init__(wav.filename)
        else:
            # create Wav object
            super(SplittedWav, self).__init__(wav)

    # make the Object iterable
    def __getitem__(self, index):
        """
        This makes the SplittedWav class an iterable object.

        The audio is split into frames (of length frame_size) automatically.
        Two frames are located hop_size frames apart. hop_size can be float,
        normal rounding applies.

        Note: index -1 refers NOT to the last frame, but to the frame directly
        left of frame 0. Although this is contrary to common behavior, being
        able to access these frames is important, because if the frames overlap
        frame -1 contains parts of the audio signal of frame 0.

        """
        # a slice is given
        if isinstance(index, slice):
            # return the frames given by the slice
            return [self[i] for i in xrange(*index.indices(self.frames))]
        # a single index is given
        elif isinstance(index, int):
#            # TODO: use this code if normal indexing behavior is needed
#            if index < 0:
#                index += self.frames
            # seek to the correct position in the audio signal
            if self.online:
                # the current position is the right edge of the frame
                # step back a complete frame size
                seek = int(index * self.hop_size - self.frame_size)
            else:
                # step back half of the window size
                # the current position is the center of the frame
                seek = int(index * self.hop_size - self.frame_size / 2.)
            # read in the right portion of the audio
            if seek < - self.frame_size:
                # more padding than a whole frame size needed
                raise IndexError("seek before start of audio")
            elif seek < 0:
                # start before the actual audio start, pad zeros accordingly
                zeros = np.zeros(-seek, dtype=self.audio.dtype)
                return np.append(zeros, self.audio[0:seek + self.frame_size])
            elif seek >= self.samples:
                # seek after end of audio
                raise IndexError("seek after end of audio")
            elif seek + self.frame_size > self.samples:
                # end behind the actual audio end, append zeros accordingly
                zeros = np.zeros(seek + self.frame_size - self.samples, dtype=self.audio.dtype)
                return np.append(self.audio[seek:], zeros)
            else:
                # normal read operation
                return self.audio[seek:seek + self.frame_size]
        # other index types are invalid
        else:
            raise TypeError("Invalid argument type.")

    # FIXME: what is the length? samples, frames?
    def __len__(self):
        return self.frames

    @property
    def frames(self):
        """Number of frames."""
        # add half a frame size to the number of samples in offline mode, since
        # the frames can cover up to this amount of data
        if self.online:
            samples = self.samples
        else:
            samples = self.samples + self.frame_size / 2.
        return int(np.ceil((samples) / self.hop_size))
