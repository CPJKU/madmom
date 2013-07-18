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


#FIXME: move this to audio.py, but this files needs to be cleaned up first
class Audio(object):
    """
    Audio is a very simple class which just stores the audio and the samplerate
    and provides some basic methods for signal processing.

    """
    def __init__(self, audio, samplerate):
        """
        Creates a new Audio object instance.

        :param audio: the audio signal [numpy array]
        :param samplerate: samplerate of the audio

        """
        if not isinstance(audio, np.ndarray):
            # make sure the audio is a numpy array
            raise TypeError("Invalid type for audio.")
        self.audio = audio
        self.samplerate = samplerate

    @property
    def num_samples(self):
        """Number of samples."""
        return np.shape(self.audio)[0]

    @property
    def num_channels(self):
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
        return float(self.num_samples) / float(self.samplerate)

    # downmix to mono
    def downmix(self):
        """Down-mix the audio signal to mono."""
        if self.num_channels > 1:
            self.audio = np.mean(self.audio, -1)

    # normalize the audio
    def normalize(self):
        """Normalise the audio signal."""
        self.audio = self.audio.astype(float) / np.max(self.audio)

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
                raise ValueError("offset must be < length of audio")
            self.audio = self.audio[(offset * self.samplerate):]
        # truncate the end
        if length != None:
            # check length
            if length <= 0:
                raise ValueError("a positive value must given")
            if length * self.samplerate > self.num_samples:
                raise ValueError("length must be < length of audio")
            self.audio = self.audio[:length * self.samplerate + 1]

    # downsample
    def downsample(self, factor=2):
        """
        Downsamples the audio signal by the given factor.

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

    # TODO: make this nicer!
    def __str__(self):
        return "%s length: %i samples (%.2f seconds) samplerate: %i" % (self.__class__, self.num_samples, self.length, self.samplerate)


class FramedAudio(Audio):
    """
    FramedAudio splits an audio signal into frames and makes them iterable.

    """
    def __init__(self, audio, samplerate, frame_size=2048, hop_size=441.0, online=False):
        """
        Creates a new FramedAudio object instance.

        :param audio: the audio signal [numpy array]
        :param samplerate: samplerate of the audio

        :param frame_size: size of one frame [default=2048]
        :param hop_size: progress N samples between adjacent frames [default=441.0]
        :param online: use only past information [default=False]

        Note: the FramedAudio class is implemented as an iterator. It splits the
        audio automatically into frames (of frame_size length) and progresses
        hop_size samples (can be float, with normal rounding applied) between
        frames.

        In offline mode, the frame is centered around the current position;
        whereas in online mode, the frame is always positioned left to the
        current position.

        """
        # instantiate a Audio object
        super(FramedAudio, self).__init__(audio, samplerate)
        # arguments for splitting the audio into frames
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.online = online

    # make the Object iterable
    def __getitem__(self, index):
        """
        This makes the FramedAudio class an iterable object.

        The audio is split into frames (of length frame_size) automatically.
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
#            # TODO: use this code if normal indexing behavior is needed
#            if index < 0:
#                index += self.frames
            # seek to the correct position in the audio signal
            seek = int(index * self.hop_size)
            # depending on online/offline mode position the moving window
            if self.online:
                # the current position is the right edge of the frame
                # step back a complete frame size
                start = seek - self.frame_size
                stop = seek
            else:
                # the current position is the center of the frame
                # step back half of the window size
                start = seek - self.frame_size / 2
                stop = seek + self.frame_size / 2
            # raise errors if outside the audio range
            if stop < 0:
                # more padding than a whole frame size needed
                raise IndexError("seek before start of audio")
            if start > self.num_samples:
                # seek after end of audio
                raise IndexError("seek after end of audio")
            # read in the right portion of the audio
            if start < 0:
                # start before the actual audio start, pad zeros accordingly
                zeros = np.zeros(-start, dtype=self.audio.dtype)
                return np.append(zeros, self.audio[0:stop])
            elif stop > self.num_samples:
                # end behind the actual audio end, append zeros accordingly
                zeros = np.zeros(stop - self.num_samples, dtype=self.audio.dtype)
                return np.append(self.audio[start:], zeros)
            else:
                # normal read operation
                return self.audio[start:stop]
        # other index types are invalid
        else:
            raise TypeError("Invalid argument type.")

    # len() should return the number of frames, since it iterates over frames
    def __len__(self):
        return self.num_frames

    @property
    def num_frames(self):
        """Number of frames."""
        #return int(np.ceil((np.shape(self.audio)[0]) / self.hop_size))
        return int(np.ceil(((np.shape(self.audio)[0]) + self.frame_size) / self.hop_size))

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


class Wav(FramedAudio):
    """
    Wav Class is a wrapper around scipy.io.wavfile.

    """

    def __init__(self, filename, frame_size=2048, hop_size=441.0, online=False):
        """
        Creates a new Wav object instance.

        :param filename: name of the .wav file or file handle
        :param frame_size: size of one frame [default=2048]
        :param hop_size: progress N samples between adjacent frames [default=441.0]
        :param online: use only past information [default=False]

        """
        # init variables
        self.filename = filename        # the name of the file
        # read in the audio from the file
        from scipy.io import wavfile
        samplerate, audio = wavfile.read(self.filename)
        # instantiate a FramedAudio object
        super(Wav, self).__init__(audio, samplerate, frame_size, hop_size, online)

    # TODO: make this nicer!
    def __str__(self):
        return "%s file: %s length: %i samples (%.2f seconds) samplerate: %i frames: %i (%i samples %.1f hopsize)" % (self.__class__, self.filename, self.num_samples, self.length, self.samplerate, self.frames, self.frame_size, self.hop_size)
