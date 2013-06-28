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

    """Wav Class is a simple wrapper around scipy.io.wavfile"""

    def __init__(self, filename=None):
        """
        Creates a new Wav object instance.

        :param filename: name of the .wav file

        """
        # init variables
        self.filename = filename  # the name of the file
        self.audio = None         # the real audio (unscaled)
        self.samplerate = None    # with that samplerate
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
        self.samplerate, self.audio = wavfile.read(self.filename)

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
