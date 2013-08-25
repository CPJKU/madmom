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

from scipy.io import wavfile
from audio import FramedAudio


class Wav(FramedAudio):
    """
    Wav Class is a simple wrapper around scipy.io.wavfile and makes the .wav
    file iterable.

    """

    def __init__(self, filename, *args, **kwargs):
        """
        Creates a new Wav object instance.

        :param filename: name of the .wav file or file handle

        """
        # init variables
        self.filename = filename        # the name of the file
        # read in the audio from the file
        samplerate, signal = wavfile.read(self.filename)
        # instantiate a FramedAudio object
        super(Wav, self).__init__(signal, samplerate, *args, **kwargs)

    # TODO: make this nicer!
    def __str__(self):
        return "%s file: %s length: %i samples (%.2f seconds) samplerate: %i frames: %i (%i samples %.1f hopsize)" % (self.__class__, self.filename, self.num_samples, self.length, self.samplerate, self.frames, self.frame_size, self.hop_size)
