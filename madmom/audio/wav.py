#!/usr/bin/env python
# encoding: utf-8
"""
This file contains wav file handling functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

from scipy.io import wavfile
from .audio import Signal


class Wav(Signal):
    """
    Wav Class is a subclass of ASignaland simple wrapper around scipy.io.wavfile.

    """

    def __init__(self, filename, *args, **kwargs):
        """
        Creates a new Wav object instance.

        :param filename: name of the .wav file or file handle

        """
        # init variables
        self.filename = filename        # the name of the file
        # read in the audio from the file
        sample_rate, data = wavfile.read(self.filename)
        # instantiate a FramedAudio object
        super(Wav, self).__init__(data, sample_rate, *args, **kwargs)

    def __str__(self):
        return "%s file: %s length: %i samples (%.2f seconds) sample rate: %i" % (self.__class__, self.filename, self.num_samples, self.length, self.sample_rate)
