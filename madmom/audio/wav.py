#!/usr/bin/env python
# encoding: utf-8
"""
This file contains wav file handling functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

from scipy.io import wavfile
from .signal import Signal


class Wav(Signal):
    """
    The Wav class is a subclass of Signal and a simple wrapper around
    scipy.io.wavfile.

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
