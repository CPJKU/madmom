#!/usr/bin/env python
# encoding: utf-8
"""
This file contains basic audio input/output functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""


# function for automatically determining how to open audio files
def load_audio_file(filename, sample_rate=None):
    """
    Load the audio data from the given file and return it as a numpy array.

    :param filename:    name of the file or file handle
    :param sample_rate: sample rate of the signal [Hz]
    :return:            tuple (signal, sample_rate)

    """
    # determine the name of the file
    if isinstance(filename, file):
        # open file handle
        filename = filename.name
    # how to handle the file?
    if filename.endswith(".wav"):
        # wav file
        from scipy.io import wavfile
        sample_rate, signal = wavfile.read(filename)
    # generic signal converter
    else:
        # FIXME: use sox instead to convert from different input signals
        # use the given sample rate to resample the signal on the fly if needed
        raise NotImplementedError('please integrate sox signal handling.')
    return signal, sample_rate

# TODO: add sox audio file handling
