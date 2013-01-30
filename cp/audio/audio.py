"""
    Some useful functions for audio processing
"""

__docformat__ = "restructuredtext en"

import numpy as np
import scipy.fftpack
import scikits.audiolab
from utilities import segment_axis


def read(filename):
    """
    Reads an audio file into a numpy array. This is just a wrapper for
    scikits.audiolab. Supports all formats supported by
    libsndfile (http://www.mega-nerd.com/libsndfile/).

    :Parameters:
        - `filename`: filename of the audiofile to read

    :Returns:
        - A tuple (data, samplerate)
          For single-channel audio `data` is an one-dimensional array, for
          multi-channel audio `data` will be a two-dimensional array of shape
          (number_of_channels, audio_length)
    """
    sound_file = scikits.audiolab.Sndfile(filename, 'r')
    data = sound_file.read_frames(sound_file.nframes)
    return (data.T, sound_file.samplerate)


def split(signal, window_length, hop_size, end='cut', endvalue=0):
    """
    Splits the signal into chunks of the specified length using the specified
    hop_size.

    :Parameters:
        - `signal`: np.array containing the signal to split
        - `window_length`: desired length of the audio chunks
        - `hop_size`: desired distance in samples between the audio chunks
        - `end`: What to do with the last frame, if the signal is not evenly
                 divisible into pieces. Options are:

                 - 'cut': Discard the extra values
                 - 'wrap': Copy values from the beginning of the array
                 - 'pad': Pad with a constant value
        - `endvalue`: Value to use for end='pad'

    :Returns:
          A reshaped array of audio chunks of length `window_length`, spaced
          by hop_size samples
    """
    return segment_axis(signal, window_length, window_length - hop_size,
                        end=end, endvalue=endvalue)


def join(splits, hop_size):
    """
    Joins a set of audio chunks to a single audio signal. For overlapping
    segments the mean of all values is used.

    This function can be used to join splits obtained by the split() function.

    :Parameters:
        - `splits`: Numpy array or list of audio chunks of same length
        - `hop_size`: Distance in samples between the beginnings of each
                      audio chunk. This should be smaller than the chunk
                      length

    :Returns:
          Joined audio signal computed from the audio chunks
    """
    window_length = len(splits[0])
    signal_length = (len(splits) - 1) * hop_size + window_length
    signal = np.zeros(signal_length)
    normalisation = np.zeros(signal_length)

    sig_pos = 0
    for split in splits:
        signal[sig_pos:sig_pos + window_length] += split
        normalisation[sig_pos:sig_pos + window_length] += 1
        sig_pos += hop_size

    return signal / normalisation


def spectrogram(signal, window_length, hop_size, window_func=np.hanning,
                fft_size=None, normalise=True):
    """
    Convenience function that computes the spectrogram of a signal. It assumes
    a real-valued input.

    :Parameters:
        - `signal`: Audio signal to compute the spectrogram on
        - `window_length`: Desired length of audio chunks
        - `hop_size`: Desired distance in samples between spectrogram frames
        - `window_func`: Window function to apply on each audio chunk. This
                         function must return a real-valued numpy array of
                         length `window_length`
        - `fft_size`: Length of the array used for fft-transformation. If
                      this is larger than window_length, each audio chunk will
                      be zero-padded.
    :Returns:
          Spectrogram of the passed signal
    """
    # using the scipy.fftpack.fft function, is has proven faster than
    # np.fft AND np.rfft. scipy.fftpack.rfft is even faster, but returns a
    # strange format to work with.

    if fft_size is None:
        fft_size = window_length

    window = window_func(window_length)
    windowed_signal = split(signal, window_length, hop_size) * window
    spec = scipy.fftpack.fft(windowed_signal, n=fft_size)[:, 0:(fft_size / 2 + 1)]

    if normalise:
        spec *= 2.0 / window.sum()

    return spec
