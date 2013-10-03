#!/usr/bin/env python
# encoding: utf-8
"""
This file contains basic signal processing functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import numpy as np


# signal functions
def attenuate(signal, attenuation):
    """"
    Attenuate the audio signal.

    :param signal:      signal
    :param attenuation: attenuation level [dB]
    :returns:           attenuated signal

    """
    # FIXME: attenuating the signal and keeping the original dtype makes the
    # following signal processing steps well-behaved, since these rely on
    # the dtype of the array to determine the correct value range.
    # But this introduces rounding (truncating) errors in case of signals
    # with int dtypes. But these errors should be negligible.
    return np.asarray(signal / np.power(np.sqrt(10.), attenuation / 10.), dtype=signal.dtype)


def normalize(signal):
    """
    Normalize the audio signal to the range -1..+1

    :param signal: signal
    :returns:      normalized signal

    """
    return signal.astype(float) / np.max(signal)


def downmix(signal):
    """
    Down-mix the audio signal to mono.

    :param signal: signal
    :returns:      mono signal

    """
    if signal.ndim > 1:
        # FIXME: taking the mean and keeping the original dtype makes the
        # following signal processing steps well-behaved, since these rely on
        # the dtype of the array to determine the correct value range.
        # But this introduces rounding (truncating) errors in case of signals
        # with int dtypes. But these errors should be negligible.
        return np.mean(signal, axis=-1, dtype=signal.dtype)
    else:
        return signal


def downsample(signal, factor=2):
    """
    Down-samples the audio signal by the given factor

    :param signal: signal
    :param factor: down-sampling factor [default=2]
    :returns:      down-sampled signal

    """
    # signal must be mono
    if signal.ndim > 1:
        # FIXME: please implement stereo (or multi-channel) handling
        raise NotImplementedError("please implement stereo functionality")
    # when downsampling by an integer factor, a simple view is more efficient
    if type(factor) == int:
        return signal[::factor]
    # otherwise do more or less propoer down-sampling
    # TODO: maybe use sox to implement this
    from scipy.signal import decimate
    # naive down-sampling
    return np.hstack(decimate(signal, factor))


def trim(signal):
    """
    Trim leading and trailing zeros of the audio signal.

    :param signal: signal
    :returns:      trimmed signal

    """
    # signal must be mono
    if signal.ndim > 1:
        # FIXME: please implement stereo (or multi-channel) handling
        # maybe it works, haven't checked
        raise NotImplementedError("please implement stereo functionality")
    return np.trim_zeros(signal, 'fb')


def root_mean_square(signal):
    """
    Computes the root mean square of the signal. This can be used as a
    measurement of power.

    :param signal: the audio signal
    :returns:      root mean square of the signal

    """
    # make sure the signal is a numpy array
    if not isinstance(signal, np.ndarray):
        raise TypeError("Invalid type for audio signal.")
    # signal must be mono
    if signal.ndim > 1:
        # FIXME: please implement stereo (or multi-channel) handling
        raise NotImplementedError("please implement stereo functionality")
    # Note: type conversion needed because of integer overflows
    if signal.dtype != np.float:
        signal = signal.astype(np.float)
    # return
    return np.sqrt(np.dot(signal, signal) / signal.size)


def sound_pressure_level(signal, p_ref=1.0):
    """
    Computes the sound pressure level of a signal.

    :param signal: the audio signal
    :param p_ref:  reference sound pressure level [default=1.0]
    :returns:      sound pressure level of the signal

    From http://en.wikipedia.org/wiki/Sound_pressure:
    Sound pressure level (SPL) or sound level is a logarithmic measure of the
    effective sound pressure of a sound relative to a reference value.
    It is measured in decibels (dB) above a standard reference level.

    """
    # compute the RMS
    rms = root_mean_square(signal)
    # compute the SPL
    if rms == 0:
        # return the smallest possible negative number
        return -np.finfo(float).max
    else:
        # normal SPL computation
        return 20.0 * np.log10(rms / p_ref)


# function for automatically determining how to open audio files
def magic_signal_handler(filename, sample_rate=None):
    """
    Magic Signal opener. Tries to guess how to open a file.

    :param filename:    name of the file or file handle
    :param sample_rate: sample rate of the signal [default=None]
    :returns:           tuple (signal, sample_rate)

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


# default Audio values
MONO = False
NORM = False
ATT = 0


class Audio(object):
    """
    Audio is a very simple class which just stores the signal and the sample
    rate and provides some basic methods for signal processing.

    """
    def __init__(self, signal, sample_rate=None, mono=MONO, norm=NORM, att=ATT):
        """
        Creates a new Audio object instance.

        :param signal:      audio signal (numpy array or file or file handle)
        :param sample_rate: sample rate of the signal [default=None]
        :param mono:        downmix the signal to mono [default=False]
        :param norm:        normalize the signal [default=False]
        :param att:         attenuate the signal by N dB [default=0]

        """
        # signal handling
        if isinstance(signal, np.ndarray) and sample_rate is not None:
            # input is an numpy array + sample rate, use as is
            self.signal = signal
            self.sample_rate = sample_rate
        else:
            # try the magic_signal_handler
            self.signal, self.sample_rate = magic_signal_handler(signal, sample_rate)

        # convenience handling of mono down-mixing and normalization
        if mono:
            # down-mix to mono
            self.downmix()
        if norm:
            # normalize signal
            self.normalize()
        if att != 0:
            # attenuate signal
            self.attenuate(att)

    @property
    def num_samples(self):
        """Number of samples."""
        # TODO: cache this value and invalidate when signal changes
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
        return float(self.num_samples) / float(self.sample_rate)

    # downmix to mono
    def downmix(self):
        """Down-mix the audio signal to mono."""
        self.signal = downmix(self.signal)

    # normalize the signal
    def normalize(self):
        """Normalize the audio signal."""
        self.signal = normalize(self.signal)

    # attenuate the signal
    def attenuate(self, attenuation):
        """
        Attenuate the audio signal.

        :param attenuation: attenuation level [dB]

        """
        self.signal = attenuate(self.signal, attenuation)

    # truncate
    # FIXME: remove this method, since we can limit the range of interest in
    # the FramedAudio for example without having to write a new array.
    def truncate(self, offset=None, length=None):
        """
        Truncate the audio signal.

        :param offset: offset / start [seconds]
        :param length: length [seconds]

        """
        # truncate the beginning
        if offset != None:
            # check offset
            if offset <= 0:
                raise ValueError("offset must be positive")
            if offset * self.sample_rate > self.num_samples:
                raise ValueError("offset must be < length of signal")
            self.signal = self.signal[(offset * self.sample_rate):]
        # truncate the end
        if length != None:
            # check length
            if length <= 0:
                raise ValueError("a positive value must given")
            if length * self.sample_rate > self.num_samples:
                raise ValueError("length must be < length of signal")
            self.signal = self.signal[:length * self.sample_rate + 1]

    # downsample
    def downsample(self, factor=2):
        """
        Downsamples the audio signal by the given factor.

        :param factor: down-sampling factor [default=2]

        """
        self.signal = downsample(self.signal, factor)
        self.sample_rate /= factor

    # trim zeros
    # FIXME: remove this methods, since we can adjust the range of interest in
    # the FramedAudio for example without having to write a new array.
    def trim(self):
        """
        Trim leading and trailing zeros of the audio signal permanently.

        """
        self.signal = np.trim_zeros(self.signal, 'fb')

    # TODO: make this nicer!
    def __str__(self):
        return "%s length: %i samples (%.2f seconds) sample rate: %i" % (self.__class__, self.num_samples, self.length, self.sample_rate)


def signal_frame(signal, index, frame_size, hop_size, online):
    """
    This function returns frame[index] of the signal.

    :param signal:     the audio signal
    :param frame_size: size of one frame
    :param hop_size:   progress N samples between adjacent frames
    :param online:     use only past information
    :returns:          the single frame of the audio signal

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

    :param signal:     the discrete signal
    :param frame_size: size of each frame
    :param hop_size:   the hop size in samples between adjacent frames
    :returns:          the framed audio signal

    Note: This function is here only for completeness. It is faster only in rare
          circumstances. Also, seeking to the right position is only working
          properly, if integer hop_sizes are used.

    """
    # init variables
    samples = np.shape(signal)[0]
    # FIXME: does not perform the seeking the right way (only int working properly)
    # see http://www.scipy.org/Cookbook/SegmentAxis for a more detailed example
    as_strided = np.lib.stride_tricks.as_strided
    # return the strided array
    return as_strided(signal, (samples, frame_size), (signal.strides[0], signal.strides[0]))[::hop_size]


# default values
FRAME_SIZE = 2048
HOP_SIZE = 441.
FPS = 100.
ONLINE = False


class FramedAudio(object):
    """
    FramedAudio splits an audio signal into frames and makes them iterable.

    """
    def __init__(self, audio, frame_size=FRAME_SIZE, hop_size=HOP_SIZE,
                 online=ONLINE, fps=None, *args, **kwargs):
        """
        Creates a new FramedAudio object instance.

        :param audio:       an Audio object
        :param frame_size:  size of one frame [default=2048]
        :param hop_size:    progress N samples between adjacent frames [default=441]
        :param online:      use only past information [default=False]
        :param fps:         use N frames per second instead of setting the hop_size;
                            if set, this overwrites the hop_size value [default=None]

        Note: The FramedAudio class is implemented as an iterator. It splits the
              given Audio automatically into frames (of frame_size length) and
              progresses hop_size samples (can be float, with normal rounding
              applied) between frames.

              In offline mode the frame is centered around the current position;
              whereas in online mode, the frame is always positioned left to the
              current position.

        """
        # instantiate a Audio object if needed
        if isinstance(audio, Audio):
            self.audio = audio
        else:
            self.audio = Audio(audio, *args, **kwargs)
        # arguments for splitting the signal into frames
        self.frame_size = frame_size
        self.hop_size = float(hop_size)
        self.online = online

        # set fps instead of hop_size
        if fps:
            # Note: the default FPS is not used in __init__(), because usually
            # FRAME_SIZE and HOP_SIZE are used, but setting the fps overwrites
            # the hop_size automatically
            self.fps = fps

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
            # use this code if normal indexing behavior is needed
            # TODO: make index -1 work so that the diff of a spectrogram for the
            # first frame can be calculated in the correct way. Right now it is
            # just 0...
            if index < 0:
                index += self.num_frames
            return signal_frame(self.audio.signal, index, self.frame_size, self.hop_size, self.online)
        # other index types are invalid
        else:
            raise TypeError("Invalid argument type.")

    # len() should return the number of frames, since it iterates over frames
    def __len__(self):
        return self.num_frames

    @property
    def num_frames(self):
        """Number of frames."""
        return int(np.ceil(self.audio.num_samples / float(self.hop_size)))

    @property
    def fps(self):
        """Frames per second."""
        return float(self.audio.sample_rate) / float(self.hop_size)

    @fps.setter
    def fps(self, fps):
        """Frames per second."""
        # set the hop size accordingly
        self.hop_size = self.audio.sample_rate / float(fps)

    @property
    def overlap_factor(self):
        """Overlap factor of two adjacent frames."""
        return 1.0 - self.hop_size / self.window.size

    # TODO: make this nicer!
    def __str__(self):
        return "%s length: %i samples (%.2f seconds) sample rate: %i frames: %i (%i num_samples %.1f hop size)" % (self.__class__, self.num_samples, self.length, self.sample_rate, self.frames, self.frame_size, self.hop_size)
