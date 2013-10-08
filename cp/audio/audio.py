#!/usr/bin/env python
# encoding: utf-8
"""
This file contains basic signal processing functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import numpy as np


def root_mean_square(signal):
    """
    Computes the root mean square of the signal. This can be used as
    a measurement of power.

    :param signal: the audio signal
    :returns:      root mean square of the signal

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
        # return the largest possible negative number
        return -np.finfo(float).max
    else:
        # normal SPL computation
        return 20.0 * np.log10(rms / p_ref)


def signal_frame(signal, index, frame_size, hop_size, origin=0):
    """
    This function returns frame[index] of the signal.

    :param signal:     the audio signal as a numpy array
    :param index:      the index of the frame to return
    :param frame_size: size of one frame in samples
    :param hop_size:   progress N samples between adjacent frames
    :param origin:     location of the window relative to the signal position
    :returns:          the requested single frame of the audio signal

    The first frame (index == 0) refers to the first sample of the signal, and
    each following frame is placed `hop_size` samples after the previous one.

    An `origin` of zero centers the frame around its reference sample,
    an `origin` of `+(frame_size-1)/2` places the frame to the left of the reference
    sample, with the reference forming the last sample of the frame, and
    an `origin` of `-frame_size/2` places the frame to the right of the reference
    sample, with the reference forming the first sample of the frame.
    """
    # length of the signal
    num_samples = len(signal)
    # seek to the correct position in the audio signal
    ref_sample = int(index * hop_size)
    # position the window
    start = ref_sample - frame_size / 2 - origin
    stop = start + frame_size
    # return the requested portion of the signal
    if (stop < 0) or (start > num_samples):
        # window falls completely outside the actual signal, return just zeros
        return np.zeros((frame_size,) + signal.shape[1:], dtype=signal.dtype)
    elif start < 0:
        # window crosses left edge of actual signal, pad zeros from left
        frame = np.empty((frame_size,) + signal.shape[1:], dtype=signal.dtype)
        frame[:-start] = 0
        frame[-start:] = signal[:stop]
        return frame
    elif stop > num_samples:
        # window crosses right edge of actual signal, pad zeros from right
        frame = np.empty((frame_size,) + signal.shape[1:], dtype=signal.dtype)
        frame[:num_samples - start] = signal[start:]
        frame[num_samples - start:] = 0
        return frame
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
MONO = False
NORM = False
ATT = 0


class Audio(object):
    """
    Audio is a very simple class which just stores the signal and the sample
    rate and provides some basic methods for signal processing.

    """
    def __init__(self, signal, sample_rate, mono=MONO, norm=NORM, att=ATT):
        """
        Creates a new Audio object instance.

        :param signal:      the audio signal [numpy array]
        :param sample_rate: sample rate of the signal
        :param mono:        downmix the signal to mono [default=False]
        :param norm:        normalize the signal [default=False]
        :param att:         attenuate the signal by N dB [default=0]

        """
        if not isinstance(signal, np.ndarray):
            # make sure the signal is a numpy array
            raise TypeError("Invalid type for audio signal.")
        self.signal = signal
        self.sample_rate = sample_rate

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
        if self.num_channels > 1:
            self.signal = np.mean(self.signal, -1)

    # normalize the signal
    def normalize(self):
        """Normalize the audio signal."""
        self.signal = self.signal.astype(float) / np.max(self.signal)

    # attenuate the signal
    def attenuate(self, attenuation):
        """
        Attenuate the audio signal.

        :param attenuation: attenuation level [dB]

        """
        self.signal /= np.power(np.sqrt(10.), attenuation / 10.)

    # truncate
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
        from scipy.signal import decimate
        self.signal = np.hstack(decimate(self.signal, factor))
        self.sample_rate /= factor

    # trim zeros
    def trim(self):
        """
        Trim leading and trailing zeros of the audio signal permanently.

        """
        self.signal = np.trim_zeros(self.signal, 'fb')

    # TODO: make this nicer!
    def __str__(self):
        return "%s length: %i samples (%.2f seconds) sample rate: %i" % (self.__class__, self.num_samples, self.length, self.sample_rate)


# default values
FRAME_SIZE = 2048
HOP_SIZE = 441.
FPS = 100.
ORIGIN = 0
MODE = 'extend'
ONLINE = False


class FramedAudio(Audio):
    """
    FramedAudio splits an audio signal into frames and makes them iterable.

    """
    def __init__(self, signal, sample_rate, frame_size=FRAME_SIZE, hop_size=HOP_SIZE,
                 origin=ORIGIN, mode=MODE, fps=None, online=None, *args, **kwargs):
        """
        Creates a new FramedAudio object instance.

        :param signal:      the audio signal [numpy array]
        :param sample_rate: sample_rate of the signal
        :param frame_size:  size of one frame [default=2048]
        :param hop_size:    progress N samples between adjacent frames [default=441]
        :param origin:      location of the window relative to the signal position [default=0]
        :param mode:        TODO: meaningful description [default=extend]
        :param fps:         use N frames per second instead of setting the hop_size;
                            if set, this overwrites the hop_size value [default=None]
        :param online:      set origin and mode to only use past information [default=None]

        Note: The FramedAudio class is implemented as an iterator. It splits the
              signal automatically into frames (of frame_size length) and
              progresses hop_size samples (can be float, with normal rounding
              applied) between frames.

              The location of the window relative to its reference sample can
              be set with the `origin` parameter. It can have the following
              literal values:
              - 'center', 'offline': the window is centered on its reference sample
              - 'left', 'past', 'online': the window is located to the left of
                its reference sample (including the reference sample)
              - 'right', 'future': the window is located to the right of its
                reference sample
              Additionally, arbitrary integer values can be given
              - zero centers the window on its reference sample
              - negative values shift the window to the right
              - positive values shift the window to the left

              `mode` handles how far frames may reach past the end of the signal
              - TODO: mode descriptions go here

        """
        # instantiate a Audio object
        super(FramedAudio, self).__init__(signal, sample_rate, *args, **kwargs)
        # arguments for splitting the signal into frames
        self.frame_size = frame_size
        self.hop_size = float(hop_size)
        # set fps instead of hop_size
        if fps:
            # Note: the default FPS is not used in __init__(), because usually
            # FRAME_SIZE and HOP_SIZE are used, but setting the fps overwrites
            # the hop_size automatically
            self.fps = fps
        # set origin and mode to reflect `online mode`
        if online:
            origin = 'left'
            mode = 'normal'
        # location of the window
        if origin in ('center', 'offline'):
            # the current position is the center of the frame
            self.origin = 0
        elif origin in ('left', 'past', 'online'):
            # the current position is the right edge of the frame
            # this is usually used when simulating online mode, where only past
            # information of the audio signal can be used
            self.origin = +(frame_size - 1) / 2
        elif origin in ('right', 'future'):
            self.origin = -(frame_size / 2)
        else:
            try:
                self.origin = int(origin)
            except ValueError:
                raise ValueError('invalid origin')
        # signal range handling
        if mode == 'extend':
            self.mode = 'extend'
        else:
            self.mode = 'normal'

    # make the Object indexable
    def __getitem__(self, index):
        """
        This makes the FramedAudio class an indexable object.

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
            # return a single frame
            # (as specified above, negative indices do not wrap around)
            return signal_frame(self.signal, index, self.frame_size, self.hop_size, self.origin)
        # other index types are invalid
        else:
            raise TypeError("frame indices must be integers, not %s" % index.__class__.__name__)

    # len() returns the number of frames, consistent with __getitem__()
    def __len__(self):
        return self.num_frames

    @property
    def num_frames(self):
        """Number of frames."""
        if self.mode == 'extend':
            return int(np.floor((np.shape(self.signal)[0]) / float(self.hop_size)) + 1)
        else:
            return int(np.ceil((np.shape(self.signal)[0]) / float(self.hop_size)))

    @property
    def fps(self):
        """Frames per second."""
        return float(self.sample_rate) / float(self.hop_size)

    @fps.setter
    def fps(self, fps):
        """Frames per second."""
        # set the hop size accordingly
        self.hop_size = self.sample_rate / float(fps)

    @property
    def overlap_factor(self):
        """Overlap factor of two adjacent frames."""
        return 1.0 - self.hop_size / self.window.size

    # TODO: make this nicer!
    def __str__(self):
        return "%s length: %i samples (%.2f seconds) sample rate: %i frames: %i (%i num_samples %.1f hop size)" % (self.__class__, self.num_samples, self.length, self.sample_rate, self.frames, self.frame_size, self.hop_size)
