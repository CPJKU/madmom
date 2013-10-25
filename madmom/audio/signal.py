#!/usr/bin/env python
# encoding: utf-8
"""
This file contains basic signal processing functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import numpy as np


# signal functions
def attenuate(x, attenuation):
    """"
    Attenuate the signal.

    :param x:           signal (numpy array)
    :param attenuation: attenuation level [dB]
    :returns:           attenuated signal

    """
    # FIXME: attenuating the signal and keeping the original dtype makes the
    # following signal processing steps well-behaved, since these rely on
    # the dtype of the array to determine the correct value range.
    # But this introduces rounding (truncating) errors in case of signals
    # with int dtypes. But these errors should be negligible.
    return np.asarray(x / np.power(np.sqrt(10.), attenuation / 10.), dtype=x.dtype)


def normalize(x):
    """
    Normalize the signal to the range -1..+1

    :param x: signal (numpy array)
    :returns: normalized signal

    """
    return x.astype(float) / np.max(x)


def downmix(x):
    """
    Down-mix the signal to mono.

    :param x: signal (numpy array)
    :returns: mono signal

    """
    if x.ndim > 1:
        # FIXME: taking the mean and keeping the original dtype makes the
        # following signal processing steps well-behaved, since these rely on
        # the dtype of the array to determine the correct value range.
        # But this introduces rounding (truncating) errors in case of signals
        # with int dtypes. But these errors should be negligible.
        return np.mean(x, axis=-1, dtype=x.dtype)
    else:
        return x


def downsample(x, factor=2):
    """
    Down-samples the signal by the given factor

    :param x:      signal (numpy array)
    :param factor: down-sampling factor [default=2]
    :returns:      down-sampled signal

    """
    # signal must be mono
    if x.ndim > 1:
        # FIXME: please implement stereo (or multi-channel) handling
        raise NotImplementedError("please implement stereo functionality")
    # when downsampling by an integer factor, a simple view is more efficient
    if type(factor) == int:
        return x[::factor]
    # otherwise do more or less propoer down-sampling
    # TODO: maybe use sox to implement this
    from scipy.signal import decimate
    # naive down-sampling
    return np.hstack(decimate(x, factor))


def trim(x):
    """
    Trim leading and trailing zeros of the signal.

    :param x: signal (numpy array)
    :returns: trimmed signal

    """
    # signal must be mono
    if x.ndim > 1:
        # FIXME: please implement stereo (or multi-channel) handling
        # maybe it works, haven't checked
        raise NotImplementedError("please implement stereo functionality")
    return np.trim_zeros(x, 'fb')


def root_mean_square(x):
    """
    Computes the root mean square of the signal. This can be used as a
    measurement of power.

    :param x: signal (numpy array)
    :returns: root mean square of the signal

    """
    # make sure the signal is a numpy array
    if not isinstance(x, np.ndarray):
        raise TypeError("Invalid type for signal.")
    # signal must be mono
    if x.ndim > 1:
        # FIXME: please implement stereo (or multi-channel) handling
        raise NotImplementedError("please implement stereo functionality")
    # Note: type conversion needed because of integer overflows
    if x.dtype != np.float:
        x = x.astype(np.float)
    # return
    return np.sqrt(np.dot(x, x) / x.size)


def sound_pressure_level(x, p_ref=1.0):
    """
    Computes the sound pressure level of a signal.

    :param x:     signal (numpy array)
    :param p_ref: reference sound pressure level [default=1.0]
    :returns:     sound pressure level of the signal

    From http://en.wikipedia.org/wiki/Sound_pressure:
    Sound pressure level (SPL) or sound level is a logarithmic measure of the
    effective sound pressure of a sound relative to a reference value.
    It is measured in decibels (dB) above a standard reference level.

    """
    # compute the RMS
    rms = root_mean_square(x)
    # compute the SPL
    if rms == 0:
        # return the smallest possible negative number
        return -np.finfo(float).max
    else:
        # normal SPL computation
        return 20.0 * np.log10(rms / p_ref)


# default Signal values
MONO = False
NORM = False
ATT = 0


class Signal(object):
    """
    Signal is a very simple class which just stores a reference to the signal
    and the sample rate and provides some basic methods for signal processing.

    """
    def __init__(self, data, sample_rate=None, mono=MONO, norm=NORM, att=ATT):
        """
        Creates a new Signal object instance.

        :param data:        numpy array (`sample_rate` must be given as well)
                            or Signal instance or file name or file handle
        :param sample_rate: sample rate of the signal [default=None]
        :param mono:        downmix the signal to mono [default=False]
        :param norm:        normalize the signal [default=False]
        :param att:         attenuate the signal by N dB [default=0]

        """
        # data handling
        if isinstance(data, np.ndarray) and sample_rate is not None:
            # data is an numpy array, use it directly
            self._data = data
        elif isinstance(data, Signal):
            # already a Signal, copy the object attributes (which can be
            # overwritten by passing other values to the constructor)
            self._data = data.data
            self._sample_rate = data.sample_rate
        else:
            # try to load an audio file
            from .io import load_audio_file
            self._data, self._sample_rate = load_audio_file(data, sample_rate)
        # set sample rate
        if sample_rate is not None:
            self._sample_rate = sample_rate
        # convenience handling of mono down-mixing and normalization
        if mono:
            # down-mix to mono
            self._data = downmix(self._data)
        if norm:
            # normalize signal
            self._data = normalize(self._data)
        if att != 0:
            # attenuate signal
            self._data = attenuate(self._data)

    @property
    def data(self):
        # make data immutable
        return self._data

    @property
    def sample_rate(self):
        # make sample rate immutable
        return self._sample_rate

    # len() returns the number of samples
    def __len__(self):
        return self.num_samples

    @property
    def num_samples(self):
        """Number of samples."""
        return len(self.data)

    @property
    def num_channels(self):
        """Number of channels."""
        try:
            # multi channel files
            return np.shape(self.data)[1]
        except IndexError:
            # catch mono files
            return 1

    @property
    def length(self):
        """Length of signal in seconds."""
        return float(self.num_samples) / float(self.sample_rate)

    # TODO: make this nicer!
    def __str__(self):
        return "%s length: %i samples (%.2f seconds) sample rate: %i" % (self.__class__, self.num_samples, self.length, self.sample_rate)


# function for splitting a signal into frames
def signal_frame(x, index, frame_size, hop_size, origin=0):
    """
    This function returns frame[index] of the signal.

    :param x:          the signal (numpy array)
    :param index:      the index of the frame to return
    :param frame_size: size of one frame in samples
    :param hop_size:   progress N samples between adjacent frames
    :param origin:     location of the window relative to the signal position
    :returns:          the requested single frame of the audio signal

    The first frame (index == 0) refers to the first sample of the signal, and
    each following frame is placed `hop_size` samples after the previous one.

    An `origin` of zero centers the frame around its reference sample,
    an `origin` of `+(frame_size-1)/2` places the frame to the left of the
    reference sample, with the reference forming the last sample of the frame,
    and an `origin` of `-frame_size/2` places the frame to the right of the
    reference sample, with the reference forming the first sample of the frame.
    """
    # length of the signal
    num_samples = len(x)
    # seek to the correct position in the audio signal
    ref_sample = int(index * hop_size)
    # position the window
    start = ref_sample - frame_size / 2 - origin
    stop = start + frame_size
    # return the requested portion of the signal
    if (stop < 0) or (start > num_samples):
        # window falls completely outside the actual signal, return just zeros
        return np.zeros((frame_size,) + x.shape[1:], dtype=x.dtype)
    elif start < 0:
        # window crosses left edge of actual signal, pad zeros from left
        frame = np.empty((frame_size,) + x.shape[1:], dtype=x.dtype)
        frame[:-start] = 0
        frame[-start:] = x[:stop]
        return frame
    elif stop > num_samples:
        # window crosses right edge of actual signal, pad zeros from right
        frame = np.empty((frame_size,) + x.shape[1:], dtype=x.dtype)
        frame[:num_samples - start] = x[start:]
        frame[num_samples - start:] = 0
        return frame
    else:
        # normal read operation
        return x[start:stop]


def strided_frames(x, frame_size, hop_size):
    """
    Returns a 2D representation of the signal with overlapping frames.

    :param x:          the signal (numpy array)
    :param frame_size: size of each frame
    :param hop_size:   the hop size in samples between adjacent frames
    :returns:          the framed signal

    Note: This function is here only for completeness. It is faster only in rare
          circumstances. Also, seeking to the right position is only working
          properly, if integer hop_sizes are used.

    """
    # init variables
    samples = np.shape(x)[0]
    # FIXME: does not perform the seeking the right way (only int working properly)
    # see http://www.scipy.org/Cookbook/SegmentAxis for a more detailed example
    as_strided = np.lib.stride_tricks.as_strided
    # return the strided array
    return as_strided(x, (samples, frame_size), (x.strides[0], x.strides[0]))[::hop_size]


# default values for splitting a signal into overlapping frames
FRAME_SIZE = 2048
HOP_SIZE = 441.
FPS = 100.
ORIGIN = 0
MODE = 'extend'
ONLINE = False


class FramedSignal(object):
    """
    FramedSignal splits a signal into frames and makes them iterable.

    """
    def __init__(self, signal, frame_size=FRAME_SIZE, hop_size=HOP_SIZE,
                 origin=ORIGIN, mode=MODE, online=ONLINE, fps=None,
                 start=None, length=None, *args, **kwargs):
        """
        Creates a new FramedSignal object instance.

        :param signal:     a Signal or FramedSignal instance
                           or anything a Signal can be instantiated from
        :param frame_size: size of one frame [default=2048]
        :param hop_size:   progress N samples between adjacent frames [default=441]
        :param origin:     location of the window relative to the signal position [default=0]
        :param mode:       TODO: meaningful description [default=extend]
        :param online:     use only past information [default=False]
        :param fps:        use N frames per second instead of setting the hop_size;
                           if set, this overwrites the hop_size value [default=None]
        :param start:      start sample [default=0]
        :param length:     length in frames [default=None]

        The FramedSignal class is implemented as an iterator. It splits the
        given Signal automatically into frames (of `frame_size` length) and
        progresses `hop_size` samples (can be float, normal rounding applies)
        between frames.

        The location of the window relative to its reference sample can be set
        with the `origin` parameter. Arbitrary integer values can be given
        - zero centers the window on its reference sample
        - negative values shift the window to the right
        - positive values shift the window to the left
        Additionally, it can have the following literal values:
        - 'center', 'offline': the window is centered on its reference sample
        - 'left', 'past', 'online': the window is located to the left of
          its reference sample (including the reference sample)
        - 'right', 'future': the window is located to the right of its
          reference sample

        `mode` handles how far frames may reach past the end of the signal
        - TODO: mode descriptions go here

        `online` is a shortcut switch for backwards compatibility and sets the
        `origin` to `left` and the `mode` to `normal`. This parameter will be
        removed in the near future!

        """
        # set default start position to 0
        self._start = 0
        # signal handling
        if isinstance(signal, FramedSignal):
            # already a FramedSignal, copy the object attributes (which can be
            # overwritten by passing other values to the constructor)
            self._signal = signal.signal
            self._frame_size = signal.frame_size
            self._hop_size = signal.hop_size
            self._origin = signal.origin
            self._mode = signal.mode
            self._start = signal.start
        else:
            # try to instantiate a Signal
            self._signal = Signal(signal, *args, **kwargs)

        # arguments for splitting the signal into frames
        if frame_size:
            self._frame_size = int(frame_size)
        if hop_size:
            self._hop_size = float(hop_size)
        # use fps instead of hop_size
        if fps:
            # Note: using fps overwrites the hop_size
            self._hop_size = self._signal.sample_rate / float(fps)

        # set origin and mode to reflect `online mode`
        if online:
            origin = 'left'
            mode = 'normal'

        # location of the window
        if origin in ('center', 'offline'):
            # the current position is the center of the frame
            self._origin = 0
        elif origin in ('left', 'past', 'online'):
            # the current position is the right edge of the frame
            # this is usually used when simulating online mode, where only past
            # information of the audio signal can be used
            self._origin = +(frame_size - 1) / 2
        elif origin in ('right', 'future'):
            self._origin = -(frame_size / 2)
        else:
            try:
                self._origin = int(origin)
            except ValueError:
                raise ValueError('invalid origin')

        # mode
        if mode == 'extend':
            # FIXME: should we save the mode, or is it enough to use it for the
            # calculation of the number of frames
            self._mode = 'extend'
            self._num_frames = int(np.floor(len(self.signal.data) / float(self.hop_size)) + 1)
        else:
            self._mode = 'normal'
            self._num_frames = int(np.ceil(len(self.signal.data) / float(self.hop_size)))

        # start and length
        if start:
            # the internal start position is stored in samples
            self._start = start
        if length:
            # set the length to the given number of frames
            self._num_frames = length

    # make the Object indexable
    def __getitem__(self, index):
        """
        This makes the FramedSignal class an indexable object.

        The signal is split into frames (of length frame_size) automatically.
        Two frames are located hop_size samples apart. hop_size can be float,
        normal rounding applies.

        Note: index -1 refers NOT to the last frame, but to the frame directly
        left of frame 0. Although this is contrary to common behavior, being
        able to access these frames is important, because if the frames overlap
        frame -1 contains parts of the signal of frame 0.

        """
        # a slice is given
        if isinstance(index, slice):
            # determine the start position (in samples) of the new object
            if index.start:
                # FIXME: should we set this to integers or allow floats as well?
                start = int(index.start * self.hop_size)
            else:
                start = 0
            # determine the length of the new object
            if index.stop and index.start:
                length = index.stop - index.start
            elif index.stop:
                length = index.stop
            elif index.start:
                length = self.num_frames - index.start
            else:
                length = self.num_frames
            # just allow normal steps
            if (index.step is not None) and (index.step != 1):
                raise ValueError('only slices with a step size of 1 are supported')
            # create a new FramedSignal object and return it
            return FramedSignal(signal=self.signal, frame_size=self.frame_size,
                                hop_size=self.hop_size, origin=self.origin,
                                start=start, length=length)
        # a single index is given
        elif isinstance(index, int):
            # return a single frame
            if index > self.num_frames:
                raise IndexError("end of signal reached")
            # return the frame at this index
            # subtract the start position (in samples) from the origin and use
            # it as the origin (negative origin shifts to the right)
            return signal_frame(self.signal.data, index, self.frame_size,
                                self.hop_size, self.origin - self.start)
        # other index types are invalid
        else:
            raise TypeError("frame indices must be integers, not %s" % index.__class__.__name__)

    @property
    def signal(self):
        return self._signal

    @property
    def frame_size(self):
        return self._frame_size

    @property
    def hop_size(self):
        return self._hop_size

    @property
    def mode(self):
        # FIXME: do we need to save and access this property?
        return self._mode

    @property
    def origin(self):
        return self._origin

    @property
    def start(self):
        return self._start

    # len() returns the number of frames, consistent with __getitem__()
    def __len__(self):
        return self.num_frames

    @property
    def num_frames(self):
        return self._num_frames

    @property
    def fps(self):
        """Frames per second."""
        return float(self.signal.sample_rate) / float(self.hop_size)

    @property
    def overlap_factor(self):
        """Overlap factor of two adjacent frames."""
        return 1.0 - self.hop_size / self.window.size

    # TODO: make this nicer!
    def __str__(self):
        return "%s length: %i samples (%.2f seconds) sample rate: %i frames: %i (%i num_samples %.1f hop size)" % (self.__class__, self.signal.num_samples, self.signal.length, self.signal.sample_rate, self.num_frames, self.frame_size, self.hop_size)
