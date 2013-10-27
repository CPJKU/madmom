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
def signal_frame(x, index, frame_size, hop_size, offset=0):
    """
    This function returns frame[index] of the signal.

    :param x:          the signal (numpy array)
    :param index:      the index of the frame to return
    :param frame_size: size of one frame in samples
    :param hop_size:   progress N samples between adjacent frames
    :param offset:     position of the first sample inside the signal
    :returns:          the requested single frame of the audio signal

    The first frame (index == 0) refers to the first sample of the signal, and
    each following frame is placed `hop_size` samples after the previous one.
    The window is always centered around this reference sample.

    `offset` sets the position of the first reference sample inside the signal.

    """
    # length of the signal
    num_samples = len(x)
    # seek to the correct position in the audio signal
    ref_sample = int(index * hop_size)
    # position the window
    start = ref_sample - frame_size / 2 + offset
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
START = 0
STOP = None


class FramedSignal(object):
    """
    FramedSignal splits a signal into frames and makes them iterable.

    """
    def __init__(self, signal, frame_size=None, hop_size=None, fps=None,
                 origin=None, mode=MODE, start=None, stop=None, *args, **kwargs):
        """
        Creates a new FramedSignal object instance.

        :param signal:     a Signal or FramedSignal instance
                           or anything a Signal can be instantiated from
        :param frame_size: size of one frame [default=2048]
        :param hop_size:   progress N samples between adjacent frames [default=441]
        :param fps:        use N frames per second instead of setting the hop_size;
                           if set, this overwrites the hop_size value [default=None]
        :param origin:     location of the window relative to the signal position [default=0]
        :param mode:       end of signal handling [default='extend']
        :param start:      start sample [default=0]
        :param stop:       stop sample [default=None]

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

        The `mode` parameter is used to set the end of signal handling
        - 'normal': the origin of the last frame has to be within the signal
        - 'extend': frames are returned as long as part of the frame overlaps
          with the signal

        If only a certain part of the Signal is wanted, `start` and `stop` can
        be used to set these values accordingly.
        `start` is given in samples and marks the first sample of the audio.
        `stop` is given in samples and marks the last sample to return.

        """
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
            self._stop = signal.stop
        else:
            # try to instantiate a Signal
            self._signal = Signal(signal, *args, **kwargs)
            # set attributes to default values (which can be overwritten by
            # passing other values to the constructor)
            self._frame_size = FRAME_SIZE
            self._hop_size = HOP_SIZE
            self._origin = ORIGIN
            self._mode = MODE
            self._start = START
            self._stop = STOP

        # arguments for splitting the signal into frames
        if frame_size:
            self._frame_size = int(frame_size)
        if hop_size:
            self._hop_size = float(hop_size)
        # use fps instead of hop_size
        if fps:
            # Note: using fps overwrites the hop_size
            self._hop_size = self._signal.sample_rate / float(fps)

        # translate literal values
        if origin in ('center', 'offline'):
            # the current position is the center of the frame
            origin = 0
        elif origin in ('left', 'past', 'online'):
            # the current position is the right edge of the frame
            # this is usually used when simulating online mode, where only past
            # information of the audio signal can be used
            origin = +(frame_size - 1) / 2
        elif origin in ('right', 'future'):
            # the current position is the left edge of the frame
            origin = -(frame_size / 2)
            # location of the window
        if origin is not None:
            # overwrite the default origin
            self._origin = origin

        # start/stop position
        if start:
            # the start position of the signal (in samples)
            self._start = int(start)
        if stop:
            # the stop position of the signal (in samples)
            self._stop = int(stop)
        if self._stop is None:
            # if no stop position is given, use the last sample
            self._stop = self.signal.num_samples

        # length of the signal
        length = self._stop - self._start

        # number of frames handling
        self._mode = mode
        if self._mode == 'extend':
            # return frames as long as frame covers any signal
            self._num_frames = int(np.floor((length) / float(self.hop_size) + 1))
        elif self._mode == 'normal':
            # return frames as long as the origin sample covers the signal
            self._num_frames = int(np.ceil(length / float(self.hop_size)))
        else:
            raise ValueError('invalid mode')

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
        # a single index is given
        if isinstance(index, int):
            # return a single frame
            if index <= self.num_frames:
                # return the frame at this index
                # subtract the origin from the start position and use as the offset
                return signal_frame(self.signal.data, index, frame_size=self.frame_size,
                                    hop_size=self.hop_size, offset=(self.start - self.origin))
            # otherwise raise an error to indicate the end of signal
            raise IndexError("end of signal reached")
        # a slice is given
        elif isinstance(index, slice):
            # create a new FramedSignal object which covers the signal part
            # determine the start frame
            if index.start:
                start = index.start
            else:
                start = 0
            # determine the stop frame
            if index.stop and index.start:
                stop = index.stop - index.start
            elif index.stop:
                stop = index.stop
            elif index.start:
                stop = self.num_frames - index.start
            else:
                stop = self.num_frames
            # convert start and stop to samples
            start *= self.hop_size
            stop *= self.hop_size
            # just allow normal steps
            if (index.step is not None) and (index.step != 1):
                raise ValueError('only slices with a step size of 1 are supported')
            # create the object and return it
            return FramedSignal(self.signal, start=start, stop=stop)
        # other index types are invalid
        else:
            raise TypeError("frame indices must be slices or integers, not %s" % index.__class__.__name__)

    @property
    def signal(self):
        """The underlying (audio) signal."""
        return self._signal

    @property
    def frame_size(self):
        """Size of one frame."""
        return self._frame_size

    @property
    def hop_size(self):
        """Number of samples between adjacent frames."""
        return self._hop_size

    @property
    def origin(self):
        """Origin of the frame center relative to the signal position."""
        return self._origin

    @property
    def mode(self):
        """End of signal handling mode."""
        return self._mode

    @property
    def start(self):
        """Start sample of the (audio) signal."""
        return self._start

    @property
    def stop(self):
        """Stop sample of the (audio) signal."""
        return self._stop

    @property
    def num_frames(self):
        """Number of frames."""
        return self._num_frames

    # len() returns the number of frames, consistent with __getitem__()
    def __len__(self):
        return self.num_frames

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
