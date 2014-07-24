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
    :return:            attenuated signal

    """
    # FIXME: attenuating the signal and keeping the original dtype makes the
    # following signal processing steps well-behaved, since these rely on
    # the dtype of the array to determine the correct value range.
    # But this introduces rounding (truncating) errors in case of signals
    # with int dtypes. But these errors should be negligible.
    return np.asarray(x / np.power(np.sqrt(10.), attenuation / 10.),
                      dtype=x.dtype)


def normalize(x):
    """
    Normalize the signal to the range -1..+1

    :param x: signal (numpy array)
    :return:  normalized signal

    """
    return x.astype(np.float) / np.max(x)


def downmix(x):
    """
    Down-mix the signal to mono.

    :param x: signal (numpy array)
    :return:  mono signal

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
    :param factor: down-sampling factor
    :return:       down-sampled signal

    """
    # signal must be mono
    if x.ndim > 1:
        # FIXME: please implement stereo (or multi-channel) handling
        raise NotImplementedError("please implement stereo functionality")
    # when down-sampling by an integer factor, a simple view is more efficient
    if type(factor) == int:
        return x[::factor]
    # otherwise do more or less proper down-sampling
    # TODO: maybe use sox to implement this
    from scipy.signal import decimate
    # naive down-sampling
    return np.hstack(decimate(x, factor))


def trim(x):
    """
    Trim leading and trailing zeros of the signal.

    :param x: signal (numpy array)
    :return:  trimmed signal

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
    :return:  root mean square of the signal

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
    :param p_ref: reference sound pressure level
    :return:      sound pressure level of the signal

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
        Creates a new Signal instance.

        :param data:        numpy array (`sample_rate` must be given as well)
                            or Signal instance or file name or file handle
        :param sample_rate: sample rate of the signal [Hz]
        :param mono:        down-mix the signal to mono
        :param norm:        normalize the signal
        :param att:         attenuate the signal [dB]

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
        if att is not None and att != 0:
            # attenuate signal
            self._data = attenuate(self._data, att)

    @property
    def data(self):
        """The raw audio signal data."""
        return self._data

    @property
    def sample_rate(self):
        """Sample rate of the audio signal."""
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

    def copy(self, sample_rate=None, mono=None, norm=None, att=None):
        """
        Copies the Signal object and adjusts some parameters.

        :param sample_rate: sample rate of the signal [Hz]
        :param mono:        down-mix the signal to mono
        :param norm:        normalize the signal
        :param att:         attenuate the signal [dB]
        :return:            Signal object with given parameters

        """
        # copy the object attributes unless overwritten by passing other values
        if sample_rate is None:
            sample_rate = self.sample_rate
        # return a new Signal
        return Signal(self.data, sample_rate=sample_rate, mono=mono, norm=norm,
                      att=att)

    def __str__(self):
        return "Signal: %d samples (%.2f sec); %d channel(s); %d Hz sample " \
               "rate" % (self.num_samples, self.length, self.num_channels,
                         self.sample_rate)

    @staticmethod
    def add_arguments(parser, norm=NORM, att=ATT):
        """
        Add audio signal related arguments to an existing parser object.

        :param parser: existing argparse parser object
        :param norm:   normalize the signal
        :param att:    attenuate the signal [dB]
        :return:       audio signal argument parser group object

        """
        # add wav options to the existing parser
        g = parser.add_argument_group('audio signal arguments')
        if norm is not None:
            g.add_argument('--norm', action='store_true', default=norm,
                           help='normalize the audio signal')
        if att is not None:
            g.add_argument('--att', action='store', type=float, default=att,
                           help='attenuate the audio signal [dB]')
        # return the argument group so it can be modified if needed
        return g


# function for splitting a signal into frames
def signal_frame(x, index, frame_size, hop_size, offset=0):
    """
    This function returns frame[index] of the signal.

    :param x:          the signal (numpy array)
    :param index:      the index of the frame to return
    :param frame_size: size of each frame in samples
    :param hop_size:   the hop size in samples between adjacent frames
    :param offset:     position of the first sample inside the signal
    :return:           the requested frame of the signal

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
    elif (start < 0) and (stop > num_samples):
        # window surrounds the actual signal, position signal accordingly
        frame = np.zeros((frame_size,) + x.shape[1:], dtype=x.dtype)
        frame[-start:num_samples - start] = x
        return frame
    elif start < 0:
        # window crosses left edge of actual signal, pad zeros from left
        frame = np.zeros((frame_size,) + x.shape[1:], dtype=x.dtype)
        frame[-start:] = x[:stop]
        return frame
    elif stop > num_samples:
        # window crosses right edge of actual signal, pad zeros from right
        frame = np.zeros((frame_size,) + x.shape[1:], dtype=x.dtype)
        frame[:num_samples - start] = x[start:]
        return frame
    else:
        # normal read operation
        return x[start:stop]


def strided_frames(x, frame_size, hop_size):
    """
    Returns a 2D representation of the signal with overlapping frames.

    :param x:          the signal (numpy array)
    :param frame_size: size of each frame in samples
    :param hop_size:   the hop size in samples between adjacent frames
    :return:           2D array with overlapping frames

    Note: This function is here only for completeness. It is faster only in
          rare circumstances. Also, seeking to the right position is only
          working properly, if an integer `hop_size` are used.

    """
    # init variables
    samples = np.shape(x)[0]
    # FIXME: does seeking only the right way for integer hop_size
    # see http://www.scipy.org/Cookbook/SegmentAxis for a more detailed example
    as_strided = np.lib.stride_tricks.as_strided
    # return the strided array
    return as_strided(x, (samples, frame_size),
                      (x.strides[0], x.strides[0]))[::hop_size]


# taken from: http://www.scipy.org/Cookbook/SegmentAxis
def segment_axis(x, frame_size, hop_size=0, axis=None, end='cut', end_value=0):
    """
    Generate a new array that chops the given array along the given axis into
    overlapping frames.

    :param x:          the signal (numpy array)
    :param frame_size: size of each frame in samples
    :param hop_size:   the hop size in samples between adjacent frames
    :param axis:       axis to operate on; if None, act on the flattened array
    :param end:        what to do with the last frame, if the array is not
                       evenly divisible into pieces; possible values:
                       'cut'  simply discard the extra values
                       'wrap' copy values from the beginning of the array
                       'pad'  pad with a constant value
    :param end_value:  value to use for end='pad'
    :return:           2D array with overlapping frames

    The array is not copied unless necessary (either because it is unevenly
    strided and being flattened or because end is set to 'pad' or 'wrap').

    Example:
    >>> segment_axis(np.arange(10), 4, 2)
    array([[0, 1, 2, 3],
           [2, 3, 4, 5],
           [4, 5, 6, 7],
           [6, 7, 8, 9]])

    """

    if axis is None:
        x = np.ravel(x)  # may copy
        axis = 0

    length = x.shape[axis]

    if hop_size <= 0:
        raise ValueError("hop_size must be positive.")
    if frame_size <= 0:
        raise ValueError("frame_size must be positive.")

    if length < frame_size or (length - frame_size) % hop_size:
        if length > frame_size:
            round_up = (frame_size + (1 + (length - frame_size) // hop_size) *
                        hop_size)
            round_down = (frame_size + ((length - frame_size) // hop_size) *
                          hop_size)
        else:
            round_up = frame_size
            round_down = 0
        assert round_down < length < round_up
        assert round_up == round_down + hop_size or (round_up == frame_size and
                                                     round_down == 0)
        x = x.swapaxes(-1, axis)

        if end == 'cut':
            x = x[..., :round_down]
        elif end in ['pad', 'wrap']:
            # need to copy
            s = list(x.shape)
            s[-1] = round_up
            y = np.empty(s, dtype=x.dtype)
            y[..., :length] = x
            if end == 'pad':
                y[..., length:] = end_value
            elif end == 'wrap':
                y[..., length:] = x[..., :round_up - length]
            x = y

        x = x.swapaxes(-1, axis)

    length = x.shape[axis]
    if length == 0:
        raise ValueError("Not enough data points to segment array in 'cut' "
                         "mode; try end='pad' or end='wrap'")
    assert length >= frame_size
    assert (length - frame_size) % hop_size == 0
    n = 1 + (length - frame_size) // hop_size
    s = x.strides[axis]
    new_shape = x.shape[:axis] + (n, frame_size) + x.shape[axis + 1:]
    new_strides = x.strides[:axis] + (hop_size * s, s) + x.strides[axis + 1:]

    try:
        return np.ndarray.__new__(np.ndarray, strides=new_strides,
                                  shape=new_shape, buffer=x, dtype=x.dtype)
    except TypeError:
        # TODO: remove warning?
        import warnings
        warnings.warn("Problem with ndarray creation forces copy.")
        x = x.copy()
        # Shape doesn't change but strides does
        new_strides = (x.strides[:axis] + (hop_size * s, s) +
                       x.strides[axis + 1:])
        return np.ndarray.__new__(np.ndarray, strides=new_strides,
                                  shape=new_shape, buffer=x, dtype=x.dtype)


# default values for splitting a signal into overlapping frames
FRAME_SIZE = 2048
HOP_SIZE = 441.
FPS = 100.
ORIGIN = 0
START = 0
NUM_FRAMES = 'extend'


class FramedSignal(object):
    """
    FramedSignal splits a signal into frames and makes them iterable.

    """
    def __init__(self, signal, frame_size=FRAME_SIZE, hop_size=HOP_SIZE,
                 fps=None, origin=ORIGIN, start=START, num_frames=NUM_FRAMES,
                 *args, **kwargs):
        """
        Creates a new FramedSignal instance.

        :param signal:     a Signal or FramedSignal instance
                           or anything a Signal can be instantiated from
        :param frame_size: size of one frame [int]
        :param hop_size:   progress N samples between adjacent frames [float]
        :param fps:        use given frames per second (instead of using
                           `hop_size`; if set, this overwrites the `hop_size`
                           value) [float]
        :param origin:     location of the window relative to the signal
                           position [int]
        :param start:      start sample [int]
        :param num_frames: number of frames to return (see below)

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

        If only a certain part of the Signal is wanted, `start` (in samples)
        and `num_frames` (in frames) can be used to set the range accordingly.
        Additionally, the `num_frames` parameter can have the following literal
        values to cover the whole signal to the end with the following end of
        signal behavior.
        - 'normal': the origin of the last frame has to be within the signal
        - 'extend': frames are returned as long as part of the frame overlaps
          with the signal [default]

        """
        # signal handling
        if isinstance(signal, Signal):
            # already a signal
            self._signal = signal
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
            # overwrite the hop_size
            self._hop_size = self._signal.sample_rate / float(fps)

        # translate literal values
        if origin in ('center', 'offline'):
            # the current position is the center of the frame
            self._origin = 0
        elif origin in ('left', 'past', 'online'):
            # the current position is the right edge of the frame
            # this is usually used when simulating online mode, where only past
            # information of the audio signal can be used
            self._origin = +(frame_size - 1) / 2
        elif origin in ('right', 'future'):
            # the current position is the left edge of the frame
            self._origin = -(frame_size / 2)
            # location of the window
        else:
            self._origin = origin

        # start position of the signal (in samples)
        self._start = int(start)

        # number of frames handling
        if num_frames == 'extend':
            # return frames as long as a frame covers any signal
            self._num_frames = int(np.floor(len(self.signal) /
                                            float(self.hop_size) + 1))
        elif num_frames == 'normal':
            # return frames as long as the origin sample covers the signal
            self._num_frames = int(np.ceil(len(self.signal) /
                                           float(self.hop_size)))
        else:
            self._num_frames = int(num_frames)

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
            if index < self.num_frames:
                # return the frame at this index
                # subtract the origin from the start position and use as offset
                return signal_frame(self.signal.data, index,
                                    frame_size=self.frame_size,
                                    hop_size=self.hop_size,
                                    offset=(self.start - self.origin))
            # otherwise raise an error to indicate the end of signal
            raise IndexError("end of signal reached")
        # a slice is given
        elif isinstance(index, slice):
            # determine the frames to return
            start, stop, step = index.indices(self.num_frames)
            # allow only normal steps
            if step != 1:
                raise ValueError('only slices with a step size of 1 supported')
            # determine the number of frames
            num_frames = stop - start
            # determine the start sample
            start_sample = self.start + self.hop_size * start
            # return a new FramedSignal instance covering the requested frames
            return FramedSignal(self.signal, frame_size=self.frame_size,
                                hop_size=self.hop_size, origin=self.origin,
                                start=start_sample, num_frames=num_frames)
        # other index types are invalid
        else:
            raise TypeError("frame indices must be slices or integers")

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
    def start(self):
        """Origin sample of the first frame."""
        return self._start

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
        return 1.0 - self.hop_size / self.frame_size

    def copy(self, frame_size=None, hop_size=None, fps=None, origin=None,
             start=None, num_frames=None):
        """
        Copies the FramedSignal object and adjusts some parameters.

        :param frame_size: size of one frame [int]
        :param hop_size:   progress N samples between adjacent frames [float]
        :param fps:        use given frames per second (instead of using
                           `hop_size`; if set, this overwrites the `hop_size`
                           value) [float]
        :param origin:     location of the window relative to the signal
                           position [int]
        :param start:      start sample [int]
        :param num_frames: number of frames to return
        :return:           FramedSignal object with given parameters

        """
        # copy the object attributes unless overwritten by passing other values
        if frame_size is None:
            frame_size = self.frame_size
        if hop_size is None:
            hop_size = self.hop_size
        if fps is None:
            fps = self.fps
        if origin is None:
            origin = self.origin
        if start is None:
            start = self.start
        if num_frames is None:
            num_frames = self.num_frames
        # return a new FramedSignal
        return FramedSignal(self.signal, frame_size=frame_size,
                            hop_size=hop_size, fps=fps, origin=origin,
                            start=start, num_frames=num_frames)

    def __str__(self):
        return "FramedSignal: %d frame(s); %d frame size; %.1f hop size\n %s"\
               % (self.num_frames, self.frame_size, self.hop_size,
                  str(self.signal))

    @staticmethod
    def add_arguments(parser, online=None, fps=FPS, frame_size=FRAME_SIZE):
        """
        Add audio frames related arguments to an existing parser object.

        :param parser:     existing argparse parser object
        :param online:     online mode
        :param fps:        frames per second
        :param frame_size: size of one frame in samples
        :return:           audio frames argument parser group object

        """
        # add wav options to the existing parser
        g = parser.add_argument_group('audio frame arguments')
        if online is not None:
            g.add_argument('--online', dest='online', action='store_true',
                           default=online,
                           help='operate in online mode [default=%(default)s]')
        if fps is not None:
            g.add_argument('--fps', action='store', type=int, default=fps,
                           help='frames per second [default=%(default)i]')
        if frame_size is not None:
            g.add_argument('--frame_size', action='store', type=int,
                           default=frame_size,
                           help='frame size [samples, default=%(default)i]')
        # return the argument group so it can be modified if needed
        return g
