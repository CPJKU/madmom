#!/usr/bin/env python
# encoding: utf-8
"""
This file contains basic signal processing functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import numpy as np

from madmom.processors import Processor


# signal functions
def smooth(signal, kernel):
    """
    Smooth the signal along the first axis.

    :param signal: signal [numpy array]
    :param kernel: smoothing kernel [numpy array or int]
    :return:       smoothed signal

    Note: If `kernel` is an integer, a Hamming window of that length will be
          used as a smoothing kernel.

    """
    # check if a kernel is given
    if kernel is None:
        return signal
    # size for the smoothing kernel is given
    elif isinstance(kernel, int):
        if kernel == 0:
            return signal
        elif kernel > 1:
            # use a Hamming window of given length
            kernel = np.hamming(kernel)
        else:
            raise ValueError("can't create a smoothing window of size %d" %
                             kernel)
    # otherwise use the given smoothing kernel directly
    elif isinstance(kernel, np.ndarray):
        if len(kernel) > 1:
            kernel = kernel
    else:
        raise ValueError("can't smooth signal with %s" % kernel)
    # convolve with the kernel and return
    if signal.ndim == 1:
        return np.convolve(signal, kernel, 'same')
    elif signal.ndim == 2:
        from scipy.signal import convolve2d
        return convolve2d(signal, kernel[:, np.newaxis], 'same')
    else:
        raise ValueError('signal must be either 1D or 2D')


def attenuate(signal, attenuation):
    """"
    Attenuate the signal.

    :param signal:      signal [numpy array]
    :param attenuation: attenuation level [dB, float]
    :return:            attenuated signal

    Note: The signal is returned with the same type, thus in case of integer
          dtypes, rounding errors may occur.

    """
    # return the signal unaltered if no attenuation is given
    if attenuation == 0:
        return signal
    # FIXME: attenuating the signal and keeping the original dtype makes the
    #        following signal processing steps well-behaved, since these rely
    #        on the dtype of the array to determine the correct value range.
    #        This introduces rounding (truncating) errors in case of signals
    #        with integer dtypes. But these errors should be negligible.
    # Note: np.asanyarray returns the signal's ndarray subclass
    return np.asanyarray(signal / np.power(np.sqrt(10.), attenuation / 10.),
                         dtype=signal.dtype)


def normalize(signal):
    """
    Normalize the signal to the range -1..+1

    :param signal: signal [numpy array]
    :return:       normalized signal

    Note: The signal is always returned with np.float dtype.

    """
    # Note: np.asanyarray returns the signal's ndarray subclass
    return np.asanyarray(signal.astype(np.float) / np.max(signal))


def remix(signal, num_channels):
    """
    Remix the signal to have the desired number of channels.

    :param signal:       signal [numpy array]
    :param num_channels: desired number of channels
    :return:             remixed signal (with same dtype)

    Note: This function does not support arbitrary channel number conversions.
          Only down-mixing to and up-mixing from mono signals is supported.
          The signal is returned with the same dtype, thus in case of
          down-mixing signals with integer dtypes, rounding errors may occur.

    """
    # convert to the desired number of channels
    if num_channels == signal.ndim or num_channels is None:
        # return as many channels as there are
        return signal
    elif num_channels == 1 and signal.ndim > 1:
        # down-mix to mono (keep the original dtype)
        # TODO: add weighted mixing
        return np.mean(signal, axis=-1, dtype=signal.dtype)
    elif num_channels > 1 and signal.ndim == 1:
        # up-mix a mono signal simply by copying channels
        return np.tile(signal[:, np.newaxis], num_channels)
    else:
        # any other channel conversion is not supported
        raise NotImplementedError("Requested %d channels, but got %d channels "
                                  "and channel conversion is not implemented."
                                  % (num_channels, signal.shape[1]))


def trim(signal):
    """
    Trim leading and trailing zeros of the signal.

    :param signal: signal [numpy array]
    :return:       trimmed signal

    """
    # signal must be mono
    if signal.ndim > 1:
        # FIXME: please implement stereo (or multi-channel) handling
        #        maybe it works, haven't checked
        raise NotImplementedError("please implement multi-dim functionality")
    return np.trim_zeros(signal)


def root_mean_square(signal):
    """
    Computes the root mean square of the signal. This can be used as a
    measurement of power.

    :param signal: signal [numpy array]
    :return:       root mean square of the signal

    """
    # make sure the signal is a numpy array
    if not isinstance(signal, np.ndarray):
        raise TypeError("Invalid type for signal, must be a numpy array.")
    # signal must be mono
    if signal.ndim > 1:
        # FIXME: please implement stereo (or multi-channel) handling
        raise NotImplementedError("please implement multi-dim functionality")
    # Note: type conversion needed because of integer overflows
    if signal.dtype != np.float:
        signal = signal.astype(np.float)
    # return
    return np.sqrt(np.dot(signal, signal) / signal.size)


def sound_pressure_level(signal, p_ref=1.0):
    """
    Computes the sound pressure level of a signal.

    :param signal: signal [numpy array]
    :param p_ref:  reference sound pressure level
    :return:       sound pressure level of the signal

    From http://en.wikipedia.org/wiki/Sound_pressure:
    Sound pressure level (SPL) or sound level is a logarithmic measure of the
    effective sound pressure of a sound relative to a reference value. It is
    measured in decibels (dB) above a standard reference level.

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


def load_wave_file(filename, sample_rate=None, num_channels=None, start=None,
                   stop=None):
    """
    Load the audio data from the given file and return it as a numpy array.
    Only supports wave files, does not support re-sampling or arbitrary
    channel number conversions. Reads the data as a memory-mapped file with
    copy-on-write semantics to defer I/O costs until needed.

    :param filename:     name of the file
    :param sample_rate:  desired sample rate of the signal in Hz [int], or
                         `None` to return the signal in its original rate
    :param num_channels: reduce or expand the signal to N channels [int], or
                         `None` to return the signal with its original channels
    :param start:        start position (seconds) [float]
    :param stop:         stop position (seconds) [float]
    :return:             tuple (signal, sample_rate)

    """
    from scipy.io import wavfile
    file_sample_rate, signal = wavfile.read(filename, mmap=True)
    # if the sample rate is not the desired one, raise exception
    if sample_rate is not None and sample_rate != file_sample_rate:
        raise NotImplementedError("Requested sample rate of %f Hz, but got %f "
                                  "Hz and resampling is not implemented." %
                                  (sample_rate, file_sample_rate))
    # only request the desired part of the signal
    if start is not None:
        start = int(start * file_sample_rate)
    if stop is not None:
        stop = min(len(signal), int(stop * file_sample_rate))
    if start is not None or stop is not None:
        signal = signal[start: stop]
    # up-/down-mix if needed
    if num_channels is not None:
        signal = remix(signal, num_channels)
    # return the signal
    return signal, file_sample_rate


def load_ffmpeg_file(filename, sample_rate=None, num_channels=None, start=None,
                     stop=None, dtype=np.int16, cmd_decode='ffmpeg',
                     cmd_probe='ffprobe'):
    """
    Load the audio data from the given file and return it as a numpy array.
    This uses ffmpeg (or avconv) and thus supports a lot of different file
    formats, resampling and channel conversions. The file will be fully decoded
    into memory if no start and stop positions are given.

    :param filename:     name of the file
    :param sample_rate:  desired sample rate of the signal in Hz [int], or
                         `None` to return the signal in its original rate
    :param num_channels: reduce or expand the signal to N channels [int], or
                         `None` to return the signal with its original channels
    :param start:        start position (seconds) [float]
    :param stop:         stop position (seconds) [float]
    :param dtype:        numpy dtype to return the signal in (supports signed
                         and unsigned 8/16/32-bit integers, and single and
                         double precision floats, each in little or big endian)
    :return:             tuple (signal, sample_rate)
    """
    from .ffmpeg import decode_to_memory, get_file_info
    # convert dtype to sample type
    # (all ffmpeg PCM sample types: ffmpeg -formats | grep PCM)
    dtype = np.dtype(dtype)
    # - unsigned int, signed int, floating point:
    sample_type = {'u': 'u', 'i': 's', 'f': 'f'}.get(dtype.kind)
    # - sample size in bits:
    sample_type += str(8 * dtype.itemsize)
    # - little endian or big endian:
    if dtype.byteorder == '=':
        import sys
        sample_type += sys.byteorder[0] + 'e'
    else:
        sample_type += {'|': '', '<': 'le', '>': 'be'}.get(dtype.byteorder)
    # start and stop position
    if start is None:
        start = 0
    max_len = None
    if stop is not None:
        max_len = stop - start
    # convert the audio signal using ffmpeg
    signal = np.frombuffer(decode_to_memory(filename, fmt=sample_type,
                                            sample_rate=sample_rate,
                                            num_channels=num_channels,
                                            skip=start, max_len=max_len,
                                            cmd=cmd_decode),
                           dtype=dtype)
    # get the needed information from the file
    if sample_rate is None or num_channels is None:
        info = get_file_info(filename, cmd=cmd_probe)
        if sample_rate is None:
            sample_rate = info['sample_rate']
        if num_channels is None:
            num_channels = info['num_channels']
    # reshape the audio signal
    if num_channels > 1:
        signal = signal.reshape((-1, num_channels))
    return signal, sample_rate


# function for automatically determining how to open audio files
def load_audio_file(filename, sample_rate=None, num_channels=None, start=None,
                    stop=None):
    """
    Load the audio data from the given file and return it as a numpy array.
    This tries load_wave_file and load_ffmpeg_file (for ffmpeg and avconv).

    :param filename:     name of the file or file handle
    :param sample_rate:  desired sample rate of the signal in Hz [int], or
                         `None` to return the signal in its original rate
    :param num_channels: reduce or expand the signal to N channels [int], or
                         `None` to return the signal with its original channels
    :param start:        start position (seconds) [float]
    :param stop:         stop position (seconds) [float]
    :return:             tuple (signal, sample_rate)

    """
    # determine the name of the file if it is a file handle
    if isinstance(filename, file):
        # open file handle
        filename = filename.name
    # try reading as a wave file, ffmpeg or avconv (in this order)
    try:
        return load_wave_file(filename, sample_rate=sample_rate,
                              num_channels=num_channels, start=start,
                              stop=stop)
    except Exception:
        pass
    try:
        return load_ffmpeg_file(filename, sample_rate=sample_rate,
                                num_channels=num_channels, start=start,
                                stop=stop)
    except Exception:
        pass
    try:
        return load_ffmpeg_file(filename, sample_rate=sample_rate,
                                num_channels=num_channels, start=start,
                                stop=stop, cmd_decode='avconv',
                                cmd_probe='avprobe')
    except Exception:
        pass
    raise RuntimeError("All attempts to load audio file %r failed." % filename)


# signal classes
class Signal(np.ndarray):
    """
    Signal extends a numpy ndarray with a 'sample_rate' and some other useful
    attributes.

    """

    def __new__(cls, data, sample_rate=None, num_channels=None, start=None,
                stop=None):
        """
        Creates a new Signal instance.

        :param data:         numpy array or audio file name or file handle
        :param sample_rate:  sample rate of the signal [int]
        :param num_channels: number of channels [int]
        :param start:        start position (seconds) [float]
        :param stop:         stop position (seconds) [float]

        Note: `sample_rate` or `num_channels` can be used to set the desired
              sample rate and number of channels if the audio is read from
              file. If set to 'None' the audio signal is used as is, i.e. the
              sample rate and number of channels are determined directly from
              the audio file.
              If the `data` is a numpy array, the `sample_rate` is set to the
              given value and `num_channels` is set to the number of columns
              of the array.

        """
        # try to load an audio file if the data is not a numpy array
        if not isinstance(data, np.ndarray):
            data, sample_rate = load_audio_file(data, sample_rate=sample_rate,
                                                num_channels=num_channels,
                                                start=start, stop=stop)
        # cast as Signal
        obj = np.asarray(data).view(cls)
        if sample_rate is not None:
            sample_rate = float(sample_rate)
        obj.sample_rate = sample_rate
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here, also needed for views of the Signal
        self.sample_rate = getattr(obj, 'sample_rate', None)

    def __reduce__(self):
        # needed for correct pickling
        # source: http://stackoverflow.com/questions/26598109/
        # get the parent's __reduce__ tuple
        pickled_state = super(Signal, self).__reduce__()
        # create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.sample_rate,)
        # return a tuple that replaces the parent's __reduce__ tuple
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):
        # needed for correct un-pickling
        # set the sample_rate
        self.sample_rate = state[-1]
        # call the parent's __setstate__ with the other tuple elements
        super(Signal, self).__setstate__(state[0:-1])

    @property
    def num_samples(self):
        """Number of samples."""
        return len(self)

    @property
    def num_channels(self):
        """Number of channels."""
        try:
            # multi channel files
            return np.shape(self)[1]
        except IndexError:
            # catch mono files
            return 1

    @property
    def length(self):
        """Length of signal in seconds."""
        # n/a if the signal has no sample rate
        if self.sample_rate is None:
            return None
        return float(self.num_samples) / self.sample_rate


class SignalProcessor(Processor):
    """
    SignalProcessor is a basic signal processor.

    """
    SAMPLE_RATE = None
    NUM_CHANNELS = None
    START = None
    STOP = None
    NORM = False
    ATT = 0.

    def __init__(self, sample_rate=SAMPLE_RATE, num_channels=NUM_CHANNELS,
                 start=START, stop=STOP, norm=NORM, att=ATT, **kwargs):
        """
        Creates a new SignalProcessor instance.

        :param sample_rate:  sample rate of the signal [Hz]
        :param num_channels: reduce the signal to N channels [int]
        :param start:        start position (seconds) [float]
        :param stop:         stop position (seconds) [float]
        :param norm:         normalize the signal [bool]
        :param att:          attenuate the signal [dB]

        Note: If `sample_rate` is set, the signal will be re-sampled to that
              sample rate; if 'None' the sample rate of the audio file will be
              used.
              If `num_channels` is set, the signal will be reduced to that
              number of channels; if 'None' as many channels as present in the
              audio file are returned.

        """
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.norm = norm
        self.att = att

    def process(self, data, **kwargs):
        """
        Processes the given audio file.

        :param data:   file name or handle
        :param kwargs: keyword arguments passed to Signal
        :return:       Signal instance with processed signal

        """
        # instantiate a Signal (with the given sample rate if set)
        data = Signal(data, self.sample_rate, self.num_channels, **kwargs)
        # process it if needed
        if self.norm:
            # normalize signal
            data = normalize(data)
        if self.att is not None and self.att != 0:
            # attenuate signal
            data = attenuate(data, self.att)
        # return processed data
        return data

    @classmethod
    def add_arguments(cls, parser, sample_rate=None, mono=None, start=None,
                      stop=None, norm=None, att=None):
        """
        Add signal processing related arguments to an existing parser.

        :param parser:      existing argparse parser object
        :param sample_rate: re-sample the signal to this sample rate [Hz]
        :param mono:        down-mix the signal to mono [bool]
        :param start:       start position (seconds) [float]
        :param stop:        stop position (seconds) [float]
        :param norm:        normalize the signal [bool]
        :param att:         attenuate the signal [dB, float]
        :return:            signal processing argument parser group

        Parameters are included in the group only if they are not 'None'.

        """
        # add signal processing options to the existing parser
        g = parser.add_argument_group('signal processing arguments')
        if sample_rate is not None:
            g.add_argument('--sample_rate', action='store_true',
                           default=sample_rate,
                           help='re-sample the signal to this sample rate '
                                '[Hz]')
        if mono is not None:
            g.add_argument('--mono', dest='num_channels', action='store_const',
                           const=1,
                           help='down-mix the signal to mono')
        if start is not None:
            g.add_argument('--start', action='store', type=float,
                           help='start position of the signal [seconds]')
        if stop is not None:
            g.add_argument('--stop', action='store', type=float,
                           help='stop position of the signal [seconds]')
        if norm is not None:
            g.add_argument('--norm', action='store_true', default=norm,
                           help='normalize the signal [default=%(default)s]')
        if att is not None:
            g.add_argument('--att', action='store', type=float, default=att,
                           help='attenuate the signal '
                                '[dB, default=%(default).1f]')
        # return the argument group so it can be modified if needed
        return g


# functions for splitting a signal into frames
def signal_frame(signal, index, frame_size, hop_size, origin=0):
    """
    This function returns frame[index] of the signal.

    :param signal:     signal [numpy array]
    :param index:      index of the frame to return [int]
    :param frame_size: size of each frame in samples [int]
    :param hop_size:   hop size in samples between adjacent frames [float]
    :param origin:     location of the window center relative to the signal
                       position [int]
    :return:           the requested frame of the signal

    The reference sample of the first frame (index == 0) refers to the first
    sample of the signal, and each following frame is placed `hop_size` samples
    after the previous one.

    The window is always centered around this reference sample. Its location
    relative to the reference sample can be set with the `origin` parameter.
    Arbitrary integer values can be given:
      - zero centers the window on its reference sample
      - negative values shift the window to the right
      - positive values shift the window to the left
    An `origin` of half the size of the `frame_size` results in windows located
    to the left of the reference sample, i.e. the first frame starts at the
    first sample of the signal.

    The part of the frame which is not covered by the signal is padded with 0s.

    """
    # length of the signal
    num_samples = len(signal)
    # seek to the correct position in the audio signal
    ref_sample = int(index * hop_size)
    # position the window
    start = ref_sample - frame_size // 2 - int(origin)
    stop = start + frame_size
    # return the requested portion of the signal
    # Note: usually np.zeros_like(signal[:frame_size]) is exactly what we want
    #       (i.e. zeros of frame_size length and the same type/class as the
    #       signal and not just the dtype), but since we have no guarantee that
    #       the signal is that long, we have to use the np.repeat workaround
    if (stop < 0) or (start > num_samples):
        # window falls completely outside the actual signal, return just zeros
        frame = np.repeat(signal[:1] * 0, frame_size, axis=0)
        return frame
    elif (start < 0) and (stop > num_samples):
        # window surrounds the actual signal, position signal accordingly
        frame = np.repeat(signal[:1] * 0, frame_size, axis=0)
        frame[-start:num_samples - start] = signal
        return frame
    elif start < 0:
        # window crosses left edge of actual signal, pad zeros from left
        frame = np.repeat(signal[:1] * 0, frame_size, axis=0)
        frame[-start:] = signal[:stop]
        return frame
    elif stop > num_samples:
        # window crosses right edge of actual signal, pad zeros from right
        frame = np.repeat(signal[:1] * 0, frame_size, axis=0)
        frame[:num_samples - start] = signal[start:]
        return frame
    else:
        # normal read operation
        return signal[start:stop]


def framed_signal_generator(signal, frame_size, hop_size, origin=0,
                            end='extend', num_frames=None, batch_size=1):
    """

    :param signal:     signal [Signal or numpy array]
    :param frame_size: size of each frame in samples [int]
    :param hop_size:   hop size in samples between adjacent frames [float]
    :param origin:     location of the window center relative to the signal
                       position [int]
    :param end:        end of signal behaviour (see `FramedSignal`)
    :param num_frames: yield this number of frames [int]
    :param batch_size: yield batches of this size [int]
    :return:           generator which yields the signal chopped into frames

    """
    # TODO: implement batch processing and set to a sensible default
    if batch_size != 1:
        raise ValueError('please implement batch processing')
    # translate literal window location values to numeric origin
    if origin in ('center', 'offline'):
        # the current position is the center of the frame
        origin = 0
    elif origin in ('left', 'past', 'online'):
        # the current position is the right edge of the frame
        # this is usually used when simulating online mode, where only past
        # information of the audio signal can be used
        origin = (frame_size - 1) / 2
    elif origin in ('right', 'future'):
        # the current position is the left edge of the frame
        origin = -(frame_size / 2)
    origin = int(origin)
    # number of frames determination
    if num_frames is None:
        if end == 'extend':
            # return frames as long as a frame covers any signal
            num_frames = np.floor(len(signal) / float(hop_size) + 1)
        elif end == 'normal':
            # return frames as long as the origin sample covers the signal
            num_frames = np.ceil(len(signal) / float(hop_size))
        else:
            raise ValueError("end of signal handling '%s' unknown" % end)
    # yield frames as long as there is a signal
    index = 0
    while index < num_frames:
         yield signal_frame(signal, index, frame_size, hop_size, origin)
         index += batch_size


# taken from: http://www.scipy.org/Cookbook/SegmentAxis
def segment_axis(signal, frame_size, hop_size=1, axis=None, end='cut',
                 end_value=0):
    """
    Generate a new array that chops the given array along the given axis into
    (overlapping) frames.

    :param signal:     signal [numpy array]
    :param frame_size: size of each frame in samples [int]
    :param hop_size:   hop size in samples between adjacent frames [int]
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

    The returned array is always of type np.ndarray.

    Example:
    >>> segment_axis(np.arange(10), 4, 2)
    array([[0, 1, 2, 3],
           [2, 3, 4, 5],
           [4, 5, 6, 7],
           [6, 7, 8, 9]])

    """
    # make sure that both frame_size and hop_size are integers
    frame_size = int(frame_size)
    hop_size = int(hop_size)
    # TODO: add comments!
    if axis is None:
        signal = np.ravel(signal)  # may copy
        axis = 0
    if axis != 0:
        raise ValueError('please check if the resulting array is correct.')

    length = signal.shape[axis]

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
        signal = signal.swapaxes(-1, axis)

        if end == 'cut':
            signal = signal[..., :round_down]
        elif end in ['pad', 'wrap']:
            # need to copy
            s = list(signal.shape)
            s[-1] = round_up
            y = np.empty(s, dtype=signal.dtype)
            y[..., :length] = signal
            if end == 'pad':
                y[..., length:] = end_value
            elif end == 'wrap':
                y[..., length:] = signal[..., :round_up - length]
            signal = y

        signal = signal.swapaxes(-1, axis)

    length = signal.shape[axis]
    if length == 0:
        raise ValueError("Not enough data points to segment array in 'cut' "
                         "mode; try end='pad' or end='wrap'")
    assert length >= frame_size
    assert (length - frame_size) % hop_size == 0
    n = 1 + (length - frame_size) // hop_size
    s = signal.strides[axis]
    new_shape = (signal.shape[:axis] + (n, frame_size) +
                 signal.shape[axis + 1:])
    new_strides = (signal.strides[:axis] + (hop_size * s, s) +
                   signal.strides[axis + 1:])

    try:
        return np.ndarray.__new__(np.ndarray, strides=new_strides,
                                  shape=new_shape, buffer=signal,
                                  dtype=signal.dtype)
    except TypeError:
        # TODO: remove warning?
        import warnings
        warnings.warn("Problem with ndarray creation forces copy.")
        signal = signal.copy()
        # Shape doesn't change but strides does
        new_strides = (signal.strides[:axis] + (hop_size * s, s) +
                       signal.strides[axis + 1:])
        return np.ndarray.__new__(np.ndarray, strides=new_strides,
                                  shape=new_shape, buffer=signal,
                                  dtype=signal.dtype)


# classes for splitting a signal into frames
class FramedSignal(object):
    """
    FramedSignal splits a Signal into frames and makes it iterable and
    indexable.

    """

    def __init__(self, signal, frame_size=2048, hop_size=441., fps=None,
                 origin=0, end='extend', num_frames=None, **kwargs):
        """
        Creates a new FramedSignal instance from the given Signal.

        :param signal:     Signal instance (or anything a Signal can be
                           instantiated from)
        :param frame_size: size of one frame [int]
        :param hop_size:   progress N samples between adjacent frames [float]
        :param fps:        use given frames per second (if set, this
                           overwrites the given `hop_size` value) [float]
        :param origin:     location of the window relative to the reference
                           sample of a frame [int]
        :param end:        end of signal handling (see below)
        :param num_frames: number of frames to return [int]

        If no Signal instance was given, one is instantiated and these
        arguments are passed:

        :param kwargs:     additional keyword arguments passed to Signal()

        The FramedSignal class is implemented as an iterator. It splits the
        given Signal automatically into frames of `frame_size` length with
        `hop_size` samples (can be float, normal rounding applies) between the
        frames. The reference sample of the first frame refers to the first
        sample of the signal

        The location of the window relative to the reference sample of a frame
        can be set with the `origin` parameter. Arbitrary integer values can
        be given:
          - zero centers the window on its reference sample
          - negative values shift the window to the right
          - positive values shift the window to the left

        Additionally, it can have the following literal values:
          - 'center', 'offline':      the window is centered on its reference
                                      sample
          - 'left', 'past', 'online': the window is located to the left of its
                                      reference sample (including the reference
                                      sample)
          - 'right', 'future':        the window is located to the right of its
                                      reference sample

        The `end` parameter is used to handle the end of signal behaviour and
        can have these values:
          - 'normal': stop as soon as the whole signal got covered by at least
                      one frame, i.e. pad maximally one frame
          - 'extend': frames are returned as long as part of the frame overlaps
                      with the signal to cover the whole signal

        Alternatively, `num_frames` can be used to retrieve a fixed number of
        frames.

        Note: We do not use the `frame_size` for the calculation of the number
              of frames in order to be able to stack multiple frames obtained
              with different frame sizes. Thus it is not guaranteed that every
              sample of the signal is returned in a frame unless the `origin`
              is either 'right' or 'future'.

        """
        # signal handling
        if not isinstance(signal, Signal):
            # try to instantiate a Signal
            signal = Signal(signal, **kwargs)
        # save the signal
        self.signal = signal

        # arguments for splitting the signal into frames
        if frame_size:
            self.frame_size = int(frame_size)
        if hop_size:
            self.hop_size = float(hop_size)
        # use fps instead of hop_size
        if fps:
            # overwrite the hop_size
            self.hop_size = self.signal.sample_rate / float(fps)

        # translate literal window location values to numeric origin
        if origin in ('center', 'offline'):
            # the current position is the center of the frame
            origin = 0
        elif origin in ('left', 'past', 'online'):
            # the current position is the right edge of the frame
            # this is usually used when simulating online mode, where only past
            # information of the audio signal can be used
            origin = (frame_size - 1) / 2
        elif origin in ('right', 'future'):
            # the current position is the left edge of the frame
            origin = -(frame_size / 2)
        self.origin = int(origin)

        # number of frames determination
        if num_frames is None:
            if end == 'extend':
                # return frames as long as a frame covers any signal
                num_frames = np.floor(len(self.signal) /
                                      float(self.hop_size) + 1)
            elif end == 'normal':
                # return frames as long as the origin sample covers the signal
                num_frames = np.ceil(len(self.signal) / float(self.hop_size))
            else:
                raise ValueError("end of signal handling '%s' unknown" %
                                 end)
        self.num_frames = int(num_frames)

    # make the object indexable / iterable
    def __getitem__(self, index):
        """
        This makes the FramedSignal class indexable and/or iterable.

        The signal is split into frames (of length `frame_size`) automatically.
        Two frames are located `hop_size` samples apart. If `hop_size` is a
        float, normal rounding applies.

        Note: Index -1 refers NOT to the last frame, but to the frame directly
              left of frame 0. Although this is contrary to common behavior,
              being able to access these frames can be important, e.g. if the
              frames overlap, frame -1 contains parts of the signal of frame 0.

        """
        # a single index is given
        if isinstance(index, int):
            # return a single frame
            if index < self.num_frames:
                # return the frame at this index
                return signal_frame(self.signal, index,
                                    frame_size=self.frame_size,
                                    hop_size=self.hop_size, origin=self.origin)
            # otherwise raise an error to indicate the end of signal
            raise IndexError("end of signal reached")
        # a slice is given
        elif isinstance(index, slice):
            # determine the frames to return (limited to the number of frames)
            start, stop, step = index.indices(self.num_frames)
            # allow only normal steps
            if step != 1:
                raise ValueError('only slices with a step size of 1 supported')
            # determine the number of frames
            num_frames = stop - start
            # determine the new origin, i.e. start position
            origin = self.origin + self.hop_size * start
            # return a new FramedSignal instance covering the requested frames
            return FramedSignal(self.signal, frame_size=self.frame_size,
                                hop_size=self.hop_size, origin=origin,
                                num_frames=num_frames)
        # other index types are invalid
        else:
            raise TypeError("frame indices must be slices or integers")

    # len() returns the number of frames, consistent with __getitem__()
    def __len__(self):
        return self.num_frames

    @property
    def frame_rate(self):
        """Frame rate."""
        # n/a if the signal has no sample rate
        if self.signal.sample_rate is None:
            return None
        return float(self.signal.sample_rate) / self.hop_size

    @property
    def fps(self):
        """Frames per second."""
        return self.frame_rate

    @property
    def overlap_factor(self):
        """Overlapping factor of two adjacent frames."""
        return 1.0 - self.hop_size / self.frame_size

    @property
    def shape(self):
        """Shape of the FramedSignal (frames x samples)."""
        return self.num_frames, self.frame_size


class FramedSignalProcessor(Processor):
    """
    Slice a Signal into frames.

    """
    FRAME_SIZE = 2048
    HOP_SIZE = 441.
    FPS = 100.
    ONLINE = False
    START = 0
    END_OF_SIGNAL = 'extend'

    def __init__(self, frame_size=FRAME_SIZE, hop_size=HOP_SIZE, fps=None,
                 online=ONLINE, end=END_OF_SIGNAL, **kwargs):
        """
        Creates a new FramedSignalProcessor instance.

        :param frame_size: size of one frame in samples [int]
        :param hop_size:   progress N samples between adjacent frames [float]
        :param fps:        use frames per second (compute the needed `hop_size`
                           instead of using the given `hop_size` value) [float]
        :param online:     operate in online mode (see below) [bool]
        :param end:        end of signal handling (see below)

        The location of the window relative to its reference sample can be set
        with the `online` parameter:
          - 'True':  the window is located to the left of its reference sample
                     (including the reference sample), i.e. only past
                     information is used
          - 'False': the window is centered on its reference sample [default]

        The end of the signal handling can be set with the `end` parameter,
        it accepts the following literal values:
          - 'normal': the origin of the last frame has to be within the signal
          - 'extend': frames are returned as long as part of the frame overlaps
                      with the signal [default]

        """
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.fps = fps  # do not convert here, pass it to FramedSignal
        self.online = online
        self.end = end

    def process(self, data, **kwargs):
        """
        Slice the signal into (overlapping) frames.

        :param data:   signal to be sliced into frames [Signal]
        :param kwargs: keyword arguments passed to FramedSignal
        :return:       FramedSignal instance

        """
        # translate online / offline mode
        if self.online:
            origin = 'online'
        else:
            origin = 'offline'
        # instantiate a FramedSignal from the data and return it
        return FramedSignal(data, frame_size=self.frame_size,
                            hop_size=self.hop_size, fps=self.fps,
                            origin=origin, end=self.end, **kwargs)

    @classmethod
    def add_arguments(cls, parser, frame_size=FRAME_SIZE, fps=FPS,
                      online=ONLINE):
        """
        Add signal framing related arguments to an existing parser.

        :param parser:     existing argparse parser object
        :param frame_size: size of one frame in samples [int]
        :param fps:        frames per second [float]
        :param online:     online mode [bool]
        :return:           signal framing argument parser group

        Parameters are included in the group only if they are not 'None'.

        """
        # add signal framing options to the existing parser
        g = parser.add_argument_group('signal framing arguments')
        if frame_size is not None:
            # depending on the type, use different options
            if isinstance(frame_size, int):
                g.add_argument('--frame_size', action='store', type=int,
                               default=frame_size,
                               help='frame size [samples, '
                                    'default=%(default)i]')
            elif isinstance(frame_size, list):
                from madmom.utils import OverrideDefaultListAction
                g.add_argument('--frame_size', type=int, default=frame_size,
                               action=OverrideDefaultListAction,
                               help='frame size(s) to use, multiple values '
                                    'be given, one per argument. [samples, '
                                    'default=%(default)s]')
        if fps is not None:
            g.add_argument('--fps', action='store', type=float, default=fps,
                           help='frames per second [default=%(default).1f]')
        if online is not None:
            g.add_argument('--online', dest='online', action='store_true',
                           default=online,
                           help='operate in online mode [default=%(default)s]')

        # TODO: include end_of_signal handling!?
        # return the argument group so it can be modified if needed
        return g
