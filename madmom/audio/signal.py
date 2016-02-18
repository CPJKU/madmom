# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains basic signal processing functionality.

"""

from __future__ import absolute_import, division, print_function

import numpy as np

from madmom.processors import Processor


# signal functions
def smooth(signal, kernel):
    """
    Smooth the signal along its first axis.

    Parameters
    ----------
    signal : numpy array
        Signal to be smoothed.
    kernel : numpy array or int
        Smoothing kernel (size).

    Returns
    -------
    numpy array
        Smoothed signal.

    Notes
    -----
    If `kernel` is an integer, a Hamming window of that length will be used
    as a smoothing kernel.

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
            raise ValueError("can't create a smoothing kernel of size %d" %
                             kernel)
    # otherwise use the given smoothing kernel directly
    elif isinstance(kernel, np.ndarray) and len(kernel) > 1:
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


def adjust_gain(signal, gain):
    """"
    Adjust the gain of the signal.

    Parameters
    ----------
    signal : numpy array
        Signal to be adjusted.
    gain : float
        Gain adjustment level [dB].

    Returns
    -------
    numpy array
        Signal with adjusted gain.

    Notes
    -----
    The signal is returned with the same dtype, thus rounding errors may occur
    with integer dtypes.

    `gain` values > 0 amplify the signal and are only supported for signals
    with float dtype to prevent clipping and integer overflows.

    """
    # convert the gain in dB to a scaling factor
    gain = np.power(np.sqrt(10.), 0.1 * gain)
    # prevent overflow and clipping
    if gain > 1 and np.issubdtype(signal.dtype, np.int):
        raise ValueError('positive gain adjustments are only supported for '
                         'float dtypes.')
    # Note: np.asanyarray returns the signal's ndarray subclass
    return np.asanyarray(signal * gain, dtype=signal.dtype)


def attenuate(signal, attenuation):
    """
    Attenuate the signal.

    Parameters
    ----------
    signal : numpy array
        Signal to be attenuated.
    attenuation :  float
        Attenuation level [dB].

    Returns
    -------
    numpy array
        Attenuated signal (same dtype as `signal`).

    Notes
    -----
    The signal is returned with the same dtype, thus rounding errors may occur
    with integer dtypes.

    """
    # return the signal unaltered if no attenuation is given
    if attenuation == 0:
        return signal
    return adjust_gain(signal, -attenuation)


def normalize(signal):
    """
    Normalize the signal to have maximum amplitude.

    Parameters
    ----------
    signal : numpy array
        Signal to be normalized.

    Returns
    -------
    numpy array
        Normalized signal.

    Notes
    -----
    Signals with float dtypes cover the range [-1, +1], signals with integer
    dtypes will cover the maximally possible range, e.g. [-32768, 32767] for
    np.int16.

    The signal is returned with the same dtype, thus rounding errors may occur
    with integer dtypes.

    """
    # scaling factor to be applied
    scaling = float(np.max(np.abs(signal)))
    if np.issubdtype(signal.dtype, np.int):
        if signal.dtype in (np.int16, np.int32):
            scaling /= np.iinfo(signal.dtype).max
        else:
            raise ValueError('only float and np.int16/32 dtypes supported, '
                             'not %s.' % signal.dtype)
    # Note: np.asanyarray returns the signal's ndarray subclass
    return np.asanyarray(signal / scaling, dtype=signal.dtype)


def remix(signal, num_channels):
    """
    Remix the signal to have the desired number of channels.

    Parameters
    ----------
    signal : numpy array
        Signal to be remixed.
    num_channels : int
        Number of channels.

    Returns
    -------
    numpy array
        Remixed signal (same dtype as `signal`).

    Notes
    -----
    This function does not support arbitrary channel number conversions.
    Only down-mixing to and up-mixing from mono signals is supported.

    The signal is returned with the same dtype, thus rounding errors may occur
    with integer dtypes.

    If the signal should be down-mixed to mono and has an integer dtype, it
    will be converted to float internally and then back to the original dtype
    to prevent clipping of the signal. To avoid this double conversion,
    convert the dtype first.

    """
    # convert to the desired number of channels
    if num_channels == signal.ndim or num_channels is None:
        # return as many channels as there are.
        return signal
    elif num_channels == 1 and signal.ndim > 1:
        # down-mix to mono
        # Note: to prevent clipping, the signal is converted to float first
        #       and then converted back to the original dtype
        # TODO: add weighted mixing
        return np.mean(signal, axis=-1).astype(signal.dtype)
    elif num_channels > 1 and signal.ndim == 1:
        # up-mix a mono signal simply by copying channels
        return np.tile(signal[:, np.newaxis], num_channels)
    else:
        # any other channel conversion is not supported
        raise NotImplementedError("Requested %d channels, but got %d channels "
                                  "and channel conversion is not implemented."
                                  % (num_channels, signal.shape[1]))


def rescale(signal, dtype=np.float32):
    """
    Rescale the signal to range [-1, 1] and return as float dtype.

    Parameters
    ----------
    signal : numpy array
        Signal to be remixed.
    dtype : numpy dtype
        Data type of the signal.

    Returns
    -------
    numpy array
        Signal rescaled to range [-1, 1].

    """
    # allow only float dtypes
    if not np.issubdtype(dtype, np.float):
        raise ValueError('only float dtypes are supported, not %s.' % dtype)
    # float signals don't need rescaling
    if np.issubdtype(signal.dtype, np.float):
        return signal.astype(dtype)
    elif np.issubdtype(signal.dtype, np.int):
        return signal.astype(dtype) / np.iinfo(signal.dtype).max
    else:
        # TODO: not sure if this can happen or not. Either add the
        #       functionality if it is supposed to work or add a test
        raise ValueError('unsupported signal dtypes: %s.' % signal.dtype)


def trim(signal, where='fb'):
    """
    Trim leading and trailing zeros of the signal.

    Parameters
    ----------
    signal : numpy array
        Signal to be trimmed.
    where : str, optional
        A string with 'f' representing trim from front and 'b' to trim from
        back. Default is 'fb', trim zeros from both ends of the signal.

    Returns
    -------
    numpy array
        Trimmed signal.

    """
    # code borrowed from np.trim_zeros()
    first = 0
    where = where.upper()
    if 'F' in where:
        for i in signal:
            if np.sum(i) != 0.:
                break
            else:
                first += 1
    last = len(signal)
    if 'B' in where:
        for i in signal[::-1]:
            if np.sum(i) != 0.:
                break
            else:
                last -= 1
    return signal[first:last]


def root_mean_square(signal):
    """
    Computes the root mean square of the signal. This can be used as a
    measurement of power.

    Parameters
    ----------
    signal : numpy array
        Signal.

    Returns
    -------
    rms : float
        Root mean square of the signal.

    """
    # make sure the signal is a numpy array
    if not isinstance(signal, np.ndarray):
        raise TypeError("Invalid type for signal, must be a numpy array.")
    # Note: type conversion needed because of integer overflows
    if signal.dtype != np.float:
        signal = signal.astype(np.float)
    # return
    return np.sqrt(np.dot(signal.flatten(), signal.flatten()) / signal.size)


def sound_pressure_level(signal, p_ref=None):
    """
    Computes the sound pressure level of a signal.

    Parameters
    ----------
    signal : numpy array
        Signal.
    p_ref : float, optional
        Reference sound pressure level; if 'None', take the max amplitude
        value for the data-type, if the data-type is float, assume amplitudes
        are between -1 and +1.

    Returns
    -------
    spl : float
        Sound pressure level of the signal [dB].

    Notes
    -----
    From http://en.wikipedia.org/wiki/Sound_pressure: Sound pressure level
    (SPL) or sound level is a logarithmic measure of the effective sound
    pressure of a sound relative to a reference value. It is measured in
    decibels (dB) above a standard reference level.

    """
    # compute the RMS
    rms = root_mean_square(signal)
    # compute the SPL
    if rms == 0:
        # return the smallest possible negative number
        return -np.finfo(float).max
    else:
        if p_ref is None:
            # find a reasonable default reference value
            if np.issubdtype(signal.dtype, np.integer):
                p_ref = float(np.iinfo(signal.dtype).max)
            else:
                p_ref = 1.0
        # normal SPL computation
        return 20.0 * np.log10(rms / p_ref)


def load_wave_file(filename, sample_rate=None, num_channels=None, start=None,
                   stop=None, dtype=None):
    """
    Load the audio data from the given file and return it as a numpy array.

    Only supports wave files, does not support re-sampling or arbitrary
    channel number conversions. Reads the data as a memory-mapped file with
    copy-on-write semantics to defer I/O costs until needed.

    Parameters
    ----------
    filename : string
        Name of the file.
    sample_rate : int, optional
        Desired sample rate of the signal [Hz], or 'None' to return the
        signal in its original rate.
    num_channels : int, optional
        Reduce or expand the signal to `num_channels` channels, or 'None'
        to return the signal with its original channels.
    start : float, optional
        Start position [seconds].
    stop : float, optional
        Stop position [seconds].
    dtype : numpy data type, optional
        The data is returned with the given dtype. If 'None', it is returned
        with its original dtype, otherwise the signal gets rescaled. Integer
        dtypes use the complete value range, float dtypes the range [-1, +1].


    Returns
    -------
    signal : numpy array
        Audio signal.
    sample_rate : int
        Sample rate of the signal [Hz].

    Notes
    -----
    The `start` and `stop` positions are rounded to the closest sample; the
    sample corresponding to the `stop` value is not returned, thus consecutive
    segment starting with the previous `stop` can be concatenated to obtain
    the original signal without gaps or overlaps.

    """
    from scipy.io import wavfile
    file_sample_rate, signal = wavfile.read(filename, mmap=True)
    # if the sample rate is not the desired one, raise exception
    if sample_rate is not None and sample_rate != file_sample_rate:
        raise ValueError('Requested sample rate of %f Hz, but got %f Hz and '
                         're-sampling is not implemented.' %
                         (sample_rate, file_sample_rate))
    # same for the data type
    if dtype is not None and signal.dtype != dtype:
        raise ValueError('Requested dtype %s, but got %s and re-scaling is '
                         'not implemented.' % (dtype, signal.dtype))
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


class LoadAudioFileError(Exception):
    """
    Exception to be raised whenever an audio file could not be loaded.

    """
    # pylint: disable=super-init-not-called

    def __init__(self, value=None):
        if value is None:
            value = 'Could not load audio file.'
        self.value = value

    def __str__(self):
        return repr(self.value)


# function for automatically determining how to open audio files
def load_audio_file(filename, sample_rate=None, num_channels=None, start=None,
                    stop=None, dtype=None):
    """
    Load the audio data from the given file and return it as a numpy array.
    This tries load_wave_file() load_ffmpeg_file() (for ffmpeg and avconv).

    Parameters
    ----------
    filename : str or file handle
        Name of the file or file handle.
    sample_rate : int, optional
        Desired sample rate of the signal [Hz], or 'None' to return the
        signal in its original rate.
    num_channels: int, optional
        Reduce or expand the signal to `num_channels` channels, or 'None'
        to return the signal with its original channels.
    start : float, optional
        Start position [seconds].
    stop : float, optional
        Stop position [seconds].
    dtype : numpy data type, optional
        The data is returned with the given dtype. If 'None', it is returned
        with its original dtype, otherwise the signal gets rescaled. Integer
        dtypes use the complete value range, float dtypes the range [-1, +1].

    Returns
    -------
    signal : numpy array
        Audio signal.
    sample_rate : int
        Sample rate of the signal [Hz].

    Notes
    -----
    For wave files, the `start` and `stop` positions are rounded to the closest
    sample; the sample corresponding to the `stop` value is not returned, thus
    consecutive segment starting with the previous `stop` can be concatenated
    to obtain the original signal without gaps or overlaps.
    For all other audio files, this can not be guaranteed.

    """
    from subprocess import CalledProcessError
    from .ffmpeg import load_ffmpeg_file

    # determine the name of the file if it is a file handle
    if not isinstance(filename, str):
        # close the file handle if it is open
        filename.close()
        # use the file name
        filename = filename.name
    # try reading as a wave file
    error = "All attempts to load audio file %r failed." % filename
    try:
        return load_wave_file(filename, sample_rate=sample_rate,
                              num_channels=num_channels, start=start,
                              stop=stop, dtype=dtype)
    except ValueError:
        pass
    # not a wave file (or other sample rate requested), try ffmpeg
    try:
        return load_ffmpeg_file(filename, sample_rate=sample_rate,
                                num_channels=num_channels, start=start,
                                stop=stop, dtype=dtype)
    except OSError:
        # ffmpeg is not present, try avconv
        try:
            return load_ffmpeg_file(filename, sample_rate=sample_rate,
                                    num_channels=num_channels, start=start,
                                    stop=stop, dtype=dtype,
                                    cmd_decode='avconv', cmd_probe='avprobe')
        except OSError:
            error += " Try installing ffmpeg (or avconv on Ubuntu Linux)."
        except CalledProcessError:
            pass
    except CalledProcessError:
        pass
    raise LoadAudioFileError(error)


# signal classes
class Signal(np.ndarray):
    """
    The :class:`Signal` class represents a signal as a (memory-mapped) numpy
    array and enhances it with a number of attributes.

    Parameters
    ----------
    data : numpy array, str or file handle
        Signal data or file name or file handle.
    sample_rate : int, optional
        Desired sample rate of the signal [Hz], or 'None' to return the
        signal in its original rate.
    num_channels : int, optional
        Reduce or expand the signal to `num_channels` channels, or 'None'
        to return the signal with its original channels.
    start : float, optional
        Start position [seconds].
    stop : float, optional
        Stop position [seconds].
    norm : bool, optional
        Normalize the signal to the range [-1, +1].
    gain : float, optional
        Adjust the gain of the signal [dB].
    dtype : numpy data type, optional
        The data is returned with the given dtype. If 'None', it is returned
        with its original dtype, otherwise the signal gets rescaled. Integer
        dtypes use the complete value range, float dtypes the range [-1, +1].

    Notes
    -----
    `sample_rate` or `num_channels` can be used to set the desired sample rate
    and number of channels if the audio is read from file. If set to 'None'
    the audio signal is used as is, i.e. the sample rate and number of channels
    are determined directly from the audio file.

    If the `data` is a numpy array, the `sample_rate` is set to the given value
    and `num_channels` is set to the number of columns of the array.

    The `gain` can be used to adjust the level of the signal.

    If both `norm` and `gain` are set, the signal is first normalized and then
    the gain is applied afterwards.

    If `norm` or `gain` is set, the selected part of the signal is loaded into
    memory completely, i.e. .wav files are not memory-mapped any more.

    """
    # pylint: disable=super-on-old-class
    # pylint: disable=super-init-not-called
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, data, sample_rate=None, num_channels=None, start=None,
                 stop=None, norm=False, gain=0, dtype=None):
        # this method is for documentation purposes only
        pass

    def __new__(cls, data, sample_rate=None, num_channels=None, start=None,
                stop=None, norm=False, gain=0, dtype=None):
        # try to load an audio file if the data is not a numpy array
        if not isinstance(data, np.ndarray):
            data, sample_rate = load_audio_file(data, sample_rate=sample_rate,
                                                num_channels=num_channels,
                                                start=start, stop=stop,
                                                dtype=dtype)
        # process it if needed
        if norm:
            # normalize signal
            data = normalize(data)
        if gain is not None and gain != 0:
            # adjust the gain
            data = adjust_gain(data, gain)
        # cast as Signal
        obj = np.asarray(data).view(cls)
        if sample_rate is not None:
            sample_rate = sample_rate
        obj.sample_rate = sample_rate
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here, also needed for views of the Signal
        self.sample_rate = getattr(obj, 'sample_rate', None)

    @property
    def num_samples(self):
        """Number of samples."""
        return len(self)

    @property
    def num_channels(self):
        """Number of channels."""
        if self.ndim == 1:
            # mono file
            return 1
        else:
            # multi channel file
            return np.shape(self)[1]

    @property
    def length(self):
        """Length of signal in seconds."""
        # n/a if the signal has no sample rate
        if self.sample_rate is None:
            return None
        return float(self.num_samples) / self.sample_rate


class SignalProcessor(Processor):
    """
    The :class:`SignalProcessor` class is a basic signal processor.

    Parameters
    ----------
    sample_rate : int, optional
        Sample rate of the signal [Hz]; if set the signal will be re-sampled
        to that sample rate; if 'None' the sample rate of the audio file will
        be used.
    num_channels : int, optional
        Number of channels of the signal; if set, the signal will be reduced
        to that number of channels; if 'None' as many channels as present in
        the audio file are returned.
    start : float, optional
        Start position [seconds].
    stop : float, optional
        Stop position [seconds].
    norm : bool, optional
        Normalize the signal to the range [-1, +1].
    att : float, optional
        Deprecated in version 0.13, use `gain` instead.
    gain : float, optional
        Adjust the gain of the signal [dB].
    dtype : numpy data type, optional
        The data is returned with the given dtype. If 'None', it is returned
        with its original dtype, otherwise the signal gets rescaled. Integer
        dtypes use the complete value range, float dtypes the range [-1, +1].

    """
    SAMPLE_RATE = None
    NUM_CHANNELS = None
    START = None
    STOP = None
    NORM = False
    ATT = None
    GAIN = 0.

    def __init__(self, sample_rate=SAMPLE_RATE, num_channels=NUM_CHANNELS,
                 start=None, stop=None, norm=NORM, att=ATT, gain=GAIN,
                 **kwargs):
        # pylint: disable=unused-argument
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.start = start
        self.stop = stop
        self.norm = norm
        if att is not None:
            raise DeprecationWarning('`att` has been renamed to `gain` in '
                                     'v0.13 and will be removed in version '
                                     'v0.14')
        self.gain = gain

    @property
    def att(self):
        """Attenuation of the signal [dB]."""
        raise DeprecationWarning('`att` has been renamed to `gain` in v0.13 '
                                 'and will be removed in version v0.14.')
        return -self.gain

    def process(self, data, start=None, stop=None, **kwargs):
        """
        Processes the given audio file.

        Parameters
        ----------
        data : numpy array, str or file handle
            Data to be processed.
        start : float, optional
            Start position [seconds].
        stop : float, optional
            Stop position [seconds].

        Returns
        -------
        signal : :class:`Signal` instance
            :class:`Signal` instance.

        """
        # pylint: disable=unused-argument
        # overwrite the default start & stop time
        if start is None:
            start = self.start
        if stop is None:
            stop = self.stop
        # instantiate a Signal
        data = Signal(data, sample_rate=self.sample_rate,
                      num_channels=self.num_channels, start=start, stop=stop,
                      norm=self.norm, gain=self.gain)
        # return processed data
        return data

    @staticmethod
    def add_arguments(parser, sample_rate=None, mono=None, start=None,
                      stop=None, norm=None, gain=None):
        """
        Add signal processing related arguments to an existing parser.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        sample_rate : int, optional
            Re-sample the signal to this sample rate [Hz].
        mono : bool, optional
            Down-mix the signal to mono.
        start : float, optional
            Start position [seconds].
        stop : float, optional
            Stop position [seconds].
        norm : bool, optional
            Normalize the signal to the range [-1, +1].
        gain : float, optional
            Adjust the gain of the signal [dB].

        Returns
        -------
        argparse argument group
            Signal processing argument parser group.

        Notes
        -----
        Parameters are included in the group only if they are not 'None'. To
        include `start` and `stop` arguments with a default value of 'None',
        i.e. do not set any start or stop time, they can be set to 'True'.

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
        if gain is not None:
            g.add_argument('--gain', action='store', type=float, default=gain,
                           help='adjust the gain of the signal '
                                '[dB, default=%(default).1f]')
        # return the argument group so it can be modified if needed
        return g


# functions for splitting a signal into frames
def signal_frame(signal, index, frame_size, hop_size, origin=0):
    """
    This function returns frame at `index` of the `signal`.

    Parameters
    ----------
    signal : numpy array
        Signal.
    index : int
        Index of the frame to return.
    frame_size : int
        Size of each frame in samples.
    hop_size : float
        Hop size in samples between adjacent frames.
    origin : int
        Location of the window center relative to the signal position.

    Returns
    -------
    frame : numpy array
        Requested frame of the signal.

    Notes
    -----
    The reference sample of the first frame (index == 0) refers to the first
    sample of the `signal`, and each following frame is placed `hop_size`
    samples after the previous one.

    The window is always centered around this reference sample. Its location
    relative to the reference sample can be set with the `origin` parameter.
    Arbitrary integer values can be given:

    - zero centers the window on its reference sample
    - negative values shift the window to the right
    - positive values shift the window to the left

    An `origin` of half the size of the `frame_size` results in windows located
    to the left of the reference sample, i.e. the first frame starts at the
    first sample of the signal.

    The part of the frame which is not covered by the signal is padded with
    zeros.

    This function is totally independent of the length of the signal. Thus,
    contrary to common indexing, the index '-1' refers NOT to the last frame
    of the signal, but instead the frame left of the first frame is returned.

    """

    # length of the signal
    num_samples = len(signal)
    # seek to the correct position in the audio signal
    ref_sample = int(index * hop_size)
    # position the window
    start = ref_sample - frame_size // 2 - int(origin)
    stop = start + frame_size
    # return the requested portion of the signal
    # Note: np.pad(signal[from: to], (pad_left, pad_right), mode='constant')
    #       always returns a ndarray, not the subclass (and is slower);
    #       usually np.zeros_like(signal[:frame_size]) is exactly what we want
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


# classes for splitting a signal into frames
class FramedSignal(object):
    """
    The :class:`FramedSignal` splits a :class:`Signal` into frames and makes it
    iterable and indexable.

    Parameters
    ----------
    signal : :class:`Signal` instance
        Signal to be split into frames.
    frame_size : int, optional
        Size of one frame [samples].
    hop_size : float, optional
        Progress `hop_size` samples between adjacent frames.
    fps : float, optional
        Use given frames per second; if set, this computes and overwrites the
        given `hop_size` value.
    origin : int, optional
        Location of the window relative to the reference sample of a frame.
    end : int or str, optional
        End of signal handling (see notes below).
    num_frames : int, optional
        Number of frames to return.
    kwargs : dict, optional
        If no :class:`Signal` instance was given, one is instantiated with
        these additional keyword arguments.

    Notes
    -----
    The :class:`FramedSignal` class is implemented as an iterator. It splits
    the given `signal` automatically into frames of `frame_size` length with
    `hop_size` samples (can be float, normal rounding applies) between the
    frames. The reference sample of the first frame refers to the first sample
    of the `signal`.

    The location of the window relative to the reference sample of a frame can
    be set with the `origin` parameter (with the same behaviour as used by
    ``scipy.ndimage`` filters). Arbitrary integer values can be given:

    - zero centers the window on its reference sample,
    - negative values shift the window to the right,
    - positive values shift the window to the left.

    Additionally, it can have the following literal values:

    - 'center', 'offline': the window is centered on its reference sample,
    - 'left', 'past', 'online': the window is located to the left of its
      reference sample (including the reference sample),
    - 'right', 'future': the window is located to the right of its reference
      sample.

    The `end` parameter is used to handle the end of signal behaviour and
    can have these values:

    - 'normal': stop as soon as the whole signal got covered by at least one
      frame (i.e. pad maximally one frame),
    - 'extend': frames are returned as long as part of the frame overlaps
      with the signal to cover the whole signal.

    Alternatively, `num_frames` can be used to retrieve a fixed number of
    frames.

    In order to be able to stack multiple frames obtained with different frame
    sizes, the number of frames to be returned must be independent from the set
    `frame_size`. It is not guaranteed that every sample of the signal is
    returned in a frame unless the `origin` is either 'right' or 'future'.

    """

    def __init__(self, signal, frame_size=2048, hop_size=441., fps=None,
                 origin=0, end='normal', num_frames=None, **kwargs):
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
        This makes the :class:`FramedSignal` class indexable and/or iterable.

        The signal is split into frames (of length `frame_size`) automatically.
        Two frames are located `hop_size` samples apart. If `hop_size` is a
        float, normal rounding applies.

        """
        # a single index is given
        if isinstance(index, int):
            # negative indices
            if index < 0:
                index += self.num_frames
            # return the frame at the given index
            if index < self.num_frames:
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
            origin = self.origin - self.hop_size * start
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
        """Frame rate (same as fps)."""
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
        shape = self.num_frames, self.frame_size
        if self.signal.num_channels != 1:
            shape += (self.signal.num_channels, )
        return shape

    @property
    def ndim(self):
        """Dimensionality of the FramedSignal."""
        return len(self.shape)


class FramedSignalProcessor(Processor):
    """
    Slice a Signal into frames.

    Parameters
    ----------
    frame_size : int, optional
        Size of one frame [samples].
    hop_size : float, optional
        Progress `hop_size` samples between adjacent frames.
    fps : float, optional
        Use given frames per second; if set, this computes and overwrites the
        given `hop_size` value.
    online : bool, optional
        Operate in online mode (see notes below).
    end : int or str, optional
        End of signal handling (see :class:`FramedSignal`).
    num_frames : int, optional
        Number of frames to return.
    kwargs : dict, optional
        If no :class:`Signal` instance was given, one is instantiated with
        these additional keyword arguments.

    Notes
    -----
    The location of the window relative to its reference sample can be set
    with the `online` parameter:

    - 'False': the window is centered on its reference sample,
    - 'True': the window is located to the left of its reference sample
      (including the reference sample), i.e. only past information is used.


    """
    FRAME_SIZE = 2048
    HOP_SIZE = 441.
    FPS = 100.
    START = 0
    END_OF_SIGNAL = 'normal'

    def __init__(self, frame_size=FRAME_SIZE, hop_size=HOP_SIZE, fps=None,
                 online=False, end=END_OF_SIGNAL, **kwargs):
        # pylint: disable=unused-argument
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.fps = fps  # do not convert here, pass it to FramedSignal
        self.online = online
        self.end = end

    def process(self, data, **kwargs):
        """
        Slice the signal into (overlapping) frames.

        Parameters
        ----------
        data : :class:`Signal` instance
            Signal to be sliced into frames.
        kwargs : dict
            Keyword arguments passed to :class:`FramedSignal` to instantiate
            the returned object.

        Returns
        -------
        frames : :class:`FramedSignal` instance
            FramedSignal instance

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

    @staticmethod
    def add_arguments(parser, frame_size=FRAME_SIZE, fps=FPS,
                      online=None):
        """
        Add signal framing related arguments to an existing parser.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        frame_size : int, optional
            Size of one frame in samples.
        fps : float, optional
            Frames per second.
        online : bool, optional
            Online mode (use only past signal information, i.e. align the
            window to the left of the reference sample).

        Returns
        -------
        argparse argument group
            Signal framing argument parser group.

        Notes
        -----
        Parameters are included in the group only if they are not 'None'.

        """
        # add signal framing options to the existing parser
        g = parser.add_argument_group('signal framing arguments')
        # depending on the type of frame_size, use different options
        if isinstance(frame_size, int):
            g.add_argument('--frame_size', action='store', type=int,
                           default=frame_size,
                           help='frame size [samples, default=%(default)i]')
        elif isinstance(frame_size, list):
            # Note: this option is used for e.g. stacking multiple spectrograms
            #       with different frame sizes
            from madmom.utils import OverrideDefaultListAction
            g.add_argument('--frame_size', type=int, default=frame_size,
                           action=OverrideDefaultListAction, sep=',',
                           help='(comma separated list of) frame size(s) to '
                                'use [samples, default=%(default)s]')
        if fps is not None:
            g.add_argument('--fps', action='store', type=float, default=fps,
                           help='frames per second [default=%(default).1f]')
        if online is False:
            g.add_argument('--online', dest='online', action='store_true',
                           help='operate in online mode [default=offline]')
        elif online is True:
            g.add_argument('--offline', dest='online', action='store_false',
                           help='operate in offline mode [default=online]')
        # return the argument group so it can be modified if needed
        return g
