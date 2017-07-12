# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains basic signal processing functionality.

"""

from __future__ import absolute_import, division, print_function

import warnings
import numpy as np

from ..processors import BufferProcessor, Processor


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
    elif isinstance(kernel, (int, np.integer)):
        if kernel == 0:
            return signal
        elif kernel > 1:
            # use a Hamming window of given length
            kernel = np.hamming(kernel)
        else:
            raise ValueError("can't create a smoothing kernel of size %d" %
                             kernel)
    # otherwise use the given smoothing kernel directly
    elif isinstance(kernel, np.ndarray):
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


def resample(signal, sample_rate, **kwargs):
    """
    Resample the signal.

    Parameters
    ----------
    signal : numpy array or Signal
        Signal to be resampled.
    sample_rate : int
        Sample rate of the signal.
    kwargs : dict, optional
        Keyword arguments passed to :func:`load_ffmpeg_file`.

    Returns
    -------
    numpy array or Signal
        Resampled signal.

    Notes
    -----
    This function uses ``ffmpeg`` to resample the signal.

    """
    from ..io.audio import load_ffmpeg_file
    # is the given signal a Signal?
    if not isinstance(signal, Signal):
        raise ValueError('only Signals can resampled, not %s' % type(signal))
    if signal.sample_rate == sample_rate:
        return signal
    # per default use the signal's dtype and num_channels
    dtype = kwargs.get('dtype', signal.dtype)
    num_channels = kwargs.get('num_channels', signal.num_channels)
    # resample the signal
    signal, sample_rate = load_ffmpeg_file(signal, sample_rate=sample_rate,
                                           num_channels=num_channels,
                                           dtype=dtype)
    # return it
    return Signal(signal, sample_rate=sample_rate)


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


def energy(signal):
    """
    Compute the energy of a (framed) signal.

    Parameters
    ----------
    signal : numpy array
        Signal.

    Returns
    -------
    energy : float
        Energy of the signal.

    Notes
    -----
    If `signal` is a `FramedSignal`, the energy is computed for each frame
    individually.

    """
    # compute the energy for every frame of the signal
    if isinstance(signal, FramedSignal):
        return np.array([energy(frame) for frame in signal])
    # make sure the signal is a numpy array
    if not isinstance(signal, np.ndarray):
        raise TypeError("Invalid type for signal, must be a numpy array.")
    # take the abs if the signal is complex
    if np.iscomplex(signal).any():
        signal = np.abs(signal)
    # Note: type conversion needed because of integer overflows
    if signal.dtype != np.float:
        signal = signal.astype(np.float)
    # return energy
    return np.dot(signal.flatten(), signal.flatten())


def root_mean_square(signal):
    """
    Compute the root mean square of a (framed) signal. This can be used as a
    measurement of power.

    Parameters
    ----------
    signal : numpy array
        Signal.

    Returns
    -------
    rms : float
        Root mean square of the signal.

    Notes
    -----
    If `signal` is a `FramedSignal`, the root mean square is computed for each
    frame individually.

    """
    # compute the root mean square for every frame of the signal
    if isinstance(signal, FramedSignal):
        return np.array([root_mean_square(frame) for frame in signal])
    return np.sqrt(energy(signal) / signal.size)


def sound_pressure_level(signal, p_ref=None):
    """
    Compute the sound pressure level of a (framed) signal.

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

    If `signal` is a `FramedSignal`, the sound pressure level is computed for
    each frame individually.

    """
    # compute the sound pressure level for every frame of the signal
    if isinstance(signal, FramedSignal):
        return np.array([sound_pressure_level(frame) for frame in signal])
    # compute the RMS
    rms = root_mean_square(signal)
    # find a reasonable default reference value if None is given
    if p_ref is None:
        if np.issubdtype(signal.dtype, np.integer):
            p_ref = float(np.iinfo(signal.dtype).max)
        else:
            p_ref = 1.0
    # normal SPL computation. ignore warnings when taking the log of 0,
    # then replace the resulting -inf values with the smallest finite number
    with np.errstate(divide='ignore'):
        return np.nan_to_num(20.0 * np.log10(rms / p_ref))


# functions to load / write audio files
class LoadAudioFileError(Exception):
    """
    Deprecated as of version 0.16. Please use
    madmom.io.audio.LoadAudioFileError instead. Will be removed in version
    0.17.

    """
    # pylint: disable=super-init-not-called

    def __init__(self, value=None):
        warnings.warn(LoadAudioFileError.__doc__)
        if value is None:
            value = 'Could not load audio file.'
        self.value = value

    def __str__(self):
        return repr(self.value)


def load_wave_file(*args, **kwargs):
    """
    Deprecated as of version 0.16. Please use madmom.io.audio.load_wave_file
    instead. Will be removed in version 0.17.

    """
    warnings.warn(load_audio_file.__doc__)
    from ..io.audio import load_wave_file
    return load_wave_file(*args, **kwargs)


def write_wave_file(*args, **kwargs):
    """
    Deprecated as of version 0.16. Please use madmom.io.audio.write_wave_file
    instead. Will be removed in version 0.17.

    """
    warnings.warn(load_audio_file.__doc__)
    from ..io.audio import write_wave_file
    return write_wave_file(*args, **kwargs)


# function for automatically determining how to open audio files
def load_audio_file(*args, **kwargs):
    """
    Deprecated as of version 0.16. Please use madmom.io.audio.load_audio_file
    instead. Will be removed in version 0.17.

    """
    warnings.warn(load_audio_file.__doc__)
    from ..io.audio import load_wave_file
    return load_wave_file(*args, **kwargs)


# signal classes
SAMPLE_RATE = None
NUM_CHANNELS = None
START = None
STOP = None
NORM = False
GAIN = 0.
DTYPE = None


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
        Normalize the signal to maximum range of the data type.
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

    Examples
    --------
    Load a mono audio file:

    >>> sig = Signal('tests/data/audio/sample.wav')
    >>> sig
    Signal([-2494, -2510, ...,   655,   639], dtype=int16)
    >>> sig.sample_rate
    44100

    Load a stereo audio file, down-mix it to mono:

    >>> sig = Signal('tests/data/audio/stereo_sample.flac', num_channels=1)
    >>> sig
    Signal([ 36,  36, ..., 524, 495], dtype=int16)
    >>> sig.num_channels
    1

    Load and re-sample an audio file:

    >>> sig = Signal('tests/data/audio/sample.wav', sample_rate=22050)
    >>> sig
    Signal([-2470, -2553, ...,   517,   677], dtype=int16)
    >>> sig.sample_rate
    22050

    Load an audio file with `float32` data type (i.e. rescale it to [-1, 1]):

    >>> sig = Signal('tests/data/audio/sample.wav', dtype=np.float32)
    >>> sig
    Signal([-0.07611, -0.0766 , ...,  0.01999,  0.0195 ], dtype=float32)
    >>> sig.dtype
    dtype('float32')

    """
    # pylint: disable=super-on-old-class
    # pylint: disable=super-init-not-called
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, data, sample_rate=SAMPLE_RATE,
                 num_channels=NUM_CHANNELS, start=START, stop=STOP, norm=NORM,
                 gain=GAIN, dtype=DTYPE, **kwargs):
        # this method is for documentation purposes only
        pass

    def __new__(cls, data, sample_rate=SAMPLE_RATE, num_channels=NUM_CHANNELS,
                start=START, stop=STOP, norm=NORM, gain=GAIN, dtype=DTYPE,
                **kwargs):
        from ..io.audio import load_audio_file
        # try to load an audio file if the data is not a numpy array
        if not isinstance(data, np.ndarray):
            data, sample_rate = load_audio_file(data, sample_rate=sample_rate,
                                                num_channels=num_channels,
                                                start=start, stop=stop,
                                                dtype=dtype)
        # cast as Signal if needed
        if not isinstance(data, Signal):
            data = np.asarray(data).view(cls)
            data.sample_rate = sample_rate
        # normalize signal if needed
        if norm:
            data = normalize(data)
        # adjust the gain if needed
        if gain is not None and gain != 0:
            data = adjust_gain(data, gain)
        # resample if needed
        if sample_rate != data.sample_rate:
            data = resample(data, sample_rate)
        # save start and stop position
        if start is not None:
            # FIXME: start and stop settings are not checked
            data.start = start
            data.stop = start + float(len(data)) / sample_rate
        # return the object
        return data

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here, also needed for views of the Signal
        self.sample_rate = getattr(obj, 'sample_rate', None)
        self.start = getattr(obj, 'start', None)
        self.stop = getattr(obj, 'stop', None)

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

    def write(self, filename):
        """
        Write the signal to disk as a .wav file.

        Parameters
        ----------
        filename : str
            Name of the file.

        Returns
        -------
        filename : str
            Name of the written file.

        """
        return write_wave_file(self, filename)

    def energy(self):
        """Energy of signal."""
        return energy(self)

    def root_mean_square(self):
        """Root mean square of signal."""
        return root_mean_square(self)

    rms = root_mean_square

    def sound_pressure_level(self):
        """Sound pressure level of signal."""
        return sound_pressure_level(self)

    spl = sound_pressure_level


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
    gain : float, optional
        Adjust the gain of the signal [dB].
    dtype : numpy data type, optional
        The data is returned with the given dtype. If 'None', it is returned
        with its original dtype, otherwise the signal gets rescaled. Integer
        dtypes use the complete value range, float dtypes the range [-1, +1].

    Examples
    --------
    Processor for loading the first two seconds of an audio file, re-sampling
    it to 22.05 kHz and down-mixing it to mono:

    >>> proc = SignalProcessor(sample_rate=22050, num_channels=1, stop=2)
    >>> sig = proc('tests/data/audio/sample.wav')
    >>> sig
    Signal([-2470, -2553, ...,  -173,  -265], dtype=int16)
    >>> sig.sample_rate
    22050
    >>> sig.num_channels
    1
    >>> sig.length
    2.0

    """

    def __init__(self, sample_rate=SAMPLE_RATE, num_channels=NUM_CHANNELS,
                 start=START, stop=STOP, norm=NORM, gain=GAIN, **kwargs):
        # pylint: disable=unused-argument
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.start = start
        self.stop = stop
        self.norm = norm
        self.gain = gain

    def process(self, data, **kwargs):
        """
        Processes the given audio file.

        Parameters
        ----------
        data : numpy array, str or file handle
            Data to be processed.
        kwargs : dict, optional
            Keyword arguments passed to :class:`Signal`.

        Returns
        -------
        signal : :class:`Signal` instance
            :class:`Signal` instance.

        """
        # pylint: disable=unused-argument
        # update arguments passed to FramedSignal
        args = dict(sample_rate=self.sample_rate,
                    num_channels=self.num_channels, start=self.start,
                    stop=self.stop, norm=self.norm, gain=self.gain)
        args.update(kwargs)
        # instantiate a Signal and return it
        return Signal(data, **args)

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
            g.add_argument('--sample_rate', action='store', type=int,
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
    # cast variables to int
    frame_size = int(frame_size)
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


FRAME_SIZE = 2048
HOP_SIZE = 441.
FPS = None
ORIGIN = 0
END_OF_SIGNAL = 'normal'
NUM_FRAMES = None


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
    - 'right', 'future', 'stream': the window is located to the right of its
      reference sample.

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

    If used in online real-time mode the parameters `origin` and `num_frames`
    should be set to 'stream' and 1, respectively.

    Examples
    --------
    To chop a :class:`Signal` (or anything a :class:`Signal` can be
    instantiated from) into overlapping frames of size 2048 with adjacent
    frames being 441 samples apart:

    >>> sig = Signal('tests/data/audio/sample.wav')
    >>> sig
    Signal([-2494, -2510, ...,   655,   639], dtype=int16)
    >>> frames = FramedSignal(sig, frame_size=2048, hop_size=441)
    >>> frames  # doctest: +ELLIPSIS
    <madmom.audio.signal.FramedSignal object at 0x...>
    >>> frames[0]
    Signal([    0,     0, ..., -4666, -4589], dtype=int16)
    >>> frames[10]
    Signal([-6156, -5645, ...,  -253,   671], dtype=int16)
    >>> frames.fps
    100.0

    Instead of passing a :class:`Signal` instance as the first argument,
    anything a :class:`Signal` can be instantiated from (e.g. a file name) can
    be used. We can also set the frames per second (`fps`) instead, they get
    converted to `hop_size` based on the `sample_rate` of the signal:

    >>> frames = FramedSignal('tests/data/audio/sample.wav', fps=100)
    >>> frames  # doctest: +ELLIPSIS
    <madmom.audio.signal.FramedSignal object at 0x...>
    >>> frames[0]
    Signal([    0,     0, ..., -4666, -4589], dtype=int16)
    >>> frames.frame_size, frames.hop_size
    (2048, 441.0)

    When trying to access an out of range frame, an IndexError is raised. Thus
    the FramedSignal can be used the same way as a numpy array or any other
    iterable.

    >>> frames = FramedSignal('tests/data/audio/sample.wav')
    >>> frames.num_frames
    281
    >>> frames[281]
    Traceback (most recent call last):
    IndexError: end of signal reached
    >>> frames.shape
    (281, 2048)

    Slices are FramedSignals itself:

    >>> frames[:4]  # doctest: +ELLIPSIS
    <madmom.audio.signal.FramedSignal object at 0x...>

    To obtain a numpy array from a FramedSignal, simply use np.array() on the
    full FramedSignal or a slice of it. Please note, that this requires a full
    memory copy.

    >>> np.array(frames[2:4])
    array([[    0,     0, ..., -5316, -5405],
           [ 2215,  2281, ...,   561,   653]], dtype=int16)

    """

    def __init__(self, signal, frame_size=FRAME_SIZE, hop_size=HOP_SIZE,
                 fps=FPS, origin=ORIGIN, end=END_OF_SIGNAL,
                 num_frames=NUM_FRAMES, **kwargs):

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
            # window centered around the origin
            origin = 0
        elif origin in ('left', 'past', 'online'):
            # origin is the right edge of the frame, i.e. window to the left
            # Note: used when simulating online mode, where only past
            #       information of the audio signal can be used
            origin = (frame_size - 1) / 2
        elif origin in ('right', 'future', 'stream'):
            # origin is the left edge of the frame, i.e. window to the right
            # Note: used when operating on live audio streams where we want
            #       to retrieve a single frame. Instead of using 'online', we
            #       "fake" the origin in order to retrieve the complete frame
            #       provided by FramedSignalProcessor. This is a workaround to
            #       be able to use the same processing chain in different modes
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
        """
        Shape of the FramedSignal (num_frames, frame_size[, num_channels]).

        """
        shape = self.num_frames, self.frame_size
        if self.signal.num_channels != 1:
            shape += (self.signal.num_channels, )
        return shape

    @property
    def ndim(self):
        """Dimensionality of the FramedSignal."""
        return len(self.shape)

    def energy(self):
        """Energy of the individual frames."""
        return energy(self)

    def root_mean_square(self):
        """Root mean square of the individual frames."""
        return root_mean_square(self)

    rms = root_mean_square

    def sound_pressure_level(self):
        """Sound pressure level of the individual frames."""
        return sound_pressure_level(self)

    spl = sound_pressure_level


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
    origin : int, optional
        Location of the window relative to the reference sample of a frame.
    end : int or str, optional
        End of signal handling (see :class:`FramedSignal`).
    num_frames : int, optional
        Number of frames to return.
    kwargs : dict, optional
        If no :class:`Signal` instance was given, one is instantiated with
        these additional keyword arguments.

    Notes
    -----
    When operating on live audio signals, `origin` must be set to 'stream' in
    order to retrieve always the last `frame_size` samples.

    Examples
    --------
    Processor for chopping a :class:`Signal` (or anything a :class:`Signal` can
    be instantiated from) into overlapping frames of size 2048, and a frame
    rate of 100 frames per second:

    >>> proc = FramedSignalProcessor(frame_size=2048, fps=100)
    >>> frames = proc('tests/data/audio/sample.wav')
    >>> frames  # doctest: +ELLIPSIS
    <madmom.audio.signal.FramedSignal object at 0x...>
    >>> frames[0]
    Signal([    0,     0, ..., -4666, -4589], dtype=int16)
    >>> frames[10]
    Signal([-6156, -5645, ...,  -253,   671], dtype=int16)
    >>> frames.hop_size
    441.0

    """

    def __init__(self, frame_size=FRAME_SIZE, hop_size=HOP_SIZE, fps=FPS,
                 origin=ORIGIN, end=END_OF_SIGNAL, num_frames=NUM_FRAMES,
                 **kwargs):
        # pylint: disable=unused-argument
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.fps = fps  # do not convert here, pass it to FramedSignal
        self.origin = origin
        self.end = end
        self.num_frames = num_frames

    def process(self, data, **kwargs):
        """
        Slice the signal into (overlapping) frames.

        Parameters
        ----------
        data : :class:`Signal` instance
            Signal to be sliced into frames.
        kwargs : dict, optional
            Keyword arguments passed to :class:`FramedSignal`.

        Returns
        -------
        frames : :class:`FramedSignal` instance
            FramedSignal instance

        """
        # update arguments passed to FramedSignal
        args = dict(frame_size=self.frame_size, hop_size=self.hop_size,
                    fps=self.fps, origin=self.origin, end=self.end,
                    num_frames=self.num_frames)
        args.update(kwargs)
        # always use the last `frame_size` samples if we operate on a live
        # audio stream, otherwise we get the wrong portion of the signal
        if self.origin == 'stream':
            data = data[-self.frame_size:]
        # instantiate a FramedSignal from the data and return it
        return FramedSignal(data, **args)

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
            g.add_argument('--online', dest='origin', action='store_const',
                           const='online', default='offline',
                           help='operate in online mode [default=offline]')
        elif online is True:
            g.add_argument('--offline', dest='origin', action='store_const',
                           const='offline', default='online',
                           help='operate in offline mode [default=online]')
        # return the argument group so it can be modified if needed
        return g


# class for online processing
class Stream(object):
    """
    A Stream handles live (i.e. online, real-time) audio input via PyAudio.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the signal.
    num_channels : int, optional
        Number of channels.
    dtype : numpy dtype, optional
        Data type for the signal.
    frame_size : int, optional
        Size of one frame [samples].
    hop_size : int, optional
        Progress `hop_size` samples between adjacent frames.
    fps : float, optional
        Use given frames per second; if set, this computes and overwrites the
        given `hop_size` value (the resulting `hop_size` must be an integer).
    queue_size : int
        Size of the FIFO (first in first out) queue. If the queue is full and
        new audio samples arrive, the oldest item in the queue will be dropped.

    Notes
    -----
    Stream is implemented as an iterable which blocks until enough new data is
    available.

    """

    def __init__(self, sample_rate=SAMPLE_RATE, num_channels=NUM_CHANNELS,
                 dtype=np.float32, frame_size=FRAME_SIZE, hop_size=HOP_SIZE,
                 fps=FPS, **kwargs):
        # import PyAudio here and not at the module level
        import pyaudio
        # set attributes
        self.sample_rate = sample_rate
        self.num_channels = 1 if None else num_channels
        self.dtype = dtype
        if frame_size:
            self.frame_size = int(frame_size)
        if fps:
            # use fps instead of hop_size
            hop_size = self.sample_rate / float(fps)
        if int(hop_size) != hop_size:
            raise ValueError(
                'only integer `hop_size` supported, not %s' % hop_size)
        self.hop_size = int(hop_size)
        # init PyAudio
        self.pa = pyaudio.PyAudio()
        # init a stream to read audio samples from
        self.stream = self.pa.open(rate=self.sample_rate,
                                   channels=self.num_channels,
                                   format=pyaudio.paFloat32, input=True,
                                   frames_per_buffer=self.hop_size,
                                   start=True)
        # create a buffer
        self.buffer = BufferProcessor(self.frame_size)
        # frame index counter
        self.frame_idx = 0
        # PyAudio flags
        self.paComplete = pyaudio.paComplete
        self.paContinue = pyaudio.paContinue

    def __iter__(self):
        return self

    def __next__(self):
        # get the desired number of samples (block until all are present)
        data = self.stream.read(self.hop_size, exception_on_overflow=False)
        # convert it to a numpy array
        data = np.fromstring(data, 'float32').astype(self.dtype, copy=False)
        # buffer the data (i.e. append hop_size samples and rotate)
        data = self.buffer(data)
        # wrap the last frame_size samples as a Signal
        # TODO: check float / int hop size; theoretically a float hop size
        #       can be accomplished by making the buffer N samples bigger and
        #       take the correct portion of the buffer
        start = (self.frame_idx * float(self.hop_size) / self.sample_rate)
        signal = Signal(data[-self.frame_size:], sample_rate=self.sample_rate,
                        dtype=self.dtype, num_channels=self.num_channels,
                        start=start)
        # increment the frame index
        self.frame_idx += 1
        return signal

    next = __next__

    def is_running(self):
        return self.stream.is_active()

    def close(self):
        self.stream.close()
        # TODO: is this the correct place to terminate PyAudio?
        self.pa.terminate()

    @property
    def shape(self):
        """Shape of the Stream (None, frame_size[, num_channels])."""
        shape = None, self.frame_size
        if self.signal.num_channels != 1:
            shape += (self.signal.num_channels,)
        return shape
