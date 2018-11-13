# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains Short-Time Fourier Transform (STFT) related functionality.

"""

from __future__ import absolute_import, division, print_function

import warnings

import numpy as np
import scipy.fftpack as fftpack

try:
    from pyfftw.builders import rfft as rfft_builder
except ImportError:
    def rfft_builder(*args, **kwargs):
        return None

from ..processors import Processor
from .signal import Signal, FramedSignal

STFT_DTYPE = np.complex64


def fft_frequencies(num_fft_bins, sample_rate):
    """
    Frequencies of the FFT bins.

    Parameters
    ----------
    num_fft_bins : int
        Number of FFT bins (i.e. half the FFT length).
    sample_rate : float
        Sample rate of the signal.

    Returns
    -------
    fft_frequencies : numpy array
        Frequencies of the FFT bins [Hz].

    """
    return np.fft.fftfreq(num_fft_bins * 2, 1. / sample_rate)[:num_fft_bins]


def stft(frames, window, fft_size=None, circular_shift=False,
         include_nyquist=False, fftw=None):
    """
    Calculates the complex Short-Time Fourier Transform (STFT) of the given
    framed signal.

    Parameters
    ----------
    frames : numpy array or iterable, shape (num_frames, frame_size)
        Framed signal (e.g. :class:`FramedSignal` instance)
    window : numpy array, shape (frame_size,)
        Window (function).
    fft_size : int, optional
        FFT size (should be a power of 2); if 'None', the 'frame_size' given
        by `frames` is used; if the given `fft_size` is greater than the
        'frame_size', the frames are zero-padded, if smaller truncated.
    circular_shift : bool, optional
        Circular shift the individual frames before performing the FFT;
        needed for correct phase.
    include_nyquist : bool, optional
        Include the Nyquist frequency bin (sample rate / 2) in returned STFT.
    fftw : :class:`pyfftw.FFTW` instance, optional
        If a :class:`pyfftw.FFTW` object is given it is used to compute the
        STFT with the FFTW library. Requires 'pyfftw'.

    Returns
    -------
    stft : numpy array, shape (num_frames, frame_size)
        The complex STFT of the framed signal.

    """
    # check for correct shape of input
    if frames.ndim != 2:
        # TODO: add multi-channel support
        raise ValueError('frames must be a 2D array or iterable, got %s with '
                         'shape %s.' % (type(frames), frames.shape))

    # shape of the frames
    num_frames, frame_size = frames.shape

    # FFT size to use
    if fft_size is None:
        fft_size = frame_size
    # number of FFT bins to return
    num_fft_bins = fft_size >> 1
    if include_nyquist:
        num_fft_bins += 1

    # size of the FFT circular shift (needed for correct phase)
    if circular_shift:
        fft_shift = frame_size >> 1

    # init objects
    data = np.empty((num_frames, num_fft_bins), STFT_DTYPE)

    # iterate over all frames
    for f, frame in enumerate(frames):
        if circular_shift:
            # if we need to circular shift the signal for correct phase, we
            # first multiply the signal frame with the window (or just use it
            # as it is if no window function is given)
            if window is not None:
                signal = np.multiply(frame, window)
            else:
                signal = frame
            # then swap the two halves of the windowed signal; if the FFT size
            # is bigger than the frame size, we need to pad the (windowed)
            # signal with additional zeros in between the two halves
            fft_signal = np.zeros(fft_size)
            fft_signal[:fft_shift] = signal[fft_shift:]
            fft_signal[-fft_shift:] = signal[:fft_shift]
        else:
            # multiply the signal frame with the window and or save it directly
            # to fft_signal (i.e. bypass the additional copying step above)
            if window is not None:
                fft_signal = np.multiply(frame, window)
            else:
                fft_signal = frame
        # perform DFT
        if fftw:
            data[f] = fftw(fft_signal)[:num_fft_bins]
        else:
            data[f] = fftpack.fft(fft_signal, fft_size, axis=0)[:num_fft_bins]
    # return STFT
    return data


def phase(stft):
    """
    Returns the phase of the complex STFT of a signal.

    Parameters
    ----------
    stft : numpy array, shape (num_frames, frame_size)
        The complex STFT of a signal.

    Returns
    -------
    phase : numpy array
        Phase of the STFT.

    """
    return np.angle(stft)


def local_group_delay(phase):
    """
    Returns the local group delay of the phase of a signal.

    Parameters
    ----------
    phase : numpy array, shape (num_frames, frame_size)
        Phase of the STFT of a signal.

    Returns
    -------
    lgd : numpy array
        Local group delay of the phase.

    """
    # check for correct shape of input
    if phase.ndim != 2:
        raise ValueError('phase must be a 2D array')
    # unwrap phase
    unwrapped_phase = np.unwrap(phase)
    # local group delay is the derivative over frequency
    unwrapped_phase[:, :-1] -= unwrapped_phase[:, 1:]
    # set the highest frequency to 0
    unwrapped_phase[:, -1] = 0
    # return the local group delay
    return unwrapped_phase


# alias
lgd = local_group_delay


# mixin providing `num_frames` & `num_bins` properties
class _PropertyMixin(object):
    # pylint: disable=missing-docstring

    @property
    def num_frames(self):
        """Number of frames."""
        return len(self)

    @property
    def num_bins(self):
        """Number of bins."""
        return int(self.shape[1])


# short-time Fourier transform class
class ShortTimeFourierTransform(_PropertyMixin, np.ndarray):
    """
    ShortTimeFourierTransform class.

    Parameters
    ----------
    frames : :class:`.audio.signal.FramedSignal` instance
        Framed signal.
    window : numpy ufunc or numpy array, optional
        Window (function); if a function (e.g. `np.hanning`) is given, a window
        with the frame size of `frames` and the given shape is created.
    fft_size : int, optional
        FFT size (should be a power of 2); if 'None', the `frame_size` given by
        `frames` is used, if the given `fft_size` is greater than the
        `frame_size`, the frames are zero-padded accordingly.
    circular_shift : bool, optional
        Circular shift the individual frames before performing the FFT;
        needed for correct phase.
    include_nyquist : bool, optional
        Include the Nyquist frequency bin (sample rate / 2).
    fftw : :class:`pyfftw.FFTW` instance, optional
        If a :class:`pyfftw.FFTW` object is given it is used to compute the
        STFT with the FFTW library. If 'None', a new :class:`pyfftw.FFTW`
        object is built. Requires 'pyfftw'.
    kwargs : dict, optional
        If no :class:`.audio.signal.FramedSignal` instance was given, one is
        instantiated with these additional keyword arguments.

    Notes
    -----
    If the :class:`Signal` (wrapped in the :class:`FramedSignal`) has an
    integer dtype, the `window` is automatically scaled as if the `signal` had
    a float dtype with the values being in the range [-1, 1]. This results in
    same valued STFTs independently of the dtype of the signal. On the other
    hand, this prevents extra memory consumption since the data-type of the
    signal does not need to be converted (and if no decoding is needed, the
    audio signal can be memory-mapped).

    Examples
    --------
    Create a :class:`ShortTimeFourierTransform` from a :class:`Signal` or
    :class:`FramedSignal`:

    >>> sig = Signal('tests/data/audio/sample.wav')
    >>> sig
    Signal([-2494, -2510, ...,   655,   639], dtype=int16)
    >>> frames = FramedSignal(sig, frame_size=2048, hop_size=441)
    >>> frames  # doctest: +ELLIPSIS
    <madmom.audio.signal.FramedSignal object at 0x...>
    >>> stft = ShortTimeFourierTransform(frames)
    >>> stft  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    ShortTimeFourierTransform([[-3.15249+0.j     ,  2.62216-3.02425j, ...,
                                -0.03634-0.00005j,  0.0367 +0.00029j],
                               [-4.28429+0.j     ,  2.02009+2.01264j, ...,
                                -0.01981-0.00933j, -0.00536+0.02162j],
                               ...,
                               [-4.92274+0.j     ,  4.09839-9.42525j, ...,
                                 0.0055 -0.00257j,  0.00137+0.00577j],
                               [-9.22709+0.j     ,  8.76929+4.0005j , ...,
                                 0.00981-0.00014j, -0.00984+0.00006j]],
                              dtype=complex64)

    A ShortTimeFourierTransform can be instantiated directly from a file name:

    >>> stft = ShortTimeFourierTransform('tests/data/audio/sample.wav')
    >>> stft  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    ShortTimeFourierTransform([[...]], dtype=complex64)

    Doing the same with a Signal of float data-type will result in a STFT of
    same value range (rounding errors will occur of course):

    >>> sig = Signal('tests/data/audio/sample.wav', dtype=np.float)
    >>> sig  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    Signal([-0.07611, -0.0766 , ...,  0.01999,  0.0195 ])
    >>> frames = FramedSignal(sig, frame_size=2048, hop_size=441)
    >>> frames  # doctest: +ELLIPSIS
    <madmom.audio.signal.FramedSignal object at 0x...>
    >>> stft = ShortTimeFourierTransform(frames)
    >>> stft  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    ShortTimeFourierTransform([[-3.1524 +0.j     ,  2.62208-3.02415j, ...,
                                -0.03633-0.00005j,  0.0367 +0.00029j],
                               [-4.28416+0.j     ,  2.02003+2.01257j, ...,
                                -0.01981-0.00933j, -0.00536+0.02162j],
                               ...,
                               [-4.92259+0.j     ,  4.09827-9.42496j, ...,
                                 0.0055 -0.00257j,  0.00137+0.00577j],
                               [-9.22681+0.j     ,  8.76902+4.00038j, ...,
                                 0.00981-0.00014j, -0.00984+0.00006j]],
                              dtype=complex64)

    Additional arguments are passed to :class:`FramedSignal` and
    :class:`Signal` respectively:

    >>> stft = ShortTimeFourierTransform('tests/data/audio/sample.wav', \
frame_size=2048, fps=100, sample_rate=22050)
    >>> stft.frames  # doctest: +ELLIPSIS
    <madmom.audio.signal.FramedSignal object at 0x...>
    >>> stft.frames.frame_size
    2048
    >>> stft.frames.hop_size
    220.5
    >>> stft.frames.signal.sample_rate
    22050

    """
    # pylint: disable=super-on-old-class
    # pylint: disable=super-init-not-called
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, frames, window=np.hanning, fft_size=None,
                 circular_shift=False, include_nyquist=False, fft_window=None,
                 fftw=None, **kwargs):
        # this method is for documentation purposes only
        pass

    def __new__(cls, frames, window=np.hanning, fft_size=None,
                circular_shift=False, include_nyquist=False, fft_window=None,
                fftw=None, **kwargs):
        # pylint: disable=unused-argument
        if isinstance(frames, ShortTimeFourierTransform):
            # already a STFT, use the frames thereof
            frames = frames.frames
        # instantiate a FramedSignal if needed
        if not isinstance(frames, FramedSignal):
            frames = FramedSignal(frames, **kwargs)

        # size of the frames
        frame_size = frames.shape[1]

        if fft_window is None:
            # if a callable window function is given, use the frame size to
            # create a window of this size
            if hasattr(window, '__call__'):
                window = window(frame_size)
            # window used for FFT
            try:
                # if the signal is not scaled, scale the window accordingly
                max_range = float(np.iinfo(frames.signal.dtype).max)
                try:
                    # scale the window by the max_range
                    fft_window = window / max_range
                except TypeError:
                    # if the window is None we can't scale it, thus create a
                    # uniform window and scale it accordingly
                    fft_window = np.ones(frame_size) / max_range
            except ValueError:
                # no scaling needed, use the window as is (can also be None)
                fft_window = window

        # use FFTW to speed up STFT
        try:
            # Note: use fft_window instead of a frame because it has already
            #       the correct dtype (frames are multiplied with this window)
            fftw = rfft_builder(fft_window, fft_size, axis=0)
        except AttributeError:
            pass
        # calculate the STFT
        data = stft(frames, fft_window, fft_size=fft_size,
                    circular_shift=circular_shift,
                    include_nyquist=include_nyquist, fftw=fftw)

        # cast as ShortTimeFourierTransform
        obj = np.asarray(data).view(cls)
        # save the other parameters
        obj.frames = frames
        obj.window = window
        obj.fft_window = fft_window
        obj.fft_size = fft_size if fft_size else frame_size
        obj.circular_shift = circular_shift
        obj.include_nyquist = include_nyquist
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here, also needed for views
        self.frames = getattr(obj, 'frames', None)
        self.window = getattr(obj, 'window', np.hanning)
        self.fft_window = getattr(obj, 'fft_window', None)
        self.fftw = getattr(obj, 'fftw', None)
        self.fft_size = getattr(obj, 'fft_size', None)
        self.circular_shift = getattr(obj, 'circular_shift', False)

    @property
    def bin_frequencies(self):
        """Bin frequencies."""
        return fft_frequencies(self.num_bins, self.frames.signal.sample_rate)

    def spec(self, **kwargs):
        """
        Returns the magnitude spectrogram of the STFT.

        Parameters
        ----------
        kwargs : dict, optional
            Keyword arguments passed to
            :class:`.audio.spectrogram.Spectrogram`.

        Returns
        -------
        spec : :class:`.audio.spectrogram.Spectrogram`
            :class:`.audio.spectrogram.Spectrogram` instance.

        """
        # import Spectrogram here, otherwise we have circular imports
        from .spectrogram import Spectrogram
        return Spectrogram(self, **kwargs)

    def phase(self, **kwargs):
        """
        Returns the phase of the STFT.

        Parameters
        ----------
        kwargs : dict, optional
            keyword arguments passed to :class:`Phase`.

        Returns
        -------
        phase : :class:`Phase`
            :class:`Phase` instance.

        """
        return Phase(self, **kwargs)


STFT = ShortTimeFourierTransform


class ShortTimeFourierTransformProcessor(Processor):
    """
    ShortTimeFourierTransformProcessor class.

    Parameters
    ----------
    window : numpy ufunc, optional
        Window function.
    fft_size : int, optional
        FFT size (should be a power of 2); if 'None', it is determined by the
        size of the frames; if is greater than the frame size, the frames are
        zero-padded accordingly.
    circular_shift : bool, optional
        Circular shift the individual frames before performing the FFT;
        needed for correct phase.
    include_nyquist : bool, optional
        Include the Nyquist frequency bin (sample rate / 2).

    Examples
    --------
    Create a :class:`ShortTimeFourierTransformProcessor` and call it with
    either a file name or a the output of a (Framed-)SignalProcessor to obtain
    a :class:`ShortTimeFourierTransform` instance.

    >>> proc = ShortTimeFourierTransformProcessor()
    >>> stft = proc('tests/data/audio/sample.wav')
    >>> stft  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    ShortTimeFourierTransform([[-3.15249+0.j     ,  2.62216-3.02425j, ...,
                                -0.03634-0.00005j,  0.0367 +0.00029j],
                               [-4.28429+0.j     ,  2.02009+2.01264j, ...,
                                -0.01981-0.00933j, -0.00536+0.02162j],
                               ...,
                               [-4.92274+0.j     ,  4.09839-9.42525j, ...,
                                 0.0055 -0.00257j,  0.00137+0.00577j],
                               [-9.22709+0.j     ,  8.76929+4.0005j , ...,
                                 0.00981-0.00014j, -0.00984+0.00006j]],
                              dtype=complex64)

    """

    def __init__(self, window=np.hanning, fft_size=None, circular_shift=False,
                 include_nyquist=False, **kwargs):
        # pylint: disable=unused-argument
        self.window = window
        self.fft_size = fft_size
        self.circular_shift = circular_shift
        self.include_nyquist = include_nyquist
        # caching only, not intended for general use
        self.fft_window = None
        self.fftw = None

    def process(self, data, **kwargs):
        """
        Perform FFT on a framed signal and return the STFT.

        Parameters
        ----------
        data : numpy array
            Data to be processed.
        kwargs : dict, optional
            Keyword arguments passed to :class:`ShortTimeFourierTransform`.

        Returns
        -------
        stft : :class:`ShortTimeFourierTransform`
            :class:`ShortTimeFourierTransform` instance.

        """
        # instantiate a STFT
        data = ShortTimeFourierTransform(data, window=self.window,
                                         fft_size=self.fft_size,
                                         circular_shift=self.circular_shift,
                                         include_nyquist=self.include_nyquist,
                                         fft_window=self.fft_window,
                                         fftw=self.fftw, **kwargs)
        # cache the window used for FFT
        # Note: depending on the signal this may be scaled already
        self.fft_window = data.fft_window
        self.fftw = data.fftw
        return data

    @staticmethod
    def add_arguments(parser, window=None, fft_size=None):
        """
        Add STFT related arguments to an existing parser.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser.
        window : numpy ufunc, optional
            Window function.
        fft_size : int, optional
            Use this size for FFT (should be a power of 2).

        Returns
        -------
        argparse argument group
            STFT argument parser group.

        Notes
        -----
        Parameters are included in the group only if they are not 'None'.

        """
        # add filterbank related options to the existing parser
        g = parser.add_argument_group('short-time Fourier transform arguments')
        if window is not None:
            g.add_argument('--window', dest='window',
                           action='store', default=window,
                           help='window function to use for FFT')
        if fft_size is not None:
            g.add_argument('--fft_size', action='store', type=int,
                           default=fft_size,
                           help='use this size for FFT (should be a power of '
                                '2) [default=%(default)i]')
        # return the group
        return g


STFTProcessor = ShortTimeFourierTransformProcessor


# phase of STFT
class Phase(_PropertyMixin, np.ndarray):
    """
    Phase class.

    Parameters
    ----------
    stft : :class:`ShortTimeFourierTransform` instance
         :class:`ShortTimeFourierTransform` instance.
    kwargs : dict, optional
        If no :class:`ShortTimeFourierTransform` instance was given, one is
        instantiated with these additional keyword arguments.

    Examples
    --------
    Create a :class:`Phase` from a :class:`ShortTimeFourierTransform` (or
    anything it can be instantiated from:

    >>> stft = ShortTimeFourierTransform('tests/data/audio/sample.wav')
    >>> phase = Phase(stft)
    >>> phase  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    Phase([[ 3.14159, -0.85649, ..., -3.14016,  0.00779],
           [ 3.14159,  0.78355, ..., -2.70136,  1.81393],
           ...,
           [ 3.14159, -1.16063, ..., -0.4373 ,  1.33774],
           [ 3.14159,  0.42799, ..., -0.0142 ,  3.13592]], dtype=float32)

    """
    # pylint: disable=super-on-old-class
    # pylint: disable=super-init-not-called
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, stft, **kwargs):
        # this method is for documentation purposes only
        pass

    def __new__(cls, stft, **kwargs):
        # pylint: disable=unused-argument
        # if a Phase object is given use its STFT
        if isinstance(stft, Phase):
            stft = stft.stft
        # instantiate a ShortTimeFourierTransform object if needed
        if not isinstance(stft, ShortTimeFourierTransform):
            # set circular_shift if it was not disables explicitly
            circular_shift = kwargs.pop('circular_shift', True)
            stft = ShortTimeFourierTransform(stft,
                                             circular_shift=circular_shift,
                                             **kwargs)
        # TODO: just recalculate with circular_shift set?
        if not stft.circular_shift:
            warnings.warn("`circular_shift` of the STFT must be set to 'True' "
                          "for correct phase", RuntimeWarning)
        # process the STFT and cast the result as Phase
        obj = np.asarray(phase(stft)).view(cls)
        # save additional attributes
        obj.stft = stft
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here, also needed for views
        self.stft = getattr(obj, 'stft', None)

    @property
    def bin_frequencies(self):
        """Bin frequencies."""
        return self.stft.bin_frequencies

    def local_group_delay(self, **kwargs):
        """
        Returns the local group delay of the phase.

        Parameters
        ----------
        kwargs : dict, optional
            Keyword arguments passed to :class:`LocalGroupDelay`.

        Returns
        -------
        lgd : :class:`LocalGroupDelay` instance
            :class:`LocalGroupDelay` instance.

        """
        return LocalGroupDelay(self, **kwargs)

    lgd = local_group_delay


# local group delay of STFT
class LocalGroupDelay(_PropertyMixin, np.ndarray):
    """
    Local Group Delay class.

    Parameters
    ----------
    stft : :class:`Phase` instance
         :class:`Phase` instance.
    kwargs : dict, optional
        If no :class:`Phase` instance was given, one is instantiated with
        these additional keyword arguments.

    Examples
    --------
    Create a :class:`LocalGroupDelay` from a :class:`ShortTimeFourierTransform`
    (or anything it can be instantiated from:

    >>> stft = ShortTimeFourierTransform('tests/data/audio/sample.wav')
    >>> lgd = LocalGroupDelay(stft)
    >>> lgd  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    LocalGroupDelay([[-2.2851 , -2.25605, ...,  3.13525,  0. ],
                     [ 2.35804,  2.53786, ...,  1.76788,  0. ],
                     ...,
                     [-1.98..., -2.93039, ..., -1.77505,  0. ],
                     [ 2.7136 ,  2.60925, ...,  3.13318,  0. ]])


    """
    # pylint: disable=super-on-old-class
    # pylint: disable=super-init-not-called
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, phase, **kwargs):
        # this method is for documentation purposes only
        pass

    def __new__(cls, phase, **kwargs):
        # pylint: disable=unused-argument
        # try to instantiate a Phase object
        if not isinstance(stft, Phase):
            phase = Phase(phase, circular_shift=True, **kwargs)
        if not phase.stft.circular_shift:
            warnings.warn("`circular_shift` of the STFT must be set to 'True' "
                          "for correct local group delay")
        # process the phase and cast the result as LocalGroupDelay
        obj = np.asarray(local_group_delay(phase)).view(cls)
        # save additional attributes
        obj.phase = phase
        obj.stft = phase.stft
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here, also needed for views
        self.phase = getattr(obj, 'phase', None)
        self.stft = getattr(obj, 'stft', None)

    @property
    def bin_frequencies(self):
        """Bin frequencies."""
        return self.stft.bin_frequencies


LGD = LocalGroupDelay
