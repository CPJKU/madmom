# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains Short-Time Fourier Transform (STFT) related functionality.

"""

from __future__ import absolute_import, division, print_function

import numpy as np

from madmom.processors import Processor


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


def stft(frames, window, fft_size=None, circular_shift=False):
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
        by the `frames` is used; if the given `fft_size` is greater than the
        'frame_size', the frames are zero-padded accordingly.
    circular_shift : bool, optional
        Circular shift the individual frames before performing the FFT;
        needed for correct phase.

    Returns
    -------
    stft : numpy array, shape (num_frames, frame_size)
        The complex STFT of the framed signal.

    """
    import scipy.fftpack as fft
    # check for correct shape of input
    if frames.ndim != 2:
        # TODO: add multi-channel support
        raise ValueError('frames must be a 2D array or iterable')

    # size of the frames
    frame_size = frames.shape[1]

    # window size must match frame size
    if window is not None and len(window) != frame_size:
        raise ValueError('window size must match frame size')

    # FFT size to use
    if fft_size is None:
        fft_size = frame_size
    # fft size must be at least the frame size
    if fft_size < frame_size:
        raise ValueError('FFT size must greater or equal the frame size')
    # number of FFT bins to store
    num_fft_bins = fft_size >> 1

    # size of the FFT circular shift (needed for correct phase)
    if circular_shift:
        fft_shift = frame_size >> 1

    # init objects
    data = np.empty((len(frames), num_fft_bins), STFT_DTYPE)
    signal = np.zeros(frame_size)
    fft_signal = np.zeros(fft_size)

    # iterate over all frames
    for f, frame in enumerate(frames):
        if circular_shift:
            # if we need to circular shift the signal for correct phase, we
            # first multiply the signal frame with the window (or just use it
            # as it is if no window function is given)
            if window is not None:
                np.multiply(frame, window, out=signal)
            else:
                signal = frame
            # then swap the two halves of the windowed signal; if the FFT size
            # is bigger than the frame size, we need to pad the (windowed)
            # signal with additional zeros in between the two halves
            fft_signal[:fft_shift] = signal[fft_shift:]
            fft_signal[-fft_shift:] = signal[:fft_shift]
        else:
            # multiply the signal frame with the window and or save it directly
            # to fft_signal (i.e. bypass the additional copying step above)
            if window is not None:
                np.multiply(frame, window, out=fft_signal[:frame_size])
            else:
                fft_signal[:frame_size] = frame
        # perform DFT
        data[f] = fft.fft(fft_signal, axis=0)[:num_fft_bins]
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


# mixin for some basic properties of all classes
class PropertyMixin(object):
    """
    Mixin which provides `num_frames`, `num_bins` properties to classes.

    """

    @property
    def num_frames(self):
        """Number of frames."""
        return len(self)

    @property
    def num_bins(self):
        """Number of bins."""
        return self.shape[1]


# short-time Fourier transform class
class ShortTimeFourierTransform(PropertyMixin, np.ndarray):
    """
    ShortTimeFourierTransform class.

    Parameters
    ----------
    frames : :class:`.audio.signal.FramedSignal` instance
        FramedSignal instance.
    window : numpy ufunc or numpy array, optional
        Window (function); if a function (e.g. np.hanning) is given, a window
        of the given shape of size of the `frames` is used.
    fft_size : int, optional
        FFT size (should be a power of 2); if 'None', the `frame_size` given by
        the `frames` is used, if the given `fft_size` is greater than the
        `frame_size`, the frames are zero-padded accordingly.
    circular_shift : bool, optional
        Circular shift the individual frames before performing the FFT;
        needed for correct phase.
    kwargs : dict, optional
        If no :class:`.audio.signal.FramedSignal` instance was given, one is
        instantiated with these additional keyword arguments.

    Notes
    -----
    If the :class:`Signal` (wrapped in the :class:`FramedSignal`) has an
    integer dtype, it is automatically scaled as if it has a float dtype with
    the values being in the range [-1, 1]. This results in same valued STFTs
    independently of the dtype of the signal. On the other hand, this prevents
    extra memory consumption since the data-type of the signal does not need to
    be converted (and if no decoding is needed, the audio signal can be memory
    mapped).

    """
    # pylint: disable=super-on-old-class
    # pylint: disable=super-init-not-called
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, frames, window=np.hanning, fft_size=None,
                 circular_shift=False, **kwargs):
        # this method is for documentation purposes only
        pass

    def __new__(cls, frames, window=np.hanning, fft_size=None,
                circular_shift=False, **kwargs):
        # pylint: disable=unused-argument
        from .signal import FramedSignal
        # take the FramedSignal from the given STFT
        if isinstance(frames, ShortTimeFourierTransform):
            # already a STFT
            frames = frames.frames
        # instantiate a FramedSignal if needed
        if not isinstance(frames, FramedSignal):
            frames = FramedSignal(frames, **kwargs)

        # size of the frames
        frame_size = frames.shape[1]

        # if a callable window function is given, use the frame size to create
        # a window of this size
        if hasattr(window, '__call__'):
            window = window(frame_size)
        # window used for FFT
        try:
            # if the audio signal is not scaled, scale the window accordingly
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

        # calculate the STFT
        data = stft(frames, fft_window, fft_size=fft_size,
                    circular_shift=circular_shift)

        # cast as ShortTimeFourierTransform
        obj = np.asarray(data).view(cls)
        # save the other parameters
        obj.frames = frames
        obj.bin_frequencies = fft_frequencies(obj.shape[1],
                                              frames.signal.sample_rate)
        obj.window = window
        obj.fft_window = fft_window
        obj.fft_size = fft_size if fft_size else frame_size
        obj.circular_shift = circular_shift
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here, also needed for views
        self.frames = getattr(obj, 'frames', None)
        self.bin_frequencies = getattr(obj, 'bin_frequencies', None)
        self.window = getattr(obj, 'window', np.hanning)
        self.fft_window = getattr(obj, 'fft_window', None)
        self.fft_size = getattr(obj, 'fft_size', None)
        self.circular_shift = getattr(obj, 'circular_shift', False)

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

    """

    def __init__(self, window=np.hanning, fft_size=None, circular_shift=False,
                 **kwargs):
        # pylint: disable=unused-argument
        self.window = window
        self.fft_size = fft_size
        self.circular_shift = circular_shift

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
        return ShortTimeFourierTransform(data, window=self.window,
                                         fft_size=self.fft_size,
                                         circular_shift=self.circular_shift,
                                         **kwargs)

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
        argpase argument group
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
class Phase(PropertyMixin, np.ndarray):
    """
    Phase class.

    Parameters
    ----------
    stft : :class:`ShortTimeFourierTransform` instance
         :class:`ShortTimeFourierTransform` instance.
    kwargs : dict, optional
        If no :class:`ShortTimeFourierTransform` instance was given, one is
        instantiated with these additional keyword arguments.

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
            import warnings
            warnings.warn("`circular_shift` of the STFT must be set to 'True' "
                          "for correct phase")
        # process the STFT and cast the result as Phase
        obj = np.asarray(phase(stft)).view(cls)
        # save additional attributes
        obj.stft = stft
        obj.bin_frequencies = stft.bin_frequencies
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here, also needed for views
        self.stft = getattr(obj, 'stft', None)
        self.bin_frequencies = getattr(obj, 'bin_frequencies', None)

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
class LocalGroupDelay(PropertyMixin, np.ndarray):
    """
    Local Group Delay class.

    Parameters
    ----------
    stft : :class:`Phase` instance
         :class:`Phase` instance.
    kwargs : dict, optional
        If no :class:`Phase` instance was given, one is instantiated with
        these additional keyword arguments.

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
            import warnings
            warnings.warn("`circular_shift` of the STFT must be set to 'True' "
                          "for correct local group delay")
        # process the phase and cast the result as LocalGroupDelay
        obj = np.asarray(local_group_delay(phase)).view(cls)
        # save additional attributes
        obj.phase = phase
        obj.stft = phase.stft
        obj.bin_frequencies = phase.bin_frequencies
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here, also needed for views
        self.phase = getattr(obj, 'phase', None)
        self.stft = getattr(obj, 'stft', None)
        self.bin_frequencies = getattr(obj, 'bin_frequencies', None)


LGD = LocalGroupDelay
