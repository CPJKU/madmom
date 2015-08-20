#!/usr/bin/env python
# encoding: utf-8
"""
This file contains STFT related functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import numpy as np
import scipy.fftpack as fft

from madmom.processors import Processor


def fft_frequencies(num_fft_bins, sample_rate):
    """
    Frequencies of the FFT bins.

    :param num_fft_bins: number of FFT bins (i.e. half the FFT length)
    :param sample_rate:  sample rate of the signal
    :return:             frequencies of the FFT bins

    """
    return np.fft.fftfreq(num_fft_bins * 2, 1. / sample_rate)[:num_fft_bins]


def stft(frames, window, fft_size=None, circular_shift=False):
    """
    Calculates the complex Short-Time Fourier Transform (STFT) of the given
    framed signal.

    :param frames:         framed signal [2D numpy array or iterable]
    :param window:         window function [1D numpy array]
    :param fft_size:       use this size for FFT [int, should be a power of 2]
    :param circular_shift: circular shift for correct phase [bool]
    :return:               the complex STFT of the signal

    """
    # size of the window
    window_size = len(window)
    # size of the FFT circular shift (needed for correct phase)
    if circular_shift:
        fft_shift = window_size >> 1
    # FFT size to use
    if fft_size is None:
        fft_size = window_size
    # number of FFT bins to store
    num_fft_bins = fft_size >> 1

    # init objects
    data = np.empty((len(frames), num_fft_bins), np.complex64)
    signal = np.zeros(window_size)
    fft_signal = np.zeros(fft_size)

    # iterate over all frames
    for f, frame in enumerate(frames):
        # multiply the signal frame with the window and take the DFT
        if circular_shift:
            # if we need a circular shift of the signal for correct phase
            # we first multiply the signal frame with the window
            np.multiply(frame, window, out=signal)
            # and then swap the two halves of the windowed signal
            # if we have a bigger FFT size than frame size, wee need to
            # pad the additional zeros in between the two halves
            fft_signal[:fft_shift] = signal[fft_shift:]
            fft_signal[-fft_shift:] = signal[:fft_shift]
        else:
            # just multiply the signal frame with the window and save it
            # directly in fft_signal (i.e. bypass the signal)
            np.multiply(frame, window, out=fft_signal[:window_size])
        # perform DFT and return the signal
        data[f] = fft.fft(fft_signal, axis=0)[:num_fft_bins]
    # return STFT
    return data


def phase(stft):
    """
    Returns the phase of the complex Short Time Fourier Transform of a signal.

    :param stft: complex STFT of a signal
    :return:     phase

    """
    return np.angle(stft)


def local_group_delay(phase):
    """
    Returns the local group delay of the phase Short Time Fourier Transform of
    a signal.

    :param phase: phase of the STFT of a signal
    :return:      local group delay

    """
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

    @property
    def num_frames(self):
        """Number of frames."""
        return len(self)

    @property
    def num_bins(self):
        """Number of bins."""
        return self.shape[1]

    @property
    def bin_frequencies(self):
        """Frequencies of the bins."""
        try:
            return self.filterbank.center_frequencies
        except AttributeError:
            return fft_frequencies(self.num_bins,
                                   self.frames.signal.sample_rate)


# short-time Fourier transform classes
class ShortTimeFourierTransform(PropertyMixin, np.ndarray):
    """
    ShortTimeFourierTransform class.

    """

    def __new__(cls, frames, window=np.hanning, fft_size=None,
                circular_shift=False, **kwargs):
        """
        Creates a new ShortTimeFourierTransform instance from the given
        FramedSignal.

        :param frames:         FramedSignal instance (or anything a
                               FramedSignal can be instantiated from)

        FFT parameters:

        :param window:         window function [numpy ufunc or numpy array]
        :param fft_size:       use this size for the FFT [int, power of 2]
        :param circular_shift: circular shift the signal before performing the
                               FFT; needed for correct phase

        If no FramedSignal instance was given, one is instantiated and these
        arguments are passed:

        :param kwargs:         keyword arguments passed to FramedSignal

        """
        from .signal import FramedSignal

        # take the FramedSignal from the given STFT
        if isinstance(frames, ShortTimeFourierTransform):
            # already a STFT
            frames = frames.frames
        # instantiate a FramedSignal if needed
        if not isinstance(frames, FramedSignal):
            frames = FramedSignal(frames, **kwargs)

        # check if the Signal is mono
        if frames.signal.num_channels > 1:
            raise ValueError('please implement multi-channel support')

        # determine which window to use
        if hasattr(window, '__call__'):
            # if only function is given, use the size to the audio frame size
            window = window(frames.frame_size)
            # # multi-channel window
            # if frames.signal.num_channels > 1:
            #     window = np.tile(window[:, np.newaxis],
            #                      frames.signal.num_channels)
        elif isinstance(window, np.ndarray):
            # otherwise use the given window directly
            if len(window) != frames.frame_size:
                raise ValueError('Window size must be equal to frame size.')
        else:
            # other types are not supported
            raise TypeError("Invalid window type.")
        # window used for FFT
        try:
            # the audio signal is not scaled, scale the window accordingly
            max_range = np.iinfo(frames.signal.dtype).max
            fft_window = window / max_range
        except ValueError:
            fft_window = window

        # calculate the STFT
        data = stft(frames, fft_window, fft_size=fft_size,
                    circular_shift=circular_shift)

        # cast as ShortTimeFourierTransform
        obj = np.asarray(data).view(cls)
        # save the other parameters
        obj.frames = frames
        obj.window = window
        obj.fft_window = fft_window
        obj.fft_size = fft_size
        obj.circular_shift = circular_shift
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here, also needed for views
        self.frames = getattr(obj, 'frames', None)
        self.window = getattr(obj, 'window', np.hanning)
        self.fft_window = getattr(obj, 'fft_window', None)
        self.fft_size = getattr(obj, 'fft_size', None)
        self.circular_shift = getattr(obj, 'circular_shift', False)

    def __reduce__(self):
        # needed for correct pickling
        # source: http://stackoverflow.com/questions/26598109/
        # get the parent's __reduce__ tuple
        pickled_state = super(ShortTimeFourierTransform, self).__reduce__()
        # create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.window, self.fft_window,
                                        self.fft_size, self.circular_shift)
        # return a tuple that replaces the parent's __reduce__ tuple
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):
        # needed for correct un-pickling
        # set the attributes
        self.window = state[-4]
        self.fft_window = state[-3]
        self.fft_size = state[-2]
        self.circular_shift = state[-1]
        # call the parent's __setstate__ with the other tuple elements
        super(ShortTimeFourierTransform, self).__setstate__(state[0:-4])

    def spec(self, **kwargs):
        """
        Compute the magnitude spectrogram of the STFT.

        :param kwargs: keyword arguments passed to Spectrogram
        :return:       Spectrogram instance

        """
        # import Spectrogram here, otherwise we have circular imports
        from .spectrogram import Spectrogram
        return Spectrogram(self, **kwargs)

    def phase(self, **kwargs):
        """
        Compute the phase of the STFT.

        :param kwargs: keyword arguments passed to Phase
        :return:       Phase instance

        """
        return Phase(self, **kwargs)


STFT = ShortTimeFourierTransform


class ShortTimeFourierTransformProcessor(Processor):
    """
    ShortTimeFourierTransformProcessor class.

    """

    def __init__(self, window=np.hanning, fft_size=None, circular_shift=False,
                 **kwargs):
        """
        Creates a new ShortTimeFourierTransformProcessor instance.

        :param window:         window function [numpy ufunc or numpy array]
        :param fft_size:       use this size for the FFT [int, power of 2]
        :param circular_shift: circular shift the signal before performing the
                               FFT; needed for correct phase

        """
        self.window = window
        self.fft_size = fft_size
        self.circular_shift = circular_shift

    def process(self, data, **kwargs):
        """
        Perform FFT on a framed signal and return the STFT.

        :param data:   data to be processed
        :param kwargs: keyword arguments passed to ShortTimeFourierTransform
        :return:       ShortTimeFourierTransform instance

        """
        # instantiate a STFT
        return ShortTimeFourierTransform(data, window=self.window,
                                         fft_size=self.fft_size,
                                         circular_shift=self.circular_shift,
                                         **kwargs)

    @classmethod
    def add_arguments(cls, parser, window=None, fft_size=None):
        """
        Add STFT related arguments to an existing parser.

        :param parser:   existing argparse parser
        :param window:   window function
        :param fft_size: use this size for FFT [int, should be a power of 2]
        :return:         STFT argument parser group

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

    """

    def __new__(cls, stft, **kwargs):
        """
        Creates a new Phase instance from the given ShortTimeFourierTransform.

        :param stft:   ShortTimeFourierTransform instance (or anything a
                       ShortTimeFourierTransform can be instantiated from)

        If no ShortTimeFourierTransform instance was given, one is instantiated
        and these arguments are passed:

        :param kwargs: keyword arguments passed to ShortTimeFourierTransform

        """
        circular_shift = kwargs.pop('circular_shift', True)
        # take the STFT
        if isinstance(stft, Phase):
            stft = stft.stft
        # instantiate a ShortTimeFourierTransform object if needed
        if not isinstance(stft, ShortTimeFourierTransform):
            stft = ShortTimeFourierTransform(stft,
                                             circular_shift=circular_shift,
                                             **kwargs)
        # TODO: just recalculate with circular_shift set?
        if not stft.circular_shift:
            import warnings
            warnings.warn("`circular_shift` of the STFT must be set to 'True' "
                          "for correct phase")
        # take the angle of the stft
        data = np.angle(stft)
        # cast as Phase
        obj = np.asarray(data).view(cls)
        # save additional attributes
        obj.stft = stft
        obj.frames = stft.frames
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here, also needed for views
        self.stft = getattr(obj, 'stft', None)
        self.frames = getattr(obj, 'frames', None)

    def local_group_delay(self, **kwargs):
        """
        Compute the local group delay of the phase.

        :param kwargs: keyword arguments passed to LocalGroupDelay
        :return:       LocalGroupDelay instance

        """
        return LocalGroupDelay(self, **kwargs)

    lgd = local_group_delay


# local group delay of STFT
class LocalGroupDelay(PropertyMixin, np.ndarray):
    """
    Local Group Delay class.

    """

    def __new__(cls, phase, **kwargs):
        """
        Creates a new LocalGroupDelay instance from the given Phase.

        :param stft:   Phase instance (or anything a Phase can be instantiated
                       from)

        If no Phase instance was given, one is instantiated and these arguments
        are passed:

        :param kwargs: keyword arguments passed to Phase

        """
        #
        if not isinstance(stft, Phase):
            # try to instantiate a Phase object
            phase = Phase(phase, circular_shift=True, **kwargs)
        if not phase.stft.circular_shift:
            import warnings
            warnings.warn("`circular_shift` of the STFT must be set to 'True' "
                          "for correct local group delay")
        # unwrap phase
        data = np.unwrap(phase)
        # local group delay is the derivative over frequency
        data[:, :-1] -= data[:, 1:]
        # set the highest frequency to 0
        data[:, -1] = 0
        # cast as LocalGroupDelay
        obj = np.asarray(data).view(cls)
        # save additional attributes
        obj.phase = phase
        obj.stft = phase.stft
        obj.frames = phase.stft.frames
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here, also needed for views
        self.phase = getattr(obj, 'phase', None)
        self.stft = getattr(obj, 'stft', None)
        self.frames = getattr(obj, 'frames', None)


LGD = LocalGroupDelay
