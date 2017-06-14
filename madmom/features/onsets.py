# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains onset detection related functionality.

"""

from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.ndimage.filters import maximum_filter

from ..processors import (Processor, OnlineProcessor, SequentialProcessor,
                          ParallelProcessor, BufferProcessor)
from ..audio.signal import smooth as smooth_signal
from ..utils import combine_events

EPSILON = np.spacing(1)


# onset detection helper functions
def wrap_to_pi(phase):
    """
    Wrap the phase information to the range -π...π.

    Parameters
    ----------
    phase : numpy array
        Phase of the STFT.

    Returns
    -------
    wrapped_phase : numpy array
        Wrapped phase.

    """
    return np.mod(phase + np.pi, 2.0 * np.pi) - np.pi


def correlation_diff(spec, diff_frames=1, pos=False, diff_bins=1):
    """
    Calculates the difference of the magnitude spectrogram relative to the
    N-th previous frame shifted in frequency to achieve the highest
    correlation between these two frames.

    Parameters
    ----------
    spec : numpy array
        Magnitude spectrogram.
    diff_frames : int, optional
        Calculate the difference to the `diff_frames`-th previous frame.
    pos : bool, optional
        Keep only positive values.
    diff_bins : int, optional
        Maximum number of bins shifted for correlation calculation.

    Returns
    -------
    correlation_diff : numpy array
        (Positive) magnitude spectrogram differences.

    Notes
    -----
    This function is only because of completeness, it is not intended to be
    actually used, since it is extremely slow. Please consider the superflux()
    function, since if performs equally well but much faster.

    """
    # init diff matrix
    diff_spec = np.zeros_like(spec)
    if diff_frames < 1:
        raise ValueError("number of `diff_frames` must be >= 1")
    # calculate the diff
    frames, bins = diff_spec.shape
    corr = np.zeros((frames, diff_bins * 2 + 1))
    for f in range(diff_frames, frames):
        # correlate the frame with the previous one
        # resulting size = bins * 2 - 1
        c = np.correlate(spec[f], spec[f - diff_frames], mode='full')
        # save the middle part
        centre = len(c) / 2
        corr[f] = c[centre - diff_bins: centre + diff_bins + 1]
        # shift the frame for difference calculation according to the
        # highest peak in correlation
        bin_offset = diff_bins - np.argmax(corr[f])
        bin_start = diff_bins + bin_offset
        bin_stop = bins - 2 * diff_bins + bin_start
        diff_spec[f, diff_bins:-diff_bins] = spec[f, diff_bins:-diff_bins] - \
            spec[f - diff_frames, bin_start:bin_stop]
    # keep only positive values
    if pos:
        np.maximum(diff_spec, 0, diff_spec)
    return np.asarray(diff_spec)


# onset detection functions pluggable into SpectralOnsetDetection
# Note: all functions here expect a Spectrogram object as their sole argument
#       thus it is not enforced that the algorithm does exactly what it is
#       supposed to do, but new configurations can be built easily
def high_frequency_content(spectrogram):
    """
    High Frequency Content.

    Parameters
    ----------
    spectrogram : :class:`Spectrogram` instance
        Spectrogram instance.

    Returns
    -------
    high_frequency_content : numpy array
        High frequency content onset detection function.

    References
    ----------
    .. [1] Paul Masri,
           "Computer Modeling of Sound for Transformation and Synthesis of
           Musical Signals",
           PhD thesis, University of Bristol, 1996.

    """
    # HFC emphasizes high frequencies by weighting the magnitude spectrogram
    # bins by their respective "number" (starting at low frequencies)
    hfc = spectrogram * np.arange(spectrogram.num_bins)
    return np.asarray(np.mean(hfc, axis=1))


def spectral_diff(spectrogram, diff_frames=None):
    """
    Spectral Diff.

    Parameters
    ----------
    spectrogram : :class:`Spectrogram` instance
        Spectrogram instance.
    diff_frames : int, optional
        Number of frames to calculate the diff to.

    Returns
    -------
    spectral_diff : numpy array
        Spectral diff onset detection function.

    References
    ----------
    .. [1] Chris Duxbury, Mark Sandler and Matthew Davis,
           "A hybrid approach to musical note onset detection",
           Proceedings of the 5th International Conference on Digital Audio
           Effects (DAFx), 2002.

    """
    from madmom.audio.spectrogram import SpectrogramDifference
    # if the diff of a spectrogram is given, do not calculate the diff twice
    if not isinstance(spectrogram, SpectrogramDifference):
        spectrogram = spectrogram.diff(diff_frames=diff_frames,
                                       positive_diffs=True)
    # Spectral diff is the sum of all squared positive 1st order differences
    return np.asarray(np.sum(spectrogram ** 2, axis=1))


def spectral_flux(spectrogram, diff_frames=None):
    """
    Spectral Flux.

    Parameters
    ----------
    spectrogram : :class:`Spectrogram` instance
        Spectrogram instance.
    diff_frames : int, optional
        Number of frames to calculate the diff to.

    Returns
    -------
    spectral_flux : numpy array
        Spectral flux onset detection function.

    References
    ----------
    .. [1] Paul Masri,
           "Computer Modeling of Sound for Transformation and Synthesis of
           Musical Signals",
           PhD thesis, University of Bristol, 1996.

    """
    from madmom.audio.spectrogram import SpectrogramDifference
    # if the diff of a spectrogram is given, do not calculate the diff twice
    if not isinstance(spectrogram, SpectrogramDifference):
        spectrogram = spectrogram.diff(diff_frames=diff_frames,
                                       positive_diffs=True)
    # Spectral flux is the sum of all positive 1st order differences
    return np.asarray(np.sum(spectrogram, axis=1))


def superflux(spectrogram, diff_frames=None, diff_max_bins=3):
    """
    SuperFlux method with a maximum filter vibrato suppression stage.

    Calculates the difference of bin k of the magnitude spectrogram relative to
    the N-th previous frame with the maximum filtered spectrogram.

    Parameters
    ----------
    spectrogram : :class:`Spectrogram` instance
        Spectrogram instance.
    diff_frames : int, optional
        Number of frames to calculate the diff to.
    diff_max_bins : int, optional
        Number of bins used for maximum filter.

    Returns
    -------
    superflux : numpy array
        SuperFlux onset detection function.

    Notes
    -----
    This method works only properly, if the spectrogram is filtered with a
    filterbank of the right frequency spacing. Filter banks with 24 bands per
    octave (i.e. quarter-tone resolution) usually yield good results. With
    `max_bins` = 3, the maximum of the bins k-1, k, k+1 of the frame
    `diff_frames` to the left is used for the calculation of the difference.

    References
    ----------
    .. [1] Sebastian Böck and Gerhard Widmer,
           "Maximum Filter Vibrato Suppression for Onset Detection",
           Proceedings of the 16th International Conference on Digital Audio
           Effects (DAFx), 2013.

    """
    from madmom.audio.spectrogram import SpectrogramDifference
    # if the diff of a spectrogram is given, do not calculate the diff twice
    if not isinstance(spectrogram, SpectrogramDifference):
        spectrogram = spectrogram.diff(diff_frames=diff_frames,
                                       diff_max_bins=diff_max_bins,
                                       positive_diffs=True)
    # SuperFlux is the sum of all positive 1st order max. filtered differences
    return np.asarray(np.sum(spectrogram, axis=1))


# TODO: should this be its own class so that we can set the filter
#       sizes in seconds instead of frames?
def complex_flux(spectrogram, diff_frames=None, diff_max_bins=3,
                 temporal_filter=3, temporal_origin=0):
    """
    ComplexFlux.

    ComplexFlux is based on the SuperFlux, but adds an additional local group
    delay based tremolo suppression.

    Parameters
    ----------
    spectrogram : :class:`Spectrogram` instance
        :class:`Spectrogram` instance.
    diff_frames : int, optional
        Number of frames to calculate the diff to.
    diff_max_bins : int, optional
        Number of bins used for maximum filter.
    temporal_filter : int, optional
        Temporal maximum filtering of the local group delay [frames].
    temporal_origin : int, optional
        Origin of the temporal maximum filter.

    Returns
    -------
    complex_flux : numpy array
        ComplexFlux onset detection function.

    References
    ----------
    .. [1] Sebastian Böck and Gerhard Widmer,
           "Local group delay based vibrato and tremolo suppression for onset
           detection",
           Proceedings of the 14th International Society for Music Information
           Retrieval Conference (ISMIR), 2013.

    """
    # create a mask based on the local group delay information
    from scipy.ndimage import maximum_filter, minimum_filter
    # take only absolute values of the local group delay and normalize them
    lgd = np.abs(spectrogram.stft.phase().lgd()) / np.pi
    # maximum filter along the temporal axis
    # TODO: use HPSS instead of simple temporal filtering
    if temporal_filter > 0:
        lgd = maximum_filter(lgd, size=[temporal_filter, 1],
                             origin=temporal_origin)
    # lgd = uniform_filter(lgd, size=[1, 3])  # better for percussive onsets
    # create the weighting mask
    try:
        # if the magnitude spectrogram was filtered, use the minimum local
        # group delay value of each filterbank (expanded by one frequency
        # bin in both directions) as the mask
        mask = np.zeros_like(spectrogram)
        num_bins = lgd.shape[1]
        for b in range(mask.shape[1]):
            # determine the corner bins for the mask
            corner_bins = np.nonzero(spectrogram.filterbank[:, b])[0]
            # always expand to the next neighbour
            start_bin = corner_bins[0] - 1
            stop_bin = corner_bins[-1] + 2
            # constrain the range
            if start_bin < 0:
                start_bin = 0
            if stop_bin > num_bins:
                stop_bin = num_bins
            # set mask
            mask[:, b] = np.amin(lgd[:, start_bin: stop_bin], axis=1)
    except AttributeError:
        # if the spectrogram is not filtered, use a simple minimum filter
        # covering only the current bin and its neighbours
        mask = minimum_filter(lgd, size=[1, 3])
    # sum all positive 1st order max. filtered and weighted differences
    diff = spectrogram.diff(diff_frames=diff_frames,
                            diff_max_bins=diff_max_bins,
                            positive_diffs=True)
    return np.asarray(np.sum(diff * mask, axis=1))


def modified_kullback_leibler(spectrogram, diff_frames=1, epsilon=EPSILON):
    """
    Modified Kullback-Leibler.

    Parameters
    ----------
    spectrogram : :class:`Spectrogram` instance
        :class:`Spectrogram` instance.
    diff_frames : int, optional
        Number of frames to calculate the diff to.
    epsilon : float, optional
        Add `epsilon` to the `spectrogram` avoid division by 0.

    Returns
    -------
    modified_kullback_leibler : numpy array
         MKL onset detection function.

    Notes
    -----
    The implementation presented in [1]_ is used instead of the original work
    presented in [2]_.

    References
    ----------
    .. [1] Paul Brossier,
           "Automatic Annotation of Musical Audio for Interactive
           Applications",
           PhD thesis, Queen Mary University of London, 2006.
    .. [2] Stephen Hainsworth and Malcolm Macleod,
           "Onset Detection in Musical Audio Signals",
           Proceedings of the International Computer Music Conference (ICMC),
           2003.

    """
    if epsilon <= 0:
        raise ValueError("a positive value must be added before division")
    mkl = np.zeros_like(spectrogram)
    mkl[diff_frames:] = (spectrogram[diff_frames:] /
                         (spectrogram[:-diff_frames] + epsilon))
    # note: the original MKL uses sum instead of mean,
    # but the range of mean is much more suitable
    return np.asarray(np.mean(np.log(1 + mkl), axis=1))


def _phase_deviation(phase):
    """
    Helper function used by phase_deviation() & weighted_phase_deviation().

    Parameters
    ----------
    phase : numpy array
        Phase of the STFT.

    Returns
    -------
    numpy array
        Phase deviation.

    """
    pd = np.zeros_like(phase)
    # instantaneous frequency is given by the first difference
    # ψ′(n, k) = ψ(n, k) − ψ(n − 1, k)
    # change in instantaneous frequency is given by the second order difference
    # ψ′′(n, k) = ψ′(n, k) − ψ′(n − 1, k)
    pd[2:] = phase[2:] - 2 * phase[1:-1] + phase[:-2]
    # map to the range -pi..pi
    return np.asarray(wrap_to_pi(pd))


def phase_deviation(spectrogram):
    """
    Phase Deviation.

    Parameters
    ----------
    spectrogram : :class:`Spectrogram` instance
        :class:`Spectrogram` instance.

    Returns
    -------
    phase_deviation : numpy array
        Phase deviation onset detection function.

    References
    ----------
    .. [1] Juan Pablo Bello, Chris Duxbury, Matthew Davies and Mark Sandler,
           "On the use of phase and energy for musical onset detection in the
           complex domain",
           IEEE Signal Processing Letters, Volume 11, Number 6, 2004.

    """
    # absolute phase changes in instantaneous frequency
    pd = np.abs(_phase_deviation(spectrogram.stft.phase()))
    return np.asarray(np.mean(pd, axis=1))


def weighted_phase_deviation(spectrogram):
    """
    Weighted Phase Deviation.

    Parameters
    ----------
    spectrogram : :class:`Spectrogram` instance
        :class:`Spectrogram` instance.

    Returns
    -------
    weighted_phase_deviation : numpy array
        Weighted phase deviation onset detection function.

    References
    ----------
    .. [1] Simon Dixon,
           "Onset Detection Revisited",
           Proceedings of the 9th International Conference on Digital Audio
           Effects (DAFx), 2006.

    """
    # cache phase
    phase = spectrogram.stft.phase()
    # make sure the spectrogram is not filtered before
    if np.shape(phase) != np.shape(spectrogram):
        raise ValueError('spectrogram and phase must be of same shape')
    # weighted_phase_deviation = spectrogram * phase_deviation
    wpd = np.abs(_phase_deviation(phase) * spectrogram)
    return np.asarray(np.mean(wpd, axis=1))


def normalized_weighted_phase_deviation(spectrogram, epsilon=EPSILON):
    """
    Normalized Weighted Phase Deviation.

    Parameters
    ----------
    spectrogram : :class:`Spectrogram` instance
        :class:`Spectrogram` instance.
    epsilon : float, optional
        Add `epsilon` to the `spectrogram` avoid division by 0.

    Returns
    -------
    normalized_weighted_phase_deviation : numpy array
        Normalized weighted phase deviation onset detection function.

    References
    ----------
    .. [1] Simon Dixon,
           "Onset Detection Revisited",
           Proceedings of the 9th International Conference on Digital Audio
           Effects (DAFx), 2006.

    """
    if epsilon <= 0:
        raise ValueError("a positive value must be added before division")
    # normalize WPD by the sum of the spectrogram
    # (add a small epsilon so that we don't divide by 0)
    norm = np.add(np.mean(spectrogram, axis=1), epsilon)
    return np.asarray(weighted_phase_deviation(spectrogram) / norm)


def _complex_domain(spectrogram):
    """
    Helper method used by complex_domain() & rectified_complex_domain().

    Parameters
    ----------
    spectrogram : :class:`Spectrogram` instance
        :class:`Spectrogram` instance.

    Returns
    -------
    numpy array
        Complex domain onset detection function.

    Notes
    -----
    We use the simple implementation presented in [1]_.

    References
    ----------
    .. [1] Simon Dixon,
           "Onset Detection Revisited",
           Proceedings of the 9th International Conference on Digital Audio
           Effects (DAFx), 2006.

    """
    # cache phase
    phase = spectrogram.stft.phase()
    # make sure the spectrogram is not filtered before
    if np.shape(phase) != np.shape(spectrogram):
        raise ValueError('spectrogram and phase must be of same shape')
    # expected spectrogram
    cd_target = np.zeros_like(phase)
    # assume constant phase change
    cd_target[1:] = 2 * phase[1:] - phase[:-1]
    # add magnitude
    cd_target = spectrogram * np.exp(1j * cd_target)
    # create complex spectrogram
    cd = spectrogram * np.exp(1j * phase)
    # subtract the target values
    cd[1:] -= cd_target[:-1]
    return np.asarray(cd)


def complex_domain(spectrogram):
    """
    Complex Domain.

    Parameters
    ----------
    spectrogram : :class:`Spectrogram` instance
        :class:`Spectrogram` instance.

    Returns
    -------
    complex_domain : numpy array
        Complex domain onset detection function.

    References
    ----------
    .. [1] Juan Pablo Bello, Chris Duxbury, Matthew Davies and Mark Sandler,
           "On the use of phase and energy for musical onset detection in the
           complex domain",
           IEEE Signal Processing Letters, Volume 11, Number 6, 2004.

    """
    # take the sum of the absolute changes
    return np.asarray(np.sum(np.abs(_complex_domain(spectrogram)), axis=1))


def rectified_complex_domain(spectrogram, diff_frames=None):
    """
    Rectified Complex Domain.

    Parameters
    ----------
    spectrogram : :class:`Spectrogram` instance
        :class:`Spectrogram` instance.
    diff_frames : int, optional
        Number of frames to calculate the diff to.

    Returns
    -------
    rectified_complex_domain : numpy array
        Rectified complex domain onset detection function.

    References
    ----------
    .. [1] Simon Dixon,
           "Onset Detection Revisited",
           Proceedings of the 9th International Conference on Digital Audio
           Effects (DAFx), 2006.

    """
    # rectified complex domain
    rcd = _complex_domain(spectrogram)
    # only keep values where the magnitude rises
    pos_diff = spectrogram.diff(diff_frames=diff_frames, positive_diffs=True)
    rcd *= pos_diff.astype(bool)
    # take the sum of the absolute changes
    return np.asarray(np.sum(np.abs(rcd), axis=1))


class SpectralOnsetProcessor(SequentialProcessor):
    """
    The SpectralOnsetProcessor class implements most of the common onset
    detection functions based on the magnitude or phase information of a
    spectrogram.

    Parameters
    ----------
    onset_method : str, optional
        Onset detection function. See `METHODS` for possible values.
    kwargs : dict, optional
        Keyword arguments passed to the pre-processing chain to obtain a
        spectral representation of the signal.

    Notes
    -----
    If the spectrogram should be filtered, the `filterbank` parameter must
    contain a valid Filterbank, if it should be scaled logarithmically, `log`
    must be set accordingly.

    References
    ----------
    .. [1] Paul Masri,
           "Computer Modeling of Sound for Transformation and Synthesis of
           Musical Signals",
           PhD thesis, University of Bristol, 1996.
    .. [2] Sebastian Böck and Gerhard Widmer,
           "Maximum Filter Vibrato Suppression for Onset Detection",
           Proceedings of the 16th International Conference on Digital Audio
           Effects (DAFx), 2013.

    Examples
    --------

    Create a SpectralOnsetProcessor and pass a file through the processor to
    obtain an onset detection function. Per default the spectral flux [1]_ is
    computed on a simple Spectrogram.

    >>> sodf = SpectralOnsetProcessor()
    >>> sodf  # doctest: +ELLIPSIS
    <madmom.features.onsets.SpectralOnsetProcessor object at 0x...>
    >>> sodf.processors[-1]  # doctest: +ELLIPSIS
    <function spectral_flux at 0x...>
    >>> sodf('tests/data/audio/sample.wav')
    ... # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    array([ 0. , 100.90121, ..., 26.30577, 20.94439], dtype=float32)

    The parameters passed to the signal pre-processing chain can be set when
    creating the SpectralOnsetProcessor. E.g. to obtain the SuperFlux [2]_
    onset detection function set these parameters:

    >>> from madmom.audio.filters import LogarithmicFilterbank
    >>> sodf = SpectralOnsetProcessor(onset_method='superflux', fps=200,
    ...                               filterbank=LogarithmicFilterbank,
    ...                               num_bands=24, log=np.log10)
    >>> sodf('tests/data/audio/sample.wav')
    ... # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    array([ 0. , 0. , 2.0868 , 1.02404, ..., 0.29888, 0.12122], dtype=float32)

    """

    METHODS = ['superflux', 'complex_flux', 'high_frequency_content',
               'spectral_diff', 'spectral_flux', 'modified_kullback_leibler',
               'phase_deviation', 'weighted_phase_deviation',
               'normalized_weighted_phase_deviation', 'complex_domain',
               'rectified_complex_domain']

    def __init__(self, onset_method='spectral_flux', **kwargs):
        import inspect
        from ..audio.signal import SignalProcessor, FramedSignalProcessor
        from ..audio.stft import ShortTimeFourierTransformProcessor
        from ..audio.spectrogram import (SpectrogramProcessor,
                                         FilteredSpectrogramProcessor,
                                         LogarithmicSpectrogramProcessor)
        # for certain methods we need to circular shift the signal before STFT
        if any(odf in onset_method for odf in ('phase', 'complex')):
            kwargs['circular_shift'] = True
        # always use mono signals
        kwargs['num_channels'] = 1
        # define processing chain
        sig = SignalProcessor(**kwargs)
        frames = FramedSignalProcessor(**kwargs)
        stft = ShortTimeFourierTransformProcessor(**kwargs)
        spec = SpectrogramProcessor(**kwargs)
        processors = [sig, frames, stft, spec]
        # filtering needed?
        if 'filterbank' in kwargs.keys() and kwargs['filterbank'] is not None:
            processors.append(FilteredSpectrogramProcessor(**kwargs))
        # scaling needed?
        if 'log' in kwargs.keys() and kwargs['log'] is not None:
            processors.append(LogarithmicSpectrogramProcessor(**kwargs))
        # odf function
        if not inspect.isfunction(onset_method):
            try:
                onset_method = globals()[onset_method]
            except KeyError:
                raise ValueError('%s not a valid onset detection function, '
                                 'choose %s.' % (onset_method, self.METHODS))
            processors.append(onset_method)
        # instantiate a SequentialProcessor
        super(SpectralOnsetProcessor, self).__init__(processors)

    @classmethod
    def add_arguments(cls, parser, onset_method=None):
        """
        Add spectral onset detection arguments to an existing parser.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        onset_method : str, optional
            Default onset detection method.

        Returns
        -------
        parser_group : argparse argument group
            Spectral onset detection argument parser group.

        """
        # add onset detection method arguments to the existing parser
        g = parser.add_argument_group('spectral onset detection arguments')
        if onset_method is not None:
            g.add_argument('--odf', dest='onset_method',
                           default=onset_method, choices=cls.METHODS,
                           help='use this onset detection function '
                                '[default=%(default)s]')
        # return the argument group so it can be modified if needed
        return g


# classes for detecting onsets with NNs
class RNNOnsetProcessor(SequentialProcessor):
    """
    Processor to get a onset activation function from multiple RNNs.

    Parameters
    ----------
    online : bool, optional
        Choose networks suitable for online onset detection, i.e. use
        unidirectional RNNs.

    Notes
    -----
    This class uses either uni- or bi-directional RNNs. Contrary to [1], it
    uses simple tanh units as in [2]. Also the input representations changed
    to use logarithmically filtered and scaled spectrograms.

    References
    ----------
    .. [1] "Universal Onset Detection with bidirectional Long Short-Term Memory
           Neural Networks"
           Florian Eyben, Sebastian Böck, Björn Schuller and Alex Graves.
           Proceedings of the 11th International Society for Music Information
           Retrieval Conference (ISMIR), 2010.
    .. [2] "Online Real-time Onset Detection with Recurrent Neural Networks"
           Sebastian Böck, Andreas Arzt, Florian Krebs and Markus Schedl.
           Proceedings of the 15th International Conference on Digital Audio
           Effects (DAFx), 2012.

    Examples
    --------
    Create a RNNOnsetProcessor and pass a file through the processor to obtain
    an onset detection function (sampled with 100 frames per second).

    >>> proc = RNNOnsetProcessor()
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.onsets.RNNOnsetProcessor object at 0x...>
    >>> proc('tests/data/audio/sample.wav') # doctest: +ELLIPSIS
    array([ 0.08313,  0.0024 ,  ...,  0.00205,  0.00527], dtype=float32)

    """

    def __init__(self, **kwargs):
        # pylint: disable=unused-argument
        from ..audio.signal import SignalProcessor, FramedSignalProcessor
        from ..audio.stft import ShortTimeFourierTransformProcessor
        from ..audio.spectrogram import (
            FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
            SpectrogramDifferenceProcessor)
        from ..models import ONSETS_RNN, ONSETS_BRNN
        from ..ml.nn import NeuralNetworkEnsemble

        # choose the appropriate models and set frame sizes accordingly
        if kwargs.get('online'):
            nn_files = ONSETS_RNN
            frame_sizes = [512, 1024, 2048]
        else:
            nn_files = ONSETS_BRNN
            frame_sizes = [1024, 2048, 4096]

        # define pre-processing chain
        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        # process the multi-resolution spec & diff in parallel
        multi = ParallelProcessor([])
        for frame_size in frame_sizes:
            # pass **kwargs in order to be able to process in online mode
            frames = FramedSignalProcessor(frame_size=frame_size, **kwargs)
            stft = ShortTimeFourierTransformProcessor()  # caching FFT window
            filt = FilteredSpectrogramProcessor(
                num_bands=6, fmin=30, fmax=17000, norm_filters=True)
            spec = LogarithmicSpectrogramProcessor(mul=5, add=1)
            diff = SpectrogramDifferenceProcessor(
                diff_ratio=0.25, positive_diffs=True, stack_diffs=np.hstack)
            # process each frame size with spec and diff sequentially
            multi.append(SequentialProcessor((frames, stft, filt, spec, diff)))
        # stack the features and processes everything sequentially
        pre_processor = SequentialProcessor((sig, multi, np.hstack))

        # process the pre-processed signal with a NN ensemble
        nn = NeuralNetworkEnsemble.load(nn_files, **kwargs)

        # instantiate a SequentialProcessor
        super(RNNOnsetProcessor, self).__init__((pre_processor, nn))


# must be a top-level function to be pickle-able
def _cnn_onset_processor_pad(data):
    """Pad the data by repeating the first and last frame 7 times."""
    pad_start = np.repeat(data[:1], 7, axis=0)
    pad_stop = np.repeat(data[-1:], 7, axis=0)
    return np.concatenate((pad_start, data, pad_stop))


class CNNOnsetProcessor(SequentialProcessor):
    """
    Processor to get a onset activation function from a CNN.

    References
    ----------
    .. [1] "Musical Onset Detection with Convolutional Neural Networks"
           Jan Schlüter and Sebastian Böck.
           Proceedings of the 6th International Workshop on Machine Learning
           and Music, 2013.

    Notes
    -----
    The implementation follows as closely as possible the original one, but
    part of the signal pre-processing differs in minor aspects, so results can
    differ slightly, too.

    Examples
    --------
    Create a CNNOnsetProcessor and pass a file through the processor to obtain
    an onset detection function (sampled with 100 frames per second).

    >>> proc = CNNOnsetProcessor()
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.onsets.CNNOnsetProcessor object at 0x...>
    >>> proc('tests/data/audio/sample.wav')  # doctest: +ELLIPSIS
    array([ 0.05369,  0.04205,  ...,  0.00024,  0.00014], dtype=float32)

    """

    def __init__(self, **kwargs):
        # pylint: disable=unused-argument
        from ..audio.signal import SignalProcessor, FramedSignalProcessor
        from ..audio.stft import ShortTimeFourierTransformProcessor
        from ..audio.filters import MelFilterbank
        from ..audio.spectrogram import (FilteredSpectrogramProcessor,
                                         LogarithmicSpectrogramProcessor)
        from ..models import ONSETS_CNN
        from ..ml.nn import NeuralNetwork

        # define pre-processing chain
        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        # process the multi-resolution spec in parallel
        multi = ParallelProcessor([])
        for frame_size in [2048, 1024, 4096]:
            frames = FramedSignalProcessor(frame_size=frame_size, fps=100)
            stft = ShortTimeFourierTransformProcessor()  # caching FFT window
            filt = FilteredSpectrogramProcessor(
                filterbank=MelFilterbank, num_bands=80, fmin=27.5, fmax=16000,
                norm_filters=True, unique_filters=False)
            spec = LogarithmicSpectrogramProcessor(log=np.log, add=EPSILON)
            # process each frame size with spec and diff sequentially
            multi.append(SequentialProcessor((frames, stft, filt, spec)))
        # stack the features (in depth) and pad at beginning and end
        stack = np.dstack
        pad = _cnn_onset_processor_pad
        # pre-processes everything sequentially
        pre_processor = SequentialProcessor((sig, multi, stack, pad))

        # process the pre-processed signal with a NN ensemble
        nn = NeuralNetwork.load(ONSETS_CNN[0])

        # instantiate a SequentialProcessor
        super(CNNOnsetProcessor, self).__init__((pre_processor, nn))


# universal peak-picking method
def peak_picking(activations, threshold, smooth=None, pre_avg=0, post_avg=0,
                 pre_max=1, post_max=1):
    """
    Perform thresholding and peak-picking on the given activation function.

    Parameters
    ----------
    activations : numpy array
        Activation function.
    threshold : float
        Threshold for peak-picking
    smooth : int or numpy array, optional
        Smooth the activation function with the kernel (size).
    pre_avg : int, optional
        Use `pre_avg` frames past information for moving average.
    post_avg : int, optional
        Use `post_avg` frames future information for moving average.
    pre_max : int, optional
        Use `pre_max` frames past information for moving maximum.
    post_max : int, optional
        Use `post_max` frames future information for moving maximum.

    Returns
    -------
    peak_idx : numpy array
        Indices of the detected peaks.

    See Also
    --------
    :func:`smooth`

    Notes
    -----
    If no moving average is needed (e.g. the activations are independent of
    the signal's level as for neural network activations), set `pre_avg` and
    `post_avg` to 0.
    For peak picking of local maxima, set `pre_max` and  `post_max` to 1.
    For online peak picking, set all `post_` parameters to 0.

    References
    ----------
    .. [1] Sebastian Böck, Florian Krebs and Markus Schedl,
           "Evaluating the Online Capabilities of Onset Detection Methods",
           Proceedings of the 13th International Society for Music Information
           Retrieval Conference (ISMIR), 2012.

    """
    # smooth activations
    activations = smooth_signal(activations, smooth)
    # compute a moving average
    avg_length = pre_avg + post_avg + 1
    if avg_length > 1:
        # TODO: make the averaging function exchangeable (mean/median/etc.)
        avg_origin = int(np.floor((pre_avg - post_avg) / 2))
        if activations.ndim == 1:
            filter_size = avg_length
        elif activations.ndim == 2:
            filter_size = [avg_length, 1]
        else:
            raise ValueError('`activations` must be either 1D or 2D')
        mov_avg = uniform_filter(activations, filter_size, mode='constant',
                                 origin=avg_origin)
    else:
        # do not use a moving average
        mov_avg = 0
    # detections are those activations above the moving average + the threshold
    detections = activations * (activations >= mov_avg + threshold)
    # peak-picking
    max_length = pre_max + post_max + 1
    if max_length > 1:
        # compute a moving maximum
        max_origin = int(np.floor((pre_max - post_max) / 2))
        if activations.ndim == 1:
            filter_size = max_length
        elif activations.ndim == 2:
            filter_size = [max_length, 1]
        else:
            raise ValueError('`activations` must be either 1D or 2D')
        mov_max = maximum_filter(detections, filter_size, mode='constant',
                                 origin=max_origin)
        # detections are peak positions
        detections *= (detections == mov_max)
    # return indices
    if activations.ndim == 1:
        return np.nonzero(detections)[0]
    elif activations.ndim == 2:
        return np.nonzero(detections)
    else:
        raise ValueError('`activations` must be either 1D or 2D')


class PeakPickingProcessor(Processor):
    """
    Deprecated as of version 0.15. Will be removed in version 0.16. Use either
    :class:`OnsetPeakPickingProcessor` or :class:`NotePeakPickingProcessor`
    instead.

    """

    def __init__(self, **kwargs):
        # pylint: disable=unused-argument
        self.kwargs = kwargs

    def process(self, activations, **kwargs):
        """
        Detect the peaks in the given activation function.

        Parameters
        ----------
        activations : numpy array
            Onset activation function.

        Returns
        -------
        peaks : numpy array
            Detected onsets [seconds[, frequency bin]].

        """
        import warnings
        if activations.ndim == 1:
            warnings.warn('`PeakPickingProcessor` is deprecated as of version '
                          '0.15 and will be removed in version 0.16. Use '
                          '`OnsetPeakPickingProcessor` instead.')
            ppp = OnsetPeakPickingProcessor(**self.kwargs)
            return ppp(activations, **kwargs)
        elif activations.ndim == 2:
            warnings.warn('`PeakPickingProcessor` is deprecated as of version '
                          '0.15 and will be removed in version 0.16. Use '
                          '`NotePeakPickingProcessor` instead.')
            from .notes import NotePeakPickingProcessor
            ppp = NotePeakPickingProcessor(**self.kwargs)
            return ppp(activations, **kwargs)

    @staticmethod
    def add_arguments(parser, **kwargs):
        """
        Deprecated as of version 0.15. Will be removed in version 0.16. Use
        either :class:`OnsetPeakPickingProcessor` or
        :class:`NotePeakPickingProcessor` instead.

        """
        return OnsetPeakPickingProcessor.add_arguments(parser, **kwargs)


class OnsetPeakPickingProcessor(OnlineProcessor):
    """
    This class implements the onset peak-picking functionality.
    It transparently converts the chosen values from seconds to frames.

    Parameters
    ----------
    threshold : float
        Threshold for peak-picking.
    smooth : float, optional
        Smooth the activation function over `smooth` seconds.
    pre_avg : float, optional
        Use `pre_avg` seconds past information for moving average.
    post_avg : float, optional
        Use `post_avg` seconds future information for moving average.
    pre_max : float, optional
        Use `pre_max` seconds past information for moving maximum.
    post_max : float, optional
        Use `post_max` seconds future information for moving maximum.
    combine : float, optional
        Only report one onset within `combine` seconds.
    delay : float, optional
        Report the detected onsets `delay` seconds delayed.
    online : bool, optional
        Use online peak-picking, i.e. no future information.
    fps : float, optional
        Frames per second used for conversion of timings.

    Returns
    -------
    onsets : numpy array
        Detected onsets [seconds].

    Notes
    -----
    If no moving average is needed (e.g. the activations are independent of
    the signal's level as for neural network activations), `pre_avg` and
    `post_avg` should be set to 0.
    For peak picking of local maxima, set `pre_max` >= 1. / `fps` and
    `post_max` >= 1. / `fps`.
    For online peak picking, all `post_` parameters are set to 0.

    References
    ----------
    .. [1] Sebastian Böck, Florian Krebs and Markus Schedl,
           "Evaluating the Online Capabilities of Onset Detection Methods",
           Proceedings of the 13th International Society for Music Information
           Retrieval Conference (ISMIR), 2012.

    Examples
    --------
    Create a PeakPickingProcessor. The returned array represents the positions
    of the onsets in seconds, thus the expected sampling rate has to be given.

    >>> proc = OnsetPeakPickingProcessor(fps=100)
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.onsets.OnsetPeakPickingProcessor object at 0x...>

    Call this OnsetPeakPickingProcessor with the onset activation function from
    an RNNOnsetProcessor to obtain the onset positions.

    >>> act = RNNOnsetProcessor()('tests/data/audio/sample.wav')
    >>> proc(act)  # doctest: +ELLIPSIS
    array([ 0.09,  0.29,  0.45,  ...,  2.34,  2.49,  2.67])

    """
    FPS = 100
    THRESHOLD = 0.5  # binary threshold
    SMOOTH = 0.
    PRE_AVG = 0.
    POST_AVG = 0.
    PRE_MAX = 0.
    POST_MAX = 0.
    COMBINE = 0.03
    DELAY = 0.
    ONLINE = False

    def __init__(self, threshold=THRESHOLD, smooth=SMOOTH, pre_avg=PRE_AVG,
                 post_avg=POST_AVG, pre_max=PRE_MAX, post_max=POST_MAX,
                 combine=COMBINE, delay=DELAY, online=ONLINE, fps=FPS,
                 **kwargs):
        # pylint: disable=unused-argument
        # instantiate OnlineProcessor
        super(OnsetPeakPickingProcessor, self).__init__(online=online)
        if self.online:
            # set some parameters to 0 (i.e. no future information available)
            smooth = 0
            post_avg = 0
            post_max = 0
            # init buffer
            self.buffer = None
            self.counter = 0
            self.last_onset = None
        # save parameters
        self.threshold = threshold
        self.smooth = smooth
        self.pre_avg = pre_avg
        self.post_avg = post_avg
        self.pre_max = pre_max
        self.post_max = post_max
        self.combine = combine
        self.delay = delay
        self.fps = fps

    def reset(self):
        """Reset OnsetPeakPickingProcessor."""
        self.buffer = None
        self.counter = 0
        self.last_onset = None

    def process_sequence(self, activations, **kwargs):
        """
        Detect the onsets in the given activation function.

        Parameters
        ----------
        activations : numpy array
            Onset activation function.

        Returns
        -------
        onsets : numpy array
            Detected onsets [seconds].

        """
        # convert timing information to frames and set default values
        # TODO: use at least 1 frame if any of these values are > 0?
        timings = np.array([self.smooth, self.pre_avg, self.post_avg,
                            self.pre_max, self.post_max]) * self.fps
        timings = np.round(timings).astype(int)
        # detect the peaks (function returns int indices)
        onsets = peak_picking(activations, self.threshold, *timings)
        # convert to timestamps
        onsets = onsets.astype(np.float) / self.fps
        # shift if necessary
        if self.delay:
            onsets += self.delay
        # combine onsets
        if self.combine:
            onsets = combine_events(onsets, self.combine, 'left')
        # return the onsets
        return np.asarray(onsets)

    process_offline = process_sequence

    def process_online(self, activations, reset=True, **kwargs):
        """
        Detect the onsets in the given activation function.

        Parameters
        ----------
        activations : numpy array
            Onset activation function.

        Returns
        -------
        onsets : numpy array
            Detected onsets [seconds].

        """
        # buffer data
        if self.buffer is None or reset:
            # reset the processor
            self.reset()
            # put 0s in front (depending on conext given by pre_max
            init = np.zeros(int(np.round(self.pre_max * self.fps)))
            buffer = np.insert(activations, 0, init, axis=0)
            # offset the counter, because we buffer the activations
            self.counter = -len(init)
            # use the data for the buffer
            self.buffer = BufferProcessor(init=buffer)
        else:
            buffer = self.buffer(activations)
        # convert timing information to frames and set default values
        # TODO: use at least 1 frame if any of these values are > 0?
        timings = np.array([self.smooth, self.pre_avg, self.post_avg,
                            self.pre_max, self.post_max]) * self.fps
        timings = np.round(timings).astype(int)
        # detect the peaks (function returns int indices)
        peaks = peak_picking(buffer, self.threshold, *timings)
        # convert to onset timings
        onsets = (self.counter + peaks) / float(self.fps)
        # increase counter
        self.counter += len(activations)
        # shift if necessary
        if self.delay:
            raise ValueError('delay not supported yet in online mode')
        # report only if there was no onset within the last combine seconds
        if self.combine and onsets.any():
            # prepend the last onset to be able to combine them correctly
            start = 0
            if self.last_onset is not None:
                onsets = np.append(self.last_onset, onsets)
                start = 1
            # combine the onsets
            onsets = combine_events(onsets, self.combine, 'left')
            # use only if the last onsets differ
            if onsets[-1] != self.last_onset:
                self.last_onset = onsets[-1]
                # remove the first onset if we added it previously
                onsets = onsets[start:]
            else:
                # don't report an onset
                onsets = np.empty(0)
        # return the onsets
        return onsets

    @staticmethod
    def add_arguments(parser, threshold=THRESHOLD, smooth=None, pre_avg=None,
                      post_avg=None, pre_max=None, post_max=None,
                      combine=COMBINE, delay=DELAY):
        """
        Add onset peak-picking related arguments to an existing parser.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        threshold : float
            Threshold for peak-picking.
        smooth : float, optional
            Smooth the activation function over `smooth` seconds.
        pre_avg : float, optional
            Use `pre_avg` seconds past information for moving average.
        post_avg : float, optional
            Use `post_avg` seconds future information for moving average.
        pre_max : float, optional
            Use `pre_max` seconds past information for moving maximum.
        post_max : float, optional
            Use `post_max` seconds future information for moving maximum.
        combine : float, optional
            Only report one onset within `combine` seconds.
        delay : float, optional
            Report the detected onsets `delay` seconds delayed.

        Returns
        -------
        parser_group : argparse argument group
            Onset peak-picking argument parser group.

        Notes
        -----
        Parameters are included in the group only if they are not 'None'.

        """
        # add onset peak-picking related options to the existing parser
        g = parser.add_argument_group('peak-picking arguments')
        g.add_argument('-t', dest='threshold', action='store', type=float,
                       default=threshold,
                       help='detection threshold [default=%(default).2f]')
        if smooth is not None:
            g.add_argument('--smooth', action='store', type=float,
                           default=smooth,
                           help='smooth the activation function over N '
                                'seconds [default=%(default).2f]')
        if pre_avg is not None:
            g.add_argument('--pre_avg', action='store', type=float,
                           default=pre_avg,
                           help='build average over N previous seconds '
                                '[default=%(default).2f]')
        if post_avg is not None:
            g.add_argument('--post_avg', action='store', type=float,
                           default=post_avg, help='build average over N '
                           'following seconds [default=%(default).2f]')
        if pre_max is not None:
            g.add_argument('--pre_max', action='store', type=float,
                           default=pre_max,
                           help='search maximum over N previous seconds '
                                '[default=%(default).2f]')
        if post_max is not None:
            g.add_argument('--post_max', action='store', type=float,
                           default=post_max,
                           help='search maximum over N following seconds '
                                '[default=%(default).2f]')
        if combine is not None:
            g.add_argument('--combine', action='store', type=float,
                           default=combine,
                           help='combine events within N seconds '
                                '[default=%(default).2f]')
        if delay is not None:
            g.add_argument('--delay', action='store', type=float,
                           default=delay,
                           help='report the events N seconds delayed '
                                '[default=%(default)i]')
        # return the argument group so it can be modified if needed
        return g
