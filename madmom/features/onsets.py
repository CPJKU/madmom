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

from madmom.processors import Processor
from madmom.audio.signal import smooth as smooth_signal
from madmom.audio.spectrogram import SpectrogramDifference


EPSILON = 1e-6


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
    return diff_spec


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
    # bins by the their respective "number" (starting at low frequencies)
    return np.mean(spectrogram * np.arange(spectrogram.num_bins), axis=1)


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
    # if the diff of a spectrogram is given, do not calculate the diff twice
    if not isinstance(spectrogram, SpectrogramDifference):
        spectrogram = spectrogram.diff(diff_frames=diff_frames,
                                       positive_diffs=True)
    # Spectral diff is the sum of all squared positive 1st order differences
    return np.sum(spectrogram ** 2, axis=1)


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
    # if the diff of a spectrogram is given, do not calculate the diff twice
    if not isinstance(spectrogram, SpectrogramDifference):
        spectrogram = spectrogram.diff(diff_frames=diff_frames,
                                       positive_diffs=True)
    # Spectral flux is the sum of all positive 1st order differences
    return np.sum(spectrogram, axis=1)


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
    # if the diff of a spectrogram is given, do not calculate the diff twice
    if not isinstance(spectrogram, SpectrogramDifference):
        spectrogram = spectrogram.diff(diff_frames=diff_frames,
                                       diff_max_bins=diff_max_bins,
                                       positive_diffs=True)
    # SuperFlux is the sum of all positive 1st order max. filtered differences
    return np.sum(spectrogram, axis=1)


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
    return np.sum(spectrogram.diff(diff_frames=diff_frames,
                                   diff_max_bins=diff_max_bins,
                                   positive_diffs=True) * mask, axis=1)


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
    return np.mean(np.log(1 + mkl), axis=1)


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
    return wrap_to_pi(pd)


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
    # take the mean of the absolute changes in instantaneous frequency
    return np.mean(np.abs(_phase_deviation(spectrogram.stft.phase())), axis=1)


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
    return np.mean(np.abs(_phase_deviation(phase) * spectrogram), axis=1)


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
    return weighted_phase_deviation(spectrogram) / norm


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
    return cd


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
    return np.sum(np.abs(_complex_domain(spectrogram)), axis=1)


def rectified_complex_domain(spectrogram, diff_frames=None,):
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
    return np.sum(np.abs(rcd), axis=1)


class SpectralOnsetProcessor(Processor):
    """
    The SpectralOnsetProcessor class implements most of the common onset
    detection functions based on the magnitude or phase information of a
    spectrogram.

    Parameters
    ----------
    onset_method : str, optional
        Onset detection function. See `METHODS` for possible values.

    """
    FRAME_SIZE = 2048
    FPS = 200
    ONLINE = False
    MAX_BINS = 3
    TEMPORAL_FILTER = 0.015
    TEMPORAL_ORIGIN = 0

    METHODS = ['superflux', 'complex_flux', 'high_frequency_content',
               'spectral_diff', 'spectral_flux', 'modified_kullback_leibler',
               'phase_deviation', 'weighted_phase_deviation',
               'normalized_weighted_phase_deviation', 'complex_domain',
               'rectified_complex_domain']

    def __init__(self, onset_method='superflux', **kwargs):
        # pylint: disable=unused-argument
        self.method = onset_method

    def process(self, spectrogram):
        """
        Detect the onsets in the given activation function.

        Parameters
        ----------
        spectrogram : :class:`Spectrogram` instance
            :class:`Spectrogram` instance.

        Returns
        -------
        odf : numpy array
            Onset detection function.

        """
        return globals()[self.method](spectrogram)

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
    smooth : int or numpy array
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
    if smooth not in (None, 0):
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
    This class implements the onset peak-picking functionality which can be
    used universally. It transparently converts the chosen values from seconds
    to frames.

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
        # TODO: make this an IOProcessor by defining input/output processings
        #       super(PeakPicking, self).__init__(peak_picking, write_events)
        #       adjust some params for online mode?
        if online:
            smooth = 0
            post_avg = 0
            post_max = 0
        self.threshold = threshold
        self.smooth = smooth
        self.pre_avg = pre_avg
        self.post_avg = post_avg
        self.pre_max = pre_max
        self.post_max = post_max
        self.combine = combine
        self.delay = delay
        self.fps = fps

    def process(self, activations):
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
        smooth = int(round(self.fps * self.smooth))
        pre_avg = int(round(self.fps * self.pre_avg))
        post_avg = int(round(self.fps * self.post_avg))
        pre_max = int(round(self.fps * self.pre_max))
        post_max = int(round(self.fps * self.post_max))
        # detect the peaks (function returns int indices)
        detections = peak_picking(activations, self.threshold, smooth,
                                  pre_avg, post_avg, pre_max, post_max)

        if isinstance(detections, tuple):
            # 2D detections (i.e. notes)
            onsets = detections[0].astype(np.float) / self.fps
            midi_notes = detections[1] + 21
            # shift if necessary
            if self.delay != 0:
                onsets += self.delay
            # combine multiple notes
            if self.combine > 0:
                detections = []
                # iterate over each detected note separately
                for note in np.unique(midi_notes):
                    # get all note detections
                    note_onsets = onsets[midi_notes == note]
                    # always use the first note
                    detections.append((note_onsets[0], note))
                    # filter all notes which occur within `combine` seconds
                    combined_note_onsets = note_onsets[1:][
                        np.diff(note_onsets) > self.combine]
                    # zip the onsets with the MIDI note number and add them to
                    # the list of detections
                    notes = zip(combined_note_onsets,
                                [note] * len(combined_note_onsets))
                    detections.extend(list(notes))
            else:
                # just zip all detected notes
                detections = list(zip(onsets, midi_notes))
            # sort the detections and save as numpy array
            detections = np.asarray(sorted(detections))
        else:
            # 1D detections (i.e. onsets)
            # convert detections to a list of timestamps
            detections = detections.astype(np.float) / self.fps
            # shift if necessary
            if self.delay != 0:
                detections += self.delay
            # always use the first detection and all others if none was
            # reported within the last `combine` seconds
            if detections.size > 1:
                # filter all detections which occur within `combine` seconds
                combined_detections = detections[1:][np.diff(detections) >
                                                     self.combine]
                # add them after the first detection
                detections = np.append(detections[0], combined_detections)
            else:
                detections = detections
        # return the detections
        return detections

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
