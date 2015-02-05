#!/usr/bin/env python
# encoding: utf-8
"""
This file contains onset detection related functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import glob

import numpy as np
from scipy.ndimage.filters import maximum_filter, uniform_filter

from .. import MODELS_PATH, SequentialProcessor, IOProcessor
from ..utils import write_events, files
from ..ml.rnn import RNNProcessor
from ..audio.signal import (SignalProcessor, FramedSignalProcessor,
                            smooth as smooth_signal)
from ..audio.spectrogram import SpectrogramProcessor, StackSpectrogramProcessor
from ..features import ActivationsProcessor

EPSILON = 1e-6


# onset detection helper functions
def wrap_to_pi(phase):
    """
    Wrap the phase information to the range -π...π.

    :param phase: phase spectrogram
    :return:      wrapped phase spectrogram

    """
    return np.mod(phase + np.pi, 2.0 * np.pi) - np.pi


def diff(spec, diff_frames=1, pos=False):
    """
    Calculates the difference of the magnitude spectrogram.

    :param spec:        the magnitude spectrogram
    :param diff_frames: calculate the difference to the N-th previous frame
    :param pos:         keep only positive values
    :return:            (positive) magnitude spectrogram differences

    """
    # init the matrix with 0s, the first N rows are 0 then
    # TODO: under some circumstances it might be helpful to init with the spec
    #       or use the frame at "real" index -N to calculate the diff to
    diff_spec = np.zeros_like(spec)
    if diff_frames < 1:
        raise ValueError("number of diff_frames must be >= 1")
    # calculate the diff
    diff_spec[diff_frames:] = spec[diff_frames:] - spec[:-diff_frames]
    # keep only positive values
    if pos:
        np.maximum(diff_spec, 0, diff_spec)
    return diff_spec


def correlation_diff(spec, diff_frames=1, pos=False, diff_bins=1):
    """
    Calculates the difference of the magnitude spectrogram relative to the
    N-th previous frame shifted in frequency to achieve the highest
    correlation between these two frames.

    :param spec:        the magnitude spectrogram
    :param diff_frames: calculate the difference to the N-th previous frame
    :param pos:         keep only positive values
    :param diff_bins:   maximum number of bins shifted for correlation
                        calculation
    :return:            (positive) magnitude spectrogram differences

    """
    # init diff matrix
    diff_spec = np.zeros_like(spec)
    if diff_frames < 1:
        raise ValueError("number of diff_frames must be >= 1")
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


# Onset Detection Functions
def high_frequency_content(spec):
    """
    High Frequency Content.

    :param spec: the magnitude spectrogram
    :return:     high frequency content onset detection function

    "Computer Modeling of Sound for Transformation and Synthesis of Musical
     Signals"
    Paul Masri
    PhD thesis, University of Bristol, 1996.

    """
    # HFC weights the magnitude spectrogram by the bin number,
    # thus emphasizing high frequencies
    return np.mean(spec * np.arange(spec.shape[1]), axis=1)


def spectral_diff(spec, diff_frames=1):
    """
    Spectral Diff.

    :param spec:        the magnitude spectrogram
    :param diff_frames: calculate the difference to the N-th previous frame
    :return:            spectral diff onset detection function

    "A hybrid approach to musical note onset detection"
    Chris Duxbury, Mark Sandler and Matthew Davis
    Proceedings of the 5th International Conference on Digital Audio Effects
    (DAFx-02), 2002.

    """
    # Spectral diff is the sum of all squared positive 1st order differences
    return np.sum(diff(spec, diff_frames=diff_frames, pos=True) ** 2, axis=1)


def spectral_flux(spec, diff_frames=1):
    """
    Spectral Flux.

    :param spec:        the magnitude spectrogram
    :param diff_frames: calculate the difference to the N-th previous frame
    :return:            spectral flux onset detection function

    "Computer Modeling of Sound for Transformation and Synthesis of Musical
     Signals"
    Paul Masri
    PhD thesis, University of Bristol, 1996.

    """
    # Spectral flux is the sum of all positive 1st order differences
    return np.sum(diff(spec, diff_frames=diff_frames, pos=True), axis=1)


def _superflux_diff_spec(spec, diff_frames=1, max_bins=3):
    """
    Internal function to calculate the difference spec used for SuperFlux

    :param spec:        the magnitude spectrogram
    :param diff_frames: calculate the difference to the N-th previous frame
    :param max_bins:    number of neighboring bins used for maximum filtering
    :return:            difference spectrogram used for SuperFlux

    """
    # init diff matrix
    diff_spec = np.zeros_like(spec)
    if diff_frames < 1:
        raise ValueError("number of diff_frames must be >= 1")
    # widen the spectrogram in frequency dimension by `max_bins`
    max_spec = maximum_filter(spec, size=[1, max_bins])
    # calculate the diff
    diff_spec[diff_frames:] = spec[diff_frames:] - max_spec[0:-diff_frames]
    # keep only positive values
    np.maximum(diff_spec, 0, diff_spec)
    # return diff spec
    return diff_spec


def superflux(spec, diff_frames=1, max_bins=3):
    """
    SuperFlux method with a maximum filter vibrato suppression stage.

    Calculates the difference of bin k of the magnitude spectrogram relative to
    the N-th previous frame with the maximum filtered spectrogram.

    :param spec:        the magnitude spectrogram
    :param diff_frames: calculate the difference to the N-th previous frame
    :param max_bins:    number of neighboring bins used for maximum filtering
    :return:            SuperFlux onset detection function

    "Maximum Filter Vibrato Suppression for Onset Detection"
    Sebastian Böck and Gerhard Widmer
    Proceedings of the 16th International Conference on Digital Audio Effects
    (DAFx-13), 2013.

    Note: this method works only properly, if the spectrogram is filtered with
          a filterbank of the right frequency spacing. Filter banks with 24
          bands per octave (i.e. quarter-tone resolution) usually yield good
          results. With `max_bins` = 3, the maximum of the bins k-1, k, k+1 of
          the frame `diff_frames` to the left is used for the calculation of
          the difference.

    """
    # SuperFlux is the sum of all positive 1st order max. filtered differences
    return np.sum(_superflux_diff_spec(spec, diff_frames, max_bins), axis=1)


def lgd_mask(spec, lgd, filterbank=None, temporal_filter=0, temporal_origin=0):
    """
    Calculates a weighting mask for the magnitude spectrogram based on the
    local group delay.

    :param spec:            the magnitude spectrogram
    :param lgd:             local group delay of the spectrogram
    :param filterbank:      filterbank used for dimensionality reduction of
                            the magnitude spectrogram
    :param temporal_filter: temporal maximum filtering of the local group delay
    :param temporal_origin: origin of the temporal maximum filter

    "Local group delay based vibrato and tremolo suppression for onset
     detection"
    Sebastian Böck and Gerhard Widmer.
    Proceedings of the 13th International Society for Music Information
    Retrieval Conference (ISMIR), 2013.

    """
    from scipy.ndimage import maximum_filter, minimum_filter
    # take only absolute values of the local group delay
    np.abs(lgd, out=lgd)

    # maximum filter along the temporal axis
    # TODO: use HPSS instead of simple temporal filtering
    if temporal_filter > 0:
        lgd = maximum_filter(lgd, size=[temporal_filter, 1],
                             origin=temporal_origin)
    # lgd = uniform_filter(lgd, size=[1, 3])  # better for percussive onsets

    # create the weighting mask
    if filterbank is not None:
        # if the magnitude spectrogram was filtered, use the minimum local
        # group delay value of each filterbank (expanded by one frequency
        # bin in both directions) as the mask
        mask = np.zeros_like(spec)
        num_bins = lgd.shape[1]
        for b in range(mask.shape[1]):
            # determine the corner bins for the mask
            corner_bins = np.nonzero(filterbank[:, b])[0]
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
    else:
        # if the spectrogram is not filtered, use a simple minimum filter
        # covering only the current bin and its neighbours
        mask = minimum_filter(lgd, size=[1, 3])
    # return the normalized mask
    return mask / np.pi


def complex_flux(spec, lgd, filterbank=None, diff_frames=1, max_bins=3,
                 temporal_filter=3, temporal_origin=0):
    """
    Complex Flux with a local group delay based tremolo suppression.

    Calculates the difference of bin k of the magnitude spectrogram relative
    to the N-th previous frame of the (maximum filtered) spectrogram.

    :param spec:            the magnitude spectrogram
    :param lgd:             local group delay of the spectrogram
    :param filterbank:      filterbank for dimensionality reduction of the spec
    :param diff_frames:     calculate the difference to the N-th previous frame
    :param max_bins:        number of neighbour bins used for maximum filtering
    :param temporal_filter: temporal maximum filtering of the local group delay
    :param temporal_origin: origin of the temporal maximum filter
    :return:                complex flux onset detection function

    "Local group delay based vibrato and tremolo suppression for onset
     detection"
    Sebastian Böck and Gerhard Widmer.
    Proceedings of the 13th International Society for Music Information
    Retrieval Conference (ISMIR), 2013.

    Note: If `max_bins` is set to any value > 1, the SuperFlux method is used
          to compute the differences of the magnitude spectrogram, otherwise
          the normal spectral flux is used.

    """
    # compute the difference spectrogram as in the SuperFlux algorithm
    diff_spec = _superflux_diff_spec(spec, diff_frames, max_bins)
    # create a mask based on the local group delay information
    mask = lgd_mask(spec, lgd, filterbank, temporal_filter, temporal_origin)
    # weight the differences with the mask
    diff_spec *= mask
    # sum all positive 1st order max. filtered and weighted differences
    return np.sum(diff_spec, axis=1)


def modified_kullback_leibler(spec, diff_frames=1, epsilon=EPSILON):
    """
    Modified Kullback-Leibler.

    :param spec:        the magnitude spectrogram
    :param diff_frames: calculate the difference to the N-th previous frame
    :param epsilon:     add epsilon to avoid division by 0
    :return:            MKL onset detection function

    Note: the implementation presented in:
    "Automatic Annotation of Musical Audio for Interactive Applications"
    Paul Brossier
    PhD thesis, Queen Mary University of London, 2006

    is used instead of the original work:
    "Onset Detection in Musical Audio Signals"
    Stephen Hainsworth and Malcolm Macleod
    Proceedings of the International Computer Music Conference (ICMC), 2003.

    """
    if epsilon <= 0:
        raise ValueError("a positive value must be added before division")
    mkl = np.zeros_like(spec)
    mkl[diff_frames:] = spec[diff_frames:] / (spec[:-diff_frames] + epsilon)
    # note: the original MKL uses sum instead of mean,
    # but the range of mean is much more suitable
    return np.mean(np.log(1 + mkl), axis=1)


def _phase_deviation(phase):
    """
    Helper method used by phase_deviation() & weighted_phase_deviation().

    :param phase: the phase spectrogram
    :return:      phase deviation

    """
    pd = np.zeros_like(phase)
    # instantaneous frequency is given by the first difference
    # ψ′(n, k) = ψ(n, k) − ψ(n − 1, k)
    # change in instantaneous frequency is given by the second order difference
    # ψ′′(n, k) = ψ′(n, k) − ψ′(n − 1, k)
    pd[2:] = phase[2:] - 2 * phase[1:-1] + phase[:-2]
    # map to the range -pi..pi
    return wrap_to_pi(pd)


def phase_deviation(phase):
    """
    Phase Deviation.

    :param phase: the phase spectrogram
    :return:      phase deviation onset detection function

    "On the use of phase and energy for musical onset detection in the complex
     domain"
    Juan Pablo Bello, Chris Duxbury, Matthew Davies and Mark Sandler
    IEEE Signal Processing Letters, Volume 11, Number 6, 2004.

    """
    # take the mean of the absolute changes in instantaneous frequency
    return np.mean(np.abs(_phase_deviation(phase)), axis=1)


def weighted_phase_deviation(spec, phase):
    """
    Weighted Phase Deviation.

    :param spec:  the magnitude spectrogram
    :param phase: the phase spectrogram
    :return:      weighted phase deviation onset detection function

    "Onset Detection Revisited"
    Simon Dixon
    Proceedings of the 9th International Conference on Digital Audio Effects
    (DAFx), 2006.

    """
    # make sure the spectrogram is not filtered before
    if np.shape(phase) != np.shape(spec):
        raise ValueError('spectrogram and phase must be of same shape')
    # weighted_phase_deviation = spec * phase_deviation
    return np.mean(np.abs(_phase_deviation(phase) * spec), axis=1)


def normalized_weighted_phase_deviation(spec, phase, epsilon=EPSILON):
    """
    Normalized Weighted Phase Deviation.

    :param spec:    the magnitude spectrogram
    :param phase:   the phase spectrogram
    :param epsilon: add epsilon to avoid division by 0
    :return:        normalized weighted phase deviation onset detection
                    function

    "Onset Detection Revisited"
    Simon Dixon
    Proceedings of the 9th International Conference on Digital Audio Effects
    (DAFx), 2006.

    """
    if epsilon <= 0:
        raise ValueError("a positive value must be added before division")
    # normalize WPD by the sum of the spectrogram
    # (add a small epsilon so that we don't divide by 0)
    norm = np.add(np.mean(spec, axis=1), epsilon)
    return weighted_phase_deviation(spec, phase) / norm


def _complex_domain(spec, phase):
    """
    Helper method used by complex_domain() & rectified_complex_domain().

    :param spec:  the magnitude spectrogram
    :param phase: the phase spectrogram
    :return:      complex domain

    Note: we use the simple implementation presented in:
    "Onset Detection Revisited"
    Simon Dixon
    Proceedings of the 9th International Conference on Digital Audio Effects
    (DAFx), 2006.

    """
    if np.shape(phase) != np.shape(spec):
        raise ValueError('spectrogram and phase must be of same shape')
    # expected spectrogram
    cd_target = np.zeros_like(phase)
    # assume constant phase change
    cd_target[1:] = 2 * phase[1:] - phase[:-1]
    # add magnitude
    cd_target = spec * np.exp(1j * cd_target)
    # create complex spectrogram
    cd = spec * np.exp(1j * phase)
    # subtract the target values
    cd[1:] -= cd_target[:-1]
    return cd


def complex_domain(spec, phase):
    """
    Complex Domain.

    :param spec:  the magnitude spectrogram
    :param phase: the phase spectrogram
    :return:      complex domain onset detection function

    "On the use of phase and energy for musical onset detection in the complex
     domain"
    Juan Pablo Bello, Chris Duxbury, Matthew Davies and Mark Sandler
    IEEE Signal Processing Letters, Volume 11, Number 6, 2004.

    """
    # take the sum of the absolute changes
    return np.sum(np.abs(_complex_domain(spec, phase)), axis=1)


def rectified_complex_domain(spec, phase):
    """
    Rectified Complex Domain.

    :param spec:  the magnitude spectrogram
    :param phase: the phase spectrogram
    :return:      rectified complex domain onset detection function

    "Onset Detection Revisited"
    Simon Dixon
    Proceedings of the 9th International Conference on Digital Audio Effects
    (DAFx), 2006.

    """
    # rectified complex domain
    rcd = _complex_domain(spec, phase)
    # only keep values where the magnitude rises
    rcd *= diff(spec, pos=True)
    # take the sum of the absolute changes
    return np.sum(np.abs(rcd), axis=1)


class SpectralOnsetProcessor(SequentialProcessor):
    """
    The SpectralOnsetProcessor class implements most of the common onset
    detection functions based on the magnitude or phase information of a
    spectrogram.

    """
    FRAME_SIZE = 2048
    FPS = 200
    ONLINE = False
    MAX_BINS = 3
    TEMPORAL_FILTER = 0.015
    TEMPORAL_ORIGIN = 0

    methods = ['superflux', 'hfc', 'sd', 'sf', 'mkl', 'pd', 'wpd', 'nwpd',
               'cd', 'rcd']

    def __init__(self, method='superflux', *args, **kwargs):
        """
        Creates a new SpectralOnsetDetection instance.

        :param method: onset detection function

        """
        # signal handling processor
        sig = SignalProcessor(mono=True, *args, **kwargs)
        frames = FramedSignalProcessor(*args, **kwargs)
        spec = SpectrogramProcessor(*args, **kwargs)
        odf = getattr(self, method)
        # print odf
        # sequentially process everything
        super(SpectralOnsetProcessor, self).__init__([sig, frames, spec, odf])
        self.method = odf

    # Onset Detection Functions
    def hfc(self, data):
        """
        High Frequency Content.

        :param data: Spectrogram instance
        :return:     High Frequency Content onset detection function

        """
        return high_frequency_content(data.spec)

    def sd(self, data):
        """
        Spectral Diff.

        :param data: Spectrogram instance
        :return:     Spectral Diff onset detection function

        """
        return spectral_diff(data.spec, diff_frames=data.num_diff_frames)

    def sf(self, data):
        """
        Spectral Flux.

        :param data: Spectrogram instance
        :return:     Spectral Flux onset detection function

        """
        return spectral_flux(data.spec, diff_frames=data.num_diff_frames)

    def superflux(self, data):
        """
        SuperFlux.

        :param data: Spectrogram instance
        :return:     SuperFlux onset detection function

        """
        # TODO: use the diff of data directly!?
        return superflux(data.spec, diff_frames=data.num_diff_frames,
                         max_bins=data.diff_max_bins)

    def complex_flux(self, data, temporal_filter=TEMPORAL_FILTER,
                     temporal_origin=TEMPORAL_ORIGIN):
        """
        Complex flux is basically the spectral flux / SuperFlux with an
        additional local group delay based tremolo suppression.

        :param data:            Spectrogram instance
        :param temporal_filter: size of the temporal maximum filtering of the
                                local group delay [seconds]
        :param temporal_origin: origin shift of the temporal maximum filter
                                [seconds]
        :return:                ComplexFlux onset detection function

        """
        # convert timing information to frames
        temporal_filter = int(round(data.frames.fps * temporal_filter))
        temporal_origin = int(round(data.frames.fps * temporal_origin))
        # touch the lgd, so that the complex stft get computed (=faster)
        data.lgd
        # compute and return the activations
        return complex_flux(spec=data.spec,
                            lgd=np.abs(data.lgd),
                            filterbank=data.filterbank,
                            diff_frames=data.num_diff_frames,
                            max_bins=data.diff_max_bins,
                            temporal_filter=temporal_filter,
                            temporal_origin=temporal_origin)

    def mkl(self, data):
        """
        Modified Kullback-Leibler.

        :param data: Spectrogram instance
        :return:     Modified Kullback-Leibler onset detection function

        """
        return modified_kullback_leibler(data.spec,
                                         diff_frames=data.num_diff_frames)

    def pd(self, data):
        """
        Phase Deviation.

        :param data: Spectrogram instance
        :return:     Phase Deviation onset detection function

        """
        return phase_deviation(data.phase)

    def wpd(self, data):
        """
        Weighted Phase Deviation.

        :param data: Spectrogram instance
        :return:     Weighted Phase Deviation onset detection function

        """
        return weighted_phase_deviation(data.spec, data.phase)

    def nwpd(self, data):
        """
        Normalized Weighted Phase Deviation.

        :param data: Spectrogram instance
        :return:     Normalized Weighted Phase Deviation onset detection
                     function

        """
        return normalized_weighted_phase_deviation(data.spec, data.phase)

    def cd(self, data):
        """
        Complex Domain.

        :param data: Spectrogram instance
        :return:     Complex Domain onset detection function

        """
        return complex_domain(data.spec, data.phase)

    def rcd(self, data):
        """
        Rectified Complex Domain.

        :param data: Spectrogram instance
        :return:     Rectified Complex Domain onset detection function

        """
        return rectified_complex_domain(data.spec, data.phase)

    @classmethod
    def add_arguments(cls, parser, method=None, methods=None):
        """
        Add spectral ODF related arguments to an existing parser.

        :param parser:   existing argparse parser
        :param method:   default ODF method
        :param methods:  list of ODF methods
        :return:         spectral onset detection argument parser group

        """
        # add other parsers
        SignalProcessor.add_arguments(parser, norm=False, att=0)
        FramedSignalProcessor.add_arguments(parser, fps=200, online=False)
        SpectrogramProcessor.add_filter_arguments(parser, bands=24, fmin=30,
                                                  fmax=17000,
                                                  norm_filters=False)
        SpectrogramProcessor.add_log_arguments(parser, log=True, mul=1, add=1)
        SpectrogramProcessor.add_diff_arguments(parser, diff_ratio=0.5,
                                                diff_max_bins=3)
        # add onset detection related options to the existing parser
        g = parser.add_argument_group('spectral onset detection arguments')
        if methods is not None:
            g.add_argument('-o', '--odf', dest='method', default=method,
                           choices=methods,
                           help='use this onset detection function '
                                '[default=%(default)s]')
        # return the argument group so it can be modified if needed
        return g


class RNNOnsetProcessor(SequentialProcessor):
    """
    Class for detecting onsets with a recurrent neural network (RNN).

    """
    BI_FILES = glob.glob("%s/onsets_brnn_[1-8].npz" % MODELS_PATH)
    UNI_FILES = glob.glob("%s/onsets_rnn_[1-8].npz" % MODELS_PATH)
    FPS = 100
    ONLINE = False

    def __init__(self, nn_files=BI_FILES, fps=FPS, online=ONLINE, *args,
                 **kwargs):
        """
        Processor for finding possible onset positions in a signal.

        :param nn_files: list of RNN model files

        """
        # signal handling processor
        sig = SignalProcessor(mono=True, *args, **kwargs)
        # TODO: this information should be stored in the nn_files
        #       also the information about mul, add & diff_ratio and so on
        frame_sizes = [512, 1024, 2048] if online else [1024, 2048, 4096]
        # parallel specs + stacking processor
        stack = StackSpectrogramProcessor(frame_sizes=frame_sizes,
                                          fps=fps, online=online, bands=6,
                                          norm_filters=True, mul=5, add=1,
                                          diff_ratio=0.25, *args, **kwargs)
        # multiple RNN processor
        rnn = RNNProcessor(nn_files=nn_files, *args, **kwargs)
        # sequentially process everything
        super(RNNOnsetProcessor, self).__init__([sig, stack, rnn])

    @classmethod
    def add_arguments(cls, parser, online=ONLINE):
        """
        Add RNN onset detection related arguments to an existing parser.

        :param parser: existing argparse parser
        :param online: settings for online mode (OnsetDetectorLL)

        """
        if online:
            nn_files = cls.UNI_FILES
            norm = None
        else:
            nn_files = cls.BI_FILES
            norm = False
        # add signal processing arguments
        SignalProcessor.add_arguments(parser, norm=norm, att=0)
        # add rnn processing arguments
        RNNProcessor.add_arguments(parser, nn_files=nn_files)


# universal peak-picking method
def peak_picking(activations, threshold, smooth=None, pre_avg=0, post_avg=0,
                 pre_max=1, post_max=1):
    """
    Perform thresholding and peak-picking on the given activation function.

    :param activations: the activation function
    :param threshold:   threshold for peak-picking
    :param smooth:      smooth the activation function with the kernel
    :param pre_avg:     use N frames past information for moving average
    :param post_avg:    use N frames future information for moving average
    :param pre_max:     use N frames past information for moving maximum
    :param post_max:    use N frames future information for moving maximum
    :return:            indices of the detected peaks

    Notes: If no moving average is needed (e.g. the activations are independent
           of the signal's level as for neural network activations), set
           `pre_avg` and `post_avg` to 0.

           For offline peak picking, set `pre_max` and `post_max` to 1.

           For online peak picking, set all `post_` parameters to 0.

    "Evaluating the Online Capabilities of Onset Detection Methods"
    Sebastian Böck, Florian Krebs and Markus Schedl
    Proceedings of the 13th International Society for Music Information
    Retrieval Conference (ISMIR), 2012.

    """
    # smooth activations
    if smooth is not None:
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
            raise ValueError('activations must be either 1D or 2D')
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
            raise ValueError('activations must be either 1D or 2D')
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
        raise ValueError('activations must be either 1D or 2D')


class OnsetDetectionProcessor(IOProcessor):
    """
    This class implements the detection (i.e. peak-picking) functionality
    which can be used universally.

    """
    FPS = 100
    THRESHOLD = 0.5  # binary threshold
    SMOOTH = 0
    PRE_AVG = 0
    POST_AVG = 0
    PRE_MAX = 1. / FPS  # corresponds to one frame
    POST_MAX = 1. / FPS
    COMBINE = 0.03
    DELAY = 0

    def __init__(self, threshold=THRESHOLD, smooth=SMOOTH, pre_avg=PRE_AVG,
                 post_avg=POST_AVG, pre_max=PRE_MAX, post_max=POST_MAX,
                 combine=COMBINE, delay=DELAY, online=False, fps=FPS,
                 *args, **kwargs):
        """
        Creates a new PeakPickingProcessor instance.

        :param threshold: threshold for peak-picking
        :param smooth:    smooth the activation function over N seconds
        :param pre_avg:   use N seconds past information for moving average
        :param post_avg:  use N seconds future information for moving average
        :param pre_max:   use N seconds past information for moving maximum
        :param post_max:  use N seconds future information for moving maximum
        :param combine:   only report one onset within N seconds
        :param delay:     report onsets N seconds delayed
        :param online:    use online peak-picking (i.e. no future information)
        :param fps:       frames per second used for conversion of timings

        Notes: If no moving average is needed (e.g. the activations are
               independent of the signal's level as for neural network
               activations), `pre_avg` and `post_avg` should be set to 0.

               For offline peak picking set `pre_max` >= 1. / `fps` and
               `post_max` >= 1. / `fps`

               For online peak picking, all `post_` parameters are set to 0.

        "Evaluating the Online Capabilities of Onset Detection Methods"
        Sebastian Böck, Florian Krebs and Markus Schedl
        Proceedings of the 13th International Society for Music Information
        Retrieval Conference (ISMIR), 2012.

        """
        # make this an IOProcessor by defining input and output processings
        super(OnsetDetectionProcessor, self).__init__(self.detect, write_events)
        # adjust some params for online mode
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

    def detect(self, activations):
        """
        Detect the onsets in the given activation function.

        :param activations: onset activation function
        :return:            detected onsets

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
        # convert detections to a list of timestamps
        detections = detections.astype(np.float) / self.fps
        # shift if necessary
        if self.delay != 0:
            detections += self.delay
        # always use the first detection and all others if none was reported
        # within the last `combine` seconds
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

    @classmethod
    def add_arguments(cls, parser, threshold=THRESHOLD, smooth=None,
                      pre_avg=None, post_avg=None, pre_max=None, post_max=None,
                      combine=COMBINE, delay=DELAY):
        """
        Add onset peak-picking related arguments to an existing parser.

        :param parser:    existing argparse parser
        :param threshold: threshold for peak-picking
        :param smooth:    smooth the activation function over N seconds
        :param pre_avg:   use N seconds past information for moving average
        :param post_avg:  use N seconds future information for moving average
        :param pre_max:   use N seconds past information for moving maximum
        :param post_max:  use N seconds future information for moving maximum
        :param combine:   only report one event within N seconds
        :param delay:     report events N seconds delayed
        :return:          onset peak-picking argument parser group

        Parameters are included in the group only if they are not 'None'.

        """
        # add onset peak-picking related options to the existing parser
        g = parser.add_argument_group('onset peak-picking arguments')
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


class NNOnsetDetectionProcessor(IOProcessor):
    """
    Class for peak-picking with neural networks.

    """
    NN_FILES = glob.glob("%s/onsets_brnn_peak_picking_[1-8].npz" % MODELS_PATH)
    FPS = 100
    THRESHOLD = 0.4
    SMOOTH = 0.07
    COMBINE = 0.04
    DELAY = 0

    def __init__(self, nn_files=NN_FILES, threshold=THRESHOLD, smooth=SMOOTH,
                 combine=COMBINE, delay=DELAY, fps=FPS, *args, **kwargs):
        """
        Creates a new NNSpectralOnsetDetection instance.

        :param nn_files:  neural network files with models for peak-picking
        :param threshold: threshold for peak-picking
        :param smooth:    smooth the activation function over N seconds
        :param combine:   only report one onset within N seconds
        :param delay:     report onsets N seconds delayed

        "Enhanced peak picking for onset detection with recurrent neural
         networks"
        Sebastian Böck, Jan Schlüter and Gerhard Widmer
        Proceedings of the 6th International Workshop on Machine Learning and
        Music (MML), 2013.

        """
        # first perform RNN processing, then onset peak-picking
        rnn = RNNProcessor(nn_files=nn_files, num_threads=1)
        pp = OnsetDetectionProcessor(threshold=threshold, smooth=smooth,
                                     pre_max=1. / fps, post_max=1. / fps,
                                     combine=combine, delay=delay, fps=fps)
        # make this an IOProcessor by defining input and output processings
        super(NNOnsetDetectionProcessor, self).__init__(rnn, pp)

    @classmethod
    def add_arguments(cls, parser, nn_files=NN_FILES, threshold=THRESHOLD,
                      smooth=SMOOTH, combine=COMBINE, delay=DELAY):
        """
        Add peak-picking related arguments to an existing parser object.

        :param parser:    existing argparse parser object
        :param nn_files:  list with files of RNN models
        :param threshold: threshold for peak-picking
        :param smooth:    smooth the activation function over N seconds
        :param combine:   only report one event within N seconds
        :param delay:     report events N seconds delayed
        :return:          peak-picking argument parser group object

        """
        # add RNN parser arguments (but without number of threads)
        RNNProcessor.add_arguments(parser, nn_files=nn_files, num_threads=0)
        OnsetDetectionProcessor.add_arguments(parser, threshold=threshold,
                                              smooth=smooth, combine=combine,
                                              delay=delay)


def parser():
    """
    Create a parser and parse the arguments.

    :return: the parsed arguments

    """
    import argparse

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    If invoked without any parameters, the software detects all onsets in the
    given input file and writes them to the output file with the SuperFlux
    algorithm introduced in:

    "Maximum Filter Vibrato Suppression for Onset Detection"
    Sebastian Böck and Gerhard Widmer
    Proceedings of the 16th International Conference on Digital Audio Effects
    (DAFx-13), 2013.

    ''')
    # general options
    p.add_argument('files', metavar='files', nargs='+',
                   help='files to be processed')
    p.add_argument('-v', dest='verbose', action='store_true',
                   help='be verbose')
    p.add_argument('--suffix', action='store', type=str, default='.txt',
                   help='suffix for detections [default=%(default)s]')
    # add arguments
    SpectralOnsetProcessor.add_arguments(p, method='superflux',
                                         methods=SpectralOnsetProcessor.methods)
    OnsetDetectionProcessor.add_arguments(p, threshold=1.1, pre_max=0.01,
                                          post_max=0.05, pre_avg=0.15,
                                          post_avg=0, combine=0.03, delay=0)
    ActivationsProcessor.add_arguments(p)
    # parse arguments
    args = p.parse_args()
    # switch to offline mode
    if args.norm:
        args.online = False
    # print arguments
    if args.verbose:
        print args
    # return
    return args


def main():
    """
    Example onset detection program.

    """
    # parse arguments
    args = parser()

    # load or create beat activations
    if args.load:
        in_processor = ActivationsProcessor(mode='r', **vars(args))
    else:
        in_processor = SpectralOnsetProcessor(**vars(args))
    # save onset activations or detect onsets
    if args.save:
        out_processor = ActivationsProcessor(mode='w', **vars(args))
    else:
        out_processor = OnsetDetectionProcessor(**vars(args))
    # prepare the processor
    processor = IOProcessor(in_processor, out_processor)

    # which files to process
    if args.load:
        # load the activations
        ext = '.activations'
    else:
        # only process .wav files
        ext = '.wav'
    # process the files
    for infile in files(args.files, ext):
        if args.verbose:
            print infile
        # append suffix to input filename
        outfile = "%s%s" % (infile, args.suffix)
        # process infile to outfile
        print infile, outfile
        processor.process(infile, outfile)


if __name__ == '__main__':
    main()
