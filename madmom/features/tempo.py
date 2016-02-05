# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains tempo related functionality.

"""

from __future__ import absolute_import, division, print_function

import numpy as np

from madmom.processors import Processor
from madmom.audio.signal import smooth as smooth_signal

NO_TEMPO = np.nan


# helper functions
def smooth_histogram(histogram, smooth):
    """
    Smooth the given histogram.

    Parameters
    ----------
    histogram : tuple
        Histogram (tuple of 2 numpy arrays, the first giving the strengths of
        the bins and the second corresponding delay values).
    smooth : int or numpy array
        Smoothing kernel (size).

    Returns
    -------
    histogram_bins : numpy array
        Bins of the smoothed histogram.
    histogram_delays : numpy array
        Corresponding delays.

    Notes
    -----
    If `smooth` is an integer, a Hamming window of that length will be used as
    a smoothing kernel.

    """
    # smooth only the the histogram bins, not the corresponding delays
    return smooth_signal(histogram[0], smooth), histogram[1]


# interval detection
def interval_histogram_acf(activations, min_tau=1, max_tau=None):
    """
    Compute the interval histogram of the given (beat) activation function via
    auto-correlation as in [1]_.

    Parameters
    ----------
    activations : numpy array
        Beat activation function.
    min_tau : int, optional
        Minimal delay for the auto-correlation function [frames].
    max_tau : int, optional
        Maximal delay for the auto-correlation function [frames].

    Returns
    -------
    histogram_bins : numpy array
        Bins of the tempo histogram.
    histogram_delays : numpy array
        Corresponding delays [frames].

    References
    ----------
    .. [1] Sebastian Böck and Markus Schedl,
           "Enhanced Beat Tracking with Context-Aware Neural Networks",
           Proceedings of the 14th International Conference on Digital Audio
           Effects (DAFx), 2011.

    """
    # set the maximum delay
    if max_tau is None:
        max_tau = len(activations) - min_tau
    # test all possible delays
    taus = list(range(min_tau, max_tau + 1))
    bins = []
    # Note: this is faster than:
    #   corr = np.correlate(activations, activations, mode='full')
    #   bins = corr[len(activations) + min_tau - 1: len(activations) + max_tau]
    for tau in taus:
        bins.append(np.sum(np.abs(activations[tau:] * activations[0:-tau])))
    # return histogram
    return np.array(bins), np.array(taus)


def interval_histogram_comb(activations, alpha, min_tau=1, max_tau=None):
    """
    Compute the interval histogram of the given (beat) activation function via
    a bank of resonating comb filters as in [1]_.

    Parameters
    ----------
    activations : numpy array
        Beat activation function.
    alpha : float or numpy array
        Scaling factor for the comb filter; if only a single value is given,
        the same scaling factor for all delays is assumed.
    min_tau : int, optional
        Minimal delay for the comb filter [frames].
    max_tau : int, optional
        Maximal delta for comb filter [frames].

    Returns
    -------
    histogram_bins : numpy array
        Bins of the tempo histogram.
    histogram_delays : numpy array
        Corresponding delays [frames].

    References
    ----------
    .. [1] Sebastian Böck, Florian Krebs and Gerhard Widmer,
           "Accurate Tempo Estimation based on Recurrent Neural Networks and
           Resonating Comb Filters",
           Proceedings of the 16th International Society for Music Information
           Retrieval Conference (ISMIR), 2015.

    """
    # import comb filter
    from madmom.audio.comb_filters import CombFilterbankProcessor
    # set the maximum delay
    if max_tau is None:
        max_tau = len(activations) - min_tau
    # get the range of taus
    taus = np.arange(min_tau, max_tau + 1)
    # create a comb filter bank instance
    cfb = CombFilterbankProcessor('backward', taus, alpha)
    # activations = np.minimum(0.9, activations)
    if activations.ndim == 1:
        # apply a bank of comb filters
        act = cfb.process(activations)
        # determine the tau with the highest value for each time step
        # sum up the maxima to yield the histogram bin values
        histogram_bins = np.sum(act * (act == np.max(act, axis=0)), axis=1)
    elif activations.ndim == 2:
        histogram_bins = np.zeros_like(taus)
        # do the same as above for all bands
        for i in range(activations.shape[1]):
            # apply a bank of comb filters
            act = cfb.process(activations[:, i])
            # determine the tau with the highest value for each time step
            # sum up the maxima to yield the histogram bin values
            histogram_bins += np.sum(act * (act == np.max(act, axis=0)),
                                     axis=1)
    else:
        raise NotImplementedError('too many dimensions for comb filter tempo '
                                  'detection.')
    # return the histogram
    return histogram_bins, taus


# helper functions
def dominant_interval(histogram, smooth=None):
    """
    Extract the dominant interval of the given histogram.

    Parameters
    ----------
    histogram : tuple
        Histogram (tuple of 2 numpy arrays, the first giving the strengths of
        the bins and the second corresponding delay values).
    smooth : int or numpy array, optional
        Smooth the histogram with the given kernel (size).

    Returns
    -------
    interval : int
        Dominant interval.

    Notes
    -----
    If `smooth` is an integer, a Hamming window of that length will be used as
    a smoothing kernel.

    """
    # smooth the histogram bins
    if smooth:
        histogram = smooth_histogram(histogram, smooth)
    # return the dominant interval
    return histogram[1][np.argmax(histogram[0])]


# extract the tempo from a histogram
def detect_tempo(histogram, fps):
    """
    Extract the tempo from the given histogram.

    Parameters
    ----------
    histogram : tuple
        Histogram (tuple of 2 numpy arrays, the first giving the strengths of
        the bins and the second corresponding delay values).
    fps : float
        Frames per second.

    Returns
    -------
    tempi : numpy array
        Numpy array with the dominant tempi [bpm] (first column) and their
        relative strengths (second column).

    """
    from scipy.signal import argrelmax
    # histogram of IBIs
    bins = histogram[0]
    # convert the histogram bin delays to tempi in beats per minute
    tempi = 60.0 * fps / histogram[1]
    # to get the two dominant tempi, just keep the peaks
    # use 'wrap' mode to also get peaks at the borders
    peaks = argrelmax(bins, mode='wrap')[0]
    # we need more than 1 peak to report multiple tempi
    if len(peaks) == 0:
        # a flat histogram has no peaks, use the center bin
        if len(bins):
            ret = np.asarray([tempi[len(bins) / 2], 1.])
        else:
            # otherwise: no peaks, no tempo
            ret = np.asarray([NO_TEMPO, 0.])
    elif len(peaks) == 1:
        # report only the strongest tempo
        ret = np.asarray([tempi[peaks[0]], 1.])
    else:
        # sort the peaks in descending order of bin heights
        sorted_peaks = peaks[np.argsort(bins[peaks])[::-1]]
        # normalize their strengths
        strengths = bins[sorted_peaks]
        strengths /= np.sum(strengths)
        # return the tempi and their normalized strengths
        ret = np.asarray(list(zip(tempi[sorted_peaks], strengths)))
    # return the tempi
    return np.atleast_2d(ret)


# tempo estimation processor class
class TempoEstimationProcessor(Processor):
    """
    Tempo Estimation Processor class.

    Parameters
    ----------
    method : {'comb', 'acf', 'dbn'}
        Method used for tempo estimation.
    min_bpm : float, optional
        Minimum tempo to detect [bpm].
    max_bpm : float, optional
        Maximum tempo to detect [bpm].
    act_smooth : float, optional (default: 0.14)
        Smooth the activation function over `act_smooth` seconds.
    hist_smooth : int, optional (default: 7)
        Smooth the tempo histogram over `hist_smooth` bins.
    alpha : float, optional
        Scaling factor for the comb filter.
    fps : float, optional
        Frames per second.

    """
    # default values for tempo estimation
    METHOD = 'comb'
    MIN_BPM = 40.
    MAX_BPM = 250.
    HIST_SMOOTH = 7
    ACT_SMOOTH = 0.14
    ALPHA = 0.79

    def __init__(self, method=METHOD, min_bpm=MIN_BPM, max_bpm=MAX_BPM,
                 act_smooth=ACT_SMOOTH, hist_smooth=HIST_SMOOTH, alpha=ALPHA,
                 fps=None, **kwargs):
        # pylint: disable=unused-argument
        # save variables
        self.method = method
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.act_smooth = act_smooth
        self.hist_smooth = hist_smooth
        self.alpha = alpha
        self.fps = fps

    @property
    def min_interval(self):
        """Minimum beat interval [frames]."""
        return int(np.floor(60. * self.fps / self.max_bpm))

    @property
    def max_interval(self):
        """Maximum beat interval [frames]."""
        return int(np.ceil(60. * self.fps / self.min_bpm))

    def process(self, activations):
        """
        Detect the tempi from the (beat) activations.

        Parameters
        ----------
        activations : numpy array
            Beat activation function.

        Returns
        -------
        tempi : numpy array
            Array with the dominant tempi [bpm] (first column) and their
            relative strengths (second column).

        """
        # smooth the activations
        act_smooth = int(round(self.fps * self.act_smooth))
        activations = smooth_signal(activations, act_smooth)
        # generate a histogram of beat intervals
        histogram = self.interval_histogram(activations.astype(np.float))
        # smooth the histogram
        histogram = smooth_histogram(histogram, self.hist_smooth)
        # detect the tempi and return them
        return detect_tempo(histogram, self.fps)

    def interval_histogram(self, activations):
        """
        Compute the histogram of the beat intervals with the selected method.

        Parameters
        ----------
        activations : numpy array
            Beat activation function.

        Returns
        -------
        histogram_bins : numpy array
            Bins of the beat interval histogram.
        histogram_delays : numpy array
            Corresponding delays [frames].

        """
        # build the tempo (i.e. inter beat interval) histogram and return it
        if self.method == 'acf':
            return interval_histogram_acf(activations, self.min_interval,
                                          self.max_interval)
        elif self.method == 'comb':
            return interval_histogram_comb(activations, self.alpha,
                                           self.min_interval,
                                           self.max_interval)
        elif self.method == 'dbn':
            from .beats import DBNBeatTrackingProcessor
            # instantiate a DBN for beat tracking
            dbn = DBNBeatTrackingProcessor(min_bpm=self.min_bpm,
                                           max_bpm=self.max_bpm,
                                           num_tempi=None, fps=self.fps)
            # get the best state path by calling the viterbi algorithm
            path, _ = dbn.hmm.viterbi(activations.astype(np.float32))
            intervals = dbn.st.state_intervals[path]
            # get the counts of the bins
            bins = np.bincount(intervals, minlength=dbn.st.intervals.max() + 1)
            # truncate everything below the minimum interval of the state space
            bins = bins[dbn.st.intervals.min():]
            # build a histogram together with the intervals and return it
            return bins, dbn.st.intervals
        else:
            raise ValueError('tempo estimation method unknown')

    def dominant_interval(self, histogram):
        """
        Extract the dominant interval of the given histogram.

        Parameters
        ----------
        histogram : tuple
            Histogram (tuple of 2 numpy arrays, the first giving the strengths
            of the bins and the second corresponding delay values).

        Returns
        -------
        interval : int
            Dominant interval.

        """
        # return the dominant interval
        return dominant_interval(histogram, self.hist_smooth)

    @staticmethod
    def add_arguments(parser, method=METHOD, min_bpm=MIN_BPM, max_bpm=MAX_BPM,
                      act_smooth=ACT_SMOOTH, hist_smooth=HIST_SMOOTH,
                      alpha=ALPHA):
        """
        Add tempo estimation related arguments to an existing parser.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser.
        method : {'comb', 'acf', 'dbn'}
            Method used for tempo estimation.
        min_bpm : float, optional
            Minimum tempo to detect [bpm].
        max_bpm : float, optional
            Maximum tempo to detect [bpm].
        act_smooth : float, optional
            Smooth the activation function over `act_smooth` seconds.
        hist_smooth : int, optional
            Smooth the tempo histogram over `hist_smooth` bins.
        alpha : float, optional
            Scaling factor for the comb filter.

        Returns
        -------
        parser_group : argparse argument group
            Tempo argument parser group.

        Notes
        -----
        Parameters are included in the group only if they are not 'None'.

        """
        # add tempo estimation related options to the existing parser
        g = parser.add_argument_group('tempo estimation arguments')
        if method is not None:
            g.add_argument('--method', action='store', type=str,
                           default=method, choices=['acf', 'comb', 'dbn'],
                           help="which method to use [default=%(default)s]")
        if min_bpm is not None:
            g.add_argument('--min_bpm', action='store', type=float,
                           default=min_bpm,
                           help='minimum tempo [bpm, default=%(default).2f]')
        if max_bpm is not None:
            g.add_argument('--max_bpm', action='store', type=float,
                           default=max_bpm,
                           help='maximum tempo [bpm, default=%(default).2f]')
        if act_smooth is not None:
            g.add_argument('--act_smooth', action='store', type=float,
                           default=act_smooth,
                           help='smooth the activations over N seconds '
                                '[default=%(default).2f]')
        if hist_smooth is not None:
            g.add_argument('--hist_smooth', action='store', type=int,
                           default=hist_smooth,
                           help='smooth the tempo histogram over N bins '
                                '[default=%(default)d]')
        if alpha is not None:
            g.add_argument('--alpha', action='store', type=float,
                           default=alpha,
                           help='alpha for comb filter tempo estimation '
                                '[default=%(default).2f]')
        # return the argument group so it can be modified if needed
        return g


# helper function for writing the detected tempi to file
def write_tempo(tempi, filename, mirex=False):
    """
    Write the most dominant tempi and the relative strength to a file.

    Parameters
    ----------
    tempi : numpy array
        Array with the detected tempi (first column) and their strengths
        (second column).
    filename : str or file handle
        Output file.
    mirex : bool, optional
        Report the lower tempo first (as required by MIREX).

    Returns
    -------
    tempo_1 : float
        The most dominant tempo.
    tempo_2 : float
        The second most dominant tempo.
    strength : float
        Their relative strength.

    """
    # make the given tempi a 2d array
    tempi = np.array(tempi, ndmin=2)
    # default values
    t1, t2, strength = 0., 0., 1.
    # only one tempo was detected
    if len(tempi) == 1:
        t1 = tempi[0][0]
        # generate a fake second tempo
        # the boundary of 68 bpm is taken from Tzanetakis 2013 ICASSP paper
        if t1 < 68:
            t2 = t1 * 2.
        else:
            t2 = t1 / 2.
    # consider only the two strongest tempi and strengths
    elif len(tempi) > 1:
        t1, t2 = tempi[:2, 0]
        strength = tempi[0, 1] / sum(tempi[:2, 1])
    # for MIREX, the lower tempo must be given first
    if mirex and t1 > t2:
        t1, t2, strength = t2, t1, 1. - strength
    # format as a numpy array
    out = np.array([t1, t2, strength], ndmin=2)
    # write to output
    np.savetxt(filename, out, fmt='%.2f\t%.2f\t%.2f')
    # also return the tempi & strength
    return t1, t2, strength
