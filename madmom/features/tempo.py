# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains tempo related functionality.

"""

from __future__ import absolute_import, division, print_function

import numpy as np
import sys

from madmom.processors import Processor, BufferProcessor
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
    # smooth only the histogram bins, not the corresponding delays
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
    if activations.ndim != 1:
        raise NotImplementedError('too many dimensions for autocorrelation '
                                  'interval histogram calculation.')
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
    if activations.ndim in (1, 2):
        # apply a bank of comb filters
        act = cfb.process(activations)
        # determine the tau with the highest value for each time step
        act_max = act == np.max(act, axis=-1)[..., np.newaxis]
        # sum up these maxima weighted by the activation value to yield the
        # histogram bin values
        histogram_bins = np.sum(act * act_max, axis=0)
    else:
        raise NotImplementedError('too many dimensions for comb filter '
                                  'interval histogram calculation.')
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
            ret = np.asarray([tempi[len(bins) // 2], 1.])
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


# tempo estimation processor classes
class BaseTempoEstimationProcessor(Processor):
    """
    Tempo Estimation Processor class.

    Parameters
    ----------
    min_bpm : float
        Minimum tempo to detect [bpm].
    max_bpm : float
        Maximum tempo to detect [bpm].
    act_smooth : float
        Smooth the activation function over `act_smooth` seconds.
    hist_smooth : int
        Smooth the tempo histogram over `hist_smooth` bins.
    fps : float, optional
        Frames per second.

    Notes
    -----
    This class provides the basic tempo estimation functionality and depends
    on an implementation of the `interval_histogram()` method. Please use one
    of the following classes:

    - :class:`CombFilterTempoEstimationProcessor`,
    - :class:`ACFTempoEstimationProcessor` or
    - :class:`DBNTempoEstimationProcessor`.

    """

    def __init__(self, min_bpm, max_bpm, act_smooth, hist_smooth, fps=None,
                 online=False, **kwargs):
        # pylint: disable=unused-argument
        # save variables
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.act_smooth = act_smooth
        self.hist_smooth = hist_smooth
        self.fps = fps
        self.online = online
        if self.online:
            self.visualize = kwargs.get('verbose', False)

    @property
    def min_interval(self):
        """Minimum beat interval [frames]."""
        return int(np.floor(60. * self.fps / self.max_bpm))

    @property
    def max_interval(self):
        """Maximum beat interval [frames]."""
        return int(np.ceil(60. * self.fps / self.min_bpm))

    def process(self, activations, **kwargs):
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
        if self.online:
            return self.process_online(activations, **kwargs)
        else:
            return self.process_offline(activations, **kwargs)

    def process_offline(self, activations, **kwargs):
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

    def process_online(self, activations, reset=True, **kwargs):
        """
        Detect the tempi from the (beat) activations in online mode.

        Parameters
        ----------
        activations : numpy array
            Beat activation function processed frame by frame.
        reset : bool, optional
            Reset the TempoEstimationProcessor to its initial state before
            processing.

        Returns
        -------
        tempi : numpy array
            Array with the dominant tempi [bpm] (first column) and their
            relative strengths (second column).

        """
        # multiple activations will result in multiple tempi
        tempi = []
        # iterate over all activations
        for activation in activations:
            # build the tempo histogram depending on the chosen method
            histogram = self.online_interval_histogram(activation, reset=reset)
            # smooth the histogram
            histogram = smooth_histogram(histogram, self.hist_smooth)
            # detect the tempo and append it to the found tempi
            tempo = detect_tempo(histogram, self.fps)
            tempi.append(tempo)
            # visualize tempo
            if self.visualize:
                display = ''
                # display the 3 most likely tempi and their strengths
                for i, display_tempo in enumerate(tempo[:3], start=1):
                    # display tempo
                    display += '| ' + str(round(display_tempo[0], 1)) + ' '
                    # display strength
                    display += min(int(display_tempo[1] * 50), 18) * '*'
                    # fill up the rest with spaces
                    display = display.ljust(i * 26)
                # print the tempi
                sys.stderr.write('\r%s' % ''.join(display) + '|')
                sys.stderr.flush()
        # return last detected tempo
        return tempi[-1]

    def reset(self):
        """
        Reset the TempoEstimationProcessor. Needs to be implemented
        by subclass.

        """
        raise NotImplementedError('Must be implemented by subclass.')

    def interval_histogram(self, activations):
        """
        Compute the histogram of the beat intervals. Needs to be implemented
        by subclass.

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
        raise NotImplementedError('Must be implemented by subclass.')

    def online_interval_histogram(self, activation, reset=True):
        """
        Compute the histogram of the beat intervals for online mode.
        Needs to be implemented by subclass.

        Parameters
        ----------
        activation : numpy float
            Beat activation function processed frame by frame.
        reset : bool, optional
            Reset the TempoEstimationProcessor to its initial state before
            processing.

        Returns
        -------
        histogram_bins : numpy array
            Bins of the beat interval histogram.
        histogram_delays : numpy array
            Corresponding delays [frames].

        """
        raise NotImplementedError('Must be implemented by subclass.')

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
    def add_arguments(parser, min_bpm=None, max_bpm=None, act_smooth=None,
                      hist_smooth=None):
        """
        Add tempo estimation related arguments to an existing parser.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser.
        min_bpm : float, optional
            Minimum tempo to detect [bpm].
        max_bpm : float, optional
            Maximum tempo to detect [bpm].
        act_smooth : float, optional
            Smooth the activation function over `act_smooth` seconds.
        hist_smooth : int, optional
            Smooth the tempo histogram over `hist_smooth` bins.

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
        # return the argument group so it can be modified if needed
        return g


class TempoEstimationProcessor(BaseTempoEstimationProcessor):
    """
    TempoEstimationProcessor is deprecated as of version 0.16 and will be
    removed in version 0.17. Use one of these dedicated tempo estimation
    processors instead:

    - :class:`CombFilterTempoEstimationProcessor`,
    - :class:`ACFTempoEstimationProcessor` or
    - :class:`DBNTempoEstimationProcessor`.

    """
    # default values for tempo estimation
    METHOD = 'comb'
    MIN_BPM = 40.
    MAX_BPM = 250.
    HIST_SMOOTH = 9
    ACT_SMOOTH = 0.14
    ALPHA = 0.79

    def __init__(self, method=METHOD, min_bpm=MIN_BPM, max_bpm=MAX_BPM,
                 act_smooth=ACT_SMOOTH, hist_smooth=HIST_SMOOTH, alpha=ALPHA,
                 fps=None, **kwargs):
        # pylint: disable=unused-argument
        super(TempoEstimationProcessor, self).__init__(
            min_bpm=min_bpm, max_bpm=max_bpm, act_smooth=act_smooth,
            hist_smooth=hist_smooth, fps=fps)
        # save variables
        self.method = method
        self.alpha = alpha

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
        import warnings
        warnings.warn(self.__doc__)
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
        import warnings
        warnings.warn(TempoEstimationProcessor.__doc__)
        # add tempo estimation related options to the existing parser
        g = CombFilterTempoEstimationProcessor.add_arguments(
            parser, min_bpm, max_bpm, act_smooth, hist_smooth, alpha)
        # add method switch
        if method is not None:
            g.add_argument('--method', action='store', type=str,
                           default=method, choices=['acf', 'comb', 'dbn'],
                           help="which method to use [default=%(default)s]")
        # return the argument group so it can be modified if needed
        return g


class CombFilterTempoEstimationProcessor(BaseTempoEstimationProcessor):
    """
    Tempo estimation witch comb filters.

    Parameters
    ----------
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
    buffer_size : float, optional
        Use a buffer of this size to sum the max. bins in online mode
        [seconds].
    fps : float, optional
        Frames per second.
    online : bool, optional
        Extend the combfilter matrix frame by frame.

    Examples
    --------
    Create a CombFilterTempoEstimationProcessor. The returned array represents
    the estimated tempi (given in beats per minute) and their relative
    strength.

    >>> proc = CombFilterTempoEstimationProcessor(fps=100)
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.tempo.CombFilterTempoEstimationProcessor object at 0x...>

    Call this CombFilterTempoEstimationProcessor with the beat activation
    function obtained by RNNBeatProcessor to estimate the tempi.

    >>> from madmom.features.beats import RNNBeatProcessor
    >>> act = RNNBeatProcessor()('tests/data/audio/sample.wav')
    >>> proc(act)  # doctest: +NORMALIZE_WHITESPACE
    array([[ 176.47059,  0.47469],
           [ 117.64706,  0.17667],
           [ 240.     ,  0.15371],
           [  68.96552,  0.09864],
           [  82.19178,  0.09629]])

    """
    # default values for tempo estimation
    MIN_BPM = 40.
    MAX_BPM = 250.
    HIST_SMOOTH = 9
    ACT_SMOOTH = 0.14
    ALPHA = 0.79
    BUFFER_SIZE = 10.

    def __init__(self, min_bpm=MIN_BPM, max_bpm=MAX_BPM, act_smooth=ACT_SMOOTH,
                 hist_smooth=HIST_SMOOTH, alpha=ALPHA, buffer_size=BUFFER_SIZE,
                 fps=None, online=False, **kwargs):
        # pylint: disable=unused-argument
        super(CombFilterTempoEstimationProcessor, self).__init__(
            min_bpm=min_bpm, max_bpm=max_bpm, act_smooth=act_smooth,
            hist_smooth=hist_smooth, fps=fps, online=online, **kwargs)
        # save additional variables
        self.alpha = alpha
        if self.online:
            self.taus = np.arange(self.min_interval, self.max_interval + 1)
            self.combfilter_matrix = []
            self.buffer = BufferProcessor((int(buffer_size * self.fps),
                                           len(self.taus)))

    def reset(self):
        """Reset the CombFilterTempoEstimationProcessor."""
        self.combfilter_matrix = []
        self.buffer.reset()

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
        return interval_histogram_comb(activations, self.alpha,
                                       self.min_interval, self.max_interval)

    def online_interval_histogram(self, activation, reset=True):
        """
        Compute the histogram of the beat intervals using a resonating
        comb filter bank for online mode.

        Parameters
        ----------
        activation : numpy float
            Beat activation function processed frame by frame.
        reset : bool, optional
            Reset the CombTempoEstimationProcessor to its initial state before
            processing.

        Returns
        -------
        histogram_bins : numpy array
            Bins of the tempo histogram.
        histogram_delays : numpy array
            Corresponding delays [frames].
        """
        # reset to initial state
        if reset:
            self.reset()
        # expand the activation for every tau
        activation = np.full(len(self.taus), activation, dtype=np.float)
        # append it to the comb filter matrix
        self.combfilter_matrix.append(activation)
        # online feed backward comb filter
        min_tau = min(self.taus)
        for t in self.taus:
            if len(self.combfilter_matrix) > t:
                self.combfilter_matrix[-1][t - min_tau] += self.alpha * \
                    self.combfilter_matrix[-1 - t][t - min_tau]
        # retrieve maxima
        act_max = self.combfilter_matrix[-1] == \
            np.max(self.combfilter_matrix[-1], axis=-1)
        # compute the max bins
        bins = self.combfilter_matrix[-1] * act_max
        # use a buffer to only keep bins of the last seconds
        # shift buffer and put new bins at end of buffer
        bins = self.buffer(bins)
        # build a histogram together with the intervals and return it
        return np.sum(bins, axis=0), np.array(self.taus)

    @staticmethod
    def add_arguments(parser, min_bpm=MIN_BPM, max_bpm=MAX_BPM,
                      act_smooth=ACT_SMOOTH, hist_smooth=HIST_SMOOTH,
                      alpha=ALPHA):
        """
        Add tempo estimation related arguments to an existing parser.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser.
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

        """
        # add tempo estimation related options to the existing parser
        g = BaseTempoEstimationProcessor.add_arguments(
            parser, min_bpm, max_bpm, act_smooth, hist_smooth)
        # add comb filter specific arguments
        g.add_argument('--alpha', action='store', type=float, default=alpha,
                       help='alpha for comb filter tempo estimation '
                            '[default=%(default).2f]')
        # return the argument group so it can be modified if needed
        return g


class ACFTempoEstimationProcessor(BaseTempoEstimationProcessor):
    """
    Tempo estimation via autocorrelation.

    Parameters
    ----------
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
    buffer_size : float, optional
        Use a buffer of this size for the activations to calculate the
        auto-correlation function [seconds].
    fps : float, optional
        Frames per second.
    online : bool, optional
        Use only the buffered activations to perform the auto correlation.

    Examples
    --------
    Create a ACFTempoEstimationProcessor. The returned array represents the
    estimated tempi (given in beats per minute) and their relative strength.

    >>> proc = ACFTempoEstimationProcessor(fps=100)
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.tempo.ACFTempoEstimationProcessor object at 0x...>

    Call this ACFTempoEstimationProcessor with the beat activation function
    obtained by RNNBeatProcessor to estimate the tempi.

    >>> from madmom.features.beats import RNNBeatProcessor
    >>> act = RNNBeatProcessor()('tests/data/audio/sample.wav')
    >>> proc(act)  # doctest: +NORMALIZE_WHITESPACE
    array([[ 176.47059,  0.47469],
           [ 117.64706,  0.17667],
           [ 240.     ,  0.15371],
           [  68.96552,  0.09864],
           [  82.19178,  0.09629]])

    """
    # default values for tempo estimation
    MIN_BPM = 40.
    MAX_BPM = 250.
    HIST_SMOOTH = 9
    ACT_SMOOTH = 0.14
    BUFFER_SIZE = 10.

    def __init__(self, min_bpm=MIN_BPM, max_bpm=MAX_BPM, act_smooth=ACT_SMOOTH,
                 hist_smooth=HIST_SMOOTH, buffer_size=BUFFER_SIZE, fps=None,
                 online=False, **kwargs):
        # pylint: disable=unused-argument
        super(ACFTempoEstimationProcessor, self).__init__(
            min_bpm=min_bpm, max_bpm=max_bpm, act_smooth=act_smooth,
            hist_smooth=hist_smooth, fps=fps, online=online, **kwargs)
        # save additional variables
        if self.online:
            self.buffer = BufferProcessor(int(buffer_size * self.fps))

    def reset(self):
        """Reset the ACFTempoEstimationProcessor."""
        self.buffer.reset()

    def interval_histogram(self, activations):
        """
        Compute the histogram of the beat intervals with the autocorrelation
        function.

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
        return interval_histogram_acf(activations, self.min_interval,
                                      self.max_interval)

    def online_interval_histogram(self, activation, reset=True):
        """
        Compute the histogram of the beat intervals using auto-correlation on
        buffered activations.

        Parameters
        ----------
        activation : numpy float
            Beat activation function processed frame by frame.
        reset : bool, optional
            Reset the ACFTempoEstimationProcessor to its initial state before
            processing.

        Returns
        -------
        histogram_bins : numpy array
            Bins of the tempo histogram.
        histogram_delays : numpy array
            Corresponding delays [frames].
        """
        # reset to initial state
        if reset:
            self.reset()
        # shift buffer and put new activation at end of buffer
        activation = self.buffer(activation)
        # use offline acf function on buffered activations
        return interval_histogram_acf(activation, self.min_interval,
                                      self.max_interval)


class DBNTempoEstimationProcessor(BaseTempoEstimationProcessor):
    """
    Tempo estimation with a dynamic Bayesian network.

    Parameters
    ----------
    min_bpm : float, optional
        Minimum tempo to detect [bpm].
    max_bpm : float, optional
        Maximum tempo to detect [bpm].
    act_smooth : float, optional
        Smooth the activation function over `act_smooth` seconds.
    hist_smooth : int, optional
        Smooth the tempo histogram over `hist_smooth` bins.
    fps : float, optional
        Frames per second.
    online : bool, optional
        Use the forward algorithm to retrieve the tempo.

    Examples
    --------
    Create a DBNTempoEstimationProcessor. The returned array represents the
    estimated tempi (given in beats per minute) and their relative strength.

    >>> proc = DBNTempoEstimationProcessor(fps=100)
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.tempo.DBNTempoEstimationProcessor object at 0x...>

    Call this DBNTempoEstimationProcessor with the beat activation function
    obtained by RNNBeatProcessor to estimate the tempi.

    >>> from madmom.features.beats import RNNBeatProcessor
    >>> act = RNNBeatProcessor()('tests/data/audio/sample.wav')
    >>> proc(act)  # doctest: +NORMALIZE_WHITESPACE
    array([[ 176.47059,  0.47469],
           [ 117.64706,  0.17667],
           [ 240.     ,  0.15371],
           [  68.96552,  0.09864],
           [  82.19178,  0.09629]])

    """
    # default values for tempo estimation
    MIN_BPM = 40.
    MAX_BPM = 250.
    HIST_SMOOTH = 9
    ACT_SMOOTH = 0.

    def __init__(self, min_bpm=MIN_BPM, max_bpm=MAX_BPM, act_smooth=ACT_SMOOTH,
                 hist_smooth=HIST_SMOOTH, fps=None, online=False, **kwargs):
        # pylint: disable=unused-argument
        super(DBNTempoEstimationProcessor, self).__init__(
            min_bpm=min_bpm, max_bpm=max_bpm, act_smooth=act_smooth,
            hist_smooth=hist_smooth, fps=fps, online=online, **kwargs)
        # save additional variables
        from .beats import DBNBeatTrackingProcessor
        self.dbn = DBNBeatTrackingProcessor(min_bpm=self.min_bpm,
                                            max_bpm=self.max_bpm,
                                            fps=self.fps, **kwargs)

    def reset(self):
        """Reset the DBNTempoEstimationProcessor."""
        self.dbn.hmm.reset()

    def interval_histogram(self, activations):
        """
        Compute the histogram of the beat intervals with a DBN.

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
        # get the best state path by calling the viterbi algorithm
        path, _ = self.dbn.hmm.viterbi(activations.astype(np.float32))
        intervals = self.dbn.st.state_intervals[path]
        # get the counts of the bins
        bins = np.bincount(intervals,
                           minlength=self.dbn.st.intervals.max() + 1)
        # truncate everything below the minimum interval of the state space
        bins = bins[self.dbn.st.intervals.min():]
        # build a histogram together with the intervals and return it
        return bins, self.dbn.st.intervals

    def online_interval_histogram(self, activation, reset=True):
        """
        Compute the histogram of the beat intervals using a DBN and the
        forward algorithm.

        Parameters
        ----------
        activation : numpy float
            Beat activation function processed frame by frame.
        reset : bool, optional
            Reset the DBNTempoEstimationProcessor to its initial state before
            processing.

        Returns
        -------
        histogram_bins : numpy array
           Bins of the tempo histogram.
        histogram_delays : numpy array
           Corresponding delays [frames].

        """
        # reset to initial state
        if reset:
            self.reset()
        # use forward path to get best state
        fwd = self.dbn.hmm.forward(activation, reset=reset)
        # choose the best state for each step
        states = np.argmax(fwd, axis=1)
        intervals = self.dbn.st.state_intervals[states]
        # get the counts of the bins
        bins = np.bincount(intervals,
                           minlength=self.dbn.st.intervals.max() + 1)
        # truncate everything below the minimum interval of the state space
        bins = bins[self.dbn.st.intervals.min():]
        # build a histogram together with the intervals and return it
        return bins, self.dbn.st.intervals


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
