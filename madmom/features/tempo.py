# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains tempo related functionality.

"""

from __future__ import absolute_import, division, print_function

import sys
import warnings
from operator import itemgetter

import numpy as np

from ..audio.signal import smooth as smooth_signal
from ..processors import BufferProcessor, OnlineProcessor

METHOD = 'comb'
ALPHA = 0.79
MIN_BPM = 40.
MAX_BPM = 250.
ACT_SMOOTH = 0.14
HIST_SMOOTH = 9
HIST_BUFFER = 10.
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
def detect_tempo(histogram, fps=None, interpolate=False):
    """
    Detect the dominant tempi from the given histogram.

    Parameters
    ----------
    histogram : tuple
        Histogram (tuple of 2 numpy arrays, the first giving the strengths of
        the bins and the second corresponding tempo/delay values).
    fps : float, optional
        Frames per second. If 'None', the second element is interpreted as
        tempo values. If set, the histogram's second element is interpreted as
        inter beat intervals (IBIs) in frames with the given rate.
    interpolate : bool, optional
        Interpolate the histogram bins.

    Returns
    -------
    tempi : numpy array
        Numpy array with the dominant tempi [bpm] (first column) and their
        relative strengths (second column).

    """
    from scipy.signal import argrelmax
    from scipy.interpolate import interp1d
    # histogram of IBIs/tempi
    bins, tempi = histogram
    # interpolate tempo
    if interpolate:
        # create interpolation function
        interpolation_fn = interp1d(tempi, bins, 'quadratic')
        # generate new intervals/tempi with 100x the resolution
        tempi = np.arange(tempi[0], tempi[-1], 0.01)
        # apply quadratic interpolation
        bins = interpolation_fn(tempi)
    # if FPS is set, treat the histogram bins as intervals and convert to BPM
    if fps is not None:
        tempi = 60.0 * fps / tempi
    # peaks represent the dominant tempi
    # Note: use 'wrap' mode to also get peaks at the borders
    peaks = argrelmax(bins, mode='wrap')[0]
    if len(peaks) == 0:
        # no peaks: no tempo
        tempi = np.asarray([NO_TEMPO, 0.])
    elif len(peaks) == 1:
        # a single peak: report only the strongest tempo
        tempi = np.asarray([tempi[peaks[0]], 1.])
    else:
        # multiple peaks: multiple tempi
        # sort the peaks in descending order of bin heights
        sorted_peaks = peaks[np.argsort(bins[peaks])[::-1]]
        # normalize their strengths
        strengths = bins[sorted_peaks]
        strengths /= np.sum(strengths)
        # return the tempi and their normalized strengths
        tempi = np.asarray(list(zip(tempi[sorted_peaks], strengths)))
    # return detected tempi
    return np.atleast_2d(tempi)


# tempo histogram processor classes
class TempoHistogramProcessor(OnlineProcessor):
    """
    Tempo Histogram Processor class.

    Parameters
    ----------
    min_bpm : float
        Minimum tempo to detect [bpm].
    max_bpm : float
        Maximum tempo to detect [bpm].
    hist_buffer : float
        Aggregate the tempo histogram over `hist_buffer` seconds.
    fps : float, optional
        Frames per second.

    Notes
    -----
    This abstract class provides the basic tempo histogram functionality.
    Please use one of the following implementations:

    - :class:`CombFilterTempoHistogramProcessor`,
    - :class:`ACFTempoHistogramProcessor` or
    - :class:`DBNTempoHistogramProcessor`.

    """

    def __init__(self, min_bpm, max_bpm, hist_buffer=HIST_BUFFER, fps=None,
                 online=False, **kwargs):
        # pylint: disable=unused-argument
        super(TempoHistogramProcessor, self).__init__(online=online)
        self.min_bpm = float(min_bpm)
        self.max_bpm = float(max_bpm)
        self.hist_buffer = hist_buffer
        self.fps = fps
        if self.online:
            self._hist_buffer = BufferProcessor((int(hist_buffer * self.fps),
                                                 len(self.intervals)))

    @property
    def min_interval(self):
        """Minimum beat interval [frames]."""
        return int(np.floor(60. * self.fps / self.max_bpm))

    @property
    def max_interval(self):
        """Maximum beat interval [frames]."""
        return int(np.ceil(60. * self.fps / self.min_bpm))

    @property
    def intervals(self):
        """Beat intervals [frames]."""
        return np.arange(self.min_interval, self.max_interval + 1)

    def reset(self):
        """Reset the tempo histogram aggregation buffer."""
        self._hist_buffer.reset()


class CombFilterTempoHistogramProcessor(TempoHistogramProcessor):
    """
    Create a tempo histogram with a bank of resonating comb filters.

    Parameters
    ----------
    min_bpm : float, optional
        Minimum tempo to detect [bpm].
    max_bpm : float, optional
        Maximum tempo to detect [bpm].
    alpha : float, optional
        Scaling factor for the comb filter.
    hist_buffer : float
        Aggregate the tempo histogram over `hist_buffer` seconds.
    fps : float, optional
        Frames per second.
    online : bool, optional
        Operate in online (i.e. causal) mode.

    """

    def __init__(self, min_bpm=MIN_BPM, max_bpm=MAX_BPM, alpha=ALPHA,
                 hist_buffer=HIST_BUFFER, fps=None, online=False, **kwargs):
        # pylint: disable=unused-argument
        super(CombFilterTempoHistogramProcessor, self).__init__(
            min_bpm=min_bpm, max_bpm=max_bpm, hist_buffer=hist_buffer, fps=fps,
            online=online, **kwargs)
        self.alpha = alpha
        if self.online:
            self._comb_buffer = BufferProcessor((self.max_interval + 1,
                                                 len(self.intervals)))

    def reset(self):
        """Reset to initial state."""
        super(CombFilterTempoHistogramProcessor, self).reset()
        self._comb_buffer.reset()

    def process_offline(self, activations, **kwargs):
        """
        Compute the histogram of the beat intervals with a bank of resonating
        comb filters.

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

    def process_online(self, activations, reset=True, **kwargs):
        """
        Compute the histogram of the beat intervals with a bank of resonating
        comb filters in online mode.

        Parameters
        ----------
        activations : numpy float
            Beat activation function.
        reset : bool, optional
            Reset to initial state before processing.

        Returns
        -------
        histogram_bins : numpy array
            Bins of the tempo histogram.
        histogram_delays : numpy array
            Corresponding delays [frames].

        """

        activations = np.array(activations, copy=False, subok=True, ndmin=1)
        # reset to initial state
        if reset:
            self.reset()
        # indices at which to retrieve y[n - τ]
        idx = [-self.intervals, np.arange(len(self.intervals))]
        # iterate over all activations
        # Note: in online mode, activations are just float values, thus cast
        #       them as 1-dimensional array
        for act in np.array(activations, copy=False, subok=True, ndmin=1):
            # online feed backward comb filter (y[n] = x[n] + α * y[n - τ])
            y_n = act + self.alpha * self._comb_buffer[idx]
            # shift output buffer with new value
            self._comb_buffer(y_n)
            # determine the tau with the highest value
            act_max = y_n == np.max(y_n, axis=-1)[..., np.newaxis]
            # compute the max bins
            bins = y_n * act_max
            # use a buffer to only keep a certain number of bins
            # shift buffer and put new bins at end of buffer
            bins = self._hist_buffer(bins)
        # build a histogram together with the intervals and return it
        return np.sum(bins, axis=0), self.intervals


class ACFTempoHistogramProcessor(TempoHistogramProcessor):
    """
    Create a tempo histogram with autocorrelation.

    Parameters
    ----------
    min_bpm : float, optional
        Minimum tempo to detect [bpm].
    max_bpm : float, optional
        Maximum tempo to detect [bpm].
    hist_buffer : float
        Aggregate the tempo histogram over `hist_buffer` seconds.
    fps : float, optional
        Frames per second.
    online : bool, optional
        Operate in online (i.e. causal) mode.

    """

    def __init__(self, min_bpm=MIN_BPM, max_bpm=MAX_BPM,
                 hist_buffer=HIST_BUFFER, fps=None, online=False, **kwargs):
        # pylint: disable=unused-argument
        super(ACFTempoHistogramProcessor, self).__init__(
            min_bpm=min_bpm, max_bpm=max_bpm, hist_buffer=hist_buffer, fps=fps,
            online=online, **kwargs)
        if self.online:
            self._act_buffer = BufferProcessor((self.max_interval + 1, 1))

    def reset(self):
        """Reset to initial state."""
        super(ACFTempoHistogramProcessor, self).reset()
        self._act_buffer.reset()

    def process_offline(self, activations, **kwargs):
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

    def process_online(self, activations, reset=True, **kwargs):
        """
        Compute the histogram of the beat intervals with the autocorrelation
        function in online mode.

        Parameters
        ----------
        activations : numpy float
            Beat activation function.
        reset : bool, optional
            Reset to initial state before processing.

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
        # iterate over all activations
        # TODO: speed this up!
        for act in activations:
            # online ACF (y[n] = x[n] * x[n - τ])
            bins = act * self._act_buffer[-self.intervals].T
            # shift activation buffer with new value
            self._act_buffer(act)
            # use a buffer to only keep a certain number of bins
            # shift buffer and put new bins at end of buffer
            bins = self._hist_buffer(bins)
        # build a histogram together with the intervals and return it
        return np.sum(bins, axis=0), self.intervals


class DBNTempoHistogramProcessor(TempoHistogramProcessor):
    """
    Create a tempo histogram with a dynamic Bayesian network (DBN).

    Parameters
    ----------
    min_bpm : float, optional
        Minimum tempo to detect [bpm].
    max_bpm : float, optional
        Maximum tempo to detect [bpm].
    hist_buffer : float
        Aggregate the tempo histogram over `hist_buffer` seconds.
    fps : float, optional
        Frames per second.
    online : bool, optional
        Operate in online (i.e. causal) mode.

    """

    def __init__(self, min_bpm=MIN_BPM, max_bpm=MAX_BPM,
                 hist_buffer=HIST_BUFFER, fps=None, online=False, **kwargs):
        # pylint: disable=unused-argument
        super(DBNTempoHistogramProcessor, self).__init__(
            min_bpm=min_bpm, max_bpm=max_bpm, hist_buffer=hist_buffer, fps=fps,
            online=online, **kwargs)
        from .beats import DBNBeatTrackingProcessor
        self.dbn = DBNBeatTrackingProcessor(
            min_bpm=self.min_bpm, max_bpm=self.max_bpm, fps=self.fps,
            online=online, **kwargs)

    def reset(self):
        """Reset DBN to initial state."""
        super(DBNTempoHistogramProcessor, self).reset()
        self.dbn.hmm.reset()

    def process_offline(self, activations, **kwargs):
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

    def process_online(self, activations, reset=True, **kwargs):
        """
        Compute the histogram of the beat intervals with a DBN using the
        forward algorithm.

        Parameters
        ----------
        activations : numpy float
            Beat activation function.
        reset : bool, optional
            Reset DBN to initial state before processing.

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
        fwd = self.dbn.hmm.forward(activations, reset=reset)
        # choose the best state for each step
        states = np.argmax(fwd, axis=1)
        intervals = self.dbn.st.state_intervals[states]
        # convert intervals to bins
        bins = np.zeros((len(activations), len(self.intervals)))
        bins[np.arange(len(activations)), intervals - self.min_interval] = 1
        # shift buffer and put new bins at end of buffer
        bins = self._hist_buffer(bins)
        # build a histogram together with the intervals and return it
        return np.sum(bins, axis=0), self.intervals


class TCNTempoHistogramProcessor(TempoHistogramProcessor):
    """
    Derive a tempo histogram from (multi-task) TCN output.

    Parameters
    ----------
    min_bpm : float, optional
        Minimum tempo to detect [bpm].
    max_bpm : float, optional
        Maximum tempo to detect [bpm].

    References
    ----------
    .. [1] Sebastian Böck, Matthew Davies and Peter Knees,
           "Multi-Task learning of tempo and beat: learning one to improve the
           other",
           Proceedings of the 20th International Society for Music Information
           Retrieval Conference (ISMIR), 2019.

    """

    def __init__(self, min_bpm=MIN_BPM, max_bpm=MAX_BPM, **kwargs):
        # pylint: disable=unused-argument
        super(TCNTempoHistogramProcessor, self).__init__(
            min_bpm=min_bpm, max_bpm=max_bpm, **kwargs)

    def process(self, data, **kwargs):
        """
        Extract tempo histogram from (multi-task) TCN output.

        Parameters
        ----------
        data : numpy array or tuple of numpy arrays
            Tempo-task (numpy array) or multi-task (tuple) output of TCN.

        Returns
        -------
        histogram_bins : numpy array
            Bins of tempo histogram, i.e. tempo strengths.
        histogram_tempi : numpy array
            Corresponding tempi [bpm].

        """
        # if data is a tuple, tempo is usually last item of TCN output
        if type(data) == tuple:
            data = itemgetter(-1)(data)
        # use a linear tempo range
        tempi = np.arange(len(data))
        # determine tempo range to consider
        min_idx = np.argmax(tempi >= self.min_bpm)
        max_idx = np.argmin(tempi <= self.max_bpm)
        # return only selected range
        return data[min_idx:max_idx], tempi[min_idx:max_idx]


class TempoEstimationProcessor(OnlineProcessor):
    """
    Tempo Estimation Processor class.

    Parameters
    ----------
    method : {'comb', 'acf', 'dbn', None}
        Method used for tempo histogram creation, e.g. from a beat
        activation function or tempo classification layer.
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
    fps : float, optional
        Frames per second.
    histogram_processor : :class:`TempoHistogramProcessor`, optional
        Processor used to create a tempo histogram.
    interpolate : bool, optional
        Interpolate tempo with quadratic interpolation.

    Examples
    --------
    Create a TempoEstimationProcessor. The returned array represents the
    estimated tempi (given in beats per minute) and their relative strength.

    >>> proc = TempoEstimationProcessor(fps=100)
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.tempo.TempoEstimationProcessor object at 0x...>

    Call this TempoEstimationProcessor with the beat activation function
    obtained by RNNBeatProcessor to estimate the tempi.

    >>> from madmom.features.beats import RNNBeatProcessor
    >>> act = RNNBeatProcessor()('tests/data/audio/sample.wav')
    >>> proc(act)  # doctest: +NORMALIZE_WHITESPACE
    array([[176.47059,  0.47469],
           [117.64706,  0.17667],
           [240.     ,  0.15371],
           [ 68.96552,  0.09864],
           [ 82.19178,  0.09629]])

    """

    def __init__(self, method=METHOD, min_bpm=MIN_BPM, max_bpm=MAX_BPM,
                 act_smooth=ACT_SMOOTH, hist_smooth=HIST_SMOOTH, fps=None,
                 online=False, histogram_processor=None, interpolate=False,
                 **kwargs):
        # pylint: disable=unused-argument
        super(TempoEstimationProcessor, self).__init__(online=online)
        if method is not None:
            warnings.warn(
                'Usage of `method` is deprecated as of version 0.17. '
                'Please pass a dedicated `TempoHistogramProcessor` '
                'instance as `histogram_processor`.'
                'Functionality will be removed in version 0.19.')
            self.method = method
        self.act_smooth = act_smooth
        self.hist_smooth = hist_smooth
        self.fps = fps
        if self.online:
            self.visualize = kwargs.get('verbose', False)
        if histogram_processor is None:
            if method == 'acf':
                histogram_processor = ACFTempoHistogramProcessor
            elif method == 'comb':
                histogram_processor = CombFilterTempoHistogramProcessor
            elif method == 'dbn':
                histogram_processor = DBNTempoHistogramProcessor
                # do not smooth the activations for the DBN
                self.act_smooth = None
            else:
                raise ValueError('tempo histogram method unknown.')
            # instantiate histogram processor
            histogram_processor = histogram_processor(
                min_bpm=min_bpm, max_bpm=max_bpm, fps=fps, online=online,
                **kwargs)
        self.histogram_processor = histogram_processor
        self.fps = fps
        self.hist_smooth = hist_smooth
        self.interpolate = interpolate
        if self.online:
            self.visualize = kwargs.get('verbose', False)

    @property
    def min_bpm(self):
        """Minimum tempo [bpm]."""
        return self.histogram_processor.min_bpm

    @property
    def max_bpm(self):
        """Maximum  tempo [bpm]."""
        return self.histogram_processor.max_bpm

    @property
    def intervals(self):
        """Beat intervals [frames]."""
        return self.histogram_processor.intervals

    @property
    def min_interval(self):
        """Minimum beat interval [frames]."""
        return self.histogram_processor.min_interval

    @property
    def max_interval(self):
        """Maximum beat interval [frames]."""
        return self.histogram_processor.max_interval

    def reset(self):
        """Reset to initial state."""
        self.histogram_processor.reset()

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
        # smooth the activations if needed
        if self.act_smooth is not None:
            act_smooth = int(round(self.fps * self.act_smooth))
            activations = smooth_signal(activations, act_smooth)
        # generate tempo histogram from beat activations/TCN classification
        histogram = self.histogram_processor(activations)
        # smooth the histogram
        histogram = smooth_histogram(histogram, self.hist_smooth)
        # detect the tempi and return them
        return detect_tempo(histogram, self.fps, interpolate=self.interpolate)

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
        # build the tempo histogram depending on the chosen method
        histogram = self.interval_histogram(activations, reset=reset)
        # smooth the histogram
        histogram = smooth_histogram(histogram, self.hist_smooth)
        # detect the tempo and append it to the found tempi
        tempo = detect_tempo(histogram, self.fps, interpolate=self.interpolate)
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
        # return tempo
        return tempo

    def interval_histogram(self, activations, **kwargs):
        """
        Compute the histogram of the beat intervals.

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
        return self.histogram_processor(activations, **kwargs)

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
    def add_arguments(parser, method=None, min_bpm=None, max_bpm=None,
                      act_smooth=None, hist_smooth=None, hist_buffer=None,
                      alpha=None, interpolate=None):
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
        hist_buffer : float, optional
            Aggregate the tempo histogram over `hist_buffer` seconds.
        alpha : float, optional
            Scaling factor for the comb filter.
        interpolate : bool, optional
            Interpolate tempo with quadratic interpolation.

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
        if hist_buffer is not None:
            g.add_argument('--hist_buffer', action='store', type=float,
                           default=hist_buffer,
                           help='aggregate the tempo histogram over N seconds '
                                'in online mode [default=%(default).2f]')
        if alpha is not None:
            g.add_argument('--alpha', action='store', type=float,
                           default=alpha,
                           help='alpha for comb filter tempo estimation '
                                '[default=%(default).2f]')
        if interpolate is not None:
            g.add_argument('--interpolate', action='store_true', default=False,
                           help='interpolate tempo with quadratic '
                                'interpolation')
        # return the argument group so it can be modified if needed
        return g
