#!/usr/bin/env python
# encoding: utf-8
"""
This file contains tempo related functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import numpy as np
from scipy.signal import argrelmax

from madmom import Processor, IOProcessor
from madmom.audio.signal import smooth as smooth_signal
from madmom.features import ActivationsProcessor
from madmom.features.beats import RNNBeatProcessing


NO_TEMPO = np.nan


# helper functions
def smooth_histogram(histogram, smooth):
    """
    Smooth the given histogram.

    :param histogram: histogram
    :param smooth:    smoothing kernel [numpy array or int]
    :return:          smoothed histogram

    Note: If 'smooth' is an integer, a Hamming window of that length will be
          used as a smoothing kernel.

    """
    # smooth only the the histogram bins, not the corresponding delays
    return smooth_signal(histogram[0], smooth), histogram[1]


# interval detection
def interval_histogram_acf(activations, min_tau=1, max_tau=None):
    """
    Compute the interval histogram of the given activation function with via
    auto-correlation.

    :param activations: the activation function
    :param min_tau:     minimal delta for correlation function [frames]
    :param max_tau:     maximal delta for correlation function [frames]
    :return:            histogram

    """
    # set the maximum delay
    if max_tau is None:
        max_tau = len(activations) - min_tau
    # test all possible delays
    taus = range(min_tau, max_tau + 1)
    bins = []
    # TODO: make this processing parallel or numpyfy if possible
    for tau in taus:
        bins.append(np.sum(np.abs(activations[tau:] * activations[0:-tau])))
    # return histogram
    return np.array(bins), np.array(taus)


def interval_histogram_comb(activations, alpha, min_tau=1, max_tau=None):
    """
    Compute the interval histogram of the given activation function via a
    bank of comb filters.

    :param activations: the activation function
    :param alpha:       scaling factor for the comb filter
    :param min_tau:     minimal delta for correlation function [frames]
    :param max_tau:     maximal delta for correlation function [frames]
    :return:            histogram

    """
    # import comb filter
    from madmom.audio.filters import CombFilterbank
    # set the maximum delay
    if max_tau is None:
        max_tau = len(activations) - min_tau
    # get the range of taus
    taus = np.arange(min_tau, max_tau + 1)
    # apply a bank of comb filters
    cfb = CombFilterbank('backward', taus, alpha).process(activations)
    # determine the tau with the highest value for each time step
    # sum up the maxima to yield the histogram bin values
    histogram_bins = np.sum(cfb * (cfb == np.max(cfb, axis=0)), axis=1)
    # return histogram
    return histogram_bins, taus


# helper functions
def dominant_interval(histogram, smooth=None):
    """
    Extract the dominant interval of the given histogram.

    :param histogram: histogram with interval distribution
    :param smooth:    smooth the histogram with the given kernel (size)
    :return:          dominant interval

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

    :param histogram: tempo histogram
    :param fps:       frames per second (needed for conversion to BPM)
    :return:          numpy array with the dominant tempi (first column)
                      and their relative strengths (second column)

    """
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
            return np.asarray([tempi[len(bins) / 2], 1.])
        # otherwise: no peaks, no tempo
        return np.asarray([NO_TEMPO, 0.])
    elif len(peaks) == 1:
        # report only the strongest tempo
        return np.asarray([tempi[peaks[0]], 1.])
    else:
        # sort the peaks in descending order of bin heights
        sorted_peaks = peaks[np.argsort(bins[peaks])[::-1]]
        # normalize their strengths
        strengths = bins[sorted_peaks]
        strengths /= np.sum(strengths)
        # return the tempi and their normalized strengths
        return np.asarray(zip(tempi[sorted_peaks], strengths))


# tempo estimation processor class
class TempoEstimation(Processor):
    """
    Tempo Estimation Processor class.

    """
    # default values for tempo estimation
    METHOD = 'comb'
    MIN_BPM = 40
    MAX_BPM = 250
    HIST_SMOOTH = 7
    ACT_SMOOTH = 0.14
    ALPHA = 0.79

    def __init__(self, method=METHOD, min_bpm=MIN_BPM, max_bpm=MAX_BPM,
                 act_smooth=ACT_SMOOTH, hist_smooth=HIST_SMOOTH, alpha=ALPHA,
                 fps=None, **kwargs):
        """
        Estimates the tempo of the signal.

        :param method:      either 'acf' or 'comb'.
        :param min_bpm:     minimum tempo to detect
        :param max_bpm:     maximum tempo to detect
        :param act_smooth:  smooth the activation function over N seconds
        :param hist_smooth: smooth the activation function over N bins
        :param alpha:       scaling factor for the comb filter

        """
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
        Detect the tempi from the beat activations.

        :param activations: RNN beat activation function
        :return:            numpy array with the dominant tempi (first column)
                            and their relative strengths (second column)

        """
        # smooth the activations
        act_smooth = int(round(self.fps * self.act_smooth))
        activations = smooth_signal(activations, act_smooth)
        # generate a histogram of beat intervals
        histogram = self.interval_histogram(activations)
        # smooth the histogram
        histogram = smooth_histogram(histogram, self.hist_smooth)
        # detect the tempi and return them
        return detect_tempo(histogram, self.fps)

    def interval_histogram(self, activations):
        """
        Compute the histogram of the beat intervals with the selected method.

        :param activations: RNN beat activation function
        :return:            beat interval histogram

        """
        # build the tempo (i.e. inter beat interval) histogram and return it
        if self.method == 'acf':
            return interval_histogram_acf(activations, self.min_interval,
                                          self.max_interval)
        elif self.method == 'comb':
            return interval_histogram_comb(activations, self.alpha,
                                           self.min_interval,
                                           self.max_interval)
        else:
            raise ValueError('tempo estimation method unknown')

    def dominant_interval(self, histogram):
        """
        Extract the dominant interval of the given histogram.

        :param histogram: histogram with interval distribution
        :return:          dominant interval

        """
        # return the dominant interval
        return dominant_interval(histogram, self.hist_smooth)

    @classmethod
    def add_arguments(cls, parser, method=METHOD, min_bpm=MIN_BPM,
                      max_bpm=MAX_BPM, act_smooth=ACT_SMOOTH,
                      hist_smooth=HIST_SMOOTH, alpha=ALPHA):
        """
        Add tempo estimation related arguments to an existing parser.

        :param parser:      existing argparse parser
        :param method:      either 'acf' or 'comb'.
        :param min_bpm:     minimum tempo [bpm]
        :param max_bpm:     maximum tempo [bpm]
        :param act_smooth:  smooth the activations over N seconds
        :param hist_smooth: smooth the tempo histogram over N bins
        :param alpha:       scaling factor of the comb filter
        :return:            tempo argument parser group

        """
        # add tempo estimation related options to the existing parser
        g = parser.add_argument_group('tempo estimation arguments')
        if method is not None:
            g.add_argument('--method', action='store', type=str,
                           default=method, choices=['acf', 'comb'],
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

    :param tempi:     tempi present
    :param filename:  output file name or file handle
    :param mirex:     report the lower tempo first (as required by MIREX)
    :return:          the most dominant tempi and the relative strength

    """
    from madmom.utils import open
    # default values
    t1, t2, strength = 0., 0., 1.
    # only one tempo was detected
    if len(tempi) == 1:
        t1 = tempi[0][0]
        # generate a fake second tempo
        if t1 > 120:
            t2 = t1 / 2.
        else:
            t2 = t1 * 2.
    # consider only the two strongest tempi and strengths
    elif len(tempi) > 1:
        t1, t2 = tempi[:2, 0]
        strength = tempi[0, 1] / sum(tempi[:2, 1])
    # for MIREX, the lower tempo must be given first
    if mirex and t1 > t2:
        t1, t2, strength = t2, t1, 1. - strength
    # write to output
    with open(filename, 'wb') as f:
        f.write("%.2f\t%.2f\t%.2f\n" % (t1, t2, strength))
    # also return the tempi & strength
    return t1, t2, strength


# wrapper function to be used as output of TempoEstimation
from functools import partial
write_tempo_mirex = partial(write_tempo, mirex=True)
write_tempo_mirex.__doc__ = 'write_tempo(tempo, filename, mirex=True)'


# RNN tempo estimation processor class
class RNNTempoEstimation(IOProcessor):
    """
    Tempo Estimation Processor class.

    """
    def __init__(self, mirex=False, load=False, save=False, **kwargs):
        """
        Estimates the tempo of the signal.

        :param mirex:
        :param load:
        :param save:

        """
        # input processing
        in_processor = RNNBeatProcessing(**kwargs)
        # TODO: this is super hackish, split RNNBeatTracking in RNN & writing
        #       parts!
        in_processor.out_processor = None
        self.fps = kwargs['fps'] = in_processor.fps
        # output processor
        writer = write_tempo_mirex if mirex else write_tempo
        out_processor = [TempoEstimation(**kwargs), writer]
        # swap in/out processors if needed
        if load:
            in_processor = ActivationsProcessor(mode='r', **kwargs)
        if save:
            out_processor = ActivationsProcessor(mode='w', **kwargs)
        # make this an IOProcessor by defining input and output processors
        super(RNNTempoEstimation, self).__init__(in_processor, out_processor)

