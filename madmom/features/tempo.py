#!/usr/bin/env python
# encoding: utf-8
"""
This file contains tempo related functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import numpy as np
from scipy.signal import argrelmax

from . import Event


# helper functions
def smooth_signal(signal, smooth):
    """
    Smooth the given signal.

    :param signal: signal
    :param smooth: smoothing kernel [array or int]
    :returns:      smoothed signal

    """
    # init smoothing kernel
    kernel = None
    # size for the smoothing kernel is given
    if isinstance(smooth, int):
        if smooth > 1:
            kernel = np.hamming(smooth)
    # otherwise use the given smoothing kernel directly
    elif isinstance(smooth, np.ndarray):
        if len(smooth) > 1:
            kernel = smooth
    # check if a kernel is given
    if kernel is None:
        raise ValueError('can not smooth signal with %s' % smooth)
    # convolve with the kernel and return
    return np.convolve(signal, kernel, 'same')


def smooth_histogram(histogram, smooth):
    """
    Smooth the given histogram.

    :param histogram: histogram
    :param smooth:    smoothing kernel [array or int]
    :returns:         smoothed histogram

    """
    # smooth only the the histogram bins, not the
    return smooth_signal(histogram[0], smooth), histogram[1]


# interval detection
def interval_histogram(activations, threshold=0, smooth=None, min_tau=1,
                       max_tau=None):
    """
    Compute the interval histogram of the given activation function.

    :param activations: the activation function
    :param threshold:   threshold for the activation function before
                        auto-correlation
    :param smooth:      kernel (size) for smoothing the activation function
                        before auto-correlating it. [array or int]
    :param min_tau:     minimal delta for correlation function [frames]
    :param max_tau:     maximal delta for correlation function [frames]
    :returns:           histogram

    """
    # smooth activations
    if smooth:
        activations = smooth_signal(activations, smooth)
    # threshold function if needed
    if threshold > 0:
        activations[activations < threshold] = 0
    # set the maximum delay
    if max_tau is None:
        max_tau = len(activations) - min_tau
    # test all possible delays
    taus = range(min_tau, max_tau)
    bins = []
    # TODO: make this processing parallel or numpyfy if possible
    for tau in taus:
        bins.append(np.sum(np.abs(activations[tau:] * activations[0:-tau])))
    # return histogram
    return np.array(bins), np.array(taus)


def dominant_interval(histogram, smooth=None):
    """
    Extract the dominant interval of the given histogram.

    :param histogram: histogram with interval distribution
    :param smooth:    smooth the histogram with the kernel
    :returns:         dominant interval

    """
    # smooth the histogram bins
    if smooth:
        histogram = smooth_histogram(histogram, smooth)
    # return the dominant interval
    return histogram[1][np.argmax(histogram[0])]


# default values for tempo estimation
THRESHOLD = 0
SMOOTH = 0.09
MIN_BPM = 60
MAX_BPM = 240
DELAY = 0


class Tempo(Event):
    """
    Tempo Class.

    """
    def __init__(self, activations, fps, sep=''):
        """
        Creates a new Tempo instance with the given activations.
        The activations can be read in from file.

        :param activations: array with the beat activations or a file (handle)
        :param fps:         frame rate of the activations
        :param sep:         separator if activations are read from file

        """
        super(Tempo, self).__init__(activations, fps, sep)

    def detect(self, threshold=THRESHOLD, smooth=SMOOTH, min_bpm=MIN_BPM,
               max_bpm=MAX_BPM, mirex=False):
        """
        Detect the tempo on basis of the given beat activation function.

        :param threshold: threshold for peak-picking
        :param smooth:    smooth the activation function over N seconds
        :param min_bpm:   minimum tempo used for beat tracking
        :param max_bpm:   maximum tempo used for beat tracking
        :param mirex:     always output the lower tempo first
        :returns:         tuple with the two most dominant tempi and the
                          relative weight of them

        """
        # convert the arguments to frames
        smooth = int(round(self.fps * smooth))
        min_tau = int(np.floor(60. * self.fps / max_bpm))
        max_tau = int(np.ceil(60. * self.fps / min_bpm))
        # generate a histogram of beat intervals
        histogram = interval_histogram(self.activations, threshold,
                                       smooth=smooth, min_tau=min_tau,
                                       max_tau=max_tau)
        # smooth the histogram again
        if smooth:
            histogram = smooth_histogram(histogram, smooth=None)
        # the histogram bins
        bins = histogram[0]
        # convert the histogram bin delays to tempi in beats per minute
        tempi = 60.0 * self.fps / histogram[1]
        # to get the two dominant tempi, just keep the peaks
        # use 'wrap' mode to also get peaks at the borders
        peaks = argrelmax(bins, mode='wrap')[0]
        # get the weights of the peaks to sort them in descending order
        strengths = bins[peaks]
        sorted_peaks = peaks[np.argsort(strengths)[::-1]]
        # we need more than 1 peak to report multiple tempi
        if len(sorted_peaks) < 2:
            # return tempi[sorted_peaks[0]], np.nan, 1.
            raise AssertionError('this should not happen!')
        # get the 2 strongest tempi
        t1, t2 = tempi[sorted_peaks[:2]]
        # calculate the relative strength
        strength = bins[sorted_peaks[0]]
        strength /= np.sum(bins[sorted_peaks[:2]])
        # return the tempi + the relative strength
        if mirex and t1 > t2:
            # for MIREX, the lower tempo must be given first
            return t2, t1, 1. - strength
        return t1, t2, strength


# def gkiokas_2010(mel_features, pcp_features):
#     """
#     Estimate the tempo using filter bank analysis and tonal features.
#
#     "Tempo Induction Using Filterbank Analysis and Tonal Features"
#     A. Gkiokas, V. Katsouros, G. Carayannis
#     Proceedings of the 11th International Society for Music Information
#     Retrieval Conference (ISMIR 2010), Utrecht, Netherlands
#
#     :param mel_features: log energies of Mel filtered spectrogram features
#     :param pcp_features: pitch class profile (tonal) features
#
#     """
#     # convolve both features with a bank of resonators
#     raise NotImplementedError
#
#
# def gkiokas_2012(spec):
#     """
#     Estimate the tempo on harmonic/percussive separated spectrograms.
#
#     :param spec: the (constant-Q) spectrogram
#
#     "Music tempo estimation and beat tracking by applying source separation and
#      metrical relations"
#     A. Gkiokas, V. Katsouros, G. Carayannis and T. Stafylakis
#     Proceedings of the 37th International Conference on Acoustics, Speech and
#     Signal Processing (ICASSP 2012), Kyoto, Japan
#
#     """
#     # TODO: check the influence/benefit of CQT compared to our spectrogram
#     #       implementation
#     raise NotImplementedError
#
#
# class TempoEstimation(object):
#
#     def __init__(self, spectrogram, *args, **kwargs):
#         """
#         Creates a new TempoEstimation instance.
#
#         :param spectrogram: the spectrogram object on which the tempo
#                             estimation operates
#
#         """
#         # import
#         from ..audio.spectrogram import Spectrogram
#
#         # check spectrogram type
#         if isinstance(spectrogram, Spectrogram):
#             # already the right format
#             self._spectrogram = spectrogram
#         else:
#             # assume a file name, try to instantiate a Spectrogram object
#             self._spectrogram = Spectrogram(spectrogram, *args, **kwargs)
#
#     @property
#     def spectrogram(self):
#         """Spectrogram."""
#         return self._spectrogram
#
#     def gkiokas_2010(self):
#         """
#         Estimate the tempo using filter bank analysis and tonal features.
#
#         """
#         from ..audio.filters import (MelFilterbank as Mel_FB,
#                                         PitchClassProfileFilterbank as Pcp_FB)
#         # first 12 features are the log energy of 12 Mel bands
#         # TODO: use a smaller frame_size for this! (20ms, 5ms hop)
#         mel_fb = Mel_FB(self.spectrogram.num_fft_bins,
#                         self.spectrogram.frames.signal.sample_rate,
#                         bands=12)
#         # init a PCP filter bank
#         # TODO: use a larger frame_size for this! (80ms, 5ms hop)
#         pcp_fb = Pcp_FB(self.spectrogram.num_fft_bins,
#                         self.spectrogram.frames.signal.sample_rate,
#                         fmin=27.5, fmax=10000)
#
#     @property
#     def pcp(self):
#         """Pitch Class Profile."""
#         if self._pcp is None:
#             # map the spectrogram to pitch classes
#             self._pcp = np.dot(self._spectrogram.spec, self._pcp_fb)
#         return self._pcp
#
#
# def parser():
#     """
#     Command line argument parser for tempo detection.
#
#     """
#     import argparse
#     from ..utils.params import (audio, spec, filtering, log, spectral_odf,
#                                 save_load)
#
#     # define parser
#     p = argparse.ArgumentParser(
#         formatter_class=argparse.RawDescriptionHelpFormatter, description="""
#     If invoked without any parameters, the software detects the tempo in
#     the given files.
#
#     """)
#     # general options
#     p.add_argument('files', metavar='files', nargs='+',
#                    help='files to be processed')
#     p.add_argument('-v', dest='verbose', action='store_true',
#                    help='be verbose')
#     p.add_argument('--ext', action='store', type=str, default='txt',
#                    help='extension for detections [default=txt]')
#     # add other argument groups
#     audio(p, online=False)
#     spec(p)
#     filtering(p, default=True)
#     log(p, default=True)
#     # spectral_odf(p)
#     # o = Tempo(p)
#     # list of offered tempo estimation methods
#     methods = ['TempoDetector', 'Gkiokas2010', 'Gkiokas2012', 'Scheirer1998']
#     p.add_argument('-m', dest='method', default='TempoDetector',
#                    help='use this tempo estimation method %s' % methods)
#     save_load(p)
#     # parse arguments
#     args = p.parse_args()
#     # print arguments
#     if args.verbose:
#         print args
#     # return args
#     return args
#
#
# def main():
#     """
#     Example tempo estimation program.
#
#     """
#     import os.path
#
#     from ..utils import files
#     from ..audio.wav import Wav
#     from ..audio.spectrogram import Spectrogram
#     from ..audio.filters import LogarithmicFilterbank
#
#     # parse arguments
#     args = parser()
#
#     # TODO: also add an option for evaluation and load the targets accordingly
#     # see cp.evaluation.helpers.match_files()
#
#     # init filterbank
#     fb = None
#
#     # which files to process
#     if args.load:
#         # load the activations
#         ext = '.activations'
#     else:
#         # only process .wav files
#         ext = '.wav'
#     # process the files
#     for f in files(args.files, ext):
#         if args.verbose:
#             print f
#
#         # use the name of the file without the extension
#         filename = os.path.splitext(f)[0]
#
#         # do the processing stuff unless the activations are loaded from file
#         if args.load:
#             # load the activations from file
#             # FIXME: fps must be encoded in the file
#             t = Tempo(f, args.fps, args.online)
#         else:
#             # create a Wav object
#             w = Wav(f, mono=True, norm=args.norm, att=args.att)
#             if args.filter:
#                 # (re-)create filterbank if the sample rate is not the same
#                 if fb is None or fb.sample_rate != w.sample_rate:
#                     # create filterbank if needed
#                     fb = LogarithmicFilterbank(num_fft_bins=args.window / 2,
#                                                sample_rate=w.sample_rate,
#                                                bands_per_octave=args.bands,
#                                                fmin=args.fmin, fmax=args.fmax,
#                                                norm=args.equal)
#             # create a Spectrogram object
#             s = Spectrogram(w, frame_size=args.window, filterbank=fb,
#                             log=args.log, mul=args.mul, add=args.add,
#                             ratio=args.ratio, diff_frames=args.diff_frames)
#             # create a SpectralTempoDetection object
#             te = TempoEstimation(s, max_bins=args.max_bins)
#             # perform detection function on the object
#             act = getattr(te, args.odf)()
#             # create an Tempo object with the activations
#             t = Tempo(act, args.fps, args.online)
#         # save Tempo activations or detect Tempos
#         if args.save:
#             # save the raw ODF activations
#             t.save_activations("%s.%s" % (filename, args.odf))
#         else:
#             # detect the Tempos
#             t.detect(args.threshold, smooth=args.smooth, min_bpm=args.min_bpm,
#                      max_bpm=args.max_bpm)
#             # write the Tempos to a file
#             t.write("%s.%s" % (filename, args.ext))
#             # also output them to stdout if verbose
#             if args.verbose:
#                 print 'detections:', t.detections
#         # continue with next file
#
# if __name__ == '__main__':
#     main()
#