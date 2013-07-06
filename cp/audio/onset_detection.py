#!/usr/bin/env python
# encoding: utf-8
"""
Copyright (c) 2013 Sebastian Böck <sebastian.boeck@jku.at>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import scipy.ndimage as sim


# helper functions
def wraptopi(phase):
    """
    Wrap the phase information to the range -π...π.

    :param phase: phase spectrogram
    :returns: wrapped phase spectrogram

    """
    return np.mod(phase + np.pi, 2.0 * np.pi) - np.pi


def diff(spec, diff_frames=1, pos=False):
    """
    Calculates the difference of the magnitude spectrogram.

    :param spec: the magnitude spectrogram
    :param diff_frames: calculate the difference to the N-th previous frame [default=1]
    :param pos: only keep positive values [default=False]
    :returns: (positive) magnitude spectrogram differences

    """
    # init the matrix with 0s, the first N rows are 0 then
    # TODO: under some circumstances it might be helpful to init with the spec
    # or use the frame at "real" index -N to calculate the diff to
    diff = np.zeros_like(spec)
    if diff_frames < 1:
        raise ValueError("number of diff_frames must be >= 1")
    # calculate the diff
    diff[diff_frames:] = spec[diff_frames:] - spec[:-diff_frames]
    # keep only positive values
    if pos:
        diff = diff * (diff > 0)
    return diff


def correlation_diff(self, spec, diff_frames=1, pos=False, diff_bins=1):
    """
    Calculates the difference of the magnitude spectrogram relative to the
    N-th previous frame shifted in frequency to achieve the highest
    correlation between these two frames.

    :param spec: the magnitude spectrogram
    :param diff_frames: calculate the difference to the N-th previous frame [default=1]
    :param pos: only keep positive values [default=False]
    :param diff_bins: maximum number of bins shifted for correlation calculation [default=1]
    :returns: (positive) magnitude spectrogram differences

    """
    # init diff matrix
    diff = np.zeros_like(spec)
    if diff_frames < 1:
        raise ValueError("number of diff_frames must be >= 1")
    # calculate the diff
    frames, bins = diff.shape
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
        diff[f, diff_bins:-diff_bins] = spec[f, diff_bins:-diff_bins] - spec[f - diff_frames, bin_start:bin_stop]
    # keep only positive values
    if pos:
        diff = diff * (diff > 0)
    return diff


# Onset Detection Functions
def high_frequency_content(spec):
    """
    High Frequency Content.

    :param spec: the magnitude spectrogram
    :returns: high frequency content onset detection function

    "Computer Modeling of Sound for Transformation and Synthesis of Musical Signals"
    Paul Masri
    PhD thesis, University of Bristol, 1996

    """
    # HFC weights the magnitude spectrogram by the bin number, thus emphasising high frequencies
    return np.mean(spec * np.arange(spec.shape[1]), axis=1)


def spectral_diff(spec, diff_frames=1):
    """
    Spectral Diff.

    :param spec: the magnitude spectrogram
    :param diff_frames: calculate the difference to the N-th previous frame [default=1]
    :returns: spectral diff onset detection function

    "A hybrid approach to musical note onset detection"
    Chris Duxbury, Mark Sandler and Matthew Davis
    Proceedings of the 5th International Conference on Digital Audio Effects (DAFx-02), 2002.

    """
    # Spectral diff is the sum of all squared positive 1st order differences
    return np.sum(diff(spec, diff_frames=diff_frames, pos=True) ** 2, axis=1)


def spectral_flux(spec, diff_frames=1):
    """
    Spectral Flux.

    :param spec: the magnitude spectrogram
    :param diff_frames: calculate the difference to the N-th previous frame [default=1]
    :returns: spectral flux onset detection function

    "Computer Modeling of Sound for Transformation and Synthesis of Musical Signals"
    Paul Masri
    PhD thesis, University of Bristol, 1996

    """
    # Spectral flux is the sum of all positive 1st order differences
    return np.sum(diff(spec, diff_frames=diff_frames, pos=True), axis=1)


def superflux(spec, diff_frames=1, max_bins=3):
    """
    SuperFlux with a maximum peak-tracking stage for difference calculation.

    Calculates the difference of bin k of the magnitude spectrogram relative to
    the N-th previous frame with a maximum filter (in the frequency axis) applied.

    :param spec: the magnitude spectrogram
    :param diff_frames: calculate the difference to the N-th previous frame [default=1]
    :param max_bins: number of neighboring bins used for maximum filtering [default=3]
    :returns: SuperFlux onset detection function

    "Maximum Filter Vibrato Suppression for Onset Detection"
    Sebastian Böck and Gerhard Widmer
    Proceedings of the 16th International Conference on Digital Audio Effects (DAFx-13), 2013.

    Note: this method works only properly, if the spectrogram is filtered with
    a filterbank of the right frequency spacing. Filterbanks with 24 bands per
    octave (i.e. quartertone resolution) usually yield good results. With the
    default 3 max_bins, the maximum of the bins k-1, k, k+1 of the frame
    diff_frames to the left is used for the calculation of the difference.

    """
    # init diff matrix
    diff = np.zeros_like(spec)
    if diff_frames < 1:
        raise ValueError("number of diff_frames must be >= 1")
    # calculate the diff
    diff[diff_frames:] = spec[diff_frames:] - sim.maximum_filter(spec, size=[1, max_bins])[0:-diff_frames]
    # keep only positive values
    diff = diff * (diff > 0)
    # SuperFlux is the sum of all positive 1st order maximum filtered differences
    return np.sum(diff, axis=1)


def modified_kullback_leibler(spec, diff_frames=1, epsilon=0.000001):
    """
    Modified Kullback-Leibler.

    :param spec: the magnitude spectrogram
    :param diff_frames: calculate the difference to the N-th previous frame [default=1]
    :param epsilon: add epsilon to avoid division by 0 [default=0.000001]
    :returns: MKL onset detection function

    Note: the implenmentation presented in:
    "Automatic Annotation of Musical Audio for Interactive Applications"
    Paul Brossier
    PhD thesis, Queen Mary University of London, 2006

    is used instead of the original work:
    "Onset Detection in Musical Audio Signals"
    Stephen Hainsworth and Malcolm Macleod
    Proceedings of the International Computer Music Conference (ICMC), 2003

    """
    if epsilon <= 0:
        raise ValueError("a positive value must be added before division")
    modified_kullback_leibler = np.zeros_like(spec)
    modified_kullback_leibler[diff_frames:] = spec[diff_frames:] / (spec[:-diff_frames] + epsilon)
    # note: the original MKL uses sum instead of mean, but the range of mean is much more suitable
    return np.mean(np.log(1 + modified_kullback_leibler), axis=1)


def _phase_deviation(phase):
    """
    Helper method used by phase_deviation() & weighted_phase_deviation().

    :param phase: the phase spectrogram
    :returns: phase deviation

    """
    pd = np.zeros_like(phase)
    # instantaneous frequency is given by the first difference ψ′(n, k) = ψ(n, k) − ψ(n − 1, k)
    # change in instantaneous frequency is given by the second order difference ψ′′(n, k) = ψ′(n, k) − ψ′(n − 1, k)
    pd[2:] = phase[2:] - 2 * phase[1:-1] + phase[:-2]
    # map to the range -pi..pi
    return wraptopi(pd)


def phase_deviation(phase):
    """
    Phase Deviation.

    :param phase: the phase spectrogram
    :returns: phase deviation onset detection function

    "On the use of phase and energy for musical onset detection in the complex domain"
    Juan Pablo Bello, Chris Duxbury, Matthew Davies and Mark Sandler
    IEEE Signal Processing Letters, Volume 11, Number 6, 2004

    """
    # take the mean of the absolute changes in instantaneous frequency
    return np.mean(np.abs(_phase_deviation(phase)), axis=1)


def weighted_phase_deviation(spec, phase):
    """
    Weighted Phase Deviation.

    :param spec: the magnitude spectrogram
    :param phase: the phase spectrogram
    :returns: weighted phase deviation onset detection function

    "Onset Detection Revisited"
    Simon Dixon
    Proceedings of the 9th International Conference on Digital Audio Effects (DAFx), 2006

    """
    # make sure the spectrogram is not filtered before
    if np.shape(phase) != np.shape(spec):
        raise ValueError("Magnitude spectrogram and phase must be of same shape")
    # weighted_phase_deviation = spec * phase_deviation
    return np.mean(np.abs(_phase_deviation(phase) * spec), axis=1)


def normalized_weighted_phase_deviation(spec, phase, epsilon=0.000001):
    """
    Normalized Weighted Phase Deviation.

    :param spec: the magnitude spectrogram
    :param phase: the phase spectrogram
    :param epsilon: add epsilon to avoid division by 0 [default=0.000001]
    :returns: normalized weighted phase deviation onset detection function

    "Onset Detection Revisited"
    Simon Dixon
    Proceedings of the 9th International Conference on Digital Audio Effects (DAFx), 2006

    """
    if epsilon <= 0:
        raise ValueError("a positive value must be added before division")
    # normalize WPD by the sum of the spectrogram (add a small amount so that we don't divide by 0)
    return weighted_phase_deviation(spec, phase) / np.add(np.mean(spec, axis=1), epsilon)


def _complex_domain(spec, phase):
    """
    Helper method used by complex_domain() & rectified_complex_domain().

    :param spec: the magnitude spectrogram
    :param phase: the phase spectrogram
    :returns: complex domain

    Note: we use the simple implementation presented in:
    "Onset Detection Revisited"
    Simon Dixon
    Proceedings of the 9th International Conference on Digital Audio Effects (DAFx), 2006

    """
    if np.shape(phase) != np.shape(spec):
        raise ValueError("Magnitude spectrogram and phase must be of same shape")
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

    :param spec: the magnitude spectrogram
    :param phase: the phase spectrogram
    :returns: complex domain onset detection function

    "On the use of phase and energy for musical onset detection in the complex domain"
    Juan Pablo Bello, Chris Duxbury, Matthew Davies and Mark Sandler
    IEEE Signal Processing Letters, Volume 11, Number 6, 2004

    """
    # take the sum of the absolute changes
    return np.sum(np.abs(_complex_domain(spec, phase)), axis=1)


def rectified_complex_domain(spec, phase):
    """
    Rectified Complex Domain.

    :param spec: the magnitude spectrogram
    :param phase: the phase spectrogram
    :returns: recified complex domain onset detection function

    "Onset Detection Revisited"
    Simon Dixon
    Proceedings of the 9th International Conference on Digital Audio Effects (DAFx), 2006

    """
    # rectified complex domain
    rcd = _complex_domain(spec, phase)
    # only keep values where the magnitude rises
    rcd *= diff(spec, pos=True)
    # take the sum of the absolute changes
    return np.sum(np.abs(rcd), axis=1)


class SpectralODF(object):
    """
    The SpectralODF class implements most of the common onset detection function
    based on the magnitude or phase information of a spectrogram.

    """
    def __init__(self, spectrogram, ratio=0.5, diff_frames=None):
        """
        Creates a new ODF object instance.

        :param spectrogram: the spectrogram object on which the detections functions operate
        :param ratio: calculate the difference to the frame which has the given magnitude ratio [default=0.5]
        :param diff_frames: calculate the difference to the N-th previous frame [default=None]

        """
        # import
        from spectrogram import Spectrogram
        # check spectrogram type
        if isinstance(spectrogram, Spectrogram):
            # already the right format
            self.s = spectrogram
        else:
            # try to convert
            self.s = Spectrogram(spectrogram)
        # determine the number off diff frames
        if diff_frames is None:
            # get the first sample with a higher magnitude than given ratio
            sample = np.argmax(self.s.window > ratio)
            diff_samples = self.s.window.size / 2 - sample
            # convert to frames
            diff_frames = int(round(diff_samples / self.s.hop_size))
            # set the minimum to 1
            if diff_frames < 1:
                diff_frames = 1
        # sanity check
        if diff_frames < 1:
            raise ValueError("number of diff_frames must be >= 1")
        self.diff_frames = diff_frames

    # Onset Detection Functions
    def hfc(self):
        """High Frequency Content."""
        return high_frequency_content(self.s.spec)

    def sd(self):
        """Spectral Diff."""
        return spectral_diff(self.s.spec, self.diff_frames)

    def sf(self):
        """Spectral Flux."""
        return spectral_flux(self.s.spec, self.diff_frames)

    def superflux(self, max_bins=3):
        """
        SuperFlux.

        :param max_bins: number of bins for the maximum filter [default=3]

        """
        return superflux(self.s.spec, self.diff_frames, max_bins)

    def mkl(self):
        """Modified Kullback-Leibler."""
        return modified_kullback_leibler(self.s.spec, self.diff_frames)

    def pd(self):
        """Phase Deviation."""
        return phase_deviation(self.s.phase)

    def wpd(self):
        """Weighted Phase Deviation."""
        return weighted_phase_deviation(self.s.spec, self.s.phase)

    def nwpd(self):
        """Normalized Weighted Phase Deviation."""
        return normalized_weighted_phase_deviation(self.s.spec, self.s.phase)

    def cd(self):
        """Complex Domain."""
        return complex_domain(self.s.spec, self.s.phase)

    def rcd(self):
        """Rectified Complex Domain."""
        return rectified_complex_domain(self.s.spec, self.s.phase)


class Onset(object):
    """
    Onset Class.

    """
    def __init__(self, activations, fps, online=True):
        """
        Creates a new Onset object instance with the given activations of the
        ODF (OnsetDetectionFunction). The activations can be read in from a file.

        :param activations: an array containing the activations of the ODF
        :param fps: frame rate of the activations
        :param online: work in online mode (i.e. use only past information) [default=True]

        """
        self.activations = None     # activations of the ODF
        self.fps = fps              # framerate of the activation function
        self.online = online        # online peak-picking
        self.detections = []        # list of detected onsets (in seconds)
        # set / load activations
        # TODO: decide whether we should go the common way and accept a file
        # here and go up the hierachy by creating a SpectralODF object and
        # perform a default onset detection function (e.g. superflux())
        if isinstance(activations, np.ndarray):
            # activations are given as an array
            self.activations = activations
        else:
            # read in the activations from a file
            self.load(activations)

    def detect(self, threshold, combine=30, pre_avg=100, pre_max=30, post_avg=30, post_max=70, delay=0):
        """
        Detects the onsets.

        :param threshold: threshold for peak-picking
        :param combine: only report 1 onset for N miliseconds [default=30]
        :param pre_avg: use N miliseconds past information for moving average [default=100]
        :param pre_max: use N miliseconds past information for moving maximum [default=30]
        :param post_avg: use N miliseconds future information for moving average [default=0]
        :param post_max: use N miliseconds future information for moving maximum [default=40]
        :param delay: report the onset N miliseconds delayed [default=0]

        In online mode, post_avg and post_max are set to 0.

        Implements the peak-picking method described in:

        "Evaluating the Online Capabilities of Onset Detection Methods"
        Sebastian Böck, Florian Krebs and Markus Schedl
        Proceedings of the 13th International Society for Music Information Retrieval Conference (ISMIR), 2012

        """
        # online mode?
        if self.online:
            post_max = 0
            post_avg = 0
        # convert timing information to frames
        pre_avg = int(round(self.fps * pre_avg / 1000.))
        pre_max = int(round(self.fps * pre_max / 1000.))
        post_max = int(round(self.fps * post_max / 1000.))
        post_avg = int(round(self.fps * post_avg / 1000.))
        # convert to seconds
        combine /= 1000.
        delay /= 1000.
        # init detections
        self.detections = []
        # moving maximum
        max_length = pre_max + post_max + 1
        max_origin = int(np.floor((pre_max - post_max) / 2))
        mov_max = sim.filters.maximum_filter1d(self.activations, max_length, mode='constant', origin=max_origin)
        # moving average
        avg_length = pre_avg + post_avg + 1
        avg_origin = int(np.floor((pre_avg - post_avg) / 2))
        mov_avg = sim.filters.uniform_filter1d(self.activations, avg_length, mode='constant', origin=avg_origin)
        # detections are activation equal to the maximum
        detections = self.activations * (self.activations == mov_max)
        # detections must be greater or equal than the moving average + threshold
        detections = detections * (detections >= mov_avg + threshold)
        # convert detected onsets to a list of timestamps
        last_onset = 0
        for i in np.nonzero(detections)[0]:
            onset = float(i) / float(self.fps) + delay
            # only report an onset if the last N miliseconds none was reported
            if onset > last_onset + combine:
                self.detections.append(onset)
                # save last reported onset
                last_onset = onset

    def write(self, filename):
        """
        Write the detected onsets to the given file.

        :param filename: the target file name

        Only useful if detect() was invoked before.

        """
        with open(filename, 'w') as f:
            for pos in self.detections:
                f.write(str(pos) + '\n')

    def save(self, filename):
        """
        Save the onset activations to the given file.

        :param filename: the target file name

        """
        self.activations.tofile(filename)

    def load(self, filename):
        """
        Load the onset activations from the given file.

        :param filename: the target file name

        """
        self.activations = np.fromfile(filename)


def _parser():
    """
    Parses the command line arguments.

    """
    import argparse
    # define parser
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    If invoked without any parameters, the software detects all onsets in
    the given files in online mode according to the method proposed in:

    "Maximum Filter Vibrato Suppression for Onset Detection"
    by Sebastian Böck and Gerhard Widmer
    in Proceedings of the 16th International Conference on Digital Audio Effects
    (DAFx-13), Maynooth, Ireland, September 2013

    """)
    # general options
    p.add_argument('files', metavar='files', nargs='+', help='files to be processed')
    p.add_argument('-v', dest='verbose', action='store_true', help='be verbose')
    p.add_argument('-s', dest='save', action='store_true', default=False, help='save the activations of the onset detection functions')
    p.add_argument('-l', dest='load', action='store_true', default=False, help='load the activations of the onset detection functions')
    # online / offline mode
    p.add_argument('--offline', dest='online', action='store_false', default=True, help='operate in offline mode')
    # wav options
    wav = p.add_argument_group('audio arguments')
    wav.add_argument('--norm', action='store_true', default=None, help='normalize the audio [switches to offline mode]')
    wav.add_argument('--att', action='store', type=float, default=None, help='attenuate the audio by ATT dB')
    # spectrogram options
    spec = p.add_argument_group('spectrogram arguments')
    spec.add_argument('--fps', action='store', default=200, type=int, help='frames per second')
    spec.add_argument('--window', action='store', type=int, default=2048, help='Hanning window length')
    spec.add_argument('--ratio', action='store', type=float, default=0.5, help='window magnitude ratio to calc number of diff frames')
    spec.add_argument('--diff_frames', action='store', type=int, default=None, help='diff frames')
    spec.add_argument('--max_bins', action='store', type=int, default=3, help='bins used for maximum filtering [default=3]')
    # spec-processing
    pre = p.add_argument_group('pre-processing arguments')
    # filter
    pre.add_argument('--filter', action='store_true', default=None, help='filter the magnitude spectrogram with a filterbank')
    pre.add_argument('--fmin', action='store', default=27.5, type=float, help='minimum frequency of filter in Hz [default=27.5]')
    pre.add_argument('--fmax', action='store', default=16000, type=float, help='maximum frequency of filter in Hz [default=16000]')
    pre.add_argument('--bands', action='store', type=int, default=24, help='number of bands per octave [default=24]')
    pre.add_argument('--equal', action='store_true', default=False, help='equalize triangular windows to have equal area')
    # logarithm
    pre.add_argument('--log', action='store_true', default=None, help='logarithmic magnitude')
    pre.add_argument('--mul', action='store', default=1, type=float, help='multiplier (before taking the log) [default=1]')
    pre.add_argument('--add', action='store', default=1, type=float, help='value added (before taking the log) [default=1]')
    # onset detection
    onset = p.add_argument_group('onset detection arguments')
    onset.add_argument('-o', dest='odf', default=None, help='use this onset detection function [superflux,spectral_flux,sfc,sft]')
    onset.add_argument('-t', dest='threshold', action='store', type=float, default=1.25, help='detection threshold')
    onset.add_argument('--combine', action='store', type=float, default=30, help='combine onsets within N miliseconds [default=30]')
    onset.add_argument('--pre_avg', action='store', type=float, default=100, help='build average over N previous miliseconds [default=100]')
    onset.add_argument('--pre_max', action='store', type=float, default=30, help='search maximum over N previous miliseconds [default=30]')
    onset.add_argument('--post_avg', action='store', type=float, default=70, help='build average over N following miliseconds [default=70]')
    onset.add_argument('--post_max', action='store', type=float, default=30, help='search maximum over N following miliseconds [default=30]')
    onset.add_argument('--delay', action='store', type=float, default=0, help='report the onsets N miliseconds delayed [default=0]')
    # version
    p.add_argument('--version', action='version', version='%(prog)spec 1.0 (2013-04-14)')
    # parse arguments
    args = p.parse_args()

    # list of offered ODFs
    methods = ['superflux', 'hfc', 'sd', 'sf', 'mkl', 'pd', 'wpd', 'nwpd', 'cd', 'rcd']
    # use default values if no ODF is given
    if args.odf is None:
        args.odf = 'superflux'
        if args.log is None:
            args.log = True
        if args.filter is None:
            args.filter = True
    # remove mistyped methods
    if args.odf not in methods:
        raise ValueError("at least one valid onset detection function must be given")

    # print arguments
    if args.verbose:
        print args

    # return args
    return args


def main():
    """
    Example onset detection program.

    """
    import os.path
    import glob
    import fnmatch

    from wav import Wav
    from spectrogram import Spectrogram
    from filterbank import CQFilter

    # parse arguments
    args = _parser()

    # determine the files to process
    files = []
    for f in args.files:
        # check what we have (file/path)
        if os.path.isdir(f):
            # use all files in the given path
            files = glob.glob(f + '/*.wav')
        else:
            # file was given, append to list
            files.append(f)

    # only process .wav files
    files = fnmatch.filter(files, '*.wav')
    files.sort()

    # init filterbank
    filt = None

    # process the files
    for f in files:
        if args.verbose:
            print f

        # use the name of the file without the extension
        filename = os.path.splitext(f)[0]

        # init Onset object
        o = None
        # do the processing stuff unless the activations are loaded from file
        if args.load:
            # load the activations from file
            o = Onset("%s.%s" % (filename, args.odf), args.fps, args.online)
        else:
            # open the wav file
            w = Wav(f)
            # normalize audio
            if args.norm:
                w.normalize()
                args.online = False  # switch to offline mode
            # downmix to mono
            if w.channels > 1:
                w.downmix()
            # attenuate signal
            if args.att:
                w.attenuate(args.att)
            # spectrogram
            s = Spectrogram(w, args.window, args.fps, args.online)
            # filter
            if args.filter:
                # (re-)create filterbank if the samplerate of the audio changes
                if filt is None or filt.fs != w.samplerate:
                    filt = CQFilter(args.window / 2, w.samplerate, args.bands, args.fmin, args.fmax, args.equal)
                # filter the spectrogram
                s.filter(filt.filterbank)
            # log
            if args.log:
                s.log(args.mul, args.add)
            # use the spectrogram to create an SpectralODF object
            sodf = SpectralODF(s, args.ratio, args.diff_frames)
            # perform detection function on the object
            # e.g. act = sodf.superflux(args.max_bins)
            if args.odf == 'superflux':
                act = getattr(sodf, args.odf)(args.max_bins)
            else:
                act = getattr(sodf, args.odf)()
            # create an Onset object with the activations
            o = Onset(act, args.fps, args.online)
            if args.save:
                # save the raw ODF activations
                o.save("%s.%s" % (filename, args.odf))

        # detect the onsets
        o.detect(args.threshold, args.combine, args.pre_avg, args.pre_max, args.post_avg, args.post_max, args.delay)
        # write the onsets to a file
        o.write("%s.onsets.txt" % (filename))
        # also output them to stdout if vebose
        if args.verbose:
            print 'detections:', o.detections

        # continue with next file

if __name__ == '__main__':
    main()
