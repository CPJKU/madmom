#!/usr/bin/env python
# encoding: utf-8
"""
This file contains all onset detection related functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import numpy as np
import scipy.ndimage as sim


# helper functions
def wraptopi(phase):
    """
    Wrap the phase information to the range -π...π.

    :param phase: phase spectrogram
    :returns:     wrapped phase spectrogram

    """
    return np.mod(phase + np.pi, 2.0 * np.pi) - np.pi


def diff(spec, diff_frames=1, pos=False):
    """
    Calculates the difference of the magnitude spectrogram.

    :param spec:        the magnitude spectrogram
    :param diff_frames: calculate the difference to the N-th previous frame
                        [default=1]
    :param pos:         only keep positive values [default=False]
    :returns:           (positive) magnitude spectrogram differences

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
        diff *= (diff > 0)
    return diff


def correlation_diff(spec, diff_frames=1, pos=False, diff_bins=1):
    """
    Calculates the difference of the magnitude spectrogram relative to the
    N-th previous frame shifted in frequency to achieve the highest
    correlation between these two frames.

    :param spec:        the magnitude spectrogram
    :param diff_frames: calculate the difference to the N-th previous frame
                        [default=1]
    :param pos:         only keep positive values [default=False]
    :param diff_bins:   maximum number of bins shifted for correlation
                        calculation [default=1]
    :returns:           (positive) magnitude spectrogram differences

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
        diff[f, diff_bins:-diff_bins] = spec[f, diff_bins:-diff_bins] -\
            spec[f - diff_frames, bin_start:bin_stop]
    # keep only positive values
    if pos:
        diff *= (diff > 0)
    return diff


# Onset Detection Functions
def high_frequency_content(spec):
    """
    High Frequency Content.

    :param spec: the magnitude spectrogram
    :returns:    high frequency content onset detection function

    "Computer Modeling of Sound for Transformation and Synthesis of Musical
    Signals"
    Paul Masri
    PhD thesis, University of Bristol, 1996

    """
    # HFC weights the magnitude spectrogram by the bin number,
    # thus emphasizing high frequencies
    return np.mean(spec * np.arange(spec.shape[1]), axis=1)


def spectral_diff(spec, diff_frames=1):
    """
    Spectral Diff.

    :param spec:        the magnitude spectrogram
    :param diff_frames: calculate the difference to the N-th previous frame
                        [default=1]
    :returns:           spectral diff onset detection function

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
                        [default=1]
    :returns:           spectral flux onset detection function

    "Computer Modeling of Sound for Transformation and Synthesis of Musical
    Signals"
    Paul Masri
    PhD thesis, University of Bristol, 1996

    """
    # Spectral flux is the sum of all positive 1st order differences
    return np.sum(diff(spec, diff_frames=diff_frames, pos=True), axis=1)


def superflux(spec, diff_frames=1, max_bins=3):
    """
    SuperFlux with a maximum peak-tracking stage for difference calculation.

    Calculates the difference of bin k of the magnitude spectrogram relative to
    the N-th previous frame with the maximum filtered spectrogram.

    :param spec:        the magnitude spectrogram
    :param diff_frames: calculate the difference to the N-th previous frame
                        [default=1]
    :param max_bins:    number of neighboring bins used for maximum filtering
                        [default=3]
    :returns:           SuperFlux onset detection function

    "Maximum Filter Vibrato Suppression for Onset Detection"
    Sebastian Böck and Gerhard Widmer
    Proceedings of the 16th International Conference on Digital Audio Effects
    (DAFx-13), 2013.

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
    # widen the spectrogram in frequency dimension by `max_bins`
    max_spec = sim.maximum_filter(spec, size=[1, max_bins])
    # calculate the diff
    diff[diff_frames:] = spec[diff_frames:] - max_spec[0:-diff_frames]
    # keep only positive values
    diff *= (diff > 0)
    # SuperFlux is the sum of all positive 1st order max. filtered differences
    return np.sum(diff, axis=1)


def modified_kullback_leibler(spec, diff_frames=1, epsilon=0.000001):
    """
    Modified Kullback-Leibler.

    :param spec:        the magnitude spectrogram
    :param diff_frames: calculate the difference to the N-th previous frame
                        [default=1]
    :param epsilon:     add epsilon to avoid division by 0 [default=0.000001]
    :returns:           MKL onset detection function

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
    mkl = np.zeros_like(spec)
    mkl[diff_frames:] = spec[diff_frames:] / (spec[:-diff_frames] + epsilon)
    # note: the original MKL uses sum instead of mean,
    # but the range of mean is much more suitable
    return np.mean(np.log(1 + mkl), axis=1)


def _phase_deviation(phase):
    """
    Helper method used by phase_deviation() & weighted_phase_deviation().

    :param phase: the phase spectrogram
    :returns:     phase deviation

    """
    pd = np.zeros_like(phase)
    # instantaneous frequency is given by the first difference
    # ψ′(n, k) = ψ(n, k) − ψ(n − 1, k)
    # change in instantaneous frequency is given by the second order difference
    # ψ′′(n, k) = ψ′(n, k) − ψ′(n − 1, k)
    pd[2:] = phase[2:] - 2 * phase[1:-1] + phase[:-2]
    # map to the range -pi..pi
    return wraptopi(pd)


def phase_deviation(phase):
    """
    Phase Deviation.

    :param phase: the phase spectrogram
    :returns:     phase deviation onset detection function

    "On the use of phase and energy for musical onset detection in the complex
    domain"
    Juan Pablo Bello, Chris Duxbury, Matthew Davies and Mark Sandler
    IEEE Signal Processing Letters, Volume 11, Number 6, 2004

    """
    # take the mean of the absolute changes in instantaneous frequency
    return np.mean(np.abs(_phase_deviation(phase)), axis=1)


def weighted_phase_deviation(spec, phase):
    """
    Weighted Phase Deviation.

    :param spec:  the magnitude spectrogram
    :param phase: the phase spectrogram
    :returns:     weighted phase deviation onset detection function

    "Onset Detection Revisited"
    Simon Dixon
    Proceedings of the 9th International Conference on Digital Audio Effects
    (DAFx), 2006

    """
    # make sure the spectrogram is not filtered before
    if np.shape(phase) != np.shape(spec):
        raise ValueError("Magn. spectrogram and phase must be of same shape")
    # weighted_phase_deviation = spec * phase_deviation
    return np.mean(np.abs(_phase_deviation(phase) * spec), axis=1)


def normalized_weighted_phase_deviation(spec, phase, epsilon=0.000001):
    """
    Normalized Weighted Phase Deviation.

    :param spec:    the magnitude spectrogram
    :param phase:   the phase spectrogram
    :param epsilon: add epsilon to avoid division by 0 [default=0.000001]
    :returns:       normalized weighted phase deviation onset detection
                    function

    "Onset Detection Revisited"
    Simon Dixon
    Proceedings of the 9th International Conference on Digital Audio Effects
    (DAFx), 2006

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
    :returns:     complex domain

    Note: we use the simple implementation presented in:
    "Onset Detection Revisited"
    Simon Dixon
    Proceedings of the 9th International Conference on Digital Audio Effects
    (DAFx), 2006

    """
    if np.shape(phase) != np.shape(spec):
        raise ValueError("Magn. spectrogram and phase must be of same shape")
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
    :returns:     complex domain onset detection function

    "On the use of phase and energy for musical onset detection in the complex
    domain"
    Juan Pablo Bello, Chris Duxbury, Matthew Davies and Mark Sandler
    IEEE Signal Processing Letters, Volume 11, Number 6, 2004

    """
    # take the sum of the absolute changes
    return np.sum(np.abs(_complex_domain(spec, phase)), axis=1)


def rectified_complex_domain(spec, phase):
    """
    Rectified Complex Domain.

    :param spec:  the magnitude spectrogram
    :param phase: the phase spectrogram
    :returns:     recified complex domain onset detection function

    "Onset Detection Revisited"
    Simon Dixon
    Proceedings of the 9th International Conference on Digital Audio Effects
    (DAFx), 2006

    """
    # rectified complex domain
    rcd = _complex_domain(spec, phase)
    # only keep values where the magnitude rises
    rcd *= diff(spec, pos=True)
    # take the sum of the absolute changes
    return np.sum(np.abs(rcd), axis=1)


# SpectralOnsetDetection default values
MAX_BINS = 3


class SpectralOnsetDetection(object):
    """
    The SpectralOnsetDetection class implements most of the common onset
    detection functions based on the magnitude or phase information of a
    spectrogram.

    """
    def __init__(self, spectrogram, max_bins=MAX_BINS, *args, **kwargs):
        """
        Creates a new SpectralOnsetDetection object instance.

        :param spectrogram: the spectrogram object on which the detections
                            functions operate
        :param max_bins:    number of bins for the maximum filter
                            (for SuperFlux) [default=3]

        """
        # import
        from ..audio.spectrogram import Spectrogram
        # check spectrogram type
        if isinstance(spectrogram, Spectrogram):
            # already the right format
            self.s = spectrogram
        else:
            # assume a file name, try to instantiate a Spectrogram object
            self.s = Spectrogram(spectrogram, *args, **kwargs)
        self.max_bins = max_bins

    # FIXME: do use s.spec and s.diff directly instead of passing the number of
    # diff_frames to all these functions?

    # Onset Detection Functions
    def hfc(self):
        """High Frequency Content."""
        return high_frequency_content(self.s.spec)

    def sd(self):
        """Spectral Diff."""
        return spectral_diff(self.s.spec, self.s.num_diff_frames)

    def sf(self):
        """Spectral Flux."""
        return spectral_flux(self.s.spec, self.s.num_diff_frames)

    def superflux(self, max_bins=None):
        """
        SuperFlux.

        :param max_bins: number of bins for the maximum filter [default=None]

        """
        if max_bins:
            # overwrite the number of bins used for maximum filtering
            self.max_bins = max_bins
        return superflux(self.s.spec, self.s.num_diff_frames, self.max_bins)

    def mkl(self):
        """Modified Kullback-Leibler."""
        return modified_kullback_leibler(self.s.spec, self.s.num_diff_frames)

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


# universal peak-picking method
def peak_picking(activations, threshold, smooth=None, pre_avg=0, post_avg=0,
                 pre_max=1, post_max=1):
    """
    Perform thresholding and peak-picking on the given activation function.

    :param activations: the onset activation function
    :param threshold:   threshold for peak-picking
    :param smooth:      smooth the activation function with the kernel
                        [default=None]
    :param pre_avg:     use N frames past information for moving average
                        [default=0]
    :param post_avg:    use N frames future information for moving average
                        [default=0]
    :param pre_max:     use N frames past information for moving maximum
                        [default=1]
    :param post_max:    use N frames future information for moving maximum
                        [default=1]

    Notes: If no moving average is needed (e.g. the activations are independent
           of the signal's level as for neural network activations), set
           `pre_avg` and `post_avg` to 0.

           For offline peak picking set `pre_max` and `post_max` to 1.

           For online peak picking set all `post_` parameters to 0.

    """
    # smooth activations
    kernel = None
    if isinstance(smooth, int):
        # size for the smoothing kernel is given
        if smooth > 1:
            kernel = np.hamming(smooth)
    elif isinstance(smooth, np.ndarray):
        # otherwise use the given smooth kernel directly
        if smooth.size > 1:
            kernel = smooth
    if kernel is not None:
        # convolve with the kernel
        activations = np.convolve(activations, kernel, 'same')
    # threshold activations
    avg_length = pre_avg + post_avg + 1
    if avg_length > 1:
        # compute a moving average
        avg_origin = int(np.floor((pre_avg - post_avg) / 2))
        # TODO: make the averaging function exchangable (mean/median/etc.)
        mov_avg = sim.filters.uniform_filter1d(activations, avg_length,
                                               mode='constant',
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
        mov_max = sim.filters.maximum_filter1d(detections, max_length,
                                               mode='constant',
                                               origin=max_origin)
        # detections are peak positions
        detections *= (detections == mov_max)
    # return indices
    return np.nonzero(detections)[0]


# default values for onset peak-picking
THRESHOLD = 1.25
SMOOTH = 0
PRE_AVG = 0.1
POST_AVG = 0.03
PRE_MAX = 0.03
POST_MAX = 0.07
# default values for onset reporting
COMBINE = 0.03
DELAY = 0


# TODO: common stuff should be moved into an Event class
class Onset(object):
    """
    Onset Class.

    """
    def __init__(self, activations, fps, online=False, sep=''):
        """
        Creates a new Onset object instance with the given activations of the
        ODF (OnsetDetectionFunction). The activations can be read from a file.

        :param activations: array with ODF activations or a file (handle)
        :param fps:         frame rate of the activations
        :param online:      work in online mode (i.e. use only past
                            information) [default=False]
        :param sep:         separator if activations are read from file

        """
        self.activations = None  # onset activation function
        self.fps = float(fps)    # frame rate of the activation function
        self.online = online     # online peak-picking
        # TODO: is it better to init the detections as np.zeros(0)?
        # this way the write() method would not throw an error, but the
        # evaluation might not be correct?!
        self.detections = None   # list of detected onsets [seconds]
        self.targets = None      # list of target onsets [seconds]
        # set / load activations
        if isinstance(activations, np.ndarray):
            # activations are given as an array
            self.activations = activations
        else:
            # read in the activations from a file
            self.load_activations(activations, sep)

    def detect(self, threshold, combine=COMBINE, delay=DELAY, smooth=SMOOTH,
               pre_avg=PRE_AVG, post_avg=POST_MAX, pre_max=PRE_MAX,
               post_max=POST_AVG):
        """
        Detect the onsets with a given peak-picking method.

        :param threshold: threshold for peak-picking
        :param combine:   only report one onset within N seconds [default=0.03]
        :param delay:     report onsets N seconds delayed [default=0]
        :param smooth:    smooth the activation function over N seconds
                          [default=0]
        :param pre_avg:   use N seconds past information for moving average
                          [default=0.1]
        :param post_avg:  use N seconds future information for moving average
                          [default=0.03]
        :param pre_max:   use N seconds past information for moving maximum
                          [default=0.03]
        :param post_max:  use N seconds future information for moving maximum
                          [default=0.07]

        Notes: If no moving average is needed (e.g. the activations are
               independent of the signal's level as for neural network
               activations), `pre_avg` and `post_avg` should be set to 0.

               For offline peak picking set `pre_max` >= 1/fps and
               `post_max` >= 1/fps

               For online peak picking, all `post_` parameters are set to 0.

        """
        # convert timing information to frames and set default values
        # TODO: use at least 1 frame if any of these values are > 0?
        smooth = int(round(self.fps * smooth))
        pre_avg = int(round(self.fps * pre_avg))
        post_avg = int(round(self.fps * post_avg))
        pre_max = int(round(self.fps * pre_max))
        post_max = int(round(self.fps * post_max))
        # adjust some params for online mode
        if self.online:
            smooth = 0
            post_avg = 0
            post_max = 0
        # detect onsets (function returns int indices)
        detections = peak_picking(self.activations, threshold, smooth,
                                  pre_avg, post_avg, pre_max, post_max)
        # convert detected onsets to a list of timestamps
        detections = detections.astype(np.float) / self.fps
        # shift if necessary
        if delay != 0:
            detections += delay
        # always use the first detection and all others if none was reported
        # within the last `combine` seconds
        if detections.size > 1:
            # filter all detections which occur within `combine` seconds
            combined_detections = detections[1:][np.diff(detections) > combine]
            # add them after the first detection
            self.detections = np.append(detections[0], combined_detections)
        else:
            self.detections = detections
        # also return the detections
        return self.detections

    def write(self, filename):
        """
        Write the detected onsets to a file.

        :param filename: output file name or file handle

        Note: detect() method must be called first.

        """
        # TODO: put this (and the same in the Beat class) to an Event class
        from ..utils.helpers import write_events
        write_events(self.detections, filename)

    def load(self, filename):
        """
        Load the target onsets from a file.

        :param filename: input file name or file handle

        """
        # TODO: put this (and the same in the Beat class) to an Event class
        from ..utils.helpers import load_events
        self.targets = load_events(filename)

    def evaluate(self, filename=None, window=0.025):
        """
        Evaluate the detected onsets against this target file.

        :param filename: target file name or file handle
        :param window:   evaluation window [seconds, default=0.025]

        """
        if filename:
            # load the targets
            self.load(filename)
        if self.targets is None:
            # no targets given, can't evaluate
            return None
        # evaluate
        from ..evaluation.onsets import OnsetEvaluation
        return OnsetEvaluation(self.detections, self.targets, window)

    def save_activations(self, filename, sep=''):
        """
        Save the onset activations to a file.

        :param filename: output file name or file handle
        :param sep:      separator between activation values [default='']

        Note: Empty (“”) separator means the file should be treated as binary;
              spaces (” ”) in the separator match zero or more whitespace;
              separator consisting only of spaces must match at least one
              whitespace. Binary files are not platform independen.

        """
        # TODO: put this (and the same in the Beat class) to an Event class
        # save the activations
        self.activations.tofile(filename, sep=sep)

    def load_activations(self, filename, sep=''):
        """
        Load the onset activations from a file.

        :param filename: the target file name
        :param sep:      separator between activation values [default='']

        Note: Empty (“”) separator means the file should be treated as binary;
              spaces (” ”) in the separator match zero or more whitespace;
              separator consisting only of spaces must match at least one
              whitespace. Binary files are not platform independen.

        """
        # TODO: put this (and the same in the Beat class) to an Event class
        # load the activations
        self.activations = np.fromfile(filename, sep=sep)


def parser():
    """
    Command line argument parser for onset detection.

    """
    import argparse
    from ..utils.params import (audio, spec, filtering, log, spectral_odf,
                                onset, io)

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    If invoked without any parameters, the software detects all onsets in
    the given files in online mode according to the method proposed in:

    "Maximum Filter Vibrato Suppression for Onset Detection"
    by Sebastian Böck and Gerhard Widmer
    Proceedings of the 16th International Conference on Digital Audio Effects
    (DAFx-13), Maynooth, Ireland, September 2013

    """)
    # general options
    p.add_argument('files', metavar='files', nargs='+',
                   help='files to be processed')
    p.add_argument('-v', dest='verbose', action='store_true',
                   help='be verbose')
    p.add_argument('--ext', action='store', type=str, default='txt',
                   help='extension for detections [default=txt]')
    # add other argument groups
    audio(p, online=False)
    spec(p)
    filtering(p, filtering=True)
    log(p, log=True)
    spectral_odf(p)
    o = onset(p)
    # list of offered ODFs
    methods = ['superflux', 'hfc', 'sd', 'sf', 'mkl', 'pd', 'wpd', 'nwpd',
               'cd', 'rcd']
    o.add_argument('-o', dest='odf', default='superflux',
                   help='use this onset detection function %s' % methods)
    io(p)
    # parse arguments
    args = p.parse_args()
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

    from ..utils.helpers import files
    from ..audio.wav import Wav
    from ..audio.spectrogram import Spectrogram
    from ..audio.filterbank import LogarithmicFilterBank

    # parse arguments
    args = parser()

    # TODO: also add an option for evaluation and load the targets accordingly
    # see cp.evaluation.helpers.match_files()

    # init filterbank
    fb = None

    # which files to process
    if args.load:
        # load the activations
        ext = '.activations'
    else:
        # only process .wav files
        ext = '.wav'
    # process the files
    for f in files(args.files, ext):
        if args.verbose:
            print f

        # use the name of the file without the extension
        filename = os.path.splitext(f)[0]

        # do the processing stuff unless the activations are loaded from file
        if args.load:
            # load the activations from file
            # FIXME: fps must be encoded in the file
            o = Onset(f, args.fps, args.online)
        else:
            # create a Wav object
            w = Wav(f, mono=True, norm=args.norm, att=args.att)
            if args.filter:
                # (re-)create filterbank if the sample rate is not the same
                if fb is None or fb.sample_rate != w.sample_rate:
                    # create filterbank if needed
                    fb = LogarithmicFilterBank(args.window / 2, w.sample_rate,
                                               args.bands, args.fmin,
                                               args.fmax, args.equal)
            # create a Spectrogram object
            s = Spectrogram(w, frame_size=args.window, filterbank=fb,
                            log=args.log, mul=args.mul, add=args.add,
                            ratio=args.ratio, diff_frames=args.diff_frames)
            # create a SpectralOnsetDetection object
            sodf = SpectralOnsetDetection(s, max_bins=args.max_bins)
            # perform detection function on the object
            act = getattr(sodf, args.odf)()
            # create an Onset object with the activations
            o = Onset(act, args.fps, args.online)
        # save onset activations or detect onsets
        if args.save:
            # save the raw ODF activations
            o.save_activations("%s.%s" % (filename, args.odf))
        else:
            # detect the onsets
            o.detect(args.threshold, combine=args.combine, delay=args.delay,
                     smooth=args.smooth, pre_avg=args.pre_avg,
                     post_avg=args.post_avg, pre_max=args.pre_max,
                     post_max=args.post_max)
            # write the onsets to a file
            o.write("%s.%s" % (filename, args.ext))
            # also output them to stdout if vebose
            if args.verbose:
                print 'detections:', o.detections
        # continue with next file

if __name__ == '__main__':
    main()
