#!/usr/bin/env python
# encoding: utf-8
"""
This file contains all beat tracking related functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import sys
import numpy as np


## TODO: implement some simple algorithms
#class SpectralBeatTracking(object):
#    """
#    The SpectralBeatTracking class implements some common beat tracking
#    algorithms.
#
#    """
#    def __init__(self, spectrogram):
#        """
#        Creates a new SpectralBeatTracking object instance.
#
#        :param spectrogram: the spectrogram object on which the detections
#                            functions operate
#
#        """
#        # import
#        from ..audio.spectrogram import Spectrogram
#        # check spectrogram type
#        if isinstance(spectrogram, Spectrogram):
#            # already the right format
#            self.s = spectrogram
#        else:
#            # assume a file name, try to instantiate a Spectrogram object
#            self.s = Spectrogram(spectrogram)
#
#    # beat tracking algorithms
#    def acf(self):
#        """Auto correlation beat tracking."""
#        # TODO: include code
#        raise NotImplementedError
#
#    def multiple_agents(self):
#        """Multiple agents based tracker."""
#        # TODO: include code
#        raise NotImplementedError


# interval (tempo) detection
def detect_dominant_interval(activations, threshold=0, smooth=None, min_tau=1,
                             max_tau=None):
    """
    Extract the dominant interval of the given activation function.

    :param activations: the onset activation function
    :param threshold:   threshold for activation function before
                        auto-correlation [default=0]
    :param smooth:      smooth the activation function with the kernel
                        [default=None]
    :param min_tau:     minimal delta for correlation function
                        [frames, default=1]
    :param max_tau:     maximal delta for correlation function
                        [frames, default=length of activation function]

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

    # threshold function if needed
    if threshold > 0:
        activations *= (activations >= threshold)

    # test all possible intervals
    taus = range(min_tau, max_tau)
    sums = []
    # TODO: make this processing parallel or numpyfy if possible
    for tau in taus:
        sums.append(np.sum(np.abs(activations[tau:] * activations[0:-tau])))

    # return dominant interval
    interval = np.argmax(sums) + min_tau
    return interval


def interval_histogram(activations, threshold=0, smooth=None, min_tau=1,
                       max_tau=None):
    """
    Compute the interval histogram of the given activation function.

    :param activations: the onset activation function
    :param threshold:   threshold for activation function before
                        auto-correlation [default=0]
    :param smooth:      smooth the activation function with the kernel
                        [default=None]
    :param min_tau:     minimal delta for correlation function
                        [frames, default=1]
    :param max_tau:     maximal delta for correlation function
                        [frames, default=length of activation function]
    :returns:           histogram

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

    # threshold function if needed
    if threshold > 0:
        activations *= (activations >= threshold)

    if max_tau is None:
        max_tau = len(activations) - min_tau

    # test all possible intervals
    taus = range(min_tau, max_tau)
    sums = []
    # TODO: make this processing parallel or numpyfy if possible
    for tau in taus:
        sums.append(np.sum(np.abs(activations[tau:] * activations[0:-tau])))

    # return histogram
    return np.array(sums), np.array(taus)


def dominant_interval(histogram, smooth=None):
    """
    Extract the dominant interval of the given histogram.

    :param histogram: histogram with interval distribution
    :param smooth:    smooth the histogram with the kernel [default=None]
    :returns:         dominant interval

    """
    # smooth histogram
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
        values = np.convolve(histogram[0], kernel, 'same')
    else:
        values = histogram[0]
    # return the dominant interval
    return histogram[1][np.argmax(values)]


# TODO: unify with dominant interval
def detect_tempo(histogram, fps, smooth=None):
    """
    Extract the dominant interval of the given histogram.

    :param histogram: histogram with interval distribution
    :param fps:       frame rate of the original beat activations
    :param smooth:    smooth the histogram with the kernel [default=None]
    :returns:         dominant interval

    """
    # smooth histogram
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
        values = np.convolve(histogram[0], kernel, 'same')
    else:
        values = histogram[0]
    # to get the two dominant tempi, just keep peaks
    values[:-1] = values[:-1] * ((values[1:] > values[:-1]) == False)
    # zero out values if predecessor is greater
    values[1:] = values[1:] * ((values[:-1] > values[1:]) == False)
    # strongest tempo
    interval1 = histogram[1][np.argmax(values)]
    weight1 = values[np.argmax(values)]
    tempo1 = 60.0 * fps / interval1
    # set it to 0
    values[np.argmax(values)] = 0
    # get the second strongest tempo
    interval2 = histogram[1][np.argmax(values)]
    weight2 = values[np.argmax(values)]
    tempo2 = 60.0 * fps / interval2
    # determine their relative strength
    try:
        weight = weight1 / (weight1 + weight2)
    except ZeroDivisionError:
        # just make sure we have some values
        weight = 1.0
    # return the 2 dominant tempi and their relative weight
    return tempo1, tempo2, weight


def detect_beats(activations, interval, look_aside=0.2):
    """
    Detects the beats in the given activation function.

    :param activations: array with beat activations
    :param interval:    look for the next beat each N frames
    :param look_aside:  look this fraction of the interval to the side to
                        detect the beats [default=False]

    Note: A Hamming window of 2*look_aside*interval is apllied for smoothing.

    """
    # TODO: make this faster!
    sys.setrecursionlimit(len(activations))
    # look for which starting beat the sum gets maximized
    sums = np.zeros(interval)
    positions = []
    frames_look_aside = int(interval * look_aside)
    win = np.hamming(2 * frames_look_aside)
    for i in range(interval):
        # TODO: threads?
        def recursive(pos):
            """
            Recursively detect the next beat.

            :param pos: start at this position
            :return:    the next beat position

            """
            # detect the nearest beat around the actual position
            start = pos - frames_look_aside
            end = pos + frames_look_aside
            if start < 0:
                # pad with zeros
                act = np.append(np.zeros(-start), activations[0:end])
            elif end > len(activations):
                # append zeros accordingly
                zeros = np.zeros(end - len(activations))
                act = np.append(activations[start:], zeros)
            else:
                act = activations[start:end]
            # apply a filtering window to prefer beats closer to the centre
            act = np.multiply(act, win)
            # search max
            if np.argmax(act) > 0:
                # maximum found, take that position
                pos = np.argmax(act) + start
            # add the found position
            positions.append(pos)
            # add the activation at that position
            sums[i] += activations[pos]
            # go to the next beat, until end is reached
            if pos + interval < len(activations):
                recursive(pos + interval)
            else:
                return
        # start at initial position
        recursive(i)
    # take the winning start position
    pos = np.argmax(sums)
    # and calc the beats for this start position
    positions = []
    recursive(pos)
    # return indices (as floats, since they get converted to seconds later on)
    return np.array(positions).astype(np.float)


# default values for beat tracking
THRESHOLD = 0
SMOOTH = 0.09
MIN_BPM = 60
MAX_BPM = 240
LOOK_ASIDE = 0.2
LOOK_AHEAD = 4
DELAY = 0


class Beat(object):
    """
    Beat Class.

    """
    def __init__(self, activations, fps, online=False, sep=''):
        """
        Creates a new Onset object instance with the given activations of the
        ODF (OnsetDetectionFunction). The activations can be read in from file.

        :param activations: array with ODF activations or a file (handle)
        :param fps:         frame rate of the activations
        :param online:      work in online mode (i.e. use only past
                            information) [default=False]
        :param sep:         separator if activations are read from file

        """
        if online:
            raise NotImplementedError('online mode not implemented')
        self.activations = None  # onset activation function
        self.fps = float(fps)    # frame rate of the activation function
        self.online = online     # online beat-tracking
        # TODO: is it better to init the detections as np.empty(0)?
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

    def detect(self, threshold=THRESHOLD, delay=DELAY, smooth=SMOOTH,
               min_bpm=MIN_BPM, max_bpm=MAX_BPM, look_aside=LOOK_ASIDE):
        """
        Detect the beats with a simple auto-correlation method.

        :param threshold:  threshold for peak-picking [default=0]
        :param delay:      report onsets N seconds delayed [default=0]
        :param smooth:     smooth the activation function over N seconds
                           [default=0.9]
        :param min_bpm:    minimum tempo used for beat tracking [default=60]
        :param max_bpm:    maximum tempo used for beat tracking [default=240]
        :param look_aside: look this fraction of a beat interval to the side
                           [default=0.2]

        First the global tempo is estimated and then the beats are aligned
        according to:

        "Enhanced Beat Tracking with Context-Aware Neural Networks"
        by Sebastian Böck and Markus Schedl
        Proceedings of the 14th International Conference on Digital Audio
        Effects (DAFx-11), Paris, France, September 2011

        """
        # convert timing information to frames and set default values
        # TODO: use at least 1 frame if any of these values are > 0?
        smooth = int(round(self.fps * smooth))
        min_tau = int(np.floor(60. * self.fps / max_bpm))
        max_tau = int(np.ceil(60. * self.fps / min_bpm))
        # detect the dominant interval
        interval = detect_dominant_interval(self.activations, threshold,
                                            smooth, min_tau, max_tau)
        # detect beats based on this interval
        detections = detect_beats(self.activations, interval, look_aside)
        # convert detected beats to a list of timestamps
        detections /= float(self.fps)
        # shift if necessary
        if delay != 0:
            detections += delay
        # always use the first detection and all others if none was reported
        # within the duration equivalent to max_bpm
        if detections.size > 1:
            # FIXME: check if this is needed!
            # filter all detections which occur within one beat interval
            combined_detections = detections[1:][np.diff(detections) >
                                                 (60. / max_bpm)]
            # add them after the first detection
            self.detections = np.append(detections[0], combined_detections)
        else:
            self.detections = detections
        # also return the detections
        return self.detections

    def track(self, threshold=THRESHOLD, delay=DELAY, smooth=SMOOTH,
              min_bpm=MIN_BPM, max_bpm=MAX_BPM, look_aside=LOOK_ASIDE,
              look_ahead=LOOK_AHEAD):
        """
        Track the beats with a simple auto-correlation method.

        :param threshold:  threshold for peak-picking [default=0]
        :param delay:      report onsets N seconds delayed [default=0]
        :param smooth:     smooth the activation function over N seconds
                           [default=0.9]
        :param min_bpm:    minimum tempo used for beat tracking [default=60]
        :param max_bpm:    maximum tempo used for beat tracking [default=240]
        :param look_aside: look this fraction of a beat interval to the side
                           [default=0.2]
        :param look_ahead: look N seconds ahead (and back) to determine the
                           tempo [default=4]

        First local tempo (in a range +- look_ahead seconds around the actual
        position) is estimated and then the next beat is tracked accordingly.
        Then the same procedure is repeated from this new position.

        "Enhanced Beat Tracking with Context-Aware Neural Networks"
        by Sebastian Böck and Markus Schedl
        Proceedings of the 14th International Conference on Digital Audio
        Effects (DAFx-11), Paris, France, September 2011
        """
        # convert timing information to frames and set default values
        # TODO: use at least 1 frame if any of these values are > 0?
        smooth = int(round(self.fps * smooth))
        min_tau = int(np.floor(60. * self.fps / max_bpm))
        max_tau = int(np.ceil(60. * self.fps / min_bpm))
        look_ahead_frames = int(look_ahead * self.fps)
        # detect the beats
        detections = []
        pos = 0
        # TODO: make this _much_ faster!
        while pos < len(self.activations):
            # look N frames around the actual position
            start = pos - look_ahead_frames
            end = pos + look_ahead_frames
            if start < 0:
                # pad with zeros
                act = np.append(np.zeros(-start), self.activations[0:end])
            elif end > len(self.activations):
                # append zeros accordingly
                zeros = np.zeros(end - len(self.activations))
                act = np.append(self.activations[start:], zeros)
            else:
                act = self.activations[start:end]
            # detect the dominant interval
            interval = detect_dominant_interval(act, threshold, smooth,
                                                min_tau, max_tau)
            # add the offset (i.e. the new detected start position)
            positions = np.array(detect_beats(act, interval, look_aside))
            # correct the beat positions
            positions += start
            # search the closest beat to the predicted beat position
            pos = positions[(np.abs(positions - pos)).argmin()]
            # append to the beats
            detections.append(pos)
            pos += interval

        # convert detected beats to a list of timestamps
        detections = np.array(detections) / float(self.fps)
        # shift if necessary
        if delay != 0:
            detections += delay
        # always use the first detection and all others if none was reported
        # within the duration equivalent to max_bpm
        if detections.size > 1:
            # FIXME: check if this is needed!
            # filter all detections which occur within one beat interval
            combined_detections = detections[1:][np.diff(detections) >
                                                 (60. / max_bpm)]
            # add them after the first detection
            self.detections = np.append(detections[0], combined_detections)
        else:
            self.detections = detections
        # also return the detections
        return self.detections

    # TODO: make an extra Tempo class!
    def tempo(self, threshold=THRESHOLD, smooth=SMOOTH, min_bpm=MIN_BPM,
              max_bpm=MAX_BPM, mirex=False):
        """
        Detect the tempo on basis of the beat activation function.

        :param threshold: threshold for peak-picking [default=0]
        :param smooth:    smooth the activation function over N seconds
                          [default=0.9]
        :param min_bpm:   minimum tempo used for beat tracking [default=60]
        :param max_bpm:   maximum tempo used for beat tracking [default=240]
        :param mirex:     always output the lower tempo first [default=False]
        :returns:         tuple with the two most dominant tempi and the
                          relative weight of them

        """
        # convert the arguments to frames
        smooth = int(round(self.fps * smooth))
        min_tau = int(np.floor(60. * self.fps / max_bpm))
        max_tau = int(np.ceil(60. * self.fps / min_bpm))
        # generate a histogram of beat intervals
        hist = interval_histogram(self.activations, threshold, smooth=smooth,
                                  min_tau=min_tau, max_tau=max_tau)
        t1, t2, weight = detect_tempo(hist, self.fps, smooth=None)
        # for MIREX, the lower tempo must be given first
        if mirex and t1 > t2:
            return t2, t1, 1 - weight
        return t1, t2, weight

    def write(self, filename):
        """
        Write the detected beats to a file.

        :param filename: output file name or file handle

        Note: detect() method must be called first.

        """
        # TODO: put this (and the same in the Onset class) to an Event class
        from ..utils.helpers import write_events
        write_events(self.detections, filename)

    def load(self, filename):
        """
        Load the target beats from a file.

        :param filename: input file name or file handle

        """
        # TODO: put this (and the same in the Onset class) to an Event class
        from ..utils.helpers import load_events
        self.targets = load_events(filename)

    def save_activations(self, filename, sep=''):
        """
        Save the beat activations to a file.

        :param filename: output file name or file handle
        :param sep:      separator between activation values [default='']

        Note: Empty (“”) separator means the file should be treated as binary;
              spaces (” ”) in the separator match zero or more whitespace;
              separator consisting only of spaces must match at least one
              whitespace. Binary files are not platform independen.

        """
        # TODO: put this (and the same in the Onset class) to an Event class
        # save the activations
        self.activations.tofile(filename, sep=sep)

    def load_activations(self, filename, sep=''):
        """
        Load the beat activations from a file.

        :param filename: the target file name
        :param sep:      separator between activation values [default='']

        Note: Empty (“”) separator means the file should be treated as binary;
              spaces (” ”) in the separator match zero or more whitespace;
              separator consisting only of spaces must match at least one
              whitespace. Binary files are not platform independen.

        """
        # TODO: put this (and the same in the Onset class) to an Event class
        # load the activations
        self.activations = np.fromfile(filename, sep=sep)


def parser():
    """
    Command line argument parser for beat detection.

    """
    import argparse
    from ..utils.params import audio, spec, filtering, log, spectral_odf, beat

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    If invoked without any parameters, the software detects all beats in the
    given files with the method described in:

    "Enhanced Beat Tracking with Context-Aware Neural Networks"
    by Sebastian Böck and Markus Schedl
    Proceedings of the 14th International Conference on Digital Audio Effects
    (DAFx-11), Paris, France, September 2011

    """)
    # general options
    p.add_argument('files', metavar='files', nargs='+',
                   help='files to be processed')
    p.add_argument('-v', dest='verbose', action='count',
                   help='increase verbosity level')
    p.add_argument('--track', action='store_true', default=False,
                   help='track, not detect')
    p.add_argument('--ext', action='store', type=str, default='txt',
                   help='extension for detections [default=txt]')
    # add other argument groups
    audio(p, fps=100)
    spec(p)
    filtering(p, filtering=True)
    log(p, log=True)
    spectral_odf(p)
    beat(p)
    # parse arguments
    args = p.parse_args()
    # print arguments
    if args.verbose:
        print args
    # return args
    return args


def main():
    """
    Example beat tracking program.

    """
    import os.path

    from ..utils.helpers import files
#    from ..audio.wav import Wav
#    from ..audio.spectrogram import Spectrogram
#    from ..audio.filterbank import LogarithmicFilterBank

    # parse arguments
    args = parser()

    # init filterbank
#    fb = None

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
            b = Beat(f, args.fps, args.online)
        else:
            raise NotImplementedError('only loading of beat activations '
                                      'supported so far.')
#            w = Wav(f, mono=True, norm=args.norm, att=args.att)
#            if args.filter:
#                # (re-)create filterbank if the sample rate is not the same
#                if fb is None or fb.sample_rate != w.sample_rate:
#                    # create filterbank if needed
#                    fb = LogarithmicFilterBank(args.window / 2, w.sample_rate,
#                                               args.bands, args.fmin,
#                                               args.fmax, args.equal)
#            # create a Spectrogram object
#            s = Spectrogram(w, frame_size=args.window, filterbank=fb,
#                            log=args.log, mul=args.mul, add=args.add,
#                            ratio=args.ratio, diff_frames=args.diff_frames)
#            # create a SpectralBeatTracking object
#            sbdf = SpectralBeatTracking(s)
#            # perform detection function on the object
#            act = getattr(sbdf, args.bdf)()
#            # create an Onset object with the activations
#            b = Beat(act, args.fps, args.online)
        # save onset activations or detect onsets
        if args.save:
            # save the raw beat activations
            b.save_activations("%s.activations" % filename)
        else:
            # detect the beats
            if not args.track:
                b.detect(args.threshold, delay=args.delay, smooth=args.smooth,
                         min_bpm=args.min_bpm, max_bpm=args.max_bpm)
            else:
                b.track(args.threshold, delay=args.delay, smooth=args.smooth,
                         min_bpm=args.min_bpm, max_bpm=args.max_bpm)
            # write the beats to a file
            b.write("%s.%s" % (filename, args.ext))
            # also output them to stdout if vebose
            if args.verbose > 2:
                print 'tempo:     ', 60. / np.median(np.diff(b.detections))
                print 'tempo:     ', 60. / np.mean(np.diff(b.detections))
        # continue with next file

if __name__ == '__main__':
    main()
