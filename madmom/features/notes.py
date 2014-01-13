#!/usr/bin/env python
# encoding: utf-8
"""
This file contains all note transcription related functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import numpy as np
import scipy.ndimage as sim
import scipy.signal as sig


def load_notes(filename):
    """
    Load the target notes from a file.

    :param filename: input file name or file handle
    :returns:        numpy array with notes

    """
    own_fid = False
    # open file if needed
    if isinstance(filename, basestring):
        fid = open(filename, 'rb')
        own_fid = True
    else:
        fid = filename
    try:
        # read in the events, one per line
        return np.loadtxt(fid)
    finally:
        # close file if needed
        if own_fid:
            fid.close()


# universal peak-picking method
def peak_picking(activations, threshold, smooth=None, pre_avg=0, post_avg=0,
                 pre_max=1, post_max=1):
    """
    Perform thresholding and peak-picking on the given activation function.

    :param activations: note activations (2D numpy array)
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
    dim = activations.shape[1]
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
        activations = sig.convolve2d(activations, kernel[:, np.newaxis],
                                     'same')
    # threshold activations
    avg_length = pre_avg + post_avg + 1
    if avg_length > 1:
        # compute a moving average
        avg_origin = int(np.floor((pre_avg - post_avg) / 2))
        # TODO: make the averaging function exchangable (mean/median/etc.)
        mov_avg = sim.filters.uniform_filter(activations, [avg_length, dim],
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
        mov_max = sim.filters.maximum_filter(detections, [max_length, dim],
                                             mode='constant',
                                             origin=max_origin)
        # detections are peak positions
        detections *= (detections == mov_max)
    # return indices of detections
    return np.nonzero(detections)


# default values for onset peak-picking
THRESHOLD = 0.35
SMOOTH = 0
PRE_AVG = 0
POST_AVG = 0
PRE_MAX = 0.03
POST_MAX = 0.07
# default values for onset reporting
COMBINE = 0.03
DELAY = 0


class NoteTranscription(object):
    """
    NoteTranscription class.

    """
    def __init__(self, activations, fps, sep=''):
        """
        Creates a new NoteTranscription object instance with the given
        activations (can be read in from a file).

        :param activations: array with note activations or a file (handle)
        :param fps:         frame rate of the activations
        :param sep:         separator if activations are read from file

        """
        self.activations = None  # onset activation function
        self.fps = fps           # frame rate of the activation function
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

    def detect(self, threshold, combine=COMBINE, delay=DELAY, smooth=SMOOTH,
               pre_avg=PRE_AVG, post_avg=POST_MAX, pre_max=PRE_MAX,
               post_max=POST_AVG):
        """
        Detect the notes with the given peak-picking parameters.

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

        """
        # convert timing information to frames and set default values
        # TODO: use at least 1 frame if any of these values are > 0?
        smooth = int(round(self.fps * smooth))
        pre_avg = int(round(self.fps * pre_avg))
        post_avg = int(round(self.fps * post_avg))
        pre_max = int(round(self.fps * pre_max))
        post_max = int(round(self.fps * post_max))
        # detect onsets
        detections = peak_picking(self.activations, threshold, smooth, pre_avg,
                                  post_avg, pre_max, post_max)
        # convert to seconds / MIDI note numbers
        onsets = detections[0] / float(self.fps)
        midi_notes = detections[1] + 21
        # shift if necessary
        if delay != 0:
            onsets += delay
#        # FIXME: make this work for notes instead of simple 1-D onset arrays
#        # always use the first detection and all others if none was reported
#        # within the last 'combine' seconds
#        if detections.size > 1:
#            # filter all detections which occur within `combine` seconds
#            combined_detections = detections[1:][np.diff(detections) > combine]
#            # add them after the first detection
#            self.detections = np.append(detections[0], combined_detections)
#        else:
#            self.detections = detections
        self.detections = zip(onsets, midi_notes)
        # also return the detections
        return self.detections

    def write(self, output, sep='\t'):
        """
        Write the detected notes to a file.

        :param output: output file name or file handle
        :param sep:    separator for the fields [default='\t']

        Note: detect() method must be called first.

        """
        # zip the detections
        # write the detected notes to the output
        for note in self.detections:
            output.write(sep.join([str(x) for x in note]) + '\n')

    def load(self, filename):
        """
        Load the target notes from a file.

        :param filename: input file name or file handle

        """
        own_fid = False
        # open file if needed
        if isinstance(filename, basestring):
            fid = open(filename, 'rb')
            own_fid = True
        else:
            fid = filename
        try:
            # read in the events, one per line
            # 1st column is the event's time, the rest is ignored
            return np.fromiter((float(line.split(None, 1)[0]) for line in fid
                                if not line.startswith('#')), dtype=np.float)
        finally:
            # close file if needed
            if own_fid:
                fid.close()

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
        # load the activations
        self.activations = np.fromfile(filename, sep=sep)
