# encoding: utf-8
"""
This package includes higher level features. Your definition of "higher" may
vary, but all "lower" level features can be found the `audio` package.

"""

import numpy as np


# base class for Onsets and Beats
class Event(object):
    """
    Event Class. This one should not be used directly.

    """
    def __init__(self, activations, fps, sep=''):
        """
        Creates a new Event instance with the given activations.
        The activations can be read in from file.

        :param activations: array with the beat activations or a file (handle)
        :param fps:         frame rate of the activations
        :param sep:         separator if activations are read from file

        """
        self.activations = None
        self.fps = float(fps)
        # TODO: is it better to init the detections as np.zeros(0)?
        #       this way the write() method would not throw an error, but the
        #       evaluation might not be correct/working?!
        self.detections = None
        # set / load activations
        if isinstance(activations, np.ndarray):
            # activations are given as an array
            self.activations = activations
        else:
            # read in the activations from a file
            self.load_activations(activations, sep)

    def write(self, filename):
        """
        Write the detections to a file.

        :param filename: output file name or file handle

        Note: detect() method must be called first.

        """
        from ..utils import write_events
        write_events(self.detections, filename)

    def save_activations(self, filename, sep=None):
        """
        Save the activations to a file.

        :param filename: output file name or file handle
        :param sep:      separator between activation values

        Note: An undefined or empty (“”) separator means that the file should
              be written as a numpy binary file.

        """
        # save the activations
        if sep in [None, '']:
            # numpy binary format
            np.save(filename, self.activations)
        else:
            # simple text format
            np.savetxt(filename, self.activations, fmt='%.5f', delimiter=sep)

    def load_activations(self, filename, sep=None):
        """
        Load the activations from a file.

        :param filename: the file name to load the activations from
        :param sep:      separator between activation values

        Note: An undefined or empty (“”) separator means the file should be
              treated as a numpy binary file; spaces (“ ”) in the separator
              match zero or more whitespace; separator consisting only of
              spaces must match at least one whitespace.

        """
        # load the activations
        try:
            # try to load as numpy binary format
            self.activations = np.load(filename)
        except IOError:
            # simple text format
            self.activations = np.loadtxt(filename, delimiter=sep)

import onsets
import beats
import tempo
