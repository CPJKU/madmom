# encoding: utf-8
"""
This package includes higher level features. Your definition of "higher" may
vary, but all "lower" level features can be found the `audio` package.

"""

import itertools as it
import numpy as np
from ..audio.wav import Wav
from ..audio.spectrogram import LogFiltSpec
from ..ml.rnn import RecurrentNeuralNetwork

def _process_rnn((nn_file, data)):
    """
    Loads a RNN model from the given file (first tuple item) and passes the
    given numpy array of data through it (second tuple item).

    """
    return RecurrentNeuralNetwork(nn_file).activate(data)


class RnnActivationFunction(object):

    def __init__(self, signal, nn_files, online, fps, bands_per_octave,
                 window_sizes, mul, add, norm_filters,
                 n_threads, **kwargs):

        if isinstance(signal, Wav):
            self._signal = signal
        else:
            self._signal = Wav(signal, mono=True, **kwargs)

        if online:
            raise NotImplementedError('online mode not implemented (yet)')

        self._nn_files = nn_files
        self._fps = fps
        self._bands_per_octave = bands_per_octave
        self._window_sizes = window_sizes
        self._mul = mul
        self._add = add
        self._norm_filters = norm_filters
        self._n_threads = n_threads

        self._activations = None

    @classmethod
    def from_activations(cls, activations, fps, sep=None):
        af = cls(signal=None, nn_filespy=None, online=None, fps=fps,
                 bands_per_octave=None, window_sizes=None, mul=None, add=None,
                 norm_filters=None, n_threads=None)

        # set / load activations
        if isinstance(activations, np.ndarray):
            # activations are given as an array
            af._activations = activations
        else:
            try:
                # try to load as numpy binary format
                af._activations = np.load(activations)
            except IOError:
                # simple text format
                af._activations = np.loadtxt(activations, delimiter=sep)

        return af

    @property
    def activations(self):
        if self._activations is None:
            self.compute()

        return self._activations

    def compute(self):
        if self._signal is None:
            raise RuntimeError('This activation class was loaded from a file'
                               'and thus cannot be computed!')

        specs = []
        for fs in self._window_sizes:
            s = LogFiltSpec(self._signal, frame_size=fs, fps=self._fps,
                            bands_per_octave=self._bands_per_octave,
                            mul=self._mul, add=self._add,
                            norm_filters=self._norm_filters)

            specs.append(s.spec)
            specs.append(s.pos_diff)

        data = np.hstack(specs)

        # init a pool of workers (if needed)
        map_ = map
        if self._n_threads != 1:
            map_ = mp.Pool(self._n_threads).map

        # compute predictions with all saved neural networks (in parallel)
        activations = map_(_process_rnn,
                           it.izip(self._nn_files, it.repeat(data)))

        # average activations if needed
        n_activations = len(self._nn_files)
        if n_activations > 1:
            act = sum(activations) / n_activations
        else:
            act = activations[0]

        self._activations = act.ravel()

    def save(self, filename, sep=None):
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
