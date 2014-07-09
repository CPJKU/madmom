# encoding: utf-8
"""
This package includes higher level features. Your definition of "higher" may
vary, but all "lower" level features can be found the `audio` package.

"""

import itertools as it
import numpy as np
import multiprocessing as mp
from ..audio.signal import Signal
from ..audio.spectrogram import LogFiltSpec
from ..ml.rnn import process_rnn


class EventDetection(object):

    def __init__(self, data, fps, sep=None, **kwargs):
        self._detections = None
        self._activations = None
        self.signal = None

        self.fps = fps

        if isinstance(data, np.ndarray):
            # Data are activations
            self._activations = data
        elif isinstance(data, Signal):
            # Data is a signal
            self.signal = data
        else:
            # Data is a filename. If sep is set, it is a text file containing
            # activations
            if sep is not None:
                self._activations = np.loadtxt(data, sep=sep)
            else:
                # let's see if it's a binary activation file
                try:
                    self._activations = np.load(data)
                except IOError:
                    # it has to be audio file
                    self.signal = Signal(data, **kwargs)

    @property
    def detections(self):
        if self._detections is None:
            self._detections = self.detect()
        return self._detections

    @property
    def activations(self):
        if self._activations is None:
            self._activations = self.process()
        return self._activations

    def process(self):
        # This function has to be implemented by subclasses
        raise NotImplementedError("Please implement this method")

    def detect(self):
        # This function has to be implemented by subclasses
        raise NotImplementedError("Please implement this method")

    def save_detections(self, filename):
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

    @classmethod
    def add_arguments(cls, parser, **kwargs):
        pass


class RnnEventDetection(EventDetection):
    # TODO: this information should be included/extracted in/from the NN files
    FPS = 100
    BANDS_PER_OCTAVE = 3
    WINDOW_SIZES = [1024, 2048, 4096]
    MUL = 1
    ADD = 1
    NORM_FILTERS = True
    N_THREADS = mp.cpu_count()
    ONLINE = False

    def __init__(self, data, nn_files=None, fps=FPS, online=ONLINE,
                 bands_per_octave=BANDS_PER_OCTAVE, window_sizes=WINDOW_SIZES,
                 mul=MUL, add=ADD, norm_filters=NORM_FILTERS,
                 n_threads=N_THREADS, **kwargs):

        super(RnnEventDetection, self).__init__(data, fps=fps, **kwargs)

        if nn_files is None and self._activations is None:
            raise RuntimeError("Either specify neural network files or load activations from file!")

        self.nn_files = nn_files
        self.bands_per_octave = bands_per_octave
        self.window_sizes = window_sizes
        self.mul = mul
        self.add = add
        self.norm_filters = norm_filters
        self.n_threads = n_threads
        self.origin = 'online' if online else 0


    def process(self):
        specs = []
        for fs in self.window_sizes:
            s = LogFiltSpec(self.signal, frame_size=fs, fps=self.fps,
                            bands_per_octave=self.bands_per_octave,
                            mul=self.mul, add=self.add,
                            norm_filters=self.norm_filters,
                            origin=self.origin)

            specs.append(s.spec)
            specs.append(s.pos_diff)

        data = np.hstack(specs)

        activations = process_rnn(data, self.nn_files, self.n_threads)
        return activations.ravel()


    @classmethod
    def add_arguments(cls, parser, nn_files=None, threads=N_THREADS, **kwargs):
        """
        Add neural network testing options to an existing parser object.

        :param parser:   existing argparse parser object
        :param nn_files: list of NN files
        :param threads:  number of threads to run in parallel
        :return:         neural network argument parser group object

        """
        super(RnnEventDetection, cls).add_arguments(parser, **kwargs)
        # add neural network related options to the existing parser
        g = parser.add_argument_group('neural network arguments')
        g.add_argument('--nn_files', action='append', type=str,
                       default=nn_files, help='average the predictions of '
                       'these pre-trained neural networks (multiple files '
                       'can be given, one file per argument)')
        g.add_argument('--threads', action='store', type=int, default=threads,
                       help='number of parallel threads [default=%(default)s]')
        # return the argument group so it can be modified if needed
        return g


import onsets
import beats
import tempo
