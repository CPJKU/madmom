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
    """ Base class for anything that detects events in an audio stream."""

    def __init__(self, data, fps, sep=None, **kwargs):
        """ Sets up the object.
            :param data: signal, activations of file name of the data to be
                         processed. Activations can be passed as numpy array,
                         a signal should use the Signal class. If data is a
                         filename and `sep` is not None, a text file containing
                         activations is assumed. If no value for `sep` is
                         provided, the method tries to load activations from
                         a binary .npy file. If this fails, it tries to load
                         a signal from the file, with **kwargs passed to the
                         constructor of Signal.

            :param fps:  Frames per second of the activations
            :param sep:  Separator of items in text file to be loaded. If it is
                         None, a binary file is assumed
        """
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
        """ The detected events. """
        if self._detections is None:
            self._detections = self.detect()
        return self._detections

    @property
    def activations(self):
        """ The activations used for event detection. """
        if self._activations is None:
            self._activations = self.process()
        return self._activations

    def process(self):
        """ This method processes the signal and computes the activations.
            :return: activations computed from the signal
        """
        # This function has to be implemented by subclasses
        raise NotImplementedError("Please implement this method")

    def detect(self):
        """ This method extracts the events (beats, onsets, ...) from the
            activations.
            :return: detected events.
        """
        # This function has to be implemented by subclasses
        raise NotImplementedError("Please implement this method")

    def save_detections(self, filename):
        """
        Write the detections to a file.

        :param filename: output file name or file handle

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
        """
        Add arguments to an argparse.ArgumentParser that specify parameters
        that can be changed through command line parameters. Note that
        parameters should be stored in variables named similarly to what the
        __init__ method expects as parameters. Derived classes should add
        super(<classname>, cls).add_arguments(parser, **kwargs) as first call,
        so that parent classes can add their arguments too.

        :param cls:    class for which the method is called.
        :param parser: argparse.ArgumentParser object to which arguments should
                       be added.
        :return:       If applicable, argument group to which arguments were
                       added
        """
        pass


class RnnEventDetection(EventDetection):
    """ Base class for event detectors that use RNNs on a set of logarithmic
        filtered spectrograms for signal processing
    """

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
        """
        Sets up the object. Check the docs in the EventDetection class for
        further parameters.

        :param data:         see EventDetection class
        :param nn_files:     list of files that define the RNN structure
        :param fps:          frames per second
        :param online:       sets if online processing is desired
        :param window_sizes: list of window sizes for spectrogram computation
        :param mul:          multiplier for logarithmic spectra
        :param add:          shift for logarithmic spectra
        :param norm_filters: sets if the logarithmic filterbank shall be
                             normalised
        :param n_threads:    number of threads for rnn processing
        """

        super(RnnEventDetection, self).__init__(data, fps=fps, **kwargs)

        if nn_files is None and self._activations is None:
            raise RuntimeError('Either specify neural network files or load '
                               'activations from file!')

        self.nn_files = nn_files
        self.bands_per_octave = bands_per_octave
        self.window_sizes = window_sizes
        self.mul = mul
        self.add = add
        self.norm_filters = norm_filters
        self.n_threads = n_threads
        self.origin = 'online' if online else 0


    def process(self):
        """ See EventDetection class """
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
