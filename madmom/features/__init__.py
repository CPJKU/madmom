# encoding: utf-8
"""
This package includes higher level features. Your definition of "higher" may
vary, but all "lower" level features can be found the `audio` package.

"""

import numpy as np


# helper functions
def smooth_signal(signal, smooth):
    """
    Smooth the given signal.

    :param signal: signal
    :param smooth: smoothing kernel [numpy array or int]
    :return:       smoothed signal

    Note: If 'smooth' is an integer, a Hamming window of that length will be
          used as a smoothing kernel.

    """
    # return signal if no smoothing is required
    if not smooth:
        return signal
    # init smoothing kernel
    kernel = None
    # size for the smoothing kernel is given
    if isinstance(smooth, int):
        if smooth > 1:
            # use a Hamming window of given length
            kernel = np.hamming(smooth)
    # otherwise use the given smoothing kernel directly
    elif isinstance(smooth, np.ndarray):
        if len(smooth) > 1:
            kernel = smooth
    # check if a kernel is given
    if kernel is None:
        raise ValueError('can not smooth signal with %s' % smooth)
    # convolve with the kernel and return
    if signal.ndim == 1:
        return np.convolve(signal, kernel, 'same')
    elif signal.ndim == 2:
        from scipy.signal import convolve2d
        return convolve2d(signal, kernel[:, np.newaxis], 'same')
    else:
        raise ValueError('signal must be either 1D or 2D')


class Activations(np.ndarray):
    """
    Activations class.

    """
    def __new__(cls, data, fps=None, sep=None):
        """
        Instantiate a new Activations object.

        :param data: either a numpy array or filename or file handle
        :param fps:  frames per second
        :param sep:  separator between activation values
        :return:     Activations instance

        Note: If a filename or file handle is given, an undefined or empty (“”)
              separator means that the file should be treated as a numpy binary
              file.
              Only binary files can store the frame rate of the activations.
              Text files should not be used for anything else but manual
              inspection or I/O with other programs.

              The activations are stored/saved/kept as np.float32.

        """
        # check the type of the given data
        if isinstance(data, np.ndarray):
            # cast to Activations
            obj = np.asarray(data, dtype=np.float32).view(cls)
            obj.fps = fps
        elif isinstance(data, (basestring, file)):
            # read from file or file handle
            obj = cls.load(data, fps, sep)
        else:
            raise TypeError("wrong input data for Activations")
        # frame rate must be set
        if obj.fps is None:
            raise TypeError("frame rate for Activations must be set")
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here
        self.fps = getattr(obj, 'fps', None)

    @classmethod
    def load(cls, infile, fps=None, sep=None):
        """
        Load the activations from a file.

        :param infile: input file name or file handle
        :param fps:    frame rate of the activations
                       if set, it overwrites the saved frame rate
        :param sep:    separator between activation values
        :return:       Activations instance

        Note: An undefined or empty (“”) separator means that the file should
              be treated as a numpy binary file.
              Only binary files can store the frame rate of the activations.
              Text files should not be used for anything else but manual
              inspection or I/O with other programs.

        """
        # load the activations
        if sep in [None, '']:
            # numpy binary format
            data = np.load(infile)
            if isinstance(data, np.lib.npyio.NpzFile):
                # .npz file, set the frame rate if none is given
                if fps is None:
                    fps = float(data['fps'])
                # and overwrite the data
                data = data['activations']
        else:
            # simple text format
            data = np.loadtxt(infile, delimiter=sep)
        if data.ndim > 1:
            # flatten the array if it has only 1 real dimension
            if data.shape[1] == 1:
                data = data.flatten()
        # instantiate a new object
        return cls(data, fps)

    def save(self, outfile, sep=None):
        """
        Save the activations to a file.

        :param outfile: output file name or file handle
        :param sep:     separator between activation values

        Note: An undefined or empty (“”) separator means that the file should
              be written as a numpy binary file.
              Only binary files can store the frame rate of the activations.
              Text files should not be used for anything else but manual
              inspection or I/O with other programs.

        """
        # save the activations
        if sep in [None, '']:
            # numpy binary format
            npz = {'activations': self,
                   'fps': self.fps}
            np.savez(outfile, **npz)
        else:
            # simple text format
            np.savetxt(outfile, self, fmt='%.5f', delimiter=sep)

    @staticmethod
    def add_arguments(parser):
        """
        Add options to save/load activations to an existing parser object.

        :param parser: existing argparse parser object
        :return:       input/output argument parser group object

        """
        # add onset detection related options to the existing parser
        g = parser.add_argument_group('save/load the activations')
        # add options for saving and loading the activations
        g.add_argument('-s', dest='save', action='store_true', default=False,
                       help='save the activations to file')
        g.add_argument('-l', dest='load', action='store_true', default=False,
                       help='load the activations from file')
        g.add_argument('--sep', action='store', default=None,
                       help='separator for saving/loading the activations '
                            '[default: None, i.e. numpy binary format]')
        # return the argument group so it can be modified if needed
        return g


class EventDetection(object):
    """
    Base class for anything that detects events in an audio signal.

    """

    def __init__(self, signal, *args, **kwargs):
        """
        Instantiate an EventDetection object from a Signal instance.

        :param signal: Signal instance or input file name or file handle

        :param args:   additional arguments passed to Signal()
        :param kwargs: additional arguments passed to Signal()

        Note: the method calls the pre_process() method with the Signal to
              obtain data suitable to be further processed by the process()
              method to compute the activations.

        """
        from madmom.audio.signal import Signal
        # load the Signal
        if isinstance(Signal, Signal) or signal is None:
            # already a Signal instance
            self.signal = signal
        else:
            # try to instantiate a Signal object
            self.signal = Signal(signal, *args, **kwargs)
        # init fps, data, activations and detections
        self._fps = None
        self._data = None
        self._activations = None
        self._detections = None

    @property
    def fps(self):
        """Frames rate."""
        if self._fps is None:
            # try to get the frame rate from the activations
            return self.activations.fps
        return self._fps

    @property
    def data(self):
        """The pre-processed data."""
        if self._data is None:
            self.pre_process()
        return self._data

    def pre_process(self):
        """
        Pre-process the signal and return data suitable for further processing.
        This method should be implemented by subclasses.

        :return:  data suitable for further processing.

        Note: The method is expected to pre-process the signal into data
              suitable for further processing and save it to self._data.
              Additionally it should return the data itself.

        """
        # functionality should be implemented by subclasses
        self._data = self.signal
        return self._data

    @classmethod
    def from_data(cls, data, fps=None):
        """
        Instantiate an EventDetection object from the given pre-processed data.

        :param data: data to be used for further processing
        :param fps:  frame rate of the data
        :return:     EventDetection instance

        """
        # instantiate an EventDetection object (without a signal attribute)
        obj = cls(None)
        # load the data
        obj._data = data
        # set the frame rate
        if fps:
            obj._fps = fps
        # return the newly created object
        return obj

    @property
    def activations(self):
        """The activations."""
        if self._activations is None:
            self.process()
        return self._activations

    def process(self):
        """
        Process the data and compute the activations.
        This method should be implemented by subclasses.

        :return:  activations computed from the signal

        Note: The method is expected to compute the activations from the
              data and save the activations to self._activations.
              Additionally it should return the activations itself.

        """
        # functionality should be implemented by subclasses
        self._activations = self._data
        return self._activations

    @classmethod
    def from_activations(cls, activations, fps=None, sep=None):
        """
        Instantiate an EventDetection object from an Activations instance.

        :param activations: Activations instance or input file name or file
                            handle
        :param fps:         frames per second
        :param sep:         separator between activation values
        :return:            EventDetection instance

        Note: An undefined or empty (“”) separator means that the file should
              be treated as a numpy binary file.

        """
        # instantiate an EventDetection object (without a signal attribute)
        obj = cls(None)
        # load the Activations
        if isinstance(activations, Activations):
            # already an Activations instance
            obj._activations = activations
            if fps:
                # overwrite the frame rate
                obj._activations.fps = fps
        else:
            # try to instantiate an Activations object
            obj._activations = Activations(activations, fps, sep)
        # return the newly created object
        return obj

    @property
    def detections(self):
        """The detected events."""
        if self._detections is None:
            self.detect()
        return self._detections

    def detect(self):
        """
        Extracts the events (beats, onsets, ...) from the activations.
        This method should be implemented by subclasses.

        :return:  the detected events

        Note: The method is expected to compute the detections from the
              activations and save the detections to self._detections.
              Additionally it should return the detections itself.

        """
        # functionality should be implemented by subclasses
        self._detections = self._activations
        return self._detections

    def write(self, filename):
        """
        Write the detected events to a file.

        :param filename: output file name or file handle

        """
        from madmom.utils import write_events
        write_events(self.detections, filename)


class RNNEventDetection(EventDetection):
    """
    Base class for anything that use RNNs to detects events in an audio signal.

    """
    import multiprocessing as mp
    NUM_THREADS = mp.cpu_count()

    def __init__(self, signal, nn_files, num_threads=NUM_THREADS,
                 *args, **kwargs):
        """
        Sets up the object. Check the docs in the EventDetection class for
        further parameters.

        :param signal:      Signal instance or input file name or file handle
        :param nn_files:    list of files that define the RNN
        :param num_threads: number of threads for rnn processing

        :param args:        additional arguments passed to EventDetection()
        :param kwargs:      additional arguments passed to EventDetection()

        """

        super(RNNEventDetection, self).__init__(signal, *args, **kwargs)
        self.nn_files = nn_files
        self.num_threads = num_threads

    def pre_process(self, frame_sizes, bands_per_octave, origin='offline',
                    mul=1, ratio=0.5):
        """
        Pre-process the signal to obtain a data representation suitable for RNN
        processing.

        :param frame_sizes:      frame sizes for the spectrograms
        :param bands_per_octave: bands per octave for the filterbank
        :param origin:           origin of the frames
        :param mul:              multiplication factor for logarithm
        :param ratio:            frame overlap ratio for diff
        :return:                 pre-processed data

        """
        from madmom.audio.spectrogram import LogFiltSpec
        data = []
        # FIXME: remove this hack!
        fps = 100
        # set the frame rate
        self._fps = fps
        # pre-process the signal
        for frame_size in frame_sizes:
            # TODO: the signal processing parameters should be included in and
            #       extracted from the NN model files
            s = LogFiltSpec(self.signal, frame_size=frame_size, fps=fps,
                            origin=origin, bands_per_octave=bands_per_octave,
                            mul=mul, add=1, norm_filters=True, fmin=30,
                            fmax=17000, ratio=ratio)
            # append the spec and the positive first order diff to the data
            data.append(s.spec)
            data.append(s.pos_diff)
        # stack the data and return it
        self._data = np.hstack(data)
        return self._data

    def process(self):
        """
        Computes the predictions on the data with the RNN models defined/given
        and save them as activations.

        :return: averaged RNN predictions

        """
        from madmom.ml.rnn import process_rnn
        # compute the predictions with RNNs
        predictions = process_rnn(self.data, self.nn_files, self.num_threads)
        # save the predictions as activations
        self._activations = Activations(predictions.ravel(), self.fps)
        # and return them
        return self._activations

    @classmethod
    def add_arguments(cls, parser, nn_files, num_threads=NUM_THREADS):
        """
        Add neural network testing options to an existing parser object.

        :param parser:      existing argparse parser object
        :param nn_files:    list with files of NN models
        :param num_threads: number of threads to run in parallel
        :return:            neural network argument parser group object

        """
        # add neural network related options to the existing parser
        g = parser.add_argument_group('neural network arguments')
        from madmom.utils import OverrideDefaultListAction
        g.add_argument('--nn_files', action=OverrideDefaultListAction,
                       type=str, default=nn_files,
                       help='average the predictions of these pre-trained '
                            'neural networks (multiple files can be given, '
                            'one file per argument)')
        g.add_argument('--threads', dest='num_threads', action='store',
                       type=int, default=num_threads,
                       help='number of parallel threads [default=%(default)s]')
        # return the argument group so it can be modified if needed
        return g
