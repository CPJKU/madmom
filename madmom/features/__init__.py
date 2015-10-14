# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments

"""
This package includes higher level features. Your definition of "higher" may
vary, but all "lower" level features can be found the `audio` package.

"""

import numpy as np

from madmom.processors import Processor


class Activations(np.ndarray):
    """
    The Activations class extends a numpy ndarray with a frame rate (fps)
    attribute.

    """
    # pylint: disable=super-on-old-class
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, data, fps=None, sep=None, dtype=np.float32):
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
        # this method exists only for argument documentation purposes
        # the initialisation is done in __new__() and __array_finalize__()

    def __new__(cls, data, fps=None, sep=None, dtype=np.float32):
        # check the type of the given data
        if isinstance(data, np.ndarray):
            # cast to Activations
            obj = np.asarray(data, dtype=dtype).view(cls)
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

    def __reduce__(self):
        # needed for correct pickling
        # source: http://stackoverflow.com/questions/26598109/
        # get the parent's __reduce__ tuple
        pickled_state = super(Activations, self).__reduce__()
        # create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.fps,)
        # return a tuple that replaces the parent's __reduce__ tuple
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):
        # needed for correct un-pickling
        # set the attributes
        self.fps = state[-1]
        # call the parent's __setstate__ with the other tuple elements
        super(Activations, self).__setstate__(state[0:-1])

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

    def save(self, outfile, sep=None, fmt='%.5f'):
        """
        Save the activations to a file.

        :param outfile: output file name or file handle
        :param sep:     separator between activation values (see below)
        :param fmt:     format of the values if stored as text file

        Note: An undefined or empty (“”) separator means that the file should
              be written as a numpy binary file.
              Only binary files can store the frame rate of the activations.
              Text files should not be used for anything else but manual
              inspection or I/O with other programs.
              If the activations are a 1d array, its values are interpreted as
              features of a single time step, i.e. all values are printed in a
              single line. If you want each value to appear in an individual
              line, use '\n' as a separator.
              If the activations are a 2d array, the first axis corresponds to
              the time dimension, i.e. the features are separated by `sep` and
              the time steps are printed in separate lines. If you like to swap
              the dimensions, please use the '.T' operator.
              Arrays of other dimensions are not supported.

        """
        # save the activations
        if sep in [None, '']:
            # numpy binary format
            npz = {'activations': self,
                   'fps': self.fps}
            np.savez(outfile, **npz)
        else:
            if self.ndim > 2:
                raise ValueError('Only 1D and 2D activations can be saved in '
                                 'human readable text format.')
            # simple text format
            header = "FPS:%f" % self.fps
            np.savetxt(outfile, np.atleast_2d(self), fmt=fmt, delimiter=sep,
                       header=header)


class ActivationsProcessor(Processor):
    """
    ActivationsProcessor processes a file and returns an Activations instance.

    """

    def __init__(self, mode, fps=None, sep=None, **kwargs):
        """

        :param mode: read/write mode of the Processor ['r', 'w']
        :param fps:  frame rate of the activations
                     (if set, it overwrites the saved frame rate)
        :param sep:  separator between activation values

        Note: An undefined or empty (“”) separator means that the file should
              be treated as a numpy binary file.
              Only binary files can store the frame rate of the activations.

        """
        # pylint: disable=unused-argument

        self.mode = mode
        self.fps = fps
        self.sep = sep

    def process(self, data, output=None):
        """
        Loads the data stored in the given file and returns it as an
        Activations instance.

        :param data:   input data or file to be loaded
                       [numpy array or file name or file handle]
        :param output: output file [file name or file handle]
        :return:       Activations instance

        """
        # pylint: disable=arguments-differ

        if self.mode in ('r', 'in', 'load'):
            return Activations.load(data, fps=self.fps, sep=self.sep)
        if self.mode in ('w', 'out', 'save'):
            Activations(data, fps=self.fps).save(output, sep=self.sep)
        else:
            raise ValueError("wrong mode %s; choose {'r', 'w', 'in', 'out', "
                             "'load', 'save'}")
        return data

    @classmethod
    def add_arguments(cls, parser):
        """
        Add options to save/load activations to an existing parser.

        :param parser: existing argparse parser
        :return:       input/output argument parser group

        """
        # add onset detection related options to the existing parser
        g = parser.add_argument_group('save/load the activations')
        # add options for saving and loading the activations
        g.add_argument('--save', action='store_true', default=False,
                       help='save the activations to file')
        g.add_argument('--load', action='store_true', default=False,
                       help='load the activations from file')
        g.add_argument('--sep', action='store', default=None,
                       help='separator for saving/loading the activations '
                            '[default: None, i.e. numpy binary format]')
        # return the argument group so it can be modified if needed
        return g


# finally import the submodules
from . import onsets, beats, notes, tempo
