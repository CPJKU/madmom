# encoding: utf-8
"""
This package includes higher level features. Your definition of "higher" may
vary, but all "lower" level features can be found the `audio` package.

"""

import numpy as np

from madmom import Processor


class Activations(np.ndarray):
    """
    The Activations class extends a numpy ndarray with a frame rate (fps)
    attribute.

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
