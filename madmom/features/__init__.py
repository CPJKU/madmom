# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=wrong-import-position
"""
This package includes high-level features. Your definition of "high" may
vary, but we define high-level features as the ones you want to evaluate (e.g.
onsets, beats, etc.). All lower-level features can be found the `madmom.audio`
package.

Notes
-----
All features should be implemented as classes which inherit from Processor
(or provide a XYZProcessor(Processor) variant). This way, multiple Processor
objects can be chained/combined to achieve the wanted functionality.


"""

from __future__ import absolute_import, division, print_function

import numpy as np

from madmom.processors import Processor


class Activations(np.ndarray):
    """
    The Activations class extends a numpy ndarray with a frame rate (fps)
    attribute.

    Parameters
    ----------
    data : str, file handle or numpy array
        Either file name/handle to read the data from or array.
    fps : float, optional
        Frames per second (must be set if `data` is given as an array).
    sep : str, optional
        Separator between activation values (if read from file).
    dtype : numpy dtype
        Data-type the activations are stored/saved/kept.

    Attributes
    ----------
    fps : float
        Frames per second.

    Notes
    -----
    If a filename or file handle is given, an undefined or empty separator
    means that the file should be treated as a numpy binary file.
    Only binary files can store the frame rate of the activations.
    Text files should not be used for anything else but manual inspection
    or I/O with other programs.

    """
    # pylint: disable=super-on-old-class
    # pylint: disable=super-init-not-called
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, data, fps=None, sep=None, dtype=np.float32):
        # this method is for documentation purposes only
        pass

    def __new__(cls, data, fps=None, sep=None, dtype=np.float32):
        import io

        # check the type of the given data
        if isinstance(data, np.ndarray):
            # cast to Activations
            obj = np.asarray(data, dtype=dtype).view(cls)
            obj.fps = fps
        elif isinstance(data, (str, io.IOBase)):
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

        Parameters
        ----------
        infile : str or file handle
            Input file name or file handle.
        fps : float, optional
            Frames per second; if set, it overwrites the saved frame rate.
        sep : str, optional
            Separator between activation values.

        Returns
        -------
        :class:`Activations` instance
            :class:`Activations` instance.

        Notes
        -----
        An undefined or empty separator means that the file should be treated
        as a numpy binary file.
        Only binary files can store the frame rate of the activations.
        Text files should not be used for anything else but manual inspection
        or I/O with other programs.

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
        if data.ndim > 1 and data.shape[1] == 1:
            # flatten the array if it has only 1 real dimension
            data = data.flatten()
        # instantiate a new object
        return cls(data, fps)

    def save(self, outfile, sep=None, fmt='%.5f'):
        """
        Save the activations to a file.

        Parameters
        ----------
        outfile : str or file handle
            Output file name or file handle.
        sep : str, optional
            Separator between activation values if saved as text file.
        fmt : str, optional
            Format of the values if saved as text file.

        Notes
        -----
        An undefined or empty separator means that the file should be treated
        as a numpy binary file.
        Only binary files can store the frame rate of the activations.
        Text files should not be used for anything else but manual inspection
        or I/O with other programs.

        If the activations are a 1D array, its values are interpreted as
        features of a single time step, i.e. all values are printed in a single
        line. If you want each value to appear in an individual line, use '\\n'
        as a separator.

        If the activations are a 2D array, the first axis corresponds to the
        time dimension, i.e. the features are separated by `sep` and the time
        steps are printed in separate lines. If you like to swap the
        dimensions, please use the `T` attribute.

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

    Parameters
    ----------
    mode : {'r', 'w', 'in', 'out', 'load', 'save'}
        Mode of the Processor: read/write.
    fps : float, optional
        Frame rate of the activations (if set, it overwrites the saved frame
        rate).
    sep : str, optional
        Separator between activation values if saved as text file.

    Notes
    -----
    An undefined or empty (“”) separator means that the file should be treated
    as a numpy binary file. Only binary files can store the frame rate of the
    activations.

    """

    def __init__(self, mode, fps=None, sep=None, **kwargs):
        # pylint: disable=unused-argument
        self.mode = mode
        self.fps = fps
        self.sep = sep

    def process(self, data, output=None, **kwargs):
        """
        Depending on the mode, either loads the data stored in the given file
        and returns it as an Activations instance or save the data to the given
        output.

        Parameters
        ----------
        data : str, file handle or numpy array
            Data or file to be loaded (if `mode` is 'r') or data to be saved
            to file (if `mode` is 'w').
        output : str or file handle, optional
            output file (only in write-mode)

        Returns
        -------
        :class:`Activations` instance
            :class:`Activations` instance (only in read-mode)

        """
        # pylint: disable=arguments-differ

        if self.mode in ('r', 'in', 'load'):
            return Activations.load(data, fps=self.fps, sep=self.sep)
        if self.mode in ('w', 'out', 'save'):
            # TODO: should we return the data or the Activations instance?
            Activations(data, fps=self.fps).save(output, sep=self.sep)
        else:
            raise ValueError("wrong mode %s; choose {'r', 'w', 'in', 'out', "
                             "'load', 'save'}")
        return data

    @staticmethod
    def add_arguments(parser):
        """
        Add options to save/load activations to an existing parser.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser.

        Returns
        -------
        parser_group : argparse argument group
            Input/output argument parser group.

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
from . import onsets, beats, notes, tempo, chords

# import often used classes
from .beats import (BeatDetectionProcessor, BeatTrackingProcessor,
                    CRFBeatDetectionProcessor, DBNBeatTrackingProcessor,
                    DBNDownBeatTrackingProcessor, MultiModelSelectionProcessor,
                    PatternTrackingProcessor, RNNBeatProcessor,
                    RNNDownBeatProcessor)
from .chords import (CNNChordFeatureProcessor, CRFChordRecognitionProcessor,
                     DeepChromaChordRecognitionProcessor)
from .notes import RNNPianoNoteProcessor, NotePeakPickingProcessor
from .onsets import (CNNOnsetProcessor, OnsetPeakPickingProcessor,
                     PeakPickingProcessor,
                     RNNOnsetProcessor, SpectralOnsetProcessor)
from .tempo import TempoEstimationProcessor
