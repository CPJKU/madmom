#!/usr/bin/env python
# encoding: utf-8
"""
This file contains note transcription related functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import glob
import numpy as np

from madmom import MODELS_PATH, IOProcessor
from madmom.utils import open

from . import ActivationsProcessor
from .onsets import PeakPicking
from ..audio.signal import SignalProcessor
from ..audio.spectrogram import StackSpectrogramProcessor
from ..ml.rnn import RNNProcessor, average_predictions


def load_notes(filename):
    """
    Load the target notes from a file.

    :param filename: input file name or file handle
    :return:         numpy array with notes

    """
    with open(filename, 'rb') as f:
        return np.loadtxt(f)


def write_notes(notes, filename, sep='\t'):
    """
    Write the detected notes to a file.

    :param notes:    list with notes
    :param filename: output file name or file handle
    :param sep:      separator for the fields [default='\t']

    """
    from madmom.utils import open
    # write the detected notes to the output
    with open(filename, 'wb') as f:
        for note in notes:
            f.write(sep.join([str(x) for x in note]) + '\n')


def write_midi(notes, filename, note_length=0.6, note_velocity=100):
    """
    Write the notes to a MIDI file.

    :param notes:         detected notes
    :param filename:      output filename
    :param note_velocity: default velocity of the notes
    :param note_length:   default length of the notes
    :return:              numpy array with notes (including note length &
                          velocity)

    """
    from madmom.utils.midi import process_notes
    # expand the array to have a length and velocity
    notes = np.hstack((notes, np.ones_like(notes)))
    # set dummy offset
    notes[:, 2] = notes[:, 0] + note_length
    # set dummy velocity
    notes[:, 3] *= note_velocity
    # write the notes to the file and return them
    return process_notes(notes, filename)


def write_frequencies(notes, filename, note_length=0.6):
    """
    Write the frequencies of the notes to file (i.e. MIREX format).

    :param notes:       detected notes
    :param filename:    output filename
    :param note_length: default length of the notes
    :return:            notes

    """
    from madmom.audio.filters import midi2hz
    from madmom.utils import open
    # MIREX format: onset \t offset \t frequency
    with open(filename, 'wb') as f:
        for note in notes:
            onset, midi_note = note
            offset = onset + note_length
            frequency = midi2hz(midi_note)
            f.write('%.2f\t%.2f\t%.2f\n' % (onset, offset, frequency))
    return notes


def note_reshaper(notes):
    """

    :param notes: numpy array with note activations
    :return:      reshaped array to represent the 88 MIDI notes

    """
    return notes.reshape(-1, 88)


# TODO: These do not seem to be used anywhere
FMIN = 27.5
FMAX = 18000


class RNNNoteTranscription(IOProcessor):
    """
    Class for detecting onsets with a recurrent neural network (RNN).

    """
    # NN model files
    NN_FILES = glob.glob("%s/notes_brnn*npz" % MODELS_PATH)
    # default values for note peak-picking
    THRESHOLD = 0.35
    SMOOTH = 0.09
    COMBINE = 0.05

    def __init__(self, nn_files=NN_FILES, threshold=THRESHOLD, smooth=SMOOTH,
                 combine=COMBINE, output_format=None, load=False, save=False,
                 **kwargs):
        """
        Processor for finding possible onset positions in a signal.

        :param nn_files: list of RNN model files

        """
        # FIXME: remove this hack of setting fps here
        #        all information should be stored in the nn_files or in a
        #        pickled Processor (including information about spectrograms,
        #        mul, add & diff_ratio and so on)
        kwargs['fps'] = fps = 100
        # input processor chain
        sig = SignalProcessor(mono=True, **kwargs)
        stack = StackSpectrogramProcessor(frame_sizes=[1024, 2048, 4096],
                                          bands=12, online=False,
                                          norm_filters=True, mul=5,
                                          add=1, diff_ratio=0.5, **kwargs)
        rnn = RNNProcessor(nn_files=nn_files, **kwargs)
        avg = average_predictions
        reshape = note_reshaper
        pp = PeakPicking(threshold=threshold, smooth=smooth, pre_max=1. / fps,
                         post_max=1. / fps, combine=combine)
        # define input and output processors
        in_processor = [sig, stack, rnn, avg, reshape]
        if output_format is None:
            output = write_notes
        elif output_format == 'midi':
            output = write_midi
        elif output_format == 'mirex':
            output = write_frequencies
        else:
            raise ValueError('unknown `output_format`: %s' % output_format)
        out_processor = [pp, output]
        # swap in/out processors if needed
        if load:
            in_processor = ActivationsProcessor(mode='r', **kwargs)
        if save:
            out_processor = ActivationsProcessor(mode='w', **kwargs)
        # make this an IOProcessor by defining input and output processors
        super(RNNNoteTranscription, self).__init__(in_processor, out_processor)

    @classmethod
    def add_arguments(cls, parser, nn_files=NN_FILES, threshold=THRESHOLD,
                      smooth=SMOOTH, combine=COMBINE):
        """
        Add note transcription related arguments to an existing parser.

        :param parser:    existing argparse parser
        :param nn_files:  list with files of NN models
        :param threshold: threshold for peak-picking
        :param smooth:    smooth the note activations over N seconds
        :param combine:   only report one note within N seconds and pitch
        :return:          note argument parser group

        """
        # add Activations parser
        ActivationsProcessor.add_arguments(parser)
        # add RNNEventDetection arguments
        RNNProcessor.add_arguments(parser, nn_files=nn_files)
        # add note transcription related options to the existing parser
        g = parser.add_argument_group('note transcription arguments')
        g.add_argument('-t', dest='threshold', action='store', type=float,
                       default=threshold, help='detection threshold '
                       '[default=%(default)s]')
        g.add_argument('--smooth', action='store', type=float, default=smooth,
                       help='smooth the note activations over N seconds '
                       '[default=%(default).2f]')
        g.add_argument('--combine', action='store', type=float,
                       default=combine, help='combine notes within N seconds '
                       '(per pitch) [default=%(default).2f]')
        # return the argument group so it can be modified if needed
        return g
