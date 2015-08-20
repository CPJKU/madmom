#!/usr/bin/env python
# encoding: utf-8
"""
This file contains note transcription related functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import glob

import numpy as np

from madmom import MODELS_PATH
from madmom.utils import suppress_warnings, open
from madmom.processors import SequentialProcessor
from madmom.audio.signal import SignalProcessor
from madmom.audio.spectrogram import (LogarithmicFilteredSpectrogramProcessor,
                                      SpectrogramDifferenceProcessor,
                                      StackedSpectrogramProcessor)
from madmom.ml.rnn import RNNProcessor, average_predictions


@suppress_warnings
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
    # write the notes to the output
    if filename is not None:
        with open(filename, 'wb') as f:
            for note in notes:
                f.write(sep.join([str(x) for x in note]) + '\n')
    # also return them
    return notes


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


class RNNNoteProcessor(SequentialProcessor):
    """
    Class for detecting notes with a recurrent neural network (RNN).

    """
    # NN model files
    NN_FILES = glob.glob("%s/notes_brnn*npz" % MODELS_PATH)

    def __init__(self, nn_files=NN_FILES, **kwargs):
        """
        Processor for finding possible notes positions in a signal.

        :param nn_files: list of RNN model files

        """
        # FIXME: remove this hack of setting fps here
        #        all information should be stored in the nn_files or in a
        #        pickled Processor (including information about spectrograms,
        #        mul, add & diff_ratio and so on)
        kwargs['fps'] = self.fps = 100
        # processing chain
        sig = SignalProcessor(num_channels=1, sample_rate=44100, **kwargs)
        # we need to define how specs and diffs should be stacked
        spec = LogarithmicFilteredSpectrogramProcessor(num_bands=12,
                                                       norm_filters=True,
                                                       mul=5, add=1)
        diff = SpectrogramDifferenceProcessor(diff_ratio=0.5,
                                              positive_diffs=True)
        # stack specs with the given frame sizes
        stack = StackedSpectrogramProcessor(frame_size=[1024, 2048, 4096],
                                            spectrogram=spec, difference=diff,
                                            **kwargs)
        rnn = RNNProcessor(nn_files=nn_files, **kwargs)
        avg = average_predictions
        reshape = note_reshaper
        # sequentially process everything
        super(RNNNoteProcessor, self).__init__([sig, stack, rnn, avg, reshape])

    @classmethod
    def add_arguments(cls, parser, nn_files=NN_FILES):
        """
        Add note transcription related arguments to an existing parser.

        :param parser:    existing argparse parser
        :param nn_files:  list with files of NN models
        :return:          note argument parser group

        """
        # add RNN processing arguments
        RNNProcessor.add_arguments(parser, nn_files=nn_files)
