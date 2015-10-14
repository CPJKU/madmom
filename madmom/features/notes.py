# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments

"""
This file contains note transcription related functionality.

"""

import numpy as np

from madmom.utils import suppress_warnings, open


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
    Reshapes the activations produced by a RNN to ave the right shape.

    :param notes: numpy array with note activations
    :return:      reshaped array to represent the 88 MIDI notes

    """
    return notes.reshape(-1, 88)
