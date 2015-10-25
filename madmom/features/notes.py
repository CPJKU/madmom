# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This file contains note transcription related functionality.

"""

from __future__ import absolute_import, division, print_function

import numpy as np

from madmom.utils import suppress_warnings


@suppress_warnings
def load_notes(filename):
    """
    Load the target notes from a file.

    :param filename: input file name or file handle
    :return:         numpy array with notes

    """
    return np.loadtxt(filename)


def write_notes(notes, filename, sep='\t',
                fmt=list(('%.3f', '%d', '%.3f', '%d'))):
    """
    Write the detected notes to a file.

    :param notes:         2D numpy array with notes
    :param filename:      output file name or file handle
    :param sep:           separator for the fields [default='\t']
    :param fmt:           format of the fields (i.e. columns, see below)


    Note: The `notes` must be a 2D numpy array with the individual notes as
          rows, and the columns defined as:

          'note_time' 'MIDI_note' ['duration' ['MIDI_velocity']]

          whith the duration and velocity being optional.

    """
    from madmom.utils import write_events
    # truncate to the number of colums given
    if notes.ndim == 1:
        fmt = '%f'
    elif notes.ndim == 2:
        fmt = sep.join(fmt[:notes.shape[1]])
    else:
        raise ValueError('unknown format for notes')
    # write the notes
    write_events(notes, filename, fmt=fmt)
    # also return them
    return notes


def write_midi(notes, filename, note_length=0.6, note_velocity=100):
    """
    Write the notes to a MIDI file.

    :param notes:         2D numpy array with notes
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


def write_mirex_format(notes, filename, note_length=0.6):
    """
    Write the frequencies of the notes to file (in MIREX format).

    :param notes:       detected notes
    :param filename:    output filename
    :param note_length: default length of the notes
    :return:            notes

    """
    from madmom.audio.filters import midi2hz
    # MIREX format: onset \t offset \t frequency
    notes = np.vstack((notes[:, 0], notes[:, 0] + note_length,
                       midi2hz(notes[:, 1]))).T
    write_notes(notes, filename, fmt=list(('%.3f', '%.3f', '%.1f', )))
    return notes


def note_reshaper(notes):
    """
    Reshapes the activations produced by a RNN to have the right shape.

    :param notes: numpy array with note activations
    :return:      reshaped array to represent the 88 MIDI notes

    """
    return notes.reshape(-1, 88)
