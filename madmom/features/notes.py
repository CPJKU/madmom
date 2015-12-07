# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains note transcription related functionality.

"""

from __future__ import absolute_import, division, print_function

import numpy as np

from madmom.utils import suppress_warnings


@suppress_warnings
def load_notes(filename):
    """
    Load the notes from a file.

    Parameters
    ----------
    filename : str or file handle
        Input file to load the notes from.

    Returns
    -------
    numpy array
        Notes.

    Notes
    -----
    The file format must be:

    'note_time' 'MIDI_note' ['duration' ['MIDI_velocity']]

    with one note per line and individual fields separated by whitespace.
    Duration and velocity is optional (but must be the same for all notes).

    """
    return np.loadtxt(filename)


def write_notes(notes, filename, sep='\t',
                fmt=list(('%.3f', '%d', '%.3f', '%d'))):
    """
    Write the notes to a file.

    Parameters
    ----------
    notes : numpy array, shape (num_notes, 2)
        Notes, one per row (column definition see notes).
    filename : str or file handle
        Output filename or handle.
    sep : str, optional
        Separator for the fields.
    fmt : list
        Format of the fields (i.e. columns, see notes)

    Returns
    -------
    numpy array
        Notes.

    Notes
    -----
    The note (columns) format must be:

    'note_time' 'MIDI_note' ['duration' ['MIDI_velocity']]

    The duration and velocity columns are optional.

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

    Parameters
    ----------
    notes : numpy array, shape (num_notes, 2)
        Notes, one per row (column definition see notes).
    filename : str
        Output MIDI file.
    note_length : float, optional
        Note length if not defined by `notes`.
    note_velocity : int, optional
        Note velocity if not defined by `notes`.

    Returns
    -------
    numpy array
        Notes (including note length and velocity).

    Notes
    -----
    The note (columns) format must be:

    'note_time' 'MIDI_note' ['duration' ['MIDI_velocity']]

    The duration and velocity columns are optional.

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

    Parameters
    ----------
    notes : numpy array, shape (num_notes, 2)
        Notes, one per row (column definition see notes).
    filename : str or file handle
        Output filename or handle.
    note_length : float, optional
        Note length if not defined by `notes`.

    Returns
    -------
    numpy array
        Notes (including note length and velocity).

    Notes
    -----
    The note (columns) format must be:

    'note_time' 'MIDI_note' ['duration' ['MIDI_velocity']]

    The duration and velocity columns are optional.

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

    Parameters
    ----------
    notes : numpy array
        Note activations.

    Returns
    -------
    numpy array
        Reshaped array to represent the 88 MIDI notes.

    """
    return notes.reshape(-1, 88)
