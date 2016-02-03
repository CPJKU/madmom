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
    The file format must be (duration and velocity being optional):

    'note_time' 'MIDI_note' ['duration' ['MIDI_velocity']]

    with one note per line and individual fields separated by whitespace.

    """
    return np.loadtxt(filename)


def expand_notes(notes, duration=0.6, velocity=100):
    """
    Expand the notes to include all columns.

    Parameters
    ----------
    notes : numpy array, shape (num_notes, 2)
        Notes, one per row (column definition see notes).
    duration : float, optional
        Note duration if not defined by `notes`.
    velocity : int, optional
        Note velocity if not defined by `notes`.

    Returns
    -------
    numpy array
        Notes (including note duration and velocity).

    Notes
    -----
    The note columns format must be (duration and velocity being optional):

    'note_time' 'MIDI_note' ['duration' ['MIDI_velocity']]

    """
    if not notes.ndim == 2:
        raise ValueError('unknown format for `notes`')
    rows, columns = notes.shape
    if columns == 4:
        return notes
    elif columns == 3:
        new_columns = np.ones((rows, 1)) * velocity
    elif columns == 2:
        new_columns = np.ones((rows, 2)) * velocity
        new_columns[:, 0] = duration
    else:
        raise ValueError('unable to handle `notes` with %d columns' % columns)
    # return the notes
    notes = np.hstack((notes, new_columns))
    return notes


def write_notes(notes, filename, sep='\t', fmt=None, header=''):
    """
    Write the notes to a file (as many columns as given).

    Parameters
    ----------
    notes : numpy array, shape (num_notes, 2)
        Notes, one per row (column definition see notes).
    filename : str or file handle
        Output filename or handle.
    sep : str, optional
        Separator for the fields.
    fmt : list, optional
        Format of the fields (i.e. columns, see notes)
    header : str, optional
        Header to be written (as a comment).

    Returns
    -------
    numpy array
        Notes.

    Notes
    -----
    The note columns format must be (duration and velocity being optional):

    'note_time' 'MIDI_note' ['duration' ['MIDI_velocity']]

    """
    if fmt is None:
        fmt = list(('%.3f', '%d', '%.3f', '%d'))
    from madmom.utils import write_events
    if not notes.ndim == 2:
        raise ValueError('unknown format for `notes`')
    # truncate to the number of colums given
    fmt = sep.join(fmt[:notes.shape[1]])
    # write the notes
    write_events(notes, filename, fmt=fmt, header=header)
    # also return them
    return notes


def write_midi(notes, filename, duration=0.6, velocity=100):
    """
    Write the notes to a MIDI file.

    Parameters
    ----------
    notes : numpy array, shape (num_notes, 2)
        Notes, one per row (column definition see notes).
    filename : str
        Output MIDI file.
    duration : float, optional
        Note duration if not defined by `notes`.
    velocity : int, optional
        Note velocity if not defined by `notes`.

    Returns
    -------
    numpy array
        Notes (including note length and velocity).

    Notes
    -----
    The note columns format must be (duration and velocity being optional):

    'note_time' 'MIDI_note' ['duration' ['MIDI_velocity']]

    """
    from madmom.utils.midi import process_notes
    # expand the array to have a default duration and velocity
    notes = expand_notes(notes, duration, velocity)
    # write the notes to the file and return them
    return process_notes(notes, filename)


def write_mirex_format(notes, filename, duration=0.6):
    """
    Write the frequencies of the notes to file (in MIREX format).

    Parameters
    ----------
    notes : numpy array, shape (num_notes, 2)
        Notes, one per row (column definition see notes).
    filename : str or file handle
        Output filename or handle.
    duration : float, optional
        Note duration if not defined by `notes`.

    Returns
    -------
    numpy array
        Notes in MIREX format.

    Notes
    -----
    The note columns format must be (duration and velocity being optional):

    'note_time' 'MIDI_note' ['duration' ['MIDI_velocity']]

    The output format required by MIREX is:

    'onset_time' 'offset_time' 'note_frequency'

    """
    from madmom.audio.filters import midi2hz
    # expand the notes if needed
    notes = expand_notes(notes, duration)
    # report offset time instead of duration
    notes = np.vstack((notes[:, 0], notes[:, 0] + notes[:, 2],
                       midi2hz(notes[:, 1]))).T
    # MIREX format: onset \t offset \t frequency
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
