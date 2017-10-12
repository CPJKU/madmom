# encoding: utf-8
"""
Input/output package.

"""

from __future__ import absolute_import, division, print_function

import sys

import numpy as np

from .audio import load_audio_file
from ..utils import suppress_warnings

if sys.version_info[0] < 3:
    _file_handle = file
else:
    from io import TextIOBase as _file_handle


def _open_file(fn):
    return fn if isinstance(fn, _file_handle) else open(fn, 'r')

# dtype for numpy structured arrays that contain labelled segments
# 'label' needs to be castable to str
SEGMENT_DTYPE = [('start', np.float), ('end', np.float), ('label', object)]


@suppress_warnings
def load_events(filename):
    """
    Load a events from a text file, one floating point number per line.

    Parameters
    ----------
    filename : str or file handle
        File to load the events from.

    Returns
    -------
    numpy array
        Events.

    Notes
    -----
    Comments (lines starting with '#') and additional columns are ignored,
    i.e. only the first column is returned.

    """
    # read in the events, one per line
    events = np.loadtxt(filename, ndmin=2)
    # 1st column is the event's time, the rest is ignored
    return events[:, 0]


def write_events(events, filename, fmt='%.3f', delimiter='\t', header=''):
    """
    Write the events to a file, one event per line.

    Parameters
    ----------
    events : numpy array
        Events to be written to file.
    filename : str or file handle
        File to write the events to.
    fmt : str, optional
        How to format the events.
    delimiter : str, optional
        String or character separating multiple columns.
    header : str, optional
        Header to be written (as a comment).

    Returns
    -------
    numpy array
        Events.

    Notes
    -----
    This function is just a wrapper to ``np.savetxt``, but reorders the
    arguments to used as an :class:`.processors.OutputProcessor`.

    """
    # write the events to the output
    np.savetxt(filename, np.asarray(events),
               fmt=fmt, delimiter=delimiter, header=header)
    # also return them
    return events


load_onsets = load_events
write_onsets = write_events


@suppress_warnings
def load_beats(values, downbeats=False):
    """
    Load the beats from the given values or file.

    Parameters
    ----------
    values : str, file handle, list or numpy array
        Name / values to be loaded.
    downbeats : bool, optional
        Load downbeats instead of beats.

    Returns
    -------
    numpy array
        Beats.

    Notes
    -----
    Expected format:

    'beat_time' ['beat_number']

    """
    # load the beats from the given representation
    if values is None:
        # return an empty array
        values = np.zeros(0)
    elif isinstance(values, (list, np.ndarray)):
        # convert to numpy array if possible
        # Note: use array instead of asarray because of ndmin
        values = np.array(values, dtype=np.float, ndmin=1, copy=False)
    else:
        # try to load the data from file
        values = np.loadtxt(values, ndmin=1)
    if values.ndim > 1:
        if downbeats:
            # rows with a "1" in the 2nd column are the downbeats.
            return values[values[:, 1] == 1][:, 0]
        else:
            # 1st column is the beat time, the rest is ignored
            return values[:, 0]
    return values


def write_beats(beats, filename, **kwargs):
    """
    Write the beats to a file.

    Parameters
    ----------
    beats : numpy array
        Beats to be written to file.
    filename : str or file handle
        File to write the events to.

    """
    if beats.ndim == 2:
        fmt = list(('%.3f', '%d'))
    else:
        fmt = '%.3f'
    write_events(beats, filename, fmt=fmt, **kwargs)


@suppress_warnings
def load_notes(values):
    """
    Load the notes from the given values or file.

    Parameters
    ----------
    values: str, file handle, list of tuples or numpy array
        Notes values.

    Returns
    -------
    numpy array
        Notes.

    Notes
    -----
    Expected file/tuple/row format:

    'note_time' 'MIDI_note' ['duration' ['MIDI_velocity']]

    """
    # load the notes from the given representation
    if isinstance(values, (list, np.ndarray)):
        # convert to numpy array if possible
        # Note: use array instead of asarray because of ndmin
        return np.array(values, dtype=np.float, ndmin=2, copy=False)
    else:
        # try to load the data from file
        return np.loadtxt(values, ndmin=2)


def write_notes(notes, filename, fmt=None, delimiter='\t', header=''):
    """
    Write the notes to a file.

    Parameters
    ----------
    notes : numpy array, shape (num_notes, 2)
        Notes, one per row (column definition see notes).
    filename : str or file handle
        Output filename or handle.
    fmt : list, optional
        Format of the fields (i.e. columns, see notes)
    delimiter : str, optional
        String or character separating the columns.
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

    As many columns as items per row are given are written to file.

    """
    # set default format
    if fmt is None:
        fmt = list(('%.3f', '%d', '%.3f', '%d'))
    if not notes.ndim == 2:
        raise ValueError('unknown format for `notes`')
    # truncate to the number of colums given
    fmt = delimiter.join(fmt[:notes.shape[1]])
    # write the notes
    write_events(notes, filename, fmt=fmt, header=header)
    # also return them
    return notes


def write_notes_midi(notes, filename, duration=0.6, velocity=100):
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
    from ..utils import expand_notes
    from ..utils.midi import MIDIFile
    # expand the array to have a default duration and velocity
    notes = expand_notes(notes, duration, velocity)
    # write the notes to the file and return them
    MIDIFile.from_notes(notes).write(filename)
    return notes


def write_notes_mirex(notes, filename, duration=0.6):
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
    from ..audio.filters import midi2hz
    from ..utils import expand_notes
    # expand the notes if needed
    notes = expand_notes(notes, duration)
    # report offset time instead of duration
    notes = np.vstack((notes[:, 0], notes[:, 0] + notes[:, 2],
                       midi2hz(notes[:, 1]))).T
    # MIREX format: onset \t offset \t frequency
    write_notes(notes, filename, fmt=list(('%.3f', '%.3f', '%.1f', )))
    return notes


def load_segments(filename):
    """
    Load labelled segments from a text file. Each line in the file represents
    a segment and is of the form <start> <end> <label>, where <start> and
    <end> are floating point numbers, and <label> is a string.

    Parameters
    ----------
    filename : str or file handle
        File containing labelled segments.

    Returns
    -------
    segments : numpy structured array
        Structured array with columns "start", "end", and "label",
        containing the beginning, end, and label of segments.
    """
    start, end, label = [], [], []

    with _open_file(filename) as f:
        for line in f:
            s, e, l = line.split()
            start.append(float(s))
            end.append(float(e))
            label.append(l)

    segments = np.zeros(len(start), dtype=SEGMENT_DTYPE)
    segments['start'] = start
    segments['end'] = end
    segments['label'] = label
    return segments


def write_segments(segments, filename):
    """
    Write labelled segments to a file.

    Parameters
    ----------
    segments : numpy structured array
        Labelled segments, one per row (column definition see SEGMENT_DTYPE).
    filename : str or file handle
        Output filename or handle

    Returns
    -------
    numpy structured array
        Labelled segments

    Notes
    -----
    Labelled segments are represented as numpy structured array with three
    named columns: 'start' contains the start position (e.g. seconds),
    'end' the end position, and 'label' the segment label.

    """
    np.savetxt(filename, segments, fmt=['%.3f', '%.3f', '%s'], delimiter='\t')
    return segments


load_chords = load_segments
write_chords = write_segments


def load_tempo(values, split_value=1., sort=False, norm_strengths=False,
               max_len=None):
    """
    Load tempo information from the given values or file.

    Parameters
    ----------
    values : str, file handle, list of tuples or numpy array
        Tempo values or file name/handle.
    split_value : float, optional
        Value to distinguish between tempi and strengths.
        `values` > `split_value` are interpreted as tempi [bpm],
        `values` <= `split_value` are interpreted as strengths.
    sort : bool, optional
        Sort the tempi by their strength.
    norm_strengths : bool, optional
        Normalize the strengths to sum 1.
    max_len : int, optional
        Return at most `max_len` tempi.

    Returns
    -------
    tempi : numpy array, shape (num_tempi, 2)
        Array with tempi (rows, first column) and their relative strengths
        (second column).

    Notes
    -----
    The tempo must have the one of the following formats (separated by
    whitespace if loaded from file):

    'tempo_one' 'tempo_two' 'relative_strength' (of the first tempo)
    'tempo_one' 'tempo_two' 'strength_one' 'strength_two'

    If no strengths are given, uniformly distributed strengths are returned.

    """
    # check max_len
    if max_len is not None and max_len < 1:
        raise ValueError('`max_len` must be greater or equal to 1')
    # load the tempo from the given representation
    if isinstance(values, (list, np.ndarray)):
        # convert to numpy array if possible
        # Note: use array instead of asarray because of ndmin
        values = np.array(values, dtype=np.float, ndmin=1, copy=False)
    else:
        # try to load the data from file
        values = np.loadtxt(values, ndmin=1)
    # split the values according to their values into tempi and strengths
    # TODO: this is kind of hack-ish, find a better solution
    tempi = values[values > split_value]
    strengths = values[values <= split_value]
    # make the strengths behave properly
    strength_sum = np.sum(strengths)
    # relative strengths are given (one less than tempi)
    if len(tempi) - len(strengths) == 1:
        strengths = np.append(strengths, 1. - strength_sum)
        if np.any(strengths < 0):
            raise AssertionError('strengths must be positive')
    # no strength is given, assume an evenly distributed one
    if strength_sum == 0:
        strengths = np.ones_like(tempi) / float(len(tempi))
    # normalize the strengths
    if norm_strengths:
        strengths /= float(strength_sum)
    # tempi and strengths must have same length
    if len(tempi) != len(strengths):
        raise AssertionError('tempi and strengths must have same length')
    # order the tempi according to their strengths
    if sort:
        # Note: use 'mergesort', because we want a stable sorting algorithm
        #       which keeps the order of the keys in case of duplicate keys
        #       but we need to apply this (-strengths) trick because we want
        #       tempi with uniformly distributed strengths to keep their order
        sort_idx = (-strengths).argsort(kind='mergesort')
        tempi = tempi[sort_idx]
        strengths = strengths[sort_idx]
    # return at most 'max_len' tempi and their relative strength
    return np.vstack((tempi[:max_len], strengths[:max_len])).T


def write_tempo(tempi, filename, mirex=False):
    """
    Write the most dominant tempi and the relative strength to a file.

    Parameters
    ----------
    tempi : numpy array
        Array with the detected tempi (first column) and their strengths
        (second column).
    filename : str or file handle
        Output file.
    mirex : bool, optional
        Report the lower tempo first (as required by MIREX).

    Returns
    -------
    tempo_1 : float
        The most dominant tempo.
    tempo_2 : float
        The second most dominant tempo.
    strength : float
        Their relative strength.

    """
    # make the given tempi a 2d array
    tempi = np.array(tempi, ndmin=2)
    # default values
    t1, t2, strength = 0., 0., 1.
    # only one tempo was detected
    if len(tempi) == 1:
        t1 = tempi[0][0]
        # generate a fake second tempo
        # the boundary of 68 bpm is taken from Tzanetakis 2013 ICASSP paper
        if t1 < 68:
            t2 = t1 * 2.
        else:
            t2 = t1 / 2.
    # consider only the two strongest tempi and strengths
    elif len(tempi) > 1:
        t1, t2 = tempi[:2, 0]
        strength = tempi[0, 1] / sum(tempi[:2, 1])
    # for MIREX, the lower tempo must be given first
    if mirex and t1 > t2:
        t1, t2, strength = t2, t1, 1. - strength
    # format as a numpy array
    out = np.array([t1, t2, strength], ndmin=2)
    # write to output
    np.savetxt(filename, out, fmt='%.2f\t%.2f\t%.2f')
    # also return the tempi & strength
    return t1, t2, strength


class AlignmentFormatError(Exception):
    """
    Exception to be raised whenever an incorrect alignment format is given.

    """
    # pylint: disable=super-init-not-called

    def __init__(self, value=None):
        if value is None:
            value = 'Alignment has to be at least one row of two columns ' \
                    'representing time and score position.'
        self.value = value

    def __str__(self):
        return repr(self.value)


def load_alignment(values):
    """
    Load the alignment from given values or file.

    Parameters
    ----------
    values : str, file handle, list or numpy array
        Alignment values.

    Returns
    -------
    numpy array
        Time and score position columns.

    """
    if values is None:
        # return 'empty' alignment
        return np.array([[0, -1]])
    elif isinstance(values, (list, np.ndarray)):
        values = np.atleast_2d(values)
    else:
        values = np.loadtxt(values, ndmin=2)

    if values.shape[0] < 1 or values.shape[1] < 2 or len(values.shape) > 2:
        raise AlignmentFormatError()

    return values[:, :2]


def load_key(value):
    """
    Load the key from the given file.

    Parameters
    ----------
    value : str or file handle
        File name or file to be loaded

    Returns
    -------
    str
        Key annotation.

    """
    return _open_file(value).read().strip()