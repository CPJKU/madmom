# encoding: utf-8
"""
Input/output package.

"""

from __future__ import absolute_import, division, print_function

import sys

import numpy as np

from .audio import load_audio_file
from .midi import load_midi, write_midi
from ..utils import suppress_warnings

if sys.version_info[0] < 3:
    _file_handle = file
else:
    from io import TextIOBase as _file_handle


# dtype for numpy structured arrays that contain labelled segments
# 'label' needs to be castable to str
SEGMENT_DTYPE = [('start', np.float), ('end', np.float), ('label', object)]


def _open_file(fn):
    return fn if isinstance(fn, _file_handle) else open(fn, 'r')


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
    fmt : str or sequence of strs, optional
        A single format (e.g. '%.3f'), a sequence of formats, or a multi-format
        string (e.g. '%.3f %.3f'), in which case `delimiter` is ignored.
    delimiter : str, optional
        String or character separating columns.
    header : str, optional
        String that will be written at the beginning of the file as comment.

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
    np.savetxt(filename, np.asarray(events), fmt=fmt, delimiter=delimiter,
               header=header)
    # also return them
    return events


load_onsets = load_events
write_onsets = write_events


@suppress_warnings
def load_beats(filename, downbeats=False):
    """
    Load the beats from the given file, one beat per line of format
    'beat_time' ['beat_number'].

    Parameters
    ----------
    filename : str or file handle
        File to load the beats from.
    downbeats : bool, optional
        Load only downbeats instead of beats.

    Returns
    -------
    numpy array
        Beats.

    """
    values = np.loadtxt(filename, ndmin=1)
    if values.ndim > 1:
        if downbeats:
            # rows with a "1" in the 2nd column are downbeats
            return values[values[:, 1] == 1][:, 0]
        else:
            # 1st column is the beat time, the rest is ignored
            return values[:, 0]
    return values


def write_beats(beats, filename, fmt=None, delimiter='\t', header=''):
    """
    Write the beats to a file.

    Parameters
    ----------
    beats : numpy array
        Beats to be written to file.
    filename : str or file handle
        File to write the beats to.
    fmt : str or sequence of strs, optional
        A single format (e.g. '%.3f'), a sequence of formats (e.g.
        ['%.3f', '%d']), or a multi-format string (e.g. '%.3f %d'), in which
        case `delimiter` is ignored.
    delimiter : str, optional
        String or character separating columns.
    header : str, optional
        String that will be written at the beginning of the file as comment.

    """
    if fmt is None and beats.ndim == 2:
        fmt = ['%.3f', '%d']
    elif fmt is None:
        fmt = '%.3f'
    write_events(beats, filename, fmt=fmt, delimiter=delimiter, header=header)


@suppress_warnings
def load_notes(filename):
    """
    Load the notes from the given file, one note per line of format
    'onset_time' 'note_number' ['duration' ['velocity']].

    Parameters
    ----------
    filename: str or file handle
        File to load the notes from.

    Returns
    -------
    numpy array
        Notes.

    """
    return np.loadtxt(filename, ndmin=2)


def write_notes(notes, filename, fmt=None, delimiter='\t', header=''):
    """
    Write the notes to a file.

    Parameters
    ----------
    notes : numpy array, shape (num_notes, 2)
        Notes, row format 'onset_time' 'note_number' ['duration' ['velocity']].
    filename : str or file handle
        File to write the notes to.
    fmt : str or sequence of strs, optional
        A sequence of formats (e.g. ['%.3f', '%d', '%.3f', '%d']), or a
        multi-format string, e.g. '%.3f %d %.3f %d', in which case `delimiter`
        is ignored.
    delimiter : str, optional
        String or character separating columns.
    header : str, optional
        String that will be written at the beginning of the file as comment.

    Returns
    -------
    numpy array
        Notes.

    """
    # set default format
    if fmt is None:
        fmt = ['%.3f', '%d', '%.3f', '%d']
    if not notes.ndim == 2:
        raise ValueError('unknown format for `notes`')
    # truncate format to the number of colums given
    fmt = delimiter.join(fmt[:notes.shape[1]])
    # write the notes
    write_events(notes, filename, fmt=fmt, delimiter=delimiter, header=header)
    # also return them
    return notes


def load_segments(filename):
    """
    Load labelled segments from file, one segment per line. Each segment is of
    form <start> <end> <label>, where <start> and <end> are floating point
    numbers, and <label> is a string.

    Parameters
    ----------
    filename : str or file handle
        File to read the labelled segments from.

    Returns
    -------
    segments : numpy structured array
        Structured array with columns 'start', 'end', and 'label',
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


def write_segments(segments, filename, fmt=None, delimiter='\t', header=''):
    """
    Write labelled segments to a file.

    Parameters
    ----------
    segments : numpy structured array
        Labelled segments, one per row (column definition see SEGMENT_DTYPE).
    filename : str or file handle
        Output filename or handle.
    fmt : str or sequence of strs, optional
        A sequence of formats (e.g. ['%.3f', '%.3f', '%s']), or a multi-format
        string (e.g. '%.3f %.3f %s'), in which case `delimiter` is ignored.
    delimiter : str, optional
        String or character separating columns.
    header : str, optional
        String that will be written at the beginning of the file as comment.

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
    if fmt is None:
        fmt = ['%.3f', '%.3f', '%s']
    np.savetxt(filename, segments, fmt=fmt, delimiter=delimiter, header=header)
    return segments


load_chords = load_segments
write_chords = write_segments


def load_key(filename):
    """
    Load the key from the given file.

    Parameters
    ----------
    filename : str or file handle
        File to read key information from.

    Returns
    -------
    str
        Key.

    """
    return _open_file(filename).read().strip()


def write_key(key, filename):
    """
    Write key string to a file.

    Parameters
    ----------
    key : str
        Key name.
    filename : str or file handle
        Output file.

    Returns
    -------
    key : str
        Key name.

    """
    np.savetxt(filename, [key], fmt='%s')
    return key


def load_tempo(filename, split_value=1., sort=None, norm_strengths=None,
               max_len=None):
    """
    Load tempo information from the given file.

    Tempo information must have the following format:
    'main tempo' ['secondary tempo' ['relative_strength']]

    Parameters
    ----------
    filename : str or file handle
        File to load the tempo from.
    split_value : float, optional
        Value to distinguish between tempi and strengths.
        `values` > `split_value` are interpreted as tempi [bpm],
        `values` <= `split_value` are interpreted as strengths.

    Returns
    -------
    tempi : numpy array, shape (num_tempi[, 2])
        Array with tempi. If no strength is parsed, a 1-dimensional array of
        length 'num_tempi' is returned. If strengths are given, a 2D array
        with tempi (first column) and their relative strengths (second column)
        is returned.


    """
    # try to load the data from file
    values = np.loadtxt(filename, ndmin=1)
    # split the filename according to their filename into tempi and strengths
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
    if norm_strengths is not None:
        import warnings
        warnings.warn('`norm_strengths` is deprecated as of version 0.16 and '
                      'will be removed in 0.18. Please normalize strengths '
                      'separately.')
        strengths /= float(strength_sum)
    # tempi and strengths must have same length
    if len(tempi) != len(strengths):
        raise AssertionError('tempi and strengths must have same length')
    # order the tempi according to their strengths
    if sort:
        import warnings
        warnings.warn('`sort` is deprecated as of version 0.16 and will be '
                      'removed in 0.18. Please sort the returned array '
                      'separately.')
        # Note: use 'mergesort', because we want a stable sorting algorithm
        #       which keeps the order of the keys in case of duplicate keys
        #       but we need to apply this '(-strengths)' trick because we want
        #       tempi with uniformly distributed strengths to keep their order
        sort_idx = (-strengths).argsort(kind='mergesort')
        tempi = tempi[sort_idx]
        strengths = strengths[sort_idx]
    # return at most 'max_len' tempi and their relative strength
    if max_len is not None:
        import warnings
        warnings.warn('`max_len` is deprecated as of version 0.16 and will be '
                      'removed in 0.18. Please truncate the returned array '
                      'separately.')
    return np.vstack((tempi[:max_len], strengths[:max_len])).T


def write_tempo(tempi, filename, delimiter='\t', mirex=False):
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
    np.savetxt(filename, out, fmt=['%.2f', '%.2f', '%.2f'],
               delimiter=delimiter)
    # also return the tempi & strength
    return t1, t2, strength
