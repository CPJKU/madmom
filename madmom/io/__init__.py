# encoding: utf-8
"""
Input/output package.

"""

from __future__ import absolute_import, division, print_function

import numpy as np

from .audio import load_audio_file
from ..utils import suppress_warnings


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


@suppress_warnings
def load_onsets(values):
    """
    Load the onsets from the given values or file.

    Parameters
    ----------
    values: str, file handle, list of tuples or numpy array
        Onsets values.

    Returns
    -------
    numpy array, shape (num_onsets,)
        Onsets.

    Notes
    -----
    Expected format:

    'onset_time' [additional information will be ignored]

    """
    # load the onsets from the given representation
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
    # 1st column is the onset time, the rest is ignored
    if values.ndim > 1:
        return values[:, 0]
    return values


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
