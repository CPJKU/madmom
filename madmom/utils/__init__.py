# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=wrong-import-position
"""
Utility package.

"""

from __future__ import absolute_import, division, print_function

import argparse
import contextlib
import numpy as np


# decorator to suppress warnings
def suppress_warnings(function):
    """
    Decorate the given function to suppress any warnings.

    Parameters
    ----------
    function : function
        Function to be decorated.

    Returns
    -------
    decorated function
        Decorated function.

    """

    def decorator_function(*args, **kwargs):
        """
        Decorator function to suppress warnings.

        Parameters
        ----------
        args : arguments, optional
            Arguments passed to function to be decorated.
        kwargs : keyword arguments, optional
            Keyword arguments passed to function to be decorated.

        Returns
        -------
        decorated function
            Decorated function.

        """
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return function(*args, **kwargs)

    return decorator_function


# file handling routines
def search_files(path, suffix=None):
    """
    Returns a list of files in `path` matching the given `suffix` or filters
    the given list to include only those matching the given suffix.

    Parameters
    ----------
    path : str or list
        Path or list of files to be searched / filtered.
    suffix : str, optional
        Return only files matching this suffix.

    Returns
    -------
    list
        List of files.

    """
    import os
    import glob

    # determine the files
    if isinstance(path, list):
        # a list of files or paths is given
        file_list = []
        # recursively call the function
        for f in path:
            file_list.extend(search_files(f, suffix))
    elif os.path.isdir(path):
        # use all files in the given path
        if suffix is None:
            file_list = glob.glob("{0!s}/*".format(path))
        elif isinstance(suffix, list):
            file_list = []
            for s in suffix:
                file_list.extend(glob.glob("{0!s}/*{1!s}".format(path, s)))
        else:
            file_list = glob.glob("{0!s}/*{1!s}".format(path, suffix))
    elif os.path.isfile(path):
        file_list = []
        # no matching needed
        if suffix is None:
            file_list = [path]
        # a list of suffices is given
        elif isinstance(suffix, list):
            for s in suffix:
                if path.endswith(s):
                    file_list = [path]
        # a single suffix is given
        elif path.endswith(suffix):
            file_list = [path]
    else:
        raise IOError("{0!s} does not exist.".format(path))
    # remove duplicates
    file_list = list(set(file_list))
    # sort files
    file_list.sort()
    # return the file list
    return file_list


def strip_suffix(filename, suffix=None):
    """
    Strip off the suffix of the given filename or string.

    Parameters
    ----------
    filename : str
        Filename or string to strip.
    suffix : str, optional
        Suffix to be stripped off (e.g. '.txt' including the dot).
    Returns
    -------
    str
        Filename or string without suffix.

    """
    if suffix is not None and filename.endswith(suffix):
        return filename[:-len(suffix)]
    return filename


def match_file(filename, match_list, suffix=None, match_suffix=None):
    """
    Match a filename or string against a list of other filenames or strings.

    Parameters
    ----------
    filename : str
        Filename or string to strip.
    match_list : list
        Match to this list of filenames or strings.
    suffix : str, optional
        Suffix of `filename` to be ignored.
    match_suffix
        Match only files from `match_list` with this suffix.

    Returns
    -------
    list
        List of matched files.

    """
    import os
    import fnmatch

    # get the base name without the path
    basename = os.path.basename(strip_suffix(filename, suffix))
    # init return list
    matches = []
    # look for files with the same base name in the files_list
    if match_suffix is not None:
        pattern = "*{0!s}*{1!s}".format(basename, match_suffix)
    else:
        pattern = "*{0!s}".format(basename)
    for match in fnmatch.filter(match_list, pattern):
        # base names must match exactly
        if basename == os.path.basename(strip_suffix(match, match_suffix)):
            matches.append(match)
    # return the matches
    return matches


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


def write_events(events, filename, fmt='%.3f', header=''):
    """
    Write events to a text file, one floating point number per line.

    Parameters
    ----------
    events : numpy array
        Events.
    filename : str or file handle
        File to write the events to.
    fmt : str, optional
        How to format the events.
    header : str, optional
        Header to be written (as a comment).

    Returns
    -------
    numpy array
        Events.

    Notes
    -----
    This function is just a wrapper to ``np.savetxt``, but reorders the
    arguments in a way it can be used as an :class:`OutputProcessor`.

    """
    # write the events to the output
    np.savetxt(filename, np.asarray(events), fmt=fmt, header=header)
    # also return them
    return events


def combine_events(events, delta):
    """
    Combine all events within a certain range.

    Parameters
    ----------
    events : list or numpy array
        Events to be combined.
    delta : float
        Combination delta. All events within this `delta` are combined, i.e.
        replaced by the mean of the two events.

    Returns
    -------
    numpy array
        Combined events.

    """
    # add a small value to delta, otherwise we end up in floating point hell
    delta += 1e-12
    # return immediately if possible
    if len(events) <= 1:
        return events
    # create working copy
    events = np.array(events, copy=True)
    # set start position
    idx = 0
    # get first event
    left = events[idx]
    # iterate over all remaining events
    for right in events[1:]:
        if right - left <= delta:
            # combine the two events
            left = events[idx] = 0.5 * (right + left)
        else:
            # move forward
            idx += 1
            left = events[idx] = right
    # return the combined events
    return events[:idx + 1]


def quantize_events(events, fps, length=None, shift=None):
    """
    Quantize the events with the given resolution.

    Parameters
    ----------
    events : numpy array
        Events to be quantized.
    fps : float
        Quantize with `fps` frames per second.
    length : int, optional
        Length of the returned array.
    shift : float, optional
        Shift the events by this value before quantisation

    Returns
    -------
    numpy array
        Quantized events.

    """
    # convert to numpy array if needed
    events = np.asarray(events, dtype=np.float)
    # shift all events if needed
    if shift:
        events += shift
    # determine the length for the quantized array
    if length is None:
        # set the length to be long enough to cover all events
        length = int(round(np.max(events) * float(fps))) + 1
    else:
        # else filter all events which do not fit in the array
        # since we apply rounding later, we need to subtract half a bin
        events = events[:np.searchsorted(events, float(length - 0.5) / fps)]
    # init array
    quantized = np.zeros(length)
    # quantize
    events *= fps
    # indices to be set in the quantized array
    idx = np.unique(np.round(events).astype(np.int))
    quantized[idx] = 1
    # return the quantized array
    return quantized


# argparse action to set and overwrite default lists
class OverrideDefaultListAction(argparse.Action):
    """
    OverrideDefaultListAction

    An argparse action that works similarly to the regular 'append' action.
    The default value is deleted when a new value is specified. The 'append'
    action would append the new value to the default.

    Parameters
    ----------
    sep : str, optional
        Separator to be used if multiple values should be parsed from a list.

    """
    def __init__(self, sep=None, *args, **kwargs):
        super(OverrideDefaultListAction, self).__init__(*args, **kwargs)
        self.set_to_default = True
        # save the type as the type for the list
        self.list_type = self.type
        if sep is not None:
            # if multiple values (separated by the given separator) should be
            # parsed we need to fake the type of the argument to be a string
            self.type = str
        self.sep = sep

    def __call__(self, parser, namespace, value, option_string=None):
        # if this Action is called for the first time, remove the defaults
        if self.set_to_default:
            setattr(namespace, self.dest, [])
            self.set_to_default = False
        # get the current values
        cur_values = getattr(namespace, self.dest)
        # convert to correct type and append the newly parsed values
        try:
            cur_values.extend([self.list_type(v)
                               for v in value.split(self.sep)])
        except ValueError as e:
            raise argparse.ArgumentError(self, e)


# taken from: http://www.scipy.org/Cookbook/SegmentAxis
def segment_axis(signal, frame_size, hop_size=1, axis=None, end='cut',
                 end_value=0):
    """
    Generate a new array that chops the given array along the given axis into
    (overlapping) frames.

    Parameters
    ----------
    signal : numpy array
        Signal.
    frame_size : int
        Size of each frame [samples].
    hop_size : int, optional
        Hop size between adjacent frames [samples].
    axis : int, optional
        Axis to operate on; if 'None', operate on the flattened array.
    end : {'cut', 'wrap', 'pad'}, optional
        What to do with the last frame, if the array is not evenly divisible
        into pieces; possible values:

        - 'cut'
          simply discard the extra values,
        - 'wrap'
          copy values from the beginning of the array,
        - 'pad'
          pad with a constant value.

    end_value : float, optional
        Value used to pad if `end` is 'pad'.

    Returns
    -------
    numpy array, shape (num_frames, frame_size)
        Array with overlapping frames

    Notes
    -----
    The array is not copied unless necessary (either because it is unevenly
    strided and being flattened or because end is set to 'pad' or 'wrap').

    The returned array is always of type np.ndarray.

    Examples
    --------
    >>> segment_axis(np.arange(10), 4, 2)
    array([[0, 1, 2, 3],
           [2, 3, 4, 5],
           [4, 5, 6, 7],
           [6, 7, 8, 9]])

    """
    # make sure that both frame_size and hop_size are integers
    frame_size = int(frame_size)
    hop_size = int(hop_size)
    # TODO: add comments!
    if axis is None:
        signal = np.ravel(signal)  # may copy
        axis = 0
    if axis != 0:
        raise ValueError('please check if the resulting array is correct.')

    length = signal.shape[axis]

    if hop_size <= 0:
        raise ValueError("hop_size must be positive.")
    if frame_size <= 0:
        raise ValueError("frame_size must be positive.")

    if length < frame_size or (length - frame_size) % hop_size:
        if length > frame_size:
            round_up = (frame_size + (1 + (length - frame_size) // hop_size) *
                        hop_size)
            round_down = (frame_size + ((length - frame_size) // hop_size) *
                          hop_size)
        else:
            round_up = frame_size
            round_down = 0
        assert round_down < length < round_up
        assert round_up == round_down + hop_size or (round_up == frame_size and
                                                     round_down == 0)
        signal = signal.swapaxes(-1, axis)

        if end == 'cut':
            signal = signal[..., :round_down]
        elif end in ['pad', 'wrap']:
            # need to copy
            s = list(signal.shape)
            s[-1] = round_up
            y = np.empty(s, dtype=signal.dtype)
            y[..., :length] = signal
            if end == 'pad':
                y[..., length:] = end_value
            elif end == 'wrap':
                y[..., length:] = signal[..., :round_up - length]
            signal = y

        signal = signal.swapaxes(-1, axis)

    length = signal.shape[axis]
    if length == 0:
        raise ValueError("Not enough data points to segment array in 'cut' "
                         "mode; try end='pad' or end='wrap'")
    assert length >= frame_size
    assert (length - frame_size) % hop_size == 0
    n = 1 + (length - frame_size) // hop_size
    s = signal.strides[axis]
    new_shape = (signal.shape[:axis] + (n, frame_size) +
                 signal.shape[axis + 1:])
    new_strides = (signal.strides[:axis] + (hop_size * s, s) +
                   signal.strides[axis + 1:])

    try:
        # noinspection PyArgumentList
        return np.ndarray.__new__(np.ndarray, strides=new_strides,
                                  shape=new_shape, buffer=signal,
                                  dtype=signal.dtype)
    except TypeError:
        # TODO: remove warning?
        import warnings
        warnings.warn("Problem with ndarray creation forces copy.")
        signal = signal.copy()
        # shape doesn't change but strides does
        new_strides = (signal.strides[:axis] + (hop_size * s, s) +
                       signal.strides[axis + 1:])
        # noinspection PyArgumentList
        return np.ndarray.__new__(np.ndarray, strides=new_strides,
                                  shape=new_shape, buffer=signal,
                                  dtype=signal.dtype)


# keep namespace clean
del argparse, contextlib

# finally import the submodules
from . import midi, stats
