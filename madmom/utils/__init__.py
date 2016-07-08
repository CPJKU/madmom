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
    # needed to preserve docstring of the decorated function
    from functools import wraps

    @wraps(function)
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
def filter_files(files, suffix):
    """
    Filter the list to contain only files matching the given `suffix`.

    Parameters
    ----------
    files : list
        List of files to be filtered.
    suffix : str
        Return only files matching this suffix.

    Returns
    -------
    list
        List of files.

    """
    import fnmatch
    # make sure files is a list
    if not isinstance(files, list):
        files = [files]
    # no suffix given, return the list unaltered
    if suffix is None:
        return files
    # filter the files with the given suffix
    file_list = []
    if isinstance(suffix, list):
        # a list of suffices is given
        for s in suffix:
            file_list.extend(fnmatch.filter(files, "*%s" % s))
    else:
        # a single suffix is given
        file_list.extend(fnmatch.filter(files, "*%s" % suffix))
    # return the filtered list
    return file_list


def search_path(path, recursion_depth=0):
    """
    Returns a list of files in a directory (recursively).

    Parameters
    ----------
    path : str or list
        Directory to be searched.
    recursion_depth : int, optional
        Recursively search sub-directories up to this depth.

    Returns
    -------
    list
        List of files.

    """
    # adapted from http://stackoverflow.com/a/234329
    import os
    # remove the rightmost path separator (needed for recursion depth count)
    path = path.rstrip(os.path.sep)
    # we can only handle directories
    if not os.path.isdir(path):
        raise IOError("%s is not a directory." % path)
    # files to be returned
    file_list = []
    # keep track of the initial recursion depth
    initial_depth = path.count(os.path.sep)
    for root, dirs, files in os.walk(path):
        # add all files of this directory to the list
        for f in files:
            file_list.append(os.path.join(root, f))
        # remove all subdirs exceeding the wanted recursion depth
        if initial_depth + recursion_depth <= root.count(os.path.sep):
            del dirs[:]
    # return the sorted file list
    return sorted(file_list)


def search_files(files, suffix=None, recursion_depth=0):
    """
    Returns the files matching the given `suffix`.

    Parameters
    ----------
    files : str or list
        File, path or a list thereof to be searched / filtered.
    suffix : str, optional
        Return only files matching this suffix.
    recursion_depth : int, optional
        Recursively search sub-directories up to this depth.

    Returns
    -------
    list
        List of files.

    Notes
    -----
    The list of returned files is sorted.

    """
    import os
    file_list = []
    # determine the files
    if isinstance(files, list):
        # a list is given, recursively call the function on each element
        for f in files:
            file_list.extend(search_files(f))
    elif os.path.isdir(files):
        # add all files in the given path (up to the given recursion depth)
        file_list.extend(search_path(files, recursion_depth))
    elif os.path.isfile(files):
        # add the given file
        file_list.append(files)
    else:
        raise IOError("%s does not exist." % files)
    # filter with the given sufix
    if suffix is not None:
        file_list = filter_files(file_list, suffix)
    # remove duplicates
    file_list = list(set(file_list))
    # return the sorted file list
    return sorted(file_list)


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


def match_file(filename, match_list, suffix=None, match_suffix=None,
               match_exactly=True):
    """
    Match a filename or string against a list of other filenames or strings.

    Parameters
    ----------
    filename : str
        Filename or string to match.
    match_list : list
        Match to this list of filenames or strings.
    suffix : str, optional
        Suffix of `filename` to be ignored.
    match_suffix : str, optional
        Match only files from `match_list` with this suffix.
    match_exactly : bool, optional
        Matches must be exact, i.e. have the same base name.

    Returns
    -------
    list
        List of matched files.

    Notes
    -----
    Asterisks "*" can be used to match any string or suffix.

    """
    import os
    import fnmatch

    # get the base name without the path
    basename = os.path.basename(strip_suffix(filename, suffix))
    # init return list
    matches = []
    # look for files with the same base name in the files_list
    if match_suffix is not None:
        pattern = "*%s*%s" % (basename, match_suffix)
    else:
        pattern = "*%s" % basename
    for match in fnmatch.filter(match_list, pattern):
        # base names must match exactly if indicated
        if (not match_exactly) or (basename == os.path.basename(
                strip_suffix(match, match_suffix))):
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


def write_events(events, filename, fmt='%.3f', delimiter='\t', header=''):
    """
    Write events to a text file, one event per line.

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
    arguments in a way it can be used as an
    :class:`.processors.OutputProcessor`.

    """
    # write the events to the output
    np.savetxt(filename, np.asarray(events),
               fmt=fmt, delimiter=delimiter, header=header)
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
            raise argparse.ArgumentError(self, str(e) + value)


# taken from: http://www.scipy.org/Cookbook/SegmentAxis
def segment_axis(signal, frame_size, hop_size, axis=None, end='cut',
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
    hop_size : int
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
del contextlib

# finally import the submodules
from . import midi, stats
