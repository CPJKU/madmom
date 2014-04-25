# encoding: utf-8
"""
Utility package.

"""

import os
import glob
import fnmatch
import contextlib
import __builtin__

import numpy as np


# overwrite the built-in open() to transparently apply some magic file handling
@contextlib.contextmanager
def open(filename, mode='r'):
    """
    Context manager which yields an open file or handle with the given mode
    and closes it if needed afterwards.

    :param filename: file name or open file handle
    :param mode:     mode in which to open the file
    :returns:        an open file handle

    """
    # check if we need to open the file
    if isinstance(filename, basestring):
        f = fid = __builtin__.open(filename, mode)
    else:
        f = filename
        fid = None
    # TODO: include automatic (un-)zipping here?
    # yield an open file handle
    yield f
    # close the file if needed
    if fid:
        fid.close()


def files(path, ext=None):
    """
    Returns a list of files in path matching the given extension.

    :param path: path or list of files to be filtered/searched
    :param ext:  only return files with this extension
    :returns:    list of files

    """
    # determine the detection files
    if type(path) == list:
        # a list of files or paths is given
        file_list = []
        # recursively call the function
        for f in path:
            file_list.extend(files(f, ext))
    elif os.path.isdir(path):
        # use all files in the given path
        if ext is None:
            file_list = glob.glob("%s/*" % path)
        else:
            file_list = glob.glob("%s/*%s" % (path, ext))
    elif os.path.isfile(path):
        # no matching needed
        if ext is None:
            file_list = [path]
        # file must have the correct extension
        elif path.endswith(ext):
            file_list = [path]
        # file does not match any condition
        else:
            file_list = []
    else:
        raise IOError("%s does not exist." % path)
    # sort files
    file_list.sort()
    # return list
    return file_list


def strip_suffix(filename, suffix=None):
    """
    Strip of the suffix of the given filename or string.

    :param filename: filename or string to process
    :param suffix:   suffix to be stripped off
    :returns:        filename or string without suffix

    """
    if suffix is not None and filename.endswith(suffix):
        return filename[:-len(suffix)]
    return filename


def match_file(filename, match_list, suffix=None, match_suffix=None):
    """
    Match a filename or string against a list of other filenames or strings.

    :param filename:     filename or string to be matched
    :param match_list:   match to this list of filenames or strings
    :param suffix:       ignore this suffix of the filename when matching
    :param match_suffix: only match files with this suffix
    :returns:            list of matched files

    """
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
        # base names must match exactly
        if basename == os.path.basename(strip_suffix(match, match_suffix)):
            matches.append(match)
    # return the matches
    return matches


def load_events(filename):
    """
    Load a list of events from a text file, one floating point number per line.

    :param filename: name of the file or file handle
    :return:         numpy array of events

    Note: Comments (i.e. lines tarting with '#') are ignored.

    """
    with open(filename, 'rb') as f:
        # read in the events, one per line
        # 1st column is the event's time, the rest is ignored
        return np.fromiter((float(line.split(None, 1)[0]) for line in f
                            if not line.startswith('#')), dtype=np.float)


def write_events(events, filename):
    """
    Write a list of events to a text file, one floating point number per line.

    :param events:   list of events [seconds]
    :param filename: output file name or file handle

    """
    with open(filename, 'wb') as f:
        f.writelines('%g\n' % e for e in events)


def combine_events(events, delta):
    """
    Combine all events within a certain range.

    :param events: list of events [seconds]
    :param delta:  combination length [seconds]
    :return:       list of combined events

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


def quantise_events(events, fps, length=None, shift=None):
    """
    Quantise the events with the given resolution.

    :param events: sequence of events [seconds]
    :param fps:    quantize with N frames per second
    :param length: length of the returned array [frames]
    :param shift:  shift the events by N seconds before quantisation
    :returns:      a quantized numpy array

    """
    # shift all events if needed
    if shift:
        events = np.asarray(events) + shift
    # determine the length for the quantised array
    if length is None:
        # set the length to be long enough to cover all events
        length = int(round(np.max(events) * float(fps))) + 1
    else:
        # else filter all events which do not fit in the array
        # since we apply rounding later, we need to subtract half a bin
        events = events[:np.searchsorted(events, float(length - 0.5) / fps)]
    # init array
    quantized = np.zeros(length)
    # set the events
    for event in events:
        idx = int(round(event * float(fps)))
        quantized[idx] = 1
    # return the quantized array
    return quantized
