#!/usr/bin/env python
# encoding: utf-8
"""
This file contains various helper functions used by all other modules.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import os.path
import glob
import fnmatch

import numpy as np


def files(path, ext=None):
    """
    Returns a list of files in path matching the given extension.

    :param path: path or list of files to be filtered / searched
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
        # no matchin needed
        if ext is None:
            file_list = [path]
        # file must have the correct extension
        elif path.endswith(ext):
            file_list = [path]
        # file does not match any condition
        else:
            file_list = []
    else:
        raise ValueError("only files or folders are supported.")
    # sort files
    file_list.sort()
    # return list
    return file_list


def stripext(filename, ext=None):
    """Strip of the extension."""
    if ext is not None and filename.endswith(ext):
        return filename[:-len(ext)]
    return filename


def match_file(filename, match_list, ext=None, match_ext=None):
    """
    Match a file against a list of other files.

    :param filename:   file to be matched
    :param match_list: match to this list of files (or folders)
    :param ext:        strip this extension from file for name matching
    :param match_ext:  strip this extension from file for name matching
    :returns:          list of matched files

    """
    # get the base name without the path (this part must match later)
    basename = os.path.basename(stripext(filename, ext))
    # init return list
    matches = []
    # look for files with the same base name in the match_list
    for match in match_list:
        # TODO: remove duplicate code with files()
        # if we have a path, take all files in there
        if os.path.isdir(match):
            if ext is None:
                matches.extend(glob.glob("%s/%s*" % (match, basename)))
            else:
                matches.extend(glob.glob("%s/%s*.%s" % (match, basename, match_ext)))
        elif os.path.isfile(match):
            # just use this file if the name matches
            if os.path.basename(stripext(match, match_ext)) == basename:
                matches.append(match)
    # return the matches
    return matches


def load_events(filename):
    """
    Load a list of events from file.

    :param filename: name of the file or file handle
    :return:         list of events

    """
    if not isinstance(filename, file):
        # open the file if necessary
        filename = open(filename, 'r')
    with filename:
        # Note: the loop is much faster than np.loadtxt(filename, usecols=[0])
        events = []
        # read in the events, one per line
        for line in filename:
            # 1st column is the events time, ignore the rest if present
            events.append(float(line.split()[0]))
    # return
    return np.asarray(events)


def write_events(events, filename):
    """
    Write the detected onsets to the given file.

    :param events:   list of events [seconds]
    :param filename: output file name or file handle

    """
    was_closed = False
    if not isinstance(filename, file):
        # open the file if necessary
        filename = open(filename, 'w')
        was_closed = True
    # FIXME: if we use "with filename:" here, the file handle gets closed
    # after write and the called object is not accessible afterwards. Is this
    # expected? Is this the right way to circumvent this?
    for e in events:
        filename.write(str(e) + '\n')
    # close the file again?
    if was_closed:
        filename.close()


def combine_events(events, delta):
    """
    Combine all events within a certain range.

    :param events: list of events [seconds]
    :param delta:  combination length [seconds]
    :return:       list of combined events

    """
    # return array if no events must be combined
    diff = np.diff(events)
    if not (events[1:][diff <= delta].any()):
        return events
    # array for combined events
    comb = []
    # copy the events, because the array is modified later
    events = np.copy(events)
    # iterate over all events
    idx = 0
    while idx < events.size - 1:
        # get the first event
        first = events[idx]
        # increase the events index
        idx += 1
        # get the second event
        second = events[idx]
        # combine the two events?
        if second - first <= delta:
            # two events within the combination window, combine them and replace
            # the second event in the original array with the mean of the events
            events[idx] = (first + second) / 2.
        else:
            # the two events can not be combined,
            # store the first event in the new list
            comb.append(first)
    # always append the last element of the list
    comb.append(events[-1])
    # return the combined events
    return np.asarray(comb)


def filter_events(events, key):
    raise NotImplemented


def quantize_events(events, fps, length=None):
    """
    Quantize the events.

    :param events: sequence of events [seconds]
    :param fps:    quantize with N frames per second
    :param length: length of the returned array [frames, default=last event]
    :returns:      a quantized numpy array

    """
    # length of the array
    if length is None:
        length = int(round(events[-1] * fps)) + 1
    # init array
    quantized = np.zeros(length)
    # set the events
    for event in events:
        idx = int(round(event * float(fps)))
        try:
            quantized[idx] = 1
        except IndexError:
            pass
    # return the events
    return quantized


def combine_activations(in_dirs, out_dir, ext='.activations', sep=''):
    """
    Quantize the activations of the given dirs.

    :param in_dirs: list of directories or files with activations
    :param out_dir: output directory
    :param sep:     separator between activation values [default='']

    Note: The output directory must exist, existing files are overwritten.

          Empty (“”) separator means the file should be treated as binary;
          spaces (” ”) in the separator match zero or more whitespace;
          separator consisting only of spaces must match at least one whitespace.

          If out_dir is set and multiple network files contain the same
          files, the activations get averaged.

    """
    # get a list of activation files
    file_list = []
    for in_dir in in_dirs:
        file_list.extend(files(in_dir, ext))

    # get the base names of all files
    base_names = [os.path.basename(f) for f in file_list]
    # keep only unique names
    base_names = list(set(base_names))

    # combine all activations with the same base name
    for base_name in base_names:
        # get a list of all file matches
        matched_files = match_file(base_name, file_list)
        # init activations
        activations = None
        for matched_file in matched_files:
            if activations is None:
                activations = np.fromfile(matched_file, sep=sep)
            else:
                activations += np.fromfile(matched_file, sep=sep)
        # average activations
        if len(matched_files) > 1:
            activations /= len(matched_files)
        # output file
        if activations is not None:
            out_file = "%s/%s" % (out_dir, base_name)
            activations.tofile(out_file, sep)
