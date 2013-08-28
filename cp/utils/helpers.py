#!/usr/bin/env python
# encoding: utf-8
"""
Copyright (c) 2013 Sebastian Böck <sebastian.boeck@jku.at>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

import numpy as np


def files(path, ext=None):
    """
    Returns a list of files in path matching the given extension.

    :param path: path or list of files to be filtered / searched
    :param ext:  only return files with this extension
    :returns:    list of files

    """
    import os.path
    import glob
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
        # just use this file
        if ext is None:
            file_list = [path]
        elif path.endswith(ext):
            file_list = [path]
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

    :param filename:     file to be matched
    :param match_list:   match to this list of files

    """
    import os.path
    import fnmatch
    # get the base name without the path
    basename = os.path.basename(stripext(filename, ext))
    # look for files with the same base name in the files_list
    matches = fnmatch.filter(match_list, "*%s*%s" % (basename, match_ext))
    exact_matches = []
    # base names must match exactly
    for match in matches:
        if basename == os.path.basename(stripext(match, match_ext)):
            exact_matches.append(match)
    # depending on the number of matches
    if len(exact_matches) == 0:
        # return None
        exact_matches = None
    elif len(exact_matches) == 1:
        # return a single entry
        exact_matches = exact_matches[0]
    # return a list
    return exact_matches


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
        quantized[idx] = 1
    # return the events
    return quantized
