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


def files(path, extension):
    """
    Returns a list of files in path matching the given extension.

    :param path:      folder to be searched for files
    :param extension: only return files with this extension
    :returns:         list of files

    """
    import os.path
    import glob
    import fnmatch
    # determine the detection files
    if type(path) == list:
        # a list of files or paths is given
        file_list = []
        for f in path:
            file_list.extend(files(f, extension))
    elif os.path.isdir(path):
        # use all files in the given path
        file_list = glob.glob("%s/*" % path)
    elif os.path.isfile(path):
        # just use this file
        file_list = [path]
    else:
        raise ValueError("only files or folders are supported")
    # sort files
    file_list.sort()
    # filter file list
    if extension:
        file_list = fnmatch.filter(file_list, "*%s" % extension)
    # return list
    return file_list


def match_files(det_files, tar_files=None, det_ext='*', tar_ext='*'):
    """
    Match a list of target files to the corresponding detection files.

    :param det_files: list of detection files
    :param tar_files: list of target files [default=None]
    :param det_ext:  use only detection files with that extension [default='*']
    :param tar_ext:  use only target files with that extension [default='*']

    Note: if no target files are given, the same list as the detections is used.
          Handy, if the first list contains both the detections and targets.
    """
    import os.path
    # if no targets are given, use the same as the detections
    if len(tar_files) == 0:
        tar_files = [det_files]
    # determine the detection files
    det_files = files(det_files, det_ext)
    # determine the target files
    tar_files = files(tar_files, tar_ext)
    # file list to return
    file_list = []
    # find matching target files for each detection file
    for det_file in det_files:
        # strip of possible extensions
        if det_ext:
            det_file_name = os.path.splitext(det_file)[0]
        else:
            det_file_name = det_file
        # get the base name without the path
        det_file_name = os.path.basename(det_file_name)
        # look for files with the same base name in the targets
        # TODO: is there a nice one-liner to achieve the same?
        #tar_files_ = [os.path.join(p, f) for p, f in os.path.split(tar_files) if f == det_file_name]
        tar_files_ = []
        for tar_file in tar_files:
            p, f = os.path.split(tar_file)
            if f == det_file_name:
                tar_files_.append(os.path.join(p, f))
        # append a tuple of the matching pair
        file_list.append((det_file, tar_files_))
    # return
    return file_list


def load_events(filename):
    """
    Load a list of events from file.

    :param filename: name of the file
    :return:         list of events

    """
    # Note: the loop is much faster than np.loadtxt(filename, usecols=[0])
    # array for events
    events = []
    # try to read in the events from the file
    with open(filename, 'rb') as f:
        # read in each line of the file
        for line in f:
            # append the event (1st column) to the list, ignore the rest
            # TODO: make these tuples, with all the evaluation methods just
            # take the needed values (columns) an evaluate accordingly.
            events.append(float(line.split()[0]))
    # return
    return np.asarray(events)


def write_events(events, filename):
    """
    Write the detected onsets to the given file.

    :param events:   list of events [seconds]
    :param filename: output file name

    """
    with open(filename, 'w') as f:
        for e in events:
            f.write(str(e) + '\n')


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
