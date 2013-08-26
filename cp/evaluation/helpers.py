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
# helper functions to read/write events from files to sequences (lists)
# and do some stuff with those sequences

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
    # TODO: numpyfy
    """
    Combine all events within a certain range.

    :param events: list of events [seconds]
    :param delta:  combination length [seconds]
    :return:       list of combined events

    """
    # return array if no events must be combined
    errors = calc_intervals(events)
    if not (events[errors <= delta].any()):
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


def find_closest_matches(detections, targets):
    """
    Find the closest matches for detections in targets.

    :param detections: sequence of events to be matched [seconds]
    :param targets:    sequence of possible matches [seconds]
    :returns:          a numpy array of indices with closest matches

    Note: the sequences must be ordered!

    """
    # if no targets are given
    if len(targets) == 0:
        # return a empty array
        return np.zeros(0, dtype=np.int)
        # FIXME: raise an error instead?
        #raise ValueError("at least one target must be given")
    # if only a single target is given
    if len(targets) == 1:
        # return an array as long as the detections with indices 0
        return np.zeros(len(detections), dtype=np.int)
    # solution found at: http://stackoverflow.com/questions/8914491/finding-the-nearest-value-and-return-the-index-of-array-in-python
    indices = targets.searchsorted(detections)
    indices = np.clip(indices, 1, len(targets) - 1)
    left = targets[indices - 1]
    right = targets[indices]
    indices -= detections - left < right - detections
    # return the indices of the closest matches
    return indices


def find_closest_intervals(detections, targets, matches=None):
    """
    Find the closest target interval surrounding the detections.

    :param detections: sequence of events to be matched [seconds]
    :param targets:    sequence of possible matches [seconds]
    :param matches:    indices of the closest matches [default=None]
    :returns:          a list of closest target intervals [seconds]

    Note: the sequences must be ordered! To speed up the calculation, a list of
          pre-computed indices of the closest matches can be used.

    """
    # init array
    closest_interval = np.ones_like(detections)
    # init array for intervals
    # Note: if we combine the formward and backward intervals this is faster,
    # but we need expand the size accordingly
    intervals = np.zeros(len(targets) + 1)
    # intervals to previous target
    intervals[1:-1] = np.diff(targets)
    # the interval from the first target to the left is the same as to the right
    intervals[0] = intervals[1]
    # the interval from the last target to the right is the same as to the left
    intervals[-1] = intervals[-2]
    # Note: intervals to the next target are always those at the next index
    # determine the closest targets
    if matches is None:
        matches = find_closest_matches(detections, targets)
    # calculate the absolute errors
    errors = calc_errors(detections, targets, matches)
    # if the errors are positive, the detection is after the target
    # thus, the needed interval is from the closest target towards the next one
    closest_interval[errors > 0] = intervals[matches[errors > 0] + 1]
    # if before, interval to previous target accordingly
    closest_interval[errors < 0] = intervals[matches[errors < 0]]
    # return the closest interval
    return closest_interval


def calc_errors(detections, targets, matches=None):
    """
    Calculates the errors of the detections relative to the closest targets.

    :param detections: sequence of events to be matched [seconds]
    :param targets:    sequence of possible matches [seconds]
    :param matches:    indices of the closest matches [default=None]
    :returns:          a list of errors to closest matches [seconds]

    Note: the sequences must be ordered! To speed up the calculation, a list of
          pre-computed indices of the closest matches can be used.

    """
    # determine the closest targets
    if matches is None:
        matches = find_closest_matches(detections, targets)
    # calc error relative to those targets
    errors = detections - targets[matches]
    # return the errors
    return errors


def calc_intervals(events, fwd=False):
    """
    Calculate the intervals of all events to the previous / next event.

    :param events: sequence of events to be matched [seconds]
    :param fwd:    calculate the intervals to the next event [default=False]
    :returns:      the intervals [seconds]

    Note: the sequences must be ordered!

    """
    interval = np.zeros_like(events)
    if fwd:
        interval[:-1] = np.diff(events)
        # the interval of the first event is the same as the one of the second event
        interval[-1] = interval[-2]
    else:
        interval[1:] = np.diff(events)
        # the interval of the first event is the same as the one of the second event
        interval[0] = interval[1]
    # return
    return interval


def calc_absolute_errors(detections, targets, matches=None):
    """
    Calculate absolute errors of the detections relative to the closest targets.

    :param detections: sequence of events to be matched [seconds]
    :param targets:    sequence of possible matches [seconds]
    :param matches:    indices of the closest matches [default=None]
    :returns:          a list of errors to closest matches [seconds]

    Note: the sequences must be ordered! To speed up the calculation, a list of
          pre-computed indices of the closest matches can be used.

    """
    # return the errors
    return np.abs(calc_errors(detections, targets, matches))


def calc_relative_errors(detections, targets, matches=None):
    """
    Relative errors of the detections to the closest targets.
    The absolute error is weighted by the interval of two targets surrounding
    each detection.

    :param detections: sequence of events to be matched [seconds]
    :param targets:    sequence of possible matches [seconds]
    :param matches:    indices of the closest matches [default=None]
    :returns:          a list of relative errors to closest matches [seconds]

    Note: the sequences must be ordered! To speed up the calculation, a list of
          pre-computed indices of the closest matches can be used.

    """
    # determine the closest targets
    if matches is None:
        matches = find_closest_matches(detections, targets)
    # calculate the absolute errors
    errors = calc_errors(detections, targets, matches)
    # get the closest intervals
    intervals = find_closest_intervals(detections, targets, matches)
    # return the relative errors
    return errors / intervals
