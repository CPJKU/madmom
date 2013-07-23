#!/usr/bin/env python
# encoding: utf-8
"""
Copyright (c) 2012-2013 Sebastian Böck <sebastian.boeck@jku.at>
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


# helper functions to read/write events from files and combine these events
def load_events(filename):
    """
    Load a list of events from file.

    :param filename: name of the file
    :return: list of events

    """
    # array for events
    events = []
    # try to read in the onsets from the file
    with open(filename, 'rb') as f:
        # read in each line of the file
        for line in f:
            # append the event (1st column) to the list, ignore the rest
            events.append(float(line.split()[0]))
    # return
    return events


def write_events(events, filename):
    """
    Write the detected onsets to the given file.

    :param events: list of events [seconds]
    :param filename: output file name

    """
    with open(filename, 'w') as f:
        for e in events:
            f.write(str(e) + '\n')


def combine_events(events, delta):
    """
    Combine all events within a certain range.

    :param events: list of events [seconds]
    :param delta: combination length [seconds]
    :return: list of combined events

    """
    # sort the events
    events.sort()
    events_length = len(events)
    events_index = 0
    # array for combined events
    comb = []
    # iterate over all events
    while events_index < events_length - 1:
        # get the first event
        first = events[events_index]
        # always increase the events index
        events_index += 1
        # get the second event
        second = events[events_index]
        # combine the two events?
        if second - first <= delta:
            # two events within the combination window, combine them and replace
            # the second event in the original list with the mean of the events
            events[events_index] = (first + second) / 2.
        else:
            # the two events can not be combined,
            # store the first event in the new list
            comb.append(first)
    # always append the last element of the list
    comb.append(events[-1])
    # return the combined onsets
    return comb


# helper functions for two sequences
def find_closest_match(detections, targets):
    """
    Find the closest matches for detections in targets.

    :param detections: sequence of events to be matched [seconds]
    :param targets: sequence of possible matches [seconds]
    :returns: a list of indices with closest matches

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
    if not isinstance(detections, np.ndarray):
        detections = np.array(detections)
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)
    indices = targets.searchsorted(detections)
    indices = np.clip(indices, 1, len(targets) - 1)
    left = targets[indices - 1]
    right = targets[indices]
    indices -= detections - left < right - detections
    # return the indices of the closest matches
    return indices


## this is slower!
#def find_closest_match_loop(detections, targets):
#    """
#    Find the closest matches for detections in targets.
#
#    :param detections: sequence of events to be matched [seconds]
#    :param targets: sequence of possible matches [seconds]
#    :returns: a list of indices with closest matches
#
#    """
#    # list of indices
#    indices = []
#    # evaluate
#    det_length = len(detections)
#    tar_length = len(targets)
#    det_index = 0
#    tar_index = 0
#    prev_error = None
#    while det_index < det_length and tar_index < tar_length:
#        # compare the two events
#        error = detections[det_index] - targets[tar_index]
#        if prev_error is None:
#            # this is the first error, continue with the next target
#            tar_index += 1
#        else:
#            # absulte error is smaller than the previous one, add the previous target to the list
#            if abs(error) < abs(prev_error):
#                # the previous target was closer, add it to the list
#                indices.append(tar_index - 1)
#            if error <= 0:
#                # continue with the next detection
#                det_index += 1
#            else:
#                # continue with the next target
#                tar_index += 1
#        # save the error
#        prev_error = error
#    # all remaining detections have the last target as their closest match
#    tar_index = [tar_index - 1]
#    # the number of missing targets is the det_length - det_index
#    indices.extend(tar_index * (det_length - det_index))
#    # return the list
#    return indices


def errors(detections, targets):
    """
    Errors of the detections relative to the closest targets.

    :param detections: sequence of events to be matched [seconds]
    :param targets: sequence of possible matches [seconds]
    :returns: a list of errors to closest matches [seconds]

    Note: the sequences must be ordered!

    """
    # determine the closest targets
    indices = find_closest_match(detections, targets)
    # calc error relative to those targets
    errors = np.asarray(detections) - np.asarray(targets)[indices]
    # return the errors
    return errors


def absolute_errors(detections, targets):
    """
    Absolute errors of the detections relative to the closest targets.

    :param detections: sequence of events to be matched [seconds]
    :param targets: sequence of possible matches [seconds]
    :returns: a list of errors to closest matches [seconds]

    Note: the sequences must be ordered!

    """
    # return the errors
    return np.abs(errors(detections, targets))


def relative_errors(detections, targets):
    """
    Relative errors of the detections to the closest targets.
    The absolute error is weighted by the interval of two targets surrounding
    each detection.

    :param detections: sequence of events to be matched [seconds]
    :param targets: sequence of possible matches [seconds]
    :returns: a list of relative errors to closest matches [seconds]

    Note: the sequences must be ordered!

    """
    # init array for intervals; expand the size by one so we can combine the
    # intervals to the previous and next item into one array
    intervals = np.zeros(len(targets) + 1)
    # intervals to previous target
    intervals[1:-1] = np.diff(targets)
    # the interval from the first target to the left is the same as the right
    intervals[0] = intervals[1]
    # the interval from the last target to the right is the same as the left
    intervals[-1] = intervals[-2]
    # note: intervals to the next target are always those at the next index
    # determine the closest targets
    closest = find_closest_match(detections, targets)
    # calculate the absolute errors (note: same method as in function above, but
    # doubled here to not calculate the closest targets twice)
    errors = np.asarray(detections) - np.asarray(targets)[closest]
    # if the errors are positive, the detection is after the target
    # thus, the needed interval is from the closest target towards the next one
    errors[errors > 0] /= intervals[closest[errors > 0] + 1]
    # if before, interval to previous target accordingly
    errors[errors < 0] /= intervals[closest[errors < 0]]
    # return the relative errors
    return errors


#def relative_errors_loop(detections, targets):
#    """
#    Relative errors of the detections to the closest targets.
#    The absolute error is weighted by the interval of two targets surrounding
#    each detection.
#
#    :param detections: sequence of events to be matched [seconds]
#    :param targets: sequence of possible matches [seconds]
#    :returns: a list of relative errors to closest matches [seconds]
#
#    Note: the sequences must be ordered!
#    """
#    errors = np.zeros(len(detections))
#    # determine closest targets to detections
#    closest = find_closest_match(detections, targets)
#    # store the number of targets
#    last_target = len(targets) - 1
#    # iterate over all detections
#    for det in range(len(detections)):
#        # find closest target
#        tar = closest[det]
#        # difference to this target
#        diff = detections[det] - targets[tar]
#        # calculate the relative error for this target
#        if tar == 0:
#            # closet target is the first one or before the current beat
#            # calculate the diff to the next target
#            interval = targets[tar + 1] - targets[tar]
#        elif tar == last_target:
#            # closet target is the last one or after the current beat
#            # calculate the diff to the previous target
#            interval = targets[tar] - targets[tar - 1]
#        else:
#            # normal
#            if diff > 0:
#                # closet target is before the current beat
#                interval = targets[tar + 1] - targets[tar]
#            else:
#                # closet target is after the current beat
#                interval = targets[tar] - targets[tar - 1]
#        # set the error in the array
#        errors[det] = diff / interval
#    return errors
