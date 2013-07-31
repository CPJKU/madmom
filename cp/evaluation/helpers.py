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


def load_events(filename):
    """
    Load a list of events from file.

    :param filename: name of the file
    :return: list of events

    """
    # Note: the loop is much faster than np.loadtxt(filename, usecols=[0])
    # array for events
    events = []
    # try to read in the onsets from the file
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
    # return array if no events must be combined
    errors = calc_intervals(events)
    if not (events[errors <= delta].any()):
        return events
    # array for combined events
    comb = []
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
            # the second event in the original list with the mean of the events
            events[idx] = (first + second) / 2.
        else:
            # the two events can not be combined,
            # store the first event in the new list
            comb.append(first)
    # always append the last element of the list
    comb.append(events[-1])
    # return the combined events
    return np.asarray(comb)


def combine_events_(events, delta):
    """
    Combine all events within a certain range.

    :param events: list of events [seconds]
    :param delta: combination length [seconds]
    :return: list of combined events

    """
    # return array if no events must be combined
    fwd_errors = calc_intervals(events, fwd=True)
    if not (events[fwd_errors <= delta].any()):
        return events
    # array for combined events
    bwd_errors = calc_intervals(events)
    # if the events are located far enough apart, just use them
    indices = np.intersect1d(np.nonzero(fwd_errors > delta)[0], np.nonzero(bwd_errors > delta)[0])
    print indices
    comb = events[indices].tolist()
    print 'ok', comb
    # the remaining must be merged
    indices = np.unique(np.append(np.nonzero(fwd_errors <= delta)[0], np.nonzero(bwd_errors <= delta)[0]))
    print indices
    # iterate over all events with errors <= delta
    events_ = np.copy(events)
    first = None
    for idx in indices:
        print idx
        # exit the loop
        if idx + 1 not in indices:
            # store the merged event in the new list
            comb.append(first)
            # reset the counters
            first = None
            second = None
            # continue with next segment
            continue
        # check if we already have an event to merge
        if first is None:
            # get the first event
            first = events_[idx]
            print 'first', first
        else:
            # treat this event as the second one
            second = events_[idx]
            print 'second', second
        # ok, we have two events to merge
        if first and second:
            # merge them and replace the first
            first = (first + second) / 2.
    # return all events without zeros
    comb.sort()
    return np.asarray(comb)

#def combine_events_(events, delta):
#    """
#    Combine all events within a certain range.
#
#    :param events: array with events [seconds]
#    :param delta: combination length [seconds]
#    :return: array with combined events
#
#    """
#    #
#    comb = np.zeros_like(events)
#    errors = calc_intervals(events)
#    # if the events are located far enough apart, just use them
#    comb[errors > delta] = events[errors > delta]
#    comb[errors <= delta] = np.mean((events[errors <= delta], events[1:][errors <= delta]), axis=0)
#    print comb
#    # always append the last element of the list
#    # return the combined events
#    return np.asarray(comb)


def filter_events(events, key):
    raise NotImplemented


def find_closest_matches(detections, targets):
    """
    Find the closest matches for detections in targets.

    :param detections: sequence of events to be matched [seconds]
    :param targets: sequence of possible matches [seconds]
    :returns: a numpy array of indices with closest matches

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
    :param targets: sequence of possible matches [seconds]
    :param matches: indices of the closest matches [default=None]
    :returns: a list of closest target intervals [seconds]

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


#def find_closest_intervals_(detections, targets, matches=None):
#    """
#    Find the closest target interval surrounding the detections.
#
#    :param detections: sequence of events to be matched [seconds]
#    :param targets: sequence of possible matches [seconds]
#    :param matches: indices of the closest matches [default=None]
#    :returns: a list of closest target intervals [seconds]
#
#    Note: the sequences must be ordered! To speed up the calculation, a list of
#          pre-computed indices of the closest matches can be used.
#
#    """
#    # init array
#    closest_interval = np.ones_like(detections)
#    # intervals to next target
#    fwd_intervals = calc_intervals(targets, fwd=True)
#    # intervals to previous target
#    bwd_intervals = calc_intervals(targets)
#    # determine the closest targets
#    if matches is None:
#        matches = find_closest_matches(detections, targets)
#    # calculate the absolute errors
#    errors = calc_errors(detections, targets, matches)
#    # if the errors are positive, the detection is after the target
#    # thus, the needed interval is from the closest target towards the next one
#    closest_interval[errors > 0] = fwd_intervals[matches[errors > 0]]
#    # if before, interval to previous target accordingly
#    closest_interval[errors < 0] = bwd_intervals[matches[errors < 0]]
#    # return the closest interval
#    return closest_interval


def calc_errors(detections, targets, matches=None):
    """
    Calculates the errors of the detections relative to the closest targets.

    :param detections: sequence of events to be matched [seconds]
    :param targets: sequence of possible matches [seconds]
    :param matches: indices of the closest matches [default=None]
    :returns: a list of errors to closest matches [seconds]

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
    :param fwd: calculate the intervals to the next event [default=False]
    :returns: the intervals [seconds]

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
    :param targets: sequence of possible matches [seconds]
    :param matches: indices of the closest matches [default=None]
    :returns: a list of errors to closest matches [seconds]

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
    :param targets: sequence of possible matches [seconds]
    :param matches: indices of the closest matches [default=None]
    :returns: a list of relative errors to closest matches [seconds]

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
