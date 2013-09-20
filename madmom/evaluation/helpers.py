#!/usr/bin/env python
# encoding: utf-8
"""
This file contains various helper functions used by cp.evaluation modules.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

# helper functions related to evaluation
import numpy as np


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
    # if before (or same position) use the interval to previous target accordingly
    closest_interval[errors <= 0] = intervals[matches[errors <= 0]]
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
