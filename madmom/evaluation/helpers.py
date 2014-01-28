#!/usr/bin/env python
# encoding: utf-8
"""
This file contains various evaluation helper functions.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

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
    # solution found at: http://stackoverflow.com/questions/8914491/
    indices = targets.searchsorted(detections)
    indices = np.clip(indices, 1, len(targets) - 1)
    left = targets[indices - 1]
    right = targets[indices]
    indices -= detections - left < right - detections
    # return the indices of the closest matches
    return indices


def calc_errors(detections, targets, matches=None):
    """
    Calculates the errors of the detections relative to the closest targets.

    :param detections: sequence of events to be matched [seconds]
    :param targets:    sequence of possible matches [seconds]
    :param matches:    indices of the closest matches
    :returns:          a list of errors to closest matches [seconds]

    Note: the sequences must be ordered! To speed up the calculation, a list
          of pre-computed indices of the closest matches can be used.

    """
    # determine the closest targets
    if matches is None:
        matches = find_closest_matches(detections, targets)
    # calc error relative to those targets
    errors = detections - targets[matches]
    # return the errors
    return errors


def calc_absolute_errors(detections, targets, matches=None):
    """
    Calculate absolute errors of the detections relative to the closest
    targets.

    :param detections: sequence of events to be matched [seconds]
    :param targets:    sequence of possible matches [seconds]
    :param matches:    indices of the closest matches
    :returns:          a list of errors to closest matches [seconds]

    Note: the sequences must be ordered! To speed up the calculation, a list
          of pre-computed indices of the closest matches can be used.

    """
    # return the errors
    return np.abs(calc_errors(detections, targets, matches))


def calc_relative_errors(detections, targets, matches=None):
    """
    Relative errors of the detections to the closest targets.
    The absolute error is weighted by the absolute value of the target.

    :param detections: sequence of events to be matched [seconds]
    :param targets:    sequence of possible matches [seconds]
    :param matches:    indices of the closest matches
    :returns:          a list of relative errors to closest matches [seconds]

    Note: the sequences must be ordered! To speed up the calculation, a list of
          pre-computed indices of the closest matches can be used.

    """
    # determine the closest targets
    if matches is None:
        matches = find_closest_matches(detections, targets)
    # calculate the absolute errors
    errors = calc_errors(detections, targets, matches)
    # return the relative errors
    return np.abs(1 - (errors / targets[matches]))
