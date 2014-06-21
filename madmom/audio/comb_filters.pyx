#!/usr/bin/env python
# encoding: utf-8
"""
This file contains the speed crucial filter and filterbank functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import numpy as np

cimport cython
cimport numpy as np

def feed_backward_comb_filter(x, tau, alpha):
    """
    Filter the signal with a feed backward comb filter.

    :param x:     signal
    :param tau:   delay length
    :param alpha: scaling factor
    :return:      comb filtered signal

    """
    if x.ndim == 1:
        return feed_backward_comb_filter_1d(x, tau, alpha)
    elif x.ndim == 2:
        return feed_backward_comb_filter_2d(x, tau, alpha)
    else:
        raise ValueError('signal x must be 1d or 2d')

@cython.boundscheck(False)
def feed_backward_comb_filter_1d(np.ndarray[np.float_t, ndim=1] x,
                                 unsigned int tau,
                                 float alpha):
    """
    Filter the signal with a feed backward comb filter.

    :param x:     signal
    :param tau:   delay length
    :param alpha: scaling factor
    :return:      comb filtered signal

    """
    # y[n] = x[n] + α * y[n - τ]
    if tau <= 0:
        raise ValueError('tau must be greater than 0')
    # type definitions
    cdef np.ndarray[np.float_t, ndim=1] y = np.copy(x)
    cdef unsigned int n
    cdef unsigned int len_x = len(x)
    # loop over the complete signal
    for n in xrange(tau, len(x)):
        # Note: saw this formula somewhere, but it seems to produce less
        #       accurate tempo predictions...
        #       y[n] = (1. - alpha) * x[n] + alpha * y[n - tau]
        # add a delayed version of the output signal
        y[n] = x[n] + alpha * y[n - tau]
    # return
    return y

@cython.boundscheck(False)
def feed_backward_comb_filter_2d(np.ndarray[np.float_t, ndim=2] x,
                                 unsigned int tau,
                                 float alpha):
    """
    Filter the signal with a feed backward comb filter.

    :param x:     signal
    :param tau:   delay length
    :param alpha: scaling factor
    :return:      comb filtered signal

    """
    # y[n] = x[n] + α * y[n - τ]
    if tau <= 0:
        raise ValueError('tau must be greater than 0')
    # type definitions
    cdef np.ndarray[np.float_t, ndim=2] y = np.copy(x)
    cdef unsigned int n
    cdef unsigned int len_x = len(x)
    # loop over the dimensions
    for d in xrange(2):
        # loop over the complete signal
        for n in xrange(tau, len(x)):
            # Note: saw this formula somewhere, but it seems to produce less
            #       accurate tempo predictions...
            #       y[n, d] = (1. - alpha) * x[n, d] + alpha * y[n - tau, d]
            # add a delayed version of the output signal
            y[n, d] = x[n, d] + alpha * y[n - tau, d]
    # return
    return y
