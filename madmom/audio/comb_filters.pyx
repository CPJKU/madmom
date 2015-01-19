#!/usr/bin/env python
# encoding: utf-8
"""
This file contains the speed crucial filter and filterbank functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import numpy as np

cimport cython
cimport numpy as np

def feed_backward_comb_filter(signal, tau, alpha):
    """
    Filter the signal with a feed backward comb filter.

    :param signal: signal
    :param tau:    delay length
    :param alpha:  scaling factor
    :return:       comb filtered signal

    """
    if signal.ndim == 1:
        return feed_backward_comb_filter_1d(signal, tau, alpha)
    elif signal.ndim == 2:
        return feed_backward_comb_filter_2d(signal, tau, alpha)
    else:
        raise ValueError('signal must be 1d or 2d')

@cython.boundscheck(False)
def feed_backward_comb_filter_1d(np.ndarray[np.float_t, ndim=1] signal,
                                 unsigned int tau,
                                 float alpha):
    """
    Filter the signal with a feed backward comb filter.

    :param signal: signal
    :param tau:    delay length
    :param alpha:  scaling factor
    :return:       comb filtered signal

    """
    # y[n] = x[n] + α * y[n - τ]
    if tau <= 0:
        raise ValueError('tau must be greater than 0')
    # type definitions
    cdef np.ndarray[np.float_t, ndim=1] y = np.copy(signal)
    cdef unsigned int n
    # loop over the complete signal
    for n in range(tau, len(signal)):
        # Note: saw this formula somewhere, but it seems to produce less
        #       accurate tempo predictions...
        #       y[n] = (1. - alpha) * x[n] + alpha * y[n - tau]
        # add a delayed version of the output signal
        y[n] = signal[n] + alpha * y[n - tau]
    # return
    return y

@cython.boundscheck(False)
def feed_backward_comb_filter_2d(np.ndarray[np.float_t, ndim=2] signal,
                                 unsigned int tau,
                                 float alpha):
    """
    Filter the signal with a feed backward comb filter.

    :param signal: signal
    :param tau:    delay length
    :param alpha:  scaling factor
    :return:       comb filtered signal

    """
    # y[n] = x[n] + α * y[n - τ]
    if tau <= 0:
        raise ValueError('tau must be greater than 0')
    # type definitions
    cdef np.ndarray[np.float_t, ndim=2] y = np.copy(signal)
    cdef unsigned int d, n
    # loop over the dimensions
    for d in range(2):
        # loop over the complete signal
        for n in range(tau, len(signal)):
            # Note: saw this formula somewhere, but it seems to produce less
            #       accurate tempo predictions...
            #       y[n, d] = (1. - alpha) * x[n, d] + alpha * y[n - tau, d]
            # add a delayed version of the output signal
            y[n, d] = signal[n, d] + alpha * y[n - tau, d]
    # return
    return y
