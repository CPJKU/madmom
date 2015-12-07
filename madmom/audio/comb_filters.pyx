# encoding: utf-8
# cython: embedsignature=True
"""
This module contains comb-filter and comb-filterbank functionality.

"""

from __future__ import absolute_import, division, print_function

import numpy as np

cimport cython
cimport numpy as np

from madmom.processors import Processor


# feed forward comb filter
def feed_forward_comb_filter(signal, tau, alpha):
    """
    Filter the signal with a feed forward comb filter.

    Parameters
    ----------
    signal : numpy array
        Signal.
    tau : int
        Delay length.
    alpha : float
        Scaling factor.

    Returns
    -------
    comb_filtered_signal : numpy array
        Comb filtered signal.

    Notes
    -----
    y[n] = x[n] + α * x[n - τ] is used as a filter function.

    """
    # y[n] = x[n] + α * x[n - τ]
    if tau <= 0:
        raise ValueError('tau must be greater than 0')
    y = np.copy(signal)
    # add the delayed signal
    y[tau:] += alpha * signal[:-tau]
    # return
    return y


# feed backward comb filter
def feed_backward_comb_filter(signal, tau, alpha):
    """
    Filter the signal with a feed backward comb filter.

    Parameters
    ----------
    signal : numpy array
        Signal.
    tau : int
        Delay length.
    alpha : float
        Scaling factor.

    Returns
    -------
    comb_filtered_signal : numpy array
        Comb filtered signal.

    Notes
    -----
    y[n] = x[n] + α * y[n - τ] is used as a filter function.

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

    Parameters
    ----------
    signal : 1D numpy array
        Signal.
    tau : int
        Delay length.
    alpha : float
        Scaling factor.

    Returns
    -------
    comb_filtered_signal : numpy array
        Comb filtered signal.

    Notes
    -----
    y[n] = x[n] + α * y[n - τ] is used as a filter function.

    """
    # y[n] = x[n] + α * y[n - τ]
    if tau <= 0:
        raise ValueError('tau must be greater than 0')
    # type definitions
    cdef np.ndarray[np.float_t, ndim=1] y = signal.copy()
    cdef unsigned int n
    # loop over the complete signal
    for n in range(tau, len(signal)):
        # Note: saw this formula somewhere, but it seems to produce less
        #       accurate tempo predictions...
        #       y[n] = (1. - alpha) * x[n] + alpha * y[n - tau]
        # add a delayed version of the output signal
        y[n] += alpha * y[n - tau]
    # return
    return y


@cython.boundscheck(False)
def feed_backward_comb_filter_2d(np.ndarray[np.float_t, ndim=2] signal,
                                 unsigned int tau,
                                 float alpha):
    """
    Filter the signal with a feed backward comb filter.

    Parameters
    ----------
    signal : 2D numpy array
        Signal.
    tau : int
        Delay length.
    alpha : float
        Scaling factor.

    Returns
    -------
    comb_filtered_signal : numpy array
        Comb filtered signal.

    Notes
    -----
    y[n] = x[n] + α * y[n - τ] is used as a filter function.

    """
    # y[n] = x[n] + α * y[n - τ]
    if tau <= 0:
        raise ValueError('tau must be greater than 0')
    # type definitions
    cdef np.ndarray[np.float_t, ndim=2] y = signal.copy()
    cdef unsigned int d, n
    # loop over the dimensions
    for d in range(2):
        # loop over the complete signal
        for n in range(tau, len(signal)):
            # add a delayed version of the output signal
            y[n, d] += alpha * y[n - tau, d]
    # return
    return y


# comb filter
def comb_filter(signal, filter_function, tau, alpha):
    """
    Filter the signal with a bank of either feed forward or backward comb
    filters.

    Parameters
    ----------
    signal : numpy array
        Signal.
    filter_function : {feed_forward_comb_filter, feed_backward_comb_filter}
        Filter function to use (feed forward or backward).
    tau : list or numpy array, shape (N,)
        Delay length(s) [frames].
    alpha : list or numpy array, shape (N,)
        Corresponding scaling factor(s).

    Returns
    -------
    comb_filtered_signal : numpy array
        Comb filtered signal with the different taus aligned along the (new)
        first dimension.

    Notes
    -----
    `tau` and `alpha` must be of same length.

    """
    # convert tau to a integer numpy array
    tau = np.asarray(tau, dtype=int)
    if tau.ndim != 1:
        raise ValueError('tau must be a 1D numpy array')
    # convert alpha to a numpy array
    alpha = np.asarray(alpha, dtype=float)
    if alpha.ndim != 1:
        raise ValueError('alpha must be a 1D numpy array')
    # alpha & tau must have the same size
    if tau.size != alpha.size:
        raise AssertionError('alpha & tau must have the same size')
    # determine output array size
    size = list(signal.shape)
    # add dimension of tau range size (new 1st dim)
    size.insert(0, len(tau))
    # init output array
    y = np.zeros(tuple(size))
    for i, t in np.ndenumerate(tau):
        y[i] = filter_function(signal, t, alpha[i])
    return y


class CombFilterbankProcessor(Processor):
    """
    CombFilterbankProcessor class.

    Parameters
    ----------
    filter_function : filter function or str
        Filter function to use {feed_forward_comb_filter,
        feed_backward_comb_filter} or a string literal {'forward', 'backward'}.
    tau : list or numpy array, shape (N,)
        Delay length(s) [frames].
    alpha : list or numpy array, shape (N,)
        Corresponding scaling factor(s).

    Notes
    -----
    `tau` and `alpha` must be of same length.

    """

    def __init__(self, filter_function, tau, alpha):
        # convert tau to a numpy array
        if isinstance(tau, int):
            self.tau = np.asarray([tau], dtype=int)
        elif isinstance(tau, (list, np.ndarray)):
            self.tau = np.asarray(tau, dtype=int)
        else:
            raise ValueError('`tau` must be cast-able as an int numpy array')

        # set the filter function
        if filter_function in ['forward', feed_forward_comb_filter]:
            self.comb_filter_function = feed_forward_comb_filter
        elif filter_function in ['backward', feed_backward_comb_filter]:
            self.comb_filter_function = feed_backward_comb_filter
        else:
            raise ValueError('unknown `filter_function`: %s' % filter_function)

        # convert alpha to a numpy array
        if isinstance(alpha, (float, int)):
            self.alpha = np.asarray([alpha] * len(tau), dtype=float)
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = np.asarray(alpha, dtype=float)
        else:
            raise ValueError('`alpha` must be cast-able as float numpy array')

    def process(self, data):
        """
        Process the given data with the comb filter.

        Parameters
        ----------
        data : numpy array
            Data to be filtered/processed.

        Returns
        -------
        comb_filtered_data : numpy array
            Comb filtered data with the different taus aligned along the (new)
            first dimension.

        """
        return comb_filter(data, self.comb_filter_function, self.tau,
                           self.alpha)
