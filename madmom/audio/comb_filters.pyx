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
        Comb filtered signal, float dtype

    Notes
    -----
    y[n] = x[n] + α * x[n - τ] is used as a filter function.

    Examples
    --------
    Comb filter the given signal:

    >>> x = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1])
    >>> feed_forward_comb_filter(x, tau=3, alpha=0.5)
    array([ 0. ,  0. ,  1. ,  0. ,  0. ,  1.5,  0. ,  0. ,  1.5])

    """
    # y[n] = x[n] + α * x[n - τ]
    if tau <= 0:
        raise ValueError('`tau` must be greater than 0')
    y = signal.astype(np.float)
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
        Comb filtered signal, float dtype.

    Notes
    -----
    y[n] = x[n] + α * y[n - τ] is used as a filter function.

    Examples
    --------
    Comb filter the given signal:

    >>> x = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1])
    >>> feed_backward_comb_filter(x, tau=3, alpha=0.5)
    array([ 0.  ,  0.  ,  1.  ,  0.  ,  0.  ,  1.5 ,  0.  ,  0.  ,  1.75])

    """
    if signal.ndim == 1:
        return _feed_backward_comb_filter_1d(signal.astype(np.float),
                                             tau, alpha)
    elif signal.ndim == 2:
        return _feed_backward_comb_filter_2d(signal.astype(np.float),
                                             tau, alpha)
    else:
        raise ValueError('signal must be 1d or 2d')


@cython.boundscheck(False)
def _feed_backward_comb_filter_1d(np.ndarray[np.float_t, ndim=1] signal,
                                  unsigned int tau, float alpha):
    """Feed backward comb filter for 1d signals."""
    # y[n] = x[n] + α * y[n - τ]
    if tau <= 0:
        raise ValueError('`tau` must be greater than 0')
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
def _feed_backward_comb_filter_2d(np.ndarray[np.float_t, ndim=2] signal,
                                  unsigned int tau, float alpha):
    """Feed backward comb filter for 2d signals."""
    # y[n] = x[n] + α * y[n - τ]
    if tau <= 0:
        raise ValueError('`tau` must be greater than 0')
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
        last dimension.

    Notes
    -----
    `tau` and `alpha` must be of same length.

    Examples
    --------
    Filter the given signal with a bank of resonating comb filters.

    >>> x = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1])
    >>> comb_filter(x, feed_forward_comb_filter, [2, 3], [0.5, 0.5])
    array([[ 0. ,  0. ],
           [ 0. ,  0. ],
           [ 1. ,  1. ],
           [ 0. ,  0. ],
           [ 0.5,  0. ],
           [ 1. ,  1.5],
           [ 0. ,  0. ],
           [ 0.5,  0. ],
           [ 1. ,  1.5]])

    Same for a backward filter:

    >>> comb_filter(x, feed_backward_comb_filter, [2, 3], [0.5, 0.5])
    array([[ 0.   ,  0.   ],
           [ 0.   ,  0.   ],
           [ 1.   ,  1.   ],
           [ 0.   ,  0.   ],
           [ 0.5  ,  0.   ],
           [ 1.   ,  1.5  ],
           [ 0.25 ,  0.   ],
           [ 0.5  ,  0.   ],
           [ 1.125,  1.75 ]])

    """
    # convert tau to a integer numpy array
    tau = np.array(tau, dtype=np.int, ndmin=1)
    if tau.ndim != 1:
        raise ValueError('`tau` must be a 1D numpy array')
    # convert alpha to a numpy array
    alpha = np.array(alpha, dtype=np.float, ndmin=1)
    # expand a single alpha value to same length as tau
    if len(alpha) == 1:
        alpha = np.repeat(alpha, len(tau))
    if alpha.ndim != 1:
        raise ValueError('`alpha` must be a 1D numpy array')
    # tau and alpha must have the same length
    if len(tau) != len(alpha):
        raise ValueError('`tau` and `alpha` must have the same length')
    # init output array
    y = []
    for i, t in np.ndenumerate(tau):
        y.append(filter_function(signal, t, alpha[i]))
    if signal.ndim == 1:
        return np.vstack(y).T
    elif signal.ndim == 2:
        return np.dstack(y)
    else:
        raise ValueError('only 1D and 2D signals supported')


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
    `tau` and `alpha` must have the same length.

    Examples
    --------
    Create a processor and then filter the given signal with it.
    The direction of the comb filter function can be given as a literal:

    >>> x = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1])
    >>> proc = CombFilterbankProcessor('forward', [2, 3], [0.5, 0.5])
    >>> proc(x)
    array([[ 0. ,  0. ],
           [ 0. ,  0. ],
           [ 1. ,  1. ],
           [ 0. ,  0. ],
           [ 0.5,  0. ],
           [ 1. ,  1.5],
           [ 0. ,  0. ],
           [ 0.5,  0. ],
           [ 1. ,  1.5]])

    >>> proc = CombFilterbankProcessor('backward', [2, 3], [0.5, 0.5])
    >>> proc(x)
    array([[ 0.   ,  0.   ],
           [ 0.   ,  0.   ],
           [ 1.   ,  1.   ],
           [ 0.   ,  0.   ],
           [ 0.5  ,  0.   ],
           [ 1.   ,  1.5  ],
           [ 0.25 ,  0.   ],
           [ 0.5  ,  0.   ],
           [ 1.125,  1.75 ]])

    """

    def __init__(self, filter_function, tau, alpha):
        # convert tau and alpha to a numpy arrays
        self.tau = np.array(tau, dtype=np.int, ndmin=1)
        self.alpha = np.array(alpha, dtype=np.float, ndmin=1)
        # set the filter function
        if filter_function in ['forward', feed_forward_comb_filter]:
            self.filter_function = feed_forward_comb_filter
        elif filter_function in ['backward', feed_backward_comb_filter]:
            self.filter_function = feed_backward_comb_filter
        else:
            raise ValueError('unknown `filter_function`: %s' % filter_function)

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
            last dimension.

        """
        return comb_filter(data, self.filter_function, self.tau, self.alpha)
