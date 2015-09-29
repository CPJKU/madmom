# encoding: utf-8
"""
This file contains the speed crucial Viterbi functionality for the
CRFBeatDetector plus some functions computing the distributions and
normalisation factors...

"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport INFINITY


def initial_distribution(num_states, interval):
    """
    Compute the initial distribution.

    :param num_states: number of states in the model
    :param interval:   beat interval of the piece [frames]
    :return:           initial distribution of the model

    """
    # We leave the initial distribution un-normalised because we want the
    # position of the first beat not to influence the probability of the
    # beat sequence. Normalising would favour shorter intervals.
    init_dist = np.ones(num_states, dtype=np.float32)
    init_dist[interval:] = 0
    return init_dist


def transition_distribution(interval, interval_sigma):
    """
    Compute the transition distribution between beats.

    :param interval:       interval of the piece [frames]
    :param interval_sigma: allowed deviation from the interval per beat
    :return:               transition distribution between beats

    """
    from scipy.stats import norm

    move_range = np.arange(interval * 2, dtype=np.float)
    # to avoid floating point hell due to np.log2(0)
    move_range[0] = 0.000001

    trans_dist = norm.pdf(np.log2(move_range),
                          loc=np.log2(interval),
                          scale=interval_sigma)
    trans_dist /= trans_dist.sum()
    return trans_dist.astype(np.float32)


def normalisation_factors(activations, transition_distribution):
    """
    Compute normalisation factors for model.

    :param activations:             beat activation function of the piece
    :param transition_distribution: transition distribution of the model
    :return:                        normalisation factors for model

    """
    from scipy.ndimage.filters import correlate1d
    return correlate1d(activations, transition_distribution,
                       mode='constant', cval=0,
                       origin=-int(transition_distribution.shape[0] / 2))


def best_sequence(activations, interval, interval_sigma):
    """
    Extract the best beat sequence for a piece.

    :param activations:    beat activation function
    :param interval:       beat interval of the piece
    :param interval_sigma: allowed deviation from the interval per beat
    :return:               tuple with extracted beat positions [frames]
                           and log probability of beat sequence
    """
    init = initial_distribution(activations.shape[0],
                                    interval)
    trans = transition_distribution(interval, interval_sigma)
    norm_fact = normalisation_factors(activations, trans)

    # ignore division by zero warnings when taking the logarithm of 0.0,
    # the result -inf is fine anyways!
    with np.errstate(divide='ignore'):
        init = np.log(init)
        trans = np.log(trans)
        norm_fact = np.log(norm_fact)
        log_act = np.log(activations)

    return viterbi(init, trans, norm_fact, log_act, interval)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def viterbi(float [::1] pi, float[::1] transition, float[::1] norm_factor,
            float [::1] activations, int tau):
    """
    Viterbi algorithm to compute the most likely beat sequence from the
    given activations and the dominant interval.

    :param pi:          initial distribution
    :param transition:  transition distribution
    :param norm_factor: normalisation factors
    :param activations: beat activations
    :param tau:         dominant interval [frames]
    :return:            tuple with extracted beat positions [frame indices]
                        and log probability of beat sequence

    """
    # number of states
    cdef int num_st = activations.shape[0]
    # number of transitions
    cdef int num_tr = transition.shape[0]
    # number of beat variables
    cdef int num_x = num_st / tau

    # current viterbi variables
    cdef float [::1] v_c = np.empty(num_st, dtype=np.float32)
    # previous viterbi variables. will be initialized with prior (first beat)
    cdef float [::1] v_p = np.empty(num_st, dtype=np.float32)
    # back-tracking pointers;
    cdef long [:, ::1] bps = np.empty((num_x - 1, num_st), dtype=np.int)
    # back tracked path, a.k.a. path sequence
    cdef long [::1] path = np.empty(num_x, dtype=np.int)

    # counters etc.
    cdef int k, i, j, next_state
    cdef double new_prob, path_prob

    # init first beat
    for i in range(num_st):
        v_p[i] = pi[i] + activations[i] + norm_factor[i]

    # iterate over all beats; the 1st beat is given by prior
    for k in range(num_x - 1):
        # reset all current viterbi variables
        v_c[:] = -INFINITY

        # find the best transition for each state i
        for i in range(num_st):
            # j is the number of frames we look back
            for j in range(min(i, num_tr)):
                # Important remark: the actual computation we'd have to do here
                # is v_p[i - j] + norm_factor[i - j] + transition[j] +
                # activations[i].
                #
                # For speedup, we can add the activation after
                # the loop, since it does not change with j. Additionally,
                # if we immediately add the normalisation factor to v_c[i],
                # we can skip adding norm_factor[i - j] for each v_p[i - j].
                new_prob = v_p[i - j] + transition[j]
                if new_prob > v_c[i]:
                    v_c[i] = new_prob
                    bps[k, i] = i - j

            # Add activation and norm_factor. For the last random variable,
            # we'll subtract norm_factor later when searching the maximum
            v_c[i] += activations[i] + norm_factor[i]

        v_p, v_c = v_c, v_p

    # add the final best state to the path
    path_prob = -INFINITY
    for i in range(num_st):
        # subtract the norm factor because they shouldn't have been added
        # for the last random variable
        v_p[i] -= norm_factor[i]
        if v_p[i] > path_prob:
            next_state = i
            path_prob = v_p[i]
    path[num_x - 1] = next_state

    # track the path backwards
    for i in range(num_x - 2, -1, -1):
        next_state = bps[i, next_state]
        path[i] = next_state

    # return the best sequence and its log probability
    return np.asarray(path), path_prob
