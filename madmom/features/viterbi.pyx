# encoding: utf-8
"""
This file contains the speed crucial Viterbi functionality.

@author: Filip Korzeniowski <filip.korzeniowski@jku.at>

"""
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport log


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def crf_viterbi(float [::1] pi, float[::1] transition, float[::1] norm_factor,
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
    cdef double new_prob, sum_k, path_prob, log_sum = 0.0

    # init first beat
    for i in range(num_st):
        v_p[i] = pi[i] * activations[i]
        sum_k += v_p[i]
    for i in range(num_st):
        v_p[i] = v_p[i] / sum_k

    sum_k = 0

    # iterate over all beats; the 1st beat is given by prior
    for k in range(num_x - 1):
        # reset all current viterbi variables
        for i in range(num_st):
            v_c[i] = 0.0

        # find the best transition for each state
        for i in range(num_st):
            for j in range(num_tr):
                if (i - j) < 0:
                    break

                new_prob = v_p[i - j] * transition[j] * activations[i] * \
                           norm_factor[i - j]

                if new_prob > v_c[i]:
                    v_c[i] = new_prob
                    bps[k, i] = i - j

        sum_k = 0.0
        for i in range(num_st):
            sum_k += v_c[i]
        for i in range(num_st):
            v_c[i] /= sum_k

        log_sum += log(sum_k)

        v_p, v_c = v_c, v_p

    # add the final best state to the path
    path_prob = 0.0
    for i in range(num_st):
        if v_p[i] > path_prob:
            next_state = i
            path_prob = v_p[i]
    path[num_x - 1] = next_state

    # track the path backwards
    for i in range(num_x - 2, -1, -1):
        next_state = bps[i, next_state]
        path[i] = next_state

    # return the best sequence and its log probability
    return np.asarray(path), log(path_prob) + log_sum
