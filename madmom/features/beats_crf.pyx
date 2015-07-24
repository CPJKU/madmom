# encoding: utf-8
"""
This file contains the speed crucial Viterbi functionality for the
CRFBeatDetector.

@author: Filip Korzeniowski <filip.korzeniowski@jku.at>

"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport INFINITY


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
        # add normalisation factor to activations (see remark in inner loop)
        activations[i] += norm_factor[i]
        v_p[i] = pi[i] + activations[i]

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
                # For even more speedup, we can add the norm_factors at the
                # beginning of the function to the activations.
                new_prob = v_p[i - j] + transition[j]
                if new_prob > v_c[i]:
                    v_c[i] = new_prob
                    bps[k, i] = i - j

            # Add activation and norm_factor, which was added to the
            # activation vector at the beginning of the function.
            v_c[i] += activations[i]

        v_p, v_c = v_c, v_p

    # add the final best state to the path
    path_prob = -INFINITY
    for i in range(num_st):
        # substract the norm factor because they shouldn't have been added
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
