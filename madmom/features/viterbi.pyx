# encoding: utf-8
"""
This file contains the speed crucial conditional random field related
functionality.

@author: Filip Korzeniowski <filip.korzeniowski@jku.at>

"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport log


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def crf_viterbi(np.ndarray[np.float32_t, ndim=1] pi,
                np.ndarray[np.float32_t, ndim=1] transition,
                np.ndarray[np.float32_t, ndim=1] norm_factor,
                np.ndarray[np.float32_t, ndim=1] activations,
                int tau):
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

    # back-tracking pointer sequence
    cdef list bps = []
    # current viterbi variables
    cdef np.ndarray[np.float32_t, ndim=1] v_c = np.zeros(num_st, dtype=np.float32)
    # previous viterbi variables
    cdef np.ndarray[np.float32_t, ndim=1] v_p = np.zeros(num_st, dtype=np.float32)
    # current back-tracking pointers; init them with -1
    cdef np.ndarray[np.int_t, ndim=1] bp_c = np.ones_like(v_c, dtype=int) * -1
    # back tracked path, a.k.a. path sequence
    cdef list path = []

    # counters etc.
    cdef int k, i, j, next_state
    cdef double cur, new, sum_k, log_sum = 0.0

    # init first beat
    v_p = pi * activations
    v_p /= v_p.sum()

    # iterate over all beats; the 1st beat is given by prior
    for k in range(num_x - 1):
        # reset all current viterbi variables
        for i in range(num_st):
            v_c[i] = 0.0
        # search the best transition
        for i in range(num_st):
            for j in range(num_tr):
                if (i + j) >= num_st:
                    break

                cur = v_c[i + j]
                new = v_p[i] * transition[j] * activations[i + j] * norm_factor[i]

                if new > cur:
                    v_c[i + j] = new
                    bp_c[i + j] = i

        sum_k = 0.0
        for i in range(num_st):
            sum_k += v_c[i]

        for i in range(num_st):
            v_c[i] /= sum_k

        log_sum += log(sum_k)

        v_p, v_c = v_c, v_p
        bps.append(bp_c.copy())

    # add the final best state to the path
    next_state = v_p.argmax()
    path.append(next_state)
    # track the path backwards
    for i in range(num_x - 2, -1, -1):
        next_state = bps[i][next_state]
        path.append(next_state)
    # return the best sequence and its log probability
    return np.array(path[::-1]), log(v_p.max()) + log_sum


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def mm_viterbi(np.ndarray[np.float32_t, ndim=1] activations,
               unsigned int num_beat_states=640,
               double tempo_change_probability=0.002,
               unsigned int observation_lambda=16,
               unsigned int min_tau=5,
               unsigned int max_tau=23):
    """
    Track the beats with a dynamic Bayesian network.

    :param activations:              beat activations
    :param num_beat_states:          number of beat states for one beat period
    :param tempo_change_probability: probability of a tempo change from
                                     one observation to the next one
    :param observation_lambda:       TODO: find better name + description
    :param min_tau:                  minimum number of beat cells to progress
                                     from one observation to the next one
    :param max_tau:                  maximum number of beat cells to progress
                                     from one observation to the next one
    :return:                         most probable state-space path sequence
                                     for the given activations

    """
    # number of states
    cdef unsigned int num_tempo_states = max_tau - min_tau
    cdef unsigned int num_states = num_beat_states * num_tempo_states
    # current viterbi variables
    cdef np.ndarray[np.float_t, ndim=1] current_viterbi = \
        np.zeros(num_states, dtype=np.float)
    # previous viterbi variables
    cdef np.ndarray[np.float_t, ndim=1] prev_viterbi = \
        np.ones(num_states, dtype=np.float)
    # current back-tracking pointers; init them with -1
    cdef np.ndarray[np.int_t, ndim=1] current_pointers = \
        np.ones_like(current_viterbi, dtype=int) * -1
    # back-tracking pointer sequence
    cdef list back_tracking_pointers = []
    # back tracked path, a.k.a. path sequence
    cdef list path = []

    # counters etc.
    cdef unsigned int state, prev_state, beat_state, tempo_state, tempo
    cdef double act, obs, transition_prob

    # iterate over all observations
    for act in activations:
        # reset all current viterbi variables
        for state in range(num_states):
            current_viterbi[state] = 0.0
        # search for best transitions
        for state in range(num_states):
            # position inside beat & tempo
            beat_state = state % num_beat_states
            tempo_state = state / num_beat_states
            tempo = tempo_state + min_tau
            # get the observation
            if beat_state < num_beat_states / observation_lambda:
                obs = act
            else:
                obs = (1. - act) / (observation_lambda - 1)

            # for each state check the 3 possible transitions

            # previous state with same tempo
            # Note: we add num_beat_states before the modulo operation so
            #       that it can be computed in C (which is faster)
            prev_state = ((beat_state + num_beat_states - tempo) %
                          num_beat_states + (tempo_state * num_beat_states))
            # probability for transition from same tempo
            transition_prob = (prev_viterbi[prev_state] *
                               (1. - tempo_change_probability) * obs)
            # if this transition probability is greater than the current one,
            # overwrite it and save the previous state in the current pointers
            if transition_prob > current_viterbi[state]:
                current_viterbi[state] = transition_prob
                current_pointers[state] = prev_state
            # transition from slower tempo
            if tempo_state > 0:
                # previous state with slower tempo
                # Note: we add num_beat_states before the modulo operation so
                #       that it can be computed in C (which is faster)
                prev_state = ((beat_state + num_beat_states - (tempo - 1)) %
                               num_beat_states +
                              ((tempo_state - 1) * num_beat_states))
                # probability for transition from slower tempo
                transition_prob = (prev_viterbi[prev_state] *
                                   0.5 * tempo_change_probability * obs)
                if transition_prob > current_viterbi[state]:
                    current_viterbi[state] = transition_prob
                    current_pointers[state] = prev_state
            # transition from faster tempo
            if tempo_state < num_tempo_states - 1:
                # previous state with faster tempo
                # Note: we add num_beat_states before the modulo operation so
                #       that it can be computed in C (which is faster)
                prev_state = ((beat_state + num_beat_states - (tempo + 1)) %
                              num_beat_states +
                              ((tempo_state + 1) * num_beat_states))
                # probability for transition from faster tempo
                transition_prob = (prev_viterbi[prev_state] *
                                   0.5 * tempo_change_probability * obs)
                if transition_prob > current_viterbi[state]:
                    current_viterbi[state] = transition_prob
                    current_pointers[state] = prev_state

        # append current pointers to the back-tracking pointer sequence list
        back_tracking_pointers.append(current_pointers.copy())
        # overwrite the old states with the normalised current ones
        prev_viterbi = current_viterbi / current_viterbi.max()

    # add the final best state to the path
    state = current_viterbi.argmax()
    path.append(state)
    # track the path backwards, start with the last frame and do not include
    # the back_tracking_pointers for frame 0, since it includes the transitions
    # to the prior distribution states
    cdef unsigned int frame
    for frame in range(len(back_tracking_pointers) - 1, 0, -1):
        state = back_tracking_pointers[frame][state]
        path.append(state)
    # return the tracked path
    return np.array(path[::-1])
