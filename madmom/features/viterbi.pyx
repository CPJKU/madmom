# encoding: utf-8
"""
This file contains the speed crucial Viterbi functionality.

@author: Filip Korzeniowski <filip.korzeniowski@jku.at>
@author: Sebastian Böck <sebastian.boeck@jku.at>

"""
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport log

# parallel processing stuff
from cython.parallel cimport prange
import multiprocessing as mp
NUM_THREADS = mp.cpu_count()

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

    # current viterbi variables
    cdef np.ndarray[np.float32_t, ndim=1] v_c = np.empty(num_st,
                                                         dtype=np.float32)
    # previous viterbi variables. will be initialised with prior (first beat)
    cdef np.ndarray[np.float32_t, ndim=1] v_p
    # back-tracking pointers;
    cdef np.ndarray[np.int_t, ndim=2] bps = np.empty((num_x - 1, num_st),
                                                     dtype=np.int)
    # back tracked path, a.k.a. path sequence
    cdef list path = []

    # counters etc.
    cdef int k, i, j, next_state
    cdef double new_prob, sum_k, log_sum = 0.0

    # init first beat
    v_p = pi * activations
    v_p /= v_p.sum()

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
               unsigned int max_tau=23,
               unsigned int num_threads=NUM_THREADS):
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
    :param num_threads:              number of parallel threads
    :return:                         most probable state-space path sequence
                                     for the given activations

    Note: a uniform prior distribution is assumed.

    """
    # number of tempo states
    cdef unsigned int num_tempo_states = max_tau - min_tau
    # number of tempo states
    cdef unsigned int num_states = num_beat_states * num_tempo_states
    # check that the number of states fit into unsigned int16
    cdef unsigned int max_int_value = np.iinfo(np.uint16).max
    if num_states > max_int_value:
        raise AssertionError('current implementation can handle only %i '
                             'states , not %i' % (max_int_value, num_states))
    # number of frames
    cdef unsigned int num_frames = len(activations)
    # current viterbi variables
    cdef np.ndarray[np.float_t, ndim=1] current_viterbi = \
        np.empty(num_states, dtype=np.float)
    # previous viterbi variables, init them with 1s as prior distribution
    cdef np.ndarray[np.float_t, ndim=1] prev_viterbi = \
        np.ones(num_states, dtype=np.float)
    # back-tracking pointers
    cdef np.ndarray[np.uint16_t, ndim=2] back_tracking_pointers = \
        np.empty((num_frames, num_states), dtype=np.uint16)
    # back tracked path, a.k.a. path sequence
    cdef np.ndarray[np.uint16_t, ndim=1] path = \
        np.empty(num_frames, dtype=np.uint16)

    # counters etc.
    cdef unsigned int state, prev_state, beat_state, tempo_state, tempo
    cdef double act, obs, transition_prob
    cdef int frame
    # iterate over all observations
    for frame in range(num_frames):
        # search for best transitions
        # FIXME: prange() is slower for only for 1 thread
        for state in prange(num_states, nogil=True, num_threads=num_threads,
                            schedule='static'):
            # reset the current viterbi variable
            current_viterbi[state] = 0.0
            # position inside beat & tempo
            beat_state = state % num_beat_states
            tempo_state = state / num_beat_states
            tempo = tempo_state + min_tau
            # get the observation
            if beat_state < num_beat_states / observation_lambda:
                obs = activations[frame]
            else:
                obs = (1. - activations[frame]) / (observation_lambda - 1)
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
                back_tracking_pointers[frame, state] = prev_state
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
                    back_tracking_pointers[frame, state] = prev_state
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
                    back_tracking_pointers[frame, state] = prev_state
        # overwrite the old states with the normalised current ones
        # Note: this is faster than unrolling the loop
        prev_viterbi = current_viterbi / current_viterbi.max()
    # fetch the final best state
    state = current_viterbi.argmax()
    # track the path backwards, start with the last frame and do not include
    # the back_tracking_pointers for frame 0, since it includes the transitions
    # to the prior distribution states
    for frame in range(num_frames -1, -1, -1):
        # save the state in the path
        path[frame] = state
        # fetch the next previous one
        state = back_tracking_pointers[frame, state]
    # return the tracked path
    return path
