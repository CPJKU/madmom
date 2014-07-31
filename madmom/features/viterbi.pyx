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


cdef class MultiModelDBN(object):
    """
    Dynamic Bayesian network for beat tracking.

    """
    # define some class variables
    cdef readonly unsigned int num_beat_states
    cdef readonly unsigned int observation_lambda
    cdef readonly unsigned int min_tempo
    cdef readonly unsigned int max_tempo
    cdef readonly double tempo_change_probability

    cdef readonly unsigned int num_threads
    cdef readonly np.ndarray path
    cdef readonly np.ndarray observations

    def __init__(self, num_beat_states=1280,
                 tempo_change_probability=0.008,
                 observation_lambda=16, min_tempo=11, max_tempo=47,
                 num_threads=NUM_THREADS):
        """
        Construct a new dynamic Bayesian network suitable for multi-model beat
        tracking.

        :param num_beat_states:          number of beat states for one beat
                                         period
        :param tempo_change_probability: probability of a tempo change from
                                         one observation to the next one
        :param observation_lambda:       split one beat period into N parts,
                                         the first representing beat states
                                         and the remaining non-beat states
        :param min_tempo:                minimum number of cells to progress
                                         from one observation to the next one
        :param max_tempo:                maximum number of cells to progress
                                         from one observation to the next one
        :param num_threads:              number of parallel threads

        """
        # save given variables
        self.num_beat_states = num_beat_states
        self.tempo_change_probability = tempo_change_probability
        self.observation_lambda = observation_lambda
        self.min_tempo = min_tempo
        self.max_tempo = max_tempo
        self.num_threads = num_threads
        # init observations and path (aka the best sequence)
        self.path = np.empty(0)
        self.observations = np.empty(0)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def viterbi(self, float [::1] observations):
        """
        Determine the best path given the observations.

        :param observations: the observations
        :return:             most probable state-space path sequence for the
                             given observations

        Note: a uniform prior distribution is assumed.

        """
        # save the given observations
        self.observations = np.asarray(observations)
        # cache class/instance variables needed in the loops
        cdef unsigned int num_beat_states = self.num_beat_states
        cdef double tempo_change_probability = self.tempo_change_probability
        cdef unsigned int observation_lambda = self.observation_lambda
        cdef unsigned int min_tempo = self.min_tempo
        cdef unsigned int max_tempo = self.max_tempo
        cdef unsigned int num_threads = self.num_threads
        # number of tempo states
        cdef unsigned int num_tempo_states = max_tempo - min_tempo
        # number of states
        cdef unsigned int num_states = num_beat_states * num_tempo_states
        # check that the number of states fit into unsigned int16
        if num_states > np.iinfo(np.uint16).max:
            raise AssertionError('DBN can handle only %i states, not %i' %
                                 (np.iinfo(np.uint16).max, num_states))
        # number of frames
        cdef unsigned int num_frames = len(observations)
        # current viterbi variables
        current_viterbi = np.empty(num_states, dtype=np.float)
        # typed memoryview thereof
        # Note: the ::1 notation indicates that is memory continuous
        cdef double [::1] current_viterbi_ = current_viterbi
        # previous viterbi variables, init them with 1s as prior distribution
        # TODO: allow other priors
        prev_viterbi = np.ones(num_states, dtype=np.float)
        # typed memoryview thereof
        cdef double [::1] prev_viterbi_ = prev_viterbi
        # back-tracking pointers
        back_tracking_pointers = np.empty((num_frames, num_states),
                                           dtype=np.uint16)
        # typed memoryview thereof
        cdef unsigned short [:, ::1] back_tracking_pointers_ = \
            back_tracking_pointers
        # back tracked path, a.k.a. path sequence
        path = np.empty(num_frames, dtype=np.uint16)
        # cdef unsigned short [::1] path_ = path

        # define counters etc.
        cdef unsigned int state, prev_state, beat_state, tempo_state, tempo
        cdef double act, obs, transition_prob
        cdef double same_tempo_prob = 1. - tempo_change_probability
        cdef double change_tempo_prob = 0.5 * tempo_change_probability
        cdef unsigned int beat_no_beat = num_beat_states / observation_lambda
        cdef int frame
        # iterate over all observations
        for frame in range(num_frames):
            # search for best transitions
            # FIXME: prange() is slower for only for 1 thread
            #        refactor the whole stuff as cdef class MMViterbi():
            #        and add a cdef compute_state() method which can be called
            #        via range() or prange() whichever is called depending on
            #        the number of threads used
            for state in prange(num_states, nogil=True,
                                num_threads=num_threads,
                                schedule='static'):
            # for state in range (num_states):
                # reset the current viterbi variable
                current_viterbi_[state] = 0.0
                # position inside beat & tempo
                beat_state = state % num_beat_states
                tempo_state = state / num_beat_states
                tempo = tempo_state + min_tempo
                # get the observation
                if beat_state < beat_no_beat:
                    obs = observations[frame]
                else:
                    obs = (1. - observations[frame]) / \
                          (observation_lambda - 1)
                # for each state check the 3 possible transitions
                # previous state with same tempo
                # Note: we add num_beat_states before the modulo operation so
                #       that it can be computed in C (which is faster)
                prev_state = ((beat_state + num_beat_states - tempo) %
                              num_beat_states +
                              (tempo_state * num_beat_states))
                # probability for transition from same tempo
                transition_prob = (prev_viterbi_[prev_state] *
                                   same_tempo_prob * obs)
                # if this transition probability is greater than the current
                # one, overwrite it and save the previous state in the current
                # pointers
                if transition_prob > current_viterbi_[state]:
                    current_viterbi_[state] = transition_prob
                    back_tracking_pointers_[frame, state] = prev_state
                # transition from slower tempo
                if tempo_state > 0:
                    # previous state with slower tempo
                    # Note: we add num_beat_states before the modulo operation
                    #       so that it can be computed in C (which is faster)
                    prev_state = ((beat_state + num_beat_states -
                                   (tempo - 1)) % num_beat_states +
                                  ((tempo_state - 1) * num_beat_states))
                    # probability for transition from slower tempo
                    transition_prob = (prev_viterbi_[prev_state] *
                                       change_tempo_prob * obs)
                    if transition_prob > current_viterbi_[state]:
                        current_viterbi_[state] = transition_prob
                        back_tracking_pointers_[frame, state] = prev_state
                # transition from faster tempo
                if tempo_state < num_tempo_states - 1:
                    # previous state with faster tempo
                    # Note: we add num_beat_states before the modulo operation
                    #       so that it can be computed in C (which is faster)
                    prev_state = ((beat_state + num_beat_states -
                                   (tempo + 1)) % num_beat_states +
                                  ((tempo_state + 1) * num_beat_states))
                    # probability for transition from faster tempo
                    transition_prob = (prev_viterbi_[prev_state] *
                                       change_tempo_prob * obs)
                    if transition_prob > current_viterbi_[state]:
                        current_viterbi_[state] = transition_prob
                        back_tracking_pointers_[frame, state] = prev_state
            # overwrite the old states with the normalised current ones
            # Note: this is faster than unrolling the loop! But it is a bit
            #       tricky: we need to call max() on the numpy array but do
            #       the normalisation and assignment on the memoryview
            prev_viterbi_ = current_viterbi_ / current_viterbi.max()

        # fetch the final best state
        state = current_viterbi.argmax()
        # track the path backwards, start with the last frame and do not
        # include the back_tracking_pointers for frame 0, since it includes the
        # transitions to the prior distribution states
        for frame in range(num_frames -1, -1, -1):
            # save the state in the path
            path[frame] = state
            # fetch the next previous one
            state = back_tracking_pointers[frame, state]
        # save the tracked path and return it
        self.path = path
        print path % num_beat_states
        return path

    @property
    def beat_states(self):
        """Beat states."""
        return self.path % self.num_beat_states

    @property
    def tempo_states(self):
        """Tempo states."""
        return self.path / self.num_beat_states

