# encoding: utf-8
"""
This file contains Hidden Markov Model (HMM) functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

If you want to change this module and use it interactively, use pyximport.

>>> import pyximport
>>> pyximport.install(reload_support=True,
                      setup_args={'include_dirs': np.get_include()})

"""

import numpy as np
cimport numpy as np
cimport cython

# parallel processing stuff
from cython.parallel cimport prange
NUM_THREADS = 1

cdef extern from "math.h":
    float INFINITY


# transition_model stuff
class TransitionModel(object):
    """
    Transition model class for a HMM.

    The transition model is defined similar to a scipy compressed sparse row
    matrix and holds all transition log probabilities from one state to an
    other.

    All state indices for row state s are stored in
    states[pointers[s]:pointers[s+1]]
    and their corresponding log probabilities are stored in
    log_probabilities[pointers[s]:pointers[s+1]].

    This allows for a parallel computation of the viterbi path.

    This class should be either used for loading saved transition models or
    being sub-classed to define a new transition model.

    """

    def __init__(self, states, pointers, log_probabilities):
        """
        Construct a TransitionModel instance for DBNs.

        :param states:            state indices
        :param pointers:          corresponding pointers
        :param log_probabilities: and log probabilities

        """
        # init some variables
        self.states = states
        self.pointers = pointers
        self.log_probabilities = log_probabilities

    @property
    def num_states(self):
        """Number of states."""
        return len(self.pointers) - 1

    @property
    def num_transitions(self):
        """Number of transitions."""
        return len(self.log_probabilities)

    @classmethod
    def make_sparse(cls, states, prev_states, log_probabilities):
        """
        Return a sparse representation of dense transitions.

        Three 1D numpy arrays of same length must be given. The indices
        correspond to each other, i.e. the first entry of all three arrays
        define the transition from the state defined prev_states[0] to that
        defined in states[0] with the log probability defined in
        log_probabilities[0].

        :param states:            corresponding states
        :param prev_states:       corresponding previous states
        :param log_probabilities: transition log probabilities

        This method removes all duplicate states and thus allows for parallel
        processing of the Viterbi of the HMM.

        """
        from scipy.sparse import csr_matrix
        # convert everything into a sparse CSR matrix
        transitions = csr_matrix((log_probabilities, (states, prev_states)))
        # convert to correct types
        states = transitions.indices.astype(np.uint32)
        pointers = transitions.indptr.astype(np.uint32)
        log_probabilities = transitions.data.astype(dtype=np.float)
        # instantiate a new TransitionModel and return it
        return states, pointers, log_probabilities


# observation stuff
class ObservationModel(object):
    """
    Observation model class for a DBN.

    The observation model is defined as two plain numpy arrays, log_densities
    and pointers.

    The 'log_densities' is a 2D numpy array with the number of rows being equal
    to the length of the observation_model and the columns representing the
    different observation log probability densities. The type must be np.float.

    The 'pointers' is a 1D numpy array and has a length equal to the number of
    states of the DBN and points from each state to the corresponding column
    of the 'log_densities' array. The type must be np.uint32.

    """

    def __init__(self, log_densities, pointers=None):
        """
        Construct a ObservationModel instance for a DBN.

        :param log_densities: observation log densities [numpy array]
        :param pointers:      pointers from DBN states to the correct densities
                              column [numpy array]

        If `log_densities` are 1D, they are converted to a 2D representation
        with only 1 column.
        If `pointers` is 'None', a pointers vector of the same length as the
        `log_densities` is created pointing always to the first column.

        """
        # convert the densities to a 2d numpy array if needed
        if log_densities.ndim == 1:
            log_densities = np.atleast_2d(log_densities).T
        self.log_densities = np.asarray(log_densities, dtype=np.float)
        # construct a pointers vector if needed
        if pointers is None:
            self.pointers = np.zeros(len(log_densities), dtype=np.uint32)


# inline function to determine the best previous state
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _best_prev_state(int state, int frame,
                                  double [::1] current_viterbi,
                                  double [::1] previous_viterbi,
                                  double [:, ::1] om_densities,
                                  unsigned int [::1] om_pointers,
                                  unsigned int [::1] tm_states,
                                  unsigned int [::1] tm_pointers,
                                  double [::1] tm_probabilities,
                                  unsigned int [:, ::1] pointers) nogil:
    """
    Inline function to determine the best previous state.

    :param state:            current state
    :param frame:            current frame
    :param current_viterbi:  current viterbi variables
    :param previous_viterbi: previous viterbi variables
    :param om_densities:     observation model densities
    :param om_pointers:      observation model pointers
    :param tm_states:        transition model states
    :param tm_pointers:      transition model pointers
    :param tm_probabilities: transition model probabilities
    :param pointers:         back tracking pointers

    """
    # define variables
    cdef unsigned int prev_state, pointer
    cdef double density, transition_prob
    # reset the current viterbi variable
    current_viterbi[state] = -INFINITY
    # get the observation model probability density value
    # the om_pointers array holds pointers to the correct observation
    # probability density value for the actual state (i.e. column in the
    # om_densities array)
    # Note: defining density here gives a 5% speed-up!?
    density = om_densities[frame, om_pointers[state]]
    # iterate over all possible previous states
    # the tm_pointers array holds pointers to the states which are
    # stored in the tm_states array
    for pointer in range(tm_pointers[state], tm_pointers[state + 1]):
        # get the previous state
        prev_state = tm_states[pointer]
        # weight the previous state with the transition probability
        # and the current observation probability density
        transition_prob = previous_viterbi[prev_state] + \
                          tm_probabilities[pointer] + density
        # if this transition probability is greater than the current one,
        # overwrite it and save the previous state in the current pointers
        if transition_prob > current_viterbi[state]:
            # update the transition probability
            current_viterbi[state] = transition_prob
            # update the back tracking pointers
            pointers[frame, state] = prev_state

# HMM stuff
class HiddenMarkovModel(object):
    """
    Hidden Markov Model

    To search for the best path through the state space with the Viterbi
    algorithm, a `transition_model`, `observation_model` and
    `initial_distribution` must be defined.

    """

    def __init__(self, transition_model, observation_model,
                 initial_distribution=None, num_threads=NUM_THREADS):
        """
        Construct a new Hidden Markov Model.

        :param transition_model:     transition model [TransitionModel]
        :param observation_model:    observation model [ObservationModel]
        :param initial_distribution: initial state distribution [numpy array];
                                     if 'None' is given a uniform distribution
                                     is assumed
        :param num_threads:          number of threads for parallel Viterbi
                                     decoding [int]

        """
        self.transition_model = transition_model
        self.observation_model = observation_model
        if initial_distribution is None:
            initial_distribution = np.log(np.ones(transition_model.num_states,
                                                  dtype=np.float) /
                                          transition_model.num_states)
        self.initial_distribution = initial_distribution
        if num_threads is None:
            num_threads = NUM_THREADS
        self.num_threads = num_threads

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def viterbi(self):
        """
        Determine the best path with the Viterbi algorithm.

        :return: best state-space path sequence and its log probability

        """
        # transition model stuff
        tm = self.transition_model
        cdef unsigned int [::1] tm_states = tm.states
        cdef unsigned int [::1] tm_pointers = tm.pointers
        cdef double [::1] tm_probabilities = tm.log_probabilities
        cdef unsigned int num_states = tm.num_states

        # observation model stuff
        om = self.observation_model
        cdef double [:, ::1] om_densities = om.log_densities
        cdef unsigned int [::1] om_pointers = om.pointers
        cdef unsigned int num_observations = len(om.log_densities)

        # current viterbi variables
        cdef double [::1] current_viterbi = np.empty(num_states,
                                                     dtype=np.float)

        # previous viterbi variables, init with the initial state distribution
        cdef double [::1] previous_viterbi = self.initial_distribution

        # back-tracking pointers
        cdef unsigned int [:, ::1] bt_pointers = np.empty((num_observations,
                                                           num_states),
                                                          dtype=np.uint32)
        # define counters etc.
        cdef int state, frame
        cdef unsigned int prev_state, pointer, num_threads = self.num_threads
        cdef double obs, transition_prob

        # iterate over all observation_model
        for frame in range(num_observations):
            # range() is faster than prange() for 1 thread
            if num_threads == 1:
                # search for best transition_model sequentially
                for state in range(num_states):
                    _best_prev_state(state, frame, current_viterbi,
                                          previous_viterbi, om_densities,
                                          om_pointers, tm_states, tm_pointers,
                                          tm_probabilities, bt_pointers)
            else:
                # search for best transition_model in parallel
                for state in prange(num_states, nogil=True, schedule='static',
                                    num_threads=num_threads):
                    _best_prev_state(state, frame, current_viterbi,
                                          previous_viterbi, om_densities,
                                          om_pointers, tm_states, tm_pointers,
                                          tm_probabilities, bt_pointers)

            # overwrite the old states with the current ones
            previous_viterbi[:] = current_viterbi

        # fetch the final best state
        state = np.asarray(current_viterbi).argmax()
        # set the path's probability to that of the best state
        log_probability = current_viterbi[state]
        # back tracked path, a.k.a. path sequence
        path = np.empty(num_observations, dtype=np.uint32)
        # track the path backwards, start with the last frame and do not
        # include the pointer for frame 0, since it includes the transition_model
        # to the prior distribution states
        for frame in range(num_observations -1, -1, -1):
            # save the state in the path
            path[frame] = state
            # fetch the next previous one
            state = bt_pointers[frame, state]
        # return the tracked path and its probability
        return path, log_probability