# encoding: utf-8
"""
This file contains Hidden Markov Model (HMM) functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

If you want to change this module and use it interactively, use pyximport.

>>> import pyximport
>>> pyximport.install(reload_support=True,
                      setup_args={'include_dirs': np.get_include()})

"""

import abc
import numpy as np
cimport numpy as np
cimport cython


cdef extern from "math.h":
    float INFINITY


class TransitionModel(object):
    """
    Transition model class for a HMM.

    The transition model is defined similar to a scipy compressed sparse row
    matrix and holds all transition probabilities from one state to an other.

    All states transitioning to state s are stored in:
    states[pointers[s]:pointers[s+1]]

    and their corresponding transition are stored in:
    probabilities[pointers[s]:pointers[s+1]].

    This allows for a parallel computation of the Viterbi path.

    This class should be either used for loading saved transition models or
    being sub-classed to define a specific transition model.

    """

    def __init__(self, states, pointers, probabilities):
        """
        Construct a TransitionModel instance for HMMs.

        :param states:        state indices
        :param pointers:      corresponding pointers
        :param probabilities: and probabilities

        """
        # init some variables
        self.states = states
        self.pointers = pointers
        self.probabilities = probabilities

    @property
    def num_states(self):
        """Number of states."""
        return len(self.pointers) - 1

    @property
    def num_transitions(self):
        """Number of transitions."""
        return len(self.probabilities)

    @property
    def log_probabilities(self):
        """Transition log probabilities."""
        return np.log(self.probabilities)

    @staticmethod
    def make_sparse(states, prev_states, probabilities):
        """
        Return a sparse representation of dense transitions.

        Three 1D numpy arrays of same length must be given. The indices
        correspond to each other, i.e. the first entry of all three arrays
        define the transition from the state defined prev_states[0] to that
        defined in states[0] with the probability defined in probabilities[0].

        :param states:        corresponding states
        :param prev_states:   corresponding previous states
        :param probabilities: transition probabilities
        :return:              tuple (states, pointers, probabilities)

        This method removes all duplicate states and thus allows for parallel
        Viterbi decoding of the HMM.

        """
        from scipy.sparse import csr_matrix
        # check for a proper probability distribution, i.e. the emission
        # probabilities of each prev_state must sum to 1
        if not np.allclose(np.bincount(prev_states, weights=probabilities), 1):
            raise ValueError('Not a probability distribution.')
        # convert everything into a sparse CSR matrix
        transitions = csr_matrix((probabilities, (states, prev_states)))
        # convert to correct types
        states = transitions.indices.astype(np.uint32)
        pointers = transitions.indptr.astype(np.uint32)
        probabilities = transitions.data.astype(dtype=np.float)
        # return them
        return states, pointers, probabilities

    @classmethod
    def from_dense(cls, states, prev_states, probabilities):
        """
        Instantiate a TransitionModel from dense transitions.

        Three 1D numpy arrays of same length must be given. The indices
        correspond to each other, i.e. the first entry of all three arrays
        define the transition from the state defined prev_states[0] to that
        defined in states[0] with the probability defined in probabilities[0].

        :param states:        corresponding states
        :param prev_states:   corresponding previous states
        :param probabilities: transition probabilities
        :return:              TransitionModel instance

        """
        # get a sparse representation of the transitions
        transitions = cls.make_sparse(states, prev_states, probabilities)
        # instantiate a new TransitionModel and return it
        return cls(*transitions)


class ObservationModel(object):
    """
    Observation model class for a HMM.

    The observation model is defined as two plain numpy arrays, log_densities
    and pointers.

    The observation model must have an attribute 'pointers' containing a plain
    1D numpy array of length equal to the number of states of the HMM and
    pointing from each state to the corresponding column of the 'log_densities'
    array, which gets returned by the 'compute_log_densities()' method with the
    observations as an argument. The 'pointers' type must be np.uint32.

    The 'log_densities' is a 2D numpy array with the number of rows being equal
    to the length of the observations and the columns representing the
    different observation log probability densities. The type must be np.float.

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, pointers):
        """
        Construct a ObservationModel instance for a HMM.

        :param pointers: pointers from HMM states to the correct densities
                         column [numpy array]
        """

        self.pointers = pointers

    @abc.abstractmethod
    def compute_log_densities(self, observations):
        """
        Compute the log densities (or, probabilities) of the observations for
        each state.

        :param observations: observations (list, numpy array, ...)
        :return:             log densities as a 2D numpy array with the number
                             of rows being equal to the number of observations
                             and the columns representing the different
                             observation log probability densities. The type
                             must be np.float.
        """
        return

    def compute_densities(self, observations):
        """
        Compute the densities (or, probabilities) of the observations for
        each state. This defaults to computing the exp of the log-densities.
        You can provide a special implementation for speed-up.

        :param observations: observations (list, numpy array, ...)
        :return:             densities as a 2D numpy array with the number
                             of rows being equal to the number of observations
                             and the columns representing the different
                             observation log probability densities. The type
                             must be np.float.
        """
        return np.exp(self.compute_log_densities(observations))


class DiscreteObservationModel(ObservationModel):
    """
    Simple discrete observation model that takes an observation matrix of
    the form (num_states x num_observations) containing P(obs | state)
    """

    def __init__(self, observation_probs):
        """
        :param observation_probs: observation probabilities as 2d numpy array
                                  of the form (num_states x num_observations)
                                  Has to sum to 1 over the first axis, since
                                  it represents P(obs | state)
        """
        if not np.allclose(observation_probs.sum(axis=1), 1):
            raise ValueError('Not a probability distribution.')

        self.obs_probs = observation_probs
        super(DiscreteObservationModel, self).__init__(
            np.arange(self.obs_probs.shape[0], dtype=np.uint32))

    def compute_densities(self, observations):
        return self.obs_probs[:, observations].T

    def compute_log_densities(self, observations):
        return np.log(self.compute_densities(observations))


class HiddenMarkovModel(object):
    """
    Hidden Markov Model

    To search for the best path through the state space with the Viterbi
    algorithm, a `transition_model`, `observation_model` and
    `initial_distribution` must be defined.

    """

    def __init__(self, transition_model, observation_model,
                 initial_distribution=None):
        """
        Construct a new Hidden Markov Model.

        :param transition_model:     transition model [TransitionModel]
        :param observation_model:    observation model [ObservationModel]
        :param initial_distribution: initial state distribution [numpy array];
                                     if 'None' is given a uniform distribution
                                     is assumed

        """
        self.transition_model = transition_model
        self.observation_model = observation_model
        if initial_distribution is None:
            initial_distribution = (np.ones(transition_model.num_states,
                                           dtype=np.float)
                                    / transition_model.num_states)
        if not np.allclose(initial_distribution.sum(), 1):
            raise ValueError('Initial distribution is not a probability '
                             'distribution.')
        self.initial_distribution = initial_distribution

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def viterbi(self, observations):
        """
        Determine the best path with the Viterbi algorithm.

        :param observations: observations to decode the optimal path for
        :return:             tuple with best state-space path sequence and its
                             log probability

        """
        # transition model stuff
        tm = self.transition_model
        cdef unsigned int [::1] tm_states = tm.states
        cdef unsigned int [::1] tm_pointers = tm.pointers
        cdef double [::1] tm_probabilities = tm.log_probabilities
        cdef unsigned int num_states = tm.num_states

        # observation model stuff
        om = self.observation_model
        cdef unsigned int num_observations = len(observations)
        cdef unsigned int [::1] om_pointers = om.pointers
        cdef double [:, ::1] om_densities = \
            om.compute_log_densities(observations)

        # current viterbi variables
        cdef double [::1] current_viterbi = np.empty(num_states,
                                                     dtype=np.float)

        # previous viterbi variables, init with the initial state distribution
        cdef double [::1] previous_viterbi = np.log(self.initial_distribution)

        # back-tracking pointers
        cdef unsigned int [:, ::1] bt_pointers = np.empty((num_observations,
                                                           num_states),
                                                          dtype=np.uint32)
        # define counters etc.
        cdef unsigned int state, frame, prev_state, pointer
        cdef double density, transition_prob

        # iterate over all observations
        for frame in range(num_observations):
            # search for the best transition
            for state in range(num_states):
                # reset the current viterbi variable
                current_viterbi[state] = -INFINITY
                # get the observation model probability density value
                # the om_pointers array holds pointers to the correct
                # observation probability density value for the actual state
                # (i.e. column in the om_densities array)
                # Note: defining density here gives a 5% speed-up!?
                density = om_densities[frame, om_pointers[state]]
                # iterate over all possible previous states
                # the tm_pointers array holds pointers to the states which are
                # stored in the tm_states array
                for pointer in range(tm_pointers[state],
                                     tm_pointers[state + 1]):
                    # get the previous state
                    prev_state = tm_states[pointer]
                    # weight the previous state with the transition probability
                    # and the current observation probability density
                    transition_prob = previous_viterbi[prev_state] + \
                                      tm_probabilities[pointer] + density
                    # if this transition probability is greater than the
                    # current one, overwrite it and save the previous state
                    # in the back tracking pointers
                    if transition_prob > current_viterbi[state]:
                        # update the transition probability
                        current_viterbi[state] = transition_prob
                        # update the back tracking pointers
                        bt_pointers[frame, state] = prev_state

            # overwrite the old states with the current ones
            previous_viterbi[:] = current_viterbi

        # fetch the final best state
        state = np.asarray(current_viterbi).argmax()
        # set the path's probability to that of the best state
        log_probability = current_viterbi[state]
        # back tracked path, a.k.a. path sequence
        path = np.empty(num_observations, dtype=np.uint32)
        # track the path backwards, start with the last frame and do not
        # include the pointer for frame 0, since it includes the transitions
        # to the prior distribution states
        for frame in range(num_observations -1, -1, -1):
            # save the state in the path
            path[frame] = state
            # fetch the next previous one
            state = bt_pointers[frame, state]
        # return the tracked path and its probability
        return path, log_probability

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def forward(self, observations):
        """
        Compute the forward variables at each time step. Instead of computing
        in the log domain, we normalise at each step, which is faster for
        the forward algorithm.

        :param observations: obervations to compute the forward variables for
        :returns: 2D numpy array containing the forward variables
        """

        # transition model stuff
        tm = self.transition_model
        cdef unsigned int [::1] tm_states = tm.states
        cdef unsigned int [::1] tm_ptrs = tm.pointers
        cdef double [::1] tm_probabilities = tm.probabilities
        cdef unsigned int num_states = tm.num_states

        # observation model stuff
        om = self.observation_model
        cdef unsigned int num_observations = len(observations)
        cdef unsigned int [::1] om_pointers = om.pointers
        cdef double [:, ::1] om_densities = \
            om.compute_densities(observations)

        cdef double[:, ::1] fwd = np.zeros((num_observations + 1, num_states),
                                           dtype=np.float)

        cdef unsigned int prev_ptr, frame, state, cur, prev
        cdef double prob_sum, norm_factor, trans_sum

        for i in range(self.initial_distribution.shape[0]):
            fwd[0, i] = self.initial_distribution[i]

        for frame in range(num_observations):
            cur = frame + 1
            prev = frame
            prob_sum = 0

            for state in range(num_states):
                trans_sum = 0
                for prev_ptr in range(tm_ptrs[state], tm_ptrs[state + 1]):
                    fwd[cur, state] += fwd[prev, tm_states[prev_ptr]] * \
                                       tm_probabilities[prev_ptr]
                fwd[cur, state] *= om_densities[frame, om_pointers[state]]
                prob_sum += fwd[cur, state]

            norm_factor = 1. / prob_sum
            for state in range(num_states):
                fwd[cur, state] *= norm_factor

        return np.asarray(fwd)[1:]

# alias
HMM = HiddenMarkovModel
