# encoding: utf-8
# cython: embedsignature=True
"""
This module contains Hidden Markov Model (HMM) functionality.

Notes
-----
If you want to change this module and use it interactively, use pyximport.

>>> import pyximport
>>> pyximport.install(reload_support=True,
...                   setup_args={'include_dirs': np.get_include()})
... # doctest: +ELLIPSIS
(None, <pyximport.pyximport.PyxImporter object at 0x...>)
"""

from __future__ import absolute_import, division, print_function

import warnings
import numpy as np

cimport numpy as np
cimport cython

from numpy.math cimport INFINITY


ctypedef np.uint32_t uint32_t


class TransitionModel(object):
    """
    Transition model class for a HMM.

    The transition model is defined similar to a scipy compressed sparse row
    matrix and holds all transition probabilities from one state to an other.
    This allows an efficient Viterbi decoding of the HMM.

    Parameters
    ----------
    states : numpy array
        All states transitioning to state s are stored in:
        states[pointers[s]:pointers[s+1]]
    pointers : numpy array
        Pointers for the `states` array for state s.
    probabilities : numpy array
        The corresponding transition are stored in:
        probabilities[pointers[s]:pointers[s+1]].

    Notes
    -----
    This class should be either used for loading saved transition models or
    being sub-classed to define a specific transition model.

    See Also
    --------
    scipy.sparse.csr_matrix

    Examples
    --------
    Create a simple transition model with two states using a list of
    transitions and their probabilities

    >>> tm = TransitionModel.from_dense([0, 1, 0, 1], [0, 0, 1, 1],
    ...                                 [0.8, 0.2, 0.3, 0.7])
    >>> tm  # doctest: +ELLIPSIS
    <madmom.ml.hmm.TransitionModel object at 0x...>

    TransitionModel.from_dense will check if the supplied probabilties for
    each state sum to 1 (and thus represent a correct probability distribution)

    >>> tm = TransitionModel.from_dense([0, 1], [1, 0], [0.5, 1.0])
    ... # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    ValueError: Not a probability distribution.

    """

    def __init__(self, states, pointers, probabilities):
        # save the parameters
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
    def make_dense(states, pointers, probabilities):
        """
        Return a dense representation of sparse transitions.

        Parameters
        ----------
        states : numpy array
            All states transitioning to state s are returned in:
            states[pointers[s]:pointers[s+1]]
        pointers : numpy array
            Pointers for the `states` array for state s.
        probabilities : numpy array
            The corresponding transition are returned in:
            probabilities[pointers[s]:pointers[s+1]].

        Returns
        -------
        states : numpy array, shape (num_transitions,)
            Array with states (i.e. destination states).
        prev_states : numpy array, shape (num_transitions,)
            Array with previous states (i.e. origination states).
        probabilities : numpy array, shape (num_transitions,)
            Transition probabilities.

        See Also
        --------
        :class:`TransitionModel`

        Notes
        -----
        Three 1D numpy arrays of same length must be given. The indices
        correspond to each other, i.e. the first entry of all three arrays
        define the transition from the state defined prev_states[0] to that
        defined in states[0] with the probability defined in probabilities[0].

        """
        from scipy.sparse import csr_matrix
        # convert everything into a sparse CSR matrix
        transitions = csr_matrix((np.array(probabilities),
                                  np.array(states), np.array(pointers)))
        # convert to correct types
        states, prev_states = transitions.nonzero()
        # return them
        return states, prev_states, probabilities

    @staticmethod
    def make_sparse(states, prev_states, probabilities):
        """
        Return a sparse representation of dense transitions.

        This method removes all duplicate states and thus allows an efficient
        Viterbi decoding of the HMM.

        Parameters
        ----------
        states : numpy array, shape (num_transitions,)
            Array with states (i.e. destination states).
        prev_states : numpy array, shape (num_transitions,)
            Array with previous states (i.e. origination states).
        probabilities : numpy array, shape (num_transitions,)
            Transition probabilities.

        Returns
        -------
        states : numpy array
            All states transitioning to state s are returned in:
            states[pointers[s]:pointers[s+1]]
        pointers : numpy array
            Pointers for the `states` array for state s.
        probabilities : numpy array
            The corresponding transition are returned in:
            probabilities[pointers[s]:pointers[s+1]].

        See Also
        --------
        :class:`TransitionModel`

        Notes
        -----
        Three 1D numpy arrays of same length must be given. The indices
        correspond to each other, i.e. the first entry of all three arrays
        define the transition from the state defined prev_states[0] to that
        defined in states[0] with the probability defined in probabilities[0].

        """
        from scipy.sparse import csr_matrix
        # check for a proper probability distribution, i.e. the emission
        # probabilities of each prev_state must sum to 1
        states = np.asarray(states)
        prev_states = np.asarray(prev_states, dtype=np.int)
        probabilities = np.asarray(probabilities)
        if not np.allclose(np.bincount(prev_states, weights=probabilities), 1):
            raise ValueError('Not a probability distribution.')
        # convert everything into a sparse CSR matrix, make sure it is square.
        # looking through prev_states is enough, because there *must* be a
        # transition *from* every state
        num_states = max(prev_states) + 1
        transitions = csr_matrix((probabilities, (states, prev_states)),
                                 shape=(num_states, num_states))
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

        Parameters
        ----------
        states : numpy array, shape (num_transitions,)
            Array with states (i.e. destination states).
        prev_states : numpy array, shape (num_transitions,)
            Array with previous states (i.e. origination states).
        probabilities : numpy array, shape (num_transitions,)
            Transition probabilities.

        Returns
        -------
        :class:`TransitionModel` instance
            TransitionModel instance.

        """
        # get a sparse representation of the transitions
        transitions = cls.make_sparse(states, prev_states, probabilities)
        # instantiate a new TransitionModel and return it
        return cls(*transitions)


class ObservationModel(object):
    """
    Observation model class for a HMM.

    The observation model is defined as a plain 1D numpy arrays `pointers` and
    the methods `log_densities()` and `densities()` which return 2D numpy
    arrays with the (log) densities of the observations.

    Parameters
    ----------
    pointers : numpy array (num_states,)
        Pointers from HMM states to the correct densities. The length of the
        array must be equal to the number of states of the HMM and pointing
        from each state to the corresponding column of the array returned
        by one of the `log_densities()` or `densities()` methods. The
        `pointers` type must be np.uint32.

    See Also
    --------
    ObservationModel.log_densities
    ObservationModel.densities

    """

    def __init__(self, pointers):
        # save parameters
        self.pointers = pointers

    def log_densities(self, observations):
        """
        Log densities (or probabilities) of the observations for each state.

        Parameters
        ----------
        observations : numpy array
            Observations.

        Returns
        -------
        numpy array
            Log densities as a 2D numpy array with the number of rows being
            equal to the number of observations and the columns representing
            the different observation log probability densities. The type must
            be np.float.

        """
        raise NotImplementedError('must be implemented by subclass')

    def densities(self, observations):
        """
        Densities (or probabilities) of the observations for each state.

        This defaults to computing the exp of the `log_densities`.
        You can provide a special implementation to speed-up everything.

        Parameters
        ----------
        observations : numpy array
            Observations.

        Returns
        -------
        numpy array
            Densities as a 2D numpy array with the number of rows being equal
            to the number of observations and the columns representing the
            different observation log probability densities. The type must be
            np.float.

        """
        return np.exp(self.log_densities(observations))


class DiscreteObservationModel(ObservationModel):
    """
    Simple discrete observation model that takes an observation matrix of the
    form (num_states x num_observations) containing P(observation | state).

    Parameters
    ----------
    observation_probabilities : numpy array
        Observation probabilities as a 2D array of shape (num_observations,
        num_states). Has to sum to 1 over the second axis, since it
        represents P(observation | state).

    Examples
    --------
    Assuming two states and three observation types, instantiate a discrete
    observation model:

    >>> om = DiscreteObservationModel(np.array([[0.1, 0.5, 0.4],
    ...                                         [0.7, 0.2, 0.1]]))
    >>> om  # doctest: +ELLIPSIS
    <madmom.ml.hmm.DiscreteObservationModel object at 0x...>

    If the probabilities do not sum to 1, it throws a ValueError:

    >>> om = DiscreteObservationModel(np.array([[0.5, 0.5, 0.5],
    ...                                         [0.5, 0.5, 0.5]]))
    ... # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    ValueError: Not a probability distribution.

    """

    def __init__(self, observation_probabilities):
        # check that it is a probability distribution
        if not np.allclose(observation_probabilities.sum(axis=1), 1):
            raise ValueError('Not a probability distribution.')
        # instantiate an ObservationModel
        super(DiscreteObservationModel, self).__init__(
            np.arange(observation_probabilities.shape[0], dtype=np.uint32))
        # save the observation probabilities
        self.observation_probabilities = observation_probabilities

    def densities(self, observations):
        """
        Densities of the observations.

        Parameters
        ----------
        observations : numpy array
            Observations.

        Returns
        -------
        numpy array
            Densities of the observations.

        """
        return self.observation_probabilities[:, observations].T

    def log_densities(self, observations):
        """
        Log densities of the observations.

        Parameters
        ----------
        observations : numpy array
            Observations.

        Returns
        -------
        numpy array
            Log densities of the observations.

        """
        return np.log(self.densities(observations))


class HiddenMarkovModel(object):
    """
    Hidden Markov Model

    To search for the best path through the state space with the Viterbi
    algorithm, the following parameters must be defined.

    Parameters
    ----------
    transition_model : :class:`TransitionModel` instance
        Transition model.
    observation_model : :class:`ObservationModel` instance
        Observation model.
    initial_distribution : numpy array, optional
        Initial state distribution; if 'None' a uniform distribution is
        assumed.

    Examples
    --------
    Create a simple HMM with two states and three observation types. The
    initial distribution is uniform.

    >>> tm = TransitionModel.from_dense([0, 1, 0, 1], [0, 0, 1, 1],
    ...                                 [0.7, 0.3, 0.6, 0.4])
    >>> om = DiscreteObservationModel(np.array([[0.2, 0.3, 0.5],
    ...                                         [0.7, 0.1, 0.2]]))
    >>> hmm = HiddenMarkovModel(tm, om)

    Now we can decode the most probable state sequence and get the
    log-probability of the sequence

    >>> seq, log_p = hmm.viterbi([0, 0, 1, 1, 0, 0, 0, 2, 2])
    >>> log_p  #  doctest: +ELLIPSIS
    -12.87...
    >>> seq
    array([1, 1, 0, 0, 1, 1, 1, 0, 0], dtype=uint32)

    Compute the forward variables:

    >>> hmm.forward([0, 0, 1, 1, 0, 0, 0, 2, 2])
    array([[ 0.34667,  0.65333],
           [ 0.33171,  0.66829],
           [ 0.83814,  0.16186],
           [ 0.86645,  0.13355],
           [ 0.38502,  0.61498],
           [ 0.33539,  0.66461],
           [ 0.33063,  0.66937],
           [ 0.81179,  0.18821],
           [ 0.84231,  0.15769]])
    """

    def __init__(self, transition_model, observation_model,
                 initial_distribution=None):
        # save the parameters
        self.transition_model = transition_model
        self.observation_model = observation_model
        if initial_distribution is None:
            initial_distribution = np.ones(transition_model.num_states,
                                           dtype=np.float) / \
                                   transition_model.num_states
        if not np.allclose(initial_distribution.sum(), 1):
            raise ValueError('Initial distribution is not a probability '
                             'distribution.')
        self.initial_distribution = initial_distribution
        # attributes needed for stateful processing (i.e. forward_step())
        self._prev = self.initial_distribution.copy()

    def __getstate__(self):
        # copy everything to a pickleable object
        state = self.__dict__.copy()
        # do not pickle attributes needed for stateful processing
        state.pop('_prev', None)
        return state

    def __setstate__(self, state):
        # restore pickled instance attributes
        self.__dict__.update(state)
        # add non-pickled attributes needed for stateful processing
        self._prev = self.initial_distribution.copy()

    def reset(self, initial_distribution=None):
        """
        Reset the HMM to its initial state.

        Parameters
        ----------
        initial_distribution : numpy array, optional
            Reset to this initial state distribution.

        """
        # reset initial state distribution
        self._prev = initial_distribution or self.initial_distribution.copy()

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def viterbi(self, observations):
        """
        Determine the best path with the Viterbi algorithm.

        Parameters
        ----------
        observations : numpy array
            Observations to decode the optimal path for.

        Returns
        -------
        path : numpy array
            Best state-space path sequence.
        log_prob : float
            Corresponding log probability.

        """
        # transition model stuff
        tm = self.transition_model
        cdef uint32_t [::1] tm_states = tm.states
        cdef uint32_t [::1] tm_pointers = tm.pointers
        cdef double [::1] tm_probabilities = tm.log_probabilities
        cdef unsigned int num_states = tm.num_states

        # observation model stuff
        om = self.observation_model
        cdef unsigned int num_observations = len(observations)
        cdef uint32_t [::1] om_pointers = om.pointers
        cdef double [:, ::1] om_densities = om.log_densities(observations)

        # current viterbi variables
        cdef double [::1] current_viterbi = np.empty(num_states,
                                                     dtype=np.float)

        # previous viterbi variables, init with the initial state distribution
        cdef double [::1] previous_viterbi = np.log(self.initial_distribution)

        # back-tracking pointers
        cdef uint32_t [:, ::1] bt_pointers = np.empty((num_observations,
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

        # raise warning if the sequence has -inf probability
        if np.isinf(log_probability):
            warnings.warn('-inf log probability during Viterbi decoding '
                          'cannot find a valid path', RuntimeWarning)
            # return empty path sequence
            return np.empty(0, dtype=np.uint32), log_probability

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
    @cython.initializedcheck(False)
    def forward(self, observations, reset=True):
        """
        Compute the forward variables at each time step. Instead of computing
        in the log domain, we normalise at each step, which is faster for the
        forward algorithm.

        Parameters
        ----------
        observations : numpy array, shape (num_frames, num_densities)
            Observations to compute the forward variables for.
        reset : bool, optional
            Reset the HMM to its inital state before computing the forward
            variables.

        Returns
        -------
        numpy array, shape (num_observations, num_states)
            Forward variables.

        """
        # transition model stuff
        tm = self.transition_model
        cdef uint32_t [::1] tm_states = tm.states
        cdef uint32_t [::1] tm_pointers = tm.pointers
        cdef double [::1] tm_probabilities = tm.probabilities
        cdef unsigned int num_states = tm.num_states

        # observation model stuff
        om = self.observation_model
        cdef uint32_t [::1] om_pointers = om.pointers
        cdef double [:, ::1] om_densities = om.densities(observations)
        cdef unsigned int num_observations = len(om_densities)

        # reset HMM
        if reset:
            self.reset()

        # forward variables
        cdef double[::1] fwd_prev = self._prev
        cdef double[:, ::1] fwd = np.zeros((num_observations, num_states),
                                           dtype=np.float)

        # define counters etc.
        cdef unsigned int prev_pointer, frame, state
        cdef double prob_sum, norm_factor

        # iterate over all observations
        for frame in range(num_observations):
            # keep track of the normalisation sum
            prob_sum = 0
            # iterate over all states
            for state in range(num_states):
                # sum over all possible predecessors
                for prev_pointer in range(tm_pointers[state],
                                          tm_pointers[state + 1]):
                    fwd[frame, state] += (fwd_prev[tm_states[prev_pointer]] *
                                          tm_probabilities[prev_pointer])
                # multiply with the observation probability
                fwd[frame, state] *= om_densities[frame, om_pointers[state]]
                prob_sum += fwd[frame, state]
            # normalise
            norm_factor = 1. / prob_sum
            for state in range(num_states):
                fwd[frame, state] *= norm_factor
                # also save it as the previous variables for the next frame
                fwd_prev[state] = fwd[frame, state]

        # return the forward variables
        return np.asarray(fwd)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    def forward_generator(self, observations, block_size=None):
        """
        Compute the forward variables at each time step. Instead of computing
        in the log domain, we normalise at each step, which is faster for
        the forward algorithm. This function is a generator that yields the
        forward variables for each time step individually to save memory.
        The observation densities are computed block-wise to save Python calls
        in the inner loops.

        Parameters
        ----------
        observations : numpy array
            Observations to compute the forward variables for.
        block_size : int, optional
            Block size for the block-wise computation of observation densities.
            If 'None', all observation densities will be computed at once.

        Yields
        ------
        numpy array, shape (num_states,)
            Forward variables.

        """
        # transition model stuff
        tm = self.transition_model
        cdef uint32_t [::1] tm_states = tm.states
        cdef uint32_t [::1] tm_ptrs = tm.pointers
        cdef double [::1] tm_probabilities = tm.probabilities
        cdef unsigned int num_states = tm.num_states

        # observation model stuff
        om = self.observation_model
        cdef unsigned int num_observations = len(observations)
        cdef uint32_t [::1] om_pointers = om.pointers
        cdef double [:, ::1] om_densities

        # forward variables
        cdef double[::1] fwd_cur = np.zeros(num_states, dtype=np.float)
        cdef double[::1] fwd_prev = self.initial_distribution.copy()

        # define counters etc.
        cdef unsigned int prev_pointer, state
        cdef unsigned int obs_start, obs_end, frame, block_sz
        cdef double prob_sum, norm_factor

        # keep track which observations om_densities currently contains
        # obs_start is the first observation index, obs_end the last one
        obs_start = 0
        obs_end = 0

        # compute everything at once if block_size was set to None
        block_sz = num_observations if block_size is None else block_size

        # iterate over all observations
        for frame in range(num_observations):
            # keep track of the normalisation sum
            prob_sum = 0

            # initialise forward variables
            fwd_cur[:] = 0.0

            # check if we have to compute another block of observation densities
            if frame >= obs_end:
                obs_start = frame
                obs_end = obs_start + block_sz
                om_densities = om.densities(observations[obs_start:obs_end])

            # iterate over all states
            for state in range(num_states):
                # sum over all possible predecessors
                for prev_pointer in range(tm_ptrs[state], tm_ptrs[state + 1]):
                    fwd_cur[state] += fwd_prev[tm_states[prev_pointer]] * \
                                      tm_probabilities[prev_pointer]
                # multiply with the observation probability
                fwd_cur[state] *= om_densities[frame - obs_start,
                                               om_pointers[state]]
                prob_sum += fwd_cur[state]
            # normalise
            norm_factor = 1. / prob_sum
            for state in range(num_states):
                fwd_cur[state] *= norm_factor

            # yield the current forward variables
            yield np.asarray(fwd_cur).copy()

            fwd_cur, fwd_prev = fwd_prev, fwd_cur

# alias
HMM = HiddenMarkovModel
