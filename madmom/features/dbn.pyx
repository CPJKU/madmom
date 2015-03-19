# encoding: utf-8
"""
This file contains dynamic Bayesian network (DBN) functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

If you want to change this module and use it interactively, use pyximport.

>>> import pyximport
>>> pyximport.install(reload_support=True,
                      setup_args={'include_dirs': np.get_include()})

"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport log

# parallel processing stuff
from cython.parallel cimport prange
NUM_THREADS = 1

cdef extern from "math.h":
    float INFINITY


# transition_model stuff
class TransitionModel(object):
    """
    Transition model class for a DBN.

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
        processing of the Viterbi of the DBN.

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

# DBN stuff
class DynamicBayesianNetwork(object):
    """
    Dynamic Bayesian network.

    To search for the best path through the state space with the Viterbi
    algorithm, a `transition_model`, `observation_model` and
    `initial_distribution` must be defined.

    """

    def __init__(self, transition_model, observation_model,
                 initial_distribution=None, num_threads=NUM_THREADS):
        """
        Construct a new Dynamic Bayesian network.

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


class BeatTrackingTransitionModel(TransitionModel):
    """
    Transition model for beat tracking with a DBN.

    """

    def __init__(self, num_beat_states, tempo_states, tempo_change_probability):
        """
        Construct a new BeatTrackingTransitionModel.

        :param num_beat_states:          number of beat states for one beat
                                         period
        :param tempo_states:             array with tempo states (number of
                                         beat states to progress from one
                                         observation value to the next one)
        :param tempo_change_probability: probability of a tempo change from
                                         one observation to the next one

        "A multi-model approach to beat tracking considering heterogeneous
         music styles"
        Sebastian Böck, Florian Krebs and Gerhard Widmer
        Proceedings of the 15th International Society for Music Information
        Retrieval Conference (ISMIR), 2014

        """
        # save variables
        self.num_beat_states = num_beat_states
        self.tempo_states = tempo_states
        self.tempo_change_probability = tempo_change_probability
        # compute the transitions
        transitions = self.make_sparse(*self.compute_transitions())
        # instantiate a BeatTrackingTransitionModel with the transitions
        super(BeatTrackingTransitionModel, self).__init__(*transitions)

    @property
    def num_tempo_states(self):
        """Number of tempo states."""
        return len(self.tempo_states)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def compute_transitions(self):
        """
        Compute the transitions (i.e. the log probabilities to move from any
        states to another one) and return them in a format understood by
        'make_sparse()'.

        :return: tuple with (states, pointers, log_probabilities)

        """
        # number of tempo & total states
        cdef unsigned int num_beat_states = self.num_beat_states
        cdef unsigned int num_tempo_states = len(self.tempo_states)
        cdef unsigned int num_states = num_beat_states * num_tempo_states
        # transition probabilities
        cdef double same_tempo_prob = log(1. - self.tempo_change_probability)
        cdef double change_tempo_prob = log(0.5 * self.tempo_change_probability)
        # counters etc.
        cdef unsigned int state, prev_state, beat_state, tempo_state, tempo
        # number of transition states
        # num_tempo_states * 3 because every state has a transition from the
        # same tempo and from the slower and faster one, -2 because the slowest
        # and the fastest tempi can't have transition_model from outside the tempo
        # range
        cdef int num_transition_states = (num_beat_states *
                                          (num_tempo_states * 3 - 2))
        # arrays for transition_model matrix creation
        cdef unsigned int [::1] states = \
            np.empty(num_transition_states, np.uint32)
        cdef unsigned int [::1] prev_states = \
            np.empty(num_transition_states, np.uint32)
        cdef double [::1] log_probabilities = \
            np.empty(num_transition_states, np.float)
        cdef int i = 0
        # loop over all states
        for state in range(num_states):
            # position inside beat & tempo
            beat_state = state % num_beat_states
            tempo_state = state / num_beat_states
            # get the corresponding tempo
            tempo = self.tempo_states[tempo_state]
            # for each state check the 3 possible transition_model
            # previous state with same tempo
            # Note: we add num_beat_states before the modulo operation so
            #       that it can be computed in C (which is faster)
            prev_state = ((beat_state + num_beat_states - tempo) %
                          num_beat_states +
                          (tempo_state * num_beat_states))
            # probability for transition from same tempo
            states[i] = state
            prev_states[i] = prev_state
            log_probabilities[i] = same_tempo_prob
            i += 1
            # transition from slower tempo
            if tempo_state > 0:
                # previous state with slower tempo
                prev_state = ((beat_state + num_beat_states - (tempo - 1)) %
                              num_beat_states +
                              ((tempo_state - 1) * num_beat_states))
                # probability for transition from slower tempo
                states[i] = state
                prev_states[i] = prev_state
                log_probabilities[i] = change_tempo_prob
                i += 1
            # transition from faster tempo
            if tempo_state < num_tempo_states - 1:
                # previous state with faster tempo
                # Note: we add num_beat_states before the modulo operation
                #       so that it can be computed in C (which is faster)
                prev_state = ((beat_state + num_beat_states - (tempo + 1)) %
                              num_beat_states +
                              ((tempo_state + 1) * num_beat_states))
                # probability for transition from faster tempo
                states[i] = state
                prev_states[i] = prev_state
                log_probabilities[i] = change_tempo_prob
                i += 1
        # return a TransitionModel
        return states, prev_states, log_probabilities

    # mapping functions
    def position(self, state):
        """
        Position (within the beat) for a given state sequence.

        :param state: state (sequence) [int or numpy array]
        :return:      corresponding beat state sequence

        """
        # return a value in the range of 0..1
        return state % self.num_beat_states / float(self.num_beat_states)

    def tempo(self, state):
        """
        Tempo for a given state sequence.

        :param state: state (sequence) [int or numpy array]
        :return:      corresponding tempo state sequence

        """
        # return the tempo state index
        return state // self.num_beat_states


class BeatTrackingObservationModel(ObservationModel):
    """
    Observation model for beat tracking with a DBN.

    """

    def __init__(self, transition_model, observation_lambda,
                 norm_observations=False):
        """
        Construct a new BeatTrackingDynamicObservationModel.

        :param observation_lambda:       split one beat period into N parts,
                                         the first representing beat states
                                         and the remaining non-beat states
        :param norm_observations:        normalize the observation_model

        "A multi-model approach to beat tracking considering heterogeneous
         music styles"
        Sebastian Böck, Florian Krebs and Gerhard Widmer
        Proceedings of the 15th International Society for Music Information
        Retrieval Conference (ISMIR), 2014

        """
        self.observation_lambda = observation_lambda
        self.norm_observations = norm_observations
        # shortcut
        tm = transition_model
        # compute observation pointers
        # always point to the non-beat densities
        self.pointers = np.ones(tm.num_states, dtype=np.uint32)
        # unless they are in the beat range of the state space
        border = 1. / observation_lambda
        beat_idx = tm.position(np.arange(tm.num_states)) < border
        self.pointers[beat_idx] = 0
        # instantiate an ObservationModel
        # FIXME: we don't have log_densities for instantiation yet...
        # super(BeatTrackingObservationModel, self).__init__(None, pointers)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def compute_densities(self, float [::1] observations):
        """
        Compute the observation log densities.

        :param observations: observations (i.e. activations of the NN)
        :return:             log_densities

        """
        # init variables
        cdef unsigned int i
        cdef unsigned int num_observations = len(observations)
        cdef float observation_lambda = self.observation_lambda
        # init densities
        cdef double [:, ::1] log_densities = np.empty((num_observations, 2),
                                                      dtype=np.float)
        # define the observation densities
        for i in range(num_observations):
            log_densities[i, 0] = log(observations[i])
            log_densities[i, 1] = log((1. - observations[i]) /
                                      (observation_lambda - 1))
        # save the densities and return them
        self.log_densities = np.asarray(log_densities)
        return self.log_densities


# down-beat tracking stuff
class DownBeatTrackingTransitionModel(TransitionModel):
    """
    Transition model for down-beat tracking with a DBN.

    """
    def __init__(self, num_bar_states, tempo_states,
                 tempo_change_probabilities):
        """
        Construct a transition model instance suitable for down-beat tracking.

        DownBeatTrackingTransitionModel is an extension of the
        BeatTrackingTransitionModel. Instead of modelling a single pattern, it
        allows multiple patterns. It basically accepts the same arguments as
        the BeatTrackingTransitionModel, but everything as lists, with the list
        entries at the same position corresponding to one (rhythmic) pattern.

        :param num_bar_states:            list with number of bar states for
                                          one bar period
        :param tempo_states:              list with numpy arrays with tempo
                                          states (number of bar states to
                                          progress from one observation value
                                          to the next one)
        :param tempo_change_probabilities: list with probabilities of a tempo
                                          change from one observation to the
                                          next one

        "Rhythmic pattern modeling for beat and downbeat tracking in musical
         audio"
        Florian Krebs, Sebastian Böck, and Gerhard Widmer
        Proceedings of the 14th International Society for Music Information
        Retrieval Conference (ISMIR), 2013.

        """
        # instantiate an empty TransitionModel object
        super(DownBeatTrackingTransitionModel, self).__init__(None, None, None)
        # check if all lists have the same length
        if not (len(num_bar_states) == len(tempo_states) ==
            len(tempo_change_probabilities)):
            raise ValueError("'num_bar_states', 'tempo_states' and "
                             "'tempo_change_probabilities' must have the same "
                             "length")
        # save the given arguments
        self.num_bar_states = num_bar_states
        self.tempo_states = tempo_states
        self.tempo_change_probabilities = tempo_change_probabilities
        # for each pattern, compute the transitions
        for i, (bs, ts, tcp) in enumerate(zip(num_bar_states, tempo_states,
                                              tempo_change_probabilities)):
            # make sure that the tempo_states are contiguous in memory
            ts = np.ascontiguousarray(ts, dtype=np.int32)
            # create a BeatTrackingTransitionModel
            tm = BeatTrackingTransitionModel(bs, ts, tcp)
            seq = np.arange(tm.num_states)
            # set/update the probabilities, states and pointers
            if i == 0:
                # set TM arrays
                states = tm.states
                pointers = tm.pointers
                log_probabilities = tm.log_probabilities
                # internal mapping arrays
                self.position_mapping = tm.position(seq)
                self.tempo_mapping = tm.tempo(seq)
                self.pattern_mapping = np.repeat(i, tm.num_states)
            else:
                # update TM array
                states = np.hstack((states,
                                    tm.states + len(pointers) - 1))
                pointers = np.hstack((pointers,
                                      tm.pointers[1:] + max(pointers)))
                log_probabilities = np.hstack((log_probabilities,
                                               tm.log_probabilities))
                # update internal mapping arrays
                self.position_mapping = np.hstack((self.position_mapping,
                                                   tm.position(seq)))
                self.tempo_mapping = np.hstack((self.tempo_mapping,
                                                tm.tempo(seq)))
                self.pattern_mapping = np.hstack((self.pattern_mapping,
                                                  np.repeat(i, tm.num_states)))
        # instantiate a BeatTrackingTransitionModel with the transitions
        super(DownBeatTrackingTransitionModel, self).__init__(states,
                                                              pointers,
                                                              log_probabilities)

    @property
    def num_tempo_states(self):
        """Number of tempo states."""
        return [len(t) for t in self.tempo_states]

    @property
    def num_patterns(self):
        """Number of rhythmic patterns"""
        # use the length of any of the lists as number of patterns
        return len(self.tempo_states)

    def position(self, state_sequence):
        """
        Position (within the bar) for a given state sequence.

        :param state_sequence: given state sequence
        :return:               corresponding bar state sequence

        """
        return self.position_mapping[state_sequence]

    def tempo(self, state_sequence):
        """
        Tempo for the given state sequence.

        :param state_sequence: given state sequence
        :return:               corresponding tempo state sequence

        """
        return self.tempo_mapping[state_sequence]

    def pattern(self, state_sequence):
        """
        Pattern for the given state sequence.

        :param state_sequence: given state sequence
        :return:               corresponding pattern state sequence

        """
        return self.pattern_mapping[state_sequence]


class GMMDownBeatTrackingObservationModel(ObservationModel):
    """
    Observation model for GMM based beat tracking with a DBN.

    """

    def __init__(self, gmms, transition_model, norm_observations):
        """
        Construct a observation model instance using Gaussian Mixture Models
        (GMMs).

        :param gmms:              list with fitted GMM(s), one entry per
                                  rhythmic pattern
        :param transition_model:  transition model
        :param norm_observations: normalize the observations

        """
        self.gmms = gmms
        self.transition_model = transition_model
        self.norm_observations = norm_observations
        # define the pointers of the log densities
        self.pointers = np.zeros(transition_model.num_states, dtype=np.uint32)
        states = np.arange(self.transition_model.num_states)
        pattern = self.transition_model.pattern(states)
        position = self.transition_model.position(states)
        densities_idx_offset = 0
        for p in range(len(gmms)):
            # number of fitted GMMs for this pattern
            num_gmms = len(gmms[p])
            # distribute the observation densities defined by the GMMs
            # uniformly across the entire state space (for this pattern)
            # Note: the densities of all GMMs are just stacked on top of each
            #       other, so we have to add an offset
            self.pointers[pattern == p] = (position[pattern == p] * num_gmms +
                                           densities_idx_offset)
            # increase the offset by the number of GMMs
            densities_idx_offset += num_gmms

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def compute_densities(self, observations):
        """
        Compute the observation log densities using (a) GMM(s).

        :param observations: observations (i.e. activations of the NN)
        :return:             log_densities

        """
        # counter, etc.
        cdef unsigned int i, j
        cdef unsigned int num_observations = len(observations)
        cdef unsigned int num_patterns = len(self.gmms)
        cdef unsigned int num_gmms = 0
        # maximum number of GMMs of all patterns
        for i in range(num_patterns):
            num_gmms += len(self.gmms[i])
        # init the densities
        log_densities = np.empty((num_observations, num_gmms), dtype=np.float)
        # define the observation densities
        cdef unsigned int c = 0
        for i in range(num_patterns):
            for j in range(len(self.gmms[i])):
                # get the predictions of each GMM for the observations
                # TODO: use a faster C version without sklearn!
                log_densities[:, c] = self.gmms[i][j].score(observations)
                c += 1
        # save the densities and return them
        self.log_densities = log_densities
        return self.log_densities
