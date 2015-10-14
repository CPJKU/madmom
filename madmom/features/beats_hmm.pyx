# encoding: utf-8

"""
This file contains HMM state space, transition and observation models used for
beat and downbeat tracking.

Please note that (almost) everything within this module is discretised to
integer values because of performance reasons.

If you want to change this module and use it interactively, use pyximport.

>>> import pyximport
>>> pyximport.install(reload_support=True,
                      setup_args={'include_dirs': np.get_include()})

"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport log, exp

from madmom.ml.hmm import TransitionModel, ObservationModel


class BeatTrackingStateSpace(object):
    """
    State space for beat tracking with a HMM.

    """

    def __init__(self, min_interval, max_interval, num_tempo_states=None):
        """
        Construct a new BeatTrackingStateSpace.

        :param min_interval:     minimum tempo (i.e. inter beat interval) to
                                 model [float]
        :param max_interval:     maximum tempo (i.e. inter beat interval) to
                                 model [float]
        :param num_tempo_states: number of tempo states [int] (if set, limit
                                 the number of states and use a log spacing,
                                 otherwise use a linear spacing defined by the
                                 tempo range)

        "An efficient state space model for joint tempo and meter tracking"
        Florian Krebs, Sebastian Böck and Gerhard Widmer
        Proceedings of the 16th International Society for Music Information
        Retrieval Conference (ISMIR), 2015.

        """
        # use a linear spacing as default
        states = np.arange(np.round(min_interval), np.round(max_interval) + 1)
        # if num_tempo_states is given (and smaller than the number of states
        # of the linear spacing) use a log spacing and limit the number of
        # states to the given value
        if num_tempo_states is not None and num_tempo_states < len(states):
            # we must approach num_tempo_states iteratively
            num_log_states = num_tempo_states
            states = []
            while len(states) < num_tempo_states:
                states = np.logspace(np.log2(min_interval),
                                     np.log2(max_interval),
                                     num_log_states, base=2)
                # quantize to integer tempo states
                states = np.unique(np.round(states))
                num_log_states += 1
        self.beat_states = np.ascontiguousarray(states, dtype=np.uint32)
        # compute the position and tempo mapping
        self.position_mapping, self.tempo_mapping = self.compute_mapping()

    @property
    def num_states(self):
        """Number of states."""
        return np.sum(self.beat_states)

    @property
    def num_tempo_states(self):
        """Number of tempo states."""
        return len(self.beat_states)

    @property
    def first_beat_positions(self):
        """First state for each tempo."""
        return np.cumsum(np.r_[0, self.beat_states[:-1]]).astype(np.uint32)

    @property
    def last_beat_positions(self):
        """Last state for each tempo."""
        return np.cumsum(self.beat_states).astype(np.uint32) - 1

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def compute_mapping(self):
        """
        Compute the mapping from state numbers to position and tempo states.

        :return: tuple with (position_mapping, tempo_mapping)

        """
        # counters etc.
        cdef unsigned int tempo_state, first_beat, last_beat
        cdef unsigned int num_states = np.sum(self.beat_states)
        cdef float pos, num_beat_states

        # mapping arrays from state numbers to tempo / position
        cdef unsigned int [::1] tempo = \
            np.empty(num_states, dtype=np.uint32)
        cdef double [::1] position = \
            np.empty(num_states, dtype=np.float)
        # cache variables
        cdef unsigned int [::1] beat_states = \
            self.beat_states
        cdef unsigned int [::1] first_beat_positions = \
            self.first_beat_positions
        cdef unsigned int [::1] last_beat_positions = \
            self.last_beat_positions
        # loop over all tempi
        for tempo_state in range(self.num_tempo_states):
            # first and last beat (exclusive) for tempo
            first_beat = first_beat_positions[tempo_state]
            last_beat = last_beat_positions[tempo_state]
            # number of beats for tempo
            num_beat_states = float(beat_states[tempo_state])
            # reset position counter
            pos = 0
            for state in range(first_beat, last_beat + 1):
                # tempo state mapping
                tempo[state] = tempo_state
                # position inside beat mapping
                position[state] = pos / num_beat_states
                pos += 1
        # return the mappings
        return np.asarray(position), np.asarray(tempo)

    def position(self, state):
        """
        Position (inside one beat) for a given state sequence.

        :param state: state (sequence) [int or numpy array]
        :return:      corresponding beat state sequence

        """
        return self.position_mapping[state]

    def tempo(self, state):
        """
        Tempo (i.e. inter beat interval) for a given state sequence.

        :param state: state (sequence) [int or numpy array]
        :return:      corresponding tempo state sequence

        """
        return self.tempo_mapping[state]


class BeatTrackingTransitionModel(TransitionModel):
    """
    Transition model for beat tracking with a HMM.

    """

    def __init__(self, state_space, transition_lambda):
        """
        Construct a new BeatTrackingTransitionModel.

        :param state_space:       BeatTrackingStateSpace instance
        :param transition_lambda: lambda for the exponential tempo change
                                  distribution (higher values prefer a constant
                                  tempo over a tempo change from one beat to
                                  the next one)

        "An efficient state space model for joint tempo and meter tracking"
        Florian Krebs, Sebastian Böck and Gerhard Widmer
        Proceedings of the 16th International Society for Music Information
        Retrieval Conference (ISMIR), 2015.

        """
        # save attributes
        self.state_space = state_space
        self.transition_lambda = np.asarray(transition_lambda, dtype=np.float)
        # compute the transitions
        transitions = self.make_sparse(*self.compute_transitions())
        # instantiate a TransitionModel with the transitions
        super(BeatTrackingTransitionModel, self).__init__(*transitions)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def compute_transitions(self):
        """
        Compute the transitions (i.e. the probabilities to move from any state
        to another one) and return them in a dense format understood by
        'make_sparse()'.

        :return: tuple with (states, prev_states, probabilities)

        """
        # cache variables
        cdef unsigned int [::1] beat_states = self.state_space.beat_states
        cdef double transition_lambda = self.transition_lambda
        # number of tempo & total states
        cdef unsigned int num_tempo_states = self.state_space.num_tempo_states
        cdef unsigned int num_states = self.state_space.num_states
        # counters etc.
        cdef unsigned int state, prev_state, old_tempo, new_tempo
        cdef double ratio, u, prob, prob_sum
        cdef double threshold = np.spacing(1)

        # to determine the number of transitions, we need to determine the
        # number of tempo change transitions first; also compute their
        # probabilities for later use

        # tempo changes can only occur at the beginning of a beat
        # transition matrix for the tempo changes
        cdef double [:, ::1] trans_prob = np.zeros((num_tempo_states,
                                                    num_tempo_states),
                                                   dtype=np.float)
        # iterate over all tempo states
        for old_tempo in range(num_tempo_states):
            # reset probability sum
            prob_sum = 0
            # compute transition probabilities to all other tempo states
            for new_tempo in range(num_tempo_states):
                # compute the ratio of the two tempi
                ratio = beat_states[new_tempo] / float(beat_states[old_tempo])
                # compute the probability for the tempo change following an
                # exponential distribution
                prob = exp(-transition_lambda * abs(ratio - 1))
                # keep only transition probabilities > threshold
                if prob > threshold:
                    # save the probability
                    trans_prob[old_tempo, new_tempo] = prob
                    # collect normalization data
                    prob_sum += prob
            # normalize the tempo transitions to other tempi
            for new_tempo in range(num_tempo_states):
                trans_prob[old_tempo, new_tempo] /= prob_sum

        # number of tempo transitions (= non-zero probabilities)
        cdef unsigned int num_tempo_transitions = \
            len(np.nonzero(trans_prob)[0])

        # apart from the very beginning of a beat, the tempo stays the same,
        # thus the number of transitions is equal to the total number of states
        # plus the number of tempo transitions minus the number of tempo states
        # since these transitions are already included in the tempo transitions
        cdef int num_transitions = num_states + num_tempo_transitions - \
                                   num_tempo_states
        # arrays for transition matrix creation
        cdef unsigned int [::1] states = \
            np.empty(num_transitions, dtype=np.uint32)
        cdef unsigned int [::1] prev_states = \
            np.empty(num_transitions, dtype=np.uint32)
        # init the probabilities with ones, so we have to care only about the
        # probabilities of the tempo transitions
        cdef double [::1] probabilities = \
            np.ones(num_transitions, dtype=np.float)

        # cache first and last positions
        cdef unsigned int [::1] first_beat_positions = \
            self.state_space.first_beat_positions
        cdef unsigned int [::1] last_beat_positions =\
            self.state_space.last_beat_positions
        # state counter
        cdef int i = 0
        # loop over all tempi
        for new_tempo in range(num_tempo_states):
            # generate all transitions from other tempi
            for old_tempo in range(num_tempo_states):
                # but only if it is a probable transition
                if trans_prob[old_tempo, new_tempo] != 0:
                    # generate a transition
                    prev_states[i] = last_beat_positions[old_tempo]
                    states[i] = first_beat_positions[new_tempo]
                    probabilities[i] = trans_prob[old_tempo, new_tempo]
                    # increase counter
                    i += 1
            # transitions within the same tempo
            for prev_state in range(first_beat_positions[new_tempo],
                                    last_beat_positions[new_tempo]):
                # generate a transition with probability 1
                prev_states[i] = prev_state
                states[i] = prev_state + 1
                # Note: skip setting the probability here, since they were
                #       initialised with 1
                # increase counter
                i += 1
        # return the arrays
        return states, prev_states, probabilities


class BeatTrackingObservationModel(ObservationModel):
    """
    Observation model for beat tracking with a HMM.

    """

    def __init__(self, state_space, observation_lambda,
                 norm_observations=False):
        """
        Construct a new BeatTrackingDynamicObservationModel.

        :param state_space:        BeatTrackingStateSpace instance
        :param observation_lambda: split one beat period into N parts, the
                                   first representing beat states and the
                                   remaining non-beat states
        :param norm_observations:  normalize the observations

        "A multi-model approach to beat tracking considering heterogeneous
         music styles"
        Sebastian Böck, Florian Krebs and Gerhard Widmer
        Proceedings of the 15th International Society for Music Information
        Retrieval Conference (ISMIR), 2014

        """
        self.observation_lambda = observation_lambda
        self.norm_observations = norm_observations
        # compute observation pointers
        # always point to the non-beat densities
        pointers = np.ones(state_space.num_states, dtype=np.uint32)
        # unless they are in the beat range of the state space
        border = 1. / observation_lambda
        beat_idx = state_space.position(np.arange(state_space.num_states,
                                                  dtype=np.int)) < border
        pointers[beat_idx] = 0
        # instantiate a ObservationModel with the pointers
        super(BeatTrackingObservationModel, self).__init__(pointers)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def log_densities(self, float [::1] observations):
        """
        Computes the log densities of the observations.

        :param observations: observations (i.e. activations of the NN)
        :return:             log densities of the observations

        """
        # init variables
        cdef unsigned int i
        cdef unsigned int num_observations = len(observations)
        cdef float observation_lambda = self.observation_lambda
        # norm observations
        if self.norm_observations:
            observations /= np.max(observations)
        # init densities
        cdef double [:, ::1] log_densities = np.empty((num_observations, 2),
                                                      dtype=np.float)
        # define the observation densities
        for i in range(num_observations):
            log_densities[i, 0] = log(observations[i])
            log_densities[i, 1] = log((1. - observations[i]) /
                                      (observation_lambda - 1))
        # return the densities
        return np.asarray(log_densities)


class DownBeatTrackingStateSpace(object):
    """
    State space for down-beat tracking with a HMM.

    """

    def __init__(self, min_intervals, max_intervals, num_tempo_states=None):
        """
        Construct a new BeatTrackingStateSpace (basically a stack of
        BeatTrackingStateSpaces).

        :param min_intervals:    list or array with minimum tempi (i.e. inter
                                 beat intervals) to model [float]
        :param max_intervals:    list or array with maximum tempi (i.e. inter
                                 beat intervals) to model [float]
        :param num_tempo_states: list or array with corresponding number of
                                 tempo states [int] (if set, limit the number
                                 of states and use a log spacing, otherwise use
                                 a linear spacing defined by the tempo range)

        "An efficient state space model for joint tempo and meter tracking"
        Florian Krebs, Sebastian Böck and Gerhard Widmer
        Proceedings of the 16th International Society for Music Information
        Retrieval Conference (ISMIR), 2015.

        """
        if num_tempo_states is None:
            num_tempo_states = [None] * len(min_intervals)
        # for each pattern, compute a beat state space
        state_spaces = []
        enum = enumerate(zip(min_intervals, max_intervals, num_tempo_states))
        for pattern, (min_, max_, num_) in enum:
            # create a BeatTrackingStateSpace and append it to the list
            state_spaces.append(BeatTrackingStateSpace(min_, max_, num_))
        self.pattern_state_spaces = state_spaces
        # define mappings
        self.position_mapping = \
            np.hstack([st.position(np.arange(st.num_states, dtype=np.int))
                       for st in self.pattern_state_spaces])
        self.tempo_mapping = \
            np.hstack([st.tempo(np.arange(st.num_states, dtype=np.int))
                       for st in self.pattern_state_spaces])
        self.pattern_mapping = \
            np.hstack([np.repeat(i, st.num_states)
                       for i, st in enumerate(self.pattern_state_spaces)])
        self.beat_states = [st.beat_states for st in self.pattern_state_spaces]

    @property
    def num_states(self):
        """Number of states."""
        return int(sum([st.num_states for st in self.pattern_state_spaces]))

    @property
    def num_tempo_states(self):
        """Number of tempo states for each pattern."""
        return [len(t) for t in self.beat_states]

    @property
    def num_patterns(self):
        """Number of rhythmic patterns"""
        return len(self.beat_states)

    def position(self, state):
        """
        Position (inside one bar) for a given state sequence.

        :param state: state (sequence) [int or numpy array]
        :return:      corresponding beat state sequence

        """
        return self.position_mapping[state]

    def tempo(self, state):
        """
        Tempo for a given state sequence.

        :param state: state (sequence) [int or numpy array]
        :return:      corresponding tempo state sequence

        """
        return self.tempo_mapping[state]

    def pattern(self, state):
        """
        Pattern for the given state sequence.

        :param state: state (sequence) [int or numpy array]
        :return:      corresponding pattern state sequence

        """
        return self.pattern_mapping[state]


class DownBeatTrackingTransitionModel(TransitionModel):
    """
    Transition model for down-beat tracking with a HMM.

    """

    def __init__(self, state_space, transition_lambda):
        """
        Construct a new DownBeatTrackingTransitionModel.

        Instead of modelling a single pattern (as BeatTrackingTransitionModel),
        it allows multiple patterns. It basically accepts the same arguments as
        the BeatTrackingTransitionModel, but everything as lists, with the list
        entries at the same position corresponding to one (rhythmic) pattern.

        :param state_space:       DownBeatTrackingStateSpace
        :param transition_lambda: (list with) lambda(s) for the exponential
                                  tempo change distribution of the patterns
                                  (higher values prefer a constant tempo over
                                  a tempo change from one bar to the next one)
                                  If a single value is given, the same value
                                  is assumed for all patterns.

        "An efficient state space model for joint tempo and meter tracking"
        Florian Krebs, Sebastian Böck and Gerhard Widmer
        Proceedings of the 16th International Society for Music Information
        Retrieval Conference (ISMIR), 2015.

        """
        # expand the transition lambda to a list if needed, i.e. use the same
        # value for all patterns
        if not isinstance(transition_lambda, list):
            transition_lambda = [transition_lambda] * state_space.num_patterns
        # check if all lists have the same length
        if not state_space.num_patterns == len(transition_lambda):
            raise ValueError("number of patterns of the 'state_space' and the "
                             "length 'transition_lambda' must be the same")
        # save the given arguments
        self.beat_states = state_space.beat_states
        self.transition_lambda = transition_lambda
        # compute the transitions for each pattern and stack them
        enum = enumerate(zip(state_space.pattern_state_spaces,
                             transition_lambda))
        for pattern, (state_space, transition_lambda) in enum:
            # create a BeatTrackingTransitionModel
            tm = BeatTrackingTransitionModel(state_space, transition_lambda)
            seq = np.arange(tm.num_states, dtype=np.int)
            # set/update the probabilities, states and pointers
            if pattern == 0:
                # for the first pattern, just set the TM arrays
                states = tm.states
                pointers = tm.pointers
                probabilities = tm.probabilities
            else:
                # for all consecutive patterns, stack the TM arrays after
                # applying an offset
                # Note: len(pointers) = len(states) + 1, because of the CSR
                #       format of the TM (please see ml.hmm.TransitionModel)
                # states: offset = length of the pointers - 1
                states = np.hstack((states, tm.states + len(pointers) - 1))
                # pointers: offset = current maximum of the pointers
                #           start = tm.pointers[1:]
                pointers = np.hstack((pointers, tm.pointers[1:] +
                                      max(pointers)))
                # probabilities: just stack them
                probabilities = np.hstack((probabilities, tm.probabilities))
        # instantiate a TransitionModel with the transition arrays
        transitions = states, pointers, probabilities
        super(DownBeatTrackingTransitionModel, self).__init__(*transitions)


class GMMDownBeatTrackingObservationModel(ObservationModel):
    """
    Observation model for GMM based beat tracking with a HMM.

    """

    def __init__(self, gmms, transition_model, norm_observations):
        """
        Construct a observation model instance using Gaussian Mixture Models
        (GMMs).

        :param gmms:              list with fitted GMM(s), one entry per
                                  rhythmic pattern
        :param transition_model:  DownBeatTrackingTransitionModel instance
        :param norm_observations: normalize the observations

        "Rhythmic Pattern Modeling for Beat and Downbeat Tracking in Musical
         Audio"
        Florian Krebs, Sebastian Böck and Gerhard Widmer
        Proceedings of the 15th International Society for Music Information
        Retrieval Conference (ISMIR), 2013

        """
        self.gmms = gmms
        self.transition_model = transition_model
        self.norm_observations = norm_observations
        # define the pointers of the log densities
        pointers = np.zeros(transition_model.num_states, dtype=np.uint32)
        states = np.arange(self.transition_model.num_states)
        pattern = self.transition_model.pattern(states)
        position = self.transition_model.position(states)
        # Note: the densities of all GMMs are just stacked on top of each
        #       other, so we have to to keep track of the total number of GMMs
        densities_idx_offset = 0
        for p in range(len(gmms)):
            # number of fitted GMMs for this pattern
            num_gmms = len(gmms[p])
            # distribute the observation densities defined by the GMMs
            # uniformly across the entire state space (for this pattern)
            # since the densities are just stacked, add the offset
            pointers[pattern == p] = (position[pattern == p] * num_gmms +
                                      densities_idx_offset)
            # increase the offset by the number of GMMs
            densities_idx_offset += num_gmms
        # instantiate a ObservationModel with the pointers
        super(GMMDownBeatTrackingObservationModel, self).__init__(pointers)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def log_densities(self, observations):
        """
        Computes the log densities of the observations using (a) GMM(s).

        :param observations: observations (i.e. activations of the NN)
        :return:             log densities of the observations

        """
        # counter, etc.
        cdef unsigned int i, j
        cdef unsigned int num_observations = len(observations)
        cdef unsigned int num_patterns = len(self.gmms)
        cdef unsigned int num_gmms = 0
        # norm observations
        if self.norm_observations:
            observations /= np.max(observations)
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
                log_densities[:, c] = self.gmms[i][j].score(observations)
                c += 1
        # return the densities
        return log_densities
