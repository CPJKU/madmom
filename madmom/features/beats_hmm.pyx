# encoding: utf-8
# cython: embedsignature=True
"""
This module contains HMM state space, transition and observation models used
for beat and downbeat tracking.

Notes
-----
Please note that (almost) everything within this module is discretised to
integer values because of performance reasons.

If you want to change this module and use it interactively, use pyximport.

>>> import pyximport
>>> pyximport.install(reload_support=True,
                      setup_args={'include_dirs': np.get_include()})

"""

from __future__ import absolute_import, division, print_function

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport log, exp

from madmom.ml.hmm import TransitionModel, ObservationModel


# state spaces
class BeatStateSpace(object):
    """
    State space for beat tracking with a HMM.

    Parameters
    ----------
    min_interval : float
        Minimum interval to model.
    max_interval : float
        Maximum interval to model.
    num_intervals : int, optional
        Number of intervals to model; if set, limit the number of intervals
        and use a log spacing instead of the default linear spacing.

    References
    ----------
    .. [1] Florian Krebs, Sebastian Böck and Gerhard Widmer,
           "An Efficient State Space Model for Joint Tempo and Meter Tracking",
           Proceedings of the 16th International Society for Music Information
           Retrieval Conference (ISMIR), 2015.

    """

    def __init__(self, min_interval, max_interval, num_intervals=None):
        # per default, use a linear spacing of the tempi
        intervals = np.arange(np.round(min_interval),
                              np.round(max_interval) + 1)
        # if num_intervals is given (and smaller than the length of the linear
        # spacing of the intervals) use a log spacing and limit the number of
        # intervals to the given value
        if num_intervals is not None and num_intervals < len(intervals):
            # we must approach intervals iteratively
            num_log_tempi = num_intervals
            intervals = []
            while len(intervals) < num_intervals:
                intervals = np.logspace(np.log2(min_interval),
                                     np.log2(max_interval),
                                     num_log_tempi, base=2)
                # quantize to integer tempo states
                intervals = np.unique(np.round(intervals))
                num_log_tempi += 1
        # intervals to model
        self.intervals = np.ascontiguousarray(intervals, dtype=np.uint32)
        # define the position and interval states
        self.position = np.empty(self.num_states)
        self.interval = np.empty(self.num_states, dtype=np.uint32)
        idx = interval = 0
        for i in self.intervals:
            self.position[idx: idx + i] = np.linspace(0, 1, i, endpoint=False)
            self.interval[idx: idx + i] = interval
            # increase counters
            idx += i
            interval += 1

    @property
    def num_states(self):
        """Number of states."""
        return int(np.sum(self.intervals))

    @property
    def num_intervals(self):
        """Number of different intervals."""
        return len(self.intervals)

    @property
    def first_states(self):
        """First state for each interval."""
        return np.cumsum(np.r_[0, self.intervals[:-1]]).astype(np.uint32)

    @property
    def last_states(self):
        """Last state for each interval."""
        return np.cumsum(self.intervals).astype(np.uint32) - 1


class MultiPatternStateSpace(object):
    """
    State space for rhythmic pattern tracking with a HMM.

    A rhythmic pattern is modeled similar to :class:`BeatStateSpace`,
    but models multiple rhythmic patterns instead of a single beat. The
    pattern's length can span multiple beats (e.g. 3 or 4 beats).

    Parameters
    ----------
    min_intervals : list or numpy array
        Minimum intervals (i.e. rhythmic pattern length) to model.
    max_intervals : list or numpy array
        Maximum intervals (i.e. rhythmic pattern length) to model.
    num_intervals : list or numpy array, optional
        Corresponding number of intervals; if set, limit the number of
        intervals and use a log spacing instead of the default linear spacing.

    See Also
    --------
    :class:`BeatStateSpace`

    References
    ----------
    .. [1] Florian Krebs, Sebastian Böck and Gerhard Widmer,
           "An Efficient State Space Model for Joint Tempo and Meter Tracking",
           Proceedings of the 16th International Society for Music Information
           Retrieval Conference (ISMIR), 2015.

    """

    def __init__(self, min_intervals, max_intervals, num_intervals=None):
        if num_intervals is None:
            num_intervals = [None] * len(min_intervals)
        # for each pattern, compute a bar state space (i.e. a beat state space
        # which spans a complete bar)
        bar_state_spaces = []
        enum = enumerate(zip(min_intervals, max_intervals, num_intervals))
        for pattern, (min_, max_, num_) in enum:
            # create a BeatStateSpace and append it to the list
            bar_state_spaces.append(BeatStateSpace(min_, max_, num_))
        self.bar_state_spaces = bar_state_spaces
        # define the position, interval and pattern states
        self.position = \
            np.hstack([st.position[np.arange(st.num_states, dtype=np.int)]
                       for st in self.bar_state_spaces])
        self.interval = \
            np.hstack([st.interval[np.arange(st.num_states, dtype=np.int)]
                       for st in self.bar_state_spaces])
        self.pattern = \
            np.hstack([np.repeat(i, st.num_states)
                       for i, st in enumerate(self.bar_state_spaces)])

    @property
    def num_states(self):
        """Number of states."""
        return int(sum([st.num_states for st in self.bar_state_spaces]))

    @property
    def num_patterns(self):
        """Number of rhythmic patterns"""
        return len(self.bar_state_spaces)


# transition models
class BeatTransitionModel(TransitionModel):
    """
    Transition model for beat tracking with a HMM.

    Parameters
    ----------
    state_space : :class:`BeatStateSpace` instance
        BeatStateSpace instance.
    transition_lambda : float
        Lambda for the exponential tempo change distribution (higher values
        prefer a constant tempo from one beat to the next one).

    References
    ----------
    .. [1] Florian Krebs, Sebastian Böck and Gerhard Widmer,
           "An Efficient State Space Model for Joint Tempo and Meter Tracking",
           Proceedings of the 16th International Society for Music Information
           Retrieval Conference (ISMIR), 2015.

    """

    def __init__(self, state_space, transition_lambda):
        # save attributes
        self.state_space = state_space
        self.transition_lambda = np.asarray(transition_lambda, dtype=np.float)
        # compute the transitions
        transitions = self.make_sparse(*self.compute_transitions())
        # instantiate a TransitionModel with the transitions
        super(BeatTransitionModel, self).__init__(*transitions)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def compute_transitions(self):
        """
        Compute the transitions (i.e. the probabilities to move from any state
        to another one) and return them in a dense format understood by
        :func:`.ml.hmm.TransitionModel.make_sparse`.

        Returns
        -------
        states : numpy array
            Array with states (i.e. destination states).
        prev_states : numpy array
            Array with previous states (i.e. origination states).
        probabilities : numpy array
            Transition probabilities.

        """
        # cache variables
        # Note: convert all intervals to float here
        cdef float [::1] intervals = \
            self.state_space.intervals.astype(np.float32)
        cdef double transition_lambda = self.transition_lambda
        # number of tempo & total states
        cdef unsigned int num_intervals = self.state_space.num_intervals
        cdef unsigned int num_states = self.state_space.num_states
        # counters etc.
        cdef unsigned int state, prev_state, old_interval, new_interval
        cdef double ratio, u, prob, prob_sum
        cdef double threshold = np.spacing(1)

        # to determine the number of transitions, we need to determine the
        # number of tempo change transitions first; also compute their
        # probabilities for later use

        # tempo changes can only occur at the beginning of a beat
        # transition matrix for the tempo changes
        cdef double [:, ::1] trans_prob = np.zeros((num_intervals,
                                                    num_intervals),
                                                   dtype=np.float)
        # iterate over all tempo states
        for old_interval in range(num_intervals):
            # reset probability sum
            prob_sum = 0
            # compute transition probabilities to all other tempo states
            for new_interval in range(num_intervals):
                # compute the ratio of the two tempi
                ratio = intervals[new_interval] / intervals[old_interval]
                # compute the probability for the tempo change following an
                # exponential distribution
                prob = exp(-transition_lambda * abs(ratio - 1))
                # keep only transition probabilities > threshold
                if prob > threshold:
                    # save the probability
                    trans_prob[old_interval, new_interval] = prob
                    # collect normalization data
                    prob_sum += prob
            # normalize the tempo transitions to other tempi
            for new_interval in range(num_intervals):
                trans_prob[old_interval, new_interval] /= prob_sum

        # number of tempo transitions (= non-zero probabilities)
        cdef unsigned int num_tempo_transitions = \
            len(np.nonzero(trans_prob)[0])

        # apart from the very beginning of a beat, the tempo stays the same,
        # thus the number of transitions is equal to the total number of states
        # plus the number of tempo transitions minus the number of tempo states
        # since these transitions are already included in the tempo transitions
        cdef int num_transitions = num_states + num_tempo_transitions - \
                                   num_intervals
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
            self.state_space.first_states
        cdef unsigned int [::1] last_beat_positions = \
            self.state_space.last_states
        # state counter
        cdef int i = 0
        # loop over all tempi
        for new_interval in range(num_intervals):
            # generate all transitions from other tempi
            for old_interval in range(num_intervals):
                # but only if it is a probable transition
                if trans_prob[old_interval, new_interval] != 0:
                    # generate a transition
                    prev_states[i] = last_beat_positions[old_interval]
                    states[i] = first_beat_positions[new_interval]
                    probabilities[i] = trans_prob[old_interval, new_interval]
                    # increase counter
                    i += 1
            # transitions within the same tempo
            for prev_state in range(first_beat_positions[new_interval],
                                    last_beat_positions[new_interval]):
                # generate a transition with probability 1
                prev_states[i] = prev_state
                states[i] = prev_state + 1
                # Note: skip setting the probability here, since they were
                #       initialised with 1
                # increase counter
                i += 1
        # return the arrays
        return states, prev_states, probabilities


class MultiPatternTransitionModel(TransitionModel):
    """
    Transition model for pattern tracking with a HMM.

    Instead of modelling only a single beat (as :class:`BeatTransitionModel`),
    the :class:`MultiPatternTransitionModel` models rhythmic patterns. It
    accepts the same arguments as the :class:`BeatTransitionModel`, but
    everything as lists, with the list entries at the same position
    corresponding to one rhythmic pattern.

    Parameters
    ----------
    state_space : :class:`MultiPatternTransitionModel` instance
        MultiPatternTransitionModel instance.
    transition_lambda : list
        Lambda(s) for the exponential tempo change distribution of the patterns
        (higher values prefer a constant tempo from one bar to the next one).
        If a single value is given, the same value is assumed for all patterns.

    See Also
    --------
    :class:`BeatTransitionModel`

    Notes
    -----
    This transition model differs from the one described in [1]_ in the
    following way:

    - it allows transitions only at pattern boundaries instead of beat
      boundaries,
    - it uses the new state space discretisation and tempo change distribution
      proposed in [2]_.

    References
    ----------
    .. [1] Florian Krebs, Sebastian Böck and Gerhard Widmer,
           "Rhythmic Pattern Modeling for Beat and Downbeat Tracking in Musical
           Audio",
           Proceedings of the 14th International Society for Music Information
           Retrieval Conference (ISMIR), 2013.
    .. [2] Florian Krebs, Sebastian Böck and Gerhard Widmer,
           "An Efficient State Space Model for Joint Tempo and Meter Tracking",
           Proceedings of the 16th International Society for Music Information
           Retrieval Conference (ISMIR), 2015.

    """

    def __init__(self, state_space, transition_lambda):
        # expand the transition lambda to a list if needed, i.e. use the same
        # value for all patterns
        if not isinstance(transition_lambda, list):
            transition_lambda = [transition_lambda] * state_space.num_patterns
        # check if all lists have the same length
        if not state_space.num_patterns == len(transition_lambda):
            raise ValueError('number of patterns of the `state_space` and the '
                             'length `transition_lambda` must be the same')
        # save the given arguments
        self.state_space = state_space
        self.transition_lambda = transition_lambda
        # compute the transitions for each pattern and stack them
        enum = enumerate(zip(state_space.bar_state_spaces, transition_lambda))
        for pattern, (state_space, transition_lambda) in enum:
            # create a BeatTransitionModel
            tm = BeatTransitionModel(state_space, transition_lambda)
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
        super(MultiPatternTransitionModel, self).__init__(*transitions)


# observation models
class RNNBeatTrackingObservationModel(ObservationModel):
    """
    Observation model for beat tracking with a HMM.

    Parameters
    ----------
    state_space : :class:`BeatStateSpace` instance
        BeatStateSpace instance.
    observation_lambda : int
        Split one beat period into `observation_lambda` parts, the first
        representing beat states and the remaining non-beat states.
    norm_observations : bool, optional
        Normalize the observations.

    References
    ----------
    .. [1] Sebastian Böck, Florian Krebs and Gerhard Widmer,
           "A Multi-Model Approach to Beat Tracking Considering Heterogeneous
           Music Styles",
           Proceedings of the 15th International Society for Music Information
           Retrieval Conference (ISMIR), 2014.

    """

    def __init__(self, state_space, observation_lambda,
                 norm_observations=False):
        self.observation_lambda = observation_lambda
        self.norm_observations = norm_observations
        # compute observation pointers
        # always point to the non-beat densities
        pointers = np.ones(state_space.num_states, dtype=np.uint32)
        # unless they are in the beat range of the state space
        border = 1. / observation_lambda
        beat_idx = state_space.position[:state_space.num_states] < border
        pointers[beat_idx] = 0
        # instantiate a ObservationModel with the pointers
        super(RNNBeatTrackingObservationModel, self).__init__(pointers)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def log_densities(self, float [::1] observations):
        """
        Computes the log densities of the observations.

        Parameters
        ----------
        observations : numpy array
            Observations (i.e. activations of the RNN).

        Returns
        -------
        numpy array
            Log densities of the observations.

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


class GMMPatternTrackingObservationModel(ObservationModel):
    """
    Observation model for GMM based beat tracking with a HMM.

    Parameters
    ----------
    gmms : list
        Fitted GMM(s), one entry per rhythmic pattern.
    transition_model : :class:`MultiPatternTransitionModel` instance
        MultiPatternTransitionModel instance.
    norm_observations : bool, optional
        Normalize the observations.

    References
    ----------
    .. [1] Florian Krebs, Sebastian Böck and Gerhard Widmer,
           "Rhythmic Pattern Modeling for Beat and Downbeat Tracking in Musical
           Audio",
           Proceedings of the 14th International Society for Music Information
           Retrieval Conference (ISMIR), 2013.

    """

    def __init__(self, gmms, transition_model, norm_observations=False):
        # save the parameters
        self.gmms = gmms
        self.transition_model = transition_model
        self.norm_observations = norm_observations
        # define the pointers of the log densities
        pointers = np.zeros(transition_model.num_states, dtype=np.uint32)
        pattern = self.transition_model.pattern
        position = self.transition_model.position
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
        super(GMMPatternTrackingObservationModel, self).__init__(pointers)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def log_densities(self, observations):
        """
        Computes the log densities of the observations using (a) GMM(s).

        Parameters
        ----------
        observations : numpy array
            Observations (i.e. activations of the NN).

        Returns
        -------
        numpy array
            Log densities of the observations.

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
