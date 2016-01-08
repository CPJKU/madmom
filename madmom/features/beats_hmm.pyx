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

    Attributes
    ----------
    num_states : int
        Number of states.
    intervals : numpy array
        Modeled intervals.
    num_intervals : int
        Number of intervals.
    state_positions : numpy array
        Positions of the states.
    state_intervals : numpy array
        Intervals of the states.
    first_states : numpy array
        First states for each interval.
    last_states : numpy array
        Last states for each interval.

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
        # save the intervals
        self.intervals = np.ascontiguousarray(intervals, dtype=np.uint32)
        # number of states and intervals
        self.num_states = int(np.sum(intervals))
        self.num_intervals = len(intervals)
        # define first and last states
        first_states = np.cumsum(np.r_[0, self.intervals[:-1]])
        self.first_states = first_states.astype(np.uint32)
        self.last_states = np.cumsum(self.intervals).astype(np.uint32) - 1
        # define the position and interval states
        self.state_positions = np.empty(self.num_states)
        self.state_intervals = np.empty(self.num_states, dtype=np.uint32)
        idx = 0
        for i in self.intervals:
            self.state_positions[idx: idx + i] = np.linspace(0, 1, i,
                                                             endpoint=False)
            self.state_intervals[idx: idx + i] = i
            # increase counter
            idx += i


class BarStateSpace(object):
    """
    State space for bar tracking with a HMM.

    Parameters
    ----------
    num_beats : int
        Number of beats per bar.
    min_interval : float
        Minimum beat interval to model.
    max_interval : float
        Maximum beat interval to model.
    num_intervals : int, optional
        Number of beat intervals to model; if set, limit the number of
        intervals and use a log spacing instead of the default linear spacing.

    Attributes
    ----------
    num_beats : int.
        Number of beats.
    num_states : int
        Number of states.
    num_intervals : int
        Number of intervals.
    state_positions : numpy array
        Positions of the states.
    state_intervals : numpy array
        Intervals of the states.
    beat_state_offsets : numpy array
        State offsets of the beats.
    first_states : list
        First interval states for each beat.
    last_states : list
        Last interval states for each beat.

    References
    ----------
    .. [1] Florian Krebs, Sebastian Böck and Gerhard Widmer,
           "An Efficient State Space Model for Joint Tempo and Meter Tracking",
           Proceedings of the 16th International Society for Music Information
           Retrieval Conference (ISMIR), 2015.

    """

    def __init__(self, num_beats, min_interval, max_interval,
                 num_intervals=None):
        # model N beats as a bar
        self.num_beats = int(num_beats)
        self.state_positions = np.empty(0, dtype=np.uint32)
        self.state_intervals = np.empty(0, dtype=np.uint32)
        self.beat_state_offsets = np.empty(0, dtype=np.int)
        self.num_states = 0
        # save the first and last states of the individual beats in a list
        self.first_states = []
        self.last_states = []
        # create a beat state space
        bss = BeatStateSpace(min_interval, max_interval, num_intervals)
        offset = 0
        for n in range(self.num_beats):
            # define position and interval states
            self.state_positions = np.hstack((self.state_positions,
                                              bss.state_positions + n))
            self.state_intervals = np.hstack((self.state_intervals,
                                              bss.state_intervals))
            self.num_states += bss.num_states
            self.first_states.append(bss.first_states + offset)
            self.last_states.append(bss.last_states + offset)
            # save the offsets and increase afterwards
            self.beat_state_offsets = np.hstack((self.beat_state_offsets,
                                                 offset))
            offset += bss.num_states


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
        state_spaces = []
        enum = enumerate(zip(min_intervals, max_intervals, num_intervals))
        for pattern, (min_, max_, num_) in enum:
            # create a BeatStateSpace and append it to the list
            state_spaces.append(BeatStateSpace(min_, max_, num_))
        self.state_spaces = state_spaces
        # define the position, interval and pattern states
        self.state_positions = \
            np.hstack([st.state_positions for st in self.state_spaces])
        self.state_intervals = \
            np.hstack([st.state_intervals for st in self.state_spaces])
        self.state_patterns = \
            np.hstack([np.repeat(i, st.num_states)
                       for i, st in enumerate(self.state_spaces)])

    @property
    def num_states(self):
        """Number of states."""
        return int(sum([st.num_states for st in self.state_spaces]))

    @property
    def num_patterns(self):
        """Number of rhythmic patterns"""
        return len(self.state_spaces)


# transition distributions
def exponential_transition(from_intervals, to_intervals, transition_lambda,
                           threshold=np.spacing(1), norm=True):
    """
    Exponential tempo transition.

    Parameters
    ----------
    from_intervals : numpy array
        Intervals where the transitions originate from.
    to_intervals :  : numpy array
        Intervals where the transitions destinate to.
    transition_lambda : float
        Lambda for the exponential tempo change distribution (higher values
        prefer a constant tempo from one beat/bar to the next one).
    threshold : float, optional
        Set transition probabilities below this threshold to zero.
    norm : bool, optional
        Normalize the emission probabilities to sum 1.

    Returns
    -------
    probabilities : numpy array, shape (num_from_intervals, num_to_intervals)
        Probability of each transition from an interval to another.

    References
    ----------
    .. [1] Florian Krebs, Sebastian Böck and Gerhard Widmer,
           "An Efficient State Space Model for Joint Tempo and Meter Tracking",
           Proceedings of the 16th International Society for Music Information
           Retrieval Conference (ISMIR), 2015.

    """
    # compute the transition probabilities
    ratio = to_intervals / from_intervals[:, np.newaxis]
    prob = np.exp(-transition_lambda * abs(ratio - 1.))
    # set values below threshold to 0
    prob[prob <= threshold] = 0
    # normalize the emission probabilities
    if norm:
        prob /= np.sum(prob, axis=1)[:, np.newaxis]
    return prob


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
        self.transition_lambda = float(transition_lambda)
        # intra state space connections (i.e. same tempi)
        states = np.arange(state_space.num_states, dtype=np.uint32)
        # remove the transitions into the first states
        states = np.setdiff1d(states, state_space.first_states)
        prev_states = states - 1
        probabilities = np.ones_like(states, dtype=np.float)
        # self connection of the state space (i.e. tempo changes)
        to_states = state_space.first_states
        from_states = state_space.last_states
        # generate an exponential tempo transition
        from_int = state_space.state_intervals[from_states].astype(np.float)
        to_int = state_space.state_intervals[to_states].astype(np.float)
        prob = exponential_transition(from_int, to_int, self.transition_lambda)
        # use only the states with transitions to/from != 0
        prev_states = np.hstack((prev_states,
                                 from_states[np.nonzero(prob)[0]]))
        states = np.hstack((states, to_states[np.nonzero(prob)[1]]))
        probabilities = np.hstack((probabilities, prob[prob != 0]))
        # make the transitions sparse
        transitions = self.make_sparse(states, prev_states, probabilities)
        # instantiate a TransitionModel
        super(BeatTransitionModel, self).__init__(*transitions)


class BarTransitionModel(TransitionModel):
    """
    Transition model for bar tracking with a HMM.

    Parameters
    ----------
    state_space : :class:`BarStateSpace` instance
        BarStateSpace instance.
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
        self.transition_lambda = float(transition_lambda)
        # intra state space connections (i.e. same tempi within the beats)
        states = np.arange(state_space.num_states, dtype=np.uint32)
        # remove the transitions into the first states of the individual beats
        states = np.setdiff1d(states, state_space.first_states)
        prev_states = states - 1
        probabilities = np.ones_like(states, dtype=np.float)
        # tempo transition at the beat boundaries
        for beat in range(state_space.num_beats):
            # connect to the first states of the actual beat
            to_states = state_space.first_states[beat]
            # connect from the last states of the previous beat
            from_states = state_space.last_states[beat - 1]
            # generate an exponential tempo transition
            from_int = state_space.state_intervals[from_states]
            to_int = state_space.state_intervals[to_states]
            prob = exponential_transition(from_int.astype(np.float),
                                          to_int.astype(np.float),
                                          self.transition_lambda)
            # use only the states with transitions to/from != 0
            prev_states = np.hstack((prev_states,
                                     from_states[np.nonzero(prob)[0]]))
            states = np.hstack((states, to_states[np.nonzero(prob)[1]]))
            probabilities = np.hstack((probabilities, prob[prob != 0]))
        # make the transitions sparse
        transitions = self.make_sparse(states, prev_states, probabilities)
        # instantiate a TransitionModel
        super(BarTransitionModel, self).__init__(*transitions)


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
        enum = enumerate(zip(state_space.state_spaces, transition_lambda))
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
        pointers[state_space.state_positions < border] = 0
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
        patterns = self.transition_model.state_patterns
        positions = self.transition_model.state_positions
        # Note: the densities of all GMMs are just stacked on top of each
        #       other, so we have to to keep track of the total number of GMMs
        densities_idx_offset = 0
        for p in range(len(gmms)):
            # number of fitted GMMs for this pattern
            num_gmms = len(gmms[p])
            # distribute the observation densities defined by the GMMs
            # uniformly across the entire state space (for this pattern)
            # since the densities are just stacked, add the offset
            pointers[patterns == p] = (positions[patterns == p] * num_gmms +
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
            Observations (i.e. multiband spectral flux features).

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
