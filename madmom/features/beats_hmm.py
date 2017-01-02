# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains HMM state spaces, transition and observation models used
for beat and downbeat tracking.

Notes
-----
Please note that (almost) everything within this module is discretised to
integer values because of performance reasons.

"""

from __future__ import absolute_import, division, print_function

import numpy as np

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
            # we must approach the number of intervals iteratively
            num_log_intervals = num_intervals
            intervals = []
            while len(intervals) < num_intervals:
                intervals = np.logspace(np.log2(min_interval),
                                        np.log2(max_interval),
                                        num_log_intervals, base=2)
                # quantize to integer intervals
                intervals = np.unique(np.round(intervals))
                num_log_intervals += 1
        # save the intervals
        self.intervals = np.ascontiguousarray(intervals, dtype=np.uint32)
        # number of states and intervals
        self.num_states = int(np.sum(intervals))
        self.num_intervals = len(intervals)
        # define first and last states
        first_states = np.cumsum(np.r_[0, self.intervals[:-1]])
        self.first_states = first_states.astype(np.uint32)
        self.last_states = np.cumsum(self.intervals).astype(np.uint32) - 1
        # define the positions and intervals of the states
        self.state_positions = np.empty(self.num_states)
        self.state_intervals = np.empty(self.num_states, dtype=np.uint32)
        # Note: having an index counter is faster than ndenumerate
        idx = 0
        for i in self.intervals:
            self.state_positions[idx: idx + i] = np.linspace(0, 1, i,
                                                             endpoint=False)
            self.state_intervals[idx: idx + i] = i
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
    num_beats : int
        Number of beats.
    num_states : int
        Number of states.
    num_intervals : int
        Number of intervals.
    state_positions : numpy array
        Positions of the states.
    state_intervals : numpy array
        Intervals of the states.
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
        self.state_positions = np.empty(0)
        self.state_intervals = np.empty(0, dtype=np.uint32)
        self.num_states = 0
        # save the first and last states of the individual beats in a list
        self.first_states = []
        self.last_states = []
        # create a BeatStateSpace and stack it `num_beats` times
        bss = BeatStateSpace(min_interval, max_interval, num_intervals)
        for n in range(self.num_beats):
            # define position and interval states
            self.state_positions = np.hstack((self.state_positions,
                                              bss.state_positions + n))
            self.state_intervals = np.hstack((self.state_intervals,
                                              bss.state_intervals))
            # add the current number of states as offset
            self.first_states.append(bss.first_states + self.num_states)
            self.last_states.append(bss.last_states + self.num_states)
            # finally increase the number of states
            self.num_states += bss.num_states


class MultiPatternStateSpace(object):
    """
    State space for rhythmic pattern tracking with a HMM.

    Parameters
    ----------
    state_spaces : list
        List with state spaces to model.

    References
    ----------
    .. [1] Florian Krebs, Sebastian Böck and Gerhard Widmer,
           "An Efficient State Space Model for Joint Tempo and Meter Tracking",
           Proceedings of the 16th International Society for Music Information
           Retrieval Conference (ISMIR), 2015.

    """

    def __init__(self, state_spaces):
        self.state_spaces = state_spaces
        # model the patterns as a whole
        self.num_patterns = len(self.state_spaces)
        self.state_positions = np.empty(0)
        self.state_intervals = np.empty(0, dtype=np.uint32)
        self.state_patterns = np.empty(0, dtype=np.uint32)
        self.num_states = 0
        # save the first and last states of the individual patterns in a list
        # self.first_states = []
        # self.last_states = []
        for p in range(self.num_patterns):
            pattern = self.state_spaces[p]
            # define position, interval and pattern states
            self.state_positions = np.hstack((self.state_positions,
                                              pattern.state_positions))
            self.state_intervals = np.hstack((self.state_intervals,
                                              pattern.state_intervals))
            self.state_patterns = np.hstack((self.state_patterns,
                                             np.repeat(p, pattern.num_states)))
            # TODO: first and last states should both be lists to work easily
            # self.first_states.append()
            # self.last_states.append()
            # finally increase the number of states
            self.num_states += pattern.num_states


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
        Intervals where the transitions terminate.
    transition_lambda : float
        Lambda for the exponential tempo change distribution (higher values
        prefer a constant tempo from one beat/bar to the next one). If None,
        allow only transitions from/to the same interval.
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
    # no transition lambda
    if transition_lambda is None:
        # return a diagonal matrix
        return np.diag(np.diag(np.ones((len(from_intervals),
                                        len(to_intervals)))))
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

    Within the beat the tempo stays the same; at beat boundaries transitions
    from one tempo (i.e. interval) to another following an exponential
    distribution are allowed.

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
        # same tempo transitions probabilities within the state space is 1
        # Note: use all states, but remove all first states because there are
        #       no same tempo transitions into them
        states = np.arange(state_space.num_states, dtype=np.uint32)
        states = np.setdiff1d(states, state_space.first_states)
        prev_states = states - 1
        probabilities = np.ones_like(states, dtype=np.float)
        # tempo transitions occur at the boundary between beats
        # Note: connect the beat state space with itself, the transitions from
        #       the last states to the first states follow an exponential tempo
        #       transition (with the tempi given as intervals)
        to_states = state_space.first_states
        from_states = state_space.last_states
        from_int = state_space.state_intervals[from_states].astype(np.float)
        to_int = state_space.state_intervals[to_states].astype(np.float)
        prob = exponential_transition(from_int, to_int, self.transition_lambda)
        # use only the states with transitions to/from != 0
        from_prob, to_prob = np.nonzero(prob)
        states = np.hstack((states, to_states[to_prob]))
        prev_states = np.hstack((prev_states, from_states[from_prob]))
        probabilities = np.hstack((probabilities, prob[prob != 0]))
        # make the transitions sparse
        transitions = self.make_sparse(states, prev_states, probabilities)
        # instantiate a TransitionModel
        super(BeatTransitionModel, self).__init__(*transitions)


class BarTransitionModel(TransitionModel):
    """
    Transition model for bar tracking with a HMM.

    Within the beats of the bar the tempo stays the same; at beat boundaries
    transitions from one tempo (i.e. interval) to another following an
    exponential distribution are allowed.

    Parameters
    ----------
    state_space : :class:`BarStateSpace` instance
        BarStateSpace instance.
    transition_lambda : float or list
        Lambda for the exponential tempo change distribution (higher values
        prefer a constant tempo from one beat to the next one).
        None can be used to set the tempo change probability to 0.
        If a list is given, the individual values represent the lambdas for
        each transition into the beat at this index position.

    Notes
    -----
    Bars performing tempo changes only at bar boundaries (and not at the beat
    boundaries) must have set all but the first `transition_lambda` values to
    None, e.g. [100, None, None] for a bar with 3 beats.

    References
    ----------
    .. [1] Florian Krebs, Sebastian Böck and Gerhard Widmer,
           "An Efficient State Space Model for Joint Tempo and Meter Tracking",
           Proceedings of the 16th International Society for Music Information
           Retrieval Conference (ISMIR), 2015.

    """

    def __init__(self, state_space, transition_lambda):
        # expand transition_lambda to a list if a single value is given
        if not isinstance(transition_lambda, list):
            transition_lambda = [transition_lambda] * state_space.num_beats
        if state_space.num_beats != len(transition_lambda):
            raise ValueError('length of `transition_lambda` must be equal to '
                             '`num_beats` of `state_space`.')
        # save attributes
        self.state_space = state_space
        self.transition_lambda = transition_lambda
        # TODO: this could be unified with the BeatTransitionModel
        # same tempo transitions probabilities within the state space is 1
        # Note: use all states, but remove all first states of the individual
        #       beats, because there are no same tempo transitions into them
        states = np.arange(state_space.num_states, dtype=np.uint32)
        states = np.setdiff1d(states, state_space.first_states)
        prev_states = states - 1
        probabilities = np.ones_like(states, dtype=np.float)
        # tempo transitions occur at the boundary between beats (unless the
        # corresponding transition_lambda is set to None)
        for beat in range(state_space.num_beats):
            # connect to the first states of the actual beat
            to_states = state_space.first_states[beat]
            # connect from the last states of the previous beat
            from_states = state_space.last_states[beat - 1]
            # transition follow an exponential tempo distribution
            from_int = state_space.state_intervals[from_states]
            to_int = state_space.state_intervals[to_states]
            prob = exponential_transition(from_int.astype(np.float),
                                          to_int.astype(np.float),
                                          transition_lambda[beat])
            # use only the states with transitions to/from != 0
            from_prob, to_prob = np.nonzero(prob)
            states = np.hstack((states, to_states[to_prob]))
            prev_states = np.hstack((prev_states, from_states[from_prob]))
            probabilities = np.hstack((probabilities, prob[prob != 0]))
        # make the transitions sparse
        transitions = self.make_sparse(states, prev_states, probabilities)
        # instantiate a TransitionModel
        super(BarTransitionModel, self).__init__(*transitions)


class MultiPatternTransitionModel(TransitionModel):
    """
    Transition model for pattern tracking with a HMM.

    Parameters
    ----------
    transition_models : list
        List with :class:`TransitionModel` instances.
    transition_prob : numpy array, optional
        Matrix with transition probabilities from one pattern to another.
    transition_lambda : float, optional
        Lambda for the exponential tempo change distribution (higher values
        prefer a constant tempo from one pattern to the next one).
    pattern_change_prob : float, optional
        Probability of a pattern change. With pattern_change_prob - 1 we
        maintain the old pattern.


    """

    def __init__(self, transition_models, transition_prob=None,
                 transition_lambda=None, pattern_change_prob=0.0):
        # TODO: use transition_prob to set different pattern change
        # probabilities
        if transition_prob is not None or transition_lambda is not None:
            raise NotImplementedError("please implement pattern transitions")
        # save attributes
        self.transition_models = transition_models
        self.transition_prob = transition_prob
        self.transition_lambda = transition_lambda
        num_patterns = len(self.transition_models)
        first_pattern_states = np.zeros(num_patterns, dtype=int)
        last_pattern_states = np.zeros(num_patterns, dtype=int)
        # stack the pattern transitions
        for i, tm in enumerate(self.transition_models):
            # set/update the probabilities, states and pointers
            if i == 0:
                # for the first pattern, just set the TM arrays
                states = tm.states
                pointers = tm.pointers
                probabilities = tm.probabilities
                first_pattern_states[i] = 0
                last_pattern_states[i] = tm.num_states - 1
            else:
                first_pattern_states[i] = len(pointers) - 1
                last_pattern_states[i] = first_pattern_states[i] + \
                    tm.num_states - 1
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

        states, prev_states, probabilities = self.make_dense(states, pointers,
                                                             probabilities)
        # add pattern_transitions
        if pattern_change_prob > 0 and num_patterns > 1:
            same_pattern = 1 - pattern_change_prob
            change_pattern = pattern_change_prob / (num_patterns - 1)
            for i in range(num_patterns):
                # find states at pattern borders
                idx = np.intersect1d(
                    np.where(prev_states == last_pattern_states[i])[0],
                    np.where(states == first_pattern_states[i])[0])
                probabilities[idx] = same_pattern
                # add pattern transition
                for j in range(num_patterns):
                    if i != j:
                        prev_states = np.hstack((prev_states,
                                                 last_pattern_states[i]))
                        states = np.hstack((states, first_pattern_states[j]))
                        probabilities = np.hstack((probabilities,
                                                   change_pattern))

        # make the transitions sparse
        transitions = self.make_sparse(states, prev_states, probabilities)
        # instantiate a TransitionModel
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

    References
    ----------
    .. [1] Sebastian Böck, Florian Krebs and Gerhard Widmer,
           "A Multi-Model Approach to Beat Tracking Considering Heterogeneous
           Music Styles",
           Proceedings of the 15th International Society for Music Information
           Retrieval Conference (ISMIR), 2014.

    """

    def __init__(self, state_space, observation_lambda):
        self.observation_lambda = observation_lambda
        # compute observation pointers
        # always point to the non-beat densities
        pointers = np.zeros(state_space.num_states, dtype=np.uint32)
        # unless they are in the beat range of the state space
        border = 1. / observation_lambda
        pointers[state_space.state_positions < border] = 1
        # instantiate a ObservationModel with the pointers
        super(RNNBeatTrackingObservationModel, self).__init__(pointers)

    def log_densities(self, observations):
        """
        Computes the log densities of the observations.

        Parameters
        ----------
        observations : numpy array, shape (N, )
            Observations (i.e. 1d activations of the RNN).

        Returns
        -------
        numpy array
            Log densities of the observations.

        """
        # init densities
        log_densities = np.empty((len(observations), 2), dtype=np.float)
        # Note: it's faster to call np.log 2 times instead of once on the
        #       whole 2d array
        log_densities[:, 0] = np.log((1. - observations) /
                                     (self.observation_lambda - 1))
        log_densities[:, 1] = np.log(observations)
        # return the densities
        return log_densities


class RNNDownBeatTrackingObservationModel(ObservationModel):
    """
    Observation model for downbeat tracking with a HMM.

    Parameters
    ----------
    state_space : :class:`BarStateSpace` instance
        BarStateSpace instance.
    observation_lambda : int
        Split each (down-)beat period into `observation_lambda` parts, the
        first representing (down-)beat states and the remaining non-beat
        states.

    References
    ----------
    .. [1] Sebastian Böck, Florian Krebs and Gerhard Widmer,
           "Joint Beat and Downbeat Tracking with Recurrent Neural Networks"
           Proceedings of the 17th International Society for Music Information
           Retrieval Conference (ISMIR), 2016.

    """

    def __init__(self, state_space, observation_lambda):
        self.observation_lambda = observation_lambda
        # compute observation pointers
        # always point to the non-beat densities
        pointers = np.zeros(state_space.num_states, dtype=np.uint32)
        # unless they are in the beat range of the state space
        border = 1. / observation_lambda
        pointers[state_space.state_positions % 1 < border] = 1
        # the downbeat (i.e. the first beat range) points to density column 2
        pointers[state_space.state_positions < border] = 2
        # instantiate a ObservationModel with the pointers
        super(RNNDownBeatTrackingObservationModel, self).__init__(pointers)

    def log_densities(self, observations):
        """
        Computes the log densities of the observations.

        Parameters
        ----------
        observations : numpy array, shape (N, 2)
            Observations (i.e. 2d activations of a RNN, the columns represent
            'beat' and 'downbeat' probabilities)

        Returns
        -------
        numpy array
            Log densities of the observations.

        """
        # init densities
        log_densities = np.empty((len(observations), 3), dtype=np.float)
        # Note: it's faster to call np.log multiple times instead of once on
        #       the whole 2d array
        log_densities[:, 0] = np.log((1. - np.sum(observations, axis=1)) /
                                     (self.observation_lambda - 1))
        log_densities[:, 1] = np.log(observations[:, 0])
        log_densities[:, 2] = np.log(observations[:, 1])
        # return the densities
        return log_densities


class GMMPatternTrackingObservationModel(ObservationModel):
    """
    Observation model for GMM based beat tracking with a HMM.

    Parameters
    ----------
    pattern_files : list
        List with files representing the rhythmic patterns, one entry per
        pattern; each pattern being a list with fitted GMMs.
    state_space : :class:`MultiPatternStateSpace` instance
        Multi pattern state space.

    References
    ----------
    .. [1] Florian Krebs, Sebastian Böck and Gerhard Widmer,
           "Rhythmic Pattern Modeling for Beat and Downbeat Tracking in Musical
           Audio",
           Proceedings of the 14th International Society for Music Information
           Retrieval Conference (ISMIR), 2013.

    """

    def __init__(self, pattern_files, state_space):
        # save the parameters
        self.pattern_files = pattern_files
        self.state_space = state_space
        # define the pointers of the log densities
        pointers = np.zeros(state_space.num_states, dtype=np.uint32)
        patterns = self.state_space.state_patterns
        positions = self.state_space.state_positions
        # Note: the densities of all GMMs are just stacked on top of each
        #       other, so we have to to keep track of the total number of GMMs
        densities_idx_offset = 0
        for p, gmms in enumerate(pattern_files):
            # number of fitted GMMs for this pattern
            num_gmms = len(gmms)
            # number of beats in this pattern
            num_beats = self.state_space.state_spaces[p].num_beats
            # distribute the observation densities defined by the GMMs
            # uniformly across the entire state space (for this pattern)
            # since the densities are just stacked, add the offset
            # Note: we have to divide by the number of beats, since the
            #       positions range is [0, num_beats]
            pointers[patterns == p] = (positions[patterns == p] * num_gmms /
                                       num_beats + densities_idx_offset)
            # increase the offset by the number of GMMs
            densities_idx_offset += num_gmms
        # instantiate a ObservationModel with the pointers
        super(GMMPatternTrackingObservationModel, self).__init__(pointers)

    def log_densities(self, observations):
        """
        Computes the log densities of the observations using (a) GMM(s).

        Parameters
        ----------
        observations : numpy array
            Observations (i.e. multi-band spectral flux features).

        Returns
        -------
        numpy array
            Log densities of the observations.

        """
        # number of GMMs of all patterns
        num_gmms = sum([len(pattern) for pattern in self.pattern_files])
        # init the densities
        log_densities = np.empty((len(observations), num_gmms), dtype=np.float)
        # define the observation densities
        i = 0
        for pattern in self.pattern_files:
            for gmm in pattern:
                # get the predictions of each GMM for the observations
                log_densities[:, i] = gmm.score(observations)
                i += 1
        # return the densities
        return log_densities
