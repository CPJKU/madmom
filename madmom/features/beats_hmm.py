# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains HMM state spaces, transition and observation models used
for beat, downbeat and pattern tracking.

Notes
-----
Please note that (almost) everything within this module is discretised to
integer values because of performance reasons.

"""

from __future__ import absolute_import, division, print_function

import numpy as np

from madmom.ml.hmm import ObservationModel, TransitionModel


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
        Positions of the states (i.e. 0...1).
    state_intervals : numpy array
        Intervals of the states (i.e. 1 / tempo).
    first_states : numpy array
        First state of each interval.
    last_states : numpy array
        Last state of each interval.

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
        self.intervals = np.ascontiguousarray(intervals, dtype=np.int)
        # number of states and intervals
        self.num_states = int(np.sum(intervals))
        self.num_intervals = len(intervals)
        # define first and last states
        first_states = np.cumsum(np.r_[0, self.intervals[:-1]])
        self.first_states = first_states.astype(np.int)
        self.last_states = np.cumsum(self.intervals) - 1
        # define the positions and intervals of the states
        self.state_positions = np.empty(self.num_states)
        self.state_intervals = np.empty(self.num_states, dtype=np.int)
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

    Model `num_beat` identical beats with the given arguments in a single state
    space.

    Parameters
    ----------
    num_beats : int
        Number of beats to form a bar.
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
        First states of each beat.
    last_states : list
        Last states of each beat.

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
        self.state_intervals = np.empty(0, dtype=np.int)
        self.num_states = 0
        # save the first and last states of the individual beats in a list
        self.first_states = []
        self.last_states = []
        # create a BeatStateSpace and stack it `num_beats` times
        bss = BeatStateSpace(min_interval, max_interval, num_intervals)
        for b in range(self.num_beats):
            # define position (add beat counter) and interval states
            self.state_positions = np.hstack((self.state_positions,
                                              bss.state_positions + b))
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

    Model a joint state space with the given `state_spaces` by stacking the
    individual state spaces.

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
        # combine the given state spaces in a single state space
        self.num_patterns = len(state_spaces)
        self.state_spaces = state_spaces
        self.state_positions = np.empty(0)
        self.state_intervals = np.empty(0, dtype=np.int)
        self.state_patterns = np.empty(0, dtype=np.int)
        self.num_states = 0
        # save the first and last states of the individual patterns in a list
        self.first_states = []
        self.last_states = []
        # stack the individual state spaces
        for p, pss in enumerate(state_spaces):
            # define position, interval and pattern states
            self.state_positions = np.hstack((self.state_positions,
                                              pss.state_positions))
            self.state_intervals = np.hstack((self.state_intervals,
                                              pss.state_intervals))
            self.state_patterns = np.hstack((self.state_patterns,
                                             np.repeat(p, pss.num_states)))
            # append the first and last states of each pattern
            self.first_states.append(pss.first_states[0] + self.num_states)
            self.last_states.append(pss.last_states[-1] + self.num_states)
            # finally increase the number of states
            self.num_states += pss.num_states


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
    ratio = (to_intervals.astype(np.float) /
             from_intervals.astype(np.float)[:, np.newaxis])
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
    from one tempo (i.e. interval) to another are allowed, following an
    exponential distribution.

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
        from_int = state_space.state_intervals[from_states]
        to_int = state_space.state_intervals[to_states]
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
            prob = exponential_transition(from_int, to_int,
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

    Add transitions with the given probability between the individual
    transition models. These transition models must correspond to the state
    spaces forming a :class:`MultiPatternStateSpace`.

    Parameters
    ----------
    transition_models : list
        List with :class:`TransitionModel` instances.
    transition_prob : numpy array or float, optional
        Probabilities to change the pattern at pattern boundaries. If an array
        is given, the first dimension corresponds to the origin pattern, the
        second to the destination pattern. If a single value is given, a
        uniform transition distribution to all other patterns is assumed. Set
        to None to stay within the same pattern.

    """

    def __init__(self, transition_models, transition_prob=None):
        # save attributes
        self.transition_models = transition_models
        self.transition_prob = transition_prob
        num_patterns = len(transition_models)
        # first stack all transition models
        first_states = []
        last_states = []
        for p, tm in enumerate(self.transition_models):
            # set/update the probabilities, states and pointers
            offset = 0
            if p == 0:
                # for the first pattern, just use the TM arrays
                states = tm.states
                pointers = tm.pointers
                probabilities = tm.probabilities
            else:
                # for all consecutive patterns, stack the TM arrays after
                # applying an offset
                # Note: len(pointers) = len(states) + 1, because of the CSR
                #       format of the TM (please see ml.hmm.TransitionModel)
                offset = len(pointers) - 1
                # states: offset = length of the pointers - 1
                states = np.hstack((states, tm.states + len(pointers) - 1))
                # pointers: offset = current maximum of the pointers
                #           start = tm.pointers[1:]
                pointers = np.hstack((pointers, tm.pointers[1:] +
                                      max(pointers)))
                # probabilities: just stack them
                probabilities = np.hstack((probabilities, tm.probabilities))
            # save the first/last states
            first_states.append(tm.state_space.first_states[0] + offset)
            last_states.append(tm.state_space.last_states[-1] + offset)
        # retrieve a dense representation in order to add transitions
        # TODO: operate directly on the sparse representation?
        states, prev_states, probabilities = self.make_dense(states, pointers,
                                                             probabilities)
        # translate float transition_prob value to transition_prob matrix
        if isinstance(transition_prob, float) and transition_prob:
            # create a pattern transition probability matrix
            self.transition_prob = np.ones((num_patterns, num_patterns))
            # transition to other patterns
            self.transition_prob *= transition_prob / (num_patterns - 1)
            # transition to same pattern
            diag = np.diag_indices_from(self.transition_prob)
            self.transition_prob[diag] = 1. - transition_prob
        else:
            self.transition_prob = transition_prob
        # update/add transitions between patterns
        if self.transition_prob is not None and num_patterns > 1:
            new_states = []
            new_prev_states = []
            new_probabilities = []
            for p in range(num_patterns):
                # indices of states/prev_states/probabilities
                idx = np.logical_and(np.in1d(prev_states, last_states[p]),
                                     np.in1d(states, first_states[p]))
                # transition probability
                prob = probabilities[idx]
                # update transitions to same pattern with new probability
                probabilities[idx] *= self.transition_prob[p, p]
                # distribute that part among all other patterns
                for p_ in np.setdiff1d(range(num_patterns), p):
                    idx_ = np.logical_and(
                        np.in1d(prev_states, last_states[p_]),
                        np.in1d(states, first_states[p_]))
                    # make sure idx and idx_ have same length
                    if len(np.nonzero(idx)[0]) != len(np.nonzero(idx_)[0]):
                        raise ValueError('Cannot add transition between '
                                         'patterns with different number of '
                                         'entering/exiting states.')
                    # use idx for the states and idx_ for prev_states
                    new_states.extend(states[idx])
                    new_prev_states.extend(prev_states[idx_])
                    new_probabilities.extend(prob *
                                             self.transition_prob[p, p_])
            # extend the arrays by these new transitions
            states = np.append(states, new_states)
            prev_states = np.append(prev_states, new_prev_states)
            probabilities = np.append(probabilities, new_probabilities)
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
        Compute the log densities of the observations.

        Parameters
        ----------
        observations : numpy array, shape (N, )
            Observations (i.e. 1D beat activations of the RNN).

        Returns
        -------
        numpy array, shape (N, 2)
            Log densities of the observations, the columns represent the
            observation log probability densities for no-beats and beats.

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
        Compute the log densities of the observations.

        Parameters
        ----------
        observations : numpy array, shape (N, 2)
            Observations (i.e. 2D activations of a RNN, the columns represent
            'beat' and 'downbeat' probabilities)

        Returns
        -------
        numpy array, shape (N, 3)
            Log densities of the observations, the columns represent the
            observation log probability densities for no-beats, beats and
            downbeats.

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
            # TODO: save the number of beats in the pattern files so we don't
            #       need to save references to all state spaces
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
        Compute the log densities of the observations using (a) GMM(s).

        Parameters
        ----------
        observations : numpy array
            Observations (i.e. multi-band spectral flux features).

        Returns
        -------
        numpy array, shape (N, num_gmms)
            Log densities of the observations, the columns represent the
            observation log probability densities for the individual GMMs.

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
