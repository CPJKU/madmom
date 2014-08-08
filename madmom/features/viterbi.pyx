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
def crf_viterbi(float [::1] pi, float[::1] transition, float[::1] norm_factor,
                float [::1] activations, int tau):
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
    cdef float [::1] v_c = np.empty(num_st, dtype=np.float32)
    # previous viterbi variables. will be initialised with prior (first beat)
    cdef float [::1] v_p = np.empty(num_st, dtype=np.float32)
    # back-tracking pointers;
    cdef long [:, ::1] bps = np.empty((num_x - 1, num_st), dtype=np.int)
    # back tracked path, a.k.a. path sequence
    cdef long [::1] path = np.empty(num_x, dtype=np.int)

    # counters etc.
    cdef int k, i, j, next_state
    cdef double new_prob, sum_k, path_prob, log_sum = 0.0

    # init first beat
    for i in range(num_st):
        v_p[i] = pi[i] * activations[i]
        sum_k += v_p[i]
    for i in range(num_st):
        v_p[i] = v_p[i] / sum_k

    sum_k = 0

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
    path_prob = 0.0
    for i in range(num_st):
        if v_p[i] > path_prob:
            next_state = i
            path_prob = v_p[i]
    path[num_x - 1] = next_state

    # track the path backwards
    for i in range(num_x - 2, -1, -1):
        next_state = bps[i, next_state]
        path[i] = next_state

    # return the best sequence and its log probability
    return np.asarray(path), log(path_prob) + log_sum


cdef class Transitions(object):
    """
    Transitions suitable for a DBN.

    """
    # define some variables which are also exported as Python attributes
    cdef public np.ndarray probabilities
    cdef public np.ndarray states
    cdef public np.ndarray pointers
    cdef list attributes

    def __init__(self, model=None):
        """
        Construct a transition probability object suitable for DBNs.

        :param model:  load the transitions model from the given file

        """
        # set the attributes
        self.attributes = ['probabilities', 'states', 'pointers']
        # load the transitions
        if model is not None:
            self.load(model)

    def _transitions(self, **kwargs):
        """Method to compute the transitions."""
        # The method must populate the following variables:
        # self.states, self.pointers, self.probabilities
        raise NotImplementedError('needs to be implemented by sub-classes')

    def save(self, outfile, compressed=True):
        """
        Save the transitions to a file.

        :param outfile:    file name or file handle to save the transitions to
        :param compressed: save in compressed format

        """
        # idea taken from: http://stackoverflow.com/questions/8955448
        npz = {}
        for attr in self.attributes:
            npz[attr] = self.__getattribute__(attr)
        # save in compressed or normal format?
        save_ = np.savez
        if compressed:
            save_ = np.savez_compressed
        # write everything to a file
        save_(outfile, **npz)

    def load(self, infile):
        """
        Load the transitions from a file.

        :param infile: file name or file handle with the transitions

        """
        # idea taken from: http://stackoverflow.com/questions/8955448
        data = np.load(infile)
        for key in data.keys():
            self.__setattr__(key, data[key])


cdef class BeatTrackingTransitions(Transitions):
    """
    Transitions suitable for a DBN.

    """
    # define some class variables which are also exported as Python attributes
    cdef public unsigned int num_beat_states
    cdef public np.ndarray tempo_states
    cdef public double tempo_change_probability

    # default values for beat tracking
    NUM_BEAT_STATES = 1280
    TEMPO_CHANGE_PROBABILITY = 0.008
    TEMPO_STATES = np.arange(11, 47)

    def __init__(self, model=None,
                 num_beat_states=NUM_BEAT_STATES,
                 tempo_states=TEMPO_STATES,
                 tempo_change_probability=TEMPO_CHANGE_PROBABILITY):
        """
        Construct a transition probability object suitable for beat tracking.

        :param model: load the transitions model from the given file

        If no model was given, the object is constructed with the following
        parameters:

        :param num_beat_states:          number of beat states for one beat
                                         period
        :param tempo_states:             array with tempo states (number of
                                         beat states to progress from one
                                         observation value to the next one)
        :param tempo_change_probability: probability of a tempo change from
                                         one observation to the next one

        """
        # load or instantiate a Transitions object
        super(BeatTrackingTransitions, self).__init__(None)
        # save the additional attributes
        self.attributes.extend(['num_beat_states', 'tempo_states',
                                'tempo_change_probability'])
        # load a model or compute transitions if needed
        if model is not None:
            self.load(model)
        else:
            self._transitions(num_beat_states,
                              np.ascontiguousarray(tempo_states,
                                                   dtype=np.int32),
                              tempo_change_probability)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _transitions(self,
                     unsigned int num_beat_states,
                     int [::1] tempo_states,
                     double tempo_change_probability):
        """
        Compute beat tracking transitions.

        :param num_beat_states:          number of beat states for one beat
                                         period
        :param tempo_states:             array with tempo states (number of
                                         beat states to progress from one
                                         observation value to the next one)
        :param tempo_change_probability: probability of a tempo change from
                                         one observation to the next one
        """
        from scipy.sparse import csr_matrix
        # save the given parameters
        self.num_beat_states = num_beat_states
        self.tempo_states = np.asarray(tempo_states)
        self.tempo_change_probability = tempo_change_probability
        # number of tempo & total states
        cdef unsigned int num_tempo_states = len(tempo_states)
        cdef unsigned int num_states = num_beat_states * num_tempo_states
        # transition probabilities
        cdef double same_tempo_prob = 1. - tempo_change_probability
        cdef double change_tempo_prob = 0.5 * tempo_change_probability
        # counters etc.
        cdef unsigned int state, prev_state, beat_state, tempo_state, tempo
        # lists for transitions matrix creation
        # TODO: use c++ containers? http://stackoverflow.com/questions/7403966
        cdef list states = []
        cdef list prev_states = []
        cdef list probabilities = []
        # loop over all states
        for state in range(num_states):
            # position inside beat & tempo
            beat_state = state % num_beat_states
            tempo_state = state / num_beat_states
            tempo = tempo_states[tempo_state]
            # for each state check the 3 possible transitions
            # previous state with same tempo
            # Note: we add num_beat_states before the modulo operation so
            #       that it can be computed in C (which is faster)
            prev_state = ((beat_state + num_beat_states - tempo) %
                          num_beat_states +
                          (tempo_state * num_beat_states))
            # probability for transition from same tempo
            states.append(state)
            prev_states.append(prev_state)
            probabilities.append(same_tempo_prob)
            # transition from slower tempo
            if tempo_state > 0:
                # previous state with slower tempo
                prev_state = ((beat_state + num_beat_states -
                               (tempo - 1)) % num_beat_states +
                              ((tempo_state - 1) * num_beat_states))
                # probability for transition from slower tempo
                states.append(state)
                prev_states.append(prev_state)
                probabilities.append(change_tempo_prob)
            # transition from faster tempo
            if tempo_state < num_tempo_states - 1:
                # previous state with faster tempo
                # Note: we add num_beat_states before the modulo operation
                #       so that it can be computed in C (which is faster)
                prev_state = ((beat_state + num_beat_states -
                               (tempo + 1)) % num_beat_states +
                              ((tempo_state + 1) * num_beat_states))
                # probability for transition from faster tempo
                states.append(state)
                prev_states.append(prev_state)
                probabilities.append(change_tempo_prob)
        # save everything in a sparse transitions matrix
        transitions = csr_matrix((probabilities, (states, prev_states)))
        # save the sparse array as 3 linear arrays
        # Note: saving in the format also used by scipy.sparse.csr_matrix
        #       allows us to parallelise the viterbi of the DBN, since we
        #       remove all duplicates from the states
        self.states = transitions.indices.astype(np.uint32)
        self.pointers = transitions.indptr.astype(np.uint32)
        self.probabilities = transitions.data.astype(dtype=np.float)


cdef class BeatTrackingDynamicBayesianNetwork(object):
    """
    Dynamic Bayesian network for beat tracking.

    """
    # define some class variables which are also exported as Python attributes
    cdef readonly Transitions transitions
    cdef readonly np.ndarray observations
    cdef readonly unsigned int observation_lambda
    cdef readonly bint norm_observations
    cdef readonly bint correct
    cdef readonly unsigned int num_threads
    cdef readonly double path_probability
    # hidden variable
    cdef np.ndarray _path

    # default values
    OBSERVATION_LAMBDA = 16
    CORRECT = True
    NORM_OBSERVATIONS = False

    def __init__(self, transitions=None, observations=None,
                 observation_lambda=OBSERVATION_LAMBDA,
                 norm_observations=NORM_OBSERVATIONS,
                 correct=CORRECT, num_threads=NUM_THREADS, **kwargs):
        """
        Construct a new dynamic Bayesian network suitable for beat tracking.

        :param transitions:        BeatTrackingTransitions instance or file
        :param observations:       observations
        :param observation_lambda: split one beat period into N parts,
                                   the first representing beat states
                                   and the remaining non-beat states
        :param norm_observations:  normalise the observations
        :param correct:            correct the detected beat positions
        :param num_threads:        number of parallel threads

        :param kwargs:             additional parameters for transition model
                                   creation if no 'transitions' are given

        """
        # save/init the transitions
        if isinstance(transitions, Transitions):
            self.transitions = transitions
        else:
            self.transitions = BeatTrackingTransitions(transitions, **kwargs)
        # save the observations as a contiguous numpy array
        if observations is not None:
            self.observations = np.ascontiguousarray(observations,
                                                     dtype=np.float32)
        # save other parameters
        self.observation_lambda = observation_lambda
        self.num_threads = num_threads
        self.correct = correct
        self.norm_observations = norm_observations

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def viterbi(self, observations):
        """
        Determine the best path given the observations.

        :param observations: the observations (anything that can be casted to
                             a numpy array)
        :return:             most probable state-space path sequence for the
                             given observations

        Note: a uniform prior distribution is assumed.

        """
        # save the given observations as an contiguous array
        self.observations = np.ascontiguousarray(observations)
        # typed memoryview thereof
        # Note: the ::1 notation indicates that is memory contiguous
        cdef float [::1] observations_ = self.observations
        # normalise the observations memoryview, not the observations itself
        if self.norm_observations:
            observations_ = self.observations / np.max(self.observations)
        # number of observations
        cdef unsigned int num_observations = len(observations_)
        # cache variables needed in the loops
        cdef unsigned int observation_lambda = self.observation_lambda
        cdef unsigned int num_threads = self.num_threads
        # number of beat/tempo/total states
        cdef unsigned int num_beat_states = self.transitions.num_beat_states
        cdef unsigned int num_tempo_states = len(self.transitions.tempo_states)
        cdef unsigned int num_states = num_beat_states * num_tempo_states

        # current viterbi variables
        current_viterbi = np.empty(num_states, dtype=np.float)
        # typed memoryview thereof
        cdef double [::1] current_viterbi_ = current_viterbi
        # previous viterbi variables, init them with 1s as prior distribution
        # TODO: allow other priors
        prev_viterbi = np.ones(num_states, dtype=np.float)
        # typed memoryview thereof
        cdef double [::1] prev_viterbi_ = prev_viterbi
        # back-tracking pointers
        back_tracking_pointers = np.empty((num_observations, num_states),
                                           dtype=np.uint32)
        # typed memoryview thereof
        cdef unsigned int [:, ::1] back_tracking_pointers_ = \
            back_tracking_pointers
        # back tracked path, a.k.a. path sequence
        path = np.empty(num_observations, dtype=np.uint32)

        # transition stuff
        cdef unsigned int [::1] states_ = self.transitions.states
        cdef unsigned int [::1] pointers_ = self.transitions.pointers
        cdef double [::1] probabilities_ = self.transitions.probabilities

        # define counters etc.
        cdef unsigned int prev_state, beat_state, pointer
        cdef double obs, transition_prob, viterbi_sum, path_probability = 0.0
        cdef unsigned int beat_no_beat = num_beat_states / observation_lambda
        cdef int state, frame
        # iterate over all observations
        for frame in range(num_observations):
            # search for best transitions
            for state in prange(num_states, nogil=True, schedule='static',
                                num_threads=num_threads):
                # reset the current viterbi variable
                current_viterbi_[state] = 0.0
                # position inside beat & tempo
                beat_state = state % num_beat_states
                # get the observation
                if beat_state < beat_no_beat:
                    obs = observations_[frame]
                else:
                    obs = ((1. - observations_[frame]) /
                           (observation_lambda - 1))
                # iterate over all possible previous states
                for pointer in range(pointers_[state], pointers_[state + 1]):
                    prev_state = states_[pointer]
                    # weight the previous state with the transition
                    # probability and the current observation
                    transition_prob = prev_viterbi_[prev_state] * \
                                      probabilities_[pointer] * obs
                    # if this transition probability is greater than the
                    # current, overwrite it and save the previous state
                    # in the current pointers
                    if transition_prob > current_viterbi_[state]:
                        current_viterbi_[state] = transition_prob
                        back_tracking_pointers_[frame, state] = prev_state
            # overwrite the old states with the normalised current ones
            # Note: this is faster than unrolling the loop! But it is a bit
            #       tricky: we need to do the normalisation on the numpy
            #       array but do the assignment on the memoryview
            viterbi_sum = current_viterbi.sum()
            prev_viterbi_ = current_viterbi / viterbi_sum
            # add the log sum of all viterbi variables to the overall sum
            path_probability += log(viterbi_sum)

        # fetch the final best state
        state = current_viterbi.argmax()
        # add its log probability to the sum
        path_probability += log(current_viterbi.max())
        # track the path backwards, start with the last frame and do not
        # include the back_tracking_pointers for frame 0, since it includes
        # the transitions to the prior distribution states
        for frame in range(num_observations -1, -1, -1):
            # save the state in the path
            path[frame] = state
            # fetch the next previous one
            state = back_tracking_pointers[frame, state]
        # save the tracked path and log sum and return them
        self._path = path
        self.path_probability = path_probability
        return path, path_probability

    @property
    def path(self):
        """Best path sequence."""
        if self._path is None:
            self.viterbi(self.observations)
        return self._path

    @property
    def beat_states_path(self):
        """Beat states path."""
        return self.path % self.transitions.num_beat_states

    @property
    def tempo_states_path(self):
        """Tempo states path."""
        return self.transitions.tempo_states[self.path /
                                             self.transitions.num_beat_states]

    @property
    def beats(self):
        # correct the beat positions
        """The detected beats."""
        if self.correct:
            beats = []
            # for each detection determine the "beat range", i.e. states <=
            # num_beat_states / observation_lambda and choose the frame with
            # the highest observation value
            beat_range = self.beat_states_path < \
                         (self.transitions.num_beat_states /
                          self.observation_lambda)
            # get all change points between True and False
            idx = np.nonzero(np.diff(beat_range))[0] + 1
            # if the first frame is in the beat range, prepend a 0
            if beat_range[0]:
                idx = np.r_[0, idx]
            # if the last frame is in the beat range, append the length of the
            # array
            if beat_range[-1]:
                idx = np.r_[idx, beat_range.size]
            # iterate over all regions
            for left, right in idx.reshape((-1, 2)):
                beats.append(np.argmax(self.observations[left:right]) + left)
            beats = np.asarray(beats)
        else:
            # just take the frames with the smallest beat state values
            from scipy.signal import argrelmin
            beats = argrelmin(self.beat_states_path, mode='wrap')[0]
            # recheck if they are within the "beat range", i.e. the
            # beat states < number of beat states / observation lambda
            # Note: interpolation and alignment of the beats to be at state 0
            #       does not improve results over this simple method
            beats = beats[self.beat_states_path[beats] <
                          (self.transitions.num_beat_states /
                           self.observation_lambda)]
        return beats

    @classmethod
    def add_arguments(cls, parser, observation_lambda=OBSERVATION_LAMBDA,
                      correct=CORRECT, norm_observations=NORM_OBSERVATIONS,
                      num_beat_states=BeatTrackingTransitions.NUM_BEAT_STATES,
                      tempo_states=BeatTrackingTransitions.TEMPO_STATES,
                      tempo_change_probability=
                      BeatTrackingTransitions.TEMPO_CHANGE_PROBABILITY):
        """
        Add dynamic Bayesian network related arguments to an existing parser
        object.

        :param parser:                   existing argparse parser object

        Parameters for the observation model:

        :param observation_lambda:       split one beat period into N parts,
                                         the first representing beat states
                                         and the remaining non-beat states
        :param correct:                  correct the beat positions
        :param norm_observations:        normalise the observations of the DBN

        Parameters for the transition model:

        :param num_beat_states:          number of cells for one beat period
        :param tempo_states:             list with tempo states
        :param tempo_change_probability: probability of a tempo change between
                                         two adjacent observations
        :return:                         beat argument parser group object

        """
        # add a group for DBN parameters
        g = parser.add_argument_group('dynamic Bayesian Network arguments')
        if correct:
            g.add_argument('--no_correct', dest='correct',
                           action='store_false', default=correct,
                           help='do not correct the beat positions')
        else:
            g.add_argument('--correct', dest='correct',
                           action='store_true', default=correct,
                           help='correct the beat positions')
        # observation model stuff
        g.add_argument('--observation_lambda', action='store', type=int,
                       default=observation_lambda,
                       help='split one beat period into N parts, the first '
                            'representing beat states and the remaining '
                            'non-beat states [default=%(default)i]')
        if norm_observations:
            g.add_argument('--no_norm_obs', dest='norm_observations',
                           action='store_false', default=norm_observations,
                           help='do not normalise the observations of the DBN')
        else:
            g.add_argument('--norm_obs', dest='norm_observations',
                           action='store_true', default=norm_observations,
                           help='normalise the observations of the DBN')
        # add a transition parameters
        g.add_argument('--num_beat_states', action='store', type=int,
                       default=num_beat_states,
                       help='number of beat states for one beat period '
                            '[default=%(default)i]')
        g.add_argument('--tempo_change_probability', action='store',
                       type=float, default=tempo_change_probability,
                       help='probability of a tempo between two adjacent '
                            'observations [default=%(default).4f]')
        if tempo_states is not None:
            from ..utils import OverrideDefaultListAction
            g.add_argument('--tempo_states', action=OverrideDefaultListAction,
                           type=int, default=tempo_states,
                           help='possible tempo states (multiple values can '
                                'be given)')
        # return the argument group so it can be modified if needed
        return g
