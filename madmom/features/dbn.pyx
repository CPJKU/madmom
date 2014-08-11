# encoding: utf-8
"""
This file contains dynamic Bayesian network (DBN) functionality.

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


cdef class TransitionModel(object):
    """
    Transition model for a DBN.

    The transition model is defined similar to a scipy compressed sparse row
    matrix and holds all transition probabilities from one state to an other.

    All state indices for row state s are stored in
    states[pointers[s]:pointers[s+1]]
    and their corresponding probabilities are stored in
    probabilities[pointers[s]:pointers[s+1]].

    This allows for a parallel computation of the viterbi path.

    This class should be either used for loading saved transition models or
    being sub-classed to define a new transition model.

    """
    # define some variables which are also exported as Python attributes
    cdef public np.ndarray probabilities
    cdef public np.ndarray states
    cdef public np.ndarray pointers
    # hidden list with attributes to save/load
    cdef list attributes

    def __init__(self, model=None):
        """
        Construct a TransitionModel instance for DBNs.

        :param model: load the transition model from the given file

        """
        # set the attributes which should be saved or loaded
        self.attributes = ['probabilities', 'states', 'pointers']
        # load the transitions
        if model is not None:
            self.load(model)

    def make_sparse(self, probabilities, states, prev_states):
        """
        Method to convert the given transitions to a sparse format.

        Three 1D numpy arrays of same length must be given. The indices
        correspond with each other, i.e. the first entry of all three arrays
        define the transition from the state defined prev_states[0] to the
        state states[0] with the probability defined in probabilities[0].

        :param probabilities: transition probabilities
        :param states:        corresponding states
        :param prev_states:   corresponding previous states

        This method removes all duplicate states and thus allows for parallel
        processing of the Viterbi of the DBN.

        """
        from scipy.sparse import csr_matrix
        # convert everything into a sparse CSR matrix
        transitions = csr_matrix((probabilities, (states, prev_states)))
        # save the sparse matrix as 3 linear arrays
        self.states = transitions.indices.astype(np.uint32)
        self.pointers = transitions.indptr.astype(np.uint32)
        self.probabilities = transitions.data.astype(dtype=np.float)

    def save(self, outfile, compressed=True):
        """
        Save the transitions to a file.

        :param outfile:    file name or file handle to save the transitions to
        :param compressed: save in compressed format

        """
        # populate the dictionary with attributes to save
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
        if isinstance(infile, np.lib.npyio.NpzFile):
            # file already read
            data = infile
        else:
            # load the .npz file
            data = np.load(infile)
        # load the needed attributes
        for attr in self.attributes:
            self.__setattr__(attr, data[attr])

    @property
    def num_states(self):
        """Number of states."""
        return len(self.pointers) - 1


cdef class ObservationModel(object):
    """
    Observation model for a DBN.

    An observation model is defined as two plain numpy arrays, observations
    and pointers.

    The 'observations' is a 2D numpy array with the number of rows being equal
    to the length of the observations and the columns representing different
    observation probabilities. The type must be np.float.

    The 'pointers' is a 1D numpy array and has a length equal to the number of
    states of the DBN and points from each state to the corresponding column
    of the 'observations' array. The type must be np.uint32.

    """
    # define some variables which are also exported as Python attributes
    cdef public np.ndarray observations
    cdef public np.ndarray pointers

    def __init__(self, observations=None, pointers=None):
        """
        Construct a ObservationModel instance for a DBN.

        :param observations: numpy array with the observations
        :param pointers:     numpy array with pointers from states to the
                             correct observations column or the number of
                             DBN states

        If observations are 1D, they are converted to a 2D representation with
        only 1 column. If pointers is an integer number, a pointers vector of
        that length is created, pointing always to the first column.

        """
        # observations
        if observations is None:
            pass
        elif isinstance(observations, (list, np.ndarray)):
            # convert to a 2d numpy array if needed
            if observations.ndim == 1:
                observations = np.atleast_2d(observations).T
            self.observations = np.asarray(observations, dtype=np.float)
        else:
            raise TypeError('wrong type for observations')
        # pointers
        if pointers is None:
            pass
        elif isinstance(pointers, int):
            # construct a pointers vector, always pointing to the first column
            self.pointers = np.zeros(pointers, dtype=np.uint32)
        elif isinstance(pointers, np.ndarray):
            # convert to correct type
            self.pointers = pointers.astype(np.uint32)
        else:
            raise TypeError('wrong type for pointers')


cdef class DynamicBayesianNetwork(object):
    """
    Dynamic Bayesian network.

    """
    # define some variables which are also exported as Python attributes
    cdef public TransitionModel transition_model
    cdef public ObservationModel observation_model
    cdef public unsigned int num_threads
    cdef public double path_probability
    # hidden variable
    cdef np.ndarray _path

    def __init__(self, transition_model=None, observation_model=None,
                 num_threads=NUM_THREADS):
        """
        Construct a new dynamic Bayesian network.

        :param transition_model:   TransitionModel instance or file
        :param observation_model:  ObservationModel instance or observations
        :param num_threads:        number of parallel threads

        """
        # save number of threads
        self.num_threads = num_threads
        # transition model
        if isinstance(transition_model, TransitionModel):
            # already a TransitionModel
            self.transition_model = transition_model
        else:
            # instantiate a new or load an existing TransitionModel
            self.transition_model = TransitionModel(transition_model)

        # observation model
        if isinstance(observation_model, ObservationModel):
            # already a ObservationModel
            self.observation_model = observation_model
        else:
            # instantiate a new ObservationModel
            num_states = self.transition_model.num_states
            self.observation_model = ObservationModel(observation_model,
                                                      num_states)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def viterbi(self):
        """
        Determine the best path with the Viterbi algorithm

        :return: best state-space path sequence and its log probability

        Note: a uniform prior distribution is assumed.

        """
        # transition model stuff
        cdef unsigned int [::1] tm_states = \
            self.transition_model.states
        cdef unsigned int [::1] tm_pointers = \
            self.transition_model.pointers
        cdef double [::1] tm_probabilities = \
            self.transition_model.probabilities
        cdef unsigned int num_states = self.transition_model.num_states

        # observation model stuff
        cdef double [:, ::1] om_observations = \
            self.observation_model.observations
        cdef unsigned int [::1] om_pointers = \
            self.observation_model.pointers
        cdef unsigned int num_observations = \
            len(self.observation_model.observations)

        # current viterbi variables
        current_viterbi = np.empty(num_states, dtype=np.float)
        cdef double [::1] current_viterbi_ = current_viterbi
        # previous viterbi variables, init them with 1s as prior distribution
        # TODO: allow other priors
        prev_viterbi = np.ones(num_states, dtype=np.float)
        cdef double [::1] prev_viterbi_ = prev_viterbi
        # back-tracking pointers
        pointers = np.empty((num_observations, num_states),dtype=np.uint32)
        cdef unsigned int [:, ::1] pointers_ = pointers
        # back tracked path, a.k.a. path sequence
        path = np.empty(num_observations, dtype=np.uint32)

        # define counters etc.
        cdef int state, frame
        cdef unsigned int prev_state, pointer, num_threads = self.num_threads
        cdef double obs, transition_prob, viterbi_sum, path_probability = 0.0

        # iterate over all observations
        for frame in range(num_observations):
            # search for best transitions
            for state in prange(num_states, nogil=True, schedule='static',
                                num_threads=num_threads):
                # reset the current viterbi variable
                current_viterbi_[state] = 0.0

                # get the observations
                # the om_pointers array holds pointers to the correct
                # observation value for the actual state (i.e. column in the
                # om_observations array)
                obs = om_observations[frame, om_pointers[state]]

                # iterate over all possible previous states
                # the tm_pointers array holds pointers to the states which are
                # stored in the tm_states array
                for pointer in range(tm_pointers[state],
                                     tm_pointers[state + 1]):
                    prev_state = tm_states[pointer]
                    # weight the previous state with the transition
                    # probability and the current observation
                    transition_prob = prev_viterbi_[prev_state] * \
                                      tm_probabilities[pointer] * obs
                    # if this transition probability is greater than the
                    # current, overwrite it and save the previous state
                    # in the current pointers
                    if transition_prob > current_viterbi_[state]:
                        current_viterbi_[state] = transition_prob
                        pointers_[frame, state] = prev_state
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
        # include the pointer for frame 0, since it includes the transitions
        # to the prior distribution states
        for frame in range(num_observations -1, -1, -1):
            # save the state in the path
            path[frame] = state
            # fetch the next previous one
            state = pointers[frame, state]
        # save the tracked path and log sum and return them
        self._path = path
        self.path_probability = path_probability
        return path, path_probability

    @property
    def path(self):
        """Best path sequence."""
        if self._path is None:
            self.viterbi()
        return self._path


# sub-class the previously defined classes for beat tracking
cdef class BeatTrackingTransitionModel(TransitionModel):
    """
    Transition model for beat tracking with a DBN.

    """
    # define some variables which are also exported as Python attributes
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
        Construct a transition model instance suitable for beat tracking.

        :param model: load the transition model from the given file

        If no model was given, the object is constructed with the following
        parameters:

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
        Retrieval Conference (ISMIR 2014), Taipeh, Taiwan, November 2014

        """
        # instantiate an empty TransitionModel object
        super(BeatTrackingTransitionModel, self).__init__(None)
        # define additional attributes to be saved or loaded
        self.attributes.extend(['num_beat_states', 'tempo_states',
                                'tempo_change_probability'])
        # load a model or compute transitions
        if model:
            self.load(model)
        else:
            # save the given parameters
            self.num_beat_states = num_beat_states
            self.tempo_states = np.ascontiguousarray(tempo_states,
                                                     dtype=np.int32)
            self.tempo_change_probability = tempo_change_probability
            # compute the transition probabilities
            self._transition_model(self.num_beat_states, self.tempo_states,
                             self.tempo_change_probability)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _transition_model(self, unsigned int num_beat_states,
                          int [::1] tempo_states,
                          double tempo_change_probability):
        """
        Compute the transition probabilities and save them in the correct
        format.

        :param num_beat_states:          number of beat states for one beat
                                         period
        :param tempo_states:             array with tempo states (number of
                                         beat states to progress from one
                                         observation value to the next one)
        :param tempo_change_probability: probability of a tempo change from
                                         one observation to the next one
        """
        # number of tempo & total states
        cdef unsigned int num_tempo_states = len(tempo_states)
        cdef unsigned int num_states = num_beat_states * num_tempo_states
        # transition probabilities
        cdef double same_tempo_prob = 1. - tempo_change_probability
        cdef double change_tempo_prob = 0.5 * tempo_change_probability
        # counters etc.
        cdef unsigned int state, prev_state, beat_state, tempo_state, tempo
        # number of transition states
        # num_tempo_states * 3 because every state has a transition from the
        # same tempo and from the slower and faster one, -2 because the slowest
        # and the fastest tempi can't have transitions from outside the tempo
        # range
        cdef int num_transition_states = (num_beat_states *
                                          (num_tempo_states * 3 - 2))
        # arrays for transitions matrix creation
        states = np.empty(num_transition_states, np.uint32)
        prev_states = np.empty(num_transition_states, np.uint32)
        probabilities = np.empty(num_transition_states, np.float)
        cdef unsigned int [::1] states_ = states
        cdef unsigned int [::1] prev_states_ = prev_states
        cdef double [::1] probabilities_ = probabilities
        cdef int i = 0
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
            states_[i] = state
            prev_states_[i] = prev_state
            probabilities_[i] = same_tempo_prob
            i += 1
            # transition from slower tempo
            if tempo_state > 0:
                # previous state with slower tempo
                prev_state = ((beat_state + num_beat_states -
                               (tempo - 1)) % num_beat_states +
                              ((tempo_state - 1) * num_beat_states))
                # probability for transition from slower tempo
                states_[i] = state
                prev_states_[i] = prev_state
                probabilities_[i] = change_tempo_prob
                i += 1
            # transition from faster tempo
            if tempo_state < num_tempo_states - 1:
                # previous state with faster tempo
                # Note: we add num_beat_states before the modulo operation
                #       so that it can be computed in C (which is faster)
                prev_state = ((beat_state + num_beat_states -
                               (tempo + 1)) % num_beat_states +
                              ((tempo_state + 1) * num_beat_states))
                # probability for transition from faster tempo
                states_[i] = state
                prev_states_[i] = prev_state
                probabilities_[i] = change_tempo_prob
                i += 1
        # make it sparse
        self.make_sparse(probabilities, states, prev_states)


cdef class NNBeatTrackingObservationModel(ObservationModel):
    """
    Observation model for NN based beat tracking with a DBN.

    """
    # define some variables which are also exported as Python attributes
    cdef public unsigned int observation_lambda
    cdef public bint norm_observations
    cdef public np.ndarray activations

    # default values for beat tracking
    OBSERVATION_LAMBDA = 16
    NORM_OBSERVATIONS = False

    def __init__(self, activations, num_states, num_beat_states,
                 observation_lambda=OBSERVATION_LAMBDA,
                 norm_observations=NORM_OBSERVATIONS):
        """
        Construct a observation model instance.

        :param activations:        neural network activations
        :param num_states:         number of DBN states
        :param num_beat_states:    number of DBN beat states
        :param observation_lambda: split one beat period into N parts,
                                   the first representing beat states
                                   and the remaining non-beat states
        :param norm_observations:  normalise the observations

        "A multi-model approach to beat tracking considering heterogeneous
         music styles"
        Sebastian Böck, Florian Krebs and Gerhard Widmer
        Proceedings of the 15th International Society for Music Information
        Retrieval Conference (ISMIR 2014), Taipeh, Taiwan, November 2014

        """
        # instantiate an empty ObservationModel
        super(NNBeatTrackingObservationModel, self).__init__(None, None)
        # convert the given activations to an contiguous array
        self.activations = np.ascontiguousarray(activations, dtype=np.float)
        # normalise the activations
        if norm_observations:
            self.activations /= np.max(self.activations)
        # save the given parameters
        self.observation_lambda = observation_lambda
        self.norm_observations = norm_observations
        # generate the observation model
        self._observation_model(self.activations, num_states, num_beat_states)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _observation_model(self, double [::1] activations,
                           unsigned int num_states,
                           unsigned int num_beat_states):
        """
        Compute the observation model.

        :param activations:     neural network activations
        :param num_states:      number of states
        :param num_beat_states: number of beat states

        """
        # counter, etc.
        cdef unsigned int i
        cdef unsigned int num_observations = len(activations)
        cdef unsigned int observation_lambda = self.observation_lambda
        # init observations
        observations = np.empty((num_observations, 2), dtype=np.float)
        cdef double [:, ::1] observations_ = observations
        # define the observation states
        for i in range(num_observations):
            observations_[i, 0] = activations[i]
            observations_[i, 1] = ((1. - activations[i]) /
                                   (observation_lambda - 1))
        # init observation pointers
        pointers = np.zeros(num_states, dtype=np.uint32)
        cdef unsigned int [:] pointers_ = pointers
        cdef unsigned int observation_border = (num_beat_states /
                                                observation_lambda)
        # define the observation pointers
        for i in range(num_states):
            if (i + num_beat_states) % num_beat_states < observation_border:
                pointers_[i] = 0
            else:
                pointers_[i] = 1
        # save everything
        self.observations = observations
        self.pointers = pointers


cdef class BeatTrackingDynamicBayesianNetwork(DynamicBayesianNetwork):
    """
    Dynamic Bayesian network for beat tracking.

    """
    # define some variables which are also exported as Python attributes
    cdef public bint correct

    # default values
    CORRECT = True
    TM = BeatTrackingTransitionModel
    OM = NNBeatTrackingObservationModel

    def __init__(self, transition_model=None, observation_model=None,
                 correct=CORRECT, num_threads=NUM_THREADS):
        """
        Construct a new dynamic Bayesian network suitable for beat tracking.

        :param transition_model:   TransitionModel or file
        :param observation_model:  ObservationModel or activations
        :param correct:            correct the detected beat positions
        :param num_threads:        number of parallel threads

        "A multi-model approach to beat tracking considering heterogeneous
         music styles"
        Sebastian Böck, Florian Krebs and Gerhard Widmer
        Proceedings of the 15th International Society for Music Information
        Retrieval Conference (ISMIR 2014), Taipeh, Taiwan, November 2014

        """
        # init the transition model
        if not isinstance(transition_model, BeatTrackingTransitionModel):
            transition_model = BeatTrackingTransitionModel(transition_model)

        # init the observation model
        if not isinstance(observation_model, NNBeatTrackingObservationModel):
            observation_model = NNBeatTrackingObservationModel(
                observation_model, transition_model.num_states,
                transition_model.num_beat_states)

        # instantiate DBN
        spr = super(BeatTrackingDynamicBayesianNetwork, self)
        spr.__init__(transition_model, observation_model, num_threads)
        # save other parameters
        self.correct = correct

    @property
    def beat_states_path(self):
        """Beat states path."""
        return self.path % self.transition_model.num_beat_states

    @property
    def tempo_states_path(self):
        """Tempo states path."""
        states = self.path / self.transition_model.num_beat_states
        return self.transition_model.tempo_states[states]

    @property
    def beats(self):
        """The detected beats."""
        # correct the beat positions
        if self.correct:
            beats = []
            # for each detection determine the "beat range", i.e. states <=
            # num_beat_states / observation_lambda and choose the frame with
            # the highest observation value
            beat_range = self.beat_states_path < \
                         (self.transition_model.num_beat_states /
                          self.observation_model.observation_lambda)
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
                # pick the frame with the highest activations value
                act = self.observation_model.activations[left:right]
                beats.append(np.argmax(act) + left)
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
                          (self.transition_model.num_beat_states /
                           self.observation_model.observation_lambda)]
        return beats

    @classmethod
    def add_arguments(cls, parser, correct=CORRECT,
                      num_beat_states=TM.NUM_BEAT_STATES,
                      tempo_states=TM.TEMPO_STATES,
                      tempo_change_probability=TM.TEMPO_CHANGE_PROBABILITY,
                      observation_lambda=OM.OBSERVATION_LAMBDA,
                      norm_observations=OM.NORM_OBSERVATIONS):
        """
        Add dynamic Bayesian network related arguments to an existing parser
        object.

        :param parser:                   existing argparse parser object
        :param correct:                  correct the beat positions

        Parameters for the transition model:

        :param num_beat_states:          number of cells for one beat period
        :param tempo_states:             list with tempo states
        :param tempo_change_probability: probability of a tempo change between
                                         two adjacent observations

        Parameters for the observation model:

        :param observation_lambda:       split one beat period into N parts,
                                         the first representing beat states
                                         and the remaining non-beat states
        :param norm_observations:        normalise the observations

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
        # return the argument group so it can be modified if needed
        return g
