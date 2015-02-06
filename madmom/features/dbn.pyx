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
import multiprocessing as mp
NUM_THREADS = mp.cpu_count()

cdef extern from "math.h":
    float INFINITY

# transition model stuff
cdef class TransitionModel(object):
    """
    Transition model for a DBN.

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
    # define some variables which are also exported as Python attributes
    cdef public np.ndarray log_probabilities
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
        self.attributes = ['log_probabilities', 'states', 'pointers']
        # init some variables
        self.states = None
        self.pointers = None
        self.log_probabilities = None
        # load the transitions
        if model is not None:
            self.load(model)

    def make_sparse(self, log_probabilities, states, prev_states):
        """
        Method to convert the given transitions to a sparse format.

        Three 1D numpy arrays of same length must be given. The indices
        correspond to each other, i.e. the first entry of all three arrays
        define the transition from the state defined prev_states[0] to that
        defined in states[0] with the log probability defined in
        log_probabilities[0].

        :param log_probabilities: transition log probabilities
        :param states:            corresponding states
        :param prev_states:       corresponding previous states

        This method removes all duplicate states and thus allows for parallel
        processing of the Viterbi of the DBN.

        """
        from scipy.sparse import csr_matrix
        # convert everything into a sparse CSR matrix
        transitions = csr_matrix((log_probabilities, (states, prev_states)))
        # save the sparse matrix as 3 linear arrays
        self.states = transitions.indices.astype(np.uint32)
        self.pointers = transitions.indptr.astype(np.uint32)
        self.log_probabilities = transitions.data.astype(dtype=np.float)

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


cdef class BeatTrackingTransitionModel(TransitionModel):
    """
    Transition model for beat tracking with a DBN.

    """
    # define some variables which are also exported as Python attributes
    cdef public unsigned int num_beat_states
    cdef public np.ndarray tempo_states
    cdef public double tempo_change_probability

    def __init__(self, num_beat_states, tempo_states,
                 tempo_change_probability):
        """
        Construct a transition model instance suitable for beat tracking.

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
        # compute transitions
        self.num_beat_states = num_beat_states
        self.tempo_states = np.ascontiguousarray(tempo_states, dtype=np.uint32)
        self.tempo_change_probability = tempo_change_probability
        # compute the transitions
        transitions = self._transition_model(self.num_beat_states,
                                             self.tempo_states,
                                             self.tempo_change_probability)
        # save them in sparse format
        self.make_sparse(*transitions)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _transition_model(self, unsigned int num_beat_states,
                          unsigned int [::1] tempo_states,
                          double tempo_change_probability):
        """
        Compute the transition log probabilities and return them in a format
        understood by make_sparse().

        :param num_beat_states:          number of states for one beat period
        :param tempo_states:             array with tempo states (number of
                                         beat states to progress from one
                                         observation value to the next one)
        :param tempo_change_probability: probability of a tempo change from
                                         one observation to the next one
        :return:                         tuple with (log_probabilities, states,
                                         prev_states)

        """
        # number of tempo & total states
        cdef unsigned int num_tempo_states = len(tempo_states)
        cdef unsigned int num_states = num_beat_states * num_tempo_states
        # transition probabilities
        cdef double same_tempo_prob = log(1. - tempo_change_probability)
        cdef double change_tempo_prob = log(0.5 * tempo_change_probability)
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
            tempo = tempo_states[tempo_state]
            # for each state check the 3 possible transitions
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
        # return the arrays
        return log_probabilities, states, prev_states

    def beat_state_sequence(self, state_sequence):
        """
        Beat state sequence for the given state sequence.

        :param state_sequence: numpy array with the state sequence
        :return:               corresponding beat state sequence

        """
        return state_sequence % self.num_beat_states

    def tempo_state_sequence(self, state_sequence):
        """
        Tempo state sequence for the given state sequence.

        :param state_sequence: numpy array with the state sequence
        :return:               corresponding tempo state sequence

        """
        return self.tempo_states[state_sequence / self.num_beat_states]


# observation model stuff
cdef class ObservationModel(object):
    """
    Observation model for a DBN.

    An observation model is defined as two plain numpy arrays, densities and
    pointers.

    The 'log_densities' is a 2D numpy array with the number of rows being equal
    to the length of the observations and the columns representing the
    different observation log probability densities. The type must be np.float.

    The 'pointers' is a 1D numpy array and has a length equal to the number of
    states of the DBN and points from each state to the corresponding column
    of the 'log_densities' array. The type must be np.uint32.

    """
    # define some variables which are also exported as Python attributes
    cdef public np.ndarray log_densities
    cdef public np.ndarray pointers

    def __init__(self, log_densities=None, pointers=None):
        """
        Construct a ObservationModel instance for a DBN.

        :param log_densities: numpy array with observation log probability
                              densities
        :param pointers:      numpy array with pointers from DBN states to the
                              correct densities column or the number of DBN
                              states

        If log_densities are 1D, they are converted to a 2D representation with
        only 1 column. If pointers is an integer number, a pointers vector of
        that length is created, pointing always to the first column.

        """
        # densities
        if log_densities is None:
            # init as None
            self.log_densities = None
        elif isinstance(log_densities, (list, np.ndarray)):
            # convert to a 2d numpy array if needed
            if log_densities.ndim == 1:
                log_densities = np.atleast_2d(log_densities).T
            self.log_densities = np.asarray(log_densities, dtype=np.float)
        else:
            raise TypeError('wrong type for log_densities')
        # pointers
        if pointers is None:
            # init as None
            self.pointers = None
        elif isinstance(pointers, int):
            # construct a pointers vector, always pointing to the first column
            self.pointers = np.zeros(pointers, dtype=np.uint32)
        elif isinstance(pointers, np.ndarray):
            # convert to correct type
            self.pointers = pointers.astype(np.uint32)
        else:
            raise TypeError('wrong type for pointers')


cdef class NNBeatTrackingObservationModel(ObservationModel):
    """
    Observation model for NN based beat tracking with a DBN.

    """
    # define some variables which are also exported as Python attributes
    cdef public unsigned int observation_lambda
    cdef public bint norm_observations
    cdef public np.ndarray observations

    def __init__(self, observations, num_states, num_beat_states,
                 observation_lambda, norm_observations=False):
        """
        Construct a observation model instance.

        :param observations:       observations (i.e. activations of the NN)
        :param num_states:         number of DBN states
        :param num_beat_states:    number of DBN beat states
        :param observation_lambda: split one beat period into N parts,
                                   the first representing beat states
                                   and the remaining non-beat states
        :param norm_observations:  normalize the observations

        "A multi-model approach to beat tracking considering heterogeneous
         music styles"
        Sebastian Böck, Florian Krebs and Gerhard Widmer
        Proceedings of the 15th International Society for Music Information
        Retrieval Conference (ISMIR 2014), Taipeh, Taiwan, November 2014

        """
        # instantiate an empty ObservationModel
        super(NNBeatTrackingObservationModel, self).__init__(None, None)
        # convert the given activations to an contiguous array
        self.observations = np.ascontiguousarray(observations, dtype=np.float)
        # normalize the activations
        if norm_observations:
            self.observations /= np.max(self.observations)
        # save the given parameters
        self.observation_lambda = observation_lambda
        self.norm_observations = norm_observations
        # generate the observation model
        self._observation_model(self.observations, num_states, num_beat_states)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _observation_model(self, double [::1] observations,
                           unsigned int num_states,
                           unsigned int num_beat_states):
        """
        Compute the observation model.

        :param observations:    observations (i.e. activations of the NN)
        :param num_states:      number of states
        :param num_beat_states: number of beat states

        """
        # counter, etc.
        cdef unsigned int i
        cdef unsigned int num_observations = len(observations)
        cdef unsigned int observation_lambda = self.observation_lambda
        # init densities
        cdef double [:, ::1] log_densities = np.empty((num_observations, 2),
                                                      dtype=np.float)
        # define the observation states
        for i in range(num_observations):
            log_densities[i, 0] = log(observations[i])
            log_densities[i, 1] = log((1. - observations[i]) /
                                      (observation_lambda - 1))
        # init observation pointers
        cdef unsigned int [:] pointers = np.zeros(num_states, dtype=np.uint32)
        cdef unsigned int observation_border = (num_beat_states /
                                                observation_lambda)
        # define the observation pointers
        for i in range(num_states):
            if (i + num_beat_states) % num_beat_states < observation_border:
                pointers[i] = 0
            else:
                pointers[i] = 1
        # save everything
        self.log_densities = np.asarray(log_densities)
        self.pointers = np.asarray(pointers)


# DBN stuff
cdef class DynamicBayesianNetwork(object):
    """
    Dynamic Bayesian network.

    """
    # define some variables which are also exported as Python attributes
    cdef public TransitionModel transition_model
    cdef public ObservationModel observation_model
    cdef public np.ndarray initial_states
    cdef public unsigned int num_threads
    # hidden variable
    cdef np.ndarray _path
    cdef double _log_probability

    def __init__(self, transition_model=None, observation_model=None,
                 initial_states=None, num_threads=NUM_THREADS):
        """
        Construct a new dynamic Bayesian network.

        :param transition_model:  TransitionModel instance or file
        :param observation_model: ObservationModel instance or observations
        :param initial_states:    initial state distribution; a uniform
                                  distribution is assumed if None is given
        :param num_threads:       number of parallel threads

        """
        # save number of threads
        if num_threads is None:
            num_threads = NUM_THREADS
        self.num_threads = num_threads
        # transition model
        if isinstance(transition_model, TransitionModel):
            # already a TransitionModel
            self.transition_model = transition_model
        else:
            # instantiate a new or load an existing TransitionModel
            self.transition_model = TransitionModel(transition_model)
        num_states = self.transition_model.num_states
        # observation model
        if isinstance(observation_model, ObservationModel):
            # already a ObservationModel
            self.observation_model = observation_model
        else:
            # instantiate a new ObservationModel
            self.observation_model = ObservationModel(observation_model,
                                                      num_states)
        # initial state distribution
        if initial_states is None:
            self.initial_states = np.log(np.ones(num_states, dtype=np.float) /
                                         num_states)
        else:
            self.initial_states = np.ascontiguousarray(initial_states,
                                                       dtype=np.float)
        # init path and its probability
        self._path = None
        self._log_probability = -INFINITY

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline void _best_prev_state(self, int state, int frame,
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

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def viterbi(self):
        """
        Determine the best path with the Viterbi algorithm.

        :return: best state-space path sequence and its log probability

        """
        # transition model stuff
        cdef TransitionModel tm = self.transition_model
        cdef unsigned int [::1] tm_states = tm.states
        cdef unsigned int [::1] tm_pointers = tm.pointers
        cdef double [::1] tm_probabilities = tm.log_probabilities
        cdef unsigned int num_states = tm.num_states

        # observation model stuff
        cdef ObservationModel om = self.observation_model
        cdef double [:, ::1] om_densities = om.log_densities
        cdef unsigned int [::1] om_pointers = om.pointers
        cdef unsigned int num_observations = len(om.log_densities)

        # current viterbi variables
        cdef double [::1] current_viterbi = np.empty(num_states,
                                                     dtype=np.float)
        # previous viterbi variables, init with the initial state distribution
        cdef double [::1] previous_viterbi = self.initial_states

        # back-tracking pointers
        cdef unsigned int [:, ::1] bt_pointers = np.empty((num_observations,
                                                           num_states),
                                                          dtype=np.uint32)
        # back tracked path, a.k.a. path sequence
        self._path = np.empty(num_observations, dtype=np.uint32)
        cdef unsigned int [::1] path = self._path

        # define counters etc.
        cdef int state, frame
        cdef unsigned int prev_state, pointer, num_threads = self.num_threads
        cdef double obs, transition_prob

        # iterate over all observations
        for frame in range(num_observations):
            # range() is faster than prange() for 1 thread
            if num_threads == 1:
                # search for best transitions sequentially
                for state in range(num_states):
                    self._best_prev_state(state, frame, current_viterbi,
                                          previous_viterbi, om_densities,
                                          om_pointers, tm_states, tm_pointers,
                                          tm_probabilities, bt_pointers)
            else:
                # search for best transitions in parallel
                for state in prange(num_states, nogil=True, schedule='static',
                                    num_threads=num_threads):
                    self._best_prev_state(state, frame, current_viterbi,
                                          previous_viterbi, om_densities,
                                          om_pointers, tm_states, tm_pointers,
                                          tm_probabilities, bt_pointers)

            # overwrite the old states with the current ones
            previous_viterbi[:] = current_viterbi

        # fetch the final best state
        state = np.asarray(current_viterbi).argmax()
        # set the path's probability to that of the best state
        self._log_probability = current_viterbi[state]
        # track the path backwards, start with the last frame and do not
        # include the pointer for frame 0, since it includes the transitions
        # to the prior distribution states
        for frame in range(num_observations -1, -1, -1):
            # save the state in the path
            path[frame] = state
            # fetch the next previous one
            state = bt_pointers[frame, state]
        # return the tracked path and its probability
        return self._path, self._log_probability

    @property
    def path(self):
        """Best path sequence."""
        if self._path is None:
            self.viterbi()
        return self._path

    @property
    def log_probability(self):
        """Log probability of the best path."""
        # Note: we need to check the _path, since the _log_probability can't
        #       get initialized as 'None' because of static typing
        if self._path is None:
            self.viterbi()
        return self._log_probability


cdef class BeatTrackingDynamicBayesianNetwork(DynamicBayesianNetwork):
    """
    Dynamic Bayesian network for beat tracking.

    """
    # define some variables which are also exported as Python attributes
    cdef public bint correct

    # shortcuts
    TM = BeatTrackingTransitionModel
    OM = NNBeatTrackingObservationModel

    def __init__(self, transition_model=None, observation_model=None,
                 initial_states=None, correct=True, num_threads=NUM_THREADS):
        """
        Construct a new dynamic Bayesian network suitable for beat tracking.

        :param transition_model:  TransitionModel or file
        :param observation_model: ObservationModel or observations
        :param initial_states:    initial state distribution; a uniform
                                  distribution is assumed if None is given
        :param correct:           correct the detected beat positions
        :param num_threads:       number of parallel threads

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
        super(BeatTrackingDynamicBayesianNetwork, self).__init__(
            transition_model, observation_model, initial_states, num_threads)
        # save other parameters
        self.correct = correct

    @property
    def beat_states_path(self):
        """Beat states path."""
        return self.transition_model.beat_state_sequence(self.path)

    @property
    def tempo_states_path(self):
        """Tempo states path."""
        return self.transition_model.tempo_state_sequence(self.path)

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
                # pick the frame with the highest observations value
                act = self.observation_model.observations[left:right]
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
