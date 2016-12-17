# encoding: utf-8
"""
This file contains downbeat tracking related functionality.

"""

from __future__ import absolute_import, division, print_function

import numpy as np
from madmom.utils import match_file
from madmom.processors import Processor, ParallelProcessor, SequentialProcessor

# from madmom.audio.chroma import CLPChromaProcessor
from madmom.features.beats import (RNNBeatProcessor, DBNBeatTrackingProcessor)
from madmom.features import ActivationsProcessor
from madmom.features.beats_hmm import RNNBeatTrackingObservationModel

import pickle
import os.path
from madmom.ml.hmm import ObservationModel
# from ..models import DOWNBEATS_BGRU


class BeatSyncProcessor_gmm(ParallelProcessor):
    """
    Resample features to be beat synchronous.

    """
    def __init__(self, feature, beats, beat_subdivision=16,
                 sum_func=np.max, fps=None, **kwargs):
        """
        Creates a new BeatSyncProcessor instance.

        :param feature: list of processors to compute features, which are
                        processed in sequential order
        :param beats:   processor that yields beat times
        :param beat_subdivision:
        :param sum_func:

        """
        self.fps = fps
        self.beat_subdivision = beat_subdivision
        self.sum_func = sum_func
        # FIXME: read fps from FeatureProcessor!
        # Set up parallel processor with feature and beat processor
        # First, the feature processor
        processors = [feature, [beats]]
        # FIXME: works only with a single thread
        super(BeatSyncProcessor_gmm, self).__init__(processors, num_threads=1)

    def process(self, data):
        """
        Compute features, get beats and sync features to beats.

        :param data: Audio filename
        :return:     list of [beat_sync_features, beat times]

        """
        # process everything, returns a list [features, beats]
        data = super(BeatSyncProcessor_gmm, self).process(data)
        return self.sync_features(data)

    def interpolate_missing(self, feats_sorted, feat_dim):
        nan_fill = np.empty(feat_dim) * np.nan
        means = np.array([np.mean(x, axis=0) if len(x) > 0 else nan_fill
                          for x in feats_sorted])
        good_rows = np.unique(np.where(np.logical_not(np.isnan(means)))[0])
        if len(good_rows) < self.beat_subdivision:
            bad_rows = np.unique(np.where(np.isnan(means))[0])
            # initialise missing values with empty array
            for r in bad_rows:
                feats_sorted[r] = np.empty((1, feat_dim))
            for d in range(feat_dim):
                means_p = np.interp(np.arange(0, self.beat_subdivision),
                                    good_rows,
                                    means[good_rows, d])
                for r in bad_rows:
                    feats_sorted[r][0, d] = means_p[r]
        return feats_sorted

    def sync_features(self, data):

        """
        This function organises the features of one song according to the
        given beats. First, a beat interval is divided into <beat_subdivision>
        divisions. Then all features that fall into one subdivision are
        summarised by a <summarise> function. If no feature value for a
        subdivision is found, it is interpolated.

        :param data:                list of two elements: 1) are
                                    the features e.g., MultiBandSpectrogram
                                    2) numpy array of beat times
        :return: beat_features      numpy array of features in beat sync
                                    [num_beats x beat_subdivision x feat_dim]
        :return: beat_div           numpy array of beat subdivision for each
                                    beat
        """
        features = data[0]
        # beats can be 1d (only beat times), or 2d (beat and beat counter)
        if data[1].ndim == 1:
            beats = np.atleast_2d(data[1]).T
        else:
            beats = data[1]
        if beats.size == 0:
            return np.array([]), np.array([])
        while (float(len(features)) / self.fps) < beats[-1, 0]:
            beats = beats[:-1, :]
            print('WARNING: Beat sequence too long compared to feature '
                             'sequence')
        if features.ndim > 1:
            feat_dim = features.shape[1]
        else:
            # last singleton dimension is deleted by activations.load. Here,
            #  we re-introduce it
            features = features[:, np.newaxis]
            feat_dim = 1

        n_beats = beats.shape[0]
        beat_features = np.empty((n_beats - 1, self.beat_subdivision,
                                  feat_dim))
        #beat_div = np.empty(((n_beats - 1) * self.beat_subdivision),
        # dtype=int)
        # allow the note on the first beat to be 20 ms too early
        first_next_beat_frame = int(
            np.max([0, np.floor((beats[0, 0] - 0.02) * self.fps)]))
        for i_beat in range(n_beats - 1):
            duration_beat = beats[i_beat + 1, 0] - beats[i_beat, 0]
            # gmms should be centered on the annotations and cover a
            # timespan that is duration_bar/num_gmms_per_bar. Subtract half of
            # this timespan to get the start frame of the first gmm of this
            # and the next bar
            offset = 0.5 * duration_beat / self.beat_subdivision
            # first frame of first gmm that corresponds to current beat
            beat_start_frame = first_next_beat_frame
            # first frame of first gmm that corresponds to next beat
            first_next_beat_frame = int(np.floor(
                (beats[i_beat + 1, 0] - offset) * self.fps))
            # set up array with time in sec of each frame center. last frame
            #  is the last frame of the last gmm of the current bar
            n_beat_frames = first_next_beat_frame - beat_start_frame
            # we need to put each feature frame in its corresponding
            # beat subdivison. We use a hack to get n_beat_frames equally
            # distributed bins between 0 and (self.beat_subdivision-1)
            beat_pos_of_frames = np.floor(
                np.linspace(0.00000001, self.beat_subdivision - 0.00000001,
                            n_beat_frames))
            features_beat = features[beat_start_frame:first_next_beat_frame]
            # group features in list by gmms (bar position)
            feats_sorted = [features_beat[beat_pos_of_frames == x] for x in
                            np.arange(0, self.beat_subdivision)]
            feats_sorted = self.interpolate_missing(feats_sorted, feat_dim)
            #beat_pos_idx = np.arange(i_beat * self.beat_subdivision,
            #                         (i_beat + 1) * self.beat_subdivision)
            beat_features[i_beat, :, :] = np.array([self.sum_func(x, axis=0)
                                                    for x in feats_sorted])
            # beat_div[beat_pos_idx] = np.arange(1, self.beat_subdivision + 1)
        return beat_features, beats


class BeatSyncProcessor(Processor):
    """
    Synchronize features to the beat.

    """
    def __init__(self, beat_subdivision=4, fps=100):
        """
        Creates a new BeatSyncProcessor instance.

        Parameters
        ----------

        fps :
            frames per second

        beat_subdivision :
            number of divisions of the beat

        sum_func :
            function to summarise the features that belong to the same beat
            subdivision.

        """
        self.beat_subdivision = beat_subdivision
        # sum up the feature values of one beat division
        self.feat_sum = 0.
        # count the frames since last beat division
        self.frame_counter = 0
        # length of one beat division in audio frames (depends on tempo)
        self.div_frames = 0
        self.fps = fps
        # store last beat time to compute tempo
        self.last_beat_time = 0
        # min beat period in frames
        self.min_beat_length = 5

    def process(self, data):
        """
        This function organises the features of a piece according to the
        given beats. First, a beat interval is divided into <beat_subdivision>
        divisions. Then all features that fall into one subdivision are
        summarised by a <summarise> function. If no feature value for a
        subdivision is found, it is interpolated.

        Parameters
        ----------
        data : list
            data[0]: features e.g., MultiBandSpectrogram shape(feat_dim)
            data[1]: beat position
            data[2]: beat interval

        Returns
        -------
        feats : numpy array [num_beats - 1, beat_subdivision * feat_dim]
            Beat synchronous features.

        """
        beat_time, features = data
        self.feat_sum += features
        self.frame_counter += 1
        is_beat = beat_time is not None
        mean_feat = None
        if (self.frame_counter == self.div_frames) or is_beat:
            if self.frame_counter > self.min_beat_length:
                mean_feat = self.feat_sum / self.frame_counter
                # update tempo and prevent beat detections
                if is_beat and (self.frame_counter > self.min_beat_length):
                    beat_interval = beat_time - self.last_beat_time
                    # update beat division because of potential new tempo
                    self.div_frames = np.round(beat_interval * self.fps /
                                               self.beat_subdivision)
                    # store last beat
                    self.last_beat_time = beat_time
            # reset buffer
            self.feat_sum = 0.
            self.frame_counter = 0
        return beat_time, mean_feat


class LoadBeatsProcessor(Processor):
    """
    Load beat times from file.

    """
    def __init__(self, beat_files, beat_suffix):
        """
        Creates a new LoadBeatProcessor instance.

        Parameters
        ----------
        beat_files :
            List of beat filenames.
        beat_suffix :
            Extension of beat filenames.

        """
        self.beat_files = beat_files
        self.beat_suffix = beat_suffix

    def process(self, data):
        """
        Load beat times from file. First match the given input filename to
        the pool of beat filenames, then load the beats.

        Parameters
        ----------
        data :
            Input file name.

        Returns
        -------
        beats : numpy array
            Beat positions [seconds].

        """
        if type(data) is not str:
            raise ValueError('Please supply a filename.')

        # select the filename among the beat file pool that matches the
        # input file
        basename, ext = os.path.splitext(os.path.basename(data))
        matches = match_file(basename, self.beat_files,
                             suffix=ext, match_suffix=self.beat_suffix,
                             match_exactly=True)
        beat_fln = matches[0]
        # TODO: Use load_beats function
        beats = np.loadtxt(beat_fln)
        if beats.ndim == 2:
            # only use beat times
            beats = beats[:, 0]
        return beats

    @classmethod
    def add_arguments(cls, parser):
        """
        Add options to save/load beats to an existing parser.

        Parameters
        ----------
        parser :
            existing argparse parser

        Returns
        -------
            beat parameter argument parser group

        """
        # add onset detection related options to the existing parser
        g = parser.add_argument_group('save/load the beat times')
        # add options for saving and loading the activations
        g.add_argument('--save_beats', action='store_true', default=False,
                       help='save the beats to file')
        g.add_argument('--load_beats', action='store_true', default=False,
                       help='load the beats from file')
        # return the argument group so it can be modified if needed
        return g


class DBNBarTrackingProcessor(Processor):
    """
    Downbeat tracking with a dynamic Bayesian network (DBN).

    """

    def __init__(self, observation_param=100, downbeats=False,
                 pattern_change_prob=0.0, beats_per_bar=[3, 4],
                 observation_model=RNNBeatTrackingObservationModel,
                 **kwargs):
        """
        Track the downbeats with a Dynamic Bayesian Network (DBN).

        Parameters
        ----------
        observation_lambda :
            weight for the activations at downbeat times.

        downbeats : bool, optional
            Return only the downbeat times. If false, return beat times and
            their respective position within the bar)

        pattern_change_prob :
            probability of a change in time signature.

        beats_per_bar :
            number of beats per bar to be modeled by the DBN
        """

        from madmom.ml.hmm import HiddenMarkovModel as Hmm
        from .beats_hmm import (BarStateSpace, BarTransitionModel,
                                MultiPatternStateSpace,
                                MultiPatternTransitionModel)
        self.num_beats = beats_per_bar
        num_patterns = len(self.num_beats)
        # save additional variables
        self.downbeats = downbeats
        # state space
        # tempo not relevant, use one tempo state per pattern
        num_tempo_states = [1] * len(self.num_beats)
        min_interval = [1] * len(self.num_beats)
        max_interval = min_interval
        # lambda not relevant, pass any value
        transition_lambda = 1
        state_spaces = []
        transition_models = []
        for p in range(num_patterns):
            # model each rhythmic pattern as a bar
            st = BarStateSpace(self.num_beats[p], min_interval[p],
                               max_interval[p], num_tempo_states[p])
            tm = BarTransitionModel(st, transition_lambda)
            state_spaces.append(st)
            transition_models.append(tm)
        self.st = MultiPatternStateSpace(state_spaces)
        self.tm = MultiPatternTransitionModel(
            transition_models, pattern_change_prob=pattern_change_prob)
        # observation model
        self.om = observation_model(
            self.st, observation_param)
        # instantiate a HMM
        self.hmm = Hmm(self.tm, self.om, None)

    def infer_states(self, activations):
        # get the best state path by calling the viterbi algorithm
        path, _ = self.hmm.viterbi(activations)
        return path

    def process(self, data):
        """
        Decode the beats/downbeats from the given activation function.

        Parameters
        ----------
        data : list of numpy arrays
            data[0] : activation function
            data[1] : beat times

        Returns
        -------
        numpy array
            beat or downbeat times

        """
        # if data[0].ndim == 1:
        activations = data[0]
        # else:
        #     activations = data[0][:, 0]
        beats = data[1]
        if beats.size == 0:
            return np.array([])
        # get the best state path by calling the inference algorithm
        path = self.infer_states(activations)
        # get the position inside the bar
        position = self.st.state_positions[path]
        # the beat numbers are the counters + 1 at the transition points
        beat_numbers = position.astype(int) + 1
        # add the last beat
        last_beat_number = np.mod(beat_numbers[-1], self.num_beats[
            self.st.state_patterns[path[-1]]]) + 1
        beat_numbers = np.append(beat_numbers, last_beat_number)
        # return the downbeats or beats and their beat numbers
        print(np.squeeze(beats[np.where(beat_numbers == 1), 0]))
        if self.downbeats:
            return np.squeeze(beats[np.where(beat_numbers == 1)])
        else:
            return np.vstack(zip(beats, beat_numbers))

    @classmethod
    def add_arguments(cls, parser, observation_weight=100,
                      beats_per_bar=[3, 4]):
        """
        Add DBN related arguments to an existing parser.

        Parameters
        ----------
        parser :
            existing argparse parser

        observation_weight :
            weight for the activations at downbeat times.

        beats_per_bar :
            number of beats per bar to be modeled by the DBN

        Returns
        -------
        argparse parser

        """
        from madmom.utils import OverrideDefaultListAction
        parser = parser.add_argument_group('DBN parameters')
        parser.add_argument('--beats_per_bar',
                            action=OverrideDefaultListAction,
                            default=beats_per_bar, type=int, sep=',',
                            help='number of beats per bar to be modeled (comma'
                            ' separated (no spaces!) list of bar length in '
                            'beats) [default=%(default)s]')
        parser.add_argument('--observation_weight', action='store',
                            type=int,
                            default=observation_weight, help='split one '
                                                                'beat'
                            ' period into N parts, the first '
                            'representing beat states and the remaining '
                            'non-beat states [default=%(default)i]')
        parser.add_argument('--pattern_change_prob', type=float, default=1e-7,
                            help='pattern change probability '
                            '[default=%(default).7f]', action='store')
        # add output format stuff
        parser = parser.add_argument_group('output arguments')
        parser.add_argument('--downbeats', action='store_true', default=False,
                            help='output only the downbeats')
        # return the argument group so it can be modified if needed
        return parser


class OnlineDBNBarTrackingProcessor(DBNBarTrackingProcessor):

    def __init__(self, observation_lambda=100, downbeats=False,
                 pattern_change_prob=0.0, beats_per_bar=[3, 4],
                 debug=False, **kwargs):
        """
        Track the downbeats with a Dynamic Bayesian Network (DBN).

        Parameters
        ----------
        observation_lambda :
            weight for the activations at downbeat times.

        downbeats : bool, optional
            Return only the downbeat times. If false, return beat times and
            their respective position within the bar)

        pattern_change_prob :
            probability of a change in time signature.

        beats_per_bar :
            number of beats per bar to be modeled by the DBN
        """
        # call init of the superclass
        super(OnlineDBNBarTrackingProcessor, self).__init__(
            observation_lambda=100, downbeats=False, pattern_change_prob=0.0,
            beats_per_bar=[3, 4], **kwargs)
        self.debug = debug

    def infer_states(self, activations):
        # get the filtering distributions by calling the forward algorithm
        fwd = self.hmm.forward_generator(activations)
        # loop through frames to select the "winning state"
        path = np.zeros((activations.shape[0]), dtype=int)
        if self.debug:
            fwd_mat = np.zeros((activations.shape[0], self.st.num_states))
        for i, f in enumerate(fwd):
            path[i] = np.argmax(f)
            if self.debug:
                fwd_mat[i, :] = f
        if self.debug:
            npz = {'fwd': fwd_mat, 'positions': self.st.state_positions,
                   'path': path}
            np.savez('/tmp/debug_info_OnlineDBNBarTracker.npz', **npz)
        return path

# TODO: this should be loaded from the model file!
SUB_DIVISIONS = [4, 2]


class DownbeatFeatureProcessor(Processor):

    def __init__(self, fps=100, num_bands=12):
        from ..audio.spectrogram import (
            FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
            SpectrogramDifferenceProcessor)
        from ..audio.signal import SignalProcessor, FramedSignalProcessor
        # percussive feature
        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        frames = FramedSignalProcessor(frame_size=2048, fps=fps)
        spec = FilteredSpectrogramProcessor(
            num_bands=num_bands, fmin=60., fmax=17000., norm_filters=True)
        log_spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
        diff = SpectrogramDifferenceProcessor(
            diff_ratio=0.5, positive_diffs=True, sum_diffs=True)
        self.feat_processor = SequentialProcessor((sig, frames, spec, log_spec,
                                                  diff))

    def process(self, data):
        """
        Compute features.

        Parameters
        ----------
        data :
            audio filename

        Returns
        -------
        numpy array
            feature values

        """
        return self.feat_processor(data)


class GMMBarProcessor(Processor):
    """
    Processor to get a downbeat activation function from multiple RNNs.

    """
    def __init__(self, fps=100, pattern_files=None, downbeats=False,
                 pattern_change_prob=0., **kwargs):
        # load the patterns
        patterns = []
        for pattern_file in pattern_files:
            with open(pattern_file, 'rb') as f:
                # Python 2 and 3 behave differently
                # TODO: use some other format to save the GMMs (.npz, .hdf5)
                try:
                    # Python 3
                    patterns.append(pickle.load(f, encoding='latin1'))
                except TypeError:
                    # Python 2 doesn't have/need the encoding
                    patterns.append(pickle.load(f))
        if len(patterns) == 0:
            raise ValueError('at least one rhythmical pattern must be given.')
        # extract the GMMs and number of beats
        gmms = [p['gmms'] for p in patterns]
        self.num_beats = [p['num_beats'] for p in patterns]
        # self.feat_means = patterns[0]['feat_means']
        # self.feat_stds = patterns[0]['feat_stds']
        self.num_beat_divisions = [len(g[0]) for g in gmms]
        observation_model = GMMDownBeatTrackingObservationModel
        self.dbn_processor = DBNBarTrackingProcessor(
            observation_param=gmms, beats_per_bar=self.num_beats,
            observation_model=observation_model, **kwargs)

    def process(self, data):
        """
        Compute features, get beats, sync features to beats, compute RNN
        activations and average them.

        Parameters
        ----------
        data[0] : mean_feat
        data[1] : beat_time

        Returns
        -------
        numpy array
            downbeat activation function

        """
        # extract features, compute beats, and sync features to the beats
        return self.dbn_processor(data)

    @classmethod
    def add_arguments(cls, parser, pattern_files=None):
        """
        Add HMM related arguments to an existing parser.

        :param parser:            existing argparse parser

        Parameters for the patterns (i.e. fitted GMMs):

        :param pattern_files:     load the patterns from these files

        Parameters for the transition model:

        Each of the following arguments expect a list with as many items as
        rhythmic patterns.

        :param min_bpm:           list with minimum tempi used for tracking
        :param max_bpm:           list with maximum tempi used for tracking
        :param num_tempo_states:  list with number of tempo states (if set,
                                  limit the number of states and use a log
                                  spacing, otherwise a linear spacing)
        :param transition_lambda: list with lambdas for the exponential tempo
                                  change distribution (higher values prefer a
                                  constant tempo over a tempo change from one
                                  bar to the next one)

        Parameters for the observation model:

        :param norm_observations: normalize the observations

        :return:                  downbeat argument parser group

        """
        from madmom.utils import OverrideDefaultListAction
        # add GMM options
        if pattern_files is not None:
            g = parser.add_argument_group('GMM arguments')
            g.add_argument('--pattern_files', action=OverrideDefaultListAction,
                           default=pattern_files,
                           help='load the patterns (with the fitted GMMs) '
                                'from these files (comma separated list)')
        # add HMM parser group
        # add output format stuff
        g = parser.add_argument_group('output arguments')
        g.add_argument('--downbeats', action='store_true', default=False,
                       help='output only the downbeats')
        # return the argument group so it can be modified if needed
        return g


class GMMDownBeatTrackingObservationModel(ObservationModel):
    """
    Observation model for GMM based beat tracking with a HMM.

    """

    def __init__(self, state_space, gmms, obslik_floor=None):
        """
        Construct a observation model instance using Gaussian Mixture Models
        (GMMs).

        :param gmms:              list with fitted GMM(s), one entry per
                                  rhythmic pattern
        :param state_space:  DownBeatTrackingTransitionModel instance
        :param norm_observations: normalize the observations

        "Rhythmic Pattern Modeling for Beat and Downbeat Tracking in Musical
         Audio"
        Florian Krebs, Sebastian Böck and Gerhard Widmer
        Proceedings of the 15th International Society for Music Information
        Retrieval Conference (ISMIR), 2013

        """
        self.gmms = gmms
        self.state_space = state_space
        # set observation likelihood floor
        self.obslik_floor = obslik_floor
        if obslik_floor is not None:
            self.obslik_floor = np.log(self.obslik_floor)
        # define the pointers of the log densities
        pointers = np.zeros(state_space.num_states, dtype=np.uint32)
        states = np.arange(self.state_space.num_states)
        pattern = self.state_space.state_patterns[states]
        position = self.state_space.state_positions[states]
        # Note: the densities of all GMMs are just stacked on top of each
        #       other, so we have to to keep track of the total number of GMMs
        densities_idx_offset = 0
        for p in range(len(gmms)):
            # distribute the observation densities defined by the GMMs
            # uniformly across the entire state space (for this pattern)
            # since the densities are just stacked, add the offset
            pointers[pattern == p] = (position[pattern == p] +
                                      densities_idx_offset)
            # number of beats for this pattern; for each beat an observation
            # likelihood is computed that encompasses all subdivision of
            # the beat
            num_beats = len(gmms[p])
            # increase the offset by the number of GMMs
            densities_idx_offset += num_beats
        # instantiate a ObservationModel with the pointers
        super(GMMDownBeatTrackingObservationModel, self).__init__(pointers)

    def log_densities(self, observations):
        """
        Computes the log densities of the observations using (a) GMM(s).

        :param observations: observations (i.e. activations of the NN)
                             [n_beats, n_subdivisions, feat_dim]
        :return:             log densities of the observations

        """
        # counter, etc.
        num_observations = len(observations)
        num_patterns = len(self.gmms)
        num_states = 0
        # maximum number of GMMs of all patterns
        for i in range(num_patterns):
            num_states += len(self.gmms[i])
        # init the densities
        log_densities = np.empty((num_observations, num_states), dtype=np.float)
        # define the observation densities
        c = 0
        for i in range(num_patterns):
            for j in range(len(self.gmms[i])): # over beat positions
                log_densities[:, c] = 0
                for bd in range(len(self.gmms[i][j])): # over beat divisions
                    # get the predictions of each GMM for the observations
                    log_densities[:, c] += self.gmms[i][j][bd].score(
                        np.squeeze(observations[:, [bd], :], axis=1))
                c += 1
        if self.obslik_floor is not None:
            # limit the range of the observation likelihood to avoid
            # problems if the probability gets 0 (impossible state)
            log_densities = np.maximum(log_densities, self.obslik_floor)

        # return the densities
        return log_densities
