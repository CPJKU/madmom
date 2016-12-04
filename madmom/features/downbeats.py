# encoding: utf-8
"""
This file contains downbeat tracking related functionality.

"""

from __future__ import absolute_import, division, print_function

import numpy as np
from madmom.utils import match_file
from madmom.processors import Processor, ParallelProcessor, SequentialProcessor

from madmom.audio.chroma import CLPChromaProcessor
from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
from madmom.features import ActivationsProcessor

import os.path


class BeatSyncProcessor(Processor):
    """
    Synchronize features to the beat.

    """
    def __init__(self, fps, beat_subdivision, sum_func=np.mean):
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
        self.fps = fps
        self.beat_subdivision = beat_subdivision
        self.sum_func = sum_func

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
            data[0]: features e.g., MultiBandSpectrogram
            data[1]: numpy array of beat times

        Returns
        -------
        feats : numpy array [num_beats - 1, beat_subdivision * feat_dim]
            Beat synchronous features.

        """
        features, beats = data
        if beats.ndim > 1:
            beats = beats[:, 0]
        if beats.size == 0:
            return np.array([]), np.array([])
        while (float(len(features)) / self.fps) < beats[-1]:
            beats = beats[:-1]
            import warnings
            warnings.warn('WARNING: Beat sequence too long compared to feature'
                          ' sequence. Removing last beat annotation.')
        feat_dim = features.shape[1]
        n_beats = len(beats)
        beat_features = np.empty((n_beats - 1, self.beat_subdivision,
                                  feat_dim))
        # Window of the first beat starts 20 ms before the beat
        next_beat_frame = np.max([0, np.floor((beats[0] - 0.02) *
                                              self.fps)])
        for i_beat in range(n_beats - 1):
            # We summarise all feature values that fall into a window of
            # length = ibi / beat_div, centered on the beat
            # annotations or interpolated subdivisions. We subtract half of
            # this timespan to get the start frame of the first window of this
            # and the next bar
            ibi_sec = beats[i_beat + 1] - beats[i_beat]
            offset = 0.5 * ibi_sec / self.beat_subdivision
            # The offset should be < 50 ms
            offset = np.min([offset, 0.05])
            # first frame of current beat
            beat_start_frame = next_beat_frame
            # first frame of next beat
            next_beat_frame = int(np.floor(
                (beats[i_beat + 1] - offset) * self.fps))
            num_beat_frames = next_beat_frame - beat_start_frame
            # assign each frame to a beat subdivision
            beat_pos_of_frames = np.floor(
                np.linspace(0, self.beat_subdivision, num_beat_frames,
                            endpoint=False))
            features_beat = features[beat_start_frame:next_beat_frame]
            # group features in list by bar position
            feats_sorted = [features_beat[beat_pos_of_frames == x] for x in
                            np.arange(0, self.beat_subdivision)]
            # interpolate missing feature values which occur at fast tempi
            feats_sorted = self.interpolate_missing(feats_sorted, feat_dim)
            beat_features[i_beat, :, :] = np.array([self.sum_func(x, axis=0)
                                                    for x in feats_sorted])
        return beat_features.reshape((n_beats - 1, self.beat_subdivision *
                                      feat_dim))

    def interpolate_missing(self, feats, feat_dim):
        """
        Interpolate missing feature values which occur at fast tempi.

        Parameters
        ----------
        feats : list of numpy arrays [beat_divisions](num_beats, feat_dim)
            Feature values.
        feat_dim :
            Feature dimension.

        Returns
        -------
        feats : numpy array
            Interpolated features.

        """
        nan_fill = np.empty(feat_dim) * np.nan
        means = np.array([np.mean(x, axis=0) if len(x) > 0 else nan_fill
                          for x in feats])
        good_rows = np.unique(np.where(np.logical_not(np.isnan(means)))[0])
        if len(good_rows) < self.beat_subdivision:
            bad_rows = np.unique(np.where(np.isnan(means))[0])
            # initialise missing values with empty array
            for r in bad_rows:
                feats[r] = np.empty((1, feat_dim))
            for d in range(feat_dim):
                means_p = np.interp(np.arange(0, self.beat_subdivision),
                                    good_rows, means[good_rows, d])
                for r in bad_rows:
                    feats[r][0, d] = means_p[r]
        return feats


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

    def __init__(self, observation_lambda=100, downbeats=False,
                 pattern_change_prob=0.0, beats_per_bar=[3, 4], **kwargs):
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
                                MultiPatternTransitionModel,
                                RNNBeatTrackingObservationModel)
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
        self.om = RNNBeatTrackingObservationModel(
            self.st, observation_lambda, obslik_floor=1e-6)
        # instantiate a HMM
        self.hmm = Hmm(self.tm, self.om, None)

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
        if data[0].ndim == 1:
            activations = data[0]
        else:
            activations = data[0][:, 0]
        beats = data[1]
        if beats.size == 0:
            return np.array([])
        # get the best state path by calling the viterbi algorithm
        path, log = self.hmm.viterbi(activations)
        # get the position inside the bar
        position = self.st.state_positions[path]
        # the beat numbers are the counters + 1 at the transition points
        beat_numbers = position.astype(int) + 1
        # add the last beat
        last_beat_number = np.mod(beat_numbers[-1], self.num_beats[
            self.st.state_patterns[path[-1]]]) + 1
        beat_numbers = np.append(beat_numbers, last_beat_number)
        # return the downbeats or beats and their beat numbers
        if self.downbeats:
            return np.squeeze(beats[np.where(beat_numbers == 1)])
        else:
            return np.vstack(zip(beats, beat_numbers))

    @classmethod
    def add_arguments(cls, parser, observation_lambda=100,
                      beats_per_bar=[3, 4]):
        """
        Add DBN related arguments to an existing parser.

        Parameters
        ----------
        parser :
            existing argparse parser

        observation_lambda :
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
                            ' separated list of bar length in beats) '
                            '[default=%(default)s]')
        parser.add_argument('--observation_lambda', action='store', type=int,
                            default=observation_lambda, help='split one beat'
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


# TODO: this should be loaded from the model file!
SUB_DIVISIONS = [4, 2]


class RNNBarProcessor(Processor):
    """
    Processor to get a downbeat activation function from multiple RNNs.

    """

    def __init__(self, fps=100):
        from ..audio.spectrogram import (
            FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
            SpectrogramDifferenceProcessor)
        from ..audio.signal import SignalProcessor, FramedSignalProcessor
        from ..ml.nn import NeuralNetworkEnsemble
        from ..models import DOWNBEATS_BGRU
        self.num_feats = len(DOWNBEATS_BGRU)
        # percussive feature
        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        frames = FramedSignalProcessor(frame_size=2048, fps=fps)
        spec = FilteredSpectrogramProcessor(
            num_bands=6, fmin=30., fmax=17000., norm_filters=True)
        log_spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
        diff = SpectrogramDifferenceProcessor(
            diff_ratio=0.5, positive_diffs=True)
        self.percussive = SequentialProcessor((sig, frames, spec, log_spec,
                                               diff))
        # harmonic feature
        self.harmonic = CLPChromaProcessor(
            fps=fps, fmin=27.5, fmax=4200., compression_factor=100,
            norm=True, threshold=0.001)
        self.rnn = []
        beat_sync = []
        for f in range(self.num_feats):
            beat_sync.append(BeatSyncProcessor(fps, SUB_DIVISIONS[f]))
            nn = NeuralNetworkEnsemble.load(DOWNBEATS_BGRU[f])
            self.rnn.append(SequentialProcessor([beat_sync[f], nn]))

    def process(self, data, kwargs):
        """
        Compute features, get beats, sync features to beats, compute RNN
        activations and average them.

        Parameters
        ----------
        data :
            audio filename

        Returns
        -------
        numpy array
            downbeat activation function

        """
        # check if beats shpould be loaded from file or detected
        if kwargs['load_beats']:
            beats = LoadBeatsProcessor(kwargs['beat_files'],
                                       kwargs['beat_suffix'])
        else:
            beats = DetectBeatsProcessor()
        pre_processor = ParallelProcessor([self.percussive, self.harmonic,
                                           beats])
        # extract features, compute beats, and sync features to the beats
        data = pre_processor(data)
        beats = data[-1]
        # run all RNNs and average the activations
        activations = []
        for f in range(self.num_feats):
            activations.append(self.rnn[f]([data[f], beats]))
        return np.mean(np.array(activations), 0), beats


class DetectBeatsProcessor(SequentialProcessor):
    """
    Processor to get the beat times from an audio signal.

    """
    def __init__(self):
        rnn = RNNBeatProcessor()
        dbn = DBNBeatTrackingProcessor(fps=100)
        super(DetectBeatsProcessor, self).__init__([rnn, dbn])


class BarTrackerActivationsProcessor(ActivationsProcessor):
    """
        BarTrackerActivationsProcessor processes a file and returns an
        BarTrackerActivations instance. This class extends the class
        ActivationsProcessor by saving/loading numpy arrays directly without
        instantiating Activation objects.

    """
    def process(self, data, output=None):
        """
        Depending on the mode, either loads the data stored in the given file
        and returns it as an Activations instance or save the data to the given
        output.

        Parameters
        ----------
        data : str, file handle or numpy array
            Data or file to be loaded (if `mode` is 'r') or data to be saved
            to file (if `mode` is 'w').
        output : str or file handle, optional
            output file (only in write-mode)

        Returns
        -------
        :class:`Activations` instance
            :class:`Activations` instance (only in read-mode)

        """
        # pylint: disable=arguments-differ

        if self.mode in ('r', 'in', 'load'):
            ldata = np.load(data)
            if isinstance(ldata, np.lib.npyio.NpzFile):
                # .npz file, set the frame rate if none is given
                data = list([])
                data.append(ldata['activations'])
                data.append(ldata['beats'])
        elif self.mode in ('w', 'out', 'save'):
            # numpy binary format
            npz = {'activations': data[0],
                   'beats': data[1]}
            np.savez(output, **npz)
        else:
            raise ValueError("wrong mode %s; choose {'r', 'w', 'in', 'out', "
                             "'load', 'save'}")
        return data
