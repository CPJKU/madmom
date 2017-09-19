# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.features.downbeats module.

"""

from __future__ import absolute_import, division, print_function

import unittest
from os.path import join as pj

from madmom.ml.hmm import HiddenMarkovModel

from madmom.audio.chroma import CLPChroma
from madmom.features import Activations
from madmom.features.downbeats import *
from madmom.models import PATTERNS_BALLROOM
from . import ACTIVATIONS_PATH, ANNOTATIONS_PATH, AUDIO_PATH, DETECTIONS_PATH
from .test_utils import DETECTION_FILES

sample_file = pj(AUDIO_PATH, "sample.wav")
sample_beats = np.loadtxt(pj(ANNOTATIONS_PATH, "sample.beats"))
sample_det_file = pj(DETECTIONS_PATH, 'sample.dbn_beat_tracker.txt')
sample_beat_det = np.loadtxt(sample_det_file)
sample_bar_act = Activations(pj(ACTIVATIONS_PATH, "sample.bar_tracker.npz"))
sample_downbeat_act = Activations(pj(ACTIVATIONS_PATH,
                                     "sample.downbeats_blstm.npz"))
sample_pattern_features = Activations(pj(ACTIVATIONS_PATH,
                                         "sample.gmm_pattern_tracker.npz"))


class TestRNNDownBeatProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = RNNDownBeatProcessor()

    def test_process(self):
        downbeat_act = self.processor(sample_file)
        self.assertTrue(np.allclose(downbeat_act, sample_downbeat_act,
                                    atol=1e-5))


class TestDBNDownBeatTrackingProcessorClass(unittest.TestCase):
    def setUp(self):
        self.processor = DBNDownBeatTrackingProcessor(
            [3, 4], fps=sample_downbeat_act.fps)

    def test_types(self):
        self.assertIsInstance(self.processor.correct, bool)
        # self.assertIsInstance(self.processor.st, BarStateSpace)
        # the bar lengths are modelled with individual HMMs
        self.assertIsInstance(self.processor.hmms, list)
        self.assertIsInstance(self.processor.hmms[0], HiddenMarkovModel)
        self.assertIsInstance(self.processor.hmms[0].transition_model,
                              BarTransitionModel)
        self.assertIsInstance(self.processor.hmms[0].observation_model,
                              RNNDownBeatTrackingObservationModel)

    def test_values(self):
        self.assertTrue(self.processor.correct)
        # we have to test each bar length individually
        path, prob = self.processor.hmms[0].viterbi(sample_downbeat_act)
        self.assertTrue(np.allclose(path[:13],
                                    [7682, 7683, 7684, 7685, 7686, 7687, 7688,
                                     7689, 217, 218, 219, 220, 221]))
        self.assertTrue(np.allclose(prob, -764.586595603))
        tm = self.processor.hmms[0].transition_model
        positions = tm.state_space.state_positions[path]
        self.assertTrue(np.allclose(positions[:10],
                                    [2.77142857, 2.8, 2.82857143, 2.85714286,
                                     2.88571429, 2.91428571, 2.94285714,
                                     2.97142857, 0, 0.02857143]))
        intervals = tm.state_space.state_intervals[path]
        self.assertTrue(np.allclose(intervals[:10], 35))

    def test_process(self):
        downbeats = self.processor(sample_downbeat_act)
        self.assertTrue(np.allclose(downbeats, [[0.09, 1], [0.45, 2],
                                                [0.79, 3], [1.12, 4],
                                                [1.47, 1], [1.8, 2],
                                                [2.14, 3], [2.49, 4]]))
        # test with beats at the first and last frame
        act = np.zeros((200, 2)) + 1e-4
        act[[0, 199], 1] = 1  # downbeats
        act[[49, 99, 149], 0] = 1  # beats
        downbeats = self.processor(act)
        self.assertTrue(np.allclose(downbeats, [[0, 1], [0.49, 2], [0.99, 3],
                                                [1.49, 4], [1.99, 1]]))
        # without correcting the beat positions
        self.processor.correct = False
        downbeats = self.processor(sample_downbeat_act)
        correct = np.array([[0.08, 1], [0.43, 2], [0.77, 3], [1.11, 4],
                            [1.45, 1], [1.79, 2], [2.13, 3], [2.47, 4]])
        self.assertTrue(np.allclose(downbeats, correct))
        # test threshold
        self.processor.threshold = 0.5
        downbeats = self.processor(sample_downbeat_act)
        self.assertTrue(np.allclose(downbeats, correct[1:-1]))
        self.processor.threshold = 1
        downbeats = self.processor(sample_downbeat_act)
        self.assertTrue(np.allclose(downbeats, np.empty((0, 2))))


class TestPatternTrackingProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = PatternTrackingProcessor(
            PATTERNS_BALLROOM, fps=sample_pattern_features.fps)

    def test_types(self):
        self.assertIsInstance(self.processor.num_beats, list)
        self.assertIsInstance(self.processor.st, MultiPatternStateSpace)
        self.assertIsInstance(self.processor.tm, MultiPatternTransitionModel)
        self.assertIsInstance(self.processor.om,
                              GMMPatternTrackingObservationModel)
        self.assertIsInstance(self.processor.hmm, HiddenMarkovModel)

    def test_values(self):
        self.assertTrue(self.processor.fps == 50)
        self.assertTrue(np.allclose(self.processor.num_beats, [3, 4]))
        path, prob = self.processor.hmm.viterbi(sample_pattern_features)
        self.assertTrue(np.allclose(path[:12], [5573, 5574, 5575, 5576, 6757,
                                                6758, 6759, 6760, 6761, 6762,
                                                6763, 6764]))
        self.assertTrue(np.allclose(prob, -468.8014))
        patterns = self.processor.st.state_patterns[path]
        self.assertTrue(np.allclose(patterns,
                                    np.ones(len(sample_pattern_features))))
        positions = self.processor.st.state_positions[path]
        self.assertTrue(np.allclose(positions[:6], [1.76470588, 1.82352944,
                                                    1.88235296, 1.94117648,
                                                    2, 2.0588236]))

    def test_process(self):
        beats = self.processor(sample_pattern_features)
        self.assertTrue(np.allclose(beats, [[0.08, 3], [0.42, 4], [0.76, 1],
                                            [1.1, 2], [1.44, 3], [1.78, 4],
                                            [2.12, 1], [2.46, 2], [2.8, 3]]))


class TestLoadBeatsProcessorClass(unittest.TestCase):

    def test_single(self):
        proc = LoadBeatsProcessor(sample_det_file)
        self.assertTrue(proc.mode == 'single')
        result = proc()
        self.assertTrue(np.allclose(result, sample_beat_det))

    def test_batch(self):
        proc = LoadBeatsProcessor(None, files=DETECTION_FILES,
                                  beats_suffix='.dbn_beat_tracker.txt')
        self.assertTrue(proc.mode == 'batch')
        result = proc(sample_file)
        self.assertTrue(np.allclose(result, sample_beat_det))


class TestSyncronizeFeaturesProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = SyncronizeFeaturesProcessor(beat_subdivisions=2,
                                                     fps=100)

    def test_process(self):
        data = [CLPChroma(sample_file, fps=100), sample_beats]
        feat_sync = self.processor(data)
        target = [[0.28231065, 0.14807641, 0.22790557, 0.41458403, 0.15966462,
                   0.22294236, 0.1429988, 0.16661506, 0.5978227, 0.24039252,
                   0.23444982, 0.21910049],
                  [0.25676728, 0.13382165, 0.19957431, 0.47225753, 0.18936998,
                   0.17014103, 0.14079712, 0.18317944, 0.60692955, 0.20016842,
                   0.17619181, 0.24408179]]
        self.assertTrue(np.allclose(feat_sync[0, :], target, rtol=1e-3))

    def test_corner_cases(self):
        feat_sync = self.processor([np.arange(100), np.array([])])
        self.assertTrue(np.allclose(feat_sync, [[], []]))
        feat_sync = self.processor([np.arange(100), np.array([0, 0.5, 1.5])])
        self.assertTrue(np.allclose(feat_sync, [[[11.], [33.5]]]))


class TestRNNBarProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = RNNBarProcessor(fps=100)

    def test_process(self):
        act = self.processor((sample_file, sample_beats[:, 0]))
        self.assertTrue(np.allclose(act, sample_bar_act, rtol=1e-3,
                                    equal_nan=True))


class TestDBNBarTrackingProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = DBNBarTrackingProcessor()

    def test_dbn(self):
        # check DBN output
        path, log = self.processor.hmm.viterbi(sample_bar_act[:-1, 1])
        self.assertTrue(np.allclose(path, [0, 1, 2]))
        self.assertTrue(np.allclose(log, -12.2217575073))

    def test_process(self):
        beats = self.processor(sample_bar_act)
        self.assertTrue(np.allclose(beats, [[0.0913, 1.], [0.7997, 2.],
                                            [1.4806, 3.], [2.1478, 1.]]))
