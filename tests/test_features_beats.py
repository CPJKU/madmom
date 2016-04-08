# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.features.beats module.

"""

from __future__ import absolute_import, division, print_function

import unittest
from . import AUDIO_PATH, ACTIVATIONS_PATH
from madmom.features import Activations
from madmom.features.beats import *
from madmom.features.beats_hmm import *
from madmom.ml.hmm import HiddenMarkovModel
from madmom.models import PATTERNS_BALLROOM


sample_file = "%s/sample.wav" % AUDIO_PATH
sample_beat_act = Activations("%s/sample.beats_blstm_2013.npz" %
                              ACTIVATIONS_PATH)
sample_downbeat_act = Activations("%s/sample.downbeats_blstm_2016.npz" %
                                  ACTIVATIONS_PATH)
sample_pattern_features = Activations("%s/sample.gmm_pattern_tracker.npz" %
                                      ACTIVATIONS_PATH)


class TestRNNBeatProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = RNNBeatProcessor()

    def test_process(self):
        beat_act = self.processor(sample_file)
        self.assertTrue(np.allclose(beat_act, sample_beat_act))


class TestRNNDownBeatProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = RNNDownBeatProcessor()

    def test_process(self):
        downbeat_act = self.processor(sample_file)
        self.assertTrue(np.allclose(downbeat_act, sample_downbeat_act))


class TestBeatTrackingProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = BeatTrackingProcessor(fps=sample_beat_act.fps)

    def test_process(self):
        beats = self.processor(sample_beat_act)
        self.assertTrue(np.allclose(beats, [0.11, 0.45, 0.79, 1.13, 1.47,
                                            1.81, 2.15, 2.49]))


class TestBeatDetectionProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = BeatDetectionProcessor(fps=sample_beat_act.fps)

    def test_process(self):
        beats = self.processor(sample_beat_act)
        self.assertTrue(np.allclose(beats, [0.11, 0.45, 0.79, 1.13, 1.47,
                                            1.81, 2.15, 2.49]))


class TestCRFBeatDetectionProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = CRFBeatDetectionProcessor(fps=sample_beat_act.fps)

    def test_process(self):
        beats = self.processor(sample_beat_act)
        self.assertTrue(np.allclose(beats, [0.09, 0.79, 1.49]))


class TestDBNBeatTrackingProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = DBNBeatTrackingProcessor(fps=sample_beat_act.fps)

    def test_types(self):
        self.assertIsInstance(self.processor.correct, bool)
        self.assertIsInstance(self.processor.st, BeatStateSpace)
        self.assertIsInstance(self.processor.tm, BeatTransitionModel)
        self.assertIsInstance(self.processor.om,
                              RNNBeatTrackingObservationModel)
        self.assertIsInstance(self.processor.hmm, HiddenMarkovModel)

    def test_values(self):
        self.assertTrue(self.processor.correct)
        path, prob = self.processor.hmm.viterbi(sample_beat_act)
        self.assertTrue(np.allclose(path[:15], [2030, 2031, 2032, 2033, 2034,
                                                2035, 2036, 1968, 1969, 1970,
                                                1971, 1972, 1973, 1974, 1975]))
        self.assertTrue(np.allclose(prob, -772.03353))
        positions = self.processor.st.state_positions[path]
        self.assertTrue(np.allclose(positions[:10],
                                    [0.89855075, 0.9130435, 0.92753625,
                                     0.942029, 0.95652175, 0.9710145,
                                     0.98550725, 0, 0.01449275, 0.02898551]))
        intervals = self.processor.st.state_intervals[path]
        self.assertTrue(np.allclose(intervals[:10], 69))

    def test_process(self):
        beats = self.processor(sample_beat_act)
        self.assertTrue(np.allclose(beats, [0.09, 0.8, 1.48, 2.15]))
        # set the threshold
        self.processor.threshold = 1
        beats = self.processor(sample_beat_act)
        self.assertTrue(np.allclose(beats, []))


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
                                    [11478, 11479, 11480, 11481, 11482, 11483,
                                     11484, 11485, 217, 218, 219, 220, 221]))
        self.assertTrue(np.allclose(prob, -766.314167769))
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
        self.assertTrue(np.allclose(downbeats, [[0.09, 1], [0.44, 2],
                                                [0.79, 3], [1.12, 4],
                                                [1.47, 1], [1.8, 2],
                                                [2.14, 3], [2.49, 4]]))
        # set the threshold
        self.processor.threshold = 1
        downbeats = self.processor(sample_downbeat_act)
        self.assertTrue(np.allclose(downbeats, np.empty((0, 2))))

    def test_process_downbeats(self):
        self.processor.downbeats = True
        downbeats = self.processor(sample_downbeat_act)
        self.assertTrue(np.allclose(downbeats, [0.09, 1.47]))
        # set the threshold
        self.processor.threshold = 1
        downbeats = self.processor(sample_downbeat_act)
        self.assertTrue(np.allclose(downbeats, np.empty((0, 2))))


class TestPatternTrackingProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = PatternTrackingProcessor(
            PATTERNS_BALLROOM, fps=sample_pattern_features.fps)

    def test_types(self):
        self.assertIsInstance(self.processor.downbeats, bool)
        self.assertIsInstance(self.processor.num_beats, list)
        self.assertIsInstance(self.processor.st, MultiPatternStateSpace)
        self.assertIsInstance(self.processor.tm, MultiPatternTransitionModel)
        self.assertIsInstance(self.processor.om,
                              GMMPatternTrackingObservationModel)
        self.assertIsInstance(self.processor.hmm, HiddenMarkovModel)

    def test_values(self):
        self.assertTrue(self.processor.fps == 50)
        self.assertTrue(self.processor.downbeats is False)
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

    def test_process_downbeats(self):
        self.processor.downbeats = True
        beats = self.processor(sample_pattern_features)
        self.assertTrue(np.allclose(beats, [0.76, 2.12]))
