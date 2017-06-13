# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.features.beats module.

"""

from __future__ import absolute_import, division, print_function

import unittest
from os.path import join as pj

from . import AUDIO_PATH, ACTIVATIONS_PATH
from madmom.audio.signal import FramedSignal
from madmom.features import Activations
from madmom.features.beats import *
from madmom.features.beats_hmm import *
from madmom.ml.hmm import HiddenMarkovModel
from madmom.models import PATTERNS_BALLROOM


sample_file = pj(AUDIO_PATH, "sample.wav")
sample_lstm_act = Activations(pj(ACTIVATIONS_PATH, "sample.beats_lstm.npz"))
sample_blstm_act = Activations(pj(ACTIVATIONS_PATH, "sample.beats_blstm.npz"))
sample_downbeat_act = Activations(pj(ACTIVATIONS_PATH,
                                     "sample.downbeats_blstm.npz"))
sample_pattern_features = Activations(pj(ACTIVATIONS_PATH,
                                         "sample.gmm_pattern_tracker.npz"))


class TestRNNBeatProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = RNNBeatProcessor()

    def test_process_blstm(self):
        # load bi-directional RNN models
        beat_act = self.processor(sample_file)
        self.assertTrue(np.allclose(beat_act, sample_blstm_act, atol=1e-5))

    def test_process_lstm(self):
        # load uni-directional RNN models
        self.processor = RNNBeatProcessor(online=True, origin='online')
        # process the whole sequence at once
        result = self.processor(sample_file)
        self.assertTrue(np.allclose(result, sample_lstm_act, atol=1e-5))
        # result must be the same if processed a second time
        result_1 = self.processor(sample_file)
        self.assertTrue(np.allclose(result, result_1))
        # result must be the same if processed frame-by-frame
        frames = FramedSignal(sample_file, origin='online')
        self.processor = RNNBeatProcessor(online=True, num_frames=1,
                                          origin='future')
        result_2 = np.hstack([self.processor(f, reset=False) for f in frames])
        self.assertTrue(np.allclose(result, result_2))
        # result must be different without resetting
        result_3 = np.hstack([self.processor(f, reset=False) for f in frames])
        self.assertFalse(np.allclose(result, result_3))


class TestRNNDownBeatProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = RNNDownBeatProcessor()

    def test_process(self):
        downbeat_act = self.processor(sample_file)
        self.assertTrue(np.allclose(downbeat_act, sample_downbeat_act,
                                    atol=1e-5))


class TestBeatTrackingProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = BeatTrackingProcessor(fps=sample_blstm_act.fps)

    def test_process(self):
        beats = self.processor(sample_blstm_act)
        self.assertTrue(np.allclose(beats, [0.11, 0.45, 0.79, 1.13, 1.47,
                                            1.81, 2.15, 2.49]))


class TestBeatDetectionProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = BeatDetectionProcessor(fps=sample_blstm_act.fps)

    def test_process(self):
        beats = self.processor(sample_blstm_act)
        self.assertTrue(np.allclose(beats, [0.11, 0.45, 0.79, 1.13, 1.47,
                                            1.81, 2.15, 2.49]))


class TestCRFBeatDetectionProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = CRFBeatDetectionProcessor(fps=sample_blstm_act.fps)

    def test_process(self):
        beats = self.processor(sample_blstm_act)
        self.assertTrue(np.allclose(beats, [0.09, 0.79, 1.49]))


class TestDBNBeatTrackingProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = DBNBeatTrackingProcessor(fps=sample_blstm_act.fps)

    def test_types(self):
        self.assertIsInstance(self.processor.correct, bool)
        self.assertIsInstance(self.processor.st, BeatStateSpace)
        self.assertIsInstance(self.processor.tm, BeatTransitionModel)
        self.assertIsInstance(self.processor.om,
                              RNNBeatTrackingObservationModel)
        self.assertIsInstance(self.processor.hmm, HiddenMarkovModel)

    def test_values(self):
        self.assertTrue(self.processor.correct)
        path, prob = self.processor.hmm.viterbi(sample_blstm_act)
        self.assertTrue(np.allclose(path[:15], [207, 208, 209, 210, 211, 212,
                                                213, 214, 215, 216, 183, 184,
                                                185, 186, 187]))
        self.assertTrue(np.allclose(prob, -758.193327161))
        positions = self.processor.st.state_positions[path]
        self.assertTrue(np.allclose(positions[:9],
                                    [0.70588235, 0.73529412, 0.76470588,
                                     0.79411765, 0.82352941, 0.85294118,
                                     0.88235294, 0.91176471, 0.94117647]))
        intervals = self.processor.st.state_intervals[path]
        self.assertTrue(np.allclose(intervals[:10], 34))

    def test_process(self):
        beats = self.processor(sample_blstm_act)
        self.assertTrue(np.allclose(beats, [0.1, 0.45, 0.8, 1.12, 1.48, 1.8,
                                            2.15, 2.49]))
        # set the threshold
        self.processor.threshold = 1
        beats = self.processor(sample_blstm_act)
        self.assertTrue(np.allclose(beats, []))

    def test_process_forward(self):
        processor = DBNBeatTrackingProcessor(fps=100, online=True)
        # compute the forward path at once
        beats = processor.process_forward(sample_lstm_act)
        self.assertTrue(np.allclose(beats, [0.47, 0.79, 1.48, 2.16, 2.5]))
        # compute the forward path framewise
        processor.reset()
        beats = [processor.process_forward(np.atleast_1d(act), reset=False)
                 for act in sample_lstm_act]
        self.assertTrue(np.allclose(np.nonzero(beats),
                                    [47, 79, 148, 216, 250]))
        # without resetting results are different
        beats = [processor.process_forward(np.atleast_1d(act), reset=False)
                 for act in sample_lstm_act]
        self.assertTrue(np.allclose(np.nonzero(beats), [3, 79, 149, 216, 252]))


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
