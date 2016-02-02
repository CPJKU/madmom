# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.ml.hmm module.

"""

from __future__ import absolute_import, division, print_function

import unittest
from . import ACTIVATIONS_PATH, MODELS_PATH
from madmom.ml.hmm import HiddenMarkovModel
from madmom.features.beats import *
from madmom.features.beats_hmm import *

act_file = np.load("%s/sample.beats_blstm_2013.npz" % ACTIVATIONS_PATH)
fps = act_file['fps']
act = act_file['activations']


class TestBeatTrackingProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = BeatTrackingProcessor(fps=fps)

    def test_process(self):
        beats = self.processor(act)
        self.assertTrue(np.allclose(beats, [0.11, 0.45, 0.79, 1.13, 1.47,
                                            1.81, 2.15, 2.49]))


class TestBeatDetectionProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = BeatDetectionProcessor(fps=fps)

    def test_process(self):
        beats = self.processor(act)
        self.assertTrue(np.allclose(beats, [0.11, 0.45, 0.79, 1.13, 1.47,
                                            1.81, 2.15, 2.49]))


class TestCRFBeatDetectionProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = CRFBeatDetectionProcessor(fps=fps)

    def test_process(self):
        beats = self.processor(act)
        self.assertTrue(np.allclose(beats, [0.09, 0.79, 1.49]))


class TestDBNBeatTrackingProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = DBNBeatTrackingProcessor(fps=fps)

    def test_types(self):
        self.assertIsInstance(self.processor.correct, bool)
        self.assertIsInstance(self.processor.st, BeatTrackingStateSpace)
        self.assertIsInstance(self.processor.tm, BeatTrackingTransitionModel)
        self.assertIsInstance(self.processor.om, BeatTrackingObservationModel)
        self.assertIsInstance(self.processor.hmm, HiddenMarkovModel)

    def test_values(self):
        self.assertTrue(self.processor.correct)
        path, prob = self.processor.hmm.viterbi(act)
        self.assertTrue(np.allclose(path[:15], [2030, 2031, 2032, 2033, 2034,
                                                2035, 2036, 1968, 1969, 1970,
                                                1971, 1972, 1973, 1974, 1975]))
        self.assertTrue(np.allclose(prob, -772.03353))
        position = self.processor.st.position(path)
        self.assertTrue(np.allclose(position[:11], [0.89855075, 0.9130435,
                                                    0.92753625, 0.942029,
                                                    0.95652175, 0.9710145,
                                                    0.98550725, 0, 0.01449275,
                                                    0.02898551, 0.04347826]))

    def test_process(self):
        beats = self.processor(act)
        self.assertTrue(np.allclose(beats, [0.09, 0.8, 1.48, 2.15]))


class TestPatternTrackingProcessorClass(unittest.TestCase):

    def setUp(self):
        import glob
        pattern_files = sorted(glob.glob("%s/patterns/2013/*" % MODELS_PATH))
        features = np.load("%s/sample.multiband_spectral_flux.npz"
                           % ACTIVATIONS_PATH)
        self.act = features['activations']
        self.processor = PatternTrackingProcessor(pattern_files,
                                                  fps=features['fps'])

    def test_types(self):
        self.assertIsInstance(self.processor.downbeats, bool)
        self.assertIsInstance(self.processor.num_beats, list)
        self.assertIsInstance(self.processor.st, PatternTrackingStateSpace)
        self.assertIsInstance(self.processor.tm,
                              PatternTrackingTransitionModel)
        self.assertIsInstance(self.processor.om,
                              GMMPatternTrackingObservationModel)
        self.assertIsInstance(self.processor.hmm, HiddenMarkovModel)

    def test_values(self):
        self.assertTrue(self.processor.downbeats is False)
        self.assertTrue(np.allclose(self.processor.num_beats, [3, 4]))
        path, prob = self.processor.hmm.viterbi(self.act)
        self.assertTrue(np.allclose(path[:12], [13497, 13498, 13499, 13500,
                                                13501, 13502, 13503, 13504,
                                                13505, 13506, 13507, 13508]))
        self.assertTrue(np.allclose(prob, -463.3286))
        pattern = self.processor.st.pattern(path)
        self.assertTrue(np.allclose(pattern, np.ones(len(self.act))))
        position = self.processor.st.position(path)
        self.assertTrue(np.allclose(position[:6], [0.19117647, 0.20588236,
                                                   0.22058824, 0.23529412,
                                                   0.25, 0.2647059]))

    def test_process(self):
        beats = self.processor(self.act)
        self.assertTrue(np.allclose(beats, [[0.08, 2], [0.42, 3], [0.76, 4],
                                            [1.1, 1], [1.46, 2], [1.8, 3],
                                            [2.14, 4], [2.48, 1]]))
