# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.features.notes module.

"""

from __future__ import absolute_import, division, print_function

import unittest
from os.path import join as pj

from madmom.features import Activations
from madmom.features.notes import *
from madmom.io import load_notes
from . import ACTIVATIONS_PATH, AUDIO_PATH, DETECTIONS_PATH

sample_file = pj(AUDIO_PATH, "stereo_sample.wav")
sample_act_rnn = Activations(pj(ACTIVATIONS_PATH,
                                "stereo_sample.notes_brnn.npz"))
sample_act_cnn = Activations(pj(ACTIVATIONS_PATH,
                                "stereo_sample.notes_cnn.npz"))
sample_det = load_notes(pj(DETECTIONS_PATH,
                           "stereo_sample.piano_transcriptor.txt"))


class TestRNNOnsetProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = RNNPianoNoteProcessor()

    def test_process(self):
        act = self.processor(sample_file)
        self.assertTrue(np.allclose(act, sample_act_rnn, atol=1e-6))


class TestNoteOnsetPeakPickingProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = NoteOnsetPeakPickingProcessor(
            threshold=0.35, smooth=0.09, combine=0.05, pre_max=0.01,
            post_max=0.01, pitch_offset=21, fps=100)
        self.result = np.array([[0.14, 72], [1.56, 41],
                                [2.52, 77], [3.37, 75]])

    def test_process(self):
        notes = self.processor(sample_act_rnn)
        self.assertTrue(np.allclose(notes, self.result))
        self.processor.threshold = 2
        notes = self.processor(sample_act_rnn)
        self.assertTrue(np.allclose(notes, np.zeros((0, 2))))

    def test_delay(self):
        self.processor.delay = 1
        notes = self.processor(sample_act_rnn)
        self.assertTrue(np.allclose(notes[:, 0] - 1, self.result[:, 0]))
        self.assertTrue(np.allclose(notes[:, 1], self.result[:, 1]))


class TestCNNPianoNoteProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = CNNPianoNoteProcessor()

    def test_process(self):
        act = self.processor(sample_file)
        self.assertTrue(np.allclose(act, sample_act_cnn, atol=1e-6))


class TestADSRNoteTrackingProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = ADSRNoteTrackingProcessor()

    def test_process(self):
        notes = self.processor(sample_act_cnn)
        self.assertTrue(np.allclose(notes, sample_det))
        self.assertTrue(np.allclose(notes.shape, (7, 3)))
        # do not enforce complete notes (same result, though)
        self.processor.complete = False
        notes = self.processor(sample_act_cnn)
        self.assertTrue(np.allclose(notes.shape, (7, 3)))
        # try various thresholds
        self.processor.onset_threshold = 0.75
        notes = self.processor(sample_act_cnn)
        self.assertTrue(np.allclose(notes.shape, (6, 3)))
        self.processor.note_threshold = 0.99
        notes = self.processor(sample_act_cnn)
        self.assertTrue(np.allclose(notes.shape, (5, 3)))
        self.processor.complete = False
        notes = self.processor(sample_act_cnn)
        self.assertTrue(np.allclose(notes.shape, (5, 3)))
        self.processor.onset_threshold = 1
        notes = self.processor(sample_act_cnn)
        self.assertTrue(np.allclose(notes.shape, (0, 3)))
