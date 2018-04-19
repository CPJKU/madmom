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
sample_act = Activations(pj(ACTIVATIONS_PATH, "stereo_sample.notes_brnn.npz"))
sample_det = load_notes(pj(DETECTIONS_PATH,
                           "stereo_sample.piano_transcriptor.txt"))


class TestRNNOnsetProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = RNNPianoNoteProcessor()

    def test_process(self):
        act = self.processor(sample_file)
        self.assertTrue(np.allclose(act, sample_act, atol=1e-6))


class TestNoteOnsetPeakPickingProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = NoteOnsetPeakPickingProcessor(
            threshold=0.35, smooth=0.09, combine=0.05, pre_max=0.01,
            post_max=0.01, pitch_offset=21, fps=100)

    def test_process(self):
        notes = self.processor(sample_act)
        self.assertTrue(np.allclose(notes, sample_det))
        self.processor.threshold = 2
        notes = self.processor(sample_act)
        self.assertTrue(np.allclose(notes, np.zeros((0, 2))))

    def test_delay(self):
        self.processor.delay = 1
        notes = self.processor(sample_act)
        self.assertTrue(np.allclose(notes[:, 0] - 1, sample_det[:, 0]))
        self.assertTrue(np.allclose(notes[:, 1], sample_det[:, 1]))
