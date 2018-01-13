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
from madmom.io import load_notes, write_notes
from madmom.utils import expand_notes
from . import ACTIVATIONS_PATH, ANNOTATIONS_PATH, AUDIO_PATH

sample_file = pj(AUDIO_PATH, "stereo_sample.wav")
sample_act = Activations(pj(ACTIVATIONS_PATH, "stereo_sample.notes_brnn.npz"))

NOTES = np.array([[0.147, 72, 3.323, 63], [1.567, 41, 0.223, 29],
                  [2.526, 77, 0.93, 72], [2.549, 60, 0.211, 28],
                  [2.563, 65, 0.202, 34], [2.577, 56, 0.234, 31],
                  [3.369, 75, 0.78, 64], [3.449, 43, 0.272, 35]])


class TestLoadNotesFunction(unittest.TestCase):

    def test_values(self):
        result = load_notes(pj(ANNOTATIONS_PATH, 'stereo_sample.notes'))
        self.assertTrue(np.allclose(result, NOTES))


class TestExpandNotesFunction(unittest.TestCase):

    def test_values(self):
        # only onset and MIDI note given
        result = expand_notes(NOTES[:, :2])
        self.assertTrue(np.allclose(result[:, :2], NOTES[:, :2]))
        self.assertTrue(np.allclose(result[:, 2], 0.6))
        self.assertTrue(np.allclose(result[:, 3], 100))
        # also duration given
        result = expand_notes(NOTES[:, :3], velocity=66)
        self.assertTrue(np.allclose(result[:, :3], NOTES[:, :3]))
        self.assertTrue(np.allclose(result[:, 3], 66))
        # also velocity given
        result = expand_notes(NOTES)
        self.assertTrue(np.allclose(result, NOTES))


class TestWriteNotesFunction(unittest.TestCase):

    def test_values(self):
        header = "MIDI notes for the stereo_sample.[flac|wav] file"
        write_notes(NOTES,
                    pj(ANNOTATIONS_PATH, 'stereo_sample.notes'), header=header)


class TestRNNOnsetProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = RNNPianoNoteProcessor()

    def test_process(self):
        act = self.processor(sample_file)
        self.assertTrue(np.allclose(act, sample_act, atol=1e-6))
