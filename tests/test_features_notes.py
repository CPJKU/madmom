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
from . import ACTIVATIONS_PATH, AUDIO_PATH

sample_file = pj(AUDIO_PATH, "stereo_sample.wav")
sample_act = Activations(pj(ACTIVATIONS_PATH, "stereo_sample.notes_brnn.npz"))


class TestRNNOnsetProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = RNNPianoNoteProcessor()

    def test_process(self):
        act = self.processor(sample_file)
        self.assertTrue(np.allclose(act, sample_act, atol=1e-6))
