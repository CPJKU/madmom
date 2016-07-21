# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.audio.chroma module.

"""

from __future__ import absolute_import, division, print_function

import numpy as np
import unittest
from os.path import join as pj
from . import AUDIO_PATH, ACTIVATIONS_PATH
from madmom.audio.chroma import DeepChromaProcessor
from madmom.features import Activations


sample_files = [pj(AUDIO_PATH, sf) for sf in ['sample.wav', 'sample2.wav']]
sample_acts = [Activations(pj(ACTIVATIONS_PATH, af))
               for af in ['sample.deep_chroma.npz', 'sample2.deep_chroma.npz']]


class TestDeepChromaProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = DeepChromaProcessor()

    def test_process(self):
        for sample_file, sample_act in zip(sample_files, sample_acts):
            chroma_act = self.processor(sample_file)
            self.assertTrue(np.allclose(chroma_act, sample_act))
