# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.audio.chroma module.

"""

from __future__ import absolute_import, division, print_function

import numpy as np
import unittest
from . import AUDIO_PATH, ACTIVATIONS_PATH
from madmom.audio.chroma import DeepChromaProcessor, CLPChroma
from madmom.features import Activations


sample_file = "%s/sample.wav" % AUDIO_PATH
sample_act_deep_chroma = Activations("%s/sample.deep_chroma.npz" %
                                     ACTIVATIONS_PATH)
sample_act_clp_chroma = Activations("%s/sample.clp_chroma.npz" %
                                    ACTIVATIONS_PATH)


class TestDeepChromaProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = DeepChromaProcessor()

    def test_process(self):
        chroma_act = self.processor(sample_file)
        self.assertTrue(np.allclose(chroma_act, sample_act_deep_chroma))


class TestCLPChromaClass(unittest.TestCase):

    def test_process(self):
        chroma_act = CLPChroma(sample_file)
        self.assertTrue(np.allclose(chroma_act, sample_act_clp_chroma))
