# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.features.drums module.

"""

from __future__ import absolute_import, division, print_function

import unittest
from os.path import join as pj

from . import AUDIO_PATH, ACTIVATIONS_PATH, DETECTIONS_PATH

from madmom.features import Activations
from madmom.features.drums import *

sample_file = pj(AUDIO_PATH, 'sample.wav')
sample_act = Activations(pj(ACTIVATIONS_PATH, 'sample.drums_crnn.npz'))


class TestDrumProcessorClass(unittest.TestCase):
    def setUp(self):
        self.processor = CRNNDrumProcessor()

    def test_process(self):
        act = self.processor(sample_file)
        self.assertTrue(np.allclose(act, sample_act))


class TestDrumPeakPickingProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = DrumPeakPickingProcessor(fps=sample_act.fps)

    def test_process(self):
        drums = self.processor(sample_act)
        print(drums)
        self.assertTrue(np.allclose(drums, [[0.09, 0], [0.09, 2], [0.44, 2],
                                            [0.61, 0], [0.76, 0], [0.80, 1],
                                            [1.12, 0], [1.12, 2], [1.62, 1],
                                            [1.80, 0], [1.80, 2], [2.14, 1],
                                            [2.14, 2], [2.66, 0]]))
