# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.features.key module.

"""

from __future__ import absolute_import, division, print_function

import unittest
from os.path import join as pj

from madmom.features import Activations
from madmom.features.key import *
from . import AUDIO_PATH, ACTIVATIONS_PATH

sample_file = pj(AUDIO_PATH, 'sample.wav')
sample2_file = pj(AUDIO_PATH, 'sample2.wav')
sample_key_act = Activations(pj(ACTIVATIONS_PATH, 'sample.key_cnn.npz'))
sample2_key_act = Activations(pj(ACTIVATIONS_PATH, 'sample2.key_cnn.npz'))


class TestHelperFunctions(unittest.TestCase):

    def test_key_prediction_to_label_function(self):
        self.assertEqual(key_prediction_to_label(sample_key_act), 'Ab major')
        self.assertEqual(
            key_prediction_to_label(sample_key_act[0]), 'Ab major')
        self.assertEqual(
            key_prediction_to_label(np.roll(sample_key_act[0], 1)), 'A minor')
        self.assertEqual(
            key_prediction_to_label(np.roll(sample_key_act[0], -3)), 'F major')

        self.assertEqual(key_prediction_to_label(sample2_key_act), 'A minor')
        self.assertEqual(
            key_prediction_to_label(sample2_key_act[0]), 'A minor')
        self.assertEqual(
            key_prediction_to_label(np.roll(sample2_key_act[0], 1)),
            'Bb minor')
        self.assertEqual(
            key_prediction_to_label(np.roll(sample2_key_act[0], -3)),
            'F# major')


class TestCNNKeyRecognitionProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = CNNKeyRecognitionProcessor()

    def test_process(self):
        act = self.processor(sample_file)
        self.assertTrue(np.allclose(act, sample_key_act))

        act = self.processor(sample2_file)
        self.assertTrue(np.allclose(act, sample2_key_act))
