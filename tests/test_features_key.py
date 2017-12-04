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
from . import AUDIO_PATH, ACTIVATIONS_PATH, DETECTIONS_PATH


class TestHelperFunctions(unittest.TestCase):

    def test_key_prediction_to_label_function(self):
        self.assertFalse(True, 'Implement this test!')

    def test_write_key_function(self):
        self.assertFalse(True, 'Implement this test?')


class TestCNNKeyRecognitionProcessorClass(unittest.TestCase):

    def test_init(self):
        self.assertFalse(True, 'Implement this test!')

