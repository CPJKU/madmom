# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.features.chords module.

"""

from __future__ import absolute_import, division, print_function

import unittest
from . import AUDIO_PATH, ACTIVATIONS_PATH

from madmom.features import Activations
from madmom.features.chords import *


sample_file = "%s/sample.wav" % AUDIO_PATH
sample_cnn_act = Activations('%s/sample.cnn_chord_features.npz' %
                             ACTIVATIONS_PATH)
sample_deep_chroma_act = Activations('%s/sample.deep_chroma.npz' %
                                     ACTIVATIONS_PATH)
sample_labels = [(0.0, 2.9, 'G#:maj')]


def _compare_labels(test_case, l1, l2):
    for l, tl in zip(l1, l2):
        test_case.assertAlmostEqual(l[0], tl[0])
        test_case.assertAlmostEqual(l[1], tl[1])
        test_case.assertEqual(l[2], tl[2])


class TestMajMinTargetsToChordLabelsFunction(unittest.TestCase):

    def test_all_labels(self):
        fps = 10.
        targets = range(25)
        target_labels = [(0.0, 0.1, 'A:maj'),
                         (0.1, 0.2, 'A#:maj'),
                         (0.2, 0.3, 'B:maj'),
                         (0.3, 0.4, 'C:maj'),
                         (0.4, 0.5, 'C#:maj'),
                         (0.5, 0.6, 'D:maj'),
                         (0.6, 0.7, 'D#:maj'),
                         (0.7, 0.8, 'E:maj'),
                         (0.8, 0.9, 'F:maj'),
                         (0.9, 1.0, 'F#:maj'),
                         (1.0, 1.1, 'G:maj'),
                         (1.1, 1.2, 'G#:maj'),
                         (1.2, 1.3, 'A:min'),
                         (1.3, 1.4, 'A#:min'),
                         (1.4, 1.5, 'B:min'),
                         (1.5, 1.6, 'C:min'),
                         (1.6, 1.7, 'C#:min'),
                         (1.7, 1.8, 'D:min'),
                         (1.8, 1.9, 'D#:min'),
                         (1.9, 2.0, 'E:min'),
                         (2.0, 2.1, 'F:min'),
                         (2.1, 2.2, 'F#:min'),
                         (2.2, 2.3, 'G:min'),
                         (2.3, 2.4, 'G#:min'),
                         (2.4, 2.5, 'N')]

        labels = majmin_targets_to_chord_labels(targets, fps)
        _compare_labels(self, labels, target_labels)

    def test_frame_join(self):
        fps = 10.
        targets = [0, 0, 4, 4, 4, 4, 24, 8, 8]
        target_labels = [(0.0, 0.2, 'A:maj'),
                         (0.2, 0.6, 'C#:maj'),
                         (0.6, 0.7, 'N'),
                         (0.7, 0.9, 'F:maj')]
        labels = majmin_targets_to_chord_labels(targets, fps)
        _compare_labels(self, labels, target_labels)


class TestCNNChordFeatureProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = CNNChordFeatureProcessor()

    def test_process(self):
        act = self.processor(sample_file)
        self.assertTrue(np.allclose(act, sample_cnn_act))


class TestCRFChordRecognitionProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = CRFChordRecognitionProcessor()

    def test_process(self):
        labels = self.processor(sample_cnn_act)
        _compare_labels(self, labels, sample_labels)


class TestDeepChromaChordRecognitionProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = DeepChromaChordRecognitionProcessor()

    def test_process(self):
        labels = self.processor(sample_deep_chroma_act)
        _compare_labels(self, labels, sample_labels)
