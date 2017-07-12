# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.features.chords module.

"""

from __future__ import absolute_import, division, print_function

import unittest
from os.path import join as pj

from madmom.features import Activations
from madmom.features.chords import *
from madmom.io import load_chords
from . import ACTIVATIONS_PATH, AUDIO_PATH, DETECTIONS_PATH

sample_files = [pj(AUDIO_PATH, sf) for sf in ['sample.wav', 'sample2.wav']]

sample_cnn_acts = [Activations(pj(ACTIVATIONS_PATH, af))
                   for af in ['sample.cnn_chord_features.npz',
                              'sample2.cnn_chord_features.npz']]

sample_cnn_labels = [load_chords(pj(DETECTIONS_PATH, df))
                     for df in ['sample.cnn_chord_recognition.txt',
                                'sample2.cnn_chord_recognition.txt']]

sample_deep_chroma_acts = [Activations(pj(ACTIVATIONS_PATH, af))
                           for af in ['sample.deep_chroma.npz',
                                      'sample2.deep_chroma.npz']]

sample_deep_chroma_labels = [load_chords(pj(DETECTIONS_PATH, df))
                             for df in ['sample.dc_chord_recognition.txt',
                                        'sample2.dc_chord_recognition.txt']]


def _compare_labels(test_case, labels, reference_labels):
    test_case.assertTrue(
        np.allclose(labels['start'], reference_labels['start']))
    test_case.assertTrue(np.allclose(labels['end'], reference_labels['end']))
    test_case.assertTrue((labels['label'] == reference_labels['label']).all())


class TestLoadSegmentsFunction(unittest.TestCase):
    def test_read_segments_from_file(self):
        chords = load_chords(pj(DETECTIONS_PATH,
                                'sample2.dc_chord_recognition.txt'))
        self.assertIsInstance(chords, np.ndarray)

    def test_read_segments_from_file_handle(self):
        with open(pj(DETECTIONS_PATH,
                     'sample2.dc_chord_recognition.txt')) as file_handle:
            chords = load_chords(file_handle)
            self.assertIsInstance(chords, np.ndarray)

    def test_read_segment_annotations(self):
        chords = load_chords(pj(DETECTIONS_PATH,
                                'sample2.dc_chord_recognition.txt'))
        _compare_labels(self, chords,
                        np.array([(0.0, 1.6, 'F:maj'),
                                  (1.6, 2.5, 'A:maj'),
                                  (2.5, 4.1, 'D:maj')], dtype=CHORD_DTYPE))

        chords = load_chords(pj(DETECTIONS_PATH,
                                'sample.dc_chord_recognition.txt'))
        _compare_labels(self, chords,
                        np.array([(0.0, 2.9, 'G#:maj')], dtype=CHORD_DTYPE))


class TestMajMinTargetsToChordLabelsFunction(unittest.TestCase):
    def test_all_labels(self):
        fps = 10.
        targets = range(25)
        target_labels = np.array([(0.0, 0.1, 'A:maj'),
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
                                  (2.4, 2.5, 'N')],
                                 dtype=CHORD_DTYPE)

        labels = majmin_targets_to_chord_labels(targets, fps)
        _compare_labels(self, labels, target_labels)

    def test_frame_join(self):
        fps = 10.
        targets = [0, 0, 4, 4, 4, 4, 24, 8, 8]
        target_labels = np.array([(0.0, 0.2, 'A:maj'),
                                  (0.2, 0.6, 'C#:maj'),
                                  (0.6, 0.7, 'N'),
                                  (0.7, 0.9, 'F:maj')], dtype=CHORD_DTYPE)
        labels = majmin_targets_to_chord_labels(targets, fps)
        _compare_labels(self, labels, target_labels)


class TestCNNChordFeatureProcessorClass(unittest.TestCase):
    def setUp(self):
        self.processor = CNNChordFeatureProcessor()

    def test_process(self):
        for audio_file, true_activation in zip(sample_files, sample_cnn_acts):
            act = self.processor(audio_file)
            self.assertTrue(np.allclose(act, true_activation))


class TestCRFChordRecognitionProcessorClass(unittest.TestCase):
    def setUp(self):
        self.processor = CRFChordRecognitionProcessor()

    def test_process(self):
        for activation, true_labels in zip(sample_cnn_acts, sample_cnn_labels):
            labels = self.processor(activation)
            _compare_labels(self, labels, true_labels)


class TestDeepChromaChordRecognitionProcessorClass(unittest.TestCase):
    def setUp(self):
        self.processor = DeepChromaChordRecognitionProcessor()

    def test_process(self):
        for activation, true_labels in zip(sample_deep_chroma_acts,
                                           sample_deep_chroma_labels):
            labels = self.processor(activation)
            _compare_labels(self, labels, true_labels)
