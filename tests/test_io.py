# encoding: utf-8
# pylint: skip-file
"""
This file contains test functions for the madmom.io module.

Please note that most loading functions are tested from within the respective
evaluation module tests, because the expected values are defined in these
tests.

"""

from __future__ import absolute_import, division, print_function

import unittest
from os.path import join as pj, join

from madmom.io import *
from madmom.io import load_beats, load_key, load_notes, load_onsets, load_tempo
from tests import ANNOTATIONS_PATH, DETECTIONS_PATH
from . import DATA_PATH

EVENTS = [1, 1.02, 1.5, 2.0, 2.03, 2.05, 2.5, 3]


class TestLoadEventsFunction(unittest.TestCase):

    def test_read_events_from_file(self):
        events = load_events(pj(DATA_PATH, 'events.txt'))
        self.assertIsInstance(events, np.ndarray)

    def test_read_events_from_file_handle(self):
        file_handle = open(pj(DATA_PATH, 'events.txt'))
        events = load_events(file_handle)
        self.assertIsInstance(events, np.ndarray)
        file_handle.close()

    def test_load_file_with_comments_and_empty_lines(self):
        events = load_events(pj(DATA_PATH, 'commented_txt'))
        self.assertTrue(np.allclose(events, [1.1, 2.1]))


class TestWriteEventsFunction(unittest.TestCase):

    def test_write_events_to_file(self):
        write_events(EVENTS, pj(DATA_PATH, 'events.txt'))
        annotations = load_events(pj(DATA_PATH, 'events.txt'))
        self.assertTrue(np.allclose(annotations, EVENTS))

    def test_write_events_to_file_handle(self):
        file_handle = open(pj(DATA_PATH, 'events.txt'), 'wb')
        write_events(EVENTS, file_handle)
        file_handle.close()
        annotations = load_events(pj(DATA_PATH, 'events.txt'))
        self.assertTrue(np.allclose(annotations, EVENTS))

    def test_write_and_read_events(self):
        write_events(EVENTS, pj(DATA_PATH, 'events.txt'))
        annotations = load_events(pj(DATA_PATH, 'events.txt'))
        self.assertTrue(np.allclose(annotations, EVENTS))


class TestLoadBeatsFunction(unittest.TestCase):

    def test_load_beats_from_file(self):
        beats = load_beats(pj(ANNOTATIONS_PATH, 'sample.beats'))
        from tests.test_evaluation_beats import SAMPLE_BEAT_ANNOTATIONS
        self.assertTrue(np.allclose(beats, SAMPLE_BEAT_ANNOTATIONS))

    def test_load_downbeats_from_file(self):
        downbeats = load_beats(pj(ANNOTATIONS_PATH, 'sample.beats'),
                               downbeats=True)
        self.assertTrue(np.allclose(downbeats, 0.0913))


class TestLoadChordsFunction(unittest.TestCase):

    def test_read_chords_from_file(self):
        chords = load_chords(pj(DETECTIONS_PATH,
                             'sample.dc_chord_recognition.txt'))
        self.assertIsInstance(chords, np.ndarray)
        self.assertEqual(chords[0][0], 0.0)
        self.assertEqual(chords[0][1], 2.9)
        self.assertEqual(chords[0][2], 'G#:maj')

    def test_read_chords_from_file_handle(self):
        with open(pj(DETECTIONS_PATH,
                     'sample.dc_chord_recognition.txt')) as file_handle:
            chords = load_chords(file_handle)
            self.assertIsInstance(chords, np.ndarray)
            self.assertEqual(chords[0][0], 0.0)
            self.assertEqual(chords[0][1], 2.9)
            self.assertEqual(chords[0][2], 'G#:maj')


class TestLoadKeyFunction(unittest.TestCase):

    def test_load_key_from_file(self):
        key = load_key(join(ANNOTATIONS_PATH, 'dummy.key'))
        self.assertEqual(key, 'F# minor')
        key = load_key(join(DETECTIONS_PATH, 'dummy.key.txt'))
        self.assertEqual(key, 'a maj')
        key = load_key(open(join(ANNOTATIONS_PATH, 'dummy.key')))
        self.assertEqual(key, 'F# minor')
        key = load_key(open(join(DETECTIONS_PATH, 'dummy.key.txt')))
        self.assertEqual(key, 'a maj')


class TestLoadNotesFunction(unittest.TestCase):

    def test_load_notes_from_file(self):
        annotations = load_notes(pj(ANNOTATIONS_PATH, 'stereo_sample.notes'))
        self.assertIsInstance(annotations, np.ndarray)

    def test_load_notes_from_file_handle(self):
        file_handle = open(pj(ANNOTATIONS_PATH, 'stereo_sample.notes'))
        annotations = load_notes(file_handle)
        self.assertIsInstance(annotations, np.ndarray)
        file_handle.close()

    def test_load_notes_annotations(self):
        from tests.test_evaluation_notes import ANNOTATIONS
        annotations = load_notes(pj(ANNOTATIONS_PATH, 'stereo_sample.notes'))
        self.assertIsInstance(annotations, np.ndarray)
        self.assertEqual(annotations.shape, (8, 4))
        self.assertTrue(np.allclose(annotations, ANNOTATIONS))


class TestLoadOnsetsFunction(unittest.TestCase):

    def test_load_onsets(self):
        from tests.test_evaluation_onsets import SAMPLE_ANNOTATIONS
        events = load_onsets(pj(ANNOTATIONS_PATH, 'sample.onsets'))
        self.assertTrue(np.allclose(events, SAMPLE_ANNOTATIONS))

    def test_load_onsets_without_comments(self):
        from tests.test_evaluation_onsets import SAMPLE_DETECTIONS

        events = load_onsets(pj(DETECTIONS_PATH, 'sample.super_flux.txt'))
        self.assertTrue(np.allclose(events, SAMPLE_DETECTIONS))

    def test_onsets_with_comments_and_empty_lines(self):
        events = load_onsets(pj(DATA_PATH, 'commented_txt'))
        self.assertTrue(np.allclose(events, [1.1, 2.1]))

    def test_load_timestamps_only(self):
        events = load_onsets(pj(ANNOTATIONS_PATH, 'stereo_sample.notes'))
        self.assertTrue(np.allclose(events, [0.147, 1.567, 2.526, 2.549, 2.563,
                                             2.577, 3.369, 3.449]))


class TestLoadTempoFunction(unittest.TestCase):

    def test_load_tempo_from_file(self):
        annotations = load_tempo(pj(ANNOTATIONS_PATH, 'sample.tempo'))
        self.assertIsInstance(annotations, np.ndarray)

    def test_load_tempo_from_file_handle(self):
        file_handle = open(pj(ANNOTATIONS_PATH, 'sample.tempo'))
        annotations = load_tempo(file_handle)
        self.assertIsInstance(annotations, np.ndarray)
        file_handle.close()

    def test_load_tempo_annotations(self):
        from tests.test_evaluation_tempo import (ANNOTATIONS, ANN_TEMPI,
                                                 ANN_STRENGTHS)
        annotations = load_tempo(pj(ANNOTATIONS_PATH, 'sample.tempo'))
        self.assertIsInstance(annotations, np.ndarray)
        self.assertEqual(annotations.shape, (2, 2))
        self.assertTrue(np.allclose(annotations, ANNOTATIONS))
        self.assertTrue(np.allclose(annotations[:, 0], ANN_TEMPI))
        self.assertTrue(np.allclose(annotations[:, 1], ANN_STRENGTHS))
