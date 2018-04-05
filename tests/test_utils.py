# encoding: utf-8
# pylint: skip-file
"""
This file contains test functions for the madmom.utils module.

"""

from __future__ import absolute_import, division, print_function

import unittest
from os.path import join as pj

from madmom.io import load_events
from madmom.utils import *
from . import (ACTIVATIONS_PATH, ANNOTATIONS_PATH, AUDIO_PATH, DATA_PATH,
               DETECTIONS_PATH)
from .test_features_notes import NOTES

FILE_LIST = [pj(DATA_PATH, 'README'),
             pj(DATA_PATH, 'commented_txt'),
             pj(DATA_PATH, 'events.txt')]

AUDIO_FILES = [pj(AUDIO_PATH, 'sample.wav'),
               pj(AUDIO_PATH, 'sample2.wav'),
               pj(AUDIO_PATH, 'sample_22050.wav'),
               pj(AUDIO_PATH, 'stereo_chirp.wav'),
               pj(AUDIO_PATH, 'stereo_sample.flac'),
               pj(AUDIO_PATH, 'stereo_sample.wav')]

ACTIVATION_FILES = [pj(ACTIVATIONS_PATH, 'sample.bar_tracker.npz'),
                    pj(ACTIVATIONS_PATH, 'sample.beats_blstm.npz'),
                    pj(ACTIVATIONS_PATH, 'sample.beats_blstm_mm.npz'),
                    pj(ACTIVATIONS_PATH, 'sample.beats_lstm.npz'),
                    pj(ACTIVATIONS_PATH, 'sample.cnn_chord_features.npz'),
                    pj(ACTIVATIONS_PATH, 'sample.downbeats_blstm.npz'),
                    pj(ACTIVATIONS_PATH, 'sample.deep_chroma.npz'),
                    pj(ACTIVATIONS_PATH, 'sample.complex_flux.npz'),
                    pj(ACTIVATIONS_PATH, 'sample.gmm_pattern_tracker.npz'),
                    pj(ACTIVATIONS_PATH, 'sample.key_cnn.npz'),
                    pj(ACTIVATIONS_PATH, 'sample2.key_cnn.npz'),
                    pj(ACTIVATIONS_PATH, 'sample.log_filt_spec_flux.npz'),
                    pj(ACTIVATIONS_PATH, 'sample.onsets_cnn.npz'),
                    pj(ACTIVATIONS_PATH, 'sample.onsets_brnn.npz'),
                    pj(ACTIVATIONS_PATH, 'sample.onsets_rnn.npz'),
                    pj(ACTIVATIONS_PATH, 'sample.spectral_flux.npz'),
                    pj(ACTIVATIONS_PATH, 'sample.super_flux.npz'),
                    pj(ACTIVATIONS_PATH, 'sample.super_flux_nn.npz'),
                    pj(ACTIVATIONS_PATH, 'sample2.cnn_chord_features.npz'),
                    pj(ACTIVATIONS_PATH, 'sample2.deep_chroma.npz'),
                    pj(ACTIVATIONS_PATH, 'stereo_sample.notes_brnn.npz')]

ANNOTATION_FILES = [pj(ANNOTATIONS_PATH, 'dummy.chords'),
                    pj(ANNOTATIONS_PATH, 'sample.beats'),
                    pj(ANNOTATIONS_PATH, 'dummy.key'),
                    pj(ANNOTATIONS_PATH, 'sample.onsets'),
                    pj(ANNOTATIONS_PATH, 'sample.sv'),
                    pj(ANNOTATIONS_PATH, 'sample.tempo'),
                    pj(ANNOTATIONS_PATH, 'stereo_sample.mid'),
                    pj(ANNOTATIONS_PATH, 'stereo_sample.notes'),
                    pj(ANNOTATIONS_PATH, 'stereo_sample.notes.mirex'),
                    pj(ANNOTATIONS_PATH, 'stereo_sample.sv'),
                    pj(ANNOTATIONS_PATH, 'multitrack.mid'),
                    pj(ANNOTATIONS_PATH, 'piano_sample.mid'),
                    pj(ANNOTATIONS_PATH, 'piano_sample.notes_in_beats')]

DETECTION_FILES = [pj(DETECTIONS_PATH, 'dummy.chords.txt'),
                   pj(DETECTIONS_PATH, 'dummy.key.txt'),
                   pj(DETECTIONS_PATH, 'sample.beat_detector.txt'),
                   pj(DETECTIONS_PATH, 'sample.beat_tracker.txt'),
                   pj(DETECTIONS_PATH, 'sample.cnn_chord_recognition.txt'),
                   pj(DETECTIONS_PATH, 'sample.cnn_onset_detector.txt'),
                   pj(DETECTIONS_PATH, 'sample.complex_flux.txt'),
                   pj(DETECTIONS_PATH, 'sample.crf_beat_detector.txt'),
                   pj(DETECTIONS_PATH, 'sample.dbn_beat_tracker.txt'),
                   pj(DETECTIONS_PATH, 'sample.dbn_downbeat_tracker.txt'),
                   pj(DETECTIONS_PATH, 'sample.dc_chord_recognition.txt'),
                   pj(DETECTIONS_PATH, 'sample.gmm_pattern_tracker.txt'),
                   pj(DETECTIONS_PATH, 'sample.key_recognition.txt'),
                   pj(DETECTIONS_PATH, 'sample2.key_recognition.txt'),
                   pj(DETECTIONS_PATH, 'sample.log_filt_spec_flux.txt'),
                   pj(DETECTIONS_PATH, 'sample.mm_beat_tracker.txt'),
                   pj(DETECTIONS_PATH, 'sample.onset_detector.txt'),
                   pj(DETECTIONS_PATH, 'sample.onset_detector_ll.txt'),
                   pj(DETECTIONS_PATH, 'sample.spectral_flux.txt'),
                   pj(DETECTIONS_PATH, 'sample.super_flux.txt'),
                   pj(DETECTIONS_PATH, 'sample.super_flux_nn.txt'),
                   pj(DETECTIONS_PATH, 'sample.tempo_detector.txt'),
                   pj(DETECTIONS_PATH, 'sample2.cnn_chord_recognition.txt'),
                   pj(DETECTIONS_PATH, 'sample2.dc_chord_recognition.txt'),
                   pj(DETECTIONS_PATH, 'stereo_sample.piano_transcriptor.txt')]

EVENTS = [1, 1.02, 1.5, 2.0, 2.03, 2.05, 2.5, 3]

ONSET_ANNOTATIONS = [0.0943, 0.2844, 0.4528, 0.6160, 0.7630, 0.8025, 0.9847,
                     1.1233, 1.4820, 1.6276, 1.8032, 2.1486, 2.3351, 2.4918,
                     2.6710]
ONSET_DETECTIONS = [0.01, 0.085, 0.275, 0.445, 0.61, 0.795, 0.98, 1.115, 1.365,
                    1.475, 1.62, 1.795, 2.14, 2.33, 2.485, 2.665]


class TestFilterFilesFunction(unittest.TestCase):

    def test_single_file(self):
        # no suffix
        result = filter_files(pj(DATA_PATH, 'README'), None)
        self.assertEqual(result, [pj(DATA_PATH, 'README')])
        # single suffix
        result = filter_files(pj(DATA_PATH, 'README'), suffix='.txt')
        self.assertEqual(result, [])
        # suffix list
        result = filter_files(pj(DATA_PATH, 'events.txt'),
                              suffix=['.txt', None])
        self.assertEqual(result, [pj(DATA_PATH, 'events.txt')])

    def test_file_list(self):
        # no suffix
        result = filter_files(FILE_LIST, None)
        self.assertEqual(result, FILE_LIST)
        # single suffix
        result = filter_files(FILE_LIST, suffix='txt')
        self.assertEqual(result, [pj(DATA_PATH, 'commented_txt'),
                                  pj(DATA_PATH, 'events.txt')])
        # suffix list
        result = filter_files(FILE_LIST, suffix=['.txt'])
        self.assertEqual(result, [pj(DATA_PATH, 'events.txt')])


class TestSearchPathFunction(unittest.TestCase):

    def test_path(self):
        result = search_path(DATA_PATH)
        self.assertEqual(result, FILE_LIST)

    def test_recursion(self):
        result = search_path(DATA_PATH, 1)
        all_files = (FILE_LIST + AUDIO_FILES + ANNOTATION_FILES +
                     DETECTION_FILES + ACTIVATION_FILES)
        self.assertEqual(result, sorted(all_files))

    def test_errors(self):
        with self.assertRaises(IOError):
            search_path(pj(DATA_PATH, 'README'))


class TestSearchFilesFunction(unittest.TestCase):

    def test_file(self):
        # no suffix
        result = search_files(pj(DATA_PATH, 'README'))
        self.assertEqual(result, [pj(DATA_PATH, 'README')])
        # single suffix
        result = search_files(pj(DATA_PATH, 'README'), suffix='.txt')
        self.assertEqual(result, [])
        # suffix list
        result = search_files(pj(DATA_PATH, 'README'), suffix=['.txt', 'txt'])
        self.assertEqual(result, [])
        # non-existing file
        with self.assertRaises(IOError):
            search_files(pj(DATA_PATH, 'non_existing'))

    def test_path(self):
        # no suffix
        result = search_files(DATA_PATH)
        self.assertEqual(result, sorted(FILE_LIST))
        # single suffix
        result = search_files(DATA_PATH, suffix='txt')
        file_list = [pj(DATA_PATH, 'commented_txt'),
                     pj(DATA_PATH, 'events.txt')]
        self.assertEqual(result, sorted(file_list))
        # another suffix
        result = search_files(DATA_PATH, suffix='.txt')
        file_list = [pj(DATA_PATH, 'events.txt')]
        self.assertEqual(result, sorted(file_list))
        # suffix list
        result = search_files(DATA_PATH, suffix=['.txt', 'txt'])
        file_list = [pj(DATA_PATH, 'commented_txt'),
                     pj(DATA_PATH, 'events.txt')]
        self.assertEqual(result, sorted(file_list))

    def test_file_list(self):
        # no suffix
        result = search_files(FILE_LIST)
        self.assertEqual(result, sorted(FILE_LIST))
        # single suffix
        result = search_files(FILE_LIST, suffix='.txt')
        self.assertEqual(result, [pj(DATA_PATH, 'events.txt')])
        # suffix list
        result = search_files(FILE_LIST, suffix=['.txt', 'txt'])
        self.assertEqual(result, [pj(DATA_PATH, 'commented_txt'),
                                  pj(DATA_PATH, 'events.txt')])


class TestStripSuffixFunction(unittest.TestCase):
    # tests for strip_suffix(filename, ext=None)
    def test_strip_txt_suffix(self):
        self.assertEqual(strip_suffix('file.txt', 'txt'), 'file.')
        self.assertEqual(strip_suffix('/path/file.txt', 'txt'), '/path/file.')

    def test_strip_dot_txt_suffix(self):
        self.assertEqual(strip_suffix('file.txt', '.txt'), 'file')
        self.assertEqual(strip_suffix('/path/file.txt', '.txt'), '/path/file')


class TestMatchFileFunction(unittest.TestCase):
    # test for match_file(filename, match_list, ext=None, match_suffix=None)
    def test_match_dot_txt_suffix(self):
        match_list = ['file.txt', '/path/file.txt', '/path/file.txt.other']
        result = match_file('file.txt', match_list)
        self.assertEqual(result, ['file.txt', '/path/file.txt'])
        result = match_file('file.txt', match_list, match_exactly=False)
        self.assertEqual(result, ['file.txt', '/path/file.txt'])

    def test_match_other_suffix(self):
        match_list = ['file.txt', '/path/file.txt', '/path/file.txt.other']
        result = match_file('file.txt', match_list, match_suffix='other')
        self.assertEqual(result, [])
        result = match_file('file.txt', match_list, match_suffix='other',
                            match_exactly=False)
        self.assertEqual(result, ['/path/file.txt.other'])

    def test_match_dot_other_suffix(self):
        match_list = ['file.txt', '/path/file.txt', '/path/file.txt.other']
        result = match_file('file.txt', match_list, match_suffix='.other')
        self.assertEqual(result, ['/path/file.txt.other'])
        result = match_file('txt', match_list, match_suffix='.other',
                            match_exactly=False)
        self.assertEqual(result, ['/path/file.txt.other'])
        result = match_file('other', match_list, match_exactly=False)
        self.assertEqual(result, ['/path/file.txt.other'])

    def test_match_any_suffix(self):
        match_list = ['file.txt', '/path/file.txt', '/path/file.txt.other']
        result = match_file('file.txt', match_list, match_suffix='*')
        self.assertEqual(result, ['file.txt', '/path/file.txt'])
        result = match_file('file.txt', match_list, match_suffix='*',
                            match_exactly=False)
        self.assertEqual(result, match_list)


class TestCombineEventsFunction(unittest.TestCase):

    def test_combine_mean(self):
        # EVENTS =           [1, 1.02, 1.5, 2.0, 2.03, 2.05, 2.5, 3]
        comb = combine_events(EVENTS, 0.)
        correct = np.asarray([1, 1.02, 1.5, 2.0, 2.03, 2.05, 2.5, 3])
        self.assertTrue(np.allclose(comb, correct))
        comb = combine_events(EVENTS, 0.01)
        correct = np.asarray([1, 1.02, 1.5, 2.0, 2.03, 2.05, 2.5, 3])
        self.assertTrue(np.allclose(comb, correct))
        comb = combine_events(EVENTS, 0.03)
        correct = np.asarray([1.01, 1.5, 2.015, 2.05, 2.5, 3])
        self.assertTrue(np.allclose(comb, correct))
        comb = combine_events(EVENTS, 0.035)
        correct = np.asarray([1.01, 1.5, 2.0325, 2.5, 3])
        self.assertTrue(np.allclose(comb, correct))
        comb = combine_events([1], 0.035)
        correct = np.asarray([1])
        self.assertTrue(np.allclose(comb, correct))

    def test_combine_left(self):
        # EVENTS =           [1, 1.02, 1.5, 2.0, 2.03, 2.05, 2.5, 3]
        comb = combine_events(EVENTS, 0., 'left')
        correct = np.asarray([1, 1.02, 1.5, 2.0, 2.03, 2.05, 2.5, 3])
        self.assertTrue(np.allclose(comb, correct))
        comb = combine_events(EVENTS, 0.01, 'left')
        correct = np.asarray([1, 1.02, 1.5, 2.0, 2.03, 2.05, 2.5, 3])
        self.assertTrue(np.allclose(comb, correct))
        comb = combine_events(EVENTS, 0.03, 'left')
        correct = np.asarray([1, 1.5, 2, 2.05, 2.5, 3])
        self.assertTrue(np.allclose(comb, correct))
        comb = combine_events(EVENTS, 0.035, 'left')
        correct = np.asarray([1, 1.5, 2, 2.05, 2.5, 3])
        self.assertTrue(np.allclose(comb, correct))
        comb = combine_events(EVENTS, 0.05, 'left')
        correct = np.asarray([1, 1.5, 2, 2.5, 3])
        self.assertTrue(np.allclose(comb, correct))

    def test_combine_right(self):
        # EVENTS =           [1, 1.02, 1.5, 2.0, 2.03, 2.05, 2.5, 3]
        comb = combine_events(EVENTS, 0., 'right')
        correct = np.asarray([1, 1.02, 1.5, 2.0, 2.03, 2.05, 2.5, 3])
        self.assertTrue(np.allclose(comb, correct))
        comb = combine_events(EVENTS, 0.01, 'right')
        correct = np.asarray([1, 1.02, 1.5, 2.0, 2.03, 2.05, 2.5, 3])
        self.assertTrue(np.allclose(comb, correct))
        comb = combine_events(EVENTS, 0.03, 'right')
        correct = np.asarray([1.02, 1.5, 2.05, 2.5, 3])
        self.assertTrue(np.allclose(comb, correct))
        comb = combine_events(EVENTS, 0.035, 'right')
        correct = np.asarray([1.02, 1.5, 2.05, 2.5, 3])
        self.assertTrue(np.allclose(comb, correct))
        comb = combine_events(EVENTS, 0.05, 'right')
        correct = np.asarray([1.02, 1.5, 2.05, 2.5, 3])
        self.assertTrue(np.allclose(comb, correct))

    def test_errors(self):
        with self.assertRaises(ValueError):
            combine_events(np.arange(6).reshape((2, 3)), 0.5)
        with self.assertRaises(ValueError):
            combine_events(EVENTS, 0.5, 'foo')


class TestQuantizeEventsFunction(unittest.TestCase):

    def test_fps(self):
        # 10 FPS
        quantized = quantize_events(EVENTS, 10)
        idx = np.nonzero(quantized)[0]
        # tar: [1, 1.02, 1.5, 2.0, 2.03, 2.05, 2.5, 3]
        self.assertTrue(np.allclose(idx, [10, 15, 20, 25, 30]))
        # 100 FPS with numpy arrays (array must not be changed)
        events = np.array(EVENTS)
        events_ = np.copy(events)
        quantized = quantize_events(events, 100)
        idx = np.nonzero(quantized)[0]
        # tar: [1, 1.02, 1.5, 2.0, 2.03, 2.05, 2.5, 3]
        correct = [100, 102, 150, 200, 203, 205, 250, 300]
        self.assertTrue(np.allclose(idx, correct))
        self.assertTrue(np.allclose(events, events_))

    def test_length(self):
        # length = 280
        quantized = quantize_events(EVENTS, 100, length=280)
        idx = np.nonzero(quantized)[0]
        # targets: [1, 1.02, 1.5, 2.0, 2.03, 2.05, 2.5, 3]
        self.assertTrue(np.allclose(idx, [100, 102, 150, 200, 203, 205, 250]))

    def test_rounding(self):
        # without length
        quantized = quantize_events([3.95], 10)
        idx = np.nonzero(quantized)[0]
        self.assertTrue(np.allclose(idx, [40]))
        # with length
        quantized = quantize_events([3.95], 10, length=39)
        idx = np.nonzero(quantized)[0]
        self.assertTrue(np.allclose(idx, []))
        # round down with length
        quantized = quantize_events([3.9499999], 10, length=40)
        idx = np.nonzero(quantized)[0]
        self.assertTrue(np.allclose(idx, [39]))

    def test_shift(self):
        # no length
        quantized = quantize_events(EVENTS, 10, shift=1)
        idx = np.nonzero(quantized)[0]
        self.assertTrue(np.allclose(idx, [20, 25, 30, 35, 40]))
        # limited length
        quantized = quantize_events(EVENTS, 10, shift=1, length=35)
        idx = np.nonzero(quantized)[0]
        correct = [20, 25, 30]
        self.assertTrue(np.allclose(idx, correct))

    def test_errors(self):
        with self.assertRaises(ValueError):
            quantize_events(1, fps=100)
        with self.assertRaises(ValueError):
            quantize_events([[0], [1], [2]], fps=100)
        with self.assertRaises(ValueError):
            quantize_events(np.arange(9).reshape((3, 3, 3)), fps=100)


class TestQuantizeNotesFunction(unittest.TestCase):

    def test_fps(self):
        # 10 FPS
        fps = 10
        quantized = quantize_notes(NOTES, fps=fps)
        self.assertTrue(quantized.shape == (42, 78))
        idx = np.nonzero(quantized)
        correct = np.arange(np.round(NOTES[0, 0] * fps),
                            np.round((NOTES[0, 0] + NOTES[0, 2]) * fps) + 1)
        self.assertTrue(np.allclose(idx[0][idx[1] == 72], correct))
        # 100 FPS with numpy arrays (array must not be changed)
        fps = 100
        notes = np.array(NOTES)
        notes_ = np.copy(notes)
        quantized = quantize_notes(notes, fps=fps)
        self.assertTrue(quantized.shape == (416, 78))
        idx = np.nonzero(quantized)
        correct = np.arange(np.round(NOTES[1, 0] * fps),
                            np.round((NOTES[1, 0] + NOTES[1, 2]) * fps) + 1)
        self.assertTrue(np.allclose(idx[0][idx[1] == 41], correct))
        self.assertTrue(np.allclose(notes, notes_))

    def test_length(self):
        fps = 100
        length = 280
        quantized = quantize_notes(NOTES, fps=fps, length=length)
        self.assertTrue(quantized.shape == (280, 78))
        idx = np.nonzero(quantized)
        correct = np.arange(np.round(NOTES[0, 0] * fps), length)
        self.assertTrue(np.allclose(idx[0][idx[1] == 72], correct))

    def test_rounding(self):
        # rounding towards next even number
        quantized = quantize_notes([[0.95, 0], [1.95, 1]], fps=10)
        self.assertTrue(np.allclose(np.nonzero(quantized), [[10, 20], [0, 1]]))
        quantized = quantize_notes([[0.85, 0], [1.85, 1]], fps=10)
        self.assertTrue(np.allclose(np.nonzero(quantized), [[8, 18], [0, 1]]))
        # round down
        quantized = quantize_notes([[0.9499999, 0]], fps=10)
        self.assertTrue(np.allclose(np.nonzero(quantized), [[9], [0]]))
        # with length
        quantized = quantize_notes([[0.95, 0], [1.95, 1]], fps=10, length=15)
        self.assertTrue(np.allclose(np.nonzero(quantized), [[10], [0]]))

    def test_num_notes(self):
        fps = 10
        num_pitches = 73
        quantized = quantize_notes(NOTES, fps=fps, num_pitches=num_pitches)
        self.assertTrue(quantized.shape == (42, 73))
        idx = np.nonzero(quantized)
        correct = np.arange(np.round(NOTES[0, 0] * fps),
                            np.round((NOTES[0, 0] + NOTES[0, 2]) * fps) + 1)
        self.assertTrue(np.allclose(idx[0][idx[1] == 72], correct))
        num_pitches = 72
        quantized = quantize_notes(NOTES, fps=fps, num_pitches=num_pitches),
        idx = np.nonzero(quantized)
        self.assertTrue(np.allclose(idx[0][idx[1] == 72], []))

    def test_velocity(self):
        fps = 10
        quantized = quantize_notes(NOTES, fps=fps)
        self.assertTrue(quantized.shape == (42, 78))
        self.assertTrue(np.max(quantized) == 72)
        self.assertTrue(np.allclose(quantized[:, 72][0], 0))
        self.assertTrue(np.allclose(quantized[:, 72][1: 36], 63))
        self.assertTrue(np.allclose(quantized[:, 72][36:], 0))
        # set velocity
        quantized = quantize_notes(NOTES, fps=fps, velocity=1.5)
        self.assertTrue(np.max(quantized) == 1.5)
        # default velocity if not given by the notes
        quantized = quantize_notes([[0.95, 0], [1.95, 1]], fps=10)
        self.assertTrue(np.allclose(quantized[np.nonzero(quantized)], 1))
        quantized = quantize_notes([[0.95, 0], [1.95, 1]], fps=10, velocity=5)
        self.assertTrue(np.allclose(quantized[np.nonzero(quantized)], 5))

    def test_errors(self):
        with self.assertRaises(ValueError):
            quantize_notes([0, 1, 2], fps=100)
        with self.assertRaises(ValueError):
            quantize_notes([[0], [1], [2]], fps=100)
        with self.assertRaises(ValueError):
            quantize_notes(np.arange(8).reshape((2, 2, 2)), fps=100)


class TestSegmentAxisFunction(unittest.TestCase):

    def test_types(self):
        result = segment_axis(np.arange(10), 4, 2)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(result.dtype == np.int)
        result = segment_axis(np.arange(10, dtype=np.float), 4, 2)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(result.dtype == np.float)
        # test with a Signal
        from madmom.audio.signal import Signal
        signal = Signal(pj(AUDIO_PATH, 'sample.wav'))
        result = segment_axis(signal, 4, 2)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(result.dtype == np.int16)

    def test_errors(self):
        # test wrong axis
        with self.assertRaises(ValueError):
            segment_axis(np.arange(10), 4, 2, axis=1)
        # testing 0 frame_size
        with self.assertRaises(ValueError):
            segment_axis(np.arange(10), 0, 2)
        # testing 0 hop_size
        with self.assertRaises(ValueError):
            segment_axis(np.arange(10), 4, 0)
        with self.assertRaises(ValueError):
            # not enough data points for frame length
            segment_axis(np.arange(3), 4, 2)

    def test_values(self):
        result = segment_axis(np.arange(10), 4, 2)
        self.assertTrue(np.allclose(result, [[0, 1, 2, 3], [2, 3, 4, 5],
                                             [4, 5, 6, 7], [6, 7, 8, 9]]))
        result = segment_axis(np.arange(10), 4, 3, end='pad')
        self.assertTrue(np.allclose(result, [[0, 1, 2, 3], [3, 4, 5, 6],
                                             [6, 7, 8, 9]]))
        result = segment_axis(np.arange(11), 4, 3, end='pad')
        self.assertTrue(np.allclose(result, [[0, 1, 2, 3], [3, 4, 5, 6],
                                             [6, 7, 8, 9], [9, 10, 0, 0]]))
        result = segment_axis(np.arange(11), 4, 3, end='pad', end_value=1)
        self.assertTrue(np.allclose(result, [[0, 1, 2, 3], [3, 4, 5, 6],
                                             [6, 7, 8, 9], [9, 10, 1, 1]]))
        result = segment_axis(np.arange(11), 4, 3, end='wrap')
        self.assertTrue(np.allclose(result, [[0, 1, 2, 3], [3, 4, 5, 6],
                                             [6, 7, 8, 9], [9, 10, 0, 1]]))
        result = segment_axis(np.arange(11), 4, 3, end='cut')
        self.assertTrue(np.allclose(result, [[0, 1, 2, 3], [3, 4, 5, 6],
                                             [6, 7, 8, 9]]))
        result = segment_axis(np.arange(11), 4, 3, axis=0)
        self.assertTrue(np.allclose(result, [[0, 1, 2, 3], [3, 4, 5, 6],
                                             [6, 7, 8, 9]]))
        result = segment_axis(np.arange(3), 4, 2, end='wrap')
        self.assertTrue(np.allclose(result, [[0, 1, 2, 0]]))
        result = segment_axis(np.arange(3), 4, 2, end='pad', end_value=9)
        self.assertTrue(np.allclose(result, [[0, 1, 2, 9]]))
