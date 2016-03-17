# encoding: utf-8
# pylint: skip-file
"""
This file contains test functions for the madmom.utils module.

"""

from __future__ import absolute_import, division, print_function

import unittest

from madmom.utils import *

from . import (DATA_PATH, AUDIO_PATH, ANNOTATIONS_PATH, ACTIVATIONS_PATH,
               DETECTIONS_PATH)

FILE_LIST = [DATA_PATH + 'README',
             DATA_PATH + 'commented_txt',
             DATA_PATH + 'events.txt']

AUDIO_FILES = [AUDIO_PATH + 'sample.wav',
               AUDIO_PATH + 'stereo_chirp.wav',
               AUDIO_PATH + 'stereo_sample.flac',
               AUDIO_PATH + 'stereo_sample.wav']

ACTIVATION_FILES = [ACTIVATIONS_PATH + 'sample.beats_blstm_2013.npz',
                    ACTIVATIONS_PATH + 'sample.onsets_brnn_2013.npz',
                    ACTIVATIONS_PATH + 'sample.onsets_rnn_2013.npz',
                    ACTIVATIONS_PATH + 'sample.logfiltspecflux.npz',
                    ACTIVATIONS_PATH + 'sample.superflux.npz',
                    ACTIVATIONS_PATH + 'sample.complexflux.npz',
                    ACTIVATIONS_PATH + 'sample.multiband_spectral_flux.npz',
                    ACTIVATIONS_PATH + 'stereo_sample.notes_brnn_2013.npz']

ANNOTATION_FILES = [ANNOTATIONS_PATH + 'sample.beats',
                    ANNOTATIONS_PATH + 'sample.onsets',
                    ANNOTATIONS_PATH + 'sample.sv',
                    ANNOTATIONS_PATH + 'sample.tempo',
                    ANNOTATIONS_PATH + 'stereo_sample.mid',
                    ANNOTATIONS_PATH + 'stereo_sample.notes',
                    ANNOTATIONS_PATH + 'stereo_sample.notes.mirex',
                    ANNOTATIONS_PATH + 'stereo_sample.sv',
                    ANNOTATIONS_PATH + 'piano_sample.mid',
                    ANNOTATIONS_PATH + 'piano_sample.notes_in_beats']

DETECTION_FILES = [DETECTIONS_PATH + 'sample.onsets.txt',
                   DETECTIONS_PATH + 'sample.tempo.txt']

EVENTS = [1, 1.02, 1.5, 2.0, 2.03, 2.05, 2.5, 3]

ONSET_ANNOTATIONS = [0.0943, 0.2844, 0.4528, 0.6160, 0.7630, 0.8025, 0.9847,
                     1.1233, 1.4820, 1.6276, 1.8032, 2.1486, 2.3351, 2.4918,
                     2.6710]
ONSET_DETECTIONS = [0.01, 0.085, 0.275, 0.445, 0.61, 0.795, 0.98, 1.115, 1.365,
                    1.475, 1.62, 1.795, 2.14, 2.33, 2.485, 2.665]


class TestFilterFilesFunction(unittest.TestCase):

    def test_single_file(self):
        # no suffix
        result = filter_files(DATA_PATH + 'README', None)
        self.assertEqual(result, [DATA_PATH + 'README'])
        # single suffix
        result = filter_files(DATA_PATH + 'README', suffix='.txt')
        self.assertEqual(result, [])
        # suffix list
        result = filter_files(DATA_PATH + 'events.txt', suffix=['.txt', None])
        self.assertEqual(result, [DATA_PATH + 'events.txt'])

    def test_file_list(self):
        # no suffix
        result = filter_files(FILE_LIST, None)
        self.assertEqual(result, FILE_LIST)
        # single suffix
        result = filter_files(FILE_LIST, suffix='txt')
        self.assertEqual(result, [DATA_PATH + 'commented_txt',
                                  DATA_PATH + 'events.txt'])
        # suffix list
        result = filter_files(FILE_LIST, suffix=['.txt'])
        self.assertEqual(result, [DATA_PATH + 'events.txt'])


class TestSearchPathFunction(unittest.TestCase):

    def test_path(self):
        result = search_path(DATA_PATH)
        self.assertEqual(result, FILE_LIST)

    def test_recursion(self):
        result = search_path(DATA_PATH, 1)
        all_files = (FILE_LIST + AUDIO_FILES + ANNOTATION_FILES +
                     DETECTION_FILES + ACTIVATION_FILES)
        self.assertEqual(result, sorted(all_files))


class TestSearchFilesFunction(unittest.TestCase):

    def test_file(self):
        # no suffix
        result = search_files(DATA_PATH + 'README')
        self.assertEqual(result, [DATA_PATH + 'README'])
        # single suffix
        result = search_files(DATA_PATH + 'README', suffix='.txt')
        self.assertEqual(result, [])
        # suffix list
        result = search_files(DATA_PATH + 'README', suffix=['.txt', 'txt'])
        self.assertEqual(result, [])
        # non-existing file
        with self.assertRaises(IOError):
            search_files(DATA_PATH + 'non_existing')

    def test_path(self):
        # no suffix
        result = search_files(DATA_PATH)
        self.assertEqual(result, sorted(FILE_LIST))
        # single suffix
        result = search_files(DATA_PATH, suffix='txt')
        file_list = [DATA_PATH + 'commented_txt',
                     DATA_PATH + 'events.txt']
        self.assertEqual(result, sorted(file_list))
        # another suffix
        result = search_files(DATA_PATH, suffix='.txt')
        file_list = [DATA_PATH + 'events.txt']
        self.assertEqual(result, sorted(file_list))
        # suffix list
        result = search_files(DATA_PATH, suffix=['.txt', 'txt'])
        file_list = [DATA_PATH + 'commented_txt',
                     DATA_PATH + 'events.txt']
        self.assertEqual(result, sorted(file_list))

    def test_file_list(self):
        # no suffix
        result = search_files(FILE_LIST)
        self.assertEqual(result, sorted(FILE_LIST))
        # single suffix
        result = search_files(FILE_LIST, suffix='.txt')
        self.assertEqual(result, [DATA_PATH + 'events.txt'])
        # suffix list
        result = search_files(FILE_LIST, suffix=['.txt', 'txt'])
        self.assertEqual(result, [DATA_PATH + 'commented_txt',
                                  DATA_PATH + 'events.txt'])


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
        self.assertEqual(match_file('file.txt', match_list),
                         ['file.txt', '/path/file.txt'])

    def test_match_other_suffix(self):
        match_list = ['file.txt', '/path/file.txt', '/path/file.txt.other']
        result = match_file('file.txt', match_list, match_suffix='other')
        self.assertEqual(result, [])

    def test_match_dot_other_suffix(self):
        match_list = ['file.txt', '/path/file.txt', '/path/file.txt.other']
        result = match_file('file.txt', match_list, match_suffix='.other')
        self.assertEqual(result, ['/path/file.txt.other'])


class TestLoadEventsFunction(unittest.TestCase):

    def test_read_events_from_file(self):
        events = load_events(DATA_PATH + 'events.txt')
        self.assertIsInstance(events, np.ndarray)

    def test_read_events_from_file_handle(self):
        file_handle = open(DATA_PATH + 'events.txt')
        events = load_events(file_handle)
        self.assertIsInstance(events, np.ndarray)
        file_handle.close()

    def test_read_onset_annotations(self):
        events = load_events(ANNOTATIONS_PATH + 'sample.onsets')
        self.assertTrue(np.allclose(events, ONSET_ANNOTATIONS))

    def test_read_file_without_comments(self):
        events = load_events(DETECTIONS_PATH + 'sample.onsets.txt')
        self.assertTrue(np.allclose(events, [0.01, 0.085, 0.275, 0.445, 0.61,
                                             0.795, 0.98, 1.115, 1.365, 1.475,
                                             1.62, 1.795, 2.14, 2.33, 2.485,
                                             2.665]))

    def test_load_file_with_comments_and_empty_lines(self):
        events = load_events(DATA_PATH + 'commented_txt')
        self.assertTrue(np.allclose(events, [1.1, 2.1]))

    def test_load_only_timestamps(self):
        events = load_events(ANNOTATIONS_PATH + 'stereo_sample.notes')
        self.assertTrue(np.allclose(events, [0.147, 1.567, 2.526, 2.549, 2.563,
                                             2.577, 3.369, 3.449]))


class TestWriteEventsFunction(unittest.TestCase):

    def test_write_events_to_file(self):
        result = write_events(EVENTS, DATA_PATH + 'events.txt')
        self.assertEqual(EVENTS, result)

    def test_write_events_to_file_handle(self):
        file_handle = open(DATA_PATH + 'events.txt', 'wb')
        result = write_events(EVENTS, file_handle)
        self.assertEqual(EVENTS, result)
        file_handle.close()

    def test_write_and_read_events(self):
        write_events(EVENTS, DATA_PATH + 'events.txt')
        annotations = load_events(DATA_PATH + 'events.txt')
        self.assertTrue(np.allclose(annotations, EVENTS))


class TestCombineEventsFunction(unittest.TestCase):

    def test_combine_000(self):
        comb = combine_events(EVENTS, 0.)
        correct = np.asarray([1, 1.02, 1.5, 2.0, 2.03, 2.05, 2.5, 3])
        self.assertTrue(np.allclose(comb, correct))

    def test_combine_001(self):
        comb = combine_events(EVENTS, 0.01)
        correct = np.asarray([1, 1.02, 1.5, 2.0, 2.03, 2.05, 2.5, 3])
        self.assertTrue(np.allclose(comb, correct))

    def test_combine_003(self):
        comb = combine_events(EVENTS, 0.03)
        correct = np.asarray([1.01, 1.5, 2.015, 2.05, 2.5, 3])
        self.assertTrue(np.allclose(comb, correct))

    def test_combine_0035(self):
        comb = combine_events(EVENTS, 0.035)
        correct = np.asarray([1.01, 1.5, 2.0325, 2.5, 3])
        self.assertTrue(np.allclose(comb, correct))

    def test_combine_short(self):
        comb = combine_events([1], 0.035)
        correct = np.asarray([1])
        self.assertTrue(np.allclose(comb, correct))


class TestQuantizeEventsFunction(unittest.TestCase):

    def test_fps(self):
        # 10 FPS
        quantized = quantize_events(EVENTS, 10, length=None)
        idx = np.nonzero(quantized)[0]
        # tar: [1, 1.02, 1.5, 2.0, 2.03, 2.05, 2.5, 3]
        self.assertTrue(np.allclose(idx, [10, 15, 20, 25, 30]))
        # 100 FPS
        quantized = quantize_events(EVENTS, 100, length=None)
        idx = np.nonzero(quantized)[0]
        # tar: [1, 1.02, 1.5, 2.0, 2.03, 2.05, 2.5, 3]
        correct = [100, 102, 150, 200, 203, 205, 250, 300]
        self.assertTrue(np.allclose(idx, correct))

    def test_length(self):
        # length = 280
        quantized = quantize_events(EVENTS, 100, length=280)
        idx = np.nonzero(quantized)[0]
        # targets: [1, 1.02, 1.5, 2.0, 2.03, 2.05, 2.5, 3]
        self.assertTrue(np.allclose(idx, [100, 102, 150, 200, 203, 205, 250]))

    def test_rounding(self):
        # without length
        quantized = quantize_events([3.95], 10, length=None)
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
        signal = Signal(AUDIO_PATH + '/sample.wav')
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
