# encoding: utf-8
"""
This file contains test functions for the madmom.utils module.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""
# pylint: skip-file

import unittest
import __builtin__

from . import DATA_PATH
from madmom.utils import *


EVENTS = [1, 1.02, 1.5, 2.0, 2.03, 2.05, 2.5, 3]

ANNOTATIONS = [0.0943, 0.2844, 0.4528, 0.6160, 0.7630, 0.8025, 0.9847, 1.1233,
               1.4820, 1.6276, 1.8032, 2.1486, 2.3351, 2.4918, 2.6710]
DETECTIONS = [0.01, 0.085, 0.275, 0.445, 0.61, 0.795, 0.98, 1.115, 1.365,
              1.475, 1.62, 1.795, 2.14, 2.33, 2.485, 2.665]


class TestSearchFilesFunction(unittest.TestCase):
    # tests for files()
    def test_files_without_suffix(self):
        all_files = search_files(DATA_PATH)
        file_list = [DATA_PATH + 'README',
                     DATA_PATH + 'commented_txt',
                     DATA_PATH + 'events.txt',
                     DATA_PATH + 'sample.beats',
                     DATA_PATH + 'sample.onsets',
                     DATA_PATH + 'sample.onsets.txt',
                     DATA_PATH + 'sample.sv',
                     DATA_PATH + 'sample.tempo',
                     DATA_PATH + 'sample.wav',
                     DATA_PATH + 'stereo_sample.mid',
                     DATA_PATH + 'stereo_sample.notes',
                     DATA_PATH + 'stereo_sample.sv',
                     DATA_PATH + 'stereo_sample.flac',
                     DATA_PATH + 'stereo_sample.wav']
        self.assertEqual(all_files, sorted(file_list))

    def test_txt_files(self):
        txt_files = search_files(DATA_PATH, suffix='txt')
        file_list = [DATA_PATH + 'commented_txt',
                     DATA_PATH + 'events.txt',
                     DATA_PATH + 'sample.onsets.txt']
        self.assertEqual(txt_files, sorted(file_list))

    def test_dot_txt_files(self):
        dot_txt_files = search_files(DATA_PATH, suffix='.txt')
        file_list = [DATA_PATH + 'events.txt',
                     DATA_PATH + 'sample.onsets.txt']
        self.assertEqual(dot_txt_files, sorted(file_list))


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
        file_handle = __builtin__.open(DATA_PATH + 'events.txt')
        events = load_events(file_handle)
        self.assertIsInstance(events, np.ndarray)
        file_handle.close()

    def test_read_onset_annotations(self):
        events = load_events(DATA_PATH + 'sample.onsets')
        self.assertTrue(np.array_equal(events, ANNOTATIONS))

    def test_read_file_without_comments(self):
        events = load_events(DATA_PATH + 'sample.onsets.txt')
        self.assertTrue(events.any())

    def test_load_file_with_comments_and_empty_lines(self):
        events = load_events(DATA_PATH + 'commented_txt')
        self.assertTrue(events.any())


class TestWriteEventsFunction(unittest.TestCase):

    def test_write_events_to_file(self):
        result = write_events(EVENTS, DATA_PATH + 'events.txt')
        self.assertEqual(EVENTS, result)

    def test_write_events_to_file_handle(self):
        file_handle = __builtin__.open(DATA_PATH + 'events.txt', 'w')
        result = write_events(EVENTS, file_handle)
        self.assertEqual(EVENTS, result)
        file_handle.close()

    def test_write_and_read_events(self):
        write_events(EVENTS, DATA_PATH + 'events.txt')
        annotations = load_events(DATA_PATH + 'events.txt')
        self.assertTrue(np.array_equal(annotations, EVENTS))


class TestCombineEventsFunction(unittest.TestCase):

    def test_combine_000(self):
        comb = combine_events(EVENTS, 0.)
        correct = np.asarray([1, 1.02, 1.5, 2.0, 2.03, 2.05, 2.5, 3])
        self.assertTrue(np.array_equal(comb, correct))

    def test_combine_001(self):
        comb = combine_events(EVENTS, 0.01)
        correct = np.asarray([1, 1.02, 1.5, 2.0, 2.03, 2.05, 2.5, 3])
        self.assertTrue(np.array_equal(comb, correct))

    def test_combine_003(self):
        comb = combine_events(EVENTS, 0.03)
        correct = np.asarray([1.01, 1.5, 2.015, 2.05, 2.5, 3])
        self.assertTrue(np.allclose(comb, correct))

    def test_combine_0035(self):
        comb = combine_events(EVENTS, 0.035)
        correct = np.asarray([1.01, 1.5, 2.0325, 2.5, 3])
        self.assertTrue(np.allclose(comb, correct))


class TestQuantizeEventsFunction(unittest.TestCase):

    def test_quantize_10(self):
        quantized = quantize_events(EVENTS, 10, length=None)
        idx = np.nonzero(quantized)[0]
        # tar: [1, 1.02, 1.5, 2.0, 2.03, 2.05, 2.5, 3]
        correct = [10, 15, 20, 21, 25, 30]
        self.assertTrue(np.array_equal(idx, correct))

    def test_quantize_100(self):
        quantized = quantize_events(EVENTS, 100, length=None)
        idx = np.nonzero(quantized)[0]
        # tar: [1, 1.02, 1.5, 2.0, 2.03, 2.05, 2.5, 3]
        correct = [100, 102, 150, 200, 203, 205, 250, 300]
        self.assertTrue(np.array_equal(idx, correct))

    def test_quantize_length_280(self):
        length = 280
        quantized = quantize_events(EVENTS, 100, length=length)
        self.assertTrue(len(quantized), length)

    def test_quantize_100_length_280(self):
        quantized = quantize_events(EVENTS, 100, length=280)
        idx = np.nonzero(quantized)[0]
        # tar: [1, 1.02, 1.5, 2.0, 2.03, 2.05, 2.5, 3]
        correct = [100, 102, 150, 200, 203, 205, 250]
        self.assertTrue(np.array_equal(idx, correct))

    def test_quantize_rounding_395(self):
        quantized = quantize_events([3.95], 10, length=None)
        idx = np.nonzero(quantized)[0]
        correct = [40]
        self.assertTrue(np.array_equal(idx, correct))

    def test_quantize_rounding_395_length_40(self):
        quantized = quantize_events([3.95], 10, length=40)
        idx = np.nonzero(quantized)[0]
        correct = []
        self.assertTrue(np.array_equal(idx, correct))

    def test_quantize_rounding_39499999_length_40(self):
        quantized = quantize_events([3.9499999], 10, length=40)
        idx = np.nonzero(quantized)[0]
        correct = [39]
        self.assertTrue(np.array_equal(idx, correct))

    def test_quantize_rounding_394(self):
        quantized = quantize_events([3.949999999], 10, length=None)
        idx = np.nonzero(quantized)[0]
        correct = [39]
        self.assertTrue(np.array_equal(idx, correct))
