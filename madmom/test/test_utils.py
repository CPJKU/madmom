# encoding: utf-8
"""
This file contains test functions for the madmom.utils.helpers module.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""
#pylint: skip-file

import unittest
import __builtin__

from madmom.test import DATA_PATH
from madmom.utils import *


ANNOTATIONS = [1, 1.02, 1.5, 2.0, 2.03, 2.05, 2.5, 3]
DETECTIONS = [0.99999999, 1.02999999, 1.45, 2.01, 2.02, 2.5, 3.030000001]


class TestFileSelection(unittest.TestCase):
    # tests for files()
    def test_files_without_suffix(self):
        all_files = files(DATA_PATH)
        self.assertEqual(all_files, [DATA_PATH + 'commented_file.txt',
                                     DATA_PATH + 'file.onsets',
                                     DATA_PATH + 'file.onsets.txt',
                                     DATA_PATH + 'file.tempo',
                                     DATA_PATH + 'file_txt'])

    def test_txt_files(self):
        txt_files = files(DATA_PATH, suffix='txt')
        self.assertEqual(txt_files, [DATA_PATH + 'commented_file.txt',
                                     DATA_PATH + 'file.onsets.txt',
                                     DATA_PATH + 'file_txt'])

    def test_dot_txt_files(self):
        dot_txt_files = files(DATA_PATH, suffix='.txt')
        self.assertEqual(dot_txt_files, [DATA_PATH + 'commented_file.txt',
                                         DATA_PATH + 'file.onsets.txt'])


class TestFileMatching(unittest.TestCase):
    # tests for strip_suffix(filename, ext=None)
    def test_strip_txt_suffix(self):
        self.assertEqual(strip_suffix('file.txt', 'txt'), 'file.')
        self.assertEqual(strip_suffix('/path/file.txt', 'txt'), '/path/file.')

    def test_strip_dot_txt_suffix(self):
        self.assertEqual(strip_suffix('file.txt', '.txt'), 'file')
        self.assertEqual(strip_suffix('/path/file.txt', '.txt'), '/path/file')

    # test for match_file(filename, match_list, ext=None, match_suffix=None)
    def test_match_dot_txt_suffix(self):
        match_list = ['file.txt', '/path/file.txt', '/path/file.txt.other']
        self.assertEqual(match_file('file.txt', match_list),
                         ['file.txt', '/path/file.txt'])

    def test_match_other_suffix(self):
        match_list = ['file.txt', '/path/file.txt', '/path/file.txt.other']
        self.assertEqual(match_file('file.txt', match_list,
                                    match_suffix='other'), [])

    def test_match_dot_other_suffix(self):
        match_list = ['file.txt', '/path/file.txt', '/path/file.txt.other']
        self.assertEqual(match_file('file.txt', match_list,
                                    match_suffix='.other'),
                         ['/path/file.txt.other'])


class TestReadFiles(unittest.TestCase):

    def test_read_events_from_closed_file_handle(self):
        file_handle = __builtin__.open(DATA_PATH + 'file_txt', 'w')
        self.assertRaises(IOError, load_events, file_handle)

    def test_read_events_from_file(self):
        annotations = load_events(DATA_PATH + 'file_txt')
        self.assertIsInstance(annotations, np.ndarray)

    def test_read_events_from_file_handle(self):
        file_handle = __builtin__.open(DATA_PATH + 'file_txt', 'r')
        annotations = load_events(file_handle)
        self.assertIsInstance(annotations, np.ndarray)
        file_handle.close()

    def test_read_onset_annotations(self):
        annotations = load_events(DATA_PATH + 'file.onsets')
        self.assertTrue(np.array_equal(annotations, ANNOTATIONS))

    def test_read_onset_detections(self):
        detections = load_events(DATA_PATH + 'file.onsets.txt')
        self.assertTrue(np.array_equal(detections, DETECTIONS))

    def test_read_file_without_comments(self):
        events = load_events(DATA_PATH + 'file.onsets.txt')
        self.assertTrue(events.any())

    def test_load_file_with_comments(self):
        events = load_events(DATA_PATH + 'commented_file.txt')
        self.assertTrue(events.any())


class TestWriteFiles(unittest.TestCase):

    def test_write_events_to_closed_file_handle(self):
        file_handle = __builtin__.open(DATA_PATH + 'file_txt', 'r')
        self.assertRaises(IOError, write_events, ANNOTATIONS, file_handle)

    def test_write_events_to_file(self):
        self.assertIsNone(write_events(ANNOTATIONS, DATA_PATH + 'file_txt'))

    def test_write_events_to_file_handle(self):
        file_handle = __builtin__.open(DATA_PATH + 'file_txt', 'w')
        self.assertIsNone(write_events(ANNOTATIONS, file_handle))
        file_handle.close()

    def test_write_and_read_events(self):
        write_events(ANNOTATIONS, DATA_PATH + 'file_txt')
        annotations = load_events(DATA_PATH + 'file_txt')
        self.assertTrue(np.array_equal(annotations, ANNOTATIONS))


class TestCombineEvents(unittest.TestCase):

    def test_combine_000(self):
        comb = combine_events(ANNOTATIONS, 0.)
        correct = np.asarray([1, 1.02, 1.5, 2.0, 2.03, 2.05, 2.5, 3])
        self.assertTrue(np.array_equal(comb, correct))

    def test_combine_001(self):
        comb = combine_events(ANNOTATIONS, 0.01)
        correct = np.asarray([1, 1.02, 1.5, 2.0, 2.03, 2.05, 2.5, 3])
        self.assertTrue(np.array_equal(comb, correct))

    def test_combine_003(self):
        comb = combine_events(ANNOTATIONS, 0.03)
        correct = np.asarray([1.01, 1.5, 2.015, 2.05, 2.5, 3])
        self.assertTrue(np.allclose(comb, correct))

    def test_combine_0035(self):
        comb = combine_events(ANNOTATIONS, 0.035)
        correct = np.asarray([1.01, 1.5, 2.0325, 2.5, 3])
        self.assertTrue(np.allclose(comb, correct))


class TestQuantizeEvents(unittest.TestCase):

    def test_quantize_10(self):
        quantized = quantize_events(ANNOTATIONS, 10, length=None)
        idx = np.nonzero(quantized)[0]
        # tar: [1, 1.02, 1.5, 2.0, 2.03, 2.05, 2.5, 3]
        correct = [10, 15, 20, 21, 25, 30]
        self.assertTrue(np.array_equal(idx, correct))

    def test_quantize_100(self):
        quantized = quantize_events(ANNOTATIONS, 100, length=None)
        idx = np.nonzero(quantized)[0]
        # tar: [1, 1.02, 1.5, 2.0, 2.03, 2.05, 2.5, 3]
        correct = [100, 102, 150, 200, 203, 205, 250, 300]
        self.assertTrue(np.array_equal(idx, correct))

    def test_quantize_length_280(self):
        length = 280
        quantized = quantize_events(ANNOTATIONS, 100, length=length)
        self.assertTrue(len(quantized), length)

    def test_quantize_100_length_280(self):
        quantized = quantize_events(ANNOTATIONS, 100, length=280)
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
