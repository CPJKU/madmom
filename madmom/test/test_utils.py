# encoding: utf-8
"""
This file contains test functions for the madmom.utils.helpers module.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import unittest
import __builtin__

from madmom.test import DATA_PATH
from madmom.utils import *


ONSET_TARGETS = [1, 1.02, 1.5, 2.0, 2.03, 2.05, 2.5, 3]
ONSET_DETECTIONS = [0.99999999, 1.02999999, 1.45, 2.01, 2.02, 2.5, 3.030000001]


class TestFileSelection(unittest.TestCase):
    # tests for files()
    def test_files_without_extension(self):
        all_files = files(DATA_PATH)
        self.assertEqual(all_files, [DATA_PATH + 'commented_file.txt',
                                     DATA_PATH + 'file.onsets',
                                     DATA_PATH + 'file.onsets.txt',
                                     DATA_PATH + 'file_txt'])

    def test_txt_files(self):
        txt_files = files(DATA_PATH, ext='txt')
        self.assertEqual(txt_files, [DATA_PATH + 'commented_file.txt',
                                     DATA_PATH + 'file.onsets.txt',
                                     DATA_PATH + 'file_txt'])

    def test_dot_txt_files(self):
        dot_txt_files = files(DATA_PATH, ext='.txt')
        self.assertEqual(dot_txt_files, [DATA_PATH + 'commented_file.txt',
                                         DATA_PATH + 'file.onsets.txt'])

    def test_load_file_with_comments(self):
        self.assertRaises(ValueError, load_events,
                          (DATA_PATH + 'commented_file.txt'))


class TestFileMatching(unittest.TestCase):
    # tests for strip_ext(filename, ext=None)
    def test_strip_txt_ext(self):
        self.assertEqual(strip_ext('file.txt', 'txt'), 'file.')
        self.assertEqual(strip_ext('/path/file.txt', 'txt'), '/path/file.')

    def test_strip_dot_txt_ext(self):
        self.assertEqual(strip_ext('file.txt', '.txt'), 'file')
        self.assertEqual(strip_ext('/path/file.txt', '.txt'), '/path/file')

    # test for match_file(filename, match_list, ext=None, match_ext=None)
    def test_match_dot_txt_ext(self):
        match_list = ['file.txt', '/path/file.txt', '/path/file.txt.other']
        self.assertEqual(match_file('file.txt', match_list),
                         ['file.txt', '/path/file.txt'])

    def test_match_other_ext(self):
        match_list = ['file.txt', '/path/file.txt', '/path/file.txt.other']
        self.assertEqual(match_file('file.txt', match_list, match_ext='other'),
                         [])

    def test_match_dot_other_ext(self):
        match_list = ['file.txt', '/path/file.txt', '/path/file.txt.other']
        self.assertEqual(match_file('file.txt', match_list,
                                    match_ext='.other'),
                         ['/path/file.txt.other'])


class TestFileHandling(unittest.TestCase):
    # test file handle handling
    def test_write_events_to_closed_file_handle(self):
        file_handle = __builtin__.open(DATA_PATH + 'file_txt', 'r')
        self.assertRaises(IOError, write_events, ONSET_TARGETS, file_handle)

    def test_read_events_from_closed_file_handle(self):
        file_handle = __builtin__.open(DATA_PATH + 'file_txt', 'w')
        self.assertRaises(IOError, load_events, file_handle)

    def test_write_events_to_file(self):
        self.assertIsNone(write_events(ONSET_TARGETS, DATA_PATH + 'file_txt'))

    def test_load_events_from_file(self):
        targets = load_events(DATA_PATH + 'file_txt')
        self.assertIsInstance(targets, np.ndarray)

    def test_write_events_to_file_handle(self):
        file_handle = __builtin__.open(DATA_PATH + 'file_txt', 'w')
        self.assertIsNone(write_events(ONSET_TARGETS, file_handle))
        file_handle.close()

    def test_load_events_from_file_handle(self):
        file_handle = __builtin__.open(DATA_PATH + 'file_txt', 'r')
        targets = load_events(file_handle)
        self.assertIsInstance(targets, np.ndarray)
        file_handle.close()


class TestFileValues(unittest.TestCase):
    # test correct value writing / loading
    def test_write_and_load_events(self):
        write_events(ONSET_TARGETS, DATA_PATH + 'file_txt')
        targets = load_events(DATA_PATH + 'file_txt')
        self.assertTrue(np.array_equal(targets, ONSET_TARGETS))

    def test_load_file_without_comments(self):
        events = load_events(DATA_PATH + 'file.onsets.txt')
        self.assertTrue(events.any())

    def test_load_onset_targets(self):
        targets = load_events(DATA_PATH + 'file.onsets')
        self.assertTrue(np.array_equal(targets, ONSET_TARGETS))

    def test_load_onset_detections(self):
        detections = load_events(DATA_PATH + 'file.onsets.txt')
        self.assertTrue(np.array_equal(detections, ONSET_DETECTIONS))

    # TODO: write a test for speed
    # def test_speed_loading_files(self):
    #     """load_events() function is much faster than e.g. np.fromtxt()"""

