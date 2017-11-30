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
from os.path import join as pj

from madmom.io import *
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


class TestWriteEventsFunction(unittest.TestCase):

    def test_write_events_to_file(self):
        result = write_events(EVENTS, pj(DATA_PATH, 'events.txt'))
        self.assertEqual(EVENTS, result)

    def test_write_events_to_file_handle(self):
        file_handle = open(pj(DATA_PATH, 'events.txt'), 'wb')
        result = write_events(EVENTS, file_handle)
        self.assertEqual(EVENTS, result)
        file_handle.close()

    def test_write_and_read_events(self):
        write_events(EVENTS, pj(DATA_PATH, 'events.txt'))
        annotations = load_events(pj(DATA_PATH, 'events.txt'))
        self.assertTrue(np.allclose(annotations, EVENTS))
