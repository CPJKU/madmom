# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the /bin/evaluate script.

"""

from __future__ import absolute_import, division, print_function

import os
import sys
import types
import unittest
from importlib.machinery import SourceFileLoader

try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

import numpy as np

from . import ANNOTATIONS_PATH, DETECTIONS_PATH
from madmom.evaluation.beats import BeatIntervalError

eval_script = os.path.dirname(os.path.realpath(__file__)) + '/../bin/evaluate'

# prevent writing compiled Python files to disk
sys.dont_write_bytecode = True


def run_script(task, det_suffix=None, args=None):
    # import module, capture stdout
    loader = SourceFileLoader('test', eval_script)
    test = types.ModuleType(loader.name)
    loader.exec_module(test)
    sys.argv = [eval_script, task, '--csv', DETECTIONS_PATH, ANNOTATIONS_PATH]
    if det_suffix:
        sys.argv.extend(['-d', det_suffix])
    if args:
        sys.argv.extend(args)
    backup = sys.stdout
    sys.stdout = StringIO()
    # run evaluation script
    test.main()
    # get data from stdout, restore environment
    data = sys.stdout.getvalue()
    sys.stdout.close()
    sys.stdout = backup
    return data.splitlines()


class TestEvaluateScript(unittest.TestCase):

    def test_onsets(self):
        res = run_script('onsets', det_suffix='.super_flux.txt')
        # second line contains the summed results
        sum_res = np.fromiter(res[1].split(',')[1:], dtype=float)
        self.assertTrue(np.allclose(
            sum_res, [14, 2, 0, 1, 15, 0.875, 0.933, 0.903, 0.824]))
        # third line contains the mean results
        mean_res = np.fromiter(res[2].split(',')[1:], dtype=float)
        self.assertTrue(np.allclose(mean_res, sum_res))

    def test_beats(self):
        res = run_script('beats', det_suffix='.beat_detector.txt')
        # second line contains the results
        res = np.fromiter(res[1].split(',')[1:], dtype=float)
        self.assertTrue(np.allclose(res, [0.667, 0.5, 0.639, 1, 0, 0,
                                          0.875, 0.875, 3.322, 3.322]))

    def test_downbeats(self):
        with self.assertRaises(BeatIntervalError):
            run_script('beats', det_suffix='.dbn_downbeat_tracker.txt',
                       args=['--down'])

    def test_chords(self):
        res = run_script('chords')
        # second line contains the weighted mean results
        weighted = np.fromiter(res[1].split(',')[1:], dtype=float)
        self.assertTrue(np.allclose(weighted, [0.897, 0.641, 0.462, 0.462,
                                               0.282, 0.590, 0.590, 0.923]))
        # third line contains the piecewise mean results
        piecewise = np.fromiter(res[2].split(',')[1:], dtype=float)
        self.assertTrue(np.allclose(weighted, piecewise))

    def test_key(self):
        res = run_script('key')
        # second line contains the results
        res = np.fromiter(res[1].split(',')[1:], dtype=float)
        self.assertTrue(np.allclose(res, [0, 0, 1, 0, 0, 0.3]))

    def test_notes(self):
        res = run_script('notes', det_suffix='.piano_transcriptor.txt',
                         args=['-w', '0.05'])
        # second line contains the summed results
        sum_res = np.fromiter(res[1].split(',')[1:], dtype=float)
        self.assertTrue(
            np.allclose(sum_res, [8, 0, 0, 0, 8, 1, 1, 1, 1]))
        # third line contains the mean results
        mean_res = np.fromiter(res[2].split(',')[1:], dtype=float)
        self.assertTrue(np.allclose(mean_res, sum_res))

    def test_tempo(self):
        res = run_script('tempo', det_suffix='.tempo_detector.txt',
                         args=['-a', '.tempo'])
        # second line contains the results
        res = np.fromiter(res[1].split(',')[1:], dtype=float)
        self.assertTrue(
            np.allclose(res, [0.3, 1, 0, 0, 1]))
