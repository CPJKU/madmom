# encoding: utf-8
"""
This file contains beat evaluation tests.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import unittest

from madmom.evaluation.beats import *

TARGETS = np.asarray([1., 2, 3, 4, 5, 6, 7, 8, 9, 10])
DETECTIONS = np.asarray([1.01, 2, 2.95, 4, 6, 7, 8, 9.1, 10, 11])


# test functions
class TestCalcInterval(unittest.TestCase):

    def test_tar_bwd(self):
        intervals = calc_intervals(TARGETS)
        correct = np.asarray([1., 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.assertTrue(np.allclose(intervals, correct))
        # TODO: test with matches given

    def test_det_bwd(self):
        intervals = calc_intervals(DETECTIONS)
        correct = [0.99, 0.99, 0.95, 1.05, 2, 1, 1, 1.1, 0.9, 1]
        self.assertTrue(np.allclose(intervals, correct))
        # TODO: test with matches given

    def test_tar_fwd(self):
        intervals = calc_intervals(TARGETS, fwd=True)
        correct = np.asarray([1., 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.assertTrue(np.allclose(intervals, correct))
        # TODO: test with matches given

    def test_det_fwd(self):
        intervals = calc_intervals(DETECTIONS, fwd=True)
        correct = [0.99, 0.95, 1.05, 2, 1, 1, 1.1, 0.9, 1, 1]
        self.assertTrue(np.allclose(intervals, correct))
        # TODO: test with matches given


class TestFindClosestInterval(unittest.TestCase):

    def test_det_tar(self):
        intervals = find_closest_intervals(DETECTIONS, TARGETS)
        correct = np.asarray([1., 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.assertTrue(np.allclose(intervals, correct))
        # TODO: test with matches given
        # intervals = find_closest_intervals(DETECTIONS, TARGETS)
        # correct = np.asarray([1., 1, 1, 1, 1, 1, 1, 1, 1, 1])

    def test_tar_det(self):
        intervals = find_closest_intervals(TARGETS, DETECTIONS)
        correct = [0.99, 0.99, 1.05, 1.05, 2, 2, 1, 1, 1.1, 0.9]
        self.assertTrue(np.allclose(intervals, correct))
        # TODO: test with matches given


class TestCalcRelativeErrorsFunction(unittest.TestCase):

    def test_det_tar(self):
        rel_errors = calc_relative_errors(DETECTIONS, TARGETS)
        # det: [1.01, 2, 2.95, 4,    6, 7, 8, 9.1, 10, 11]
        # tar: [1,    2, 3,    4, 5, 6, 7, 8, 9,   10]
        correct = [0.01, 0, -0.05, 0, 0, 0, 0, 0.1, 0, 1]
        # all intervals are 1, so need for division
        self.assertTrue(np.allclose(rel_errors, correct))
        # TODO: test with matches given

    def test_tar_det(self):
        rel_errors = calc_relative_errors(TARGETS, DETECTIONS)
        # tar: [1,    2, 3,    4, 5, 6, 7, 8, 9,   10]
        # det: [1.01, 2, 2.95, 4,    6, 7, 8, 9.1, 10, 11]
        errors = np.asarray([-0.01, 0, 0.05, 0, -1, 0, 0, 0, -0.1, 0])
        intervals = np.asarray([0.99, 0.99, 1.05, 1.05, 2, 2, 1, 1, 1.1, 0.9])
        self.assertTrue(np.allclose(rel_errors, errors / intervals))
        # TODO: test with matches given


class TestPscoreFunction(unittest.TestCase):

    def test_types(self):
        score = pscore(DETECTIONS, TARGETS, 0.2)
        self.assertIsInstance(score, float)
        score = pscore([], [], 0.1)
        self.assertIsInstance(score, float)
        # score = pscore(DETECTIONS, TARGETS)
        # self.assertIsInstance(score, float)
        # score = pscore(DETECTIONS, TARGETS)
        # self.assertIsInstance(score, float)
        # score = pscore(DETECTIONS, TARGETS)
        # self.assertIsInstance(score, float)
        # all arguments must be given
        with self.assertRaises(TypeError):
            pscore(DETECTIONS, TARGETS, None)
        with self.assertRaises(TypeError):
            pscore(DETECTIONS, None, 0.2)
        with self.assertRaises(TypeError):
            pscore(None, TARGETS, 0.2)
        # tolerance must be correct type
        with self.assertRaises(ValueError):
            pscore(DETECTIONS, TARGETS, [])
        with self.assertRaises(TypeError):
            pscore(DETECTIONS, TARGETS, None)
        with self.assertRaises(TypeError):
            pscore(DETECTIONS, TARGETS, {})

    def test_values(self):
        # empty lists should return 0
        score = pscore([], [], 0.2)
        self.assertEqual(score, 0)
        # score relies on intervals, hence at least 2 targets must be given
        score = pscore(DETECTIONS, [1], 0.2)
        self.assertEqual(score, 0)
        # no detections should return 0
        score = pscore([], TARGETS, 0.2)
        self.assertEqual(score, 0)
        # normal calculation
        score = pscore(DETECTIONS, TARGETS, 0.2)
        self.assertEqual(score, 0.9)


# test evaluation class
class TestBeatEvaluationClass(unittest.TestCase):

    def test_types(self):
        e = BeatEvaluation(DETECTIONS, TARGETS)
        # from OnsetEvaluation
        self.assertIsInstance(e.num_tp, int)
        self.assertIsInstance(e.num_fp, int)
        self.assertIsInstance(e.num_tn, int)
        self.assertIsInstance(e.num_fn, int)
        self.assertIsInstance(e.precision, float)
        self.assertIsInstance(e.recall, float)
        self.assertIsInstance(e.fmeasure, float)
        self.assertIsInstance(e.accuracy, float)
        self.assertIsInstance(e.errors, np.ndarray)
        self.assertIsInstance(e.mean_error, float)
        self.assertIsInstance(e.std_error, float)
        # additional beat score types
        self.assertIsInstance(e.pscore, float)
        self.assertIsInstance(e.cemgil, float)
        self.assertIsInstance(e.cmlc, float)
        self.assertIsInstance(e.cmlt, float)
        self.assertIsInstance(e.amlc, float)
        self.assertIsInstance(e.amlt, float)
        self.assertIsInstance(e.information_gain, float)
        self.assertIsInstance(e.global_information_gain, float)
        self.assertIsInstance(e.error_histogram, np.ndarray)

    def test_conversion(self):
        # conversion from list should work
        e = BeatEvaluation([], [])
        self.assertIsInstance(e.tp, np.ndarray)
        self.assertIsInstance(e.fp, np.ndarray)
        self.assertIsInstance(e.tn, np.ndarray)
        self.assertIsInstance(e.fn, np.ndarray)
        # conversion from dict should work as well
        e = BeatEvaluation({}, {})
        self.assertIsInstance(e.tp, np.ndarray)
        self.assertIsInstance(e.fp, np.ndarray)
        self.assertIsInstance(e.tn, np.ndarray)
        self.assertIsInstance(e.fn, np.ndarray)
        # others should fail
        self.assertRaises(TypeError, BeatEvaluation, float(0), float(0))
        self.assertRaises(TypeError, BeatEvaluation, int(0), int(0))

    def test_results(self):
        e = BeatEvaluation(DETECTIONS, TARGETS)
        # tar: [1,    2, 3,    4, 5, 6, 7, 8, 9,   10]
        # det: [1.01, 2, 2.95, 4,    6, 7, 8, 9.1, 10, 11]
        # WINDOW = 0.07
        # TOLERANCE = 0.2
        # SIGMA = 0.04
        # TEMPO_TOLERANCE = 0.175
        # PHASE_TOLERANCE = 0.175
        # BINS = 40
        self.assertEqual(e.tp.tolist(), [1.01, 2, 2.95, 4, 6, 7, 8, 10])
        self.assertEqual(e.fp.tolist(), [9.1, 11])
        self.assertEqual(e.tn.tolist(), [])
        self.assertEqual(e.fn.tolist(), [5, 9])
        self.assertEqual(e.num_tp, 8)
        self.assertEqual(e.num_fp, 2)
        self.assertEqual(e.num_tn, 0)
        self.assertEqual(e.num_fn, 2)
        self.assertEqual(e.precision, 8. / 10.)
        self.assertEqual(e.recall, 8. / 10.)
        f = 2 * (8. / 10.) * (8. / 10.) / ((8. / 10.) + (8. / 10.))
        self.assertEqual(e.fmeasure, f)
        self.assertEqual(e.accuracy, (8. + 0) / (8 + 2 + 0 + 2))
        # pscore: delta <= tolerance * median(inter beat interval)
        self.assertEqual(e.pscore, 9. / 10.)
        # cemgil:
        # self.assertEqual(e.cemgil, 9. / 10.)
        # self.assertEqual(e.cmlc, 9. / 10.)
        # self.assertEqual(e.cmlt, 9. / 10.)
        # self.assertEqual(e.amlc, 9. / 10.)
        # self.assertEqual(e.amlt, 9. / 10.)
        # self.assertEqual(e.information_gain, 9. / 10.)
        # self.assertEqual(e.global_information_gain, 9. / 10.)
        # self.assertEqual(e.error_histogram, 9. / 10.)
