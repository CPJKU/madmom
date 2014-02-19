# encoding: utf-8
"""
This file contains onset evaluation tests.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import unittest

from madmom.evaluation import *
from madmom.evaluation.onsets import *

DETECTIONS = np.asarray([0.99999999, 1.02999999, 1.45, 2.01, 2.02, 2.5,
                         3.030000001])
TARGETS = np.asarray([1, 1.02, 1.5, 2.0, 2.03, 2.05, 2.5, 3])


# test types
class TestOnsetEvaluationTypes(unittest.TestCase):
    e = OnsetEvaluation(DETECTIONS, TARGETS)

    def test_num_tp_type(self):
        self.assertIsInstance(self.e.num_tp, int)

    def test_num_fp_type(self):
        self.assertIsInstance(self.e.num_fp, int)

    def test_num_tn_type(self):
        self.assertIsInstance(self.e.num_tn, int)

    def test_num_fn_type(self):
        self.assertIsInstance(self.e.num_fn, int)

    def test_precision_type(self):
        self.assertIsInstance(self.e.precision, float)

    def test_recall_type(self):
        self.assertIsInstance(self.e.recall, float)

    def test_fmeasure_type(self):
        self.assertIsInstance(self.e.fmeasure, float)

    def test_accuracy_type(self):
        self.assertIsInstance(self.e.accuracy, float)

    def test_errors_type(self):
        self.assertIsInstance(self.e.errors, np.ndarray)

    def test_mean_error_type(self):
        self.assertIsInstance(self.e.mean_error, float)

    def test_std_error_type(self):
        self.assertIsInstance(self.e.std_error, float)


class TestOnsetEvaluationAddition(unittest.TestCase):
    e = OnsetEvaluation(DETECTIONS, TARGETS)

    # add
    def test_add_evaluation_object(self):
        e = self.e + Evaluation()
        self.assertIsInstance(e, OnsetEvaluation)

    def test_add_onset_evaluation_object(self):
        e = self.e + OnsetEvaluation(DETECTIONS, TARGETS)
        self.assertIsInstance(e, OnsetEvaluation)

    # others should fail
    def test_add_simple_evaluation_object(self):
        with self.assertRaises(TypeError):
            self.e + SimpleEvaluation()

    def test_add_sum_evaluation_object(self):
        with self.assertRaises(TypeError):
            self.e + SumEvaluation()

    def test_add_mean_evaluation_object(self):
        with self.assertRaises(TypeError):
            self.e + MeanEvaluation()

    # iadd
    def test_iadd_evaluation_object(self):
        self.e += Evaluation()
        self.assertIsInstance(self.e, OnsetEvaluation)

    def test_iadd_onset_evaluation_object(self):
        self.e += OnsetEvaluation(DETECTIONS, TARGETS)
        self.assertIsInstance(self.e, OnsetEvaluation)

    # others should fail
    def test_iadd_simple_evaluation_object(self):
        with self.assertRaises(TypeError):
            self.e += SimpleEvaluation()

    def test_iadd_sum_evaluation_object(self):
        with self.assertRaises(TypeError):
            self.e += SumEvaluation()

    def test_iadd_mean_evaluation_object(self):
        with self.assertRaises(TypeError):
            self.e += MeanEvaluation()



# test results with 0.01 seconds detection window
class TestOnsetEvaluationResults001(unittest.TestCase):
    oe = OnsetEvaluation(DETECTIONS, TARGETS, 0.01)

    def test_tp(self):
        self.assertEqual(self.oe.tp.tolist(), [0.99999999, 1.02999999, 2.01,
                                               2.02, 2.5])

    def test_fp(self):
        self.assertEqual(self.oe.fp.tolist(), [1.45, 3.030000001])

    def test_tn(self):
        self.assertEqual(self.oe.tn.tolist(), [])

    def test_fn(self):
        self.assertEqual(self.oe.fn.tolist(), [1.5, 2.05, 3.0])


# test results with 0.03 seconds detection window
class TestOnsetEvaluationResults003(unittest.TestCase):
    oe = OnsetEvaluation(DETECTIONS, TARGETS, 0.03)

    def test_tp(self):
        self.assertEqual(self.oe.tp.tolist(), [0.99999999, 1.02999999, 2.01,
                                               2.02, 2.5])

    def test_fp(self):
        self.assertEqual(self.oe.fp.tolist(), [1.45, 3.030000001])

    def test_fn(self):
        self.assertEqual(self.oe.fn.tolist(), [1.5, 2.05, 3.0])


# test results with 0.04 seconds detection window
class TestOnsetEvaluationResults004(unittest.TestCase):
    oe = OnsetEvaluation(DETECTIONS, TARGETS, 0.04)

    def test_tp(self):
        self.assertEqual(self.oe.tp.tolist(), [0.99999999, 1.02999999, 2.01,
                                               2.02, 2.5, 3.030000001])

    def test_fp(self):
        self.assertEqual(self.oe.fp.tolist(), [1.45])

    def test_fn(self):
        self.assertEqual(self.oe.fn.tolist(), [1.5, 2.05])

