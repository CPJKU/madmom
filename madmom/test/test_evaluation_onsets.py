# encoding: utf-8
"""
This file contains tests for the madmom.evaluation.onsets module.

"""
# pylint: skip-file

import unittest

from madmom.evaluation import SimpleEvaluation
from madmom.evaluation.onsets import *

DETECTIONS = np.asarray([0.99999999, 1.02999999, 1.45, 2.01, 2.02, 2.5,
                         3.030000001])
ANNOTATIONS = np.asarray([1, 1.02, 1.5, 2.0, 2.03, 2.05, 2.5, 3])


# test evaluation function
class TestOnsetEvaluationFunction(unittest.TestCase):

    def test_window_001(self):
        tp, fp, tn, fn = onset_evaluation(DETECTIONS, ANNOTATIONS, 0.01)
        self.assertEqual(tp, [0.99999999, 1.02999999, 2.01, 2.02, 2.5])
        self.assertEqual(fp, [1.45, 3.030000001])
        self.assertEqual(tn, [])
        self.assertEqual(fn, [1.5, 2.05, 3.0])

    def test_window_003(self):
        tp, fp, tn, fn = onset_evaluation(DETECTIONS, ANNOTATIONS, 0.03)
        self.assertEqual(tp, [0.99999999, 1.02999999, 2.01, 2.02, 2.5])
        self.assertEqual(fp, [1.45, 3.030000001])
        self.assertEqual(tn, [])
        self.assertEqual(fn, [1.5, 2.05, 3.0])

    def test_window_004(self):
        tp, fp, tn, fn = onset_evaluation(DETECTIONS, ANNOTATIONS, 0.04)

        self.assertEqual(tp, [0.99999999, 1.02999999, 2.01, 2.02, 2.5,
                              3.030000001])
        self.assertEqual(fp, [1.45])
        self.assertEqual(tn, [])
        self.assertEqual(fn, [1.5, 2.05])


# test evaluation class
class TestOnsetEvaluationClass(unittest.TestCase):

    def test_types(self):
        e = OnsetEvaluation(DETECTIONS, ANNOTATIONS)
        self.assertIsInstance(e.num_tp, int)
        self.assertIsInstance(e.num_fp, int)
        self.assertIsInstance(e.num_tn, int)
        self.assertIsInstance(e.num_fn, int)
        self.assertIsInstance(e.precision, float)
        self.assertIsInstance(e.recall, float)
        self.assertIsInstance(e.fmeasure, float)
        self.assertIsInstance(e.accuracy, float)
        self.assertIsInstance(e.errors, list)
        self.assertIsInstance(e.mean_error, float)
        self.assertIsInstance(e.std_error, float)

    def test_conversion(self):
        # conversion from list should work
        e = OnsetEvaluation([0], [0])
        self.assertIsInstance(e.tp, list)
        self.assertIsInstance(e.fp, list)
        self.assertIsInstance(e.tn, list)
        self.assertIsInstance(e.fn, list)
        # conversion from dict should work as well
        e = OnsetEvaluation({}, {})
        self.assertIsInstance(e.tp, list)
        self.assertIsInstance(e.fp, list)
        self.assertIsInstance(e.tn, list)
        self.assertIsInstance(e.fn, list)
        # others should fail
        self.assertRaises(TypeError, OnsetEvaluation, float(0), float(0))
        self.assertRaises(TypeError, OnsetEvaluation, int(0), int(0))

    def test_add(self):
        e = OnsetEvaluation(DETECTIONS, ANNOTATIONS)
        # adding an Evaluation or OnsetEvaluation object should work
        self.assertIsInstance(e + Evaluation(), Evaluation)
        self.assertIsInstance(e + OnsetEvaluation(DETECTIONS, ANNOTATIONS),
                              Evaluation)
        # adding others should fail
        with self.assertRaises(TypeError):
            e + SimpleEvaluation()
        with self.assertRaises(TypeError):
            e + SumEvaluation()
        with self.assertRaises(TypeError):
            e + MeanEvaluation()

    def test_iadd(self):
        e = OnsetEvaluation(DETECTIONS, ANNOTATIONS)
        # adding an Evaluation
        e += Evaluation()
        self.assertIsInstance(e, OnsetEvaluation)
        # or OnsetEvaluation object should work
        e += OnsetEvaluation(DETECTIONS, ANNOTATIONS)
        self.assertIsInstance(e, OnsetEvaluation)
        # adding others should fail
        with self.assertRaises(TypeError):
            e += SimpleEvaluation()
        with self.assertRaises(TypeError):
            e += SumEvaluation()
        with self.assertRaises(TypeError):
            e += MeanEvaluation()

    def test_iadd_types(self):
        e = OnsetEvaluation(DETECTIONS, ANNOTATIONS)
        e += OnsetEvaluation(DETECTIONS, ANNOTATIONS)
        self.assertIsInstance(e.num_tp, int)
        self.assertIsInstance(e.num_fp, int)
        self.assertIsInstance(e.num_tn, int)
        self.assertIsInstance(e.num_fn, int)
        self.assertIsInstance(e.precision, float)
        self.assertIsInstance(e.recall, float)
        self.assertIsInstance(e.fmeasure, float)
        self.assertIsInstance(e.accuracy, float)
        self.assertIsInstance(e.errors, list)
        self.assertIsInstance(e.mean_error, float)
        self.assertIsInstance(e.std_error, float)

    def test_results(self):
        e = OnsetEvaluation(DETECTIONS, ANNOTATIONS)
        self.assertEqual(e.tp, [0.99999999, 1.02999999, 2.01, 2.02,
                                2.5])
        self.assertEqual(e.fp, [1.45, 3.030000001])
        self.assertEqual(e.tn, [])
        self.assertEqual(e.fn, [1.5, 2.05, 3.0])
        self.assertEqual(e.num_tp, 5)
        self.assertEqual(e.num_fp, 2)
        self.assertEqual(e.num_tn, 0)
        self.assertEqual(e.num_fn, 3)
        # p = correct / retrieved
        self.assertEqual(e.precision, 5. / 7.)
        # r = correct / relevant
        self.assertEqual(e.recall, 5. / 8.)
        # f = 2 * P * R / (P + R)
        f = 2 * (5. / 7.) * (5. / 8.) / ((5. / 7.) + (5. / 8.))
        self.assertEqual(e.fmeasure, f)
        # acc = (TP + TN) / (TP + FP + TN + FN)
        self.assertEqual(e.accuracy, (5. + 0) / (5 + 2 + 0 + 3))
        # errors
        # det 0.99999999, 1.02999999, 1.45, 2.01, 2.02,       2.5, 3.030000001
        # tar 1,          1.02,       1.5,  2.0,  2.03, 2.05, 2.5, 3
        errors = [0.99999999 - 1, 1.02999999 - 1.02,  # 1.45 - 1.5,
                  2.01 - 2, 2.02 - 2.03, 2.5 - 2.5]  # , 3.030000001 - 3
        self.assertTrue(np.array_equal(e.errors, errors))
        mean = np.mean([0.99999999 - 1, 1.02999999 - 1.02, 2.01 - 2,
                        2.02 - 2.03, 2.5 - 2.5])
        self.assertEqual(e.mean_error, mean)
        std = np.std([0.99999999 - 1, 1.02999999 - 1.02, 2.01 - 2, 2.02 - 2.03,
                      2.5 - 2.5])
        self.assertEqual(e.std_error, std)
