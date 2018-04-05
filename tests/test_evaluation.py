# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.evaluation module.

"""

from __future__ import absolute_import, division, print_function

import unittest
import math
from collections import OrderedDict

from madmom.evaluation import *


DETECTIONS = np.asarray([0.99, 1.45, 2.01, 2.015, 3.1, 8.1])
ANNOTATIONS = np.asarray([1, 1.5, 2.0, 2.03, 2.05, 2.5, 3])
MATCHES = np.asarray([0, 1, 2, 3, 6, 6])


# test functions
class TestFindClosestMatchesFunction(unittest.TestCase):

    def test_types(self):
        matches = find_closest_matches([], [])
        self.assertIsInstance(matches, np.ndarray)
        self.assertEqual(matches.dtype, np.int)
        self.assertIsInstance(find_closest_matches([], []), np.ndarray)

    def test_value(self):
        # empty sequences
        matches = find_closest_matches([], [])
        self.assertTrue(np.allclose(matches, []))
        # detections relative to annotations
        matches = find_closest_matches(DETECTIONS, ANNOTATIONS)
        self.assertTrue(np.allclose(matches, MATCHES))
        # annotations relative to detections
        matches = find_closest_matches(ANNOTATIONS, DETECTIONS)
        correct = np.asarray([0, 1, 2, 3, 3, 3, 4])
        self.assertTrue(np.allclose(matches, correct))


class TestCalcErrorsFunction(unittest.TestCase):

    def test_types(self):
        errors = calc_errors(DETECTIONS, ANNOTATIONS)
        self.assertIsInstance(errors, np.ndarray)
        self.assertEqual(errors.dtype, np.float)

    def test_values(self):
        # empty sequences
        matches = calc_errors([], [])
        self.assertTrue(np.allclose(matches, []))
        # detections relative to annotations
        errors = calc_errors(DETECTIONS, ANNOTATIONS)
        correct = np.asarray([-0.01, -0.05, 0.01, -0.015, 0.1, 5.1])
        self.assertTrue(np.allclose(errors, correct))
        # same but with matches given
        errors = calc_errors(DETECTIONS, ANNOTATIONS, MATCHES)
        self.assertTrue(np.allclose(errors, correct))
        # annotations relative to detections
        errors = calc_errors(ANNOTATIONS, DETECTIONS)
        correct = np.asarray([0.01, 0.05, -0.01, 0.015, 0.035, 0.485, -0.1])
        self.assertTrue(np.allclose(errors, correct))


class TestCalcAbsoluteErrorsFunction(unittest.TestCase):

    def test_types(self):
        errors = calc_absolute_errors(DETECTIONS, ANNOTATIONS)
        self.assertIsInstance(errors, np.ndarray)
        self.assertEqual(errors.dtype, np.float)

    def test_values(self):
        # empty sequences
        errors = calc_absolute_errors([], [])
        self.assertTrue(np.allclose(errors, []))
        # detections relative to annotations
        errors = calc_absolute_errors(DETECTIONS, ANNOTATIONS)
        correct = np.asarray([0.01, 0.05, 0.01, 0.015, 0.1, 5.1])
        self.assertTrue(np.allclose(errors, correct))
        # same but with matches given
        errors = calc_absolute_errors(DETECTIONS, ANNOTATIONS, MATCHES)
        self.assertTrue(np.allclose(errors, correct))
        # annotations relative to detections
        errors = calc_absolute_errors(ANNOTATIONS, DETECTIONS)
        correct = np.asarray([0.01, 0.05, 0.01, 0.015, 0.035, 0.485, 0.1])
        self.assertTrue(np.allclose(errors, correct))


class TestCalcRelativeErrorsFunction(unittest.TestCase):

    def test_types(self):
        errors = calc_relative_errors(DETECTIONS, ANNOTATIONS)
        self.assertIsInstance(errors, np.ndarray)

    def test_values(self):
        # empty sequences
        errors = calc_relative_errors([], [])
        self.assertTrue(np.allclose(errors, []))
        # detections relative to annotations
        errors = calc_relative_errors(DETECTIONS, ANNOTATIONS)
        # np.abs(1 - (errors / annotations[matches]))
        # det: [0.99, 1.45, 2.01, 2.015,            3.1,  8.1])
        # tar: [1,    1.5,  2.0,  2.03,  2.05, 2.5, 3])
        correct = np.abs(np.asarray([1 + 0.01 / 1, 1 + 0.05 / 1.5,
                                     1 - 0.01 / 2, 1 + 0.015 / 2.03,
                                     1 - 0.1 / 3, 1 - 5.1 / 3]))
        self.assertTrue(np.allclose(errors, correct))
        # same but with matches given
        errors = calc_relative_errors(DETECTIONS, ANNOTATIONS, MATCHES)
        self.assertTrue(np.allclose(errors, correct))
        # annotations relative to detections
        errors = calc_relative_errors(ANNOTATIONS, DETECTIONS)
        correct = np.abs(np.asarray([1 - 0.01 / 0.99, 1 - 0.05 / 1.45,
                                     1 + 0.01 / 2.01, 1 - 0.015 / 2.015,
                                     1 - 0.035 / 2.015, 1 - 0.485 / 2.015,
                                     1 + 0.1 / 3.1]))
        self.assertTrue(np.allclose(errors, correct))


# test classes
class TestSimpleEvaluationClass(unittest.TestCase):

    def test_types(self):
        e = SimpleEvaluation()
        self.assertIsNone(e.name)
        self.assertIsInstance(e.num_tp, int)
        self.assertIsInstance(e.num_fp, int)
        self.assertIsInstance(e.num_tn, int)
        self.assertIsInstance(e.num_fn, int)
        self.assertIsInstance(e.num_annotations, int)
        self.assertIsInstance(e.precision, float)
        self.assertIsInstance(e.recall, float)
        self.assertIsInstance(e.fmeasure, float)
        self.assertIsInstance(e.accuracy, float)
        self.assertIsInstance(len(e), int)
        self.assertIsInstance(e.metrics, dict)

    def test_conversion(self):
        # conversion from float should work
        e = SimpleEvaluation(float(0), float(0), float(0), float(0))
        self.assertIsInstance(e.num_tp, int)
        self.assertIsInstance(e.num_fp, int)
        self.assertIsInstance(e.num_tn, int)
        self.assertIsInstance(e.num_fn, int)
        # conversion from list or dict should fail
        self.assertRaises(TypeError, SimpleEvaluation, [0], [0], [0], [0])
        self.assertRaises(TypeError, SimpleEvaluation, {}, {}, {}, {})

    def test_results(self):
        # empty evaluation object
        e = SimpleEvaluation()
        self.assertEqual(e.num_tp, 0)
        self.assertEqual(e.num_fp, 0)
        self.assertEqual(e.num_tn, 0)
        self.assertEqual(e.num_fn, 0)
        self.assertEqual(e.num_annotations, 0)
        self.assertEqual(len(e), 0)
        # all correct (none) retrieved
        self.assertEqual(e.precision, 1)
        # all retrieved (none) are correct
        self.assertEqual(e.recall, 1)
        # 2 * P * R / (P + R)
        self.assertEqual(e.fmeasure, 1)
        # (TP + TN) / (TP + FP + TN + FN)
        self.assertEqual(e.accuracy, 1)
        # metric dictionary
        self.assertEqual(list(e.metrics.keys()),
                         ['num_tp', 'num_fp', 'num_tn', 'num_fn',
                          'num_annotations', 'precision', 'recall',
                          'fmeasure', 'accuracy'])
        correct = OrderedDict([('num_tp', 0), ('num_fp', 0), ('num_tn', 0),
                               ('num_fn', 0), ('num_annotations', 0),
                               ('precision', 1.0), ('recall', 1.0),
                               ('fmeasure', 1.0), ('accuracy', 1.0)])
        self.assertEqual(e.metrics, correct)

        # test with other values
        e = SimpleEvaluation(num_tp=5, num_fp=3, num_tn=4, num_fn=1)
        self.assertEqual(e.num_tp, 5)
        self.assertEqual(e.num_fp, 3)
        self.assertEqual(e.num_tn, 4)
        self.assertEqual(e.num_fn, 1)
        self.assertEqual(e.num_annotations, 6)
        self.assertEqual(len(e), 6)
        # correct / retrieved
        self.assertEqual(e.precision, 5. / 8.)
        # correct / relevant
        self.assertEqual(e.recall, 5. / 6.)
        # 2 * P * R / (P + R)
        f = 2 * (5. / 8.) * (5. / 6.) / ((5. / 8.) + (5. / 6.))
        self.assertEqual(e.fmeasure, f)
        # (TP + TN) / (TP + FP + TN + FN)
        self.assertEqual(e.accuracy, (5. + 4) / (5 + 3 + 4 + 1))

        # test with no true positives/negatives
        e = SimpleEvaluation(num_tp=0, num_fp=3, num_tn=0, num_fn=1)
        self.assertEqual(e.num_tp, 0)
        self.assertEqual(e.num_fp, 3)
        self.assertEqual(e.num_tn, 0)
        self.assertEqual(e.num_fn, 1)
        self.assertEqual(e.num_annotations, 1)
        self.assertEqual(len(e), 1)
        self.assertEqual(e.precision, 0)
        self.assertEqual(e.recall, 0)
        self.assertEqual(e.fmeasure, 0)
        self.assertEqual(e.accuracy, 0)


class TestEvaluationClass(unittest.TestCase):

    def test_types(self):
        e = Evaluation()
        self.assertIsNone(e.name)
        self.assertIsInstance(e.num_tp, int)
        self.assertIsInstance(e.num_fp, int)
        self.assertIsInstance(e.num_tn, int)
        self.assertIsInstance(e.num_fn, int)
        self.assertIsInstance(e.precision, float)
        self.assertIsInstance(e.recall, float)
        self.assertIsInstance(e.fmeasure, float)
        self.assertIsInstance(e.fmeasure, float)
        self.assertIsInstance(e.tp, np.ndarray)
        self.assertIsInstance(e.fp, np.ndarray)
        self.assertIsInstance(e.tn, np.ndarray)
        self.assertIsInstance(e.fn, np.ndarray)
        self.assertIsInstance(len(e), int)
        self.assertIsInstance(e.metrics, dict)

    def test_conversion(self):
        # # conversion from dict should fail
        # with self.assertRaises(TypeError):
        #     Evaluation(tp={}, fp={}, tn={}, fn={})
        # conversion from int or float should fail
        with self.assertRaises(TypeError):
            Evaluation(tp=int(0), fp=int(0), tn=int(0), fn=int(0))
        with self.assertRaises(TypeError):
            Evaluation(tp=float(0), fp=float(0), tn=float(0), fn=float(0))
        e = Evaluation(tp={}, fp={}, tn={}, fn={})
        self.assertIsInstance(e.tp, np.ndarray)
        self.assertIsInstance(e.fp, np.ndarray)
        self.assertIsInstance(e.tn, np.ndarray)
        self.assertIsInstance(e.fn, np.ndarray)

    def test_results(self):
        # empty evaluation object
        e = Evaluation()
        self.assertTrue(np.allclose(e.tp, np.empty(0)))
        self.assertTrue(np.allclose(e.fp, np.empty(0)))
        self.assertTrue(np.allclose(e.tn, np.empty(0)))
        self.assertTrue(np.allclose(e.fn, np.empty(0)))
        self.assertEqual(e.num_tp, 0)
        self.assertEqual(e.num_fp, 0)
        self.assertEqual(e.num_tn, 0)
        self.assertEqual(e.num_fn, 0)
        # p: all correct (none) retrieved
        self.assertEqual(e.precision, 1)
        # r: all retrieved (none) are correct
        self.assertEqual(e.recall, 1)
        # f: 2 * P * R / (P + R)
        self.assertEqual(e.fmeasure, 1)
        # acc: (TP + TN) / (TP + FP + TN + FN)
        self.assertEqual(e.accuracy, 1)
        # test metric dictionary keys
        self.assertEqual(list(e.metrics.keys()),
                         ['num_tp', 'num_fp', 'num_tn', 'num_fn',
                          'num_annotations', 'precision', 'recall',
                          'fmeasure', 'accuracy'])
        # test with other values
        e = Evaluation(tp=[1, 2, 3.0], fp=[1.5], fn=[0, 3.1])
        tp = np.asarray([1, 2, 3], dtype=np.float)
        self.assertTrue(np.allclose(e.tp, tp))
        fp = np.asarray([1.5], dtype=np.float)
        self.assertTrue(np.allclose(e.fp, fp))
        tn = np.asarray([], dtype=np.float)
        self.assertTrue(np.allclose(e.tn, tn))
        fn = np.asarray([0, 3.1], dtype=np.float)
        self.assertTrue(np.allclose(e.fn, fn))
        self.assertEqual(e.num_tp, 3)
        self.assertEqual(e.num_fp, 1)
        self.assertEqual(e.num_tn, 0)
        self.assertEqual(e.num_fn, 2)
        # p: correct / retrieved
        self.assertEqual(e.precision, 3. / 4.)
        # r: correct / relevant
        self.assertEqual(e.recall, 3. / 5.)
        # f: 2 * P * R / (P + R)
        f = 2 * (3. / 4.) * (3. / 5.) / ((3. / 4.) + (3. / 5.))
        self.assertEqual(e.fmeasure, f)
        # acc: (TP + TN) / (TP + FP + TN + FN)
        self.assertEqual(e.accuracy, 3. / (3 + 1 + 2))


class TestMultiClassEvaluationClass(unittest.TestCase):

    def test_types(self):
        e = MultiClassEvaluation()
        self.assertIsNone(e.name)
        self.assertIsInstance(e.num_tp, int)
        self.assertIsInstance(e.num_fp, int)
        self.assertIsInstance(e.num_tn, int)
        self.assertIsInstance(e.num_fn, int)
        self.assertIsInstance(e.precision, float)
        self.assertIsInstance(e.recall, float)
        self.assertIsInstance(e.fmeasure, float)
        self.assertIsInstance(e.fmeasure, float)
        self.assertIsInstance(e.tp, np.ndarray)
        self.assertIsInstance(e.fp, np.ndarray)
        self.assertIsInstance(e.tn, np.ndarray)
        self.assertIsInstance(e.fn, np.ndarray)
        self.assertEqual(e.tp.shape, (0, 2))
        self.assertEqual(e.fp.shape, (0, 2))
        self.assertEqual(e.tn.shape, (0, 2))
        self.assertEqual(e.fn.shape, (0, 2))
        self.assertIsInstance(len(e), int)
        self.assertIsInstance(e.metrics, dict)


class TestSumEvaluationClass(unittest.TestCase):

    def test_types(self):
        e = SumEvaluation([])
        self.assertIsInstance(e.eval_objects, list)
        self.assertIsInstance(e.name, str)
        self.assertIsInstance(e.num_tp, int)
        self.assertIsInstance(e.num_fp, int)
        self.assertIsInstance(e.num_tn, int)
        self.assertIsInstance(e.num_fn, int)
        self.assertIsInstance(e.precision, float)
        self.assertIsInstance(e.recall, float)
        self.assertIsInstance(e.fmeasure, float)
        self.assertIsInstance(e.accuracy, float)
        self.assertIsInstance(len(e), int)
        self.assertIsInstance(e.metrics, dict)

    def test_results(self):
        # empty evaluation
        e = SumEvaluation([])
        self.assertEqual(e.num_tp, 0)
        self.assertEqual(e.num_fp, 0)
        self.assertEqual(e.num_tn, 0)
        self.assertEqual(e.num_fn, 0)
        self.assertEqual(e.precision, 1)
        self.assertEqual(e.recall, 1)
        self.assertEqual(e.fmeasure, 1)
        self.assertEqual(e.accuracy, 1)
        self.assertEqual(len(e), 0)
        # empty SimpleEvaluation
        e = SumEvaluation([SimpleEvaluation()])
        self.assertEqual(e.num_tp, 0)
        self.assertEqual(e.num_fp, 0)
        self.assertEqual(e.num_tn, 0)
        self.assertEqual(e.num_fn, 0)
        self.assertEqual(e.precision, 1)
        self.assertEqual(e.recall, 1)
        self.assertEqual(e.fmeasure, 1)
        self.assertEqual(e.accuracy, 1)
        self.assertEqual(len(e), 1)
        # empty SimpleEvaluation without the list
        e = SumEvaluation(SimpleEvaluation())
        self.assertEqual(e.num_tp, 0)
        self.assertEqual(e.num_fp, 0)
        self.assertEqual(e.num_tn, 0)
        self.assertEqual(e.num_fn, 0)
        self.assertEqual(e.precision, 1)
        self.assertEqual(e.recall, 1)
        self.assertEqual(e.fmeasure, 1)
        self.assertEqual(e.accuracy, 1)
        self.assertEqual(len(e), 1)
        # empty and real SimpleEvaluation
        e1 = SimpleEvaluation()
        e2 = SimpleEvaluation(num_tp=5, num_fp=3, num_tn=4, num_fn=1)
        e = SumEvaluation([e1, e2])
        self.assertEqual(e.num_tp, 5)
        self.assertEqual(e.num_fp, 3)
        self.assertEqual(e.num_tn, 4)
        self.assertEqual(e.num_fn, 1)
        self.assertEqual(e.precision, 5. / 8.)
        self.assertEqual(e.recall, 5. / 6.)
        f = 2 * (5. / 8.) * (5. / 6.) / ((5. / 8.) + (5. / 6.))
        self.assertEqual(e.fmeasure, f)
        self.assertEqual(e.accuracy, (5. + 4) / (5 + 3 + 4 + 1))
        self.assertEqual(len(e), 2)


class TestMeanEvaluationClass(unittest.TestCase):

    def test_types(self):
        e = MeanEvaluation([])
        self.assertIsInstance(e.eval_objects, list)
        self.assertIsInstance(e.name, str)
        self.assertIsInstance(e.num_tp, float)
        self.assertIsInstance(e.num_fp, float)
        self.assertIsInstance(e.num_tn, float)
        self.assertIsInstance(e.num_fn, float)
        self.assertIsInstance(e.num_annotations, float)
        self.assertIsInstance(e.precision, float)
        self.assertIsInstance(e.recall, float)
        self.assertIsInstance(e.fmeasure, float)
        self.assertIsInstance(e.accuracy, float)
        self.assertIsInstance(len(e), int)
        self.assertIsInstance(e.metrics, dict)

    def test_results(self):
        # empty MeanEvaluation
        e = MeanEvaluation([])
        self.assertEqual(e.num_tp, 0)
        self.assertEqual(e.num_fp, 0)
        self.assertEqual(e.num_tn, 0)
        self.assertEqual(e.num_fn, 0)
        self.assertEqual(e.num_annotations, 0)
        self.assertTrue(math.isnan(e.precision))
        self.assertTrue(math.isnan(e.recall))
        self.assertTrue(math.isnan(e.fmeasure))
        self.assertTrue(math.isnan(e.accuracy))
        self.assertEqual(len(e), 0)
        # empty SimpleEvaluation
        e = MeanEvaluation([SimpleEvaluation()])
        self.assertEqual(e.num_tp, 0)
        self.assertEqual(e.num_fp, 0)
        self.assertEqual(e.num_tn, 0)
        self.assertEqual(e.num_fn, 0)
        self.assertEqual(e.num_annotations, 0)
        self.assertEqual(e.precision, 1)
        self.assertEqual(e.recall, 1)
        self.assertEqual(e.fmeasure, 1)
        self.assertEqual(e.accuracy, 1)
        self.assertEqual(len(e), 1)
        # empty SimpleEvaluation without the list
        e = MeanEvaluation(SimpleEvaluation())
        self.assertEqual(e.num_tp, 0)
        self.assertEqual(e.num_fp, 0)
        self.assertEqual(e.num_tn, 0)
        self.assertEqual(e.num_fn, 0)
        self.assertEqual(e.num_annotations, 0)
        self.assertEqual(e.precision, 1)
        self.assertEqual(e.recall, 1)
        self.assertEqual(e.fmeasure, 1)
        self.assertEqual(e.accuracy, 1)
        self.assertEqual(len(e), 1)
        # empty and real SimpleEvaluation
        e1 = SimpleEvaluation()
        e2 = SimpleEvaluation(num_tp=5, num_fp=3, num_tn=4, num_fn=1)
        e = MeanEvaluation([e1, e2])
        self.assertEqual(e.num_tp, 5 / 2.)
        self.assertEqual(e.num_fp, 3 / 2.)
        self.assertEqual(e.num_tn, 4 / 2.)
        self.assertEqual(e.num_fn, 1 / 2.)
        self.assertEqual(e.num_annotations, 6 / 2.)
        self.assertEqual(e.precision, (1 + 5. / 8.) / 2.)
        self.assertEqual(e.recall, (1 + 5. / 6.) / 2.)
        f = (1 + 2 * (5. / 8.) * (5. / 6.) / ((5. / 8.) + (5. / 6.))) / 2.
        self.assertEqual(e.fmeasure, f)
        self.assertEqual(e.accuracy, (1 + (5. + 4) / (5 + 3 + 4 + 1)) / 2.)
        self.assertEqual(len(e), 2)
