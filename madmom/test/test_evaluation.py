# encoding: utf-8
"""
This file contains onset evaluation tests.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import unittest

from madmom.evaluation import *


DETECTIONS = np.asarray([0.99, 1.45, 2.01, 2.015, 3.1, 8.1])
TARGETS = np.asarray([1, 1.5, 2.0, 2.03, 2.05, 2.5, 3])


# test functions
class TestFindClosestMatchesFunction(unittest.TestCase):

    def test_wrong_passed_type(self):
        with self.assertRaises(AttributeError):
            find_closest_matches([0, 1], [2, 3])

    def test_return_type(self):
        self.assertIsInstance(find_closest_matches([], []), np.ndarray)

    def test_return_value(self):
        matches = find_closest_matches(DETECTIONS, TARGETS)
        correct = np.asarray([0, 1, 2, 3, 6, 6])
        self.assertTrue(np.array_equal(matches, correct))

    def test_return_value_reverse(self):
        matches = find_closest_matches(TARGETS, DETECTIONS)
        correct = np.asarray([0, 1, 2, 3, 3, 3, 4])
        self.assertTrue(np.array_equal(matches, correct))


class TestCalcErrorsFunction(unittest.TestCase):

    def test_wrong_passed_type(self):
        with self.assertRaises(AttributeError):
            calc_errors([0, 1], [2, 3])

    def test_wrong_passed_matches_type(self):
        with self.assertRaises(TypeError):
            calc_errors([], [], [0, 1])

    def test_return_type(self):
        self.assertIsInstance(calc_errors(DETECTIONS, TARGETS), np.ndarray)

    def test_return_value(self):
        errors = calc_errors(DETECTIONS, TARGETS)
        correct = np.asarray([-0.01, -0.05, 0.01, -0.015, 0.1, 5.1])
        self.assertTrue(np.allclose(errors, correct))

    def test_return_value_reverse(self):
        errors = calc_errors(TARGETS, DETECTIONS)
        correct = np.asarray([0.01, 0.05, -0.01, 0.015, 0.035, 0.485, -0.1])
        self.assertTrue(np.allclose(errors, correct))


class TestCalcAbsoluteErrorsFunction(unittest.TestCase):

    def test_wrong_passed_type(self):
        with self.assertRaises(AttributeError):
            calc_absolute_errors([0, 1], [2, 3])

    def test_wrong_passed_matches_type(self):
        with self.assertRaises(TypeError):
            calc_absolute_errors([], [], [0, 1])

    def test_return_type(self):
        self.assertIsInstance(calc_absolute_errors(DETECTIONS, TARGETS),
                              np.ndarray)

    def test_return_value(self):
        errors = calc_absolute_errors(DETECTIONS, TARGETS)
        correct = np.asarray([0.01, 0.05, 0.01, 0.015, 0.1, 5.1])
        self.assertTrue(np.allclose(errors, correct))

    def test_return_value_reverse(self):
        errors = calc_absolute_errors(TARGETS, DETECTIONS)
        correct = np.asarray([0.01, 0.05, 0.01, 0.015, 0.035, 0.485, 0.1])
        self.assertTrue(np.allclose(errors, correct))


class TestCalcRelativeErrorsFunction(unittest.TestCase):

    def test_wrong_passed_type(self):
        with self.assertRaises(AttributeError):
            calc_relative_errors([0, 1], [2, 3])

    def test_wrong_passed_matches_type(self):
        with self.assertRaises(TypeError):
            calc_relative_errors([], [], [0, 1])

    def test_return_type(self):
        self.assertIsInstance(calc_relative_errors(DETECTIONS, TARGETS),
                              np.ndarray)

    def test_return_value(self):
        errors = calc_relative_errors(DETECTIONS, TARGETS)
        # np.abs(1 - (errors / targets[matches]))
        # det: [0.99, 1.45, 2.01, 2.015,            3.1,  8.1])
        # tar: [1,    1.5,  2.0,  2.03,  2.05, 2.5, 3])
        correct = np.abs(np.asarray([1 + 0.01 / 1, 1 + 0.05 / 1.5,
                                     1 - 0.01 / 2, 1 + 0.015 / 2.03,
                                     1 - 0.1 / 3, 1 - 5.1 / 3]))
        self.assertTrue(np.allclose(errors, correct))

    def test_return_value_reverse(self):
        errors = calc_relative_errors(TARGETS, DETECTIONS)
        correct = np.abs(np.asarray([1 - 0.01 / 0.99, 1 - 0.05 / 1.45,
                                     1 + 0.01 / 2.01, 1 - 0.015 / 2.015,
                                     1 - 0.035 / 2.015, 1 - 0.485 / 2.015,
                                     1 + 0.1 / 3.1]))
        self.assertTrue(np.allclose(errors, correct))


# test classes
class TestSimpleEvaluationClass(unittest.TestCase):

    def test_types(self):
        e = SimpleEvaluation()
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

    def test_add(self):
        e = SimpleEvaluation()
        self.assertIsInstance(e + Evaluation(), SimpleEvaluation)
        self.assertIsInstance(e + SimpleEvaluation(), SimpleEvaluation)
        self.assertIsInstance(e + SumEvaluation(), SimpleEvaluation)
        self.assertIsInstance(e + MeanEvaluation(), SimpleEvaluation)

    def test_iadd(self):
        e = SimpleEvaluation()
        e += SimpleEvaluation()
        self.assertIsInstance(e, SimpleEvaluation)
        e += Evaluation()
        self.assertIsInstance(e, SimpleEvaluation)
        e += SumEvaluation()
        self.assertIsInstance(e, SimpleEvaluation)
        e += MeanEvaluation()
        self.assertIsInstance(e, SimpleEvaluation)

    def test_results(self):
        e = SimpleEvaluation()
        self.assertEqual(e.num_tp, 0)
        self.assertEqual(e.num_fp, 0)
        self.assertEqual(e.num_tn, 0)
        self.assertEqual(e.num_fn, 0)
        # all correct (none) retrieved
        self.assertEqual(e.precision, 1)
        # all retrieved (none) are correct
        self.assertEqual(e.recall, 1)
        # 2 * P * R / (P + R)
        self.assertEqual(e.fmeasure, 1)
        # (TP + TN) / (TP + FP + TN + FN)
        self.assertEqual(e.accuracy, 1)
        # errors
        self.assertTrue(np.array_equal(e.errors, np.empty(0)))
        self.assertEqual(e.mean_error, 0)
        self.assertEqual(e.std_error, 0)

    def test_results_5341(self):
        e = SimpleEvaluation(num_tp=5, num_fp=3, num_tn=4, num_fn=1)
        self.assertEqual(e.num_tp, 5)
        self.assertEqual(e.num_fp, 3)
        self.assertEqual(e.num_tn, 4)
        self.assertEqual(e.num_fn, 1)
        # correct / retrieved
        self.assertEqual(e.precision, 5. / 8.)
        # correct / relevant
        self.assertEqual(e.recall, 5. / 6.)
        # 2 * P * R / (P + R)
        f = 2 * (5. / 8.) * (5. / 6.) / ((5. / 8.) + (5. / 6.))
        self.assertEqual(e.fmeasure, f)
        # (TP + TN) / (TP + FP + TN + FN)
        self.assertEqual(e.accuracy, (5. + 4) / (5 + 3 + 4 + 1))
        # array with errors
        self.assertTrue(np.array_equal(e.errors, np.zeros(0)))
        self.assertEqual(e.mean_error, 0)
        self.assertEqual(e.std_error, 0)


class TestEvaluationClass(unittest.TestCase):

    def test_types(self):
        e = Evaluation()
        self.assertIsInstance(e.num_tp, int)
        self.assertIsInstance(e.num_fp, int)
        self.assertIsInstance(e.num_tn, int)
        self.assertIsInstance(e.num_fn, int)
        self.assertIsInstance(e.precision, float)
        self.assertIsInstance(e.recall, float)
        self.assertIsInstance(e.fmeasure, float)
        self.assertIsInstance(e.fmeasure, float)
        self.assertIsInstance(e.errors, np.ndarray)
        self.assertIsInstance(e.tp, np.ndarray)
        self.assertIsInstance(e.fp, np.ndarray)
        self.assertIsInstance(e.tn, np.ndarray)
        self.assertIsInstance(e.fn, np.ndarray)

    def test_conversion(self):
        # conversion from float should work
        e = Evaluation(tp=float(0), fp=float(0), tn=float(0), fn=float(0))
        self.assertIsInstance(e.tp, np.ndarray)
        self.assertIsInstance(e.fp, np.ndarray)
        self.assertIsInstance(e.tn, np.ndarray)
        self.assertIsInstance(e.fn, np.ndarray)
        # conversion from int should work
        e = Evaluation(tp=int(0), fp=int(0), tn=int(0), fn=int(0))
        self.assertIsInstance(e.tp, np.ndarray)
        self.assertIsInstance(e.fp, np.ndarray)
        self.assertIsInstance(e.tn, np.ndarray)
        self.assertIsInstance(e.fn, np.ndarray)
        # conversion from list should work
        e = Evaluation(tp=[0], fp=[0], tn=[0], fn=[0])
        self.assertIsInstance(e.tp, np.ndarray)
        self.assertIsInstance(e.fp, np.ndarray)
        self.assertIsInstance(e.tn, np.ndarray)
        self.assertIsInstance(e.fn, np.ndarray)
        # conversion from dict should fail
        self.assertRaises(TypeError, Evaluation(), tp={}, fp={}, tn={}, fn={})

    def test_add(self):
        e = Evaluation()
        self.assertIsInstance(e + Evaluation(), Evaluation)
        # can't add the following, because the don't have TP, FP, TN, FN arrays
        with self.assertRaises(TypeError):
            e + SimpleEvaluation()
        with self.assertRaises(TypeError):
            e + SumEvaluation()
        with self.assertRaises(TypeError):
            e + MeanEvaluation()

    def test_iadd(self):
        e = Evaluation()
        e += Evaluation()
        self.assertIsInstance(e, Evaluation)
        # can't add the following, because the don't have TP, FP, TN, FN arrays
        with self.assertRaises(TypeError):
            e += SimpleEvaluation()
        with self.assertRaises(TypeError):
            e += SumEvaluation()
        with self.assertRaises(TypeError):
            e += MeanEvaluation()

    def test_results_empty(self):
        e = Evaluation()
        self.assertTrue(np.array_equal(e.tp, np.empty(0)))
        self.assertTrue(np.array_equal(e.fp, np.empty(0)))
        self.assertTrue(np.array_equal(e.tn, np.empty(0)))
        self.assertTrue(np.array_equal(e.fn, np.empty(0)))
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
        # errors
        self.assertTrue(np.array_equal(e.errors, np.empty(0)))
        self.assertEqual(e.mean_error, 0)
        self.assertEqual(e.std_error, 0)

    def test_results_3102(self):
        e = Evaluation(tp=[1, 2, 3.0], fp=[1.5], fn=[0, 3.1])
        tp = np.asarray([1, 2, 3], dtype=np.float)
        self.assertTrue(np.array_equal(e.tp, tp))
        fp = np.asarray([1.5], dtype=np.float)
        self.assertTrue(np.array_equal(e.fp, fp))
        tn = np.asarray([], dtype=np.float)
        self.assertTrue(np.array_equal(e.tn, tn))
        fn = np.asarray([0, 3.1], dtype=np.float)
        self.assertTrue(np.array_equal(e.fn, fn))
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
        # errors
        self.assertTrue(np.array_equal(e.errors, np.empty(0)))
        self.assertEqual(e.mean_error, 0)
        self.assertEqual(e.std_error, 0)


class TestSumEvaluationClass(unittest.TestCase):

    def test_types(self):
        e = SumEvaluation()
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

    def test_add(self):
        e = SumEvaluation()
        self.assertIsInstance(e + Evaluation(), SumEvaluation)
        self.assertIsInstance(e + SimpleEvaluation(), SumEvaluation)
        self.assertIsInstance(e + SumEvaluation(), SumEvaluation)
        self.assertIsInstance(e + MeanEvaluation(), SumEvaluation)

    def test_iadd(self):
        e = SumEvaluation()
        e += Evaluation()
        self.assertIsInstance(e, SumEvaluation)
        e += SimpleEvaluation()
        self.assertIsInstance(e, SumEvaluation)
        e += SumEvaluation()
        self.assertIsInstance(e, SumEvaluation)
        e += MeanEvaluation()
        self.assertIsInstance(e, SumEvaluation)

    def test_iadd_types(self):
        e = SumEvaluation()
        e += SimpleEvaluation()
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

    def test_results_empty(self):
        e = SumEvaluation()
        e += SimpleEvaluation()
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
        # errors
        self.assertTrue(np.array_equal(e.errors, np.empty(0)))
        self.assertEqual(e.mean_error, 0)
        self.assertEqual(e.std_error, 0)

    def test_results_5341(self):
        e = SumEvaluation()
        e += SimpleEvaluation()
        e += SimpleEvaluation(num_tp=5, num_fp=3, num_tn=4, num_fn=1)
        self.assertEqual(e.num_tp, 5)
        self.assertEqual(e.num_fp, 3)
        self.assertEqual(e.num_tn, 4)
        self.assertEqual(e.num_fn, 1)
        self.assertEqual(e.precision, 5. / 8.)
        self.assertEqual(e.recall, 5. / 6.)
        f = 2 * (5. / 8.) * (5. / 6.) / ((5. / 8.) + (5. / 6.))
        self.assertEqual(e.fmeasure, f)
        self.assertEqual(e.accuracy, (5. + 4) / (5 + 3 + 4 + 1))
        self.assertTrue(np.array_equal(e.errors, np.zeros(0)))
        self.assertEqual(e.mean_error, 0)
        self.assertEqual(e.std_error, 0)


class TestMeanEvaluationClass(unittest.TestCase):

    def test_types(self):
        e = MeanEvaluation()
        self.assertIsInstance(e.num_tp, float)
        self.assertIsInstance(e.num_fp, float)
        self.assertIsInstance(e.num_tn, float)
        self.assertIsInstance(e.num_fn, float)
        self.assertIsInstance(e.precision, float)
        self.assertIsInstance(e.recall, float)
        self.assertIsInstance(e.fmeasure, float)
        self.assertTrue(np.array_equal(e.errors, np.zeros(0)))

    def test_append(self):
        e = MeanEvaluation()
        e.append(Evaluation())
        self.assertIsInstance(e, MeanEvaluation)
        e.append(SimpleEvaluation())
        self.assertIsInstance(e, MeanEvaluation)
        e.append(SumEvaluation())
        self.assertIsInstance(e, MeanEvaluation)
        e.append(MeanEvaluation())
        self.assertIsInstance(e, MeanEvaluation)
        # appending something should not return anything
        self.assertEqual(e.append(Evaluation()), None)

    def test_append_types(self):
        e = MeanEvaluation()
        e.append(Evaluation())
        self.assertIsInstance(e.num_tp, float)
        self.assertIsInstance(e.num_fp, float)
        self.assertIsInstance(e.num_tn, float)
        self.assertIsInstance(e.num_fn, float)
        self.assertIsInstance(e.precision, float)
        self.assertIsInstance(e.recall, float)
        self.assertIsInstance(e.fmeasure, float)
        self.assertTrue(np.array_equal(e.errors, np.zeros(0)))

    def test_results_empty(self):
        e = MeanEvaluation()
        self.assertEqual(e.num_tp, 0)
        self.assertEqual(e.num_fp, 0)
        self.assertEqual(e.num_tn, 0)
        self.assertEqual(e.num_fn, 0)
        # p: all correct (none) retrieved
        self.assertEqual(e.precision, 0)
        # r: all retrieved (none) are correct
        self.assertEqual(e.recall, 0)
        # f: 2 * P * R / (P + R)
        self.assertEqual(e.fmeasure, 0)
        # acc: (TP + TN) / (TP + FP + TN + FN)
        self.assertEqual(e.accuracy, 0)
        # errors
        self.assertTrue(np.array_equal(e.errors, np.zeros(0)))
        self.assertEqual(e.mean_error, 0)
        self.assertEqual(e.std_error, 0)
