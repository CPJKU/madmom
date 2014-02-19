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
class TestFindClosestMatches(unittest.TestCase):

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


class TestCalcErrors(unittest.TestCase):

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


class TestCalcAbsoluteErrors(unittest.TestCase):

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


class TestCalcRelativeErrors(unittest.TestCase):

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


# SimpleEvaluation

# test types
class TestSimpleEvaluationTypes(unittest.TestCase):
    e = Evaluation()

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


class TestSimpleEvaluationAddition(unittest.TestCase):
    e = SimpleEvaluation()

    def test_add_evaluation_object(self):
        e = self.e + Evaluation()
        self.assertIsInstance(e, SimpleEvaluation)

    def test_add_simple_evaluation_object(self):
        e = self.e + SimpleEvaluation()
        self.assertIsInstance(e, SimpleEvaluation)

    def test_add_sum_evaluation_object(self):
        e = self.e + SumEvaluation()
        self.assertIsInstance(e, SimpleEvaluation)

    def test_add_mean_evaluation_object(self):
        e = self.e + MeanEvaluation()
        self.assertIsInstance(e, SimpleEvaluation)

    def test_iadd_evaluation_object(self):
        self.e += Evaluation()
        self.assertIsInstance(self.e, SimpleEvaluation)

    def test_iadd_simple_evaluation_object(self):
        self.e += SimpleEvaluation()
        self.assertIsInstance(self.e, SimpleEvaluation)

    def test_iadd_sum_evaluation_object(self):
        self.e += SumEvaluation()
        self.assertIsInstance(self.e, SimpleEvaluation)

    def test_iadd_mean_evaluation_object(self):
        self.e += MeanEvaluation()
        self.assertIsInstance(self.e, SimpleEvaluation)


# test results
class TestSimpleEvaluationResultsNone(unittest.TestCase):
    e = SimpleEvaluation()

    def test_num_tp(self):
        self.assertEqual(self.e.num_tp, 0)

    def test_num_fp(self):
        self.assertEqual(self.e.num_fp, 0)

    def test_num_tn(self):
        self.assertEqual(self.e.num_tn, 0)

    def test_num_fn(self):
        self.assertEqual(self.e.num_fn, 0)

    def test_precision(self):
        # all correct (none) retrieved
        self.assertEqual(self.e.precision, 1)

    def test_recall(self):
        # all retrieved (none) are correct
        self.assertEqual(self.e.recall, 1)

    def test_fmeasure(self):
        # 2 * P * R / (P + R)
        self.assertEqual(self.e.fmeasure, 1)

    def test_accuracy(self):
        # (TP + TN) / (TP + FP + TN + FN)
        self.assertEqual(self.e.accuracy, 1)

    def test_errors(self):
        # array with errors
        self.assertTrue(np.array_equal(self.e.errors, np.empty(0)))

    def test_mean_error(self):
        self.assertEqual(self.e.mean_error, 0)

    def test_std_error(self):
        self.assertEqual(self.e.std_error, 0)


# test results with other values
class TestSimpleEvaluationResults5341(unittest.TestCase):
    e = SimpleEvaluation(num_tp=5, num_fp=3, num_tn=4, num_fn=1)

    def test_num_tp(self):
        self.assertEqual(self.e.num_tp, 5)

    def test_num_fp(self):
        self.assertEqual(self.e.num_fp, 3)

    def test_num_tn(self):
        self.assertEqual(self.e.num_tn, 4)

    def test_num_fn(self):
        self.assertEqual(self.e.num_fn, 1)

    def test_precision(self):
        # correct / retrieved
        self.assertEqual(self.e.precision, 5. / 8.)

    def test_recall(self):
        # correct / relevant
        self.assertEqual(self.e.recall, 5. / 6.)

    def test_fmeasure(self):
        # 2 * P * R / (P + R)
        self.assertEqual(self.e.fmeasure, 2 * (5. / 8.) * (5. / 6.) /
                         ((5. / 8.) + (5. / 6.)))

    def test_accuracy(self):
        # (TP + TN) / (TP + FP + TN + FN)
        self.assertEqual(self.e.accuracy, (5. + 4) / (5 + 3 + 4 + 1))

    def test_errors(self):
        # array with errors
        self.assertTrue(np.array_equal(self.e.errors, np.zeros(0)))

    def test_mean_error(self):
        self.assertEqual(self.e.mean_error, 0)

    def test_std_error(self):
        self.assertEqual(self.e.std_error, 0)


# Evaluation

# test types
class TestEvaluationTypes(unittest.TestCase):
    e = Evaluation()

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
        self.assertIsInstance(self.e.fmeasure, float)

    def test_errors_type(self):
        self.assertIsInstance(self.e.errors, np.ndarray)

    def test_tp_type(self):
        self.assertIsInstance(self.e.tp, np.ndarray)

    def test_fp_type(self):
        self.assertIsInstance(self.e.fp, np.ndarray)

    def test_tn_type(self):
        self.assertIsInstance(self.e.tn, np.ndarray)

    def test_fn_type(self):
        self.assertIsInstance(self.e.fn, np.ndarray)

    def test_conversion_from_float(self):
        e = Evaluation(tp=float(0), fp=float(0), tn=float(0), fn=float(0))
        self.assertIsInstance(e.tp, np.ndarray)
        self.assertIsInstance(e.fp, np.ndarray)
        self.assertIsInstance(e.tn, np.ndarray)
        self.assertIsInstance(e.fn, np.ndarray)

    def test_conversion_from_int(self):
        e = Evaluation(tp=int(0), fp=int(0), tn=int(0), fn=int(0))
        self.assertIsInstance(e.tp, np.ndarray)
        self.assertIsInstance(e.fp, np.ndarray)
        self.assertIsInstance(e.tn, np.ndarray)
        self.assertIsInstance(e.fn, np.ndarray)

    def test_conversion_from_list(self):
        e = Evaluation(tp=[0], fp=[0], tn=[0], fn=[0])
        self.assertIsInstance(e.tp, np.ndarray)
        self.assertIsInstance(e.fp, np.ndarray)
        self.assertIsInstance(e.tn, np.ndarray)
        self.assertIsInstance(e.fn, np.ndarray)

    def test_conversion_from_dict(self):
        self.assertRaises(TypeError, Evaluation(), tp={}, fp={}, tn={}, fn={})


class TestEvaluationAddition(unittest.TestCase):
    e = Evaluation()

    def test_add_evaluation_object(self):
        e = self.e + Evaluation()
        self.assertIsInstance(e, Evaluation)

    def test_iadd_evaluation_object(self):
        self.e += Evaluation()
        self.assertIsInstance(self.e, Evaluation)

    # can't add the following, because the don't have TP, FP, TN, FN arrays
    def test_add_simple_evaluation_object(self):
        with self.assertRaises(TypeError):
            self.e + SimpleEvaluation()

    def test_add_sum_evaluation_object(self):
        with self.assertRaises(TypeError):
            self.e + SumEvaluation()

    def test_add_mean_evaluation_object(self):
        with self.assertRaises(TypeError):
            self.e + MeanEvaluation()

    def test_iadd_simple_evaluation_object(self):
        with self.assertRaises(TypeError):
            self.e += SimpleEvaluation()

    def test_iadd_sum_evaluation_object(self):
        with self.assertRaises(TypeError):
            self.e += SumEvaluation()

    def test_iadd_mean_evaluation_object(self):
        with self.assertRaises(TypeError):
            self.e += MeanEvaluation()


# test results
class TestEvaluationResults0000(unittest.TestCase):
    e = Evaluation()

    def test_tp(self):
        self.assertTrue(np.array_equal(self.e.tp, np.empty(0)))

    def test_fp(self):
        self.assertTrue(np.array_equal(self.e.fp, np.empty(0)))

    def test_tn(self):
        self.assertTrue(np.array_equal(self.e.tn, np.empty(0)))

    def test_fn(self):
        self.assertTrue(np.array_equal(self.e.fn, np.empty(0)))

    def test_num_tp(self):
        self.assertEqual(self.e.num_tp, 0)

    def test_num_fp(self):
        self.assertEqual(self.e.num_fp, 0)

    def test_num_tn(self):
        self.assertEqual(self.e.num_tn, 0)

    def test_num_fn(self):
        self.assertEqual(self.e.num_fn, 0)

    def test_precision(self):
        # all correct (none) retrieved
        self.assertEqual(self.e.precision, 1)

    def test_recall(self):
        # all retrieved (none) are correct
        self.assertEqual(self.e.recall, 1)

    def test_fmeasure(self):
        # 2 * P * R / (P + R)
        self.assertEqual(self.e.fmeasure, 1)

    def test_accuracy(self):
        # (TP + TN) / (TP + FP + TN + FN)
        self.assertEqual(self.e.accuracy, 1)

    def test_errors(self):
        # array with errors
        self.assertTrue(np.array_equal(self.e.errors, np.empty(0)))

    def test_mean_error(self):
        self.assertEqual(self.e.mean_error, 0)

    def test_std_error(self):
        self.assertEqual(self.e.std_error, 0)


# test results with other values
class TestEvaluationResults3102(unittest.TestCase):
    e = Evaluation(tp=[1, 2, 3.0], fp=[1.5], fn=[0, 3.1])

    def test_tp(self):
        self.assertTrue(np.array_equal(self.e.tp, np.asarray([1, 2, 3],
                                                             dtype=np.float)))

    def test_fp(self):
        self.assertTrue(np.array_equal(self.e.fp, np.asarray([1.5],
                                                             dtype=np.float)))

    def test_fn(self):
        self.assertTrue(np.array_equal(self.e.fn, np.asarray([0, 3.1],
                                                             dtype=np.float)))

    def test_num_tp(self):
        self.assertEqual(self.e.num_tp, 3)

    def test_num_fp(self):
        self.assertEqual(self.e.num_fp, 1)

    def test_num_tn(self):
        self.assertEqual(self.e.num_tn, 0)

    def test_num_fn(self):
        self.assertEqual(self.e.num_fn, 2)

    def test_precision(self):
        # correct / retrieved
        self.assertEqual(self.e.precision, 3. / 4.)

    def test_recall(self):
        # correct / relevant
        self.assertEqual(self.e.recall, 3. / 5.)

    def test_fmeasure(self):
        # 2 * P * R / (P + R)
        self.assertEqual(self.e.fmeasure, 2 * (3. / 4.) * (3. / 5.) /
                         ((3. / 4.) + (3. / 5.)))

    def test_accuracy(self):
        # (TP + TN) / (TP + FP + TN + FN)
        self.assertEqual(self.e.accuracy, 3. / (3 + 1 + 2))

    def test_errors(self):
        # array with errors
        self.assertTrue(np.array_equal(self.e.errors, np.empty(0)))

    def test_mean_error(self):
        self.assertEqual(self.e.mean_error, 0)

    def test_std_error(self):
        self.assertEqual(self.e.std_error, 0)


# SumEvaluation
class TestSumEvaluationTypes(unittest.TestCase):
    e = SumEvaluation()

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


class TestSumEvaluationAddition(unittest.TestCase):
    e = SumEvaluation()

    def test_add_evaluation_object(self):
        e = self.e + Evaluation()
        self.assertIsInstance(e, SumEvaluation)

    def test_add_simple_evaluation_object(self):
        e = self.e + SimpleEvaluation()
        self.assertIsInstance(e, SumEvaluation)

    def test_add_sum_evaluation_object(self):
        e = self.e + SumEvaluation()
        self.assertIsInstance(e, SumEvaluation)

    def test_add_mean_evaluation_object(self):
        e = self.e + MeanEvaluation()
        self.assertIsInstance(e, SumEvaluation)

    def test_iadd_evaluation_object(self):
        self.e += Evaluation()
        self.assertIsInstance(self.e, SumEvaluation)

    def test_iadd_simple_evaluation_object(self):
        self.e += SimpleEvaluation()
        self.assertIsInstance(self.e, SumEvaluation)

    def test_iadd_sum_evaluation_object(self):
        self.e += SumEvaluation()
        self.assertIsInstance(self.e, SumEvaluation)

    def test_iadd_mean_evaluation_object(self):
        self.e += MeanEvaluation()
        self.assertIsInstance(self.e, SumEvaluation)


class TestSumEvaluationResults0000(unittest.TestCase):
    e = SumEvaluation()
    e += SimpleEvaluation()

    def test_num_tp(self):
        self.assertEqual(self.e.num_tp, 0)

    def test_num_fp(self):
        self.assertEqual(self.e.num_fp, 0)

    def test_num_tn(self):
        self.assertEqual(self.e.num_tn, 0)

    def test_num_fn(self):
        self.assertEqual(self.e.num_fn, 0)

    def test_precision(self):
        # all correct (none) retrieved
        self.assertEqual(self.e.precision, 1)

    def test_recall(self):
        # all retrieved (none) are correct
        self.assertEqual(self.e.recall, 1)

    def test_fmeasure(self):
        # 2 * P * R / (P + R)
        self.assertEqual(self.e.fmeasure, 1)

    def test_accuracy(self):
        # (TP + TN) / (TP + FP + TN + FN)
        self.assertEqual(self.e.accuracy, 1)

    def test_errors(self):
        # array with errors
        self.assertTrue(np.array_equal(self.e.errors, np.empty(0)))

    def test_mean_error(self):
        self.assertEqual(self.e.mean_error, 0)

    def test_std_error(self):
        self.assertEqual(self.e.std_error, 0)


class TestSumEvaluationResults5341(unittest.TestCase):
    e = SumEvaluation()
    e += SimpleEvaluation()
    e += SimpleEvaluation(num_tp=5, num_fp=3, num_tn=4, num_fn=1)

    def test_num_tp(self):
        self.assertEqual(self.e.num_tp, 5)

    def test_num_fp(self):
        self.assertEqual(self.e.num_fp, 3)

    def test_num_tn(self):
        self.assertEqual(self.e.num_tn, 4)

    def test_num_fn(self):
        self.assertEqual(self.e.num_fn, 1)

    def test_precision(self):
        # correct / retrieved
        self.assertEqual(self.e.precision, 5. / 8.)

    def test_recall(self):
        # correct / relevant
        self.assertEqual(self.e.recall, 5. / 6.)

    def test_fmeasure(self):
        # 2 * P * R / (P + R)
        self.assertEqual(self.e.fmeasure, 2 * (5. / 8.) * (5. / 6.) /
                         ((5. / 8.) + (5. / 6.)))

    def test_accuracy(self):
        # (TP + TN) / (TP + FP + TN + FN)
        self.assertEqual(self.e.accuracy, (5. + 4) / (5 + 3 + 4 + 1))

    def test_errors(self):
        # array with errors
        self.assertTrue(np.array_equal(self.e.errors, np.zeros(0)))

    def test_mean_error(self):
        self.assertEqual(self.e.mean_error, 0)

    def test_std_error(self):
        self.assertEqual(self.e.std_error, 0)


# MeanEvaluation
class TestMeanEvaluationTypes(unittest.TestCase):
    e = MeanEvaluation()

    def test_num_tp_type(self):
        self.assertIsInstance(self.e.num_tp, float)

    def test_num_fp_type(self):
        self.assertIsInstance(self.e.num_fp, float)

    def test_num_tn_type(self):
        self.assertIsInstance(self.e.num_tn, float)

    def test_num_fn_type(self):
        self.assertIsInstance(self.e.num_fn, float)

    def test_precision_type(self):
        self.assertIsInstance(self.e.precision, float)

    def test_recall_type(self):
        self.assertIsInstance(self.e.recall, float)

    def test_fmeasure_type(self):
        self.assertIsInstance(self.e.fmeasure, float)

    def test_errors_type(self):
        self.assertTrue(np.array_equal(self.e.errors, np.zeros(0)))


class TestMeanEvaluationAppend(unittest.TestCase):
    e = MeanEvaluation()

    def test_append_evaluation_object(self):
        self.e.append(Evaluation())
        self.assertIsInstance(self.e, MeanEvaluation)

    def test_append_simple_evaluation_object(self):
        self.e.append(SimpleEvaluation())
        self.assertIsInstance(self.e, MeanEvaluation)

    def test_append_sum_evaluation_object(self):
        self.e.append(SumEvaluation())
        self.assertIsInstance(self.e, MeanEvaluation)

    def test_append_mean_evaluation_object(self):
        self.e.append(MeanEvaluation())
        self.assertIsInstance(self.e, MeanEvaluation)


# test results
class TestMeanEvaluationResults0000(unittest.TestCase):
    e = MeanEvaluation()

    def test_num_tp(self):
        self.assertEqual(self.e.num_tp, 0)

    def test_num_fp(self):
        self.assertEqual(self.e.num_fp, 0)

    def test_num_tn(self):
        self.assertEqual(self.e.num_tn, 0)

    def test_num_fn(self):
        self.assertEqual(self.e.num_fn, 0)

    def test_precision(self):
        # all correct (none) retrieved
        self.assertEqual(self.e.precision, 0)

    def test_recall(self):
        # all retrieved (none) are correct
        self.assertEqual(self.e.recall, 0)

    def test_fmeasure(self):
        # 2 * P * R / (P + R)
        self.assertEqual(self.e.fmeasure, 0)

    def test_accuracy(self):
        # (TP + TN) / (TP + FP + TN + FN)
        self.assertEqual(self.e.accuracy, 0)

    def test_errors(self):
        # array with errors
        self.assertTrue(np.array_equal(self.e.errors, np.zeros(0)))

    def test_mean_error(self):
        self.assertEqual(self.e.mean_error, 0)

    def test_std_error(self):
        self.assertEqual(self.e.std_error, 0)
