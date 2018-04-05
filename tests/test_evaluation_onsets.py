# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.evaluation.onsets module.

"""

from __future__ import absolute_import, division, print_function

import math
import unittest

from madmom.evaluation.onsets import *
from . import ANNOTATIONS_PATH, DETECTIONS_PATH

# dummy detections/annotations
DETECTIONS = [0.99999999, 1.02999999, 1.45, 2.01, 2.02, 2.5, 3.025000001]
ANNOTATIONS = [1, 1.02, 1.5, 2.0, 2.03, 2.05, 2.5, 3]
# real detections/annotations
SAMPLE_DETECTIONS = [0.01, 0.085, 0.275, 0.445, 0.61, 0.795, 0.98, 1.115,
                     1.365, 1.475, 1.62, 1.795, 2.14, 2.33, 2.485, 2.665]
SAMPLE_ANNOTATIONS = [0.0943, 0.2844, 0.4528, 0.6160, 0.7630, 0.8025, 0.9847,
                      1.1233, 1.4820, 1.6276, 1.8032, 2.1486, 2.3351, 2.4918,
                      2.6710]


# loading function
class TestOnsetConstantsClass(unittest.TestCase):

    def test_types(self):
        self.assertIsInstance(WINDOW, float)

    def test_values(self):
        self.assertEqual(WINDOW, 0.025)


# test evaluation function
class TestOnsetEvaluationFunction(unittest.TestCase):

    def test_errors(self):
        # detections / annotations must not be None
        with self.assertRaises(TypeError):
            onset_evaluation(None, ANNOTATIONS)
        with self.assertRaises(TypeError):
            onset_evaluation(DETECTIONS, None)
        # tolerance must be > 0
        with self.assertRaises(ValueError):
            onset_evaluation(DETECTIONS, ANNOTATIONS, 0)
        # tolerance must be correct type
        with self.assertRaises(TypeError):
            onset_evaluation(DETECTIONS, ANNOTATIONS, None)
        with self.assertRaises(TypeError):
            onset_evaluation(DETECTIONS, ANNOTATIONS, [])
        with self.assertRaises(TypeError):
            onset_evaluation(DETECTIONS, ANNOTATIONS, {})

    def test_results(self):
        # default window
        tp, fp, tn, fn, errors = onset_evaluation(DETECTIONS, ANNOTATIONS)
        self.assertTrue(np.allclose(tp, [0.999999, 1.029999, 2.01, 2.02, 2.5]))
        self.assertTrue(np.allclose(fp, [1.45, 3.025000001]))
        self.assertTrue(np.allclose(tn, []))
        self.assertTrue(np.allclose(fn, [1.5, 2.05, 3.0]))
        self.assertTrue(np.allclose(errors, [-0.00000001, 0.00999999, 0.01,
                                             -0.01, 0]))
        # window = 0.01
        tp, fp, tn, fn, errors = onset_evaluation(DETECTIONS, ANNOTATIONS,
                                                  window=0.01)
        self.assertTrue(np.allclose(tp, [0.999999, 1.029999, 2.01, 2.02, 2.5]))
        self.assertTrue(np.allclose(fp, [1.45, 3.025000001]))
        self.assertTrue(np.allclose(tn, []))
        self.assertTrue(np.allclose(fn, [1.5, 2.05, 3.0]))
        self.assertTrue(np.allclose(errors, [-0.00000001, 0.00999999, 0.01,
                                             -0.01, 0]))
        # window = 0.04
        tp, fp, tn, fn, errors = onset_evaluation(DETECTIONS, ANNOTATIONS,
                                                  window=0.04)
        self.assertTrue(np.allclose(tp, [0.999999, 1.029999, 2.01, 2.02, 2.5,
                                         3.025000001]))
        self.assertTrue(np.allclose(fp, [1.45]))
        self.assertTrue(np.allclose(tn, []))
        self.assertTrue(np.allclose(fn, [1.5, 2.05]))
        self.assertTrue(np.allclose(errors, [-0.00000001, 0.00999999, 0.01,
                                             -0.01, 0, 0.025]))


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
        self.assertIsInstance(e.errors, np.ndarray)
        self.assertIsInstance(e.mean_error, float)
        self.assertIsInstance(e.std_error, float)

    def test_conversion(self):
        # conversion from list should work
        e = OnsetEvaluation([0], [0])
        self.assertIsInstance(e.tp, np.ndarray)
        self.assertIsInstance(e.fp, np.ndarray)
        self.assertIsInstance(e.tn, np.ndarray)
        self.assertIsInstance(e.fn, np.ndarray)
        self.assertIsInstance(e.errors, np.ndarray)
        # conversion from single values should work
        e = OnsetEvaluation(0, 0)
        self.assertIsInstance(e.tp, np.ndarray)
        self.assertIsInstance(e.fp, np.ndarray)
        self.assertIsInstance(e.tn, np.ndarray)
        self.assertIsInstance(e.fn, np.ndarray)
        self.assertIsInstance(e.errors, np.ndarray)

    def test_results(self):
        # empty detections / annotations
        e = OnsetEvaluation([], [])
        self.assertTrue(np.allclose(e.tp, []))
        self.assertTrue(np.allclose(e.fp, []))
        self.assertTrue(np.allclose(e.tn, []))
        self.assertTrue(np.allclose(e.fn, []))
        self.assertTrue(np.allclose(e.errors, []))
        self.assertEqual(e.num_tp, 0)
        self.assertEqual(e.num_fp, 0)
        self.assertEqual(e.num_tn, 0)
        self.assertEqual(e.num_fn, 0)
        self.assertEqual(e.precision, 1)
        self.assertEqual(e.recall, 1)
        self.assertEqual(e.fmeasure, 1)
        self.assertEqual(e.accuracy, 1)
        self.assertTrue(np.allclose(e.errors, []))
        self.assertTrue(math.isnan(e.mean_error))
        self.assertTrue(math.isnan(e.std_error))

        # real detections / annotations
        e = OnsetEvaluation(DETECTIONS, ANNOTATIONS)
        self.assertTrue(np.allclose(e.tp, [0.99999, 1.02999, 2.01, 2.02, 2.5]))
        self.assertTrue(np.allclose(e.fp, [1.45, 3.025000001]))
        self.assertTrue(np.allclose(e.tn, []))
        self.assertTrue(np.allclose(e.fn, [1.5, 2.05, 3.0]))
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
        self.assertTrue(np.allclose(e.errors, errors))
        mean = np.mean([0.99999999 - 1, 1.02999999 - 1.02, 2.01 - 2,
                        2.02 - 2.03, 2.5 - 2.5])
        self.assertEqual(e.mean_error, mean)
        std = np.std([0.99999999 - 1, 1.02999999 - 1.02, 2.01 - 2, 2.02 - 2.03,
                      2.5 - 2.5])
        self.assertEqual(e.std_error, std)

    def test_tostring(self):
        print(OnsetEvaluation([], []))


class TestOnsetSumEvaluationClass(unittest.TestCase):

    def test_types(self):
        e = OnsetSumEvaluation([])
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

    def test_results(self):
        # empty sum evaluation
        e = OnsetSumEvaluation([])
        self.assertEqual(e.num_tp, 0)
        self.assertEqual(e.num_fp, 0)
        self.assertEqual(e.num_tn, 0)
        self.assertEqual(e.num_fn, 0)
        self.assertEqual(e.precision, 1)
        self.assertEqual(e.recall, 1)
        self.assertEqual(e.fmeasure, 1)
        self.assertEqual(e.accuracy, 1)
        self.assertTrue(np.allclose(e.errors, []))
        self.assertTrue(math.isnan(e.mean_error))
        self.assertTrue(math.isnan(e.std_error))
        # sum evaluation of empty onset evaluation
        e1 = OnsetEvaluation([], [])
        e = OnsetSumEvaluation([e1])
        self.assertEqual(e.num_tp, 0)
        self.assertEqual(e.num_fp, 0)
        self.assertEqual(e.num_tn, 0)
        self.assertEqual(e.num_fn, 0)
        self.assertEqual(e.precision, 1)
        self.assertEqual(e.recall, 1)
        self.assertEqual(e.fmeasure, 1)
        self.assertEqual(e.accuracy, 1)
        self.assertTrue(np.allclose(e.errors, []))
        self.assertTrue(math.isnan(e.mean_error))
        self.assertTrue(math.isnan(e.std_error))
        # sum evaluation of empty and real onset evaluation
        e2 = OnsetEvaluation(DETECTIONS, ANNOTATIONS)
        e = OnsetSumEvaluation([e1, e2])
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
        # errors is just a concatenation of all errors, i.e. those of e2
        self.assertTrue(np.allclose(e.errors, e2.errors))
        # thus mean and std of errors is those of e2
        self.assertEqual(e.mean_error, e2.mean_error)
        self.assertEqual(e.std_error, e2.std_error)

    def test_tostring(self):
        print(OnsetSumEvaluation([]))


class TestOnsetMeanEvaluationClass(unittest.TestCase):

    def test_types(self):
        e = OnsetMeanEvaluation([])
        self.assertIsInstance(e.num_tp, float)
        self.assertIsInstance(e.num_fp, float)
        self.assertIsInstance(e.num_tn, float)
        self.assertIsInstance(e.num_fn, float)
        self.assertIsInstance(e.precision, float)
        self.assertIsInstance(e.recall, float)
        self.assertIsInstance(e.fmeasure, float)
        self.assertIsInstance(e.accuracy, float)
        self.assertIsInstance(e.errors, np.ndarray)
        self.assertIsInstance(e.mean_error, float)
        self.assertIsInstance(e.std_error, float)

    def test_results(self):
        # empty mean evaluation
        e = OnsetMeanEvaluation([])
        self.assertEqual(e.num_tp, 0)
        self.assertEqual(e.num_fp, 0)
        self.assertEqual(e.num_tn, 0)
        self.assertEqual(e.num_fn, 0)
        self.assertTrue(math.isnan(e.precision))
        self.assertTrue(math.isnan(e.recall))
        self.assertTrue(math.isnan(e.fmeasure))
        self.assertTrue(math.isnan(e.accuracy))
        self.assertTrue(np.allclose(e.errors, []))
        self.assertTrue(math.isnan(e.mean_error))
        self.assertTrue(math.isnan(e.std_error))

        # mean evaluation of empty onset evaluation
        e1 = OnsetEvaluation([], [])
        e = OnsetMeanEvaluation([e1])
        self.assertEqual(e.num_tp, 0)
        self.assertEqual(e.num_fp, 0)
        self.assertEqual(e.num_tn, 0)
        self.assertEqual(e.num_fn, 0)
        self.assertEqual(e.precision, 1)
        self.assertEqual(e.recall, 1)
        self.assertEqual(e.fmeasure, 1)
        self.assertEqual(e.accuracy, 1)
        self.assertTrue(np.allclose(e.errors, []))
        self.assertTrue(math.isnan(e.mean_error))
        self.assertTrue(math.isnan(e.std_error))

        # mean evaluation of empty and real onset evaluation
        e2 = OnsetEvaluation(DETECTIONS, ANNOTATIONS)
        e3 = OnsetEvaluation(ANNOTATIONS, DETECTIONS)
        e = OnsetMeanEvaluation([e1, e2, e3])
        self.assertTrue(np.allclose(
            e.num_tp, np.mean([e_.num_tp for e_ in [e1, e2, e3]])))
        self.assertTrue(np.allclose(
            e.num_fp, np.mean([e_.num_fp for e_ in [e1, e2, e3]])))
        self.assertTrue(np.allclose(
            e.num_tn, np.mean([e_.num_tn for e_ in [e1, e2, e3]])))
        self.assertTrue(np.allclose(
            e.num_fn, np.mean([e_.num_fn for e_ in [e1, e2, e3]])))
        self.assertTrue(np.allclose(
            e.precision, np.mean([e_.precision for e_ in [e1, e2, e3]])))
        self.assertTrue(np.allclose(
            e.recall, np.mean([e_.recall for e_ in [e1, e2, e3]])))
        self.assertTrue(np.allclose(
            e.fmeasure, np.mean([e_.fmeasure for e_ in [e1, e2, e3]])))
        self.assertTrue(np.allclose(
            e.accuracy, np.mean([e_.accuracy for e_ in [e1, e2, e3]])))
        # errors is just a concatenation of all errors
        # (inherited from SumOnsetEvaluation)
        self.assertTrue(np.allclose(
            e.errors, np.concatenate([e_.errors for e_ in [e2, e3]])))
        # mean and std errors are those of e2 and e3, since those of e1 are NaN
        self.assertEqual(e.mean_error,
                         np.mean([e_.mean_error for e_ in [e2, e3]]))
        self.assertEqual(e.std_error,
                         np.mean([e_.std_error for e_ in [e2, e3]]))

    def test_tostring(self):
        print(OnsetMeanEvaluation([]))


class TestAddParserFunction(unittest.TestCase):

    def setUp(self):
        import argparse
        self.parser = argparse.ArgumentParser()
        sub_parser = self.parser.add_subparsers()
        self.sub_parser, self.group = add_parser(sub_parser)

    def test_args(self):
        args = self.parser.parse_args(['onsets', ANNOTATIONS_PATH,
                                       DETECTIONS_PATH])
        self.assertTrue(args.ann_dir is None)
        self.assertTrue(args.ann_suffix == '.onsets')
        self.assertTrue(args.combine == 0.03)
        self.assertTrue(args.delay == 0.0)
        self.assertTrue(args.det_dir is None)
        self.assertTrue(args.det_suffix == '.onsets.txt')
        self.assertTrue(args.eval == OnsetEvaluation)
        self.assertTrue(args.files == [ANNOTATIONS_PATH, DETECTIONS_PATH])
        self.assertTrue(args.ignore_non_existing is False)
        self.assertTrue(args.mean_eval == OnsetMeanEvaluation)
        # self.assertTrue(args.outfile == StringIO.StringIO)
        from madmom.evaluation import tostring
        self.assertTrue(args.output_formatter == tostring)
        self.assertTrue(args.quiet is False)
        self.assertTrue(args.sum_eval == OnsetSumEvaluation)
        self.assertTrue(args.verbose == 0)
        self.assertTrue(args.window == 0.025)
