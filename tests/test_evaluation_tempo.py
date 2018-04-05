# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.evaluation.tempo module.

"""

from __future__ import absolute_import, division, print_function

import math
import unittest

from madmom.evaluation.tempo import *
from . import ANNOTATIONS_PATH, DETECTIONS_PATH

ANNOTATIONS = np.asarray([[87.5, 0.7], [175, 0.3]])
ANN_TEMPI = np.asarray([87.5, 175])
ANN_STRENGTHS = np.asarray([0.7, 0.3])
DETECTIONS = np.asarray([[176.47, 0.6], [117.65, 0.4]])
DET_TEMPI = np.asarray([176.47, 117.65])
DET_STRENGTHS = np.asarray([0.6, 0.4])


# test functions
class TestSortTempoFunction(unittest.TestCase):

    def test_sort(self):
        result = sort_tempo([[100, 0.8], [50, 0.2]])
        self.assertTrue(np.allclose(result, [[100, 0.8], [50, 0.2]]))
        result = sort_tempo([[50, 0.2], [100, 0.8]])
        self.assertTrue(np.allclose(result, [[100, 0.8], [50, 0.2]]))
        # tempo order of 50 and 100 bpm must be kept
        result = sort_tempo([[100, 0.2], [50, 0.2], [75, 0.6]])
        self.assertTrue(np.allclose(result,
                                    [[75, 0.6], [100, 0.2], [50, 0.2]]))

    def test_error(self):
        with self.assertRaises(ValueError):
            sort_tempo([120, 60])


class TestConstantsClass(unittest.TestCase):

    def test_types(self):
        self.assertIsInstance(TOLERANCE, float)
        self.assertIsInstance(DOUBLE, bool)
        self.assertIsInstance(TRIPLE, bool)

    def test_values(self):
        self.assertEqual(TOLERANCE, 0.04)
        self.assertEqual(DOUBLE, True)
        self.assertEqual(TRIPLE, True)


class TestTempoEvaluationFunction(unittest.TestCase):

    def test_types(self):
        scores = tempo_evaluation(DETECTIONS, ANNOTATIONS)
        self.assertIsInstance(scores, tuple)
        # detections / annotations must be correct type
        scores = tempo_evaluation([], [])
        self.assertIsInstance(scores, tuple)
        scores = tempo_evaluation({}, {})
        self.assertIsInstance(scores, tuple)
        # tolerance must be correct type
        scores = tempo_evaluation(DETECTIONS, ANNOTATIONS, int(1.2))
        self.assertIsInstance(scores, tuple)

    def test_errors(self):
        # detections / annotations must not be None
        with self.assertRaises(TypeError):
            tempo_evaluation(None, ANN_TEMPI)
        with self.assertRaises(TypeError):
            tempo_evaluation(DETECTIONS, None)
        # tolerance must be > 0
        with self.assertRaises(ValueError):
            tempo_evaluation(DETECTIONS, ANNOTATIONS, 0)
        # tolerance must be correct type
        with self.assertRaises(TypeError):
            tempo_evaluation(DETECTIONS, ANN_TEMPI, None)
        with self.assertRaises(TypeError):
            tempo_evaluation(DETECTIONS, ANN_TEMPI, [])
        with self.assertRaises(TypeError):
            tempo_evaluation(DETECTIONS, ANN_TEMPI, {})

    def test_values(self):
        # no tempi should return perfect score
        scores = tempo_evaluation([], [])
        self.assertEqual(scores, (1, True, True))
        # no detections should return worst score
        scores = tempo_evaluation([], ANNOTATIONS)
        self.assertEqual(scores, (0, False, False))
        # no annotations should return worst score
        scores = tempo_evaluation(DETECTIONS, np.zeros(0))
        self.assertEqual(scores, (0, False, False))
        # normal calculation
        scores = tempo_evaluation(DETECTIONS, ANNOTATIONS)
        self.assertEqual(scores, (0.3, True, False))
        # uniform strength calculation
        scores = tempo_evaluation(DETECTIONS, ANN_TEMPI)
        self.assertEqual(scores, (0.5, True, False))


# test evaluation class
class TestTempoEvaluationClass(unittest.TestCase):

    def test_types(self):
        e = TempoEvaluation(np.zeros(0), np.zeros(0))
        self.assertIsInstance(e.pscore, float)
        self.assertIsInstance(e.any, bool)
        self.assertIsInstance(e.all, bool)
        self.assertIsInstance(e.acc1, bool)
        self.assertIsInstance(e.acc2, bool)

    def test_conversion(self):
        # conversion from list should work
        e = TempoEvaluation([], [])
        self.assertIsInstance(e.pscore, float)
        self.assertIsInstance(e.any, bool)
        self.assertIsInstance(e.all, bool)
        self.assertIsInstance(e.acc1, bool)
        self.assertIsInstance(e.acc2, bool)

    def test_results_empty(self):
        e = TempoEvaluation([], [])
        self.assertEqual(e.pscore, 1)
        self.assertEqual(e.any, True)
        self.assertEqual(e.all, True)
        self.assertEqual(e.acc1, True)
        self.assertEqual(e.acc2, True)
        self.assertEqual(len(e), 1)

    def test_results(self):
        # two detections / annotations
        e = TempoEvaluation([120, 60], [[60, 0.7], [30, 0.3]])
        self.assertEqual(e.pscore, 0.7)
        self.assertEqual(e.any, True)
        self.assertEqual(e.all, False)
        # consider only first detection / annotation
        e = TempoEvaluation([120, 60], [[60, 0.7], [30, 0.3]], max_len=1)
        self.assertEqual(e.pscore, 0)
        self.assertEqual(e.any, False)
        self.assertEqual(e.all, False)
        # only det=120 and ann=60 should be evaluated for acc
        self.assertEqual(e.acc1, False)
        self.assertEqual(e.acc2, True)

        # two detections / annotations
        e = TempoEvaluation([120, 60], [[180, 0.7], [60, 0.3]])
        self.assertEqual(e.pscore, 0.3)
        self.assertEqual(e.any, True)
        self.assertEqual(e.all, False)
        # only det=120 and ann=180 should be evaluated for acc
        self.assertEqual(e.acc1, False)
        self.assertEqual(e.acc2, False)
        # consider only first detection / annotation
        e = TempoEvaluation([120, 60], [[180, 0.7], [60, 0.3]], max_len=1)
        self.assertEqual(e.pscore, 0)
        self.assertEqual(e.any, False)
        self.assertEqual(e.all, False)

        # two detections / annotations
        e = TempoEvaluation([120, 60], [[180, 0.7], [60, 0.3]])
        self.assertEqual(e.pscore, 0.3)
        self.assertEqual(e.any, True)
        self.assertEqual(e.all, False)
        # only det=120 and ann=180 should be evaluated for acc
        self.assertEqual(e.acc1, False)
        self.assertEqual(e.acc2, False)

        # two detections / annotations
        e = TempoEvaluation([120, 60], [[180, 0.3], [60, 0.7]])
        self.assertEqual(e.pscore, 0.7)
        self.assertEqual(e.any, True)
        self.assertEqual(e.all, False)
        # only det=120 and ann=60 should be evaluated for acc
        self.assertEqual(e.acc1, False)
        self.assertEqual(e.acc2, True)
        # consider only strongest detection / annotation (sort them)
        e = TempoEvaluation([60, 120], [[180, 0.3], [60, 0.7]], max_len=1)
        self.assertEqual(e.pscore, 1)
        self.assertEqual(e.any, True)
        self.assertEqual(e.all, True)
        self.assertEqual(e.acc1, True)
        self.assertEqual(e.acc2, True)
        # same, but do not sort them
        e = TempoEvaluation([60, 120], [[180, 0.3], [60, 0.7]], max_len=1,
                            sort=False)
        self.assertEqual(e.pscore, 0)
        self.assertEqual(e.any, False)
        self.assertEqual(e.all, False)
        self.assertEqual(e.acc1, False)
        self.assertEqual(e.acc2, True)

        # only 1 annotations
        e = TempoEvaluation([120, 60], [30, 1])
        self.assertEqual(e.pscore, 0)
        self.assertEqual(e.any, False)
        self.assertEqual(e.all, False)
        # only det=120 and ann=30 should be evaluated for acc
        self.assertEqual(e.acc1, False)
        self.assertEqual(e.acc2, False)

        # only 1 annotations
        e = TempoEvaluation([60, 120], [180, 1])
        self.assertEqual(e.pscore, 0)
        self.assertEqual(e.any, False)
        self.assertEqual(e.all, False)
        # only det=60 and ann=60 should be evaluated for acc
        self.assertEqual(e.acc1, False)
        self.assertEqual(e.acc2, True)

    def test_results_no_double(self):
        # only 1 annotations
        e = TempoEvaluation([60], [30], double=False)
        self.assertEqual(e.pscore, 0)
        self.assertEqual(e.any, False)
        self.assertEqual(e.all, False)
        self.assertEqual(e.acc1, False)
        self.assertEqual(e.acc2, False)
        # only 1 annotations
        e = TempoEvaluation([60], [180], double=False)
        self.assertEqual(e.pscore, 0)
        self.assertEqual(e.any, False)
        self.assertEqual(e.all, False)
        self.assertEqual(e.acc1, False)
        self.assertEqual(e.acc2, True)

    def test_results_no_triple(self):
        # only 1 annotations
        e = TempoEvaluation([60], [30], triple=False)
        self.assertEqual(e.pscore, 0)
        self.assertEqual(e.any, False)
        self.assertEqual(e.all, False)
        self.assertEqual(e.acc1, False)
        self.assertEqual(e.acc2, True)
        # only 1 annotations
        e = TempoEvaluation([60], [180], triple=False)
        self.assertEqual(e.pscore, 0)
        self.assertEqual(e.any, False)
        self.assertEqual(e.all, False)
        self.assertEqual(e.acc1, False)
        self.assertEqual(e.acc2, False)

    def test_tostring(self):
        print(TempoEvaluation([], []))


class TestMeanTempoEvaluationClass(unittest.TestCase):

    def test_types(self):
        e = TempoMeanEvaluation([])
        self.assertIsInstance(e.pscore, float)
        self.assertIsInstance(e.any, float)
        self.assertIsInstance(e.all, float)
        self.assertIsInstance(e.acc1, float)
        self.assertIsInstance(e.acc2, float)

    def test_results(self):
        # empty mean evaluation
        e = TempoMeanEvaluation([])
        self.assertTrue(math.isnan(e.pscore))
        self.assertTrue(math.isnan(e.any))
        self.assertTrue(math.isnan(e.all))
        self.assertTrue(math.isnan(e.acc1))
        self.assertTrue(math.isnan(e.acc2))
        self.assertEqual(len(e), 0)
        # mean evaluation with empty evaluation
        e1 = TempoEvaluation([], [])
        e = TempoMeanEvaluation([e1])
        self.assertEqual(e.pscore, 1)
        self.assertEqual(e.any, 1)
        self.assertEqual(e.all, 1)
        self.assertEqual(e.acc1, 1)
        self.assertEqual(e.acc2, 1)
        self.assertEqual(len(e), 1)
        # mean evaluation of empty and real evaluation
        e2 = TempoEvaluation([120, 60], [[60, 0.7], [30, 0.3]])
        e = TempoMeanEvaluation([e1, e2])
        self.assertEqual(e.pscore, (1 + .7) / 2.)
        self.assertEqual(e.any, (1 + 1) / 2.)
        self.assertEqual(e.all, (1 + 0) / 2.)
        self.assertEqual(e.acc1, (1 + 0) / 2.)
        self.assertEqual(e.acc2, (1 + 1.) / 2.)
        self.assertEqual(len(e), 2)

    def test_tostring(self):
        print(TempoMeanEvaluation([]))


class TestAddParserFunction(unittest.TestCase):

    def setUp(self):
        import argparse
        self.parser = argparse.ArgumentParser()
        sub_parser = self.parser.add_subparsers()
        self.sub_parser, self.group = add_parser(sub_parser)

    def test_args(self):
        args = self.parser.parse_args(['tempo', ANNOTATIONS_PATH,
                                       DETECTIONS_PATH])
        self.assertTrue(args.ann_dir is None)
        self.assertTrue(args.ann_suffix == '.bpm')
        self.assertTrue(args.det_dir is None)
        self.assertTrue(args.det_suffix == '.bpm.txt')
        self.assertTrue(args.double is True)
        self.assertTrue(args.eval == TempoEvaluation)
        self.assertTrue(args.files == [ANNOTATIONS_PATH, DETECTIONS_PATH])
        self.assertTrue(args.ignore_non_existing is False)
        self.assertTrue(args.mean_eval == TempoMeanEvaluation)
        # self.assertTrue(args.outfile == StringIO.StringIO)
        from madmom.evaluation import tostring
        self.assertTrue(args.output_formatter == tostring)
        self.assertTrue(args.quiet is False)
        self.assertTrue(args.sum_eval is None)
        self.assertTrue(args.tolerance == 0.04)
        self.assertTrue(args.triple is True)
        self.assertTrue(args.verbose == 0)
