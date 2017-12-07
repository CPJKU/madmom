# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.evaluation.tempo module.

"""

from __future__ import absolute_import, division, print_function

import math
import unittest
from os.path import join as pj

from madmom.evaluation.tempo import *
from madmom.io import load_tempo
from . import ANNOTATIONS_PATH, DETECTIONS_PATH

ANNOTATIONS = np.asarray([[87.5, 0.7], [175, 0.3]])
ANN_TEMPI = np.asarray([87.5, 175])
ANN_STRENGTHS = np.asarray([0.7, 0.3])
DETECTIONS = np.asarray([[176.47, 0.6], [117.65, 0.4]])
DET_TEMPI = np.asarray([176.47, 117.65])
DET_STRENGTHS = np.asarray([0.6, 0.4])


# test functions
class TestLoadTempoFunction(unittest.TestCase):

    def test_load_tempo_from_file(self):
        annotations = load_tempo(pj(ANNOTATIONS_PATH, 'sample.tempo'))
        self.assertIsInstance(annotations, np.ndarray)

    def test_load_tempo_from_file_handle(self):
        file_handle = open(pj(ANNOTATIONS_PATH, 'sample.tempo'))
        annotations = load_tempo(file_handle)
        self.assertIsInstance(annotations, np.ndarray)
        file_handle.close()

    def test_load_tempo_annotations(self):
        annotations = load_tempo(pj(ANNOTATIONS_PATH, 'sample.tempo'))
        self.assertIsInstance(annotations, np.ndarray)
        self.assertEqual(annotations.shape, (2, 2))
        self.assertTrue(np.allclose(annotations, ANNOTATIONS))
        self.assertTrue(np.allclose(annotations[:, 0], ANN_TEMPI))
        self.assertTrue(np.allclose(annotations[:, 1], ANN_STRENGTHS))

    def test_lists(self):
        # simple lists
        result = load_tempo([100, 1])
        self.assertTrue(np.allclose(result, [[100, 1]]))
        result = load_tempo([100, 50])
        self.assertTrue(np.allclose(result, [[100, 0.5], [50, 0.5]]))
        # lists of lists
        result = load_tempo([[100], [50]])
        self.assertTrue(np.allclose(result, [[100, 0.5], [50, 0.5]]))
        result = load_tempo([[100, 0.6], [50, 0.6]])
        self.assertTrue(np.allclose(result, [[100, 0.6], [50, 0.6]]))
        # lists of tuples
        result = load_tempo([(100), (50)])
        self.assertTrue(np.allclose(result, [[100, 0.5], [50, 0.5]]))
        result = load_tempo([(100, 0.6), (50, 0.6)])
        self.assertTrue(np.allclose(result, [[100, 0.6], [50, 0.6]]))

    def test_arrays(self):
        result = load_tempo(np.asarray(100))
        self.assertTrue(np.allclose(result, [[100, 1]]))
        result = load_tempo(np.asarray((100, 50)))
        self.assertTrue(np.allclose(result, [[100, 0.5], [50, 0.5]]))
        result = load_tempo(np.asarray([100, 50]))
        self.assertTrue(np.allclose(result, [[100, 0.5], [50, 0.5]]))
        result = load_tempo(ANNOTATIONS)
        self.assertTrue(np.allclose(result, ANNOTATIONS))
        result = load_tempo(DETECTIONS)
        self.assertTrue(np.allclose(result, DETECTIONS))

    def test_missing_strength(self):
        # a strength of 1 should be added
        result = load_tempo([100])
        self.assertTrue(np.allclose(result, [[100, 1]]))
        # a strength of 0.5 should be added to both and order should be kept
        result = load_tempo([100, 50])
        self.assertTrue(np.allclose(result, [[100, 0.5], [50, 0.5]]))
        # a strength of 1/3 should be added to both and order should be kept
        result = load_tempo([50, 100, 75])
        self.assertTrue(np.allclose(result, [[50, 1. / 3],
                                             [100, 1. / 3],
                                             [75, 1. / 3]]))

    def test_relative_strengths(self):
        # the second strength should be added
        result = load_tempo([100, 50, 0.7])
        self.assertTrue(np.allclose(result, [[100, 0.7], [50, 0.3]]))
        # the strength could be somewhere
        result = load_tempo([100, 0.7, 50])
        self.assertTrue(np.allclose(result, [[100, 0.7], [50, 0.3]]))
        # the strength could be somewhere
        result = load_tempo([100, 0.5, 50, 0.3, 75])
        self.assertTrue(np.allclose(result, [[100, 0.5],
                                             [50, 0.3],
                                             [75, 0.2]]))

    def test_norm_strengths(self):
        result = load_tempo([100, 50, 0.4, 0.1], norm_strengths=True)
        self.assertTrue(np.allclose(result, [[100, 0.8], [50, 0.2]]))
        # the strength could be somewhere
        result = load_tempo([100, 0.4, 0.1, 50], norm_strengths=True)
        self.assertTrue(np.allclose(result, [[100, 0.8], [50, 0.2]]))
        # the strength could be somewhere
        result = load_tempo([100, 0.2, 50, 0.2, 0.1, 75], norm_strengths=True)
        self.assertTrue(np.allclose(result, [[100, 0.4],
                                             [50, 0.4],
                                             [75, 0.2]]))

    def test_sort(self):
        result = load_tempo([100, 50, 0.8, 0.2], sort=True)
        self.assertTrue(np.allclose(result, [[100, 0.8], [50, 0.2]]))
        result = load_tempo([50, 0.2, 100, 0.8], sort=True)
        self.assertTrue(np.allclose(result, [[100, 0.8], [50, 0.2]]))
        # third strength should be 0.6, tempo order of 50 and 100 must be kept
        result = load_tempo([100, 0.2, 50, 0.2, 75], sort=True)
        self.assertTrue(np.allclose(result, [[75, 0.6],
                                             [100, 0.2],
                                             [50, 0.2]]))

    def test_max_len(self):
        # positive values
        result = load_tempo([100, 50, 0.8, 0.2], max_len=1)
        self.assertTrue(np.allclose(result, [[100, 0.8]]))
        result = load_tempo([100, 50, 0.8, 0.2], max_len=2)
        self.assertTrue(np.allclose(result, [[100, 0.8], [50, 0.2]]))
        result = load_tempo([100, 50, 0.8, 0.2], max_len=3)
        self.assertTrue(np.allclose(result, [[100, 0.8], [50, 0.2]]))
        # third strength should be 0.6, tempo order of 50 and 100 must be kept
        result = load_tempo([100, 0.2, 50, 0.2, 75], sort=True, max_len=2)
        self.assertTrue(np.allclose(result, [[75, 0.6], [100, 0.2]]))
        # negative values are not supported
        with self.assertRaises(ValueError):
            load_tempo([100, 50, 0.8, 0.2], max_len=-1)


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
        e1 = TempoEvaluation([[]], [[]])
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
