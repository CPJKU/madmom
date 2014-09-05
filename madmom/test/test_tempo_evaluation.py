# encoding: utf-8
"""
This file contains tempo evaluation tests.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""
#pylint: skip-file

import unittest
import __builtin__

from madmom.evaluation.tempo import *
from . import DATA_PATH

TEMPI = np.asarray([120.1, 59])
STRENGTHS = np.asarray([0.6, 0.4])
DETECTIONS = np.asarray([60, 90])


# test functions
class TestLoadTempoFunction(unittest.TestCase):

    def test_load_tempo_from_file(self):
        annotations = load_tempo(DATA_PATH + 'file.tempo')
        self.assertIsInstance(annotations, tuple)

    def test_load_tempo_from_file_handle(self):
        file_handle = __builtin__.open(DATA_PATH + 'file.tempo', 'r')
        annotations = load_tempo(file_handle)
        self.assertIsInstance(annotations, tuple)
        file_handle.close()

    def test_load_tempo_annotations(self):
        annotations = load_tempo(DATA_PATH + 'file.tempo')
        self.assertIsInstance(annotations, tuple)
        print annotations
        self.assertTrue(np.allclose(annotations[0], TEMPI))
        self.assertTrue(np.allclose(annotations[1], STRENGTHS))


class TestTempoEvaluationFunction(unittest.TestCase):

    def test_types(self):
        scores = tempo_evaluation(DETECTIONS, TEMPI, STRENGTHS, 0.08)
        self.assertIsInstance(scores, tuple)
        # detections / annotations must be correct type
        scores = tempo_evaluation([], [], [], 0.08)
        self.assertIsInstance(scores, tuple)
        scores = tempo_evaluation({}, {}, {}, 0.08)
        self.assertIsInstance(scores, tuple)
        # cwe do not support normal non-empty lists
        with self.assertRaises(TypeError):
            tempo_evaluation(DETECTIONS.tolist(), TEMPI.tolist(),
                             STRENGTHS.tolist(), 0.08)
        # detections must not be None
        with self.assertRaises(TypeError):
            tempo_evaluation(None, TEMPI, STRENGTHS, 0.08)
        # annotations must not be None
        with self.assertRaises(TypeError):
            tempo_evaluation(DETECTIONS, None, STRENGTHS, 0.08)
        # strengths can be None
        scores = tempo_evaluation(DETECTIONS, TEMPI, None, 0.08)
        self.assertIsInstance(scores, tuple)
        # tolerance must be correct type
        scores = tempo_evaluation(DETECTIONS, TEMPI, STRENGTHS, int(1.2))
        self.assertIsInstance(scores, tuple)
        # various not supported versions
        with self.assertRaises(ValueError):
            tempo_evaluation(DETECTIONS, TEMPI, STRENGTHS, [])
        with self.assertRaises(ValueError):
            tempo_evaluation(DETECTIONS, TEMPI, [], 0.08)
        # TODO: what should happen if we supply a dictionary?
        # with self.assertRaises(TypeError):
        #     tempo_evaluation(DETECTIONS, TEMPI, STRENGTHS, {})

    def test_values(self):
        # tolerance must be > 0
        with self.assertRaises(ValueError):
            tempo_evaluation(DETECTIONS, TEMPI, STRENGTHS, 0)
        with self.assertRaises(ValueError):
            tempo_evaluation(DETECTIONS, TEMPI, STRENGTHS, None)
        # no tempi should return perfect score
        scores = tempo_evaluation([], [], [], 0.08)
        self.assertEqual(scores, (1, True, True))
        # no detections should return worst score
        scores = tempo_evaluation([], TEMPI, STRENGTHS, 0.08)
        self.assertEqual(scores, (0, False, False))
        # no annotations should return worst score
        scores = tempo_evaluation(DETECTIONS, np.zeros(0), STRENGTHS, 0.08)
        self.assertEqual(scores, (0, False, False))
        # normal calculation
        scores = tempo_evaluation(DETECTIONS, TEMPI, STRENGTHS, 0.08)
        self.assertEqual(scores, (0.4, True, False))
        # uniform strength calculation
        scores = tempo_evaluation(DETECTIONS, TEMPI, None, 0.08)
        self.assertEqual(scores, (0.5, True, False))


# test evaluation class
class TestTempoEvaluationClass(unittest.TestCase):

    def test_types(self):
        e = TempoEvaluation(np.zeros(0), np.zeros(0), np.zeros(0))
        self.assertIsInstance(e.pscore, float)
        self.assertIsInstance(e.any, bool)
        self.assertIsInstance(e.all, bool)
        self.assertIsInstance(e.acc1, bool)
        self.assertIsInstance(e.acc2, bool)

    def test_conversion(self):
        # conversion from list should work
        e = TempoEvaluation([], [], [])
        self.assertIsInstance(e.pscore, float)
        self.assertIsInstance(e.any, bool)
        self.assertIsInstance(e.all, bool)
        self.assertIsInstance(e.acc1, bool)
        self.assertIsInstance(e.acc2, bool)
        # others should fail
        self.assertRaises(TypeError, TempoEvaluation, float(0), float(0),
                          float(0))
        self.assertRaises(TypeError, TempoEvaluation, int(0), int(0), int(0))
        self.assertRaises(TypeError, TempoEvaluation, {}, {}, {})
        self.assertRaises(TypeError, TempoEvaluation, None, None, None)

    def test_results_empty(self):
        e = TempoEvaluation([], [], [])
        self.assertEqual(e.pscore, 1)
        self.assertEqual(e.any, True)
        self.assertEqual(e.all, True)
        self.assertEqual(e.acc1, True)
        self.assertEqual(e.acc2, True)

    def test_results(self):
        e = TempoEvaluation([120, 60], [60, 30], [0.7, 0.3])
        self.assertEqual(e.pscore, 0.7)
        self.assertEqual(e.any, True)
        self.assertEqual(e.all, False)
        self.assertEqual(e.acc1, True)
        self.assertEqual(e.acc2, True)


class TestMeanTempoEvaluationClass(unittest.TestCase):

    def test_types(self):
        e = MeanTempoEvaluation()
        self.assertIsInstance(e.pscore, float)
        self.assertIsInstance(e.any, float)
        self.assertIsInstance(e.all, float)
        self.assertIsInstance(e.acc1, float)
        self.assertIsInstance(e.acc2, float)

    def test_append(self):
        e = MeanTempoEvaluation()
        e.append(TempoEvaluation([], [], []))
        self.assertIsInstance(e, MeanTempoEvaluation)
        # appending something valid should not return anything
        self.assertEqual(e.append(TempoEvaluation([], [], [])), None)
        # appending something else should not work

    def test_append_types(self):
        e = MeanTempoEvaluation()
        e.append(TempoEvaluation([], [], []))
        self.assertIsInstance(e.pscore, float)
        self.assertIsInstance(e.any, float)
        self.assertIsInstance(e.all, float)
        self.assertIsInstance(e.acc1, float)
        self.assertIsInstance(e.acc2, float)

    def test_results_empty(self):
        e = MeanTempoEvaluation()
        e.append(TempoEvaluation([], [], []))
        self.assertEqual(e.pscore, 1.)
        self.assertEqual(e.any, 1.)
        self.assertEqual(e.all, 1.)
        self.assertEqual(e.acc1, 1.)
        self.assertEqual(e.acc2, 1.)

    def test_results(self):
        e = MeanTempoEvaluation()
        e.append(TempoEvaluation([], [], []))
        # result so far: 1, 1, 1
        e.append(TempoEvaluation([120, 60], [60, 30], [0.7, 0.3]))
        self.assertEqual(e.pscore, (1 + .7) / 2)
        self.assertEqual(e.any, 1)
        self.assertEqual(e.all, 0.5)
        self.assertEqual(e.acc1, 1.)
        self.assertEqual(e.acc2, 1.)
