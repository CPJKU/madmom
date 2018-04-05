# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.evaluation.notes module.

"""

from __future__ import absolute_import, division, print_function

import math
import unittest

from madmom.evaluation.notes import *
from . import ANNOTATIONS_PATH, DETECTIONS_PATH

DETECTIONS = np.asarray([[0.147, 72],  # TP
                         [0.147, 80],  # FP
                         [0.147, 60],  # FP, octave error
                         #  [1.567, 41], FN
                         [2.540, 77],  # 14ms too late
                         [2.520, 60],  # 29ms too early
                         #  [2.563, 65], FN
                         #  [2.577, 57], FN + FP, 1 note off
                         [3.368, 75],  # 1ms too early
                         [3.449, 43]])
ANNOTATIONS = np.asarray([[0.147, 72, 3.323, 63],
                          [1.567, 41, 0.223, 29],
                          [2.526, 77, 0.930, 72],
                          [2.549, 60, 0.211, 28],
                          [2.563, 65, 0.202, 34],
                          [2.577, 56, 0.234, 31],
                          [3.369, 75, 0.780, 64],
                          [3.449, 43, 0.272, 35]])


# test functions
class TestRemoveDuplicateNotesFunction(unittest.TestCase):

    def test_types(self):
        notes = remove_duplicate_notes(ANNOTATIONS)
        self.assertIsInstance(notes, np.ndarray)

    def test_results(self):
        ann = np.vstack((ANNOTATIONS, ANNOTATIONS[2]))
        notes = remove_duplicate_notes(ann)
        self.assertTrue(np.allclose(notes, ANNOTATIONS))


class TestNoteConstantsClass(unittest.TestCase):

    def test_types(self):
        self.assertIsInstance(WINDOW, float)

    def test_values(self):
        self.assertEqual(WINDOW, 0.025)


class TestNoteOnsetEvaluationFunction(unittest.TestCase):

    def test_types(self):
        tp, fp, tn, fn, errors = note_onset_evaluation(DETECTIONS, ANNOTATIONS,
                                                       0.025)
        self.assertIsInstance(tp, np.ndarray)
        self.assertIsInstance(fp, np.ndarray)
        self.assertIsInstance(tn, np.ndarray)
        self.assertIsInstance(fn, np.ndarray)
        self.assertIsInstance(errors, np.ndarray)
        tp, fp, tn, fn, errors = note_onset_evaluation([[]], [[]], 0.025)
        self.assertIsInstance(tp, np.ndarray)
        self.assertIsInstance(fp, np.ndarray)
        self.assertIsInstance(tn, np.ndarray)
        self.assertIsInstance(fn, np.ndarray)
        self.assertIsInstance(errors, np.ndarray)
        with self.assertRaises(ValueError):
            note_onset_evaluation([], [], 0.025)

    def test_results(self):
        # empty detections and annotations
        tp, fp, tn, fn, errors = note_onset_evaluation([[]], [[]], 0.02)
        self.assertTrue(np.allclose(tp, np.zeros((0, 2))))
        self.assertTrue(np.allclose(fp, np.zeros((0, 2))))
        self.assertTrue(np.allclose(tn, np.zeros((0, 2))))
        self.assertTrue(np.allclose(fn, np.zeros((0, 2))))
        self.assertTrue(np.allclose(errors, np.zeros((0, 2))))
        # empty annotations
        tp, fp, tn, fn, errors = note_onset_evaluation(DETECTIONS, [[]], 0.02)
        self.assertTrue(np.allclose(tp, np.zeros((0, 2))))
        self.assertTrue(np.allclose(fp, DETECTIONS))
        self.assertTrue(np.allclose(tn, np.zeros((0, 2))))
        self.assertTrue(np.allclose(fn, np.zeros((0, 2))))
        self.assertTrue(np.allclose(errors, np.zeros((0, 2))))
        # empty detections
        tp, fp, tn, fn, errors = note_onset_evaluation([[]], ANNOTATIONS, 0.02)
        self.assertTrue(np.allclose(tp, np.zeros((0, 2))))
        self.assertTrue(np.allclose(fp, np.zeros((0, 2))))
        self.assertTrue(np.allclose(tn, np.zeros((0, 2))))
        self.assertTrue(np.allclose(fn, ANNOTATIONS))
        self.assertTrue(np.allclose(errors, np.zeros((0, 2))))
        # window = 0.01
        tp, fp, tn, fn, errors = note_onset_evaluation(DETECTIONS, ANNOTATIONS,
                                                       0.01)
        self.assertTrue(np.allclose(tp, [[0.147, 72], [3.368, 75],
                                         [3.449, 43]]))
        self.assertTrue(np.allclose(fp, [[0.147, 60], [0.147, 80],
                                         [2.520, 60], [2.540, 77]]))
        self.assertTrue(np.allclose(tn, np.zeros((0, 2))))
        self.assertTrue(np.allclose(fn, [[1.567, 41], [2.526, 77], [2.549, 60],
                                         [2.563, 65], [2.577, 56]]))
        self.assertTrue(np.allclose(errors, [[0, 72], [-0.001, 75], [0, 43]]))
        # default window (= 0.025)
        tp, fp, tn, fn, errors = note_onset_evaluation(DETECTIONS, ANNOTATIONS)
        self.assertTrue(np.allclose(tp, [[0.147, 72], [2.540, 77],
                                         [3.368, 75], [3.449, 43]]))
        self.assertTrue(np.allclose(fp, [[0.147, 60], [0.147, 80],
                                         [2.520, 60]]))
        self.assertTrue(np.allclose(tn, np.zeros((0, 2))))
        self.assertTrue(np.allclose(fn, [[1.567, 41], [2.549, 60],
                                         [2.563, 65], [2.577, 56]]))
        self.assertTrue(np.allclose(errors, [[0, 72], [0.014, 77],
                                             [-0.001, 75], [0, 43]]))
        # window = 0.03
        tp, fp, tn, fn, errors = note_onset_evaluation(DETECTIONS, ANNOTATIONS,
                                                       0.03)
        self.assertTrue(np.allclose(tp, [[0.147, 72], [2.520, 60], [2.540, 77],
                                         [3.368, 75], [3.449, 43]]))
        self.assertTrue(np.allclose(fp, [[0.147, 60], [0.147, 80]]))
        self.assertTrue(np.allclose(tn, np.zeros((0, 2))))
        self.assertTrue(np.allclose(fn, [[1.567, 41], [2.563, 65],
                                         [2.577, 56]]))
        self.assertTrue(np.allclose(errors, [[0, 72], [-0.029, 60],
                                             [0.014, 77], [-0.001, 75],
                                             [0, 43]]))


# test evaluation class
class TestNoteEvaluationClass(unittest.TestCase):

    def test_types(self):
        e = NoteEvaluation(DETECTIONS, ANNOTATIONS)
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
        # conversion from list of lists should work
        e = NoteEvaluation([[0, 0]], [[0, 0]])
        self.assertIsInstance(e.tp, np.ndarray)
        self.assertIsInstance(e.fp, np.ndarray)
        self.assertIsInstance(e.tn, np.ndarray)
        self.assertIsInstance(e.fn, np.ndarray)

    def test_results(self):
        # empty detections / annotations
        e = NoteEvaluation([[]], [[]])
        self.assertTrue(np.allclose(e.tp, np.zeros((0, 2))))
        self.assertTrue(np.allclose(e.fp, np.zeros((0, 2))))
        self.assertTrue(np.allclose(e.tn, np.zeros((0, 2))))
        self.assertTrue(np.allclose(e.fn, np.zeros((0, 2))))
        self.assertEqual(e.num_tp, 0)
        self.assertEqual(e.num_fp, 0)
        self.assertEqual(e.num_tn, 0)
        self.assertEqual(e.num_fn, 0)
        self.assertEqual(e.precision, 1)
        self.assertEqual(e.recall, 1)
        self.assertEqual(e.fmeasure, 1)
        self.assertEqual(e.accuracy, 1)
        self.assertTrue(np.allclose(e.errors, np.zeros((0, 2))))
        self.assertTrue(math.isnan(e.mean_error))
        self.assertTrue(math.isnan(e.std_error))

        # real detections / annotations
        e = NoteEvaluation(DETECTIONS, ANNOTATIONS)
        self.assertTrue(np.allclose(e.tp, [[0.147, 72], [2.540, 77],
                                           [3.368, 75], [3.449, 43]]))
        self.assertTrue(np.allclose(e.fp, [[0.147, 60], [0.147, 80],
                                           [2.520, 60]]))
        self.assertTrue(np.allclose(e.tn, np.zeros((0, 2))))
        self.assertTrue(np.allclose(e.fn, [[1.567, 41], [2.549, 60],
                                           [2.563, 65], [2.577, 56]]))
        self.assertEqual(e.num_tp, 4)
        self.assertEqual(e.num_fp, 3)
        self.assertEqual(e.num_tn, 0)
        self.assertEqual(e.num_fn, 4)
        self.assertEqual(e.precision, 4. / 7.)
        self.assertEqual(e.recall, 4. / 8.)
        f = 2 * (4. / 7.) * (4. / 8.) / ((4. / 7.) + (4. / 8.))
        self.assertEqual(e.fmeasure, f)
        self.assertEqual(e.accuracy, (4. + 0) / (4 + 3 + 0 + 4))
        # errors
        # tp =  [[0.147, 72], [2.540, 77], [3.368, 75], [3.449, 43]]
        # ann = [[0.147, 72], [2.526, 77], [3.369, 75], [3.449, 43]]
        # err = [[0.   , 72], [0.014, 77], [-0.001, 75], [0.   , 43]]
        errors = np.asarray([[0., 72], [0.014, 77], [-0.001, 75], [0., 43]])
        self.assertTrue(np.allclose(e.errors, errors))
        self.assertTrue(np.allclose(e.mean_error,
                                    np.mean([0, 0.014, -0.001, 0])))
        self.assertTrue(np.allclose(e.std_error,
                                    np.std([0, 0.014, -0.001, 0])))

    def test_tostring(self):
        print(NoteEvaluation([], []))


class TestNoteSumEvaluationClass(unittest.TestCase):

    def test_types(self):
        e = NoteSumEvaluation([])
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
        e = NoteSumEvaluation([])
        self.assertEqual(e.num_tp, 0)
        self.assertEqual(e.num_fp, 0)
        self.assertEqual(e.num_tn, 0)
        self.assertEqual(e.num_fn, 0)
        self.assertEqual(e.precision, 1)
        self.assertEqual(e.recall, 1)
        self.assertEqual(e.fmeasure, 1)
        self.assertEqual(e.accuracy, 1)
        self.assertTrue(np.allclose(e.errors, np.zeros((0, 2))))
        self.assertTrue(math.isnan(e.mean_error))
        self.assertTrue(math.isnan(e.std_error))
        # sum evaluation of empty note evaluation
        e1 = NoteEvaluation([], [])
        e = NoteSumEvaluation([e1])
        self.assertEqual(e.num_tp, 0)
        self.assertEqual(e.num_fp, 0)
        self.assertEqual(e.num_tn, 0)
        self.assertEqual(e.num_fn, 0)
        self.assertEqual(e.precision, 1)
        self.assertEqual(e.recall, 1)
        self.assertEqual(e.fmeasure, 1)
        self.assertEqual(e.accuracy, 1)
        self.assertTrue(np.allclose(e.errors, np.zeros((0, 2))))
        self.assertTrue(math.isnan(e.mean_error))
        self.assertTrue(math.isnan(e.std_error))
        # sum evaluation of empty and real onset evaluation
        e2 = NoteEvaluation(DETECTIONS, ANNOTATIONS)
        e = NoteSumEvaluation([e1, e2])
        # everything must be the same as e2, since e1 was empty and thus did
        # not ad anything to the sum evaluation
        self.assertEqual(e.num_tp, e2.num_tp)
        self.assertEqual(e.num_fp, e2.num_fp)
        self.assertEqual(e.num_tn, e2.num_tn)
        self.assertEqual(e.num_fn, e2.num_fn)
        self.assertEqual(e.precision, e2.precision)
        self.assertEqual(e.recall, e2.recall)
        self.assertEqual(e.fmeasure, e2.fmeasure)
        self.assertEqual(e.accuracy, e2.accuracy)
        self.assertTrue(np.allclose(e.errors, e2.errors))
        self.assertEqual(e.mean_error, e2.mean_error)
        self.assertEqual(e.std_error, e2.std_error)

    def test_tostring(self):
        print(NoteSumEvaluation([]))


class TestNoteMeanEvaluationClass(unittest.TestCase):

    def test_types(self):
        e = NoteMeanEvaluation([])
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
        e = NoteMeanEvaluation([])
        self.assertEqual(e.num_tp, 0)
        self.assertEqual(e.num_fp, 0)
        self.assertEqual(e.num_tn, 0)
        self.assertEqual(e.num_fn, 0)
        self.assertTrue(math.isnan(e.precision))
        self.assertTrue(math.isnan(e.recall))
        self.assertTrue(math.isnan(e.fmeasure))
        self.assertTrue(math.isnan(e.accuracy))
        self.assertTrue(np.allclose(e.errors, np.zeros((0, 2))))
        self.assertTrue(math.isnan(e.mean_error))
        self.assertTrue(math.isnan(e.std_error))

        # mean evaluation of empty note evaluation
        e1 = NoteEvaluation([], [])
        e = NoteMeanEvaluation([e1])
        self.assertEqual(e.num_tp, 0)
        self.assertEqual(e.num_fp, 0)
        self.assertEqual(e.num_tn, 0)
        self.assertEqual(e.num_fn, 0)
        self.assertEqual(e.precision, 1)
        self.assertEqual(e.recall, 1)
        self.assertEqual(e.fmeasure, 1)
        self.assertEqual(e.accuracy, 1)
        self.assertTrue(np.allclose(e.errors, np.zeros((0, 2))))
        self.assertTrue(math.isnan(e.mean_error))
        self.assertTrue(math.isnan(e.std_error))

        # mean evaluation of empty and real note evaluation
        e2 = NoteEvaluation(DETECTIONS, ANNOTATIONS)
        e = NoteMeanEvaluation([e1, e2])
        self.assertTrue(np.allclose(
            e.num_tp, np.mean([e_.num_tp for e_ in [e1, e2]])))
        self.assertTrue(np.allclose(
            e.num_fp, np.mean([e_.num_fp for e_ in [e1, e2]])))
        self.assertTrue(np.allclose(
            e.num_tn, np.mean([e_.num_tn for e_ in [e1, e2]])))
        self.assertTrue(np.allclose(
            e.num_fn, np.mean([e_.num_fn for e_ in [e1, e2]])))
        self.assertTrue(np.allclose(
            e.precision, np.mean([e_.precision for e_ in [e1, e2]])))
        self.assertTrue(np.allclose(
            e.recall, np.mean([e_.recall for e_ in [e1, e2]])))
        self.assertTrue(np.allclose(
            e.fmeasure, np.mean([e_.fmeasure for e_ in [e1, e2]])))
        self.assertTrue(np.allclose(
            e.accuracy, np.mean([e_.accuracy for e_ in [e1, e2]])))
        self.assertTrue(np.allclose(
            e.errors, np.concatenate([e_.errors for e_ in [e1, e2]])))
        # mean and std errors are those of e2, since those of e1 are NaN
        self.assertEqual(e.mean_error, e2.mean_error)
        self.assertEqual(e.std_error, e2.std_error)

    def test_tostring(self):
        print(NoteMeanEvaluation([]))


class TestAddParserFunction(unittest.TestCase):

    def setUp(self):
        import argparse
        self.parser = argparse.ArgumentParser()
        sub_parser = self.parser.add_subparsers()
        self.sub_parser, self.group = add_parser(sub_parser)

    def test_args(self):
        args = self.parser.parse_args(['notes', ANNOTATIONS_PATH,
                                       DETECTIONS_PATH])
        self.assertTrue(args.ann_dir is None)
        self.assertTrue(args.ann_suffix == '.notes')
        self.assertTrue(args.det_dir is None)
        self.assertTrue(args.det_suffix == '.notes.txt')
        self.assertTrue(args.eval == NoteEvaluation)
        self.assertTrue(args.files == [ANNOTATIONS_PATH, DETECTIONS_PATH])
        self.assertTrue(args.ignore_non_existing is False)
        self.assertTrue(args.mean_eval == NoteMeanEvaluation)
        # self.assertTrue(args.outfile == StringIO.StringIO)
        from madmom.evaluation import tostring
        self.assertTrue(args.output_formatter == tostring)
        self.assertTrue(args.quiet is False)
        self.assertTrue(args.sum_eval == NoteSumEvaluation)
        self.assertTrue(args.verbose == 0)
        self.assertTrue(args.window == 0.025)
