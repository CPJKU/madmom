# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.evaluation.key module.

"""

from __future__ import absolute_import, division, print_function

import unittest
from os.path import join

from madmom.evaluation.key import *
from . import ANNOTATIONS_PATH, DETECTIONS_PATH


class TestKeyLabelToClassFunction(unittest.TestCase):

    def test_illegal_label(self):
        with self.assertRaises(ValueError):
            key_label_to_class('Z major')
        with self.assertRaises(ValueError):
            key_label_to_class('D mixolydian')
        with self.assertRaises(ValueError):
            key_label_to_class('wrongannoation')
        with self.assertRaises(ValueError):
            key_label_to_class('C:maj')

    def test_values(self):
        self.assertEqual(key_label_to_class('C major'), 0)
        self.assertEqual(key_label_to_class('C# major'), 1)
        self.assertEqual(key_label_to_class('D major'), 2)
        self.assertEqual(key_label_to_class('D# major'), 3)
        self.assertEqual(key_label_to_class('E major'), 4)
        self.assertEqual(key_label_to_class('F major'), 5)
        self.assertEqual(key_label_to_class('F# major'), 6)
        self.assertEqual(key_label_to_class('G major'), 7)
        self.assertEqual(key_label_to_class('G# major'), 8)
        self.assertEqual(key_label_to_class('A major'), 9)
        self.assertEqual(key_label_to_class('A# major'), 10)
        self.assertEqual(key_label_to_class('B major'), 11)
        self.assertEqual(key_label_to_class('C minor'), 0 + 12)
        self.assertEqual(key_label_to_class('C# minor'), 1 + 12)
        self.assertEqual(key_label_to_class('D minor'), 2 + 12)
        self.assertEqual(key_label_to_class('D# minor'), 3 + 12)
        self.assertEqual(key_label_to_class('E minor'), 4 + 12)
        self.assertEqual(key_label_to_class('F minor'), 5 + 12)
        self.assertEqual(key_label_to_class('F# minor'), 6 + 12)
        self.assertEqual(key_label_to_class('G minor'), 7 + 12)
        self.assertEqual(key_label_to_class('G# minor'), 8 + 12)
        self.assertEqual(key_label_to_class('A minor'), 9 + 12)
        self.assertEqual(key_label_to_class('A# minor'), 10 + 12)
        self.assertEqual(key_label_to_class('B minor'), 11 + 12)
        self.assertEqual(key_label_to_class('C major'),
                         key_label_to_class('C maj'))
        self.assertEqual(key_label_to_class('F minor'),
                         key_label_to_class('F min'))
        self.assertEqual(key_label_to_class('gb maj'),
                         key_label_to_class('F# major'))


class TestKeyClassToRootAndModeFunction(unittest.TestCase):

    def test_values(self):
        self.assertEqual(key_class_to_root_and_mode(0), (0, 0))
        self.assertEqual(key_class_to_root_and_mode(1), (1, 0))
        self.assertEqual(key_class_to_root_and_mode(2), (2, 0))
        self.assertEqual(key_class_to_root_and_mode(3), (3, 0))
        self.assertEqual(key_class_to_root_and_mode(4), (4, 0))
        self.assertEqual(key_class_to_root_and_mode(5), (5, 0))
        self.assertEqual(key_class_to_root_and_mode(6), (6, 0))
        self.assertEqual(key_class_to_root_and_mode(7), (7, 0))
        self.assertEqual(key_class_to_root_and_mode(8), (8, 0))
        self.assertEqual(key_class_to_root_and_mode(9), (9, 0))
        self.assertEqual(key_class_to_root_and_mode(10), (10, 0))
        self.assertEqual(key_class_to_root_and_mode(11), (11, 0))
        self.assertEqual(key_class_to_root_and_mode(12), (0, 1))
        self.assertEqual(key_class_to_root_and_mode(13), (1, 1))
        self.assertEqual(key_class_to_root_and_mode(14), (2, 1))
        self.assertEqual(key_class_to_root_and_mode(15), (3, 1))
        self.assertEqual(key_class_to_root_and_mode(16), (4, 1))
        self.assertEqual(key_class_to_root_and_mode(17), (5, 1))
        self.assertEqual(key_class_to_root_and_mode(18), (6, 1))
        self.assertEqual(key_class_to_root_and_mode(19), (7, 1))
        self.assertEqual(key_class_to_root_and_mode(20), (8, 1))
        self.assertEqual(key_class_to_root_and_mode(21), (9, 1))
        self.assertEqual(key_class_to_root_and_mode(22), (10, 1))
        self.assertEqual(key_class_to_root_and_mode(23), (11, 1))
        with self.assertRaises(ValueError):
            key_class_to_root_and_mode(-4)
        with self.assertRaises(ValueError):
            key_class_to_root_and_mode(24)


class TestErrorTypeFunction(unittest.TestCase):

    def _compare_error_types(self, correct, fifth_strict, fifth_lax, relative,
                             relative_of_fifth_up, relative_of_fifth_down,
                             parallel):
        for det_key in range(24):
            cat = error_type(det_key, correct)
            cat_st = error_type(det_key, correct, strict_fifth=True)
            cat_rf = error_type(det_key, correct, relative_of_fifth=True)
            cat_st_rf = error_type(det_key,
                                   correct,
                                   strict_fifth=True,
                                   relative_of_fifth=True)
            if det_key == correct:
                self.assertEqual(cat, 'correct')
                self.assertEqual(cat_st, cat)
                self.assertEqual(cat_rf, cat)
                self.assertEqual(cat_st_rf, cat)
            elif det_key == fifth_strict:
                self.assertEqual(cat, 'fifth')
                self.assertEqual(cat_st, cat)
                self.assertEqual(cat_rf, cat)
                self.assertEqual(cat_st_rf, cat)
            elif det_key == fifth_lax:
                self.assertEqual(cat, 'fifth')
                self.assertEqual(cat_st, 'other')
                self.assertEqual(cat_rf, 'fifth')
                self.assertEqual(cat_st_rf, 'other')
            elif det_key == relative:
                self.assertEqual(cat, 'relative')
                self.assertEqual(cat_st, cat)
                self.assertEqual(cat_rf, cat)
                self.assertEqual(cat_st_rf, cat)
            elif det_key == relative_of_fifth_down:
                self.assertEqual(cat, 'other')
                self.assertEqual(cat_st, cat)
                self.assertEqual(cat_rf, 'relative_of_fifth')
                self.assertEqual(cat_st_rf, cat)
            elif det_key == relative_of_fifth_up:
                self.assertEqual(cat, 'other')
                self.assertEqual(cat_st, cat)
                self.assertEqual(cat_rf, 'relative_of_fifth')
                self.assertEqual(cat_st_rf, 'relative_of_fifth')
            elif det_key == parallel:
                self.assertEqual(cat, 'parallel')
                self.assertEqual(cat_st, cat)
                self.assertEqual(cat_rf, cat)
                self.assertEqual(cat_st_rf, cat)

    def test_values(self):
        self._compare_error_types(
            correct=key_label_to_class('c maj'),
            fifth_strict=key_label_to_class('g maj'),
            fifth_lax=key_label_to_class('f maj'),
            relative=key_label_to_class('a min'),
            relative_of_fifth_up=key_label_to_class('e min'),
            relative_of_fifth_down=key_label_to_class('d min'),
            parallel=key_label_to_class('c min')
        )

        self._compare_error_types(
            correct=key_label_to_class('eb maj'),
            fifth_strict=key_label_to_class('bb maj'),
            fifth_lax=key_label_to_class('ab maj'),
            relative=key_label_to_class('c min'),
            relative_of_fifth_up=key_label_to_class('g min'),
            relative_of_fifth_down=key_label_to_class('f min'),
            parallel=key_label_to_class('eb min')
        )

        self._compare_error_types(
            correct=key_label_to_class('a min'),
            fifth_strict=key_label_to_class('e min'),
            fifth_lax=key_label_to_class('d min'),
            relative=key_label_to_class('c maj'),
            relative_of_fifth_up=key_label_to_class('g maj'),
            relative_of_fifth_down=key_label_to_class('f maj'),
            parallel=key_label_to_class('a maj')
        )

        self._compare_error_types(
            correct=key_label_to_class('b min'),
            fifth_strict=key_label_to_class('gb min'),
            fifth_lax=key_label_to_class('e min'),
            relative=key_label_to_class('d maj'),
            relative_of_fifth_up=key_label_to_class('a maj'),
            relative_of_fifth_down=key_label_to_class('g maj'),
            parallel=key_label_to_class('b maj')
        )


class TestKeyEvaluationClass(unittest.TestCase):

    def setUp(self):
        # this one should have a score of 1
        self.eval_correct = KeyEvaluation(
            load_key(join(DETECTIONS_PATH, 'dummy.correct.key.txt')),
            load_key(join(ANNOTATIONS_PATH, 'dummy.key')),
            name='eval_correct'
        )
        # this one should have a score of 0.5
        self.eval_fifth = KeyEvaluation(
            load_key(join(DETECTIONS_PATH, 'dummy.fifth.key.txt')),
            load_key(join(ANNOTATIONS_PATH, 'dummy.key')),
            name='eval_fifth'
        )

        # this one should have a score of 0.3
        self.eval_relative = KeyEvaluation(
            load_key(join(DETECTIONS_PATH, 'dummy.key.txt')),
            load_key(join(ANNOTATIONS_PATH, 'dummy.key')),
            name='eval_relative'
        )
        # this one should have a score of 0.2
        self.eval_parallel = KeyEvaluation(
            load_key(join(DETECTIONS_PATH, 'dummy.parallel.key.txt')),
            load_key(join(ANNOTATIONS_PATH, 'dummy.key')),
            name='eval_parallel'
        )
        # this one should have a score of 0.0
        self.eval_relative_of_fifth = KeyEvaluation(
            load_key(join(DETECTIONS_PATH, 'dummy.relative_of_fifth.key.txt')),
            load_key(join(ANNOTATIONS_PATH, 'dummy.key')),
            relative_of_fifth=True,
            name='eval_relative_of_fifth'
        )
        # this one should have a score of 0.0
        self.eval_other = KeyEvaluation(
            load_key(join(DETECTIONS_PATH, 'dummy.other.key.txt')),
            load_key(join(ANNOTATIONS_PATH, 'dummy.key')),
            name='eval_other'
        )

    def test_init(self):
        self.assertTrue(self.eval_relative.name == 'eval_relative')
        self.assertTrue(self.eval_relative.detection, 9)
        self.assertTrue(self.eval_relative.annotation, 18)

    def test_results(self):
        # Correct
        self.assertEqual(self.eval_correct.error_category, 'correct')
        self.assertEqual(self.eval_correct.score, 1.0)
        # Fifth
        self.assertEqual(self.eval_fifth.error_category, 'fifth')
        self.assertEqual(self.eval_fifth.score, 0.5)
        # Relative
        self.assertEqual(self.eval_relative.error_category, 'relative')
        self.assertEqual(self.eval_relative.score, 0.3)
        # Relative of Fifth
        self.assertEqual(self.eval_relative_of_fifth.error_category,
                         'relative_of_fifth')
        self.assertEqual(self.eval_relative_of_fifth.score, 0.0)
        # Parallel
        self.assertEqual(self.eval_parallel.error_category, 'parallel')
        self.assertEqual(self.eval_parallel.score, 0.2)
        # Other
        self.assertEqual(self.eval_other.error_category, 'other')
        self.assertEqual(self.eval_other.score, 0.0)


class TestKeyMeanEvaluation(unittest.TestCase):

    def setUp(self):
        # this one should have a score of 1
        self.eval_correct = KeyEvaluation(
            load_key(join(DETECTIONS_PATH, 'dummy.correct.key.txt')),
            load_key(join(ANNOTATIONS_PATH, 'dummy.key')),
            name='eval_correct'
        )
        # this one should have a score of 0.2
        self.eval_parallel = KeyEvaluation(
            load_key(join(DETECTIONS_PATH, 'dummy.parallel.key.txt')),
            load_key(join(ANNOTATIONS_PATH, 'dummy.key')),
            name='eval_parallel'
        )
        # this one should have a score of 0.0
        self.eval_relative = KeyEvaluation(
            load_key(join(DETECTIONS_PATH, 'dummy.key.txt')),
            load_key(join(ANNOTATIONS_PATH, 'dummy.key')),
            name='eval_relative'
        )
        # this one should have a score of 0.0
        self.eval_other = KeyEvaluation(
            load_key(join(DETECTIONS_PATH, 'dummy.other.key.txt')),
            load_key(join(ANNOTATIONS_PATH, 'dummy.key')),
            name='eval_other'
        )
        # this one has has the same key BUT a different set of error scores
        self.eval_different_scores = KeyEvaluation(
            load_key(join(DETECTIONS_PATH, 'dummy.correct.key.txt')),
            load_key(join(ANNOTATIONS_PATH, 'dummy.key')),
            name='eval_correct_different_scores'
        )
        self.eval_different_scores.error_scores = {'correct': 0.5}

        self.eval_correct_w_rel_of_fifth = KeyEvaluation(
            load_key(join(DETECTIONS_PATH, 'dummy.correct.key.txt')),
            load_key(join(ANNOTATIONS_PATH, 'dummy.key')),
            relative_of_fifth=True,
            name='eval_correct_w_rel_of_fifth'
        )

        self.eval_rel_of_fifth = KeyEvaluation(
            load_key(join(DETECTIONS_PATH, 'dummy.relative_of_fifth.key.txt')),
            load_key(join(ANNOTATIONS_PATH, 'dummy.key')),
            relative_of_fifth=True,
            name='eval_rel_of_fifth'
        )

    def test_check_key_eval_objects(self):
        evals = [self.eval_correct, self.eval_parallel,
                 self.eval_different_scores, self.eval_other]
        with self.assertRaises(ValueError):
            KeyMeanEvaluation(evals)

        evals = [self.eval_correct, self.eval_parallel,
                 self.eval_rel_of_fifth]
        with self.assertRaises(ValueError):
            KeyMeanEvaluation(evals)

    def test_empty_eval_list(self):
        with self.assertRaises(ValueError):
            KeyMeanEvaluation([])

    def test_mean_results(self):
        evals = [self.eval_correct, self.eval_parallel, self.eval_relative,
                 self.eval_other]

        mean_eval = KeyMeanEvaluation(evals)

        self.assertAlmostEqual(mean_eval.correct, 1.0 / len(evals))
        self.assertAlmostEqual(mean_eval.fifth, 0.0)
        self.assertAlmostEqual(mean_eval.relative, 1.0 / len(evals))
        self.assertAlmostEqual(mean_eval.relative_of_fifth, 0.0)
        self.assertAlmostEqual(mean_eval.parallel, 1.0 / len(evals))
        self.assertAlmostEqual(mean_eval.other, 1.0 / len(evals))
        self.assertAlmostEqual(mean_eval.weighted, 0.375)
        self.assertEqual(mean_eval.tostring(),
                         'mean for 4 files\n  '
                         'Weighted: 0.375  '
                         'Correct: 0.250  '
                         'Fifth: 0.000  '
                         'Relative: 0.250  '
                         'Parallel: 0.250  '
                         'Other: 0.250')

    def test_mean_results_w_rel_of_fifth(self):
        evals = [self.eval_correct_w_rel_of_fifth,
                 self.eval_rel_of_fifth]

        mean_eval = KeyMeanEvaluation(evals, name='Jean-Guy')

        self.assertAlmostEqual(mean_eval.correct, 1.0 / len(evals))
        self.assertAlmostEqual(mean_eval.fifth, 0.0)
        self.assertAlmostEqual(mean_eval.relative, 0.0)
        self.assertAlmostEqual(mean_eval.relative_of_fifth, 1.0 / len(evals))
        self.assertAlmostEqual(mean_eval.parallel, 0.0 / len(evals))
        self.assertAlmostEqual(mean_eval.other, 0.0 / len(evals))
        self.assertAlmostEqual(mean_eval.weighted, 0.5)
        self.assertEqual(mean_eval.tostring(),
                         'Jean-Guy\n  '
                         'Weighted: 0.500  '
                         'Correct: 0.500  '
                         'Fifth: 0.000  '
                         'Relative: 0.000  '
                         'Relative of fifth: 0.500  '
                         'Parallel: 0.000  '
                         'Other: 0.000')


class TestAddParserFunction(unittest.TestCase):

    def setUp(self):
        import argparse
        self.parser = argparse.ArgumentParser()
        sub_parser = self.parser.add_subparsers()
        self.sub_parser = add_parser(sub_parser)

    def test_args(self):
        args = self.parser.parse_args(['key', ANNOTATIONS_PATH,
                                       DETECTIONS_PATH])
        self.assertTrue(args.ann_dir is None)
        self.assertTrue(args.ann_suffix == '.key')
        self.assertTrue(args.det_dir is None)
        self.assertTrue(args.det_suffix == '.key.txt')
        self.assertTrue(args.eval == KeyEvaluation)
        self.assertTrue(args.files == [ANNOTATIONS_PATH, DETECTIONS_PATH])
        self.assertTrue(args.mean_eval == KeyMeanEvaluation)
        self.assertTrue(args.sum_eval is None)
        from madmom.evaluation import tostring
        self.assertTrue(args.output_formatter == tostring)
