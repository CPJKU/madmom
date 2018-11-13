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


class TestErrorTypeFunction(unittest.TestCase):

    def _compare_scores(self, correct, fifth_strict, fifth_lax, relative,
                        parallel):
        for det_key in range(24):
            score, cat = error_type(det_key, correct)
            score_st, cat_st = error_type(det_key, correct, strict_fifth=True)
            if det_key == correct:
                self.assertEqual(cat, 'correct')
                self.assertEqual(score, 1.0)
                self.assertEqual(cat_st, cat)
                self.assertEqual(score_st, score)
            if det_key == fifth_strict:
                self.assertEqual(cat, 'fifth')
                self.assertEqual(score, 0.5)
                self.assertEqual(cat_st, cat)
                self.assertEqual(score_st, score)
            if det_key == fifth_lax:
                self.assertEqual(cat, 'fifth')
                self.assertEqual(score, 0.5)
                self.assertEqual(cat_st, 'other')
                self.assertEqual(score_st, 0.0)
            if det_key == relative:
                self.assertEqual(cat, 'relative')
                self.assertEqual(score, 0.3)
                self.assertEqual(cat_st, cat)
                self.assertEqual(score_st, score)
            if det_key == parallel:
                self.assertEqual(cat, 'parallel')
                self.assertEqual(score, 0.2)
                self.assertEqual(cat_st, cat)
                self.assertEqual(score_st, score)

    def test_values(self):
        self._compare_scores(
            correct=key_label_to_class('c maj'),
            fifth_strict=key_label_to_class('g maj'),
            fifth_lax=key_label_to_class('f maj'),
            relative=key_label_to_class('a min'),
            parallel=key_label_to_class('c min')
        )

        self._compare_scores(
            correct=key_label_to_class('eb maj'),
            fifth_strict=key_label_to_class('bb maj'),
            fifth_lax=key_label_to_class('ab maj'),
            relative=key_label_to_class('c min'),
            parallel=key_label_to_class('eb min')
        )

        self._compare_scores(
            correct=key_label_to_class('a min'),
            fifth_strict=key_label_to_class('e min'),
            fifth_lax=key_label_to_class('d min'),
            relative=key_label_to_class('c maj'),
            parallel=key_label_to_class('a maj')
        )

        self._compare_scores(
            correct=key_label_to_class('b min'),
            fifth_strict=key_label_to_class('gb min'),
            fifth_lax=key_label_to_class('e min'),
            relative=key_label_to_class('d maj'),
            parallel=key_label_to_class('b maj')
        )


class TestKeyEvaluationClass(unittest.TestCase):

    def setUp(self):
        self.eval = KeyEvaluation(
            load_key(join(DETECTIONS_PATH, 'dummy.key.txt')),
            load_key(join(ANNOTATIONS_PATH, 'dummy.key')),
            name='TestEval'
        )

    def test_init(self):
        self.assertTrue(self.eval.name == 'TestEval')
        self.assertTrue(self.eval.detection, 9)
        self.assertTrue(self.eval.annotation, 18)

    def test_results(self):
        self.assertEqual(self.eval.error_category, 'relative')
        self.assertEqual(self.eval.score, 0.3)


class TestKeyMeanEvaluation(unittest.TestCase):

    def setUp(self):
        # this one should have a score of 1
        self.eval1 = KeyEvaluation(
            load_key(join(DETECTIONS_PATH, 'dummy.key.txt')),
            load_key(join(DETECTIONS_PATH, 'dummy.key.txt')),
            name='eval1'
        )
        # this one should have a score of 0.3
        self.eval2 = KeyEvaluation(
            load_key(join(DETECTIONS_PATH, 'dummy.key.txt')),
            load_key(join(ANNOTATIONS_PATH, 'dummy.key')),
            name='eval2'
        )

    def test_mean_results(self):
        mean_eval = KeyMeanEvaluation([self.eval1, self.eval2])
        self.assertAlmostEqual(mean_eval.correct, 0.5)
        self.assertAlmostEqual(mean_eval.fifth, 0.)
        self.assertAlmostEqual(mean_eval.relative, 0.5)
        self.assertAlmostEqual(mean_eval.parallel, 0.0)
        self.assertAlmostEqual(mean_eval.other, 0.0)
        self.assertAlmostEqual(mean_eval.weighted, 0.65)


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
