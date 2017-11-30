# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.evaluation.chords module.

"""

from __future__ import absolute_import, division, print_function

import unittest
from os.path import join

from madmom.evaluation.chords import *
from . import ANNOTATIONS_PATH, DETECTIONS_PATH

DUMMY_ANNOTATIONS = np.array(
    [(0.1, 1.0, (9, 0, [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])),
     (1.0, 2.0, (5, 0, [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])),
     (2.0, 3.0, (0, 0, [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])),
     (3.0, 4.0, (7, 0, [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]))],
    dtype=CHORD_ANN_DTYPE
)

DUMMY_DUPL_ANNOTATIONS = np.array(
    [(0.1, 1.0, (9, 0, [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])),
     (1.0, 2.0, (5, 0, [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])),
     (2.0, 3.0, (5, 0, [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])),
     (3.0, 4.0, (5, 1, [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]))],
    dtype=CHORD_ANN_DTYPE
)

DUMMY_MERGED_DUPL_ANNOTATIONS = np.array(
    [(0.1, 1.0, (9, 0, [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])),
     (1.0, 3.0, (5, 0, [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])),
     (3.0, 4.0, (5, 1, [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]))],
    dtype=CHORD_ANN_DTYPE
)


class TestChordParsing(unittest.TestCase):

    def test_modify(self):
        self.assertEqual(modify(0, 'b'), -1)
        self.assertEqual(modify(0, '#'), 1)
        self.assertEqual(modify(5, 'b'), 4)
        self.assertEqual(modify(10, '#'), 11)
        self.assertEqual(modify(0, 'bb'), -2)
        self.assertEqual(modify(0, '###'), 3)
        self.assertEqual(modify(5, 'b#'), 5)
        self.assertEqual(modify(10, '#b'), 10)
        self.assertEqual(modify(5, 'b#bb#'), 4)
        self.assertRaises(ValueError, modify, 0, 'ab#')

    def test_pitch(self):
        # test natural pitches
        for pn, pid in zip('CDEFGAB', [0, 2, 4, 5, 7, 9, 11]):
            self.assertEqual(pitch(pn), pid)

        # test modifiers
        self.assertEqual(pitch('C#'), 1)
        self.assertEqual(pitch('Cb'), 11)
        self.assertEqual(pitch('E#'), 5)
        self.assertEqual(pitch('Bb'), 10)

        # test multiple modifiers
        self.assertEqual(pitch('Bbb'), 9)
        self.assertEqual(pitch('G#b'), 7)
        self.assertEqual(pitch('Dbb'), 0)

    def test_interval(self):
        # test 'natural' intervals
        for int_name, int_id in zip(['{}'.format(i) for i in range(1, 14)],
                                    [i % 12 for i in [0, 2, 4, 5, 7, 9, 11, 12,
                                                      14, 16, 17, 19, 21]]):
            self.assertEqual(interval(int_name), int_id)

        # test modifiers
        self.assertEqual(interval('b3'), 3)
        self.assertEqual(interval('#4'), 6)
        self.assertEqual(interval('b7'), 10)
        self.assertEqual(interval('#7'), 0)

        # test multiple modifiers
        self.assertEqual(interval('##1'), 2)
        self.assertEqual(interval('#b5'), 7)
        self.assertEqual(interval('b#b6'), 8)

    def assertIntervalsEqual(self, i1, i2):
        self.assertTrue((i1 == i2).all())

    def test_interval_list(self):
        # test interval creation
        self.assertIntervalsEqual(
            interval_list('(1,3,5)'),
            np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]))
        self.assertIntervalsEqual(
            interval_list('(1,b3,5)'),
            np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]))
        self.assertIntervalsEqual(
            interval_list('(1,b3,5,b7)'),
            np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]))

        # test interval subtraction
        self.assertIntervalsEqual(
            interval_list('(*3)',
                          np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])),
            np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]))

        # test interval addition
        self.assertIntervalsEqual(
            interval_list('(3, b7)',
                          np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])),
            np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]))

    def test_intervals(self):
        # test some common interval annotations
        self.assertIntervalsEqual(chord_intervals('maj'),
                                  interval_list('(1,3,5)'))
        self.assertIntervalsEqual(chord_intervals('min'),
                                  interval_list('(1,b3,5)'))
        self.assertIntervalsEqual(chord_intervals('maj7'),
                                  interval_list('(1,3,5,7)'))

        # test addition of intervals
        self.assertIntervalsEqual(chord_intervals('maj(7)'),
                                  chord_intervals('maj7'))
        self.assertIntervalsEqual(chord_intervals('dim(bb7)'),
                                  chord_intervals('dim7'))

        # test removal of intervals
        self.assertIntervalsEqual(chord_intervals('maj9(*9)'),
                                  chord_intervals('maj7'))
        self.assertIntervalsEqual(chord_intervals('min7(*b7)'),
                                  chord_intervals('min'))

        # test addition and removal of intervals
        self.assertIntervalsEqual(chord_intervals('maj(*3,2)'),
                                  chord_intervals('sus2'))
        self.assertIntervalsEqual(chord_intervals('min(*b3,3,7)'),
                                  chord_intervals('maj7'))

    def assertChordEqual(self, c1, c2):
        self.assertEqual(c1[0], c2[0])
        self.assertEqual(c1[1], c2[1])
        self.assertIntervalsEqual(c1[2], c2[2])

    def test_chord(self):
        # pitch explicit, intervals and bass implicit
        self.assertChordEqual(chord('C'),
                              (pitch('C'), 0, chord_intervals('maj')))
        # pitch and bass explicit, intervals implicit
        self.assertChordEqual(chord('G#b/5'),
                              (pitch('G'), interval('5'),
                               chord_intervals('maj')))
        # pitch and intervals in shorthand explicit, bass implicit
        self.assertChordEqual(chord('Cb:sus4'), (pitch('Cb'), 0,
                                                 chord_intervals('sus4')))
        # pitch and intervals as list explicit, bass implicit
        self.assertChordEqual(chord('F:(1,3,5,9)'),
                              (pitch('F'), 0, chord_intervals('(1,3,5,9)')))
        # pitch and intervals, both shorthand and list, explicit bass implicit
        self.assertChordEqual(chord('Db:min6(*b3)'),
                              (pitch('Db'), 0, chord_intervals('(1,5,6)')))
        # everything explicit
        self.assertChordEqual(chord('A#:minmaj7/b3'),
                              (pitch('A#'), interval('b3'),
                               chord_intervals('minmaj7')))
        # test no-chord and unknown-chord
        self.assertChordEqual(chord('N'), NO_CHORD)
        self.assertChordEqual(chord('X'), UNKNOWN_CHORD)

    def test_chords(self):
        # test whether the chords() function creates a proper array of
        # chords
        labels = ['F', 'C:maj', 'D:(1,b3,5)', 'Bb:maj7']
        for lbl, crd in zip(labels, chords(labels)):
            self.assertChordEqual(chord(lbl), crd)

    def test_encode_func(self):
        crds = encode(
            load_chords(join(ANNOTATIONS_PATH, 'dummy.chords')))
        self.assertTrue((crds == DUMMY_ANNOTATIONS).all())

    def test_merge_func(self):
        merged_chords = merge_chords(DUMMY_DUPL_ANNOTATIONS)
        self.assertTrue((merged_chords == DUMMY_MERGED_DUPL_ANNOTATIONS).all())


class TestChordEvaluation(unittest.TestCase):

    def setUp(self):
        self.ann = encode(
            load_chords(join(ANNOTATIONS_PATH, 'dummy.chords')))
        self.unadjusted_det = encode(
            load_chords(join(DETECTIONS_PATH, 'dummy.chords.txt')))
        self.det = adjust(self.unadjusted_det, self.ann)
        self.ev_ann, self.ev_det, self.ev_dur = evaluation_pairs(self.det,
                                                                 self.ann)

    def assertIntervalsEqual(self, i1, i2):
        self.assertTrue((i1 == i2).all())

    def assertChordEqual(self, c1, c2):
        self.assertEqual(c1[0], c2[0])
        self.assertEqual(c1[1], c2[1])
        self.assertIntervalsEqual(c1[2], c2[2])

    def test_adjust(self):
        self.assertAlmostEqual(self.det[0]['start'], self.ann[0]['start'])
        self.assertAlmostEqual(self.det[-1]['end'], self.ann[-1]['end'])
        # the last 'N' chord should have been removed
        self.assertChordEqual(self.det[-1]['chord'], chord('G:aug'))

        det = self.unadjusted_det.copy()
        # make detections shorter than annotations
        det = det[:-2]
        det[0]['start'] = 0.5
        det = adjust(det, self.ann)

        self.assertAlmostEqual(det[0]['start'], self.ann[0]['start'])
        self.assertAlmostEqual(det[-1]['end'], self.ann[-1]['end'])
        # should have filled up the chord sequence with a no-chord
        self.assertChordEqual(det[-1]['chord'], chord('N'))

    def test_evaluation_pairs(self):
        true_ev_ann, true_ev_det, true_ev_dur = zip(*[
            (chord('A:min'), chord('N'), 0.1),
            (chord('A:min'), chord('A:min'), 0.6),
            (chord('A:min'), chord('A:min7'), 0.2),
            (chord('F:maj'), chord('F:dim'), 0.5),
            (chord('F:maj'), chord('F:maj'), 0.5),
            (chord('C:maj'), chord('F:maj'), 0.2),
            (chord('C:maj'), chord('C:maj/3'), 0.7),
            (chord('C:maj'), chord('G:maj7/7'), 0.1),
            (chord('G:maj'), chord('G:maj7/7'), 0.5),
            (chord('G:maj'), chord('G:aug'), 0.5),
        ])
        true_ev_ann = np.array(list(true_ev_ann), dtype=CHORD_DTYPE)
        true_ev_det = np.array(list(true_ev_det), dtype=CHORD_DTYPE)
        true_ev_dur = np.array(list(true_ev_dur))

        self.assertTrue((self.ev_ann == true_ev_ann).all())
        self.assertTrue((self.ev_det == true_ev_det).all())
        self.assertTrue(np.allclose(self.ev_dur, true_ev_dur))

    def test_score_root(self):
        score = score_root(self.ev_det, self.ev_ann)
        self.assertTrue(np.allclose(score, np.array([
            0., 1., 1., 1., 1., 0., 1., 0., 1., 1.
        ])))

    def test_score_exact(self):
        score = score_exact(self.ev_det, self.ev_ann)
        self.assertTrue(np.allclose(score, np.array([
            0., 1., 0., 0., 1., 0., 0., 0., 0., 0.
        ])))

    def test_select_majmin(self):
        # normally, you would not apply this function to the detected
        # evaluation  pairs - see evaluation.chords for correct usage.
        # However, ev_det contains a good variety of chords, so let's use it
        sel = select_majmin(self.ev_det)
        self.assertTrue((sel == np.array([True, True, False, False, True, True,
                                          True, False, False, False])).all())

    def test_select_sevenths(self):
        sel = select_sevenths(self.ev_det)
        self.assertTrue((sel == np.array([True, True, True, False, True, True,
                                          True, True, True, False])).all())

    def test_segmentation(self):
        self.assertAlmostEqual(
            segmentation(self.ann['start'], self.ann['end'],
                         self.det['start'], self.det['end']),
            0.41025641025641025641)
        self.assertAlmostEqual(
            segmentation(self.det['start'], self.det['end'],
                         self.ann['start'], self.ann['end']),
            0.07692307692307692308)

    def test_reduce_to_triads(self):
        true_red_wo_bass = chords(['N', 'A:min', 'A:min', 'F:dim', 'F:maj',
                                   'C:maj', 'G:maj', 'G:aug'])
        reduced_wo_bass = reduce_to_triads(self.det['chord'], keep_bass=False)
        self.assertTrue((reduced_wo_bass == true_red_wo_bass).all())

        true_red_w_bass = chords(['N', 'A:min', 'A:min', 'F:dim', 'F:maj',
                                  'C:maj/3', 'G:maj', 'G:aug'])
        reduced_w_bass = reduce_to_triads(self.det['chord'], keep_bass=True)
        self.assertTrue((reduced_w_bass == true_red_w_bass).all())

        # test some further mappings
        src = chords(['A:hdim7', 'B:min6/5', 'C:sus4/4', 'G:(1,5,b7)'])
        trg = chords(['A:dim', 'B:min/5', 'C:sus4/4', 'G:(1,5)'])
        self.assertTrue((reduce_to_triads(src, keep_bass=True) == trg).all())

    def test_reduce_to_tetrads(self):
        true_red_wo_bass = chords(['N', 'A:min', 'A:min7', 'F:dim', 'F:maj',
                                   'C:maj', 'G:maj7', 'G:aug'])
        reduced_wo_bass = reduce_to_tetrads(self.det['chord'], keep_bass=False)
        self.assertTrue((reduced_wo_bass == true_red_wo_bass).all())

        reduced_w_bass = reduce_to_tetrads(self.det['chord'], keep_bass=True)
        self.assertTrue((reduced_w_bass == self.det['chord']).all())

        # test some further mappings
        src = chords(['A:maj9', 'Cb:9', 'E:min9/9', 'E:min9/b7'])
        trg = chords(['A:maj7', 'Cb:7', 'E:min7', 'E:min7/b7'])
        self.assertTrue((reduce_to_tetrads(src, keep_bass=True) == trg).all())


class TestChordEvaluationClass(unittest.TestCase):

    def test_init(self):
        eval = ChordEvaluation(
            load_chords(join(DETECTIONS_PATH, 'dummy.chords.txt')),
            load_chords(join(ANNOTATIONS_PATH, 'dummy.chords')),
            name='TestEval'
        )
        self.assertTrue(eval.name == 'TestEval')
        ann = encode(
            load_chords(join(ANNOTATIONS_PATH, 'dummy.chords')))
        det = encode(
            load_chords(join(DETECTIONS_PATH, 'dummy.chords.txt')))
        det = adjust(det, ann)
        self.assertTrue((eval.ann_chords == ann).all())
        self.assertTrue((eval.det_chords == det).all())
        ann, det, dur = evaluation_pairs(eval.det_chords, eval.ann_chords)
        self.assertTrue((ann == eval.annotations).all())
        self.assertTrue((det == eval.detections).all())
        self.assertTrue((dur == eval.durations).all())

    def test_results(self):
        eval = ChordEvaluation(
            load_chords(join(DETECTIONS_PATH, 'dummy.chords.txt')),
            load_chords(join(ANNOTATIONS_PATH, 'dummy.chords')),
            name='TestEval'
        )
        self.assertAlmostEqual(eval.length, 3.9)
        self.assertAlmostEqual(eval.root, 0.8974358974358975)
        self.assertAlmostEqual(eval.majmin, 0.6410256410256411)
        self.assertAlmostEqual(eval.majminbass, 0.46153846153846156)
        self.assertAlmostEqual(eval.sevenths, 0.46153846153846156)
        self.assertAlmostEqual(eval.seventhsbass, 0.2820512820512821)
        self.assertAlmostEqual(eval.undersegmentation,
                               1. - 0.07692307692307692308)
        self.assertAlmostEqual(eval.oversegmentation,
                               1. - 0.41025641025641025641)
        self.assertAlmostEqual(eval.segmentation,
                               1. - 0.41025641025641025641)


class TestAggregateChordEvaluation(unittest.TestCase):

    def setUp(self):
        # this one should have a score of 1 everywhere and length 4.3
        self.eval1 = ChordEvaluation(
            load_chords(join(DETECTIONS_PATH, 'dummy.chords.txt')),
            load_chords(join(DETECTIONS_PATH, 'dummy.chords.txt')),
            name='TestEval'
        )
        self.eval2 = ChordEvaluation(
            load_chords(join(DETECTIONS_PATH, 'dummy.chords.txt')),
            load_chords(join(ANNOTATIONS_PATH, 'dummy.chords')),
            name='TestEval'
        )

    def test_mean_results(self):
        mean_eval = ChordMeanEvaluation([self.eval1, self.eval2])
        self.assertAlmostEqual(mean_eval.root, 0.9487179487179487)
        self.assertAlmostEqual(mean_eval.majmin, 0.8205128205128205)
        self.assertAlmostEqual(mean_eval.majminbass, 0.7307692307692308)
        self.assertAlmostEqual(mean_eval.sevenths, 0.7307692307692308)
        self.assertAlmostEqual(mean_eval.seventhsbass, 0.6410256410256411)
        self.assertAlmostEqual(mean_eval.undersegmentation, 0.9615384615384616)
        self.assertAlmostEqual(mean_eval.oversegmentation, 0.7948717948717949)
        self.assertAlmostEqual(mean_eval.segmentation, 0.7948717948717949)

    def test_sum_results(self):
        sum_eval = ChordSumEvaluation([self.eval1, self.eval2])
        self.assertAlmostEqual(sum_eval.root, 0.951219512195122)
        self.assertAlmostEqual(sum_eval.majmin, 0.8028169014084507)
        self.assertAlmostEqual(sum_eval.majminbass, 0.7042253521126761)
        self.assertAlmostEqual(sum_eval.sevenths, 0.7042253521126761)
        self.assertAlmostEqual(sum_eval.seventhsbass, 0.6056338028169015)
        self.assertAlmostEqual(sum_eval.undersegmentation, 0.9634146341463415)
        self.assertAlmostEqual(sum_eval.oversegmentation, 0.8048780487804879)
        self.assertAlmostEqual(sum_eval.segmentation, 0.8048780487804879)


class TestAddParserFunction(unittest.TestCase):

    def setUp(self):
        import argparse
        self.parser = argparse.ArgumentParser()
        sub_parser = self.parser.add_subparsers()
        self.sub_parser = add_parser(sub_parser)

    def test_args(self):
        args = self.parser.parse_args(['chords', ANNOTATIONS_PATH,
                                       DETECTIONS_PATH])
        self.assertTrue(args.ann_dir is None)
        self.assertTrue(args.ann_suffix == '.chords')
        self.assertTrue(args.det_dir is None)
        self.assertTrue(args.det_suffix == '.chords.txt')
        self.assertTrue(args.eval == ChordEvaluation)
        self.assertTrue(args.files == [ANNOTATIONS_PATH, DETECTIONS_PATH])
        self.assertTrue(args.mean_eval == ChordMeanEvaluation)
        self.assertTrue(args.sum_eval == ChordSumEvaluation)
        from madmom.evaluation import tostring
        self.assertTrue(args.output_formatter == tostring)
