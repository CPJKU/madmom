import unittest
from madmom.evaluation.chords import *
from . import ANNOTATIONS_PATH, DETECTIONS_PATH
from os.path import join


DUMMY_ANNOTATIONS = np.array(
    [(0.1, 1.0, (9, 0, [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])),
     (1.0, 2.0, (5, 0, [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])),
     (2.0, 3.0, (0, 0, [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])),
     (3.0, 4.0, (7, 0, [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]))],
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


class TestChordLoading(unittest.TestCase):

    def test_load_func(self):
        crds = load_chords(join(ANNOTATIONS_PATH, 'dummy.chords'))
        self.assertTrue((crds == DUMMY_ANNOTATIONS).all())


class TestChordEvaluation(unittest.TestCase):

    def assertIntervalsEqual(self, i1, i2):
        self.assertTrue((i1 == i2).all())

    def assertChordEqual(self, c1, c2):
        self.assertEqual(c1[0], c2[0])
        self.assertEqual(c1[1], c2[1])
        self.assertIntervalsEqual(c1[2], c2[2])

    def test_adjust(self):
        ann = load_chords(join(ANNOTATIONS_PATH, 'dummy.chords'))
        det = load_chords(join(DETECTIONS_PATH, 'dummy.chords.txt'))
        det = adjust(det, ann)

        self.assertAlmostEqual(det[0]['start'], ann[0]['start'])
        self.assertAlmostEqual(det[-1]['end'], ann[-1]['end'])
        # the last 'N' chord should have been removed
        self.assertChordEqual(det[-1]['chord'], chord('G:min'))

        ann = load_chords(join(ANNOTATIONS_PATH, 'dummy.chords'))
        det = load_chords(join(DETECTIONS_PATH, 'dummy.chords.txt'))
        # make detections shorter than annotations
        det = det[:-2]
        det[0]['start'] = 0.5
        det = adjust(det, ann)

        self.assertAlmostEqual(det[0]['start'], ann[0]['start'])
        self.assertAlmostEqual(det[-1]['end'], ann[-1]['end'])
        # should have filled up the chord sequence with a no-chord
        self.assertChordEqual(det[-1]['chord'], chord('N'))

    def test_evaluation_pairs(self):
        ann = load_chords(join(ANNOTATIONS_PATH, 'dummy.chords'))
        det = load_chords(join(DETECTIONS_PATH, 'dummy.chords.txt'))
        det = adjust(det, ann)

        ev_ann, ev_det, ev_dur = evaluation_pairs(det, ann)
        true_ev_ann, true_ev_det, true_ev_dur = zip(*[
            (chord('A:min'), chord('N'), 0.1),
            (chord('A:min'), chord('A:min'), 0.6),
            (chord('A:min'), chord('A:min7'), 0.2),
            (chord('F:maj'), chord('F:min'), 0.5),
            (chord('F:maj'), chord('F:maj'), 0.5),
            (chord('C:maj'), chord('F:maj'), 0.2),
            (chord('C:maj'), chord('C:maj'), 0.7),
            (chord('C:maj'), chord('G:maj7/7'), 0.1),
            (chord('G:maj'), chord('G:maj7/7'), 0.5),
            (chord('G:maj'), chord('G:min'), 0.5),
        ])
        true_ev_ann = np.array(list(true_ev_ann), dtype=CHORD_DTYPE)
        true_ev_det = np.array(list(true_ev_det), dtype=CHORD_DTYPE)
        true_ev_dur = np.array(list(true_ev_dur))

        self.assertTrue((ev_ann == true_ev_ann).all())
        self.assertTrue((ev_det == true_ev_det).all())
        self.assertTrue(np.allclose(ev_dur, true_ev_dur))

    def test_score_root(self):
        ann = load_chords(join(ANNOTATIONS_PATH, 'dummy.chords'))
        det = load_chords(join(DETECTIONS_PATH, 'dummy.chords.txt'))
        det = adjust(det, ann)
        ev_ann, ev_det, ev_dur = evaluation_pairs(det, ann)

        score = score_root(ev_det, ev_ann)
        self.assertTrue(np.allclose(score, np.array([
            0., 1., 1., 1., 1., 0., 1., 0., 1., 1.
        ])))

    def test_score_exact(self):
        ann = load_chords(join(ANNOTATIONS_PATH, 'dummy.chords'))
        det = load_chords(join(DETECTIONS_PATH, 'dummy.chords.txt'))
        det = adjust(det, ann)
        ev_ann, ev_det, ev_dur = evaluation_pairs(det, ann)

        score = score_exact(ev_det, ev_ann)
        self.assertTrue(np.allclose(score, np.array([
            0., 1., 0., 0., 1., 0., 1., 0., 0., 0.
        ])))

    def test_select_majmin(self):
        ann = load_chords(join(ANNOTATIONS_PATH, 'dummy.chords'))
        det = load_chords(join(DETECTIONS_PATH, 'dummy.chords.txt'))
        det = adjust(det, ann)
        ev_ann, ev_det, ev_dur = evaluation_pairs(det, ann)

        # normally, you would not apply this function to the detected
        # evaluation  pairs - see evaluation.chords for correct usage.
        # However, ev_det contains a good variety of chords, so let's use it
        sel = select_majmin(ev_det)
        self.assertTrue((sel == np.array([True, True, False, True, True, True,
                                          True, False, False, True])).all())

    def test_segmentation(self):
        ann = load_chords(join(ANNOTATIONS_PATH, 'dummy.chords'))
        det = load_chords(join(DETECTIONS_PATH, 'dummy.chords.txt'))
        det = adjust(det, ann)

        self.assertAlmostEqual(
            segmentation(ann['start'], ann['end'], det['start'], det['end']),
            0.41025641025641025641)
        self.assertAlmostEqual(
            segmentation(det['start'], det['end'], ann['start'], ann['end']),
            0.07692307692307692308)
