import unittest
from madmom.evaluation.chords import *


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
        self.assertIntervalsEqual(intervals('maj'), interval_list('(1,3,5)'))
        self.assertIntervalsEqual(intervals('min'), interval_list('(1,b3,5)'))
        self.assertIntervalsEqual(intervals('maj7'), interval_list('(1,3,5,7)'))

        # test addition of intervals
        self.assertIntervalsEqual(intervals('maj(7)'), intervals('maj7'))
        self.assertIntervalsEqual(intervals('dim(bb7)'), intervals('dim7'))

        # test removal of intervals
        self.assertIntervalsEqual(intervals('maj9(*9)'), intervals('maj7'))
        self.assertIntervalsEqual(intervals('min7(*b7)'), intervals('min'))

        # test addition and removal of intervals
        self.assertIntervalsEqual(intervals('maj(*3,2)'), intervals('sus2'))
        self.assertIntervalsEqual(intervals('min(*b3,3,7)'), intervals('maj7'))

    def assertChordEqual(self, c1, c2):
        self.assertEqual(c1[0], c2[0])
        self.assertEqual(c1[1], c2[1])
        self.assertIntervalsEqual(c1[2], c2[2])

    def test_chord(self):
        # pitch explicit, intervals and bass implicit
        self.assertChordEqual(chord('C'), (pitch('C'), 0, intervals('maj')))
        # pitch and bass explicit, intervals implicit
        self.assertChordEqual(chord('G#b/5'), (pitch('G'), interval('5'),
                                               intervals('maj')))
        # pitch and intervals in shorthand explicit, bass implicit
        self.assertChordEqual(chord('Cb:sus4'), (pitch('Cb'), 0,
                                                 intervals('sus4')))
        # pitch and intervals as list explicit, bass implicit
        self.assertChordEqual(chord('F:(1,3,5,9)'), (pitch('F'), 0,
                                                     intervals('(1,3,5,9)')))
        # pitch and intervals, both shorthand and list, explicit bass implicit
        self.assertChordEqual(chord('Db:min6(*b3)'), (pitch('Db'), 0,
                                                      intervals('(1,5,b6)')))
        # everything explicit
        self.assertChordEqual(chord('A#:minmaj7/b3'), (pitch('A#'),
                                                       interval('b3'),
                                                       intervals('minmaj7')))

    def test_chords(self):
        # test whether the chords() function creates a proper array of
        # chords
        labels = ['F', 'C:maj', 'D:(1,b3,5)', 'Bb:maj7']
        for lbl, crd in zip(labels, chords(labels)):
            self.assertChordEqual(chord(lbl), crd)


