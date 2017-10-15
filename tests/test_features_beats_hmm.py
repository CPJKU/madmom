# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.features.beats_hmm module.

"""

from __future__ import absolute_import, division, print_function

import unittest
from madmom.ml.hmm import *
from madmom.features.beats_hmm import *


# state spaces
class TestBeatStateSpaceClass(unittest.TestCase):

    def test_types(self):
        bss = BeatStateSpace(1, 4)
        self.assertIsInstance(bss.intervals, np.ndarray)
        self.assertIsInstance(bss.state_positions, np.ndarray)
        self.assertIsInstance(bss.state_intervals, np.ndarray)
        self.assertIsInstance(bss.first_states, np.ndarray)
        self.assertIsInstance(bss.last_states, np.ndarray)
        self.assertIsInstance(bss.num_states, int)
        self.assertIsInstance(bss.num_intervals, int)
        # dtypes
        self.assertTrue(bss.intervals.dtype == np.int)
        self.assertTrue(bss.state_positions.dtype == np.float)
        self.assertTrue(bss.state_intervals.dtype == np.int)
        self.assertTrue(bss.first_states.dtype == np.int)
        self.assertTrue(bss.last_states.dtype == np.int)

    def test_values(self):
        bss = BeatStateSpace(1, 4)
        self.assertTrue(np.allclose(bss.intervals, [1, 2, 3, 4]))
        self.assertTrue(np.allclose(bss.state_positions,
                                    [0, 0, 0.5, 0, 1. / 3, 2. / 3,
                                     0, 0.25, 0.5, 0.75]))
        self.assertTrue(np.allclose(bss.state_intervals,
                                    [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]))
        self.assertTrue(np.allclose(bss.first_states, [0, 1, 3, 6]))
        self.assertTrue(np.allclose(bss.last_states, [0, 2, 5, 9]))
        self.assertTrue(bss.num_states == 10)
        self.assertTrue(bss.num_intervals == 4)
        # other intervals
        bss = BeatStateSpace(2, 6)
        self.assertTrue(np.allclose(bss.intervals, [2, 3, 4, 5, 6]))
        self.assertTrue(np.allclose(bss.state_positions,
                                    [0, 0.5,
                                     0, 1. / 3, 2. / 3,
                                     0, 0.25, 0.5, 0.75,
                                     0, 0.2, 0.4, 0.6, 0.8,
                                     0, 1. / 6, 2. / 6, 0.5, 4. / 6, 5. / 6]))
        self.assertTrue(np.allclose(bss.state_intervals,
                                    [2, 2, 3, 3, 3, 4, 4, 4, 4,
                                     5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6]))
        self.assertTrue(np.allclose(bss.first_states, [0, 2, 5, 9, 14]))
        self.assertTrue(np.allclose(bss.last_states, [1, 4, 8, 13, 19]))
        self.assertTrue(bss.num_states == 20)
        self.assertTrue(bss.num_intervals == 5)


class TestBarStateSpaceClass(unittest.TestCase):

    def test_types(self):
        bss = BarStateSpace(2, 1, 4)
        self.assertIsInstance(bss.num_beats, int)
        self.assertIsInstance(bss.num_states, int)
        # self.assertIsInstance(bss.intervals, np.ndarray)
        self.assertIsInstance(bss.state_positions, np.ndarray)
        self.assertIsInstance(bss.state_intervals, np.ndarray)
        self.assertIsInstance(bss.first_states, list)
        self.assertIsInstance(bss.last_states, list)
        # dtypes
        self.assertTrue(bss.state_positions.dtype == np.float)
        self.assertTrue(bss.state_intervals.dtype == np.int)

    def test_values(self):
        # 2 beats, intervals 1 to 4
        bss = BarStateSpace(2, 1, 4)
        self.assertTrue(bss.num_beats == 2)
        self.assertTrue(bss.num_states == 20)
        # self.assertTrue(np.allclose(bss.intervals, [1, 2, 3, 4]))
        self.assertTrue(np.allclose(bss.state_positions,
                                    [0, 0, 0.5, 0, 1. / 3, 2. / 3,
                                     0, 0.25, 0.5, 0.75,
                                     1, 1, 1.5, 1, 4. / 3, 5. / 3,
                                     1, 1.25, 1.5, 1.75]))
        self.assertTrue(np.allclose(bss.state_intervals,
                                    [1, 2, 2, 3, 3, 3, 4, 4, 4, 4,
                                     1, 2, 2, 3, 3, 3, 4, 4, 4, 4]))
        self.assertTrue(np.allclose(bss.first_states, [[0, 1, 3, 6],
                                                       [10, 11, 13, 16]]))
        self.assertTrue(np.allclose(bss.last_states, [[0, 2, 5, 9],
                                                      [10, 12, 15, 19]]))
        # other values: 1 beat, intervals 2 to 6
        bss = BarStateSpace(1, 2, 6)
        self.assertTrue(bss.num_beats == 1)
        self.assertTrue(bss.num_states == 20)
        # self.assertTrue(np.allclose(bss.intervals, [2, 3, 4, 5, 6]))
        self.assertTrue(np.allclose(bss.state_positions,
                                    [0, 0.5,
                                     0, 1. / 3, 2. / 3,
                                     0, 0.25, 0.5, 0.75,
                                     0, 0.2, 0.4, 0.6, 0.8,
                                     0, 1. / 6, 2. / 6, 0.5, 4. / 6, 5. / 6]))
        self.assertTrue(np.allclose(bss.state_intervals,
                                    [2, 2, 3, 3, 3, 4, 4, 4, 4,
                                     5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6]))
        self.assertTrue(np.allclose(bss.first_states, [[0, 2, 5, 9, 14]]))
        self.assertTrue(np.allclose(bss.last_states, [[1, 4, 8, 13, 19]]))


class TestMultiPatternStateSpaceClass(unittest.TestCase):

    def test_types(self):
        # test with 2 BeatStateSpaces as before
        # mpss = MultiPatternStateSpace([1, 2], [4, 6])
        bss1 = BeatStateSpace(1, 4)
        bss2 = BeatStateSpace(2, 6)
        mpss = MultiPatternStateSpace([bss1, bss2])
        self.assertIsInstance(mpss.state_spaces, list)
        self.assertIsInstance(mpss.state_positions, np.ndarray)
        self.assertIsInstance(mpss.state_intervals, np.ndarray)
        self.assertIsInstance(mpss.num_states, int)
        # self.assertIsInstance(mpss.num_intervals, int)
        self.assertIsInstance(mpss.num_patterns, int)
        # dtypes
        self.assertTrue(mpss.state_positions.dtype == np.float)
        self.assertTrue(mpss.state_intervals.dtype == np.int)

    def test_values_beat(self):
        # test with 2 BeatStateSpaces as before
        # mpss = MultiPatternStateSpace([1, 2], [4, 6])
        bss1 = BeatStateSpace(1, 4)
        bss2 = BeatStateSpace(2, 6)
        mpss = MultiPatternStateSpace([bss1, bss2])
        self.assertTrue(np.allclose(mpss.state_spaces[0].intervals,
                                    [1, 2, 3, 4]))
        self.assertTrue(np.allclose(mpss.state_spaces[1].intervals,
                                    [2, 3, 4, 5, 6]))
        self.assertTrue(mpss.num_states == 30)
        # self.assertTrue(mpss.num_intervals == [4, 5])
        self.assertTrue(mpss.num_patterns == 2)
        # first pattern
        self.assertTrue(np.allclose(mpss.state_positions[:10],
                                    [0, 0, 0.5, 0, 1. / 3, 2. / 3,
                                     0, 0.25, 0.5, 0.75]))
        self.assertTrue(np.allclose(mpss.state_intervals[:10],
                                    [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]))
        self.assertTrue(np.allclose(mpss.state_patterns[:10], 0))
        # second pattern
        self.assertTrue(np.allclose(mpss.state_positions[10:],
                                    [0, 0.5,
                                     0, 1. / 3, 2. / 3,
                                     0, 0.25, 0.5, 0.75,
                                     0, 0.2, 0.4, 0.6, 0.8,
                                     0, 1. / 6, 2. / 6, 0.5, 4. / 6, 5. / 6]))
        self.assertTrue(np.allclose(mpss.state_intervals[10:],
                                    [2, 2, 3, 3, 3, 4, 4, 4, 4,
                                     5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6]))
        self.assertTrue(np.allclose(mpss.state_patterns[10:], 1))

    def test_values_bar(self):
        # test with 2 BarStateSpaces
        bss1 = BarStateSpace(2, 1, 4)
        bss2 = BarStateSpace(1, 2, 6)
        mpss = MultiPatternStateSpace([bss1, bss2])
        # self.assertTrue(np.allclose(mpss.state_spaces[0].intervals,
        #                             [1, 2, 3, 4]))
        # self.assertTrue(np.allclose(mpss.state_spaces[1].intervals,
        #                             [2, 3, 4, 5, 6]))
        self.assertTrue(mpss.num_states == 40)
        # self.assertTrue(mpss.num_intervals == [4, 5])
        self.assertTrue(mpss.num_patterns == 2)
        # first pattern
        self.assertTrue(np.allclose(mpss.state_positions[:20],
                                    [0, 0, 0.5, 0, 1. / 3, 2. / 3,
                                     0, 0.25, 0.5, 0.75,
                                     1, 1, 1.5, 1, 4. / 3, 5. / 3,
                                     1, 1.25, 1.5, 1.75]))
        self.assertTrue(np.allclose(mpss.state_intervals[:20],
                                    [1, 2, 2, 3, 3, 3, 4, 4, 4, 4,
                                     1, 2, 2, 3, 3, 3, 4, 4, 4, 4]))
        self.assertTrue(np.allclose(mpss.state_patterns[:20], 0))
        # self.assertTrue(np.allclose(mpss.first_states[0],
        #                             [[0, 1, 3, 6], [10, 11, 13, 16]]))
        # self.assertTrue(np.allclose(mpss.last_states[0],
        #                             [[0, 2, 5, 9], [10, 12, 15, 19]]))
        # second pattern
        self.assertTrue(np.allclose(mpss.state_positions[20:],
                                    [0, 0.5,
                                     0, 1. / 3, 2. / 3,
                                     0, 0.25, 0.5, 0.75,
                                     0, 0.2, 0.4, 0.6, 0.8,
                                     0, 1. / 6, 2. / 6, 0.5, 4. / 6, 5. / 6]))
        self.assertTrue(np.allclose(mpss.state_intervals[20:],
                                    [2, 2, 3, 3, 3, 4, 4, 4, 4,
                                     5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6]))
        self.assertTrue(np.allclose(mpss.state_patterns[20:], 1))
        # self.assertTrue(np.allclose(mpss.first_states[1],
        #                             [[0, 2, 5, 9, 14]]))
        # self.assertTrue(np.allclose(mpss.last_states[1],
        #                             [[1, 4, 8, 13, 19]]))


# transition models
class TestBeatTransitionModelClass(unittest.TestCase):

    def test_types(self):
        bss = BeatStateSpace(1, 4)
        tm = BeatTransitionModel(bss, 100)
        self.assertIsInstance(tm, BeatTransitionModel)
        self.assertIsInstance(tm, TransitionModel)
        self.assertIsInstance(tm.state_space, BeatStateSpace)
        self.assertIsInstance(tm.transition_lambda, float)
        self.assertIsInstance(tm.states, np.ndarray)
        self.assertIsInstance(tm.pointers, np.ndarray)
        self.assertIsInstance(tm.probabilities, np.ndarray)
        self.assertIsInstance(tm.log_probabilities, np.ndarray)
        self.assertIsInstance(tm.num_states, int)
        self.assertIsInstance(tm.num_transitions, int)
        self.assertTrue(tm.states.dtype == np.uint32)
        self.assertTrue(tm.pointers.dtype == np.uint32)
        self.assertTrue(tm.probabilities.dtype == np.float)
        self.assertTrue(tm.log_probabilities.dtype == np.float)

    def test_values(self):
        bss = BeatStateSpace(1, 4)
        tm = BeatTransitionModel(bss, 100)
        self.assertTrue(np.allclose(tm.states,
                                    [0, 2, 5, 1, 5, 9, 3, 4, 5, 9, 6, 7, 8]))
        self.assertTrue(np.allclose(tm.pointers,
                                    [0, 1, 3, 4, 6, 7, 8, 10, 11, 12, 13]))
        self.assertTrue(np.allclose(tm.probabilities,
                                    [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1]))
        self.assertTrue(np.allclose(tm.log_probabilities,
                                    [0, 0, -33.33333, 0, 0, -25,
                                     0, 0, -33.33333, 0, 0, 0, 0]))
        self.assertTrue(tm.num_states == 10)
        self.assertTrue(tm.num_transitions == 13)


class TestBarTransitionModelClass(unittest.TestCase):

    def test_types(self):
        bss = BarStateSpace(2, 1, 4)
        tm = BarTransitionModel(bss, 100)
        self.assertIsInstance(tm, BarTransitionModel)
        self.assertIsInstance(tm, TransitionModel)
        self.assertIsInstance(tm.state_space, BarStateSpace)
        self.assertIsInstance(tm.transition_lambda, list)
        self.assertIsInstance(tm.states, np.ndarray)
        self.assertIsInstance(tm.pointers, np.ndarray)
        self.assertIsInstance(tm.probabilities, np.ndarray)
        self.assertIsInstance(tm.log_probabilities, np.ndarray)
        self.assertIsInstance(tm.num_states, int)
        self.assertIsInstance(tm.num_transitions, int)
        self.assertTrue(tm.states.dtype == np.uint32)
        self.assertTrue(tm.pointers.dtype == np.uint32)
        self.assertTrue(tm.probabilities.dtype == np.float)
        self.assertTrue(tm.log_probabilities.dtype == np.float)

    def test_values(self):
        bss = BarStateSpace(2, 1, 4)
        tm = BarTransitionModel(bss, 100)
        self.assertTrue(np.allclose(tm.states,
                                    [10, 12, 15, 1, 15, 19, 3, 4, 15, 19, 6, 7,
                                     8, 0, 2, 5, 11, 5, 9, 13, 14, 5, 9, 16,
                                     17, 18]))
        self.assertTrue(np.allclose(tm.pointers,
                                    [0, 1, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14,
                                     16, 17, 19, 20, 21, 23, 24, 25, 26]))
        self.assertTrue(np.allclose(tm.probabilities,
                                    [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1,
                                     1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1]))
        self.assertTrue(np.allclose(tm.log_probabilities,
                                    [0, 0, -33.33333, 0, 0, -25,
                                     0, 0, -33.33333, 0, 0, 0, 0,
                                     0, 0, -33.33333, 0, 0, -25,
                                     0, 0, -33.33333, 0, 0, 0, 0]))
        self.assertTrue(tm.num_states == 20)
        self.assertTrue(tm.num_transitions == 26)


class TestMultiPatternTransitionModelClass(unittest.TestCase):

    def test_types(self):
        bss1 = BeatStateSpace(1, 4)
        bss2 = BeatStateSpace(2, 6)
        btm1 = BeatTransitionModel(bss1, 100)
        btm2 = BeatTransitionModel(bss2, 100)
        tm = MultiPatternTransitionModel([btm1, btm2])
        self.assertIsInstance(tm, MultiPatternTransitionModel)
        self.assertIsInstance(tm, TransitionModel)
        self.assertIsInstance(tm.transition_models, list)
        self.assertIsNone(tm.transition_prob)
        self.assertIsInstance(tm.states, np.ndarray)
        self.assertIsInstance(tm.pointers, np.ndarray)
        self.assertIsInstance(tm.probabilities, np.ndarray)
        self.assertIsInstance(tm.log_probabilities, np.ndarray)
        self.assertIsInstance(tm.num_states, int)
        self.assertIsInstance(tm.num_transitions, int)
        self.assertTrue(tm.states.dtype == np.uint32)
        self.assertTrue(tm.pointers.dtype == np.uint32)
        self.assertTrue(tm.probabilities.dtype == np.float)
        self.assertTrue(tm.log_probabilities.dtype == np.float)

    def test_values_beat(self):
        # test with 2 BeatStateSpaces
        bss1 = BeatStateSpace(1, 4)
        bss2 = BeatStateSpace(2, 6)
        btm1 = BeatTransitionModel(bss1, 100)
        btm2 = BeatTransitionModel(bss2, 100)
        tm = MultiPatternTransitionModel([btm1, btm2])

        self.assertTrue(tm.num_states == 10 + 20)
        self.assertTrue(tm.num_transitions == 13 + 28)
        # the first pattern has 13 transitions
        self.assertTrue(np.allclose(tm.states[:13],
                                    [0, 2, 5, 1, 5, 9, 3, 4, 5, 9, 6, 7, 8]))
        # the second 28
        self.assertTrue(np.allclose(tm.states[13:],
                                    [11, 14, 10, 14, 18, 12, 13, 14, 18, 23,
                                     29, 15, 16, 17, 18, 23, 29, 19, 20, 21,
                                     22, 23, 29, 24, 25, 26, 27, 28]))
        # the first pattern has 10 states (pointers has one more element)
        self.assertTrue(np.allclose(tm.pointers[:11],
                                    [0, 1, 3, 4, 6, 7, 8, 10, 11, 12, 13]))
        # the second has 20
        self.assertTrue(np.allclose(tm.pointers[11:],
                                    [15, 16, 18, 19, 20, 24, 25, 26, 27, 30,
                                     31, 32, 33, 34, 36, 37, 38, 39, 40, 41]))
        # transition probabilities
        self.assertTrue(np.allclose(tm.probabilities,
                                    [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1,
                                     0, 1, 1, 0, 1, 1, 0, 1, 2.06e-09, 0, 1, 1,
                                     1, 0, 1, 5.78e-08, 1, 1, 1, 1, 2.06e-09,
                                     1, 1, 1, 1, 1, 1]))
        self.assertTrue(np.allclose(tm.log_probabilities,
                                    [0, 0, -33.33333, 0, 0, -25, 0, 0,
                                     -33.33333, 0, 0, 0, 0, 0,
                                     -33.33333, 0, 0, -25, 0, 0,
                                     -33.33333, 0, -20, -33.33333, 0, 0,
                                     0, -25, -4.1e-09, -16.6666, 0, 0, 0,
                                     0, -20, -5.78e-08, 0, 0, 0, 0, 0]))

    def test_values_bar(self):
        # test with 2 BarStateSpaces
        bss1 = BarStateSpace(2, 1, 4)
        bss2 = BarStateSpace(1, 2, 6)
        btm1 = BarTransitionModel(bss1, 100)
        btm2 = BarTransitionModel(bss2, 100)
        tm = MultiPatternTransitionModel([btm1, btm2])
        self.assertTrue(tm.num_states == 20 + 20)
        self.assertTrue(tm.num_transitions == 26 + 28)
        # the first pattern has 26 transitions
        self.assertTrue(np.allclose(tm.states[:26],
                                    [10, 12, 15, 1, 15, 19, 3, 4, 15, 19, 6, 7,
                                     8, 0, 2, 5, 11, 5, 9, 13, 14, 5, 9, 16,
                                     17, 18]))
        # the second 28
        self.assertTrue(np.allclose(tm.states[26:],
                                    [21, 24, 20, 24, 28, 22, 23, 24, 28, 33,
                                     39, 25, 26, 27, 28, 33, 39, 29, 30, 31,
                                     32, 33, 39, 34, 35, 36, 37, 38]))
        # the first pattern has 20 states (pointers has one more element)
        self.assertTrue(np.allclose(tm.pointers[:21],
                                    [0, 1, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14,
                                     16, 17, 19, 20, 21, 23, 24, 25, 26]))
        # the second has 20
        self.assertTrue(np.allclose(tm.pointers[21:],
                                    [28, 29, 31, 32, 33, 37, 38, 39, 40, 43,
                                     44, 45, 46, 47, 49, 50, 51, 52, 53, 54]))
        # transition probabilities
        self.assertTrue(np.allclose(tm.probabilities,
                                    [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1,
                                     1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0,
                                     1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1,
                                     0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                                    atol=1e-7))
        self.assertTrue(np.allclose(tm.log_probabilities,
                                    [0, 0, -33.33333, 0, 0, -25,
                                     0, 0, -33.33333, 0, 0, 0, 0,
                                     0, 0, -33.33333, 0, 0, -25,
                                     0, 0, -33.33333, 0, 0, 0,
                                     0, 0, -33.33333, 0, 0, -25,
                                     0, 0, -33.33333, 0, -20, -33.33333, 0,
                                     0, 0, -25, 0, -16.6666, 0, 0,
                                     0, 0, -20, -5.78e-08, 0, 0, 0, 0, 0]))

    def test_values_meter_transition(self):
        # test with 2 BarStateSpaces
        bss1 = BarStateSpace(2, 1, 1)  # states 01
        bss2 = BarStateSpace(3, 1, 1)  # states 234
        btm1 = BarTransitionModel(bss1, 100)
        btm2 = BarTransitionModel(bss2, 100)
        tm = MultiPatternTransitionModel([btm1, btm2], transition_prob=0.25)
        self.assertIsInstance(tm.transition_prob, np.ndarray)
        self.assertTrue(tm.num_states == 5)
        self.assertTrue(tm.num_transitions == 7)
        self.assertTrue(np.allclose(tm.states, [1, 4, 0, 1, 4, 2, 3]))
        self.assertTrue(np.allclose(tm.pointers, [0, 2, 3, 5, 6, 7]))
        self.assertTrue(np.allclose(tm.probabilities,
                                    [0.75, 0.25, 1, 0.25, 0.75, 1, 1]))
        states, prev_states, probs = tm.make_dense(tm.states, tm.pointers,
                                                   tm.probabilities)
        self.assertTrue(np.allclose(prev_states, [1, 4, 0, 1, 4, 2, 3]))
        self.assertTrue(np.allclose(states, [0, 0, 1, 2, 2, 3, 4]))
        self.assertTrue(np.allclose(probs, [0.75, 0.25, 1, 0.25, 0.75, 1, 1]))
        # same with 3 bar lengths
        bss3 = BarStateSpace(4, 1, 1)  # states 5678
        btm3 = BarTransitionModel(bss3, 100)
        trans = np.array([[0.6, 0.3, 0.25],
                          [0.15, 0.6, 0.15],
                          [0.25, 0.1, 0.6]])
        tm = MultiPatternTransitionModel([btm1, btm2, btm3],
                                         transition_prob=trans)
        self.assertIsInstance(tm.transition_prob, np.ndarray)
        self.assertTrue(tm.num_states == 9)
        self.assertTrue(tm.num_transitions == 15)
        self.assertTrue(np.allclose(tm.states, [1, 4, 8, 0, 1, 4, 8, 2, 3, 1,
                                                4, 8, 5, 6, 7]))
        self.assertTrue(np.allclose(tm.pointers, [0, 3, 4, 7, 8, 9, 12, 13,
                                                  14, 15]))
        self.assertTrue(np.allclose(tm.probabilities,
                                    [0.6, 0.3, 0.25, 1, 0.15, 0.6, 0.15, 1, 1,
                                     0.25, 0.1, 0.6, 1, 1, 1]))
        states, prev_states, probs = tm.make_dense(tm.states, tm.pointers,
                                                   tm.probabilities)
        self.assertTrue(np.allclose(prev_states, [1, 4, 8, 0, 1, 4, 8, 2, 3,
                                                  1, 4, 8, 5, 6, 7]))
        self.assertTrue(np.allclose(states, [0, 0, 0, 1, 2, 2, 2, 3, 4,
                                             5, 5, 5, 6, 7, 8]))
        self.assertTrue(np.allclose(probs, [0.6, 0.3, 0.25, 1, 0.15, 0.6, 0.15,
                                            1, 1, 0.25, 0.1, 0.6, 1, 1, 1]))
        # test with 2 BarStateSpaces with more tempi
        bss1 = BarStateSpace(2, 2, 5)
        bss2 = BarStateSpace(3, 2, 4)
        btm1 = BarTransitionModel(bss1, 100)
        btm2 = BarTransitionModel(bss2, 100)
        tm = MultiPatternTransitionModel([btm1, btm2])
        self.assertTrue(tm.num_states == 55)
        self.assertTrue(tm.num_transitions == 74)
        with self.assertRaises(ValueError):
            MultiPatternTransitionModel([btm1, btm2], transition_prob=0.25)
        # same with same number of tempi
        bss1 = BarStateSpace(2, 2, 4)
        bss2 = BarStateSpace(3, 2, 4)
        btm1 = BarTransitionModel(bss1, 100)
        btm2 = BarTransitionModel(bss2, 100)
        tm = MultiPatternTransitionModel([btm1, btm2], transition_prob=0.25)
        self.assertTrue(tm.num_states == 45)
        self.assertTrue(tm.num_transitions == 72)


# observation models
class TestRNNBeatTrackingObservationModelClass(unittest.TestCase):

    def setUp(self):
        btss = BeatStateSpace(1, 4)
        self.om = RNNBeatTrackingObservationModel(btss, 4)
        self.obs = np.asarray([1, 0.1, 0.01, 0], dtype=np.float32)

    def test_types(self):
        self.assertIsInstance(self.om.pointers, np.ndarray)
        self.assertIsInstance(self.om.densities(self.obs), np.ndarray)
        self.assertIsInstance(self.om.log_densities(self.obs), np.ndarray)
        self.assertTrue(self.om.pointers.dtype == np.uint32)
        self.assertTrue(self.om.densities(self.obs).dtype == np.float)
        self.assertTrue(self.om.log_densities(self.obs).dtype == np.float)

    def test_values(self):
        self.assertTrue(np.allclose(self.om.pointers,
                                    [1, 1, 0, 1, 0, 0, 1, 0, 0, 0]))
        self.assertTrue(np.allclose(self.om.densities(self.obs),
                                    [[0, 1], [0.3, 0.1],
                                     [0.33, 0.01], [1. / 3, 0]]))
        self.assertTrue(np.allclose(self.om.log_densities(self.obs),
                                    [[-np.inf, 0], [-1.20397281, -2.30258508],
                                     [-1.10866262, -4.60517021],
                                     [-1.09861229, -np.inf]]))
