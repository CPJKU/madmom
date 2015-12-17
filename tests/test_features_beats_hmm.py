# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.ml.hmm module.

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
        self.assertIsInstance(bss.position, np.ndarray)
        self.assertIsInstance(bss.interval, np.ndarray)
        self.assertIsInstance(bss.num_states, int)
        self.assertIsInstance(bss.num_intervals, int)
        self.assertIsInstance(bss.first_states, np.ndarray)
        self.assertIsInstance(bss.last_states, np.ndarray)

    def test_values(self):
        bss = BeatStateSpace(1, 4)
        self.assertTrue(np.allclose(bss.intervals, [1, 2, 3, 4]))
        self.assertTrue(np.allclose(bss.position,
                                    [0, 0, 0.5, 0, 1. / 3, 2. / 3,
                                     0, 0.25, 0.5, 0.75]))
        self.assertTrue(np.allclose(bss.interval,
                                    [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]))
        self.assertTrue(np.allclose(bss.first_states, [0, 1, 3, 6]))
        self.assertTrue(np.allclose(bss.last_states, [0, 2, 5, 9]))
        self.assertTrue(bss.num_states == 10)
        self.assertTrue(bss.num_intervals == 4)
        # other intervals
        bss = BeatStateSpace(2, 6)
        self.assertTrue(np.allclose(bss.intervals, [2, 3, 4, 5, 6]))
        self.assertTrue(np.allclose(bss.position,
                                    [0, 0.5,
                                     0, 1. / 3, 2. / 3,
                                     0, 0.25, 0.5, 0.75,
                                     0, 0.2, 0.4, 0.6, 0.8,
                                     0, 1. / 6, 2. / 6, 0.5, 4. / 6, 5. / 6]))
        self.assertTrue(np.allclose(bss.interval,
                                    [0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3,
                                     4, 4, 4, 4, 4, 4]))
        self.assertTrue(np.allclose(bss.first_states, [0, 2, 5, 9, 14]))
        self.assertTrue(np.allclose(bss.last_states, [1, 4, 8, 13, 19]))
        self.assertTrue(bss.num_states == 20)
        self.assertTrue(bss.num_intervals == 5)


class TestMultiPatternStateSpaceClass(unittest.TestCase):

    def setUp(self):
        self.ptss = MultiPatternStateSpace([1, 2], [4, 6])

    def test_types(self):
        self.assertIsInstance(self.ptss.bar_state_spaces, list)
        self.assertIsInstance(self.ptss.position, np.ndarray)
        self.assertIsInstance(self.ptss.interval, np.ndarray)
        self.assertIsInstance(self.ptss.num_states, int)
        # self.assertIsInstance(self.ptss.num_intervals, list)
        self.assertIsInstance(self.ptss.num_patterns, int)

    def test_values(self):
        self.assertTrue(np.allclose(self.ptss.bar_state_spaces[0].intervals,
                                    [1, 2, 3, 4]))
        self.assertTrue(np.allclose(self.ptss.bar_state_spaces[1].intervals,
                                    [2, 3, 4, 5, 6]))
        self.assertTrue(self.ptss.num_states == 30)
        # self.assertTrue(self.ptss.num_intervals == [4, 5])
        self.assertTrue(self.ptss.num_patterns == 2)
        # first pattern
        self.assertTrue(np.allclose(self.ptss.position[:10],
                                    [0, 0, 0.5, 0, 1. / 3, 2. / 3,
                                     0, 0.25, 0.5, 0.75]))
        self.assertTrue(np.allclose(self.ptss.interval[:10],
                                    [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]))
        self.assertTrue(np.allclose(self.ptss.pattern[:10], 0))
        # second pattern
        self.assertTrue(np.allclose(self.ptss.position[10:],
                                    [0, 0.5,
                                     0, 1. / 3, 2. / 3,
                                     0, 0.25, 0.5, 0.75,
                                     0, 0.2, 0.4, 0.6, 0.8,
                                     0, 1. / 6, 2. / 6, 0.5, 4. / 6, 5. / 6]))
        self.assertTrue(np.allclose(self.ptss.interval[10:],
                                    [0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3,
                                     4, 4, 4, 4, 4, 4]))
        self.assertTrue(np.allclose(self.ptss.pattern[10:], 1))


# transition models
class TestBeatTransitionModelClass(unittest.TestCase):

    def setUp(self):
        btss = BeatStateSpace(1, 4)
        self.tm = BeatTransitionModel(btss, 100)

    def test_types(self):
        self.assertIsInstance(self.tm, BeatTransitionModel)
        self.assertIsInstance(self.tm, TransitionModel)
        self.assertIsInstance(self.tm.state_space, BeatStateSpace)
        self.assertIsInstance(self.tm.transition_lambda, np.ndarray)
        self.assertIsInstance(self.tm.states, np.ndarray)
        self.assertIsInstance(self.tm.pointers, np.ndarray)
        self.assertIsInstance(self.tm.probabilities, np.ndarray)
        self.assertIsInstance(self.tm.log_probabilities, np.ndarray)
        self.assertIsInstance(self.tm.num_states, int)
        self.assertIsInstance(self.tm.num_transitions, int)
        self.assertTrue(self.tm.states.dtype == np.uint32)
        self.assertTrue(self.tm.pointers.dtype == np.uint32)
        self.assertTrue(self.tm.probabilities.dtype == np.float)
        self.assertTrue(self.tm.log_probabilities.dtype == np.float)

    def test_values(self):
        self.assertTrue(np.allclose(self.tm.states,
                                    [0, 2, 5, 1, 5, 9, 3, 4, 5, 9, 6, 7, 8]))
        self.assertTrue(np.allclose(self.tm.pointers,
                                    [0, 1, 3, 4, 6, 7, 8, 10, 11, 12, 13]))
        self.assertTrue(np.allclose(self.tm.probabilities,
                                    [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1]))
        self.assertTrue(np.allclose(self.tm.log_probabilities,
                                    [0, 0, -33.3333333, 0, 0, -25, 0, 0,
                                     -33.3333333, 0, 0, 0, 0]))
        self.assertTrue(self.tm.num_states == 10)
        self.assertTrue(self.tm.num_transitions == 13)


class TestPatternTrackingTransitionModelClass(unittest.TestCase):

    def setUp(self):
        ptss = MultiPatternStateSpace([1, 2], [4, 6])
        self.tm = MultiPatternTransitionModel(ptss, 100)

    def test_types(self):
        self.assertIsInstance(self.tm, MultiPatternTransitionModel)
        self.assertIsInstance(self.tm, TransitionModel)
        # self.assertIsInstance(self.tm.state_space, PatternTrackingStateSpace)
        self.assertIsInstance(self.tm.transition_lambda, list)
        self.assertIsInstance(self.tm.states, np.ndarray)
        self.assertIsInstance(self.tm.pointers, np.ndarray)
        self.assertIsInstance(self.tm.probabilities, np.ndarray)
        self.assertIsInstance(self.tm.log_probabilities, np.ndarray)
        self.assertIsInstance(self.tm.num_states, int)
        self.assertIsInstance(self.tm.num_transitions, int)
        self.assertTrue(self.tm.states.dtype == np.uint32)
        self.assertTrue(self.tm.pointers.dtype == np.uint32)
        self.assertTrue(self.tm.probabilities.dtype == np.float)
        self.assertTrue(self.tm.log_probabilities.dtype == np.float)

    def test_values(self):
        print(self.tm.probabilities)
        print(self.tm.log_probabilities)
        # the first pattern has 13 transitions
        self.assertTrue(np.allclose(self.tm.states[:13],
                                    [0, 2, 5, 1, 5, 9, 3, 4, 5, 9, 6, 7, 8]))
        self.assertTrue(np.allclose(self.tm.states[13:],
                                    [11, 14, 10, 14, 18, 12, 13, 14, 18, 23,
                                     29, 15, 16, 17, 18, 23, 29, 19, 20, 21,
                                     22, 23, 29, 24, 25, 26, 27, 28]))
        # the first pattern has 10 states (pointers has one more element)
        self.assertTrue(np.allclose(self.tm.pointers[:11],
                                    [0, 1, 3, 4, 6, 7, 8, 10, 11, 12, 13]))
        self.assertTrue(np.allclose(self.tm.pointers[11:],
                                    [15, 16, 18, 19, 20, 24, 25, 26, 27, 30,
                                     31, 32, 33, 34, 36, 37, 38, 39, 40, 41]))
        self.assertTrue(np.allclose(self.tm.probabilities,
                                    [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1,
                                     0, 1, 1, 0, 1, 1, 0, 1, 2.06e-09, 0, 1, 1,
                                     1, 0, 1, 5.78e-08, 1, 1, 1, 1, 2.06e-09,
                                     1, 1, 1, 1, 1, 1]))
        self.assertTrue(np.allclose(self.tm.log_probabilities,
                                    [0, 0, -33.3333333, 0, 0, -25, 0, 0,
                                     -33.3333333, 0, 0, 0, 0, 0,
                                     -33.3333333, 0, 0, -25, 0, 0,
                                     -33.3333333, 0, -20, -33.3333334, 0, 0,
                                     0, -25, -4.1e-09, -16.666666, 0, 0, 0,
                                     0, -20, -5.78e-08, 0, 0, 0, 0, 0]))
        self.assertTrue(self.tm.num_states == 30)
        self.assertTrue(self.tm.num_transitions == 41)


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
                                    [0, 0, 1, 0, 1, 1, 0, 1, 1, 1]))
        self.assertTrue(np.allclose(self.om.densities(self.obs),
                                    [[1, 0], [0.1, 0.3],
                                     [0.01, 0.33], [0, 1. / 3]]))
        self.assertTrue(np.allclose(self.om.log_densities(self.obs),
                                    [[0, -np.inf], [-2.30258508, -1.20397281],
                                     [-4.60517021, -1.10866262],
                                     [-np.inf, -1.09861229]]))
