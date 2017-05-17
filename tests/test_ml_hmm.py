# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.ml.hmm module.

"""

from __future__ import absolute_import, division, print_function

import sys
import unittest
from madmom.ml.hmm import *


PRIOR = np.array([0.6, 0.2, 0.2])

TRANSITIONS = [(0, 0, 0.7),
               (0, 1, 0.3),
               (1, 0, 0.1),
               (1, 1, 0.6),
               (1, 2, 0.3),
               (2, 1, 0.3),
               (2, 2, 0.7)]

OBS_PROB = np.array([[0.7, 0.15, 0.15],
                     [0.3, 0.5, 0.2],
                     [0.2, 0.4, 0.4]])

OBS_SEQ = np.array([0, 2, 2, 0, 0, 1, 1, 2, 0, 2, 1, 1,
                    1, 2, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0])

CORRECT_FWD = np.array([[0.6754386, 0.23684211, 0.0877193],
                        [0.369291, 0.36798608, 0.26272292],
                        [0.18146746, 0.33625874, 0.4822738],
                        [0.35097423, 0.37533682, 0.27368895],
                        [0.51780506, 0.32329768, 0.15889725],
                        [0.17366244, 0.58209473, 0.24424283],
                        [0.06699296, 0.58957189, 0.34343515],
                        [0.05708114, 0.3428725, 0.60004636],
                        [0.18734426, 0.43567034, 0.3769854],
                        [0.09699435, 0.31882203, 0.58418362],
                        [0.03609747, 0.47711943, 0.4867831],
                        [0.02569311, 0.52002881, 0.45427808],
                        [0.02452257, 0.53259115, 0.44288628],
                        [0.03637171, 0.31660931, 0.64701899],
                        [0.02015006, 0.46444741, 0.51540253],
                        [0.02118133, 0.51228818, 0.46653049],
                        [0.16609052, 0.48889238, 0.3450171],
                        [0.06141349, 0.55365814, 0.38492837],
                        [0.2327641, 0.47273564, 0.29450026],
                        [0.42127593, 0.37947727, 0.1992468],
                        [0.57132392, 0.30444215, 0.12423393],
                        [0.66310201, 0.25840843, 0.07848956],
                        [0.23315472, 0.59876843, 0.16807684],
                        [0.43437318, 0.40024174, 0.16538507],
                        [0.58171672, 0.30436365, 0.11391962]])


class TestTransitionModelClass(unittest.TestCase):

    def setUp(self):
        frm, to, prob = list(zip(*TRANSITIONS))
        self.tm = TransitionModel.from_dense(to, frm, prob)

    def test_types(self):
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
        self.assertTrue(np.allclose(self.tm.states, [0, 1, 0, 1, 2, 1, 2]))
        self.assertTrue(np.allclose(self.tm.pointers, [0, 2, 5, 7]))
        self.assertTrue(np.allclose(self.tm.probabilities,
                                    [0.7, 0.1, 0.3, 0.6, 0.3, 0.3, 0.7]))
        log_prob = [-0.35667494, -2.30258509, -1.2039728, -0.51082562,
                    -1.2039728, -1.2039728, -0.35667494]
        self.assertTrue(np.allclose(self.tm.log_probabilities, log_prob))
        self.assertTrue(self.tm.num_states == 3)
        self.assertTrue(self.tm.num_transitions == 7)

    def test_num_states_unreachable(self):
        for r in range(3):
            trans = np.array([[.5, .5, .0],
                              [.5, .5, .0],
                              [.5, .5, .0]])
            trans = np.roll(trans, shift=r, axis=1)
            frm, to = trans.nonzero()
            tm = TransitionModel.from_dense(to, frm, trans[frm, to])
            self.assertTrue(tm.num_states == 3)


class TestDiscreteObservationModelClass(unittest.TestCase):

    def setUp(self):
        self.om = DiscreteObservationModel(OBS_PROB)

    def test_types(self):
        self.assertIsInstance(self.om.pointers, np.ndarray)
        self.assertIsInstance(self.om.densities(OBS_SEQ), np.ndarray)
        self.assertIsInstance(self.om.log_densities(OBS_SEQ), np.ndarray)
        self.assertTrue(self.om.pointers.dtype == np.uint32)
        self.assertTrue(self.om.densities(OBS_SEQ).dtype == np.float)
        self.assertTrue(self.om.log_densities(OBS_SEQ).dtype == np.float)

    def test_values(self):
        self.assertTrue(np.allclose(self.om.pointers, [0, 1, 2]))
        self.assertTrue(np.allclose(self.om.observation_probabilities,
                                    OBS_PROB))
        self.assertTrue(np.allclose(self.om.densities(OBS_SEQ),
                                    OBS_PROB[:, OBS_SEQ].T))
        self.assertTrue(np.allclose(self.om.log_densities(OBS_SEQ),
                                    np.log(OBS_PROB[:, OBS_SEQ].T)))


class TestHiddenMarkovModelClass(unittest.TestCase):

    def setUp(self):
        frm, to, prob = list(zip(*TRANSITIONS))
        tm = TransitionModel.from_dense(to, frm, prob)
        om = DiscreteObservationModel(OBS_PROB)
        self.hmm = HiddenMarkovModel(tm, om, PRIOR)

    def test_viterbi(self):
        correct_state_seq = np.array([0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2,
                                      2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        correct_log_p = -35.2104311327
        state_seq, log_p = self.hmm.viterbi(OBS_SEQ)
        self.assertTrue((state_seq == correct_state_seq).all())
        self.assertAlmostEqual(log_p, correct_log_p)

    def test_forward(self):
        fwd = self.hmm.forward(OBS_SEQ)
        self.assertTrue(np.allclose(fwd, CORRECT_FWD))
        # two runs must yield identical results
        fwd = self.hmm.forward(OBS_SEQ)
        self.assertTrue(np.allclose(fwd, CORRECT_FWD))
        # after resetting the HMM, it must produce the same output as before
        self.hmm.reset()
        fwd = np.vstack([self.hmm.forward(np.atleast_1d(o), reset=False)
                         for o in OBS_SEQ])
        self.assertTrue(np.allclose(fwd, CORRECT_FWD))
        # without resetting it produces different results
        fwd = np.vstack([self.hmm.forward(np.atleast_1d(o), reset=False)
                         for o in OBS_SEQ])
        self.assertFalse(np.allclose(fwd, CORRECT_FWD))
        # after resetting it must yield the correct result again
        self.hmm.reset()
        fwd = np.vstack([self.hmm.forward(np.atleast_1d(o), reset=False)
                         for o in OBS_SEQ])
        self.assertTrue(np.allclose(fwd, CORRECT_FWD))
        # initialisation must not change
        self.assertTrue(np.allclose(self.hmm.initial_distribution, PRIOR))

    def test_forward_generator(self):
        fwd = np.vstack(self.hmm.forward_generator(OBS_SEQ, block_size=5))
        self.assertTrue(np.allclose(fwd, CORRECT_FWD))

    def test_invalid_sequence(self):
        transitions = [(0, 0, 0.1), (0, 1, 0.9), (0, 2, 0),
                       (1, 0, 0), (1, 1, 1), (1, 2, 0),
                       (2, 0, 0), (2, 1, 0), (2, 2, 1)]
        frm, to, prob = list(zip(*transitions))
        tm = TransitionModel.from_dense(to, frm, prob)
        obs_prob = np.array([[0.7, 0.3, 0],
                             [0.3, 0.7, 0],
                             [0, 0, 1]])
        om = DiscreteObservationModel(obs_prob)
        hmm = HiddenMarkovModel(tm, om)
        state_seq, log_p = hmm.viterbi([0, 1, 0, 2])
        self.assertTrue(np.allclose(state_seq, []))
        self.assertAlmostEqual(log_p, -np.inf)
        # TODO: assertWarns exist only for Python 3.2+, test in all versions
        if sys.version_info >= (3, 2):
            with self.assertWarns(RuntimeWarning):
                hmm.viterbi([0, 1, 0, 2])
        state_seq, log_p = hmm.viterbi([0, 0, 1, 1])
        self.assertTrue((state_seq == [1, 1, 1, 1]).all())
        self.assertAlmostEqual(log_p, -4.219907785197447)
