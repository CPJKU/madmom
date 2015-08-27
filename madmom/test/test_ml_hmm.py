# encoding: utf-8
"""
This file contains tests for the madmom.ml.hmm module.

@author: Filip Korzeniowski <filip.korzeniowski@jku.at>

"""
# pylint: skip-file

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
                     [0.3, 0.5,  0.2],
                     [0.2, 0.4,  0.4]])

OBS_SEQ = np.array([0, 2, 2, 0, 0, 1, 1, 2, 0, 2, 1, 1,
                    1, 2, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0])

CORRECT_FWD = np.array(
        [[0.6754386,  0.23684211, 0.0877193],
         [0.369291,   0.36798608, 0.26272292],
         [0.18146746, 0.33625874, 0.4822738],
         [0.35097423, 0.37533682, 0.27368895],
         [0.51780506, 0.32329768, 0.15889725],
         [0.17366244, 0.58209473, 0.24424283],
         [0.06699296, 0.58957189, 0.34343515],
         [0.05708114, 0.3428725,  0.60004636],
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
         [0.2327641,  0.47273564, 0.29450026],
         [0.42127593, 0.37947727, 0.1992468],
         [0.57132392, 0.30444215, 0.12423393],
         [0.66310201, 0.25840843, 0.07848956],
         [0.23315472, 0.59876843, 0.16807684],
         [0.43437318, 0.40024174, 0.16538507],
         [0.58171672, 0.30436365, 0.11391962]])


class TestHmmInference(unittest.TestCase):

    def setUp(self):
        frm, to, prob = zip(*TRANSITIONS)

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

    def test_forward_generator(self):
        fwd = np.vstack(list(self.hmm.forward_generator(OBS_SEQ, block_size=5)))
        self.assertTrue(np.allclose(fwd, CORRECT_FWD))
