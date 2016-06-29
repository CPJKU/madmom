# encoding: utf-8
# pylint: skip-file
"""
This file contains test for the madmom.ml.crf module.

"""

from __future__ import absolute_import, division, print_function

import unittest
from madmom.ml.crf import *

eta = 0.000000000000001  # numerical stability
PI = np.log(np.array([0.6, 0.2, 0.1, 0.1], dtype=np.float64))
TAU = np.log(np.ones(4, dtype=np.float64))
C = np.log(np.ones(4, dtype=np.float64))

A = np.log(np.array([[0.8, 0.2, 0.0, 0.0],
                     [0.1, 0.6, 0.3, 0.0],
                     [0.0, 0.2, 0.7, 0.1],
                     [0.0, 0.0, 0.4, 0.6]]) + eta).astype(np.float64)

W = np.log(np.array([[0.7, 0.1, 0.2, 0.3],
                     [0.15, 0.4, 0.7, 0.1],
                     [0.15, 0.5, 0.1, 0.6]]) + eta).astype(np.float64)


def _to_onehot(seq, num_states):
    oh = np.zeros((len(seq), num_states))
    oh[range(len(seq)), seq] = 1
    return oh

OBS_SEQ_1 = _to_onehot(np.array([0, 0, 1, 0, 0, 2, 1, 0, 2, 1, 0, 1, 1, 1, 0,
                                 2, 0, 2, 0, 1, 1, 2, 0, 0, 0, 1]), 3)
OBS_SEQ_2 = _to_onehot(np.array([2, 2, 2, 2, 1, 0, 2, 0, 0, 0, 1, 1, 1, 2, 0,
                                 2, 2, 2, 0, 1, 1, 1, 1, 1, 1, 1]), 3)


class TestConditionalRandomFieldClass(unittest.TestCase):

    def setUp(self):
        self.crf = ConditionalRandomField(PI, TAU, C, A, W)

    def test_decode(self):
        correct_state_seq1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2,
                                       2, 3, 3, 3, 3, 3, 2, 2, 1, 0, 0, 0, 0])
        # correct_p_seq1 = -36.94762254
        correct_state_seq2 = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 2, 2,
                                       3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2])
        # correct_p_seq2 = -34.03217714

        state_seq = self.crf.process(OBS_SEQ_1)
        self.assertTrue((state_seq == correct_state_seq1).all())

        state_seq = self.crf.process(OBS_SEQ_2)
        self.assertTrue((state_seq == correct_state_seq2).all())
