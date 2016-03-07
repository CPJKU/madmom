# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.ml.io module

"""

from __future__ import absolute_import, division, print_function

import unittest
from nose.tools import raises
from madmom.ml.nn import *
from madmom.ml.io import *

from . import MODELS_PATH

MODEL_FILE = MODELS_PATH + 'simple_bdrnn.hdf5'


class TestLoadFunction(unittest.TestCase):

    def test_load_hdf5(self):
        models = load(MODEL_FILE)
        self.assertTrue(models.keys() == ['rnn'])

        rnn = models['rnn']
        self.assertEqual(type(rnn), RecurrentNeuralNetwork)
        self.assertEqual(len(rnn.layers), 2)

        for hl in rnn.layers:
            self.assertEqual(type(hl), layers.BidirectionalLayer)
            for rl in [hl.fwd_layer, hl.bwd_layer]:
                self.assertEqual(type(rl), layers.RecurrentLayer)
                self.assertTrue(np.allclose(rl.recurrent_weights,
                                            np.zeros((3, 3))))
                self.assertTrue(np.allclose(rl.weights, np.zeros((3, 3))))
                self.assertTrue(np.allclose(rl.bias, np.zeros(3)))
                self.assertEqual(rl.transfer_fn, transfer_fns.tanh)

    @raises(NotImplementedError)
    def test_load_npz(self):
        load(MODEL_FILE, file_type='npz')

    @raises(ValueError)
    def test_load_others(self):
        load(MODEL_FILE, file_type='abc')
