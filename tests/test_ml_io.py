# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.ml.io module

"""

from __future__ import absolute_import, division, print_function

import unittest
from tempfile import NamedTemporaryFile
from nose.tools import raises
from madmom.ml.rnn import *
from madmom.ml.io import *

from . import MODELS_PATH

HDF5_MODEL_FILE = MODELS_PATH + 'simple_bdrnn.hdf5'
NPZ_MODEL_FILE = MODELS_PATH + 'simple_bdrnn.npz'


class TestLoadFunction(unittest.TestCase):

    def _check_simple_bdrnn(self, rnn):
        self.assertEqual(type(rnn), RecurrentNeuralNetwork)
        self.assertEqual(len(rnn.layers), 2)

        for hl in rnn.layers:
            self.assertEqual(type(hl), BidirectionalLayer)
            for rl in [hl.fwd_layer, hl.bwd_layer]:
                self.assertEqual(type(rl), RecurrentLayer)
                self.assertTrue(np.allclose(rl.recurrent_weights,
                                            np.zeros((3, 3))))
                self.assertTrue(np.allclose(rl.weights, np.zeros((3, 3))))
                self.assertTrue(np.allclose(rl.bias, np.zeros(3)))
                self.assertEqual(rl.transfer_fn, tanh)

    def test_load_hdf5(self):
        models = load(HDF5_MODEL_FILE)
        self.assertTrue(models.keys() == ['rnn'])
        self._check_simple_bdrnn(models['rnn'])

    def test_load_npz(self):
        models = load(NPZ_MODEL_FILE)
        self.assertTrue(models.keys() == ['rnn'])
        self._check_simple_bdrnn(models['rnn'])

    @raises(ValueError)
    def test_load_others(self):
        load(HDF5_MODEL_FILE, file_type='abc')

    def test_hdf5_to_npz(self):
        with NamedTemporaryFile() as temp:
            hdf5_to_npz(HDF5_MODEL_FILE, temp)
            models = load(temp.name, file_type='npz')
            self.assertTrue(models.keys() == ['rnn'])
            self._check_simple_bdrnn(models['rnn'])
