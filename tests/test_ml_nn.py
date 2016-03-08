# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.ml.nn module.

"""

from __future__ import absolute_import, division, print_function

import unittest

from madmom import MODELS_PATH
from madmom.ml.nn import *

ONSETS_RNN = "%s/onsets/2013/onsets_rnn_1.pkl" % MODELS_PATH
ONSETS_BRNN = "%s/onsets/2013/onsets_brnn_1.pkl" % MODELS_PATH
ONSETS_BRNN_PP = "%s/onsets/2014/onsets_brnn_pp_1.pkl" % MODELS_PATH
NOTES_BRNN = "%s/notes/2013/notes_brnn.pkl" % MODELS_PATH
BEATS_BLSTM = "%s/beats/2013/beats_blstm_1.pkl" % MODELS_PATH


class TestNeuralNetworkClass(unittest.TestCase):

    def test_rnn(self):
        rnn = NeuralNetwork.load(ONSETS_RNN)
        input_size = rnn.layers[0].weights.shape[0]
        data = np.zeros((4, input_size))
        data[1] = 1.
        result = rnn.process(data)
        self.assertTrue(np.allclose(result, [1.78801871e-04, 8.00144131e-01,
                                             3.30476369e-05, 1.36037513e-04]))

    def test_brnn(self):
        rnn = NeuralNetwork.load(ONSETS_BRNN)
        input_size = rnn.layers[0].fwd_layer.weights.shape[0]
        data = np.zeros((4, input_size))
        data[1] = 1.
        result = rnn.process(data)
        self.assertTrue(np.allclose(result, [0.00461393, 0.46032878,
                                             0.04824624, 0.00083493]))

    def test_brnn_pp(self):
        rnn = NeuralNetwork.load(ONSETS_BRNN_PP)
        input_size = rnn.layers[0].fwd_layer.weights.shape[0]
        data = np.zeros((4, input_size))
        data[1] = 1.
        result = rnn.process(data)
        self.assertTrue(np.allclose(result, [3.88076517e-03, 1.67354920e-03,
                                             1.14450835e-03, 5.01533471e-05]))

    def test_brnn_regression(self):
        rnn = NeuralNetwork.load(NOTES_BRNN)
        input_size = rnn.layers[0].fwd_layer.weights.shape[0]
        data = np.zeros((4, input_size))
        data[1] = 1.
        result = rnn.process(data)
        self.assertEqual(result.shape, (4, 88))
        print(result[:, :2])
        self.assertTrue(np.allclose(result[:, :2],
                                    [[6.50841586e-05, 4.06891153e-04],
                                     [-9.74552809e-04, -3.86762259e-03],
                                     [1.09878686e-04, 1.54044293e-04],
                                     [-8.16427571e-04, 4.62550714e-04]]))

    def test_blstm(self):
        rnn = NeuralNetwork.load(BEATS_BLSTM)
        input_size = rnn.layers[0].fwd_layer.cell.weights.shape[0]
        print(input_size)
        data = np.zeros((4, input_size), dtype=np.float32)
        data[1] = 1.
        result = rnn.process(data)
        self.assertTrue(np.allclose(result, [0.01389176, 0.12165674,
                                             0.01439718, 0.00706945]))
