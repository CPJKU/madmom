# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.ml.nn module.

"""

from __future__ import absolute_import, division, print_function

import unittest

from madmom.models import *
from madmom.ml.nn import *


class TestNeuralNetworkClass(unittest.TestCase):

    def test_rnn(self):
        rnn = NeuralNetwork.load(ONSETS_RNN[0])
        input_size = rnn.layers[0].weights.shape[0]
        data = np.zeros((4, input_size))
        data[1] = 1.
        result = rnn.process(data)
        self.assertTrue(np.allclose(result, [1.78801871e-04, 8.00144131e-01,
                                             3.30476369e-05, 1.36037513e-04]))

    def test_brnn(self):
        rnn = NeuralNetwork.load(ONSETS_BRNN[0])
        input_size = rnn.layers[0].fwd_layer.weights.shape[0]
        data = np.zeros((4, input_size))
        data[1] = 1.
        result = rnn.process(data)
        self.assertTrue(np.allclose(result, [0.00461393, 0.46032878,
                                             0.04824624, 0.00083493]))

    def test_brnn_pp(self):
        rnn = NeuralNetwork.load(ONSETS_BRNN_PP[0])
        input_size = rnn.layers[0].fwd_layer.weights.shape[0]
        data = np.zeros((4, input_size))
        data[1] = 1.
        result = rnn.process(data)
        self.assertTrue(np.allclose(result, [3.88076517e-03, 1.67354920e-03,
                                             1.14450835e-03, 5.01533471e-05]))

    def test_brnn_regression(self):
        rnn = NeuralNetwork.load(NOTES_BRNN[0])
        input_size = rnn.layers[0].fwd_layer.weights.shape[0]
        data = np.zeros((4, input_size))
        data[1] = 1.
        result = rnn.process(data)
        self.assertEqual(result.shape, (4, 88))
        self.assertTrue(np.allclose(result[:, :2],
                                    [[6.50841586e-05, 4.06891153e-04],
                                     [-9.74552809e-04, -3.86762259e-03],
                                     [1.09878686e-04, 1.54044293e-04],
                                     [-8.16427571e-04, 4.62550714e-04]]))

    def test_blstm(self):
        rnn = NeuralNetwork.load(BEATS_BLSTM[0])
        input_size = rnn.layers[0].fwd_layer.cell.weights.shape[0]
        data = np.zeros((4, input_size))
        data[1] = 1.
        result = rnn.process(data)
        self.assertTrue(np.allclose(result, [0.0815198, 0.24451593,
                                             0.08786312, 0.01776425]))

    def test_cnn(self):
        cnn = NeuralNetwork.load(ONSETS_CNN[0])
        data = np.zeros((19, 80, 3), dtype=np.float32)
        data[10] = 1.
        result = cnn.process(data)
        self.assertTrue(np.allclose(result,
                                    [0.0026428, 0.09070455, 0.96606344,
                                     0.99829632, 0.7015394]))
