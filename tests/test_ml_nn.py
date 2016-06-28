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


class TestBatchNormLayerClass(unittest.TestCase):

    IN = np.array([[[0.32400414, 0.31483042],
                    [0.38269293, 0.04822304],
                    [0.03791266, 0.34776369]],
                   [[0.87113619, 0.62172854],
                    [0.87353969, 0.92837042],
                    [0.70359915, 0.49917081]],
                   [[0.42643583, 0.74653631],
                    [0.08519834, 0.35423595],
                    [0.34863797, 0.44895086]]])

    BN_PARAMS = [np.array([-0.00098404, 0.00185387]),
                 np.array([-0.0068268, 0.0068859]),
                 np.array([-0.00289366, 0.00742069]),
                 np.array([-0.00177374, -0.00444383])]

    OUT = np.array([[[-0.00098008, 0.00184446],
                     [-0.00097937, 0.00185262],
                     [-0.00098355, 0.00184345]],
                    [[-0.00097346, 0.00183507],
                     [-0.00097343, 0.00182569],
                     [-0.00097549, 0.00183882]],
                    [[-0.00097884, 0.00183125],
                     [-0.00098297, 0.00184326],
                     [-0.00097978, 0.00184036]]])

    def test_batch_norm(self):
        params = TestBatchNormLayerClass.BN_PARAMS
        x = TestBatchNormLayerClass.IN
        y_true = TestBatchNormLayerClass.OUT
        bnl = layers.BatchNormLayer(
            params[0], params[1], params[2], params[3], activations.linear)

        y = bnl.activate(x)

        self.assertTrue(np.allclose(y, y_true))
