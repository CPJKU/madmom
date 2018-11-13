# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.ml.nn module.

"""

from __future__ import absolute_import, division, print_function

import unittest

from madmom.models import *
from madmom.ml.nn import *
from madmom.ml.nn.layers import *


class TestRNNClass(unittest.TestCase):

    def setUp(self):
        # uni-directional RNN
        self.rnn = NeuralNetwork.load(ONSETS_RNN[0])
        self.data = np.zeros((4, self.rnn.layers[0].weights.shape[0]))
        self.data[1] = 1.
        self.result = [1.78801871e-04, 8.00144131e-01,
                       3.30476369e-05, 1.36037513e-04]

    def test_process(self):
        # process the whole sequence at once
        result = self.rnn.process(self.data)
        self.assertTrue(np.allclose(result, self.result))
        # two runs must produce the same output
        result_1 = self.rnn.process(self.data)
        self.assertTrue(np.allclose(result_1, self.result))
        # after resetting the RNN, it must produce the same output as before
        self.rnn.reset()
        result_2 = [self.rnn.process(np.atleast_2d(d), reset=False)
                    for d in self.data]
        self.assertTrue(np.allclose(np.hstack(result_2), self.result))
        # without resetting it produces different results
        result_3 = [self.rnn.process(np.atleast_2d(d), reset=False)
                    for d in self.data]
        self.assertTrue(np.allclose(np.hstack(result_3),
                                    [9.15636891e-04, 9.74331021e-01,
                                     4.83996118e-05, 2.72355013e-04]))


class TestLSTMClass(unittest.TestCase):
    def setUp(self):
        # uni-directional LSTM-RNN
        self.rnn = NeuralNetwork.load(BEATS_LSTM[0])
        self.data = np.zeros((4, self.rnn.layers[0].cell.weights.shape[0]))
        self.data[1] = 1.
        self.result = [0.00126955, 0.03134079, 0.01535073, 0.00207471]

    def test_process(self):
        # process the whole sequence at once
        result = self.rnn.process(self.data)
        self.assertTrue(np.allclose(result, self.result))
        # two runs must produce the same output
        result_1 = self.rnn.process(self.data)
        self.assertTrue(np.allclose(result_1, self.result))
        # after resetting the RNN, it must produce the same output
        self.rnn.reset()
        result_2 = [self.rnn.process(np.atleast_2d(d), reset=False)
                    for d in self.data]
        self.assertTrue(np.allclose(np.hstack(result_2), self.result))
        # without resetting it produces different output
        result_3 = [self.rnn.process(np.atleast_2d(d), reset=False)
                    for d in self.data]
        self.assertTrue(np.allclose(np.hstack(result_3),
                                    [0.00054101, 0.05323271,
                                     0.0548761, 0.00785541]))


# class for testing all other (offline-only) networks
class TestNeuralNetworkClass(unittest.TestCase):

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
        self.assertTrue(np.allclose(result,
                                    [3.88076517e-03, 1.67354920e-03,
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
        data = np.zeros((19, 80, 3))
        data[10] = 1.
        result = cnn.process(data)
        self.assertTrue(np.allclose(result, [0.0021432, 0.02647826, 0.92750794,
                                             0.84207922, 0.21631248]))


class TestFeedForwardLayerClass(unittest.TestCase):

    def setUp(self):
        # borrow an FeedForwardLayer from an existing network
        rnn = NeuralNetwork.load(ONSETS_RNN[0])
        self.layer = rnn.layers[-1]
        input_size = self.layer.weights.shape[0]
        self.data = np.zeros((4, input_size))
        self.data[1] = 1.
        self.result = np.array([[0.16283005], [0.14362903],
                                [0.16283005], [0.16283005]])

    def test_types(self):
        self.assertTrue(isinstance(self.layer, FeedForwardLayer))
        self.assertTrue(self.layer.activation_fn == sigmoid)

    def test_init(self):
        self.assertFalse(hasattr(self.layer, 'init'))
        self.assertFalse(hasattr(self.layer, '_prev'))

    def test_activate(self):
        # test result
        result_1 = self.layer(self.data)
        self.assertTrue(np.allclose(result_1, self.result))
        # two runs must yield identical results
        result_2 = self.layer(self.data)
        self.assertTrue(np.allclose(result_1, result_2))
        # calling frame-by-frame must yield identical results
        result_3 = [self.layer.activate(d) for d in self.data]
        self.assertTrue(np.allclose(result_3, self.result))
        # calling frame-by-frame without resetting also
        result_4 = [self.layer.activate(d, reset=False) for d in self.data]
        self.assertTrue(np.allclose(result_4, self.result))


class TestRecurrentLayerClass(unittest.TestCase):

    def setUp(self):
        # borrow an RecurrentLayer from an existing network
        rnn = NeuralNetwork.load(ONSETS_RNN[0])
        self.layer = rnn.layers[0]
        input_size = self.layer.weights.shape[0]
        self.data = np.zeros((4, input_size))
        self.data[1] = 1.

    def test_types(self):
        self.assertTrue(isinstance(self.layer, RecurrentLayer))
        self.assertTrue(self.layer.activation_fn == tanh)

    def test_init(self):
        self.assertTrue(hasattr(self.layer, 'init'))
        self.assertTrue(hasattr(self.layer, '_prev'))

    def test_activate(self):
        # test result
        result_1 = self.layer(self.data)
        self.assertTrue(np.allclose(result_1[0, :2],
                                    [-0.33919713, -0.02091585]))
        self.assertTrue(np.allclose(result_1[-1, -2:],
                                    [0.4419672, -0.261151]))
        # two runs must yield identical results
        result_2 = self.layer(self.data)
        self.assertTrue(np.allclose(result_2, result_1))
        # last step must be preserved
        self.assertTrue(np.allclose(self.layer._prev, result_2[-1]))
        # initialisation must not change
        self.assertTrue(np.allclose(self.layer.init, np.zeros(25)))
        # reset layer, activate framewise must yield the same result
        self.layer.reset()
        result_3 = [self.layer.activate(np.atleast_2d(d), reset=False)
                    for d in self.data]
        result_3 = np.vstack(result_3)
        self.assertTrue(np.allclose(result_3, result_1))
        self.assertTrue(np.allclose(self.layer._prev, result_3[-1]))
        # activate framewise without resetting must yield a different result
        result_4 = [self.layer.activate(np.atleast_2d(d), reset=False)
                    for d in self.data]
        result_4 = np.vstack(result_4)
        self.assertFalse(np.allclose(result_4, result_1))
        self.assertTrue(np.allclose(result_4[0, :2],
                                    [-3.14807342e-01, -2.22700375e-01]))
        self.assertTrue(np.allclose(result_4[-1, -2:],
                                    [4.40259654e-01, -2.59141315e-01]))
        # last step must be preserved
        self.assertTrue(np.allclose(self.layer._prev, result_4[-1]))
        # initialisation must not change
        self.assertTrue(np.allclose(self.layer.init, np.zeros(25)))


class TestLSTMLayerClass(unittest.TestCase):

    def setUp(self):
        # borrow an LSTMLayer from an existing network
        rnn = NeuralNetwork.load(BEATS_BLSTM[0])
        self.layer = rnn.layers[0].fwd_layer
        input_size = self.layer.cell.weights.shape[0]
        self.data = np.zeros((4, input_size))
        self.data[1] = 1.

    def test_types(self):
        self.assertTrue(isinstance(self.layer, LSTMLayer))
        self.assertTrue(isinstance(self.layer.input_gate, Gate))
        self.assertTrue(isinstance(self.layer.forget_gate, Gate))
        self.assertTrue(isinstance(self.layer.cell, Cell))
        self.assertTrue(isinstance(self.layer.output_gate, Gate))
        self.assertTrue(self.layer.activation_fn == tanh)
        self.assertTrue(self.layer.input_gate.activation_fn == sigmoid)
        self.assertTrue(self.layer.forget_gate.activation_fn == sigmoid)
        self.assertTrue(self.layer.cell.activation_fn == tanh)
        self.assertTrue(self.layer.output_gate.activation_fn == sigmoid)

    def test_init(self):
        self.assertTrue(hasattr(self.layer, 'init'))
        self.assertTrue(hasattr(self.layer, '_prev'))

    def test_activate(self):
        # test result
        result_1 = self.layer(self.data)
        self.assertTrue(np.allclose(result_1[0, :2],
                                    [8.15829188e-02, 1.14209838e-02]))
        self.assertTrue(np.allclose(result_1[-1, -2:],
                                    [-1.81968838e-01, -2.32963227e-02]))
        # two runs must yield identical results
        result_2 = self.layer(self.data)
        self.assertTrue(np.allclose(result_2, result_1))
        # last step must be preserved
        self.assertTrue(np.allclose(self.layer._prev, result_2[-1]))
        # reset layer, activate framewise must yield the same result
        self.layer.reset()
        result_3 = [self.layer.activate(np.atleast_2d(d), reset=False)
                    for d in self.data]
        self.assertTrue(np.allclose(np.vstack(result_3), result_1))
        # activate framewise without resetting must yield a different result
        result_4 = [self.layer.activate(np.atleast_2d(d), reset=False)
                    for d in self.data]
        result_4 = np.vstack(result_4)
        self.assertFalse(np.allclose(result_4, result_1))
        self.assertTrue(np.allclose(result_4[0, :2],
                                    [2.34878659e-01, -1.26488507e-03]))
        self.assertTrue(np.allclose(result_4[-1, -2:],
                                    [-2.45573238e-01, -2.92062219e-02]))
        # last step must be preserved
        self.assertTrue(np.allclose(self.layer._prev, result_4[-1]))
        # initialisation must not change
        self.assertTrue(np.allclose(self.layer.init, np.zeros(25)))


class TestGRUClass(unittest.TestCase):

    W_xr = np.array([[-0.42948743, -1.29989187],
                     [0.77213901, 0.86070993],
                     [1.13791823, -0.87066225]])
    W_xu = np.array([[0.44875312, 0.07172084],
                     [-0.24292999, 1.318794],
                     [1.0270179, 0.16293946]])
    W_xhu = np.array([[0.8812559, 1.35859991],
                      [1.04311944, -0.25449358],
                      [-1.09539597, 1.19808424]])
    W_hr = np.array([[0.96696973, 0.1384294],
                     [-0.09561655, -1.23413809]])
    W_hu = np.array([[0.04664641, 0.59561686],
                     [1.00325841, -0.11574791]])
    W_hhu = np.array([[1.19742848, 1.07850016],
                      [0.35234964, -1.45348681]])
    b_r = np.array([1.41851288, -0.39743243])
    b_u = np.array([-0.78729095, 0.83385797])
    b_hu = np.array([1.25143065, -0.97715625])

    IN = np.array([[0.91298812, -1.47626202, -1.08667502],
                   [0.49814883, -0.0104938, 0.93869008],
                   [-1.12282135, 0.3780883, 1.42017503],
                   [0.62669439, 0.89438929, -0.69354132],
                   [0.16162221, -1.00166208, 0.23579985]])
    OUT = np.array([[0.22772433, -0.13181415],
                    [0.49479958, 0.51224858],
                    [0.08539771, -0.56119639],
                    [0.1946809, -0.50421363],
                    [0.17403202, -0.27258521]])
    H = np.array([0.02345737, 0.34454183])

    def setUp(self):
        self.reset_gate = layers.Gate(
            TestGRUClass.W_xr, TestGRUClass.b_r, TestGRUClass.W_hr,
            activation_fn=activations.sigmoid)
        self.update_gate = layers.Gate(
            TestGRUClass.W_xu, TestGRUClass.b_u, TestGRUClass.W_hu,
            activation_fn=activations.sigmoid)
        self.gru_cell = layers.GRUCell(
            TestGRUClass.W_xhu, TestGRUClass.b_hu, TestGRUClass.W_hhu)
        self.gru_1 = layers.GRULayer(self.reset_gate, self.update_gate,
                                     self.gru_cell)
        self.gru_2 = layers.GRULayer(self.reset_gate, self.update_gate,
                                     self.gru_cell, init=TestGRUClass.H)

    def test_activate(self):
        self.assertTrue(
            np.allclose(self.reset_gate.activate(TestGRUClass.IN[0, :],
                        TestGRUClass.H), np.array([0.20419282, 0.08861294])))
        self.assertTrue(
            np.allclose(self.update_gate.activate(TestGRUClass.IN[0, :],
                        TestGRUClass.H), np.array([0.31254834, 0.2226105])))
        self.assertTrue(
            np.allclose(self.gru_cell.activate(TestGRUClass.IN[0, :],
                        TestGRUClass.H, TestGRUClass.H),
                        np.array([0.9366396, -0.67876764])))
        # activating the layer normally
        self.assertTrue(np.allclose(self.gru_1.activate(TestGRUClass.IN),
                                    TestGRUClass.OUT))
        # activating the layer a second time must give the same results
        self.assertTrue(np.allclose(self.gru_1.activate(TestGRUClass.IN),
                                    TestGRUClass.OUT))
        # activating the other layer
        self.assertTrue(
            np.allclose(self.gru_2.activate(TestGRUClass.IN),
                        np.array([[0.30988133, 0.13258138],
                                  [0.60639685, 0.55714613],
                                  [0.21366976, -0.55568963],
                                  [0.30860096, -0.43686554],
                                  [0.28866628, -0.23025239]])))
        # reset layer, activate framewise must yield the same result
        self.gru_1.reset()
        result_1 = [self.gru_1.activate(np.atleast_2d(d), reset=False)
                    for d in self.IN]
        self.assertTrue(np.allclose(np.vstack(result_1), self.OUT))
        # the previous state must be the last output
        self.assertTrue(np.allclose(self.gru_1._prev, self.OUT[-1]))
        # activate with same data without resetting
        result_2 = [self.gru_1.activate(np.atleast_2d(d), reset=False)
                    for d in self.IN]
        # results must differ
        self.assertFalse(np.allclose(result_1, result_2))
        self.assertTrue(np.allclose(np.vstack(result_2),
                                    [[0.3254016, -0.33195618],
                                     [0.53048187, 0.51293057],
                                     [0.12495148, -0.559039],
                                     [0.22988503, -0.48455557],
                                     [0.20934506, -0.25959042]]))
        # the previous state must be the last output
        self.assertTrue(np.allclose(self.gru_1._prev,
                                    [0.20934506, -0.25959042]))
        # initialisation must not change
        self.assertTrue(np.allclose(self.gru_1.init, [0, 0]))


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


class TestAverageLayerClass(unittest.TestCase):

    IN = np.array([[[0.32400414, 0.31483042],
                    [0.38269293, 0.04822304],
                    [0.03791266, 0.34776369]],
                   [[0.87113619, 0.62172854],
                    [0.87353969, 0.92837042],
                    [0.70359915, 0.49917081]],
                   [[0.42643583, 0.74653631],
                    [0.08519834, 0.35423595],
                    [0.34863797, 0.44895086]]])

    OUT_AVG = 0.46460927444444444
    OUT_02 = np.array([0.55077857, 0.44537673, 0.39767252])
    OUT_02_KD = np.array([[[0.55077857], [0.44537673], [0.39767252]]])

    def test_average_layer(self):
        al = layers.AverageLayer()
        out = al(TestAverageLayerClass.IN)
        self.assertAlmostEqual(out, TestAverageLayerClass.OUT_AVG)

        al = layers.AverageLayer(axis=(0, 2))
        out = al(TestAverageLayerClass.IN)
        self.assertEqual(out.shape, (3,))
        self.assertTrue(np.allclose(out, TestAverageLayerClass.OUT_02))

        al = layers.AverageLayer(axis=(0, 2), keepdims=True)
        out = al(TestAverageLayerClass.IN)
        self.assertEqual(out.shape, (1, 3, 1))
        self.assertTrue(np.allclose(out, TestAverageLayerClass.OUT_02_KD))

        al = layers.AverageLayer(axis=(0, 2), dtype=np.float32)
        out = al(TestAverageLayerClass.IN)
        self.assertEqual(out.dtype, np.float32)


class TestReshapeLayerClass(unittest.TestCase):

    IN = np.random.random((2, 3, 4))

    def test_reshape_layer(self):
        rl = layers.ReshapeLayer(newshape=(3, 4, 2))
        self.assertEqual(rl(TestReshapeLayerClass.IN).shape, (3, 4, 2))
        rl = layers.ReshapeLayer(newshape=(3, -1, 2))
        self.assertEqual(rl(TestReshapeLayerClass.IN).shape, (3, 4, 2))
        rl = layers.ReshapeLayer(newshape=(-1,))
        self.assertEqual(rl(TestReshapeLayerClass.IN).shape, (24,))

        with self.assertRaises(ValueError):
            rl = layers.ReshapeLayer(newshape=(3, 2, 2))
            rl(TestReshapeLayerClass.IN)


class TestTransposeLayerClass(unittest.TestCase):

    IN = np.random.random((2, 3, 4, 5))

    def test_transpose_layer(self):
        tl = layers.TransposeLayer()
        self.assertEqual(tl(TestTransposeLayerClass.IN).shape, (5, 4, 3, 2))

        tl = layers.TransposeLayer(axes=(2, 0, 1, 3))
        self.assertEqual(tl(TestTransposeLayerClass.IN).shape, (4, 2, 3, 5))

        with self.assertRaises(ValueError):
            tl = layers.TransposeLayer(axes=(0, 1, 3))
            tl(TestTransposeLayerClass.IN)

        with self.assertRaises(ValueError):
            tl = layers.TransposeLayer(axes=(0, 1, 2, 3, 4))
            tl(TestTransposeLayerClass.IN)

        with self.assertRaises(ValueError):
            tl = layers.TransposeLayer(axes=(0, 1, 1, 2))
            tl(TestTransposeLayerClass.IN)


class TestPadLayerClass(unittest.TestCase):

    def test_constant_padding(self):
        pl = layers.PadLayer(width=2, axes=(0, 1), value=10.)
        data = np.arange(40).reshape(5, 4, 2).astype(np.float)
        out = pl(data)

        self.assertEqual(out.shape, (9, 8, 2))
        self.assertTrue(np.allclose(out[2:-2, 2:-2, :], data))
        self.assertTrue(np.allclose(out[:2, :, :], 10.))
        self.assertTrue(np.allclose(out[-2:, :, :], 10.))
        self.assertTrue(np.allclose(out[:, :2, :], 10.))
        self.assertTrue(np.allclose(out[:, -2:, :], 10.))

        pl = layers.PadLayer(width=3, axes=(2,), value=2.2)
        out = pl(data)

        self.assertEqual(out.shape, (5, 4, 8))
        self.assertTrue(np.allclose(out[:, :, 3:-3], data))
        self.assertTrue(np.allclose(out[:, :, :3], 2.2))
        self.assertTrue(np.allclose(out[:, :, -3:], 2.2))


class ConvolutionalLayerClassTest(unittest.TestCase):

    W1 = np.array([[[[0.69557322]],
                    [[0.45649655]],
                    [[0.58179561]]],
                   [[[0.20438251]],
                    [[0.17404747]],
                    [[0.41624290]]]])

    W3 = np.array([[[[0.57353216, 0.72422232, 0.15716315],
                     [0.82000373, 0.26902348, 0.69203708],
                     [0.45564084, 0.89265194, 0.98080186]],
                    [[0.44920649, 0.52442715, 0.33103038],
                     [0.24536095, 0.49307102, 0.28850389],
                     [0.38324254, 0.46965330, 0.76865911]],
                    [[0.44225901, 0.34989312, 0.92381997],
                     [0.32123710, 0.04856574, 0.87387125],
                     [0.70175767, 0.38149251, 0.40178089]]],
                   [[[0.28197446, 0.35315104, 0.53862099],
                     [0.01224023, 0.94672135, 0.87194315],
                     [0.69193064, 0.27611521, 0.51076897]],
                    [[0.22228372, 0.58605351, 0.17730248],
                     [0.10949298, 0.43124835, 0.71336330],
                     [0.57694486, 0.44623928, 0.11774881]],
                    [[0.76850363, 0.46740177, 0.76900027],
                     [0.61551742, 0.62841514, 0.05235070],
                     [0.01321052, 0.93591818, 0.61256317]]]])

    B = np.array([0.27614033, 0.87995416, 0.23540803])

    O1 = np.array([[[0.97171354, 1.33645070, 0.81720366],
                    [0.27614033, 0.87995416, 0.23540803],
                    [0.27614033, 0.87995416, 0.23540803],
                    [0.27614033, 0.87995416, 0.23540803],
                    [0.48052284, 1.05400163, 0.65165093]],
                   [[0.27614033, 0.87995416, 0.23540803],
                    [0.97171354, 1.33645070, 0.81720366],
                    [0.27614033, 0.87995416, 0.23540803],
                    [0.48052284, 1.05400163, 0.65165093],
                    [0.27614033, 0.87995416, 0.23540803]],
                   [[0.27614033, 0.87995416, 0.23540803],
                    [0.27614033, 0.87995416, 0.23540803],
                    [1.17609602, 1.51049817, 1.23344656],
                    [0.27614033, 0.87995416, 0.23540803],
                    [0.27614033, 0.87995416, 0.23540803]],
                   [[0.27614033, 0.87995416, 0.23540803],
                    [0.48052284, 1.05400163, 0.65165093],
                    [0.27614033, 0.87995416, 0.23540803],
                    [0.97171354, 1.33645070, 0.81720366],
                    [0.27614033, 0.87995416, 0.23540803]],
                   [[0.48052284, 1.05400163, 0.65165093],
                    [0.27614033, 0.87995416, 0.23540803],
                    [0.27614033, 0.87995416, 0.23540803],
                    [0.27614033, 0.87995416, 0.23540803],
                    [0.97171354, 1.33645070, 0.81720366]]])

    O3 = np.array([[[2.38147223, 2.81317455, 1.89651736],
                    [2.05779099, 2.38843173, 2.54209157],
                    [2.61057651, 2.39648026, 2.56985398]],
                   [[2.35418737, 2.29051489, 2.02105685],
                    [4.27677071, 3.77638656, 2.53863951],
                    [2.84045803, 2.85248774, 2.44744130]],
                   [[2.90905416, 2.44869238, 2.34779163],
                    [3.13685429, 2.75457102, 1.92640647],
                    [2.61026680, 2.70863968, 1.74057683]]])

    def setUp(self):
        self.layer1x1 = ConvolutionalLayer(ConvolutionalLayerClassTest.W1,
                                           ConvolutionalLayerClassTest.B)
        self.layer3x3 = ConvolutionalLayer(ConvolutionalLayerClassTest.W3,
                                           ConvolutionalLayerClassTest.B)
        self.data = np.stack([np.eye(5), np.eye(5)[:, ::-1]], axis=-1)

    def test_activate(self):
        out1 = self.layer1x1.activate(self.data)
        self.assertTrue(np.allclose(out1, ConvolutionalLayerClassTest.O1))
        out3 = self.layer3x3.activate(self.data)
        self.assertTrue(np.allclose(out3, ConvolutionalLayerClassTest.O3))
