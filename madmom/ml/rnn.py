#!/usr/bin/env python
# encoding: utf-8
"""
This file contains recurrent neural network (RNN) related functionality.

It's main purpose is to serve as a substitute for testing neural networks
which were trained by other ML packages or programs without requiring these
packages or programs as dependencies.

The only allowed dependencies are Python + numpy + scipy.

The structure reflects just the needed functionality for testing networks. This
module is not meant to be a general purpose RNN with lots of functionality.
Just use one of the many NN/ML packages out there if you need training or any
other stuff.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""
import abc
import numpy as np

from madmom import Processor, ParallelProcessor

# naming infix for bidirectional layer
REVERSE = 'reverse'


# transfer functions
def linear(x, out=None):
    """
    Linear function.

    :param x:   input data
    :param out: numpy array to hold the output data
    :return:    unaltered input data

    """
    if out is None or x is out:
        return x
    out[:] = x
    return out


tanh = np.tanh


def _sigmoid(x, out=None):
    """
    Logistic sigmoid function.

    :param x:   input data
    :param out: numpy array to hold the output data
    :return:    logistic sigmoid of input data

    """
    # sigmoid = 0.5 * (1. + np.tanh(0.5 * x))
    if out is None:
        out = .5 * x
    else:
        if out is not x:
            out[:] = x
        out *= .5
    np.tanh(out, out)
    out += 1
    out *= .5
    return out

try:
    # try to use a faster sigmoid function
    from distutils.version import LooseVersion
    from scipy.version import version as scipy_version
    # we need a recent version of scipy, older have a bug in expit
    # https://github.com/scipy/scipy/issues/3385
    if LooseVersion(scipy_version) < LooseVersion("0.14"):
        raise ImportError
    from scipy.special import expit as sigmoid
except ImportError:
    sigmoid = _sigmoid


# network layer classes
class Layer(object):
    """
    Generic network Layer.

    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def activate(self, data):
        """
        Activate the layer.

        :param data: activate with this data
        :return:     activations for this data

        """
        return

    @abc.abstractproperty
    def input_size(self):
        """
        Input size of the layer.

        """
        return

    @abc.abstractproperty
    def output_size(self):
        """
        Output size of the layer.

        """
        return


class BidirectionalLayer(Layer):
    """
    Bidirectional network layer.

    """
    def __init__(self, fwd_layer, bwd_layer):
        """
        Create a new BidirectionalLayer.

        :param fwd_layer: forward layer
        :param bwd_layer: backward layer

        """
        self.fwd_layer = fwd_layer
        self.bwd_layer = bwd_layer

    def activate(self, data):
        """
        Activate the layer.

        :param data: activate with this data
        :return:     activations for this data

        """
        # activate in forward direction
        fwd = self.fwd_layer.activate(data)
        # also activate with reverse input
        bwd = self.bwd_layer.activate(data[::-1])
        # stack data
        return np.hstack((bwd[::-1], fwd))

    @property
    def input_size(self):
        """
        Input size of the layer.

        """
        # the input sizes of the forward and backward layer must match
        assert self.fwd_layer.input_size == self.bwd_layer.input_size
        return self.fwd_layer.input_size

    @property
    def output_size(self):
        """
        Output size of the layer.

        """
        return self.fwd_layer.output_size + self.bwd_layer.output_size


class FeedForwardLayer(Layer):
    """
    Feed-forward network layer.

    """
    def __init__(self, transfer_fn, weights, bias):
        """
        Create a new Layer.

        :param transfer_fn: transfer function
        :param weights:     weights (2D matrix)
        :param bias:        bias (1D vector or scalar)

        Note: The transfer function needs to support the numpy ufunc out
              argument.

        """
        self.transfer_fn = transfer_fn
        self.weights = np.copy(weights)
        self.bias = bias.flatten('A')

    def activate(self, data):
        """
        Activate the layer.

        :param data: activate with this data
        :return:     activations for this data

        """
        # weight the data, add bias and apply transfer function
        return self.transfer_fn(np.dot(data, self.weights) + self.bias)

    @property
    def input_size(self):
        """
        Output size of the layer.

        """
        return self.weights.shape[0]

    @property
    def output_size(self):
        """
        Output size of the layer.

        """
        return self.weights.shape[1]


class RecurrentLayer(FeedForwardLayer):
    """
    Recurrent network layer.

    """
    def __init__(self, transfer_fn, weights, bias, recurrent_weights=None):
        """
        Create a new Layer.

        :param transfer_fn:       transfer function
        :param weights:           weights (2D matrix)
        :param bias:              bias (1D vector or scalar)
        :param recurrent_weights: recurrent weights (2D matrix)

        Note: The transfer function needs to support the numpy ufunc out
              argument.

        """
        super(RecurrentLayer, self).__init__(transfer_fn, weights, bias)
        self.recurrent_weights = None
        if recurrent_weights is not None:
            self.recurrent_weights = np.copy(recurrent_weights)

    def activate(self, data):
        """
        Activate the layer.

        :param data: activate with this data
        :return:     activations for this data

        """
        # if we don't have recurrent weights, we don't have to loop
        if self.recurrent_weights is None:
            return super(RecurrentLayer, self).activate(data)
        size = data.shape[0]
        # FIXME: although everything seems to be ok, np.dot doesn't accept the
        #        format of the output array. Speed is almost the same, though.
        # out = np.zeros((size, len(self.bias)), dtype=np.float32)
        # tmp = np.zeros(len(self.bias), dtype=np.float32)
        # np.dot(data, self.weights, out=out)
        out = np.dot(data, self.weights)
        out += self.bias
        # loop through each time step
        for i in xrange(size):
            # add the weighted previous step
            if i >= 1:
                # np.dot(out[i - 1], self.recurrent_weights, out=tmp)
                # out[i] += tmp
                out[i] += np.dot(out[i - 1], self.recurrent_weights)
            # apply transfer function
            self.transfer_fn(out[i], out=out[i])
        # return
        return out


class LinearLayer(RecurrentLayer):
    """
    Recurrent network layer with linear transfer function.

    """
    def __init__(self, weights, bias, recurrent_weights=None):
        """
        Create a new LinearLayer.

        :param weights:           weights (2D matrix)
        :param bias:              bias (1D vector or scalar)
        :param recurrent_weights: recurrent weights (2D matrix)

        """
        super(LinearLayer, self).__init__(linear, weights, bias,
                                          recurrent_weights)


class TanhLayer(RecurrentLayer):
    """
    Recurrent network layer with tanh transfer function.

    """
    def __init__(self, weights, bias, recurrent_weights=None):
        """
        Create a new TanhLayer.

        :param weights:           weights (2D matrix)
        :param bias:              bias (1D vector or scalar)
        :param recurrent_weights: recurrent weights (2D matrix)

        """
        super(TanhLayer, self).__init__(tanh, weights, bias,
                                        recurrent_weights)


class SigmoidLayer(RecurrentLayer):
    """
    Recurrent network layer with sigmoid transfer function.

    """
    def __init__(self, weights, bias, recurrent_weights=None):
        """
        Create a new SigmoidLayer.

        :param weights:           weights (2D matrix)
        :param bias:              bias (1D vector or scalar)
        :param recurrent_weights: recurrent weights (2D matrix)

        """
        super(SigmoidLayer, self).__init__(sigmoid, weights, bias,
                                           recurrent_weights)


# LSTM stuff
class Cell(object):
    """
    Cell as used by LSTM units.

    """
    def __init__(self, weights, bias, recurrent_weights, transfer_fn=tanh):
        """
        Create a new cell as used by LSTM units.

        :param weights:           weights (2D matrix)
        :param bias:              bias (1D vector or scalar)
        :param recurrent_weights: recurrent weights (2D matrix)
        :param transfer_fn:       transfer function

        Note: The transfer function needs to support the numpy ufunc out
              argument.

        """
        self.weights = np.copy(weights)
        self.bias = bias.flatten('A')
        self.recurrent_weights = np.copy(recurrent_weights)
        self.peephole_weights = None
        self.transfer_fn = transfer_fn
        self.cell = np.zeros(self.bias.size, dtype=np.float32)
        self._tmp = np.zeros(self.bias.size, dtype=np.float32)

    def activate(self, data, prev, state=None):
        """
        Activate the cell with the given data, state (if peephole connections
        are used) and the output (if recurrent connections are used).

        :param data:  input data for the cell (1D vector or scalar)
        :param prev:  output data of the previous time step (1D vector or
                      scalar)
        :param state: state data of the {current | previous} time step (1D
                      vector or scalar)
        :return:      activations of the gate

        """
        # weight input and add bias
        np.dot(data, self.weights, out=self.cell)
        self.cell += self.bias
        # add the previous state weighted by the peephole
        if self.peephole_weights is not None:
            self.cell += state * self.peephole_weights
        # add recurrent connection
        if self.recurrent_weights is not None:
            np.dot(prev, self.recurrent_weights, out=self._tmp)
            self.cell += self._tmp
        # apply transfer function
        self.transfer_fn(self.cell, out=self.cell)
        # also return the cell itself
        return self.cell


class Gate(Cell):
    """
    Gate as used by LSTM units.

    """
    def __init__(self, weights, bias, recurrent_weights, peephole_weights):
        """
        Create a new {input, forget, output} Gate as used by LSTM units.

        :param weights:           weights (2D matrix)
        :param bias:              bias (1D vector or scalar)
        :param recurrent_weights: recurrent weights (2D matrix)
        :param peephole_weights:  peephole weights (1D vector or scalar)

        """
        super(Gate, self).__init__(weights, bias, recurrent_weights, sigmoid)
        self.peephole_weights = peephole_weights.flatten('A')


class LSTMLayer(object):
    """
    Recurrent network layer with Long Short-Term Memory units.

    """
    def __init__(self, weights, bias, recurrent_weights, peephole_weights):
        """
        Create a new LSTMLayer.

        :param weights:           weights (2D matrix)
        :param bias:              bias (1D vector or scalar)
        :param recurrent_weights: recurrent weights (2D matrix)
        :param peephole_weights:  peephole weights (1D vector or scalar)

        """
        # init the gates and memory cell
        self.input_gate = Gate(weights[0::4].T, bias[0::4].T,
                               recurrent_weights[0::4].T,
                               peephole_weights[0::3].T)
        self.forget_gate = Gate(weights[1::4].T, bias[1::4].T,
                                recurrent_weights[1::4].T,
                                peephole_weights[1::3].T)
        self.cell = Cell(weights[2::4].T, bias[2::4].T,
                         recurrent_weights[2::4].T)
        self.output_gate = Gate(weights[3::4].T, bias[3::4].T,
                                recurrent_weights[3::4].T,
                                peephole_weights[2::3].T)

    def activate(self, data):
        """
        Activate the layer.

        :param data: activate with this data
        :return:     activations for this data

        """
        # init arrays
        size = len(data)
        # output matrix for the whole sequence
        out = np.zeros((size, self.cell.bias.size), dtype=np.float32)
        # output (of the previous time step)
        out_ = np.zeros(self.cell.bias.size, dtype=np.float32)
        # state (of the previous time step)
        state_ = np.zeros(self.cell.bias.size, dtype=np.float32)
        # process the input data
        for i in xrange(size):
            data_ = data[i]
            # input gate:
            # operate on current data, previous state and previous output
            ig = self.input_gate.activate(data_, out_, state_)
            # forget gate:
            # operate on current data, previous state and previous output
            fg = self.forget_gate.activate(data_, out_, state_)
            # cell:
            # operate on current data and previous output
            cell = self.cell.activate(data_, out_)
            # internal state:
            # weight the cell with the input gate
            # and add the previous state weighted by the forget gate
            state_ = cell * ig + state_ * fg
            # state[i] = state_
            # output gate:
            # operate on current data, current state and previous output
            og = self.output_gate.activate(data_, out_, state_)
            # output:
            # apply transfer function to state and weight by output gate
            out_ = tanh(state_) * og
            out[i] = out_
        return out


# network class
class RecurrentNeuralNetwork(Processor):
    """
    Recurrent Neural Network (RNN) class.

    """
    def __init__(self, layers=None):
        """
        Create a new RecurrentNeuralNetwork object.

        :param layers: build a RNN object with the given layers

        """
        self.layers = layers

    @classmethod
    def load(cls, filename):
        """
        Load the RecurrentNeuralNetwork model from a .npz file.

        :param filename: name of the .npz file with the RNN model
        :return:         RecurrentNeuralNetwork instance

        """
        import re
        # native numpy .npz format or pickled dictionary
        data = np.load(filename)
        # determine the number of layers (i.e. all "layer_%d_" occurrences)
        num_layers = max([int(re.findall(r'layer_(\d+)_', k)[0])
                          for k in data.keys() if k.startswith('layer_')])

        # function for layer creation with the given parameters
        def create_layer(params):
            """
            Create a new network layer according to the given parameters.

            :param params: parameters for layer creation
            :return:       a network layer

            """
            # first check if we need to create a bidirectional layer
            bwd_layer = None
            if '%s_type' % REVERSE in params.keys():
                # pop the parameters needed for the reverse (backward) layer
                bwd_type = params.pop('%s_type' % REVERSE)
                bwd_params = dict((k.split('_', 1)[1], params.pop(k))
                                  for k in params.keys() if
                                  k.startswith('%s_' % REVERSE))
                # construct the layer
                bwd_layer = globals()["%sLayer" % bwd_type](**bwd_params)

            # pop the parameters needed for the normal (forward) layer
            fwd_type = params.pop('type')
            fwd_params = params
            # construct the layer
            fwd_layer = globals()["%sLayer" % fwd_type](**fwd_params)

            # return a (bidirectional) layer
            if bwd_layer is not None:
                # construct a bidirectional layer with the forward and backward
                # layers and return it
                return BidirectionalLayer(fwd_layer, bwd_layer)
            else:
                # just return the forward layer
                return fwd_layer

        # loop over all layers
        layers = []
        for i in xrange(num_layers + 1):
            # get all parameters for that layer
            layer_params = dict((k.split('_', 2)[2], data[k])
                                for k in data.keys() if
                                k.startswith('layer_%d' % i))
            # create a layer from these parameters
            layer = create_layer(layer_params)
            # add to the model
            layers.append(layer)
        # instantiate a RNN from the layers and return it
        return cls(layers)

    def process(self, data):
        """
        Process the given data with the RNN.

        :param data: activate the network with this data
        :return:     network predictions for this data

        """
        # check the dimensions of the data
        if data.ndim == 1:
            data = np.atleast_2d(data).T
        # loop over all layers
        for layer in self.layers:
            # feed the output of one layer into the next one
            data = layer.activate(data)
        # ravel the predictions if needed
        if data.ndim == 2 and data.shape[1] == 1:
            data = data.ravel()
        return data

# alias
RNN = RecurrentNeuralNetwork


class RNNProcessor(ParallelProcessor):
    """
    Recurrent Neural Network (RNN) processor class.

    """
    def __init__(self, nn_files, num_threads=None, **kwargs):
        """
        Instantiates a RNNProcessor, which loads the models from files.

        :param nn_files:    list of files with the RNN models
        :param num_threads: number of parallel working threads

        """
        nn_models = []
        for nn_file in nn_files:
            nn_models.append(RecurrentNeuralNetwork.load(nn_file))
        # instantiate ParallelProcessor
        super(RNNProcessor, self).__init__(nn_models, num_threads)

    @classmethod
    def add_arguments(cls, parser, nn_files):
        """
        Add neural network testing options to an existing parser.

        :param parser:   existing argparse parser
        :param nn_files: list with files of RNN models
        :return:         neural network argument parser group

        """
        from madmom.utils import OverrideDefaultListAction
        # add neural network options
        g = parser.add_argument_group('neural network arguments')
        g.add_argument('--nn_files', action=OverrideDefaultListAction,
                       type=str, default=nn_files,
                       help='average the predictions of these pre-trained '
                            'neural networks (multiple files can be given, '
                            'one file per argument)')
        # return the argument group so it can be modified if needed
        return g


def average_predictions(predictions):
    """
    Returns the average of all predictions from the list.

    :param predictions: list with predictions (beat activation functions)
    :return:            averaged prediction

    """
    # average predictions if needed
    if len(predictions) > 1:
        # average the predictions
        predictions = sum(predictions) / len(predictions)
    else:
        # nothing to average since we have only one prediction
        predictions = predictions[0]
    # return the (averaged) predictions
    return predictions
