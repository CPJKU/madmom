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

import numpy as np
import re

# naming infix for bidirectional layer
REVERSE = 'reverse'


# transfer functions
def linear(data):
    """
    Dummy linear function.

    :param data: input data
    :returns:    data

    """
    return data


def tanh(data):
    """
    Tanh function.

    :param data: input data
    :returns:    tanh of data

    """
    return np.tanh(data)


def sigmoid(data):
    """
    Logistic sigmoid function.

    :param data: input data
    :returns:    logistic sigmoid of data

    """
    return 0.5 * (1. + np.tanh(data / 2.))


# network layer classes
class Layer(object):
    """
    Generic network Layer.

    """

    def activate(self, data):
        """
        Activate the layer.

        :param data: activate with this data
        :returns:    activations for this data

        """
        raise NotImplementedError("To be implemented by subclass")

    @property
    def output_size(self):
        """
        Output size of the layer.

        """
        raise NotImplementedError("To be implemented by subclass")


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
        :returns:    activations for this data

        """
        # activate in forward direction
        fwd = self.fwd_layer.activate(data)
        # also activate with reverse input
        bwd = self.bwd_layer.activate(data[::-1])
        # stack data
        return np.hstack((bwd[::-1], fwd))

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

        :param transfer_fn:       transfer function
        :param weights:           weights (2D matrix)
        :param bias:              bias (1D vector or scalar)

        """
        self.transfer_fn = transfer_fn
        self.weights = weights
        self.bias = bias

    def activate(self, data):
        """
        Activate the layer.

        :param data: activate with this data
        :returns:    activations for this data

        """
        # weight the data, add bias and apply transfer function
        return np.dot(data, self.weights) + self.bias

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

        """
        super(RecurrentLayer, self).__init__(transfer_fn, weights, bias)
        self.recurrent_weights = recurrent_weights

    def activate(self, data):
        """
        Activate the layer.

        :param data: activate with this data
        :returns:    activations for this data

        """
        # if we don't have recurrent weights, we don't have to loop
        if self.recurrent_weights is None:
            return super(RecurrentLayer, self).activate(data)
        # loop through each time step of the data
        size = data.shape[0]
        out = np.zeros((size, self.bias.size))
        for i in range(size):
            # weight the data, add the bias
            cell = np.dot(data[i], self.weights) + self.bias
            # add the weighted previous step and
            cell += np.dot(out[i - 1], self.recurrent_weights)
            # apply transfer function
            out[i] = self.transfer_fn(cell)
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


class LSTMLayer(RecurrentLayer):
    """
    Recurrent network layer with Long Short-Term Memory units.

    """
    def __init__(self, weights, bias, recurrent_weights,
                 peephole_weights=None):
        """
        Create a new LSTMLayer.

        :param weights:           weights (2D matrix)
        :param bias:              bias (1D vector or scalar)
        :param recurrent_weights: recurrent weights (2D matrix)
        :param peephole_weights:  peephole weights (2D matrix)

        """
        super(LSTMLayer, self).__init__(tanh, weights, bias, recurrent_weights)
        self.peephole_weights = peephole_weights

    def activate(self, data):
        """
        Activate the layer.

        :param data: activate with this data
        :returns:    activations for this data

        """
        # init arrays
        size = data.shape[0]
        out = np.zeros((size, self.bias.size / 4))
        state = np.zeros_like(out)
        # process the input data
        for i in range(size):
            # input gate:
            # weight input and add bias
            ig = np.dot(data[i], self.weights[0::4].T) + self.bias[0::4].T
            # add the previous state weighted by the peephole
            if self.peephole_weights is not None:
                ig += state[i - 1] * self.peephole_weights[0::3].T
            # add recurrent connection
            if self.recurrent_weights is not None:
                ig += np.dot(out[i - 1], self.recurrent_weights[0::4].T)
            # apply sigmoid
            ig = sigmoid(ig)

            # forget gate:
            # weight input and add bias
            fg = np.dot(data[i], self.weights[1::4].T) + self.bias[1::4].T
            # add the previous state weighted by the peephole
            if self.peephole_weights is not None:
                fg += state[i - 1] * self.peephole_weights[1::3].T
            # add recurrent connection
            if self.recurrent_weights is not None:
                fg += np.dot(out[i - 1], self.recurrent_weights[1::4].T)
            # apply sigmoid
            fg = sigmoid(fg)

            # cell:
            # weight the input and add bias
            cell = np.dot(data[i], self.weights[2::4].T) + self.bias[2::4].T
            # add recurrent connections
            if self.recurrent_weights is not None:
                cell += np.dot(out[i - 1], self.recurrent_weights[2::4].T)
            # apply tanh
            cell = np.tanh(cell)

            # internal state:
            # weight the cell with the input gate
            # and add the previous state weighted by the forget gate
            state[i] = cell * ig + state[i - 1] * fg

            # output gate:
            # weight the input and add bias
            og = np.dot(data[i], self.weights[3::4].T) + self.bias[3::4].T
            # add the *current* state weighted by the peephole
            if self.peephole_weights is not None:
                og += state[i] * self.peephole_weights[2::3].T
            # add recurrent connection
            if self.recurrent_weights is not None:
                og += np.dot(out[i - 1], self.recurrent_weights[3::4].T)
            # apply sigmoid
            og = sigmoid(og)

            # output:
            # apply transfer function to state and weight by output gate
            out[i] = self.transfer_fn(state[i]) * og
        return out


# network class
class RecurrentNeuralNetwork(object):
    """
    Recurrent Neural Network (RNN) class.

    """
    def __init__(self, filename=None):
        """
        Create a new RecurrentNeuralNetwork object.

        :param filename: build the RNN according to the model stored in that
                         file [default=None]

        """
        self.layers = []
        if filename:
            self.load(filename)

    def load(self, filename):
        """
        Load the model of the RecurrentNeuralNetwork from a .npz file.

        :param filename: name of the .npz file with the RNN model

        """
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
            :returns:      a network layer

            """
            # first check if we need to create a bidirectional layer
            bwd_layer = None
            if '%s_type' % REVERSE in params.keys():
                # pop the parameters needed for the reverse (backward) layer
                bwd_type = params.pop('%s_type' % REVERSE)
                bwd_params = dict((k.split('_', 1)[1], params.pop(k)) \
                                  for k in params.keys() if \
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
                # layer and return it
                return BidirectionalLayer(fwd_layer, bwd_layer)
            else:
                # just return the forward layer
                return fwd_layer

        # loop over all layers and add them to the model
        for i in xrange(num_layers + 1):
            # get all parameters for that layer
            layer_params = dict((k.split('_', 2)[2], data[k])
                                for k in data.keys() if
                                k.startswith('layer_%d' % i))
            # create a layer from these parameters
            layer = create_layer(layer_params)
            # add to the model
            self.layers.append(layer)

    def activate(self, data):
        """
        Activate the RNN.

        :param data: activate with this data
        :returns:    network activations for this data

        """
        # loop over all layers
        # feed the output of one layer as input to the next one
        for layer in self.layers:
            # activate the layer
            data = layer.activate(data)
        return data

# alias
RNN = RecurrentNeuralNetwork
