# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-few-public-methods
"""
This module contains neural network layers for the ml.nn module.

"""

from __future__ import absolute_import, division, print_function

import numpy as np

from .activations import tanh, sigmoid

NN_DTYPE = np.float32


class Layer(object):
    """
    Generic callable network layer.

    """

    def __call__(self, *args):
        # this magic method makes a Layer callable
        return self.activate(*args)

    def activate(self, data):
        """
        Activate the layer.

        Parameters
        ----------
        data : numpy array
            Activate with this data.

        Returns
        -------
        numpy array
            Activations for this data.

        """
        raise NotImplementedError('must be implemented by subclass.')


class FeedForwardLayer(Layer):
    """
    Feed-forward network layer.

    Parameters
    ----------
    weights : numpy array, shape ()
        Weights.
    bias : scalar or numpy array, shape ()
        Bias.
    activation_fn : numpy ufunc
        Activation function.

    """

    def __init__(self, weights, bias, activation_fn):
        self.weights = weights.copy()
        self.bias = bias.flatten()
        self.activation_fn = activation_fn

    def activate(self, data):
        """
        Activate the layer.

        Parameters
        ----------
        data : numpy array
            Activate with this data.

        Returns
        -------
        numpy array
            Activations for this data.

        """
        # weight the data, add bias and apply transfer function
        return self.activation_fn(np.dot(data, self.weights) + self.bias)


class RecurrentLayer(FeedForwardLayer):
    """
    Recurrent network layer.

    Parameters
    ----------
    weights : numpy array, shape ()
        Weights.
    bias : scalar or numpy array, shape ()
        Bias.
    recurrent_weights : numpy array, shape ()
        Recurrent weights.
    activation_fn : numpy ufunc
        Activation function.

    """

    def __init__(self, weights, bias, recurrent_weights, activation_fn):
        super(RecurrentLayer, self).__init__(weights, bias, activation_fn)
        self.recurrent_weights = recurrent_weights.copy()

    def activate(self, data):
        """
        Activate the layer.

        Parameters
        ----------
        data : numpy array
            Activate with this data.

        Returns
        -------
        numpy array
            Activations for this data.

        """
        # if we don't have recurrent weights, we don't have to loop
        if self.recurrent_weights is None:
            return super(RecurrentLayer, self).activate(data)
        size = data.shape[0]
        # FIXME: although everything seems to be ok, np.dot doesn't accept the
        #        format of the output array. Speed is almost the same, though.
        # out = np.zeros((size, len(self.bias)), dtype=NN_DTYPE)
        # tmp = np.zeros(len(self.bias), dtype=NN_DTYPE)
        # np.dot(data, self.weights, out=out)
        out = np.dot(data, self.weights)
        out += self.bias
        # loop through each time step
        for i in range(size):
            # add the weighted previous step
            if i >= 1:
                # np.dot(out[i - 1], self.recurrent_weights, out=tmp)
                # out[i] += tmp
                out[i] += np.dot(out[i - 1], self.recurrent_weights)
            # apply transfer function
            self.activation_fn(out[i], out=out[i])
        # return
        return out


class BidirectionalLayer(Layer):
    """
    Bidirectional network layer.

    Parameters
    ----------
    fwd_layer : Layer instance
        Forward layer.
    bwd_layer : Layer instance
        Backward layer.

    """

    def __init__(self, fwd_layer, bwd_layer):
        self.fwd_layer = fwd_layer
        self.bwd_layer = bwd_layer

    def activate(self, data):
        """
        Activate the layer.

        After activating the `fwd_layer` with the data and the `bwd_layer` with
        the data in reverse temporal order, the two activations are stacked and
        returned.

        Parameters
        ----------
        data : numpy array
            Activate with this data.

        Returns
        -------
        numpy array
            Activations for this data.

        """
        # activate in forward direction
        fwd = self.fwd_layer.activate(data)
        # also activate with reverse input
        bwd = self.bwd_layer.activate(data[::-1])
        # stack data
        return np.hstack((fwd, bwd[::-1]))


# LSTM stuff
class Cell(Layer):
    """
    Cell as used by LSTM units.

    Parameters
    ----------
    weights : numpy array, shape ()
        Weights.
    bias : scalar or numpy array, shape ()
        Bias.
    recurrent_weights : numpy array, shape ()
        Recurrent weights.
    activation_fn : numpy ufunc, optional
        Activation function.

    """

    def __init__(self, weights, bias, recurrent_weights, activation_fn=tanh):
        self.weights = weights.copy()
        self.bias = bias.flatten()
        self.recurrent_weights = recurrent_weights.copy()
        # Note: define the peephole_weights here, so we don't have to define a
        #       different activate() method for the Gate subclass
        self.peephole_weights = None
        self.activation_fn = activation_fn
        self.cell = np.zeros(self.bias.size, dtype=NN_DTYPE)
        self._tmp = np.zeros(self.bias.size, dtype=NN_DTYPE)

    def activate(self, data, prev, state=None):
        """
        Activate the cell / gate with the given data, state (if peephole
        connections are used) and the output (if recurrent connections are
        used).

        Parameters
        ----------
        data : scalar or numpy array, shape ()
            Input data for the cell.
        prev : scalar or numpy array, shape ()
            Output data of the previous time step.
        state : scalar or numpy array, shape ()
            State data of the {current | previous} time step.

        Returns
        -------
        numpy array
            Activations of the gate for this data.

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
        self.activation_fn(self.cell, out=self.cell)
        # also return the cell itself
        return self.cell


class Gate(Cell):
    """
    Gate as used by LSTM units.

    Parameters
    ----------
    weights : numpy array, shape ()
        Weights.
    bias : scalar or numpy array, shape ()
        Bias.
    recurrent_weights : numpy array, shape ()
        Recurrent weights.
    peephole_weights : numpy array, shape ()
        Peephole weights.
    activation_fn : numpy ufunc, optional
        Activation function.

    """

    def __init__(self, weights, bias, recurrent_weights, peephole_weights,
                 activation_fn=sigmoid):
        super(Gate, self).__init__(weights, bias, recurrent_weights,
                                   activation_fn=activation_fn)
        self.peephole_weights = peephole_weights.flatten()


class LSTMLayer(Layer):
    """
    Recurrent network layer with Long Short-Term Memory units.

    Parameters
    ----------
    weights : numpy array, shape ()
        Weights.
    bias : scalar or numpy array, shape ()
        Bias.
    recurrent_weights : numpy array, shape ()
        Recurrent weights.
    peephole_weights : numpy array, shape ()
        Peephole weights.
    activation_fn : numpy ufunc, optional
        Activation function.

    """

    def __init__(self, weights, bias, recurrent_weights, peephole_weights,
                 activation_fn=tanh):
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
        self.activation_fn = activation_fn

    def activate(self, data):
        """
        Activate the LSTM layer.

        Parameters
        ----------
        data : numpy array
            Activate with this data.

        Returns
        -------
        numpy array
            Activations for this data.

        """
        # init arrays
        size = len(data)
        # output matrix for the whole sequence
        out = np.zeros((size, self.cell.bias.size), dtype=NN_DTYPE)
        # output (of the previous time step)
        out_ = np.zeros(self.cell.bias.size, dtype=NN_DTYPE)
        # state (of the previous time step)
        state_ = np.zeros(self.cell.bias.size, dtype=NN_DTYPE)
        # process the input data
        for i in range(size):
            # cache input data
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
            # output gate:
            # operate on current data, current state and previous output
            og = self.output_gate.activate(data_, out_, state_)
            # output:
            # apply transfer function to state and weight by output gate
            out_ = self.activation_fn(state_) * og
            out[i] = out_
        return out
