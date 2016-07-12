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
        self.weights = weights
        self.bias = bias
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
        # weight input, add bias and apply activations function
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
        # weight input and add bias
        out = np.dot(data, self.weights) + self.bias
        # loop through all time steps
        for i in range(len(data)):
            # add weighted previous step
            if i >= 1:
                out[i] += np.dot(out[i - 1], self.recurrent_weights)
            # apply activation function
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
        fwd = self.fwd_layer(data)
        # also activate with reverse input
        bwd = self.bwd_layer(data[::-1])
        # stack data
        return np.hstack((fwd, bwd[::-1]))


# LSTM stuff
class Gate(Layer):
    """
    Gate as used by LSTM layers.

    Parameters
    ----------
    weights : numpy array, shape ()
        Weights.
    bias : scalar or numpy array, shape ()
        Bias.
    recurrent_weights : numpy array, shape ()
        Recurrent weights.
    peephole_weights : numpy array, optional, shape ()
        Peephole weights.
    activation_fn : numpy ufunc, optional
        Activation function.

    """

    def __init__(self, weights, bias, recurrent_weights, peephole_weights=None,
                 activation_fn=sigmoid):
        self.weights = weights
        self.bias = bias
        self.recurrent_weights = recurrent_weights
        self.peephole_weights = peephole_weights
        self.activation_fn = activation_fn

    def activate(self, data, prev, state=None):
        """
        Activate the gate with the given data, state (if peephole connections
        are used) and the previous output (if recurrent connections are used).

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
        out = np.dot(data, self.weights) + self.bias
        # add the previous state weighted by the peephole
        if self.peephole_weights is not None:
            out += state * self.peephole_weights
        # add recurrent connection
        if self.recurrent_weights is not None:
            out += np.dot(prev, self.recurrent_weights)
        # apply activation function and return it
        return self.activation_fn(out)


class Cell(Gate):
    """
    Cell as used by LSTM layers.

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

    Notes
    -----
    A Cell is the same as a Gate except it misses peephole connections and
    has a `tanh` activation function.

    """

    def __init__(self, weights, bias, recurrent_weights, activation_fn=tanh):
        super(Cell, self).__init__(weights, bias, recurrent_weights,
                                   activation_fn=activation_fn)


class LSTMLayer(Layer):
    """
    Recurrent network layer with Long Short-Term Memory units.

    Parameters
    ----------
    input_gate : :class:`Gate`
        Input gate.
    forget_gate : :class:`Gate`
        Forget gate.
    cell : :class:`Cell`
        Cell (i.e. a Gate without peephole connections).
    output_gate : :class:`Gate`
        Output gate.
    activation_fn : numpy ufunc, optional
        Activation function.

    """

    def __init__(self, input_gate, forget_gate, cell, output_gate,
                 activation_fn=tanh):
        self.input_gate = input_gate
        self.forget_gate = forget_gate
        self.cell = cell
        self.output_gate = output_gate
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
            # apply activation function to state and weight by output gate
            out_ = self.activation_fn(state_) * og
            out[i] = out_
        return out
