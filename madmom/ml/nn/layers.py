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

from .activations import linear, sigmoid, tanh

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
        self.recurrent_weights = recurrent_weights

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


class ConvolutionalLayer(FeedForwardLayer):
    """
    Convolutional network layer.

    Parameters
    ----------
    weights : numpy array, shape (num_feature_maps, num_channels, <kernel>)
        Weights.
    bias : scalar or numpy array, shape (num_filters,)
        Bias.
    stride : int, optional
        Stride of the convolution.
    pad : {'valid', 'same', 'full'}
        A string indicating the size of the output:

        - full
            The output is the full discrete linear convolution of the inputs.
        - valid
            The output consists only of those elements that do not rely on the
            zero-padding.
        - same
            The output is the same size as the input, centered with respect to
            the ‘full’ output.

    activation_fn : numpy ufunc
        Activation function.

    """

    def __init__(self, weights, bias, stride=1, pad='valid',
                 activation_fn=linear):
        super(ConvolutionalLayer, self).__init__(weights, bias, activation_fn)
        if stride != 1:
            raise NotImplementedError('only `stride` == 1 implemented.')
        self.stride = stride
        if pad != 'valid':
            raise NotImplementedError('only `pad` == "valid" implemented.')
        self.pad = pad

    def activate(self, data):
        """
        Activate the layer.

        Parameters
        ----------
        data : numpy array (num_frames, num_bins, num_channels)
            Activate with this data.

        Returns
        -------
        numpy array
            Activations for this data.

        """
        from scipy.signal import convolve2d
        # determine output shape and allocate memory
        num_frames, num_bins, num_channels = data.shape
        num_channels, num_features, size_time, size_freq = self.weights.shape
        # adjust the output number of frames and bins depending on `pad`
        # TODO: this works only with pad='valid'
        num_frames -= (size_time - 1)
        num_bins -= (size_freq - 1)
        # init the output array with Fortran ordering (column major)
        out = np.zeros((num_frames, num_bins, num_features),
                       dtype=NN_DTYPE, order='F')
        # iterate over all channels
        for c in range(num_channels):
            channel = data[:, :, c]
            # convolve each channel separately with each filter
            for w, weights in enumerate(self.weights[c]):
                # TODO: add boundary stuff?
                conv = convolve2d(channel, weights, mode=self.pad)
                out[:, :, w] += conv
        # add bias to each feature map and apply activation function
        return self.activation_fn(out + self.bias)


class StrideLayer(Layer):
    """
    Stride network layer.

    Parameters
    ----------
    block_size : int
        Re-arrange (stride) the data in blocks of given size.

    """

    def __init__(self, block_size):
        self.block_size = block_size

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
            Strided data.

        """
        # re-arrange the data for the following dense layer
        from ...utils import segment_axis
        data = segment_axis(data, self.block_size, 1, axis=0, end='cut')
        return data.reshape(len(data), -1)


class MaxPoolLayer(Layer):
    """
    2D max-pooling network layer.

    Parameters
    ----------
    size : tuple
        The size of the pooling region in each dimension.
    stride : tuple, optional
        The strides between successive pooling regions in each dimension.
        If None `stride` = `size`.

    """

    def __init__(self, size, stride=None):
        self.size = size
        if stride is None:
            stride = size
        self.stride = stride

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
        from scipy.ndimage.filters import maximum_filter
        # define which part of the maximum filtered data to return
        slice_dim_1 = slice(self.size[0] // 2, None, self.stride[0])
        slice_dim_2 = slice(self.size[1] // 2, None, self.stride[1])
        # TODO: is constant mode the most appropriate?
        data = [maximum_filter(data[:, :, c], self.size, mode='constant')
                [slice_dim_1, slice_dim_2] for c in range(data.shape[2])]
        # join channels and return as array
        return np.dstack(data)
