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


class FeedForwardLayer(object):
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
        # weight the data, add bias and apply activation function
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
            # apply activation function
            self.activation_fn(out[i], out=out[i])
        # return
        return out


class BidirectionalLayer(object):
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
class Cell(object):
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
        # apply activation function
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


class LSTMLayer(object):
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


class StrideLayer(object):

    def __init__(self, block_size):
        self.block_size = block_size

    def activate(self, data):
        from ...utils import segment_axis
        data = segment_axis(data, self.block_size, 1, axis=0, end='cut')
        return data.reshape(len(data), -1)


class MaxPoolLayer(object):
    """
    2D Max-pooling network layer.

    Parameters
    ----------
    size : tuple
        The size of the pooling region in each dimension.
    stride : tuple, optional
        The strides between sucessive pooling regions in each dimension.
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
        # TODO: is contsant mode the most appropriate?
        data = [maximum_filter(data[:, :, c], self.size, mode='constant')[
                slice_dim_1, slice_dim_2] for c in range(data.shape[2])]
        # join channels and return as array
        return np.dstack(data)


class BatchNormLayer(object):
    """
    Batch normalization layer with activation function. The previous layer
    is usually linear with no bias - the BatchNormLayer's beta parameter
    replaces it. See [1] for a detailed understanding of the parameters.

    Parameters
    ----------
    beta : numpy array
        Values for the `beta` parameter. Must be broadcastable to the incoming
        shape.
    gamma : numpy array
        Values for the `gamma` parameter. Must be broadcastable to the incoming
        shape.
    mean : numpy array
        Mean values of incoming data. Must be broadcastable to the incoming
        shape.
    inv_std : numpy array
        Inverse standard deviation of incoming data. Must be broadcastable to
        the incoming shape.
    activation_fn : numpy ufunc
        Activation function.

    References
    ----------
    .. [1] Ioffe, Sergey and Szegedy, Christian (2015):
           Batch Normalization: Accelerating Deep Network Training by Reducing
           Internal Covariate Shift. http://arxiv.org/abs/1502.03167.
    """

    def __init__(self, beta, gamma, mean, inv_std, activation_fn):
        self.beta = beta
        self.gamma = gamma
        self.mean = mean
        self.inv_std = inv_std
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
        return self.activation_fn(
            (data - self.mean) * (self.gamma * self.inv_std) + self.beta
        )
