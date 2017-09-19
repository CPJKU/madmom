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

    def __call__(self, *args, **kwargs):
        # this magic method makes a Layer callable
        return self.activate(*args, **kwargs)

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

    def reset(self):
        """
        Reset the layer to its initial state.

        """
        return None


class FeedForwardLayer(Layer):
    """
    Feed-forward network layer.

    Parameters
    ----------
    weights : numpy array, shape (num_inputs, num_hiddens)
        Weights.
    bias : scalar or numpy array, shape (num_hiddens,)
        Bias.
    activation_fn : numpy ufunc
        Activation function.

    """

    def __init__(self, weights, bias, activation_fn):
        self.weights = weights
        self.bias = bias.flatten()
        self.activation_fn = activation_fn

    def activate(self, data, **kwargs):
        """
        Activate the layer.

        Parameters
        ----------
        data : numpy array, shape (num_frames, num_inputs)
            Activate with this data.

        Returns
        -------
        numpy array, shape (num_frames, num_hiddens)
            Activations for this data.

        """
        # weight input, add bias and apply activations function
        out = np.dot(data, self.weights) + self.bias
        return self.activation_fn(out)


class RecurrentLayer(FeedForwardLayer):
    """
    Recurrent network layer.

    Parameters
    ----------
    weights : numpy array, shape (num_inputs, num_hiddens)
        Weights.
    bias : scalar or numpy array, shape (num_hiddens,)
        Bias.
    recurrent_weights : numpy array, shape (num_hiddens, num_hiddens)
        Recurrent weights.
    activation_fn : numpy ufunc
        Activation function.
    init : numpy array, shape (num_hiddens,), optional
        Initial state of hidden units.

    """

    def __init__(self, weights, bias, recurrent_weights, activation_fn,
                 init=None):
        super(RecurrentLayer, self).__init__(weights, bias, activation_fn)
        self.recurrent_weights = recurrent_weights
        if init is None:
            init = np.zeros(self.bias.size, dtype=NN_DTYPE)
        self.init = init
        # attributes needed for stateful processing
        self._prev = self.init

    def __getstate__(self):
        # copy everything to a pickleable object
        state = self.__dict__.copy()
        # do not pickle attributes needed for stateful processing
        state.pop('_prev', None)
        return state

    def __setstate__(self, state):
        # restore pickled instance attributes
        self.__dict__.update(state)
        # TODO: old models do not have the init attribute, thus create it
        #       remove this initialisation code after updating the models
        if not hasattr(self, 'init'):
            self.init = np.zeros(self.bias.size, dtype=NN_DTYPE)
        # add non-pickled attributes needed for stateful processing
        self._prev = self.init

    def reset(self, init=None):
        """
        Reset the layer to its initial state.

        Parameters
        ----------
        init : numpy array, shape (num_hiddens,), optional
            Reset the hidden units to this initial state.

        """
        # reset previous time step to initial value
        self._prev = init if init is not None else self.init

    def activate(self, data, reset=True):
        """
        Activate the layer.

        Parameters
        ----------
        data : numpy array, shape (num_frames, num_inputs)
            Activate with this data.
        reset : bool, optional
            Reset the layer to its initial state before activating it.

        Returns
        -------
        numpy array, shape (num_frames, num_hiddens)
            Activations for this data.

        """
        # reset layer to initial state
        if reset:
            self.reset()
        # weight input and add bias
        out = np.dot(data, self.weights) + self.bias
        # loop through all time steps
        for i in range(len(data)):
            # add weighted previous step
            out[i] += np.dot(self._prev, self.recurrent_weights)
            # apply activation function
            out[i] = self.activation_fn(out[i])
            # set reference to current output
            self._prev = out[i]
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

    def activate(self, data, **kwargs):
        """
        Activate the layer.

        After activating the `fwd_layer` with the data and the `bwd_layer` with
        the data in reverse temporal order, the two activations are stacked and
        returned.

        Parameters
        ----------
        data : numpy array, shape (num_frames, num_inputs)
            Activate with this data.

        Returns
        -------
        numpy array, shape (num_frames, num_hiddens)
            Activations for this data.

        """
        # activate in forward direction
        fwd = self.fwd_layer(data, **kwargs)
        # also activate with reverse input
        bwd = self.bwd_layer(data[::-1], **kwargs)
        # stack data
        return np.hstack((fwd, bwd[::-1]))


# LSTM stuff
class Gate(RecurrentLayer):
    """
    Gate as used by LSTM layers.

    Parameters
    ----------
    weights : numpy array, shape (num_inputs, num_hiddens)
        Weights.
    bias : scalar or numpy array, shape (num_hiddens,)
        Bias.
    recurrent_weights : numpy array, shape (num_hiddens, num_hiddens)
        Recurrent weights.
    peephole_weights : numpy array, shape (num_hiddens,), optional
        Peephole weights.
    activation_fn : numpy ufunc, optional
        Activation function.

    Notes
    -----
    Gate should not be used directly, only inside an LSTMLayer.

    """

    def __init__(self, weights, bias, recurrent_weights, peephole_weights=None,
                 activation_fn=sigmoid):
        super(Gate, self).__init__(weights, bias, recurrent_weights,
                                   activation_fn=activation_fn)
        if peephole_weights is not None:
            peephole_weights = peephole_weights.flatten()
        self.peephole_weights = peephole_weights

    def activate(self, data, prev, state=None):
        """
        Activate the gate with the given data, state (if peephole connections
        are used) and the previous output (if recurrent connections are used).

        Parameters
        ----------
        data : scalar or numpy array, shape (num_hiddens,)
            Input data for the cell.
        prev : scalar or numpy array, shape (num_hiddens,)
            Output data of the previous time step.
        state : scalar or numpy array, shape (num_hiddens,)
            State data of the {current | previous} time step.

        Returns
        -------
        numpy array, shape (num_hiddens,)
            Activations of the gate for this data.

        """
        # weight input and add bias
        out = np.dot(data, self.weights) + self.bias
        # add the previous state weighted by the peephole
        if self.peephole_weights is not None:
            out += state * self.peephole_weights
        # add recurrent connection
        out += np.dot(prev, self.recurrent_weights)
        # apply activation function and return it
        return self.activation_fn(out)


class Cell(Gate):
    """
    Cell as used by LSTM layers.

    Parameters
    ----------
    weights : numpy array, shape (num_inputs, num_hiddens)
        Weights.
    bias : scalar or numpy array, shape (num_hiddens,)
        Bias.
    recurrent_weights : numpy array, shape (num_hiddens, num_hiddens)
        Recurrent weights.
    activation_fn : numpy ufunc, optional
        Activation function.

    Notes
    -----
    A Cell is the same as a Gate except it misses peephole connections and
    has a `tanh` activation function. It should not be used directly, only
    inside an LSTMLayer.

    """

    def __init__(self, weights, bias, recurrent_weights, activation_fn=tanh):
        super(Cell, self).__init__(weights, bias, recurrent_weights,
                                   activation_fn=activation_fn)


class LSTMLayer(RecurrentLayer):
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
    init : numpy array, shape (num_hiddens, ), optional
        Initial state of the layer.
    cell_init : numpy array, shape (num_hiddens, ), optional
        Initial state of the cell.

    """

    def __init__(self, input_gate, forget_gate, cell, output_gate,
                 activation_fn=tanh, init=None, cell_init=None):
        self.input_gate = input_gate
        self.forget_gate = forget_gate
        self.cell = cell
        self.output_gate = output_gate
        self.activation_fn = activation_fn
        # keep the state of the layer and cell
        if init is None:
            init = np.zeros(self.cell.bias.size, dtype=NN_DTYPE)
        self.init = init
        self._prev = self.init
        if cell_init is None:
            cell_init = np.zeros(self.cell.bias.size, dtype=NN_DTYPE)
        self.cell_init = cell_init
        self._state = self.cell_init

    def __getstate__(self):
        # copy everything to a pickleable object
        state = self.__dict__.copy()
        # do not pickle attributes needed for stateful processing
        state.pop('_prev', None)
        state.pop('_state', None)
        return state

    def __setstate__(self, state):
        # restore pickled instance attributes
        self.__dict__.update(state)
        # TODO: old models do not have the init attributes, thus create them
        #       remove this initialisation code after updating the models
        if not hasattr(self, 'init'):
            self.init = np.zeros(self.cell.bias.size, dtype=NN_DTYPE)
        if not hasattr(self, 'cell_init'):
            self.cell_init = np.zeros(self.cell.bias.size, dtype=NN_DTYPE)
        # add non-pickled attributes needed for stateful processing
        self._prev = self.init
        self._state = self.cell_init

    def reset(self, init=None, cell_init=None):
        """
        Reset the layer to its initial state.

        Parameters
        ----------
        init : numpy array, shape (num_hiddens,), optional
            Reset the hidden units to this initial state.
        cell_init : numpy array, shape (num_hiddens,), optional
            Reset the cells to this initial state.

        """
        # reset previous time step and state to initial value
        self._prev = init if init is not None else self.init
        self._state = cell_init if cell_init is not None else self.cell_init

    def activate(self, data, reset=True):
        """
        Activate the LSTM layer.

        Parameters
        ----------
        data : numpy array, shape (num_frames, num_inputs)
            Activate with this data.
        reset : bool, optional
            Reset the layer to its initial state before activating it.

        Returns
        -------
        numpy array, shape (num_frames, num_hiddens)
            Activations for this data.

        """
        # reset layer
        if reset:
            self.reset()
        # init arrays
        size = len(data)
        # output matrix for the whole sequence
        out = np.zeros((size, self.cell.bias.size), dtype=NN_DTYPE)
        # process the input data
        for i in range(size):
            # cache input data
            data_ = data[i]
            # input gate:
            # operate on current data, previous output and state
            ig = self.input_gate.activate(data_, self._prev, self._state)
            # forget gate:
            # operate on current data, previous output and state
            fg = self.forget_gate.activate(data_, self._prev, self._state)
            # cell:
            # operate on current data and previous output
            cell = self.cell.activate(data_, self._prev)
            # internal state:
            # weight the cell with the input gate
            # and add the previous state weighted by the forget gate
            self._state = cell * ig + self._state * fg
            # output gate:
            # operate on current data, previous output and current state
            og = self.output_gate.activate(data_, self._prev, self._state)
            # output:
            # apply activation function to state and weight by output gate
            out[i] = self.activation_fn(self._state) * og
            # set reference to current output
            self._prev = out[i]
        return out


class GRUCell(Cell):
    """
    Cell as used by GRU layers proposed in [1]_. The cell output is computed by

    .. math::
        h = tanh(W_{xh} * x_t + W_{hh} * h_{t-1} + b).

    Parameters
    ----------
    weights : numpy array, shape (num_inputs, num_hiddens)
        Weights of the connections between inputs and cell.
    bias : scalar or numpy array, shape (num_hiddens,)
        Bias.
    recurrent_weights : numpy array, shape (num_hiddens, num_hiddens)
        Weights of the connections between cell and cell output of the
        previous time step.
    activation_fn : numpy ufunc, optional
        Activation function.

    References
    ----------
    .. [1] Kyunghyun Cho, Bart Van Merrienboer, Dzmitry Bahdanau, and Yoshua
           Bengio,
           "On the properties of neural machine translation: Encoder-decoder
           approaches", http://arxiv.org/abs/1409.1259, 2014.

    Notes
    -----
    There are two formulations of the GRUCell in the literature. Here,
    we adopted the (slightly older) one proposed in [1]_, which is also
    implemented in the Lasagne toolbox.

    It should not be used directly, only inside a GRULayer.

    """

    def __init__(self, weights, bias, recurrent_weights, activation_fn=tanh):
        super(GRUCell, self).__init__(weights, bias, recurrent_weights,
                                      activation_fn)

    def activate(self, data, prev, reset_gate):
        """
        Activate the cell with the given input, previous output and reset gate.

        Parameters
        ----------
        data : numpy array, shape (num_inputs,)
            Input data for the cell.
        prev : numpy array, shape (num_hiddens,)
            Output of the previous time step.
        reset_gate : numpy array, shape (num_hiddens,)
            Activation of the reset gate.

        Returns
        -------
        numpy array, shape (num_hiddens,)
            Activations of the cell for this data.

        """
        # weight input and add bias
        out = np.dot(data, self.weights) + self.bias
        # weight previous cell output and reset gate
        out += reset_gate * np.dot(prev, self.recurrent_weights)
        # apply activation function and return it
        return self.activation_fn(out)


class GRULayer(RecurrentLayer):
    """
    Recurrent network layer with Gated Recurrent Units (GRU) as proposed in
    [1]_.

    Parameters
    ----------
    reset_gate : :class:`Gate`
        Reset gate.
    update_gate : :class:`Gate`
        Update gate.
    cell : :class:`GRUCell`
        GRU cell.
    init : numpy array, shape (num_hiddens,), optional
        Initial state of hidden units.

    References
    ----------
    .. [1] Kyunghyun Cho, Bart van Merriënboer, Dzmitry Bahdanau, and Yoshua
           Bengio,
           "On the properties of neural machine translation: Encoder-decoder
           approaches",
           http://arxiv.org/abs/1409.1259, 2014.

    Notes
    -----
    There are two formulations of the GRUCell in the literature. Here,
    we adopted the (slightly older) one proposed in [1], which is also
    implemented in the Lasagne toolbox.

    """

    def __init__(self, reset_gate, update_gate, cell, init=None):
        # init the gates
        self.reset_gate = reset_gate
        self.update_gate = update_gate
        self.cell = cell
        # keep the state of the layer
        if init is None:
            init = np.zeros(self.cell.bias.size, dtype=NN_DTYPE)
        self.init = init
        # keep the state of the layer
        self._prev = self.init

    def __getstate__(self):
        # copy everything to a pickleable object
        state = self.__dict__.copy()
        # do not pickle attributes needed for stateful processing
        state.pop('_prev', None)
        return state

    def __setstate__(self, state):
        # TODO: old models have a 'hid_init' instead of an 'init' attribute
        #       remove this unpickling code after updating all models
        try:
            import warnings
            warnings.warn('Please update your GRU models by loading them and '
                          'saving them again. Loading old models will not work'
                          ' from version 0.18 onwards.', RuntimeWarning)
            state['init'] = state.pop('hid_init')
        except KeyError:
            pass
        # restore pickled instance attributes
        self.__dict__.update(state)
        # TODO: old models do not have the init attributes, thus create them
        #       remove this initialisation code after updating the models
        if not hasattr(self, 'init'):
            self.init = np.zeros(self.cell.bias.size, dtype=NN_DTYPE)
        # add non-pickled attributes needed for stateful processing
        self._prev = self.init

    def reset(self, init=None):
        """
        Reset the layer to its initial state.

        Parameters
        ----------
        init : numpy array, shape (num_hiddens,), optional
            Reset the hidden units to this initial state.

        """
        # reset previous time step and state to initial value
        self._prev = init or self.init

    def activate(self, data, reset=True):
        """
        Activate the GRU layer.

        Parameters
        ----------
        data : numpy array, shape (num_frames, num_inputs)
            Activate with this data.
        reset : bool, optional
            Reset the layer to its initial state before activating it.

        Returns
        -------
        numpy array, shape (num_frames, num_hiddens)
            Activations for this data.

        """
        # reset layer
        if reset:
            self.reset()
        # init arrays
        size = len(data)
        # output matrix for the whole sequence
        out = np.zeros((size, self.cell.bias.size), dtype=NN_DTYPE)
        # process the input data
        for i in range(size):
            # cache input data
            data_ = data[i]
            # reset gate:
            # operate on current data and previous output
            rg = self.reset_gate.activate(data_, self._prev)
            # update gate:
            # operate on current data and previous output
            ug = self.update_gate.activate(data_, self._prev)
            # cell (implemented as in [1]):
            # operate on current data, previous output and reset gate
            cell = self.cell.activate(data_, self._prev, rg)
            # output:
            out[i] = ug * cell + (1 - ug) * self._prev
            # set reference to current output
            self._prev = out[i]
        return out


def _kernel_margins(kernel_shape, margin_shift):
    """
    Determine the margin that needs to be cut off when doing a "valid"
    convolution.

    Parameters
    ----------
    kernel_shape : tuple
        Shape of the convolution kernel to determine the margins for
    margin_shift : bool
        Shift the borders by one pixel if kernel is of even size

    Returns
    -------
    start_x, end_x, start_y, end_y : tuple
        Indices determining the valid part of the convolution output.
    """
    start_x = int(np.floor(kernel_shape[0] / 2.))
    start_y = int(np.floor(kernel_shape[1] / 2.))

    margin_shift = -1 if margin_shift else 0
    if kernel_shape[0] % 2 == 0:
        end_x = start_x - 1
        start_x += margin_shift
        end_x -= margin_shift
    else:
        end_x = start_x

    if kernel_shape[1] % 2 == 0:
        end_y = start_y - 1
        start_y += margin_shift
        end_y -= margin_shift
    else:
        end_y = start_y

    return start_x, -end_x, start_y, -end_y


try:
    # pylint: disable=no-name-in-module
    # pylint: disable=wrong-import-order
    # pylint: disable=wrong-import-position

    # if opencv is installed, use their convolution function, because
    # it is much faster
    from cv2 import filter2D as _do_convolve

    def _convolve(x, k):
        sx, ex, sy, ey = _kernel_margins(k.shape, margin_shift=False)
        return _do_convolve(x, -1, k[::-1, ::-1])[sx:ex, sy:ey]

except ImportError:
    # scipy.ndimage.convolution behaves slightly differently with
    # even-sized kernels. If it is used, we need to shift the margins
    from scipy.ndimage import convolve as _do_convolve

    def _convolve(x, k):
        sx, ex, sy, ey = _kernel_margins(k.shape, margin_shift=True)
        return _do_convolve(x, k)[sx:ex, sy:ey]


def convolve(data, kernel):
    """
    Convolve the data with the kernel in 'valid' mode, i.e. only where
    kernel and data fully overlaps.

    Parameters
    ----------
    data : numpy array
        Data to be convolved.
    kernel : numpy array
        Convolution kernel

    Returns
    -------
    numpy array
        Convolved data

    """
    return _convolve(data, kernel)


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

    def activate(self, data, **kwargs):
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
        # if no channel dimension given, assume 1 channel
        if len(data.shape) == 2:
            data = data.reshape(data.shape + (1,))

        # determine output shape and allocate memory
        num_frames, num_bins, num_channels = data.shape
        num_channels_w, num_features, size_time, size_freq = self.weights.shape
        if num_channels_w != num_channels:
            raise ValueError('Number of channels in weight vector different '
                             'from number of channels of input data!')
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
                conv = convolve(channel, weights)
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

    def activate(self, data, **kwargs):
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

    def activate(self, data, **kwargs):
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


class BatchNormLayer(Layer):
    """
    Batch normalization layer with activation function. The previous layer
    is usually linear with no bias - the BatchNormLayer's beta parameter
    replaces it. See [1]_ for a detailed understanding of the parameters.

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
    .. [1] "Batch Normalization: Accelerating Deep Network Training by Reducing
           Internal Covariate Shift"
           Sergey Ioffe and Christian Szegedy.
           http://arxiv.org/abs/1502.03167, 2015.
    """

    def __init__(self, beta, gamma, mean, inv_std, activation_fn):
        self.beta = beta
        self.gamma = gamma
        self.mean = mean
        self.inv_std = inv_std
        self.activation_fn = activation_fn

    def activate(self, data, **kwargs):
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
