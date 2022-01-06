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
from numpy.lib.stride_tricks import as_strided
from scipy.ndimage import convolve as _scipy_convolve
from scipy.ndimage.filters import maximum_filter

from .activations import sigmoid, tanh

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


class SequentialLayer(Layer):
    """
    SequentialLayer.

    Parameters
    ----------
    layers : list
        Activate these layers sequentially.

    """

    def __init__(self, layers):
        self.layers = layers

    def activate(self, data, **kwargs):
        """
        Activate all layers with the data sequentially.

        Parameters
        ----------
        data : numpy arrays
            Data with shape according to the first layer's input shape.

        Returns
        -------
        data : numpy array
            Activated data with shape according to the last layer's output
            shape.

        Notes
        -----
        The output of the first layer is fed into the second layer. The last
        layer's output is returned.

        """
        # sequentially process the data
        for layer in self.layers:
            data = layer(data, **kwargs)
        return data


class ParallelLayer(Layer):
    """
    ParallelLayer.

    Parameters
    ----------
    layers : list
        Activate these layers in parallel.

    """

    def __init__(self, layers):
        self.layers = layers

    def activate(self, data, **kwargs):
        """
        Activate the data with the defined layers in parallel.

        All layers are activated with the same input data.

        Parameters
        ----------
        data : numpy arrays
            Data with shape according to the layer's common input shape.

        Returns
        -------
        data : list
            List with activated data, dimensions according to each layer's
            output shape.

        """
        # process the data with each layer in parallel
        results = []
        for layer in self.layers:
            results.append(layer(data, **kwargs))
        return results


class MultiTaskLayer(Layer):
    """
    MultiTaskLayer.

    Parameters
    ----------
    layers : list
        Activate these layers individually.
    mapping : dict, optional
        Dict mapping layer indices to (input) data indices, i.e. which layer to
        activate with which input.

    """

    def __init__(self, layers, mapping=None):
        self.layers = layers
        self.mapping = mapping

    def activate(self, data, **kwargs):
        """
        Activate the defined layers with the data individually.

        Parameters
        ----------
        data : tuple or list
            Data tuple/list with numpy arrays, with shapes according to each
            layer's input shape.

        Returns
        -------
        data : tuple
            Tuple with activated data, dimensions according to each layer's
            output shape.

        Notes
        -----
        All layers are activated with the data at the respective position, i.e.
        the first layer with the first element of the data tuple and so on.

        """
        result = []
        # process (multiple) inputs with (multiple) outputs
        for i, layer in enumerate(self.layers):
            # if mapping is not defined, use a 1:1 mapping
            if self.mapping is None:
                m = i
            else:
                m = self.mapping[i]
            # activate layer with selected input data
            result.append(layer(data[m]))
        return tuple(result)


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

    def __init__(self, weights, bias, activation_fn=None):
        self.weights = weights
        self.bias = bias.flatten()
        self.activation_fn = activation_fn

    def activate(self, data, **kwargs):
        """
        Activate FeedForwardLayer.

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
        if self.activation_fn is not None:
            self.activation_fn(out, out=out)
        return out


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
    activation_fn : numpy ufunc, optional
        Activation function.
    init : numpy array, shape (num_hiddens,), optional
        Initial state of hidden units.

    """

    def __init__(self, weights, bias, recurrent_weights, activation_fn=tanh,
                 init=None):
        super(RecurrentLayer, self).__init__(weights, bias, activation_fn)
        self.recurrent_weights = recurrent_weights
        if init is None:
            init = np.zeros(self.bias.size, dtype=NN_DTYPE)
        self.init = init
        # attributes needed for stateful processing
        self._prev = self.init

    def __getstate__(self):
        # copy everything to a picklable object
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
        Reset RecurrentLayer to its initial state.

        Parameters
        ----------
        init : numpy array, shape (num_hiddens,), optional
            Reset the hidden units to this initial state.

        """
        # reset previous time step to initial value
        self._prev = init if init is not None else self.init

    def activate(self, data, reset=True):
        """
        Activate RecurrentLayer.

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
            if self.activation_fn is not None:
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
        Activate BidirectionalLayer.

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
        Activate gate with the given data, state (if peephole connections
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
        # copy everything to a picklable object
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
        Reset LSTMLayer to its initial state.

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
        Activate LSTMLayer.

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
        Activate GRU cell with the given input, previous output and reset gate.

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
        # copy everything to a picklable object
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


def _kernel_margins(kernel_shape, margin_shift, pad='valid'):
    """
    Determine the margin that needs to be cut off when doing a convolution.

    Parameters
    ----------
    kernel_shape : tuple
        Shape of the convolution kernel to determine the margins for
    margin_shift : bool
        Shift the borders by one pixel if kernel is of even size
    pad : str, optional
        Padding applied to the convolution, either 'valid' or 'same'.

    Returns
    -------
    start_x, end_x, start_y, end_y : tuple
        Indices determining the valid part of the convolution output.
    """

    if pad == 'same':
        return None, None, None, None
    elif pad != 'valid':
        raise NotImplementedError('only `pad` == "valid" implemented.')

    start_x = int(np.floor(kernel_shape[0] / 2.))
    start_y = int(np.floor(kernel_shape[1] / 2.))

    margin_shift = -1 if margin_shift else 0
    if kernel_shape[0] % 2 == 0:
        end_x = start_x - 1
        start_x += margin_shift
        end_x -= margin_shift
    else:
        end_x = start_x
    start_x = start_x if start_x > 0 else None
    end_x = -end_x if end_x > 0 else None

    if kernel_shape[1] % 2 == 0:
        end_y = start_y - 1
        start_y += margin_shift
        end_y -= margin_shift
    else:
        end_y = start_y
    start_y = start_y if start_y > 0 else None
    end_y = -end_y if end_y > 0 else None

    return start_x, end_x, start_y, end_y


try:
    # pylint: disable=no-name-in-module
    # pylint: disable=wrong-import-order
    # pylint: disable=wrong-import-position

    # opencv's convolution is much faster for certain kernel sizes
    from cv2 import filter2D, BORDER_CONSTANT

    def _convolve_opencv(x, k, pad):
        sx, ex, sy, ey = _kernel_margins(k.shape, margin_shift=False, pad=pad)
        anchor = (-1, -1)
        if pad == 'same':
            # TODO: check if this is correct in all cases
            anchor = tuple(-1 * (np.array(k.shape) % 2))
        # opencv computes a correlation, thus flip the kernel
        return filter2D(x, -1, k[::-1, ::-1], anchor=anchor,
                        borderType=BORDER_CONSTANT)[sx:ex, sy:ey]
except ImportError:
    _convolve_opencv = None


def _convolve_scipy(x, k, pad):
    # scipy.ndimage.convolve behaves slightly differently with
    # even-sized kernels, thus shift the margins
    sx, ex, sy, ey = _kernel_margins(k.shape, margin_shift=True, pad=pad)
    return _scipy_convolve(x, k, mode='constant')[sx:ex, sy:ey]


def convolve(data, kernel, pad='valid'):
    """
    Convolve data with kernel.

    Parameters
    ----------
    data : numpy array
        Data to be convolved.
    kernel : numpy array
        Convolution kernel
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

    Returns
    -------
    numpy array
        Convolved data.

    """
    t, f = kernel.shape
    # use opencv's convolution for small kernels if available
    if _convolve_opencv is None:  # or t == 1 or f == 1:
        return _convolve_scipy(data, kernel, pad)
    else:
        return _convolve_opencv(data, kernel, pad)


class ConvolutionalLayer(FeedForwardLayer):
    """
    Convolutional network layer.

    Parameters
    ----------
    weights : numpy array, shape (num_channels, num_feature_maps, <kernel>)
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

    activation_fn : numpy ufunc, optional
        Activation function.

    """

    def __init__(self, weights, bias, stride=None, pad='valid',
                 activation_fn=None):
        super(ConvolutionalLayer, self).__init__(weights, bias, activation_fn)
        self.stride = stride
        self.pad = pad

    def activate(self, data, **kwargs):
        """
        Activate ConvolutionalLayer.

        Parameters
        ----------
        data : numpy array, shape (num_frames, num_bins, num_channels)
            Activate with this data.

        Returns
        -------
        numpy array, shape (num_frames, num_bins, num_features)
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
        if self.pad == 'valid':
            num_frames -= (size_time - 1)
            num_bins -= (size_freq - 1)
        elif self.pad != 'same':
            raise NotImplementedError('`pad` is neither "valid" nor "same"')

        # init the output array with Fortran ordering (column major)
        out = np.zeros((num_frames, num_bins, num_features),
                       dtype=NN_DTYPE, order='F')
        # iterate over all channels
        for c in range(num_channels):
            channel = data[:, :, c]
            # convolve each channel separately with each filter
            for w, weights in enumerate(self.weights[c]):
                conv = convolve(channel, weights, self.pad)
                out[:, :, w] += conv
        # add bias to each feature map and apply activation function
        out += self.bias
        if self.activation_fn is not None:
            self.activation_fn(out, out=out)

        # use only selected parts of the output
        if self.stride not in (None, 1, (1, 1)):
            out = out[::self.stride[0], ::self.stride[1]]

        return out


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
        Activate StrideLayer.

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
        If 'None' `stride` = `size`.
    axis : int, optional
        Pool along the given axis. If set, `size` is ignored.

    """

    def __init__(self, size, stride=None, axis=None):
        self.size = size
        if stride is None:
            stride = size
        self.stride = stride
        self.axis = axis

    def __setstate__(self, state):
        # restore pickled instance attributes
        self.__dict__.update(state)
        # TODO: old models do not have `axis`, thus create it
        #       remove this initialisation code after updating the models
        if not hasattr(self, 'axis'):
            self.axis = None

    def activate(self, data, **kwargs):
        """
        Activate MaxPoolLayer.

        Parameters
        ----------
        data : numpy array, shape (num_frames, num_bins[, num_channels])
            Activate with this data.

        Returns
        -------
        numpy array
            Max pooled data.

        """
        if self.axis is not None:
            if self.stride is not None:
                raise NotImplementedError('`axis` with `stride` not supported')
            return np.max(data, axis=self.axis)
        # define which part of the maximum filtered data to return
        slice_dim_1 = slice(self.size[0] // 2,
                            data.shape[0] - (self.size[0] - 1) // 2,
                            self.stride[0])
        slice_dim_2 = slice(self.size[1] // 2,
                            data.shape[1] - (self.size[1] - 1) // 2,
                            self.stride[1])

        # TODO: is constant mode the most appropriate?
        if len(data.shape) == 2:
            # filter the data as is
            return maximum_filter(data, self.size,
                                  mode='constant')[slice_dim_1, slice_dim_2]
        elif len(data.shape) == 3:
            # filter each channel separately
            data = [maximum_filter(data[:, :, c], self.size, mode='constant')
                    [slice_dim_1, slice_dim_2] for c in range(data.shape[2])]
            # join channels and return as array
            return np.dstack(data)
        else:
            ValueError('`data` must bei either 2 or 3-dimensional')


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
    activation_fn : numpy ufunc, optional
        Activation function.

    References
    ----------
    .. [1] "Batch Normalization: Accelerating Deep Network Training by Reducing
           Internal Covariate Shift"
           Sergey Ioffe and Christian Szegedy.
           http://arxiv.org/abs/1502.03167, 2015.

    """

    def __init__(self, beta, gamma, mean, inv_std, activation_fn=None):
        self.beta = beta
        self.gamma = gamma
        self.mean = mean
        self.inv_std = inv_std
        self.activation_fn = activation_fn

    def activate(self, data, **kwargs):
        """
        Activate BatchNormLayer.

        Parameters
        ----------
        data : numpy array
            Activate with this data.

        Returns
        -------
        numpy array
            Normalized data.

        """
        out = (data - self.mean) * (self.gamma * self.inv_std) + self.beta
        if self.activation_fn is not None:
            self.activation_fn(out, out=out)
        return out


class TransposeLayer(Layer):
    """
    Transpose layer.

    Parameters
    ----------
    axes : list of ints, optional
        By default, reverse the dimensions of the input, otherwise permute the
        axes of the input according to the values given.

    """

    def __init__(self, axes=None):
        self.axes = axes

    def activate(self, data, **kwargs):
        """
        Activate TransposeLayer.

        Parameters
        ----------
        data : numpy array
            Activate with this data.

        Returns
        -------
        numpy array
            Transposed data.

        """
        return np.transpose(data, self.axes)


class ReshapeLayer(Layer):
    """
    Reshape Layer.

    Parameters
    ----------
    newshape : int or tuple of ints
        The new shape should be compatible with the original shape. If
        an integer, then the result will be a 1-D array of that length.
        One shape dimension can be -1. In this case, the value is
        inferred from the length of the array and remaining dimensions.
    order : {'C', 'F', 'A'}, optional
        Index order or the input. See np.reshape for a detailed description.

    """

    def __init__(self, newshape, order='C'):
        self.newshape = newshape
        self.order = order

    def activate(self, data, **kwargs):
        """
        Activate ReshapeLayer.

        Parameters
        ----------
        data : numpy array
            Activate with this data.

        Returns
        -------
        numpy array
            Reshaped data.

        """
        return np.reshape(data, self.newshape, self.order)


class AverageLayer(Layer):
    """
    Average layer.

    Parameters
    ----------
    axis : None or int or tuple of ints, optional
        Axis or axes along which the means are computed. The default is to
        compute the mean of the flattened array.
    dtype : data-type, optional
        Type to use in computing the mean. For integer inputs, the default
        is `float64`; for floating point inputs, it is the same as the
        input dtype.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one.

    """

    def __init__(self, axis=None, dtype=None, keepdims=False):
        self.axis = axis
        self.dtype = dtype
        self.keepdims = keepdims

    def activate(self, data, **kwargs):
        """
        Activate AverageLayer.

        Parameters
        ----------
        data : numpy array
            Activate with this data.

        Returns
        -------
        numpy array
            Averaged data.

        """
        return np.mean(data, axis=self.axis, dtype=self.dtype,
                       keepdims=self.keepdims)


class PadLayer(Layer):
    """
    Padding layer that pads the input with a constant value.

    Parameters
    ----------
    width : int
        Width of the padding (only one value for all dimensions)
    axes : iterable
        Indices of axes to be padded
    value : float
        Value to be used for padding.

    """

    def __init__(self, width, axes, value=0.):
        self.width = width
        self.axes = axes
        self.value = value

    def activate(self, data, **kwargs):
        """
        Activate PadLayer.

        Parameters
        ----------
        data : numpy array
            Activate with this data.

        Returns
        -------
        numpy array
            Padded data.

        """
        shape = list(data.shape)
        data_idxs = [slice(None) for _ in range(len(shape))]
        for a in self.axes:
            shape[a] += self.width * 2
            data_idxs[a] = slice(self.width, -self.width)
        data_padded = np.full(tuple(shape), self.value)
        data_padded[tuple(data_idxs)] = data
        return data_padded


class TCNBlock(Layer):
    """
    TCN Block.

    Parameters
    ----------
    dilated_conv : ConvolutionalLayer or List thereof
        Layer(s) which performs the dilated convolution.
    dilation_rate : int, optional or List of ints
        Dilation rate(s) of the `dilated_conv` layer(s).
    activation_fn : numpy ufunc, optional
        Activation function to be applied after the dilated convolution.
    skip_conv : ConvolutionalLayer, optional
        Layer which convolves the output of the dilated convolution to be used
        as skip connection and added to the residual data. If 'None', the
        output after the activation function is used directly.
    residual_conv : ConvolutionalLayer, optional
        Layer which convolves the input data to have the same output dimension
        as the main activation path. If 'None', the input data is added
        directly to the output of the skip convolution.

    """

    def __init__(self, dilated_conv, dilation_rate, activation_fn=None,
                 skip_conv=None, residual_conv=None):
        self.dilated_conv = dilated_conv
        self.dilation_rate = dilation_rate
        self.activation_fn = activation_fn
        self.skip_conv = skip_conv
        self.residual_conv = residual_conv

    @staticmethod
    def _dilate_data(data, size, dilation_rate):
        if dilation_rate is None:
            return data
        # Note: TCNBlock supports only 1D convolutions, data is given as
        #       (time, freq=1, num_features), thus reshape to given size
        #       (time, kernel_size, num_features)
        #       to be able to convolve with normal 2D convolution
        # determine data shape and number of bytes per item
        t, f, n = data.shape
        i = data.itemsize
        assert f == 1, 'TCNBlock supports only 1D dilated convolutions.'
        # to be able to use as_strided we have to pad the data accordingly
        # pad twice the dilation_rate on each side with zeros
        zeros = np.zeros((dilation_rate * 2, f, n), dtype=data.dtype)
        padded_data = np.concatenate((zeros, data, zeros))
        # return a dilated view of the data
        return as_strided(padded_data,
                          shape=(t, size, n),
                          strides=(n * i, n * i * dilation_rate, i))

    def activate(self, data, **kwargs):
        """
        Activate TCNBlock.

        Parameters
        ----------
        data : numpy array
            Activate with this data.

        Returns
        -------
        tuple
            Dilated (and activated) data, skip connection.

        """
        # the layer uses multiple dilated convolutions
        if isinstance(self.dilated_conv, list):
            # layer has multiple dilated convolutions
            out = []
            for conv, rate in zip(self.dilated_conv, self.dilation_rate):
                size = conv.weights.shape[-1]
                out.append(conv(self._dilate_data(data, size, rate)))
            # concatenate their output
            out = np.concatenate(out, axis=-1)
        else:
            # layer has only a single dilated convolutions
            size = self.dilated_conv.weights.shape[-1]
            out = self.dilated_conv(self._dilate_data(data, size,
                                                      self.dilation_rate))
        if self.activation_fn is not None:
            out = self.activation_fn(out)
        if self.skip_conv is not None:
            out = self.skip_conv(out)
        if self.residual_conv is not None:
            res = self.residual_conv(data)
        else:
            res = data
        return res + out, out


class TCNLayer(Layer):
    """
    Temporal convolutional network layer.

    Parameters
    ----------
    tcn_blocks : list of TCNBlock instances
        TCN blocks which perform the dilated convolutions.
    activation_fn : numpy ufunc, optional
        Activation function to be applied after the TCN blocks.
    skip_connections : bool, optional
        Aggregate skip connections by summation.

    """

    def __init__(self, tcn_blocks, activation_fn=None, skip_connections=False):
        self.tcn_blocks = tcn_blocks
        self.skip_connections = skip_connections
        self.activation_fn = activation_fn

    def activate(self, data, **kwargs):
        """
        Activate TCNLayer.

        Parameters
        ----------
        data : numpy array (num_frames, num_inputs)
            Activate with this data.

        Returns
        -------
        numpy array or tuple
            Activations for this data. If `skip_connections` is 'True', a tuple
            with the summed skip connections as its second element is returned.

        """
        skip_connections = None
        for i, tcn_block in enumerate(self.tcn_blocks):
            data, skip = tcn_block(data)
            if i == 0:
                skip_connections = skip
            else:
                skip_connections += skip
        if self.activation_fn is not None:
            self.activation_fn(data, out=data)
        if self.skip_connections:
            return data, skip_connections
        else:
            return data
