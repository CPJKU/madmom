# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-few-public-methods
"""
This module contains recurrent neural network (RNN) related functionality.

It's main purpose is to serve as a substitute for testing neural networks
which were trained by other ML packages or programs without requiring these
packages or programs as dependencies.

The only allowed dependencies are Python + numpy + scipy.

The structure reflects just the needed functionality for testing networks. This
module is not meant to be a general purpose RNN with lots of functionality.
Just use one of the many NN/ML packages out there if you need training or any
other stuff.

"""

from __future__ import absolute_import, division, print_function

import numpy as np

from ..processors import Processor, ParallelProcessor


# naming infix for bidirectional layer
NN_DTYPE = np.float32
REVERSE = 'reverse'


# transfer functions
def linear(x, out=None):
    """
    Linear function.

    Parameters
    ----------
    x : numpy array
        Input data.
    out : numpy array, optional
        Array to hold the output data.

    Returns
    -------
    numpy array
        Unaltered input data.

    """
    if out is None or x is out:
        return x
    out[:] = x
    return out


tanh = np.tanh


def _sigmoid(x, out=None):
    """
    Logistic sigmoid function.

    Parameters
    ----------
    x : numpy array
        Input data.
    out : numpy array, optional
        Array to hold the output data.

    Returns
    -------
    numpy array
        Logistic sigmoid of input data.

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
    # pylint: disable=no-name-in-module
    # pylint: disable=wrong-import-order
    # pylint: disable=wrong-import-position

    # try to use a faster sigmoid function
    from distutils.version import LooseVersion
    from scipy.version import version as scipy_version
    # we need a recent version of scipy, older have a bug in expit
    # https://github.com/scipy/scipy/issues/3385
    if LooseVersion(scipy_version) < LooseVersion("0.14"):
        # Note: Raising an AttributeError might not be the best idea ever
        #       (i.e. ImportError would be more appropriate), but older
        #       versions of scipy not having the expit function raise the same
        #       error. In some cases this check fails, don't know why...
        raise AttributeError
    from scipy.special import expit as sigmoid
except AttributeError:
    sigmoid = _sigmoid


def relu(x, out=None):
    """
    Rectified linear (unit) transfer function.

    Parameters
    ----------
    x : numpy array
        Input data.
    out : numpy array, optional
        Array to hold the output data.

    Returns
    -------
    numpy array
        Rectified linear of input data.

    """
    if out is None:
        return np.maximum(x, 0)
    np.maximum(x, 0, out)
    return out


def softmax(x, out=None):
    """
    Softmax transfer function.

    Parameters
    ----------
    x : numpy array
        Input data.
    out : numpy array, optional
        Array to hold the output data.

    Returns
    -------
    numpy array
        Softmax of input data.

    """
    # determine maximum (over classes)
    tmp = np.amax(x, axis=1, keepdims=True)
    # exp of the input minus the max
    if out is None:
        out = np.exp(x - tmp)
    else:
        np.exp(x - tmp, out=out)
    # normalize by the sum (reusing the tmp variable)
    np.sum(out, axis=1, keepdims=True, out=tmp)
    out /= tmp
    return out


# network layer classes
class FeedForwardLayer(object):
    """
    Feed-forward network layer.

    Parameters
    ----------
    weights : numpy array, shape ()
        Weights.
    bias : scalar or numpy array, shape ()
        Bias.
    transfer_fn : numpy ufunc
        Transfer function.

    """

    def __init__(self, weights, bias, transfer_fn):
        self.weights = weights.copy()
        self.bias = bias.flatten()
        self.transfer_fn = transfer_fn

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
        return self.transfer_fn(np.dot(data, self.weights) + self.bias)


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
    transfer_fn : numpy ufunc
        Transfer function.

    """

    def __init__(self, weights, bias, recurrent_weights, transfer_fn):
        super(RecurrentLayer, self).__init__(weights, bias, transfer_fn)
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
            self.transfer_fn(out[i], out=out[i])
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
    transfer_fn : numpy ufunc, optional
        Transfer function.

    """

    def __init__(self, weights, bias, recurrent_weights, transfer_fn=tanh):
        self.weights = weights.copy()
        self.bias = bias.flatten()
        self.recurrent_weights = recurrent_weights.copy()
        # Note: define the peephole_weights here, so we don't have to define a
        #       different activate() method for the Gate subclass
        self.peephole_weights = None
        self.transfer_fn = transfer_fn
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
        self.transfer_fn(self.cell, out=self.cell)
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
    transfer_fn : numpy ufunc, optional
        Transfer function.

    """

    def __init__(self, weights, bias, recurrent_weights, peephole_weights,
                 transfer_fn=sigmoid):
        super(Gate, self).__init__(weights, bias, recurrent_weights,
                                   transfer_fn=transfer_fn)
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
    transfer_fn : numpy ufunc, optional
        Transfer function.

    """

    def __init__(self, weights, bias, recurrent_weights, peephole_weights,
                 transfer_fn=tanh):
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
        self.transfer_fn = transfer_fn

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
            out_ = self.transfer_fn(state_) * og
            out[i] = out_
        return out


# network class
class RecurrentNeuralNetwork(Processor):
    """
    Recurrent Neural Network (RNN) class.

    Parameters
    ----------
    layers : list
        Layers of the RNN.

    """

    def __init__(self, layers=None):
        self.layers = layers

    @classmethod
    def load(cls, filename):
        """
        Instantiate a RecurrentNeuralNetwork from a .npz model file.

        Parameters
        ----------
        filename : str
            Name of the .npz file with the RNN model.

        Returns
        -------
        :class:`RecurrentNeuralNetwork` instance
            RNN instance

        """
        import re
        # native numpy .npz format or pickled dictionary
        data = np.load(filename)

        # determine the number of layers (i.e. all "layer_%d_" occurrences)
        num_layers = max([int(re.findall(r'layer_(\d+)_', k)[0]) for
                          k in list(data.keys()) if k.startswith('layer_')])

        # function for layer creation with the given parameters
        def create_layer(params):
            """

            Parameters
            ----------
            params : dict
                Parameters for layer creation.

            Returns
            -------
            layer : Layer instance
                A network layer.

            """
            # first check if we need to create a bidirectional layer
            bwd_layer = None

            if '%s_type' % REVERSE in list(params.keys()):
                # pop the parameters needed for the reverse (backward) layer
                bwd_type = bytes(params.pop('%s_type' % REVERSE))
                bwd_transfer_fn = bytes(params.pop('%s_transfer_fn' %
                                                   REVERSE))
                bwd_params = dict((k.split('_', 1)[1], params.pop(k))
                                  for k in list(params.keys()) if
                                  k.startswith('%s_' % REVERSE))
                bwd_params['transfer_fn'] = globals()[bwd_transfer_fn.decode()]
                # construct the layer
                bwd_layer = globals()['%sLayer' % bwd_type.decode()](
                    **bwd_params)

            # pop the parameters needed for the normal (forward) layer
            fwd_type = bytes(params.pop('type'))
            fwd_transfer_fn = bytes(params.pop('transfer_fn'))
            fwd_params = params
            fwd_params['transfer_fn'] = globals()[fwd_transfer_fn.decode()]
            # construct the layer
            fwd_layer = globals()['%sLayer' % fwd_type.decode()](**fwd_params)

            # return the (bidirectional) layer
            if bwd_layer is not None:
                # construct a bidirectional layer with the forward and backward
                # layers and return it
                return BidirectionalLayer(fwd_layer, bwd_layer)
            else:
                # just return the forward layer
                return fwd_layer

        # loop over all layers
        layers = []
        for i in range(num_layers + 1):
            # get all parameters for that layer
            layer_params = dict((k.split('_', 2)[2], data[k])
                                for k in list(data.keys()) if
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

        Parameters
        ----------
        data : numpy array
            Activate the network with this data.

        Returns
        -------
        numpy array
            Network predictions for this data.

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

    Parameters
    ----------
    nn_files : list
        List of files with the RNN models.
    num_threads : int, optional
        Number of parallel working threads.

    """

    def __init__(self, nn_files, num_threads=None, **kwargs):
        # pylint: disable=unused-argument
        if not nn_files:
            raise ValueError('at least one RNN model must be given.')
        nn_models = []
        for nn_file in nn_files:
            nn_models.append(RecurrentNeuralNetwork.load(nn_file))
        # instantiate ParallelProcessor
        super(RNNProcessor, self).__init__(nn_models, num_threads=num_threads)

    @staticmethod
    def add_arguments(parser, nn_files):
        """
        Add recurrent neural network testing options to an existing parser.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        nn_files : list
            RNN model files.

        Returns
        -------
        argparse argument group
            Recurrent neural network argument parser group.

        """
        # pylint: disable=signature-differs
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
    Returns the average of all predictions.

    Parameters
    ----------
    predictions : list
        Predictions (i.e. NN activation functions).

    Returns
    -------
    numpy array
        Averaged prediction.

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
