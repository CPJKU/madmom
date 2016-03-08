# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-few-public-methods
"""
Neural Network package.

"""

from __future__ import absolute_import, division, print_function

import numpy as np

from . import layers, activations
from ...processors import Processor

from .layers import (FeedForwardLayer, RecurrentLayer, BidirectionalLayer,
                     LSTMLayer)
from .activations import linear, tanh, sigmoid


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


class NeuralNetwork(Processor):
    """
    Neural Network class.

    Parameters
    ----------
    layers : list
        Layers of the Neural Network.

    """

    def __init__(self, layers=None):
        self.layers = layers

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

    @staticmethod
    def add_arguments(parser, nn_files):
        """
        Add neural network options to an existing parser.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        nn_files : list
            Neural network model files.

        Returns
        -------
        argparse argument group
            Neural network argument parser group.

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

    @classmethod
    def load_npz(cls, filename, convert=False):
        """
        Instantiate a NeuralNetwork from a .npz model file (and pickle it).

        Parameters
        ----------
        filename : str
            Name of the .npz file with the RNN model.
        convert : bool, optional
            Convert the model to the new pickle format.

        Returns
        -------
        :class:`NeuralNetwork` instance
            NeuralNetwork instance

        """
        import os
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
            reverse = 'reverse'
            bwd_layer = None

            if '%s_type' % reverse in list(params.keys()):
                # pop the parameters needed for the reverse (backward) layer
                bwd_type = bytes(params.pop('%s_type' % reverse))
                bwd_act_fn = bytes(params.pop('%s_transfer_fn' % reverse))
                bwd_params = dict((k.split('_', 1)[1], params.pop(k))
                                  for k in list(params.keys()) if
                                  k.startswith('%s_' % reverse))
                bwd_params['activation_fn'] = globals()[bwd_act_fn.decode()]
                # construct the layer
                bwd_layer = globals()['%sLayer' % bwd_type.decode()](
                    **bwd_params)

            # pop the parameters needed for the normal (forward) layer
            fwd_type = bytes(params.pop('type'))
            fwd_act_fn = bytes(params.pop('transfer_fn'))
            fwd_params = params
            fwd_params['activation_fn'] = globals()[fwd_act_fn.decode()]
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
        obj = cls(layers)
        if convert:
            obj.dump('%s.pkl' % os.path.splitext(filename)[0])
        return obj
