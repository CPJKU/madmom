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

from madmom.processors import Processor, ParallelProcessor


# naming infix for bidirectional layer
REVERSE = 'reverse'


# network layer classes
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

    # @classmethod
    # def load(cls, filename):
    #     """
    #     Instantiate a RecurrentNeuralNetwork from a .npz model file.
    #
    #     Parameters
    #     ----------
    #     filename : str
    #         Name of the .npz file with the RNN model.
    #
    #     Returns
    #     -------
    #     :class:`RecurrentNeuralNetwork` instance
    #         RNN instance
    #
    #     """
    #     import re
    #     # native numpy .npz format or pickled dictionary
    #     data = np.load(filename)
    #
    #     # determine the number of layers (i.e. all "layer_%d_" occurrences)
    #     num_layers = max([int(re.findall(r'layer_(\d+)_', k)[0]) for
    #                       k in list(data.keys()) if k.startswith('layer_')])
    #
    #     # function for layer creation with the given parameters
    #     def create_layer(params):
    #         """
    #
    #         Parameters
    #         ----------
    #         params : dict
    #             Parameters for layer creation.
    #
    #         Returns
    #         -------
    #         layer : Layer instance
    #             A network layer.
    #
    #         """
    #         # first check if we need to create a bidirectional layer
    #         bwd_layer = None
    #
    #         if '%s_type' % REVERSE in list(params.keys()):
    #             # pop the parameters needed for the reverse (backward) layer
    #             bwd_type = bytes(params.pop('%s_type' % REVERSE))
    #             bwd_transfer_fn = bytes(params.pop('%s_transfer_fn' %
    #                                                REVERSE))
    #             bwd_params = dict((k.split('_', 1)[1], params.pop(k))
    #                               for k in list(params.keys()) if
    #                               k.startswith('%s_' % REVERSE))
    #             bwd_params['transfer_fn'] = globals()[bwd_transfer_fn.decode()]
    #             # construct the layer
    #             bwd_layer = globals()['%sLayer' % bwd_type.decode()](
    #                 **bwd_params)
    #
    #         # pop the parameters needed for the normal (forward) layer
    #         fwd_type = bytes(params.pop('type'))
    #         fwd_transfer_fn = bytes(params.pop('transfer_fn'))
    #         fwd_params = params
    #         fwd_params['transfer_fn'] = globals()[fwd_transfer_fn.decode()]
    #         # construct the layer
    #         fwd_layer = globals()['%sLayer' % fwd_type.decode()](**fwd_params)
    #
    #         # return the (bidirectional) layer
    #         if bwd_layer is not None:
    #             # construct a bidirectional layer with the forward and backward
    #             # layers and return it
    #             return BidirectionalLayer(fwd_layer, bwd_layer)
    #         else:
    #             # just return the forward layer
    #             return fwd_layer
    #
    #     # loop over all layers
    #     layers = []
    #     for i in range(num_layers + 1):
    #         # get all parameters for that layer
    #         layer_params = dict((k.split('_', 2)[2], data[k])
    #                             for k in list(data.keys()) if
    #                             k.startswith('layer_%d' % i))
    #         # create a layer from these parameters
    #         layer = create_layer(layer_params)
    #         # add to the model
    #         layers.append(layer)
    #     # instantiate a RNN from the layers and return it
    #     return cls(layers)

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
