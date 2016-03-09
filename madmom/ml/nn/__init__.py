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

    def __init__(self, layers):
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
