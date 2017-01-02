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
from ...processors import Processor, ParallelProcessor, SequentialProcessor


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

    Examples
    --------
    Create a NeuralNetwork from the given layers.

    >>> from madmom.ml.nn.layers import FeedForwardLayer
    >>> from madmom.ml.nn.activations import tanh, sigmoid
    >>> l1_weights = [[0.5, -1., -0.3 , -0.2]]
    >>> l1_bias = [0.05, 0., 0.8, -0.5]
    >>> l1 = FeedForwardLayer(l1_weights, l1_bias, activation_fn=tanh)
    >>> l2_weights = [-1, 0.9, -0.2 , 0.4]
    >>> l2_bias = [0.5]
    >>> l2 = FeedForwardLayer(l2_weights, l2_bias, activation_fn=sigmoid)
    >>> nn = NeuralNetwork([l1, l2])
    >>> nn  # doctest: +ELLIPSIS
    <madmom.ml.nn.NeuralNetwork object at 0x...>
    >>> nn(np.array([0, 0.5, 1, 0, 1, 2, 0]))  # doctest: +NORMALIZE_WHITESPACE
    array([ 0.53305, 0.36903, 0.265 , 0.53305, 0.265 , 0.18612, 0.53305])

    """

    def __init__(self, layers, online=False):
        self.layers = layers
        self.online = online

    def __setstate__(self, state):
        # restore instance attributes
        self.__dict__.update(state)
        # TODO: old models do not have the online attribute, thus create it
        #       remove this initialisation code after updating the models
        if not hasattr(self, 'online'):
            self.online = None

    def process(self, data):
        """
        Process the given data with the neural network.

        Parameters
        ----------
        data : numpy array
            Activate the network with this data.

        Returns
        -------
        numpy array
            Network predictions for this data.

        Notes
        -----
        Depending on online/offline mode the predictions are either reported
        on a frame-by-frame basis or for the whole sequence, respectively.

        """
        if self.online:
            return self.process_online(data)
        return self.process_offline(data)

    def process_offline(self, data):
        """
        Process the given data with the neural network.

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
            # activate the layer and feed the output into the next one
            data = layer(data)
        # ravel the predictions if needed
        if data.ndim == 2 and data.shape[1] == 1:
            data = data.ravel()
        return data

    def process_online(self, data):
        """
        Process the given data with the neural network.

        Parameters
        ----------
        data : numpy array
            Activate the network with this data.

        Returns
        -------
        numpy array
            Network predictions for this data.

        """
        # loop over all layers
        for layer in self.layers:
            # activate the layer and feed the output into the next one
            data = layer.activate_online(data)
        # ravel the predictions if needed
        if data.ndim == 2 and data.shape[1] == 1:
            data = data.ravel()
        return data

    def reset(self):
        """
        Reset the neural network to its initial state.

        """
        for layer in self.layers:
            layer.reset()


class NeuralNetworkEnsemble(SequentialProcessor):
    """
    Neural Network ensemble class.

    Parameters
    ----------
    networks : list
        List of the Neural Networks.
    ensemble_fn : function or callable, optional
        Ensemble function to be applied to the predictions of the neural
        network ensemble (default: average predictions).
    num_threads : int, optional
        Number of parallel working threads.

    Notes
    -----
    If `ensemble_fn` is set to 'None', the predictions are returned as a list
    with the same length as the number of networks given.

    Examples
    --------
    Create a NeuralNetworkEnsemble from the networks. Instead of supplying
    the neural networks as parameter, they can also be loaded from file:

    >>> from madmom.models import ONSETS_BRNN_PP
    >>> nn = NeuralNetworkEnsemble.load(ONSETS_BRNN_PP)
    >>> nn  # doctest: +ELLIPSIS
    <madmom.ml.nn.NeuralNetworkEnsemble object at 0x...>
    >>> nn(np.array([0, 0.5, 1, 0, 1, 2, 0]))  # doctest: +NORMALIZE_WHITESPACE
    array([ 0.00116, 0.00213, 0.01428, 0.00729, 0.0088 , 0.21965, 0.00532])

    """

    def __init__(self, networks, ensemble_fn=average_predictions,
                 num_threads=None, **kwargs):
        networks_processor = ParallelProcessor(networks,
                                               num_threads=num_threads)
        super(NeuralNetworkEnsemble, self).__init__((networks_processor,
                                                     ensemble_fn))

    @classmethod
    def load(cls, nn_files, **kwargs):
        """

        Parameters
        ----------
        nn_files : list
            List of neural network model file names.
        kwargs : dict, optional
            Keyword arguments passed to NeuralNetworkEnsemble.

        Returns
        -------
        NeuralNetworkEnsemble
            NeuralNetworkEnsemble instance.

        """
        networks = [NeuralNetwork.load(f) for f in nn_files]
        return cls(networks, **kwargs)

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
