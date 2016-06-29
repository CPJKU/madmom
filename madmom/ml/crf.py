# encoding: utf-8
"""
This module contains an implementation of Conditional Random Fields (CRFs)
"""
# pylint: disable=no-member
# pylint: disable=invalid-name
import numpy as np

from ..processors import Processor


class ConditionalRandomField(Processor):
    """
    Implements a linear-chain Conditional Random Field using a
    matrix-based definition:

    .. math::
        P(Y|X) = exp[E(Y,X)] / Σ_{Y'}[E(Y', X)]

        E(Y,X) = Σ_{i=1}^{N} [y_{n-1}^T  A  y_n + y_n^T c + x_n^T W y_n ] +
                y_0^T π + y_N^T τ,

    where Y is a sequence of labels in one-hot encoding and X are the observed
    features.

    Parameters
    ----------
    initial : numpy array
        Initial potential (π) of the CRF. Also defines the number of states.
    final : numpy array
        Potential (τ) of the last variable of the CRF.
    bias : numpy array
        Label bias potential (c).
    transition : numpy array
        Matrix defining the transition potentials (A), where the rows are the
        'from' dimension, and columns the 'to' dimension.
    observation : numpy array
        Matrix defining the observation potentials (W), where the rows are the
        'observation' dimension, and columns the 'state' dimension.
    """

    def __init__(self, initial, final, bias, transition, observation):
        self.pi = initial
        self.tau = final
        self.c = bias
        self.A = transition
        self.W = observation

    def process(self, observations):
        """
        Determine the most probable configuration of Y given the state
        sequence x:

        .. math::
            y^* = argmax_y P(Y=y|X=x)

        Parameters
        ----------
        observations : numpy array
            Observations (x) to decode the most probable state sequence for.

        Returns
        -------
        y_star : numpy array
            Most probable state sequence.
        """
        num_observations = len(observations)
        num_states = len(self.pi)
        bt_pointers = np.empty((num_observations, num_states), dtype=np.uint32)
        viterbi = self.pi.copy()
        y_star = np.empty(num_observations, dtype=np.uint32)

        for i in range(num_observations):
            all_trans = self.A + viterbi[:, np.newaxis]
            best_trans = np.max(all_trans, axis=0)
            bt_pointers[i] = np.argmax(all_trans, axis=0)
            viterbi = self.c + np.dot(observations[i], self.W) + best_trans

        viterbi += self.tau

        y_star[-1] = np.argmax(viterbi)
        for i in range(len(y_star) - 1)[::-1]:
            y_star[i] = bt_pointers[i + 1, y_star[i + 1]]

        return y_star
