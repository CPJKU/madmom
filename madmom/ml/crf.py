# encoding: utf-8
"""
This module contains an implementation of Conditional Random Fields (CRFs)
"""
import numpy as np

from ..processors import Processor


class ConditionalRandomField(Processor):
    """
    Implements a linear-chain Conditional Random Field using a
    matrix-based definition:

    P(Y|X) = exp(E(Y,X)) / Σ_Y'[E(Y', X)]
    E(Y,X) = Σ_{i=1..N} [ y_{n-1}.T * A * y_n + y_n.T * c + x_n.T * W * y_n ] +
             y_0.T * π + y_N.T * τ

    Parameters:
    -----------
    pi : numpy array
        Initial potential of the CRF. Also defines the number of states.
    tau : numpy array
        Potential of the last variable of the CRF.
    c : numpy array
        Label bias potential
    A : numpy array
        Matrix defining the transition potentials, where the rows are the
        'from' dimension, and columns the 'to' dimension.
    W : numpy array
        Matrix defining the observation potentials, where the rows are the
        'observation' dimension, and columns the 'state' dimension
    """

    def __init__(self, pi, tau, c, A, W):
        self.pi = pi
        self.tau = tau
        self.c = c
        self.A = A
        self.W = W

    def process(self, observations):
        """
        Determine the most probable configuration of Y (state sequence given X:

            y* = argmax_y P(Y|X)

        Parameters
        ----------
        observations : numpy array
            Observations to decode the most probable state sequence for (X)

        Returns
        -------
        y_star : numpy array
            Most probable state sequence
        """
        num_observations = len(observations)
        num_states = len(self.pi)
        bt_pointers = np.empty((num_observations, num_states), dtype=np.uint32)
        v = self.pi.copy()
        y_star = np.empty(num_observations, dtype=np.uint32)

        for i in range(num_observations):
            all_trans = self.A + v[:, np.newaxis]
            best_trans = np.max(all_trans, axis=0)
            bt_pointers[i] = np.argmax(all_trans, axis=0)
            v = self.c + np.dot(observations[i], self.W) + best_trans

        v += self.tau

        y_star[-1] = np.argmax(v)
        for i in range(len(y_star) - 1)[::-1]:
            y_star[i] = bt_pointers[i + 1, y_star[i + 1]]

        return y_star
