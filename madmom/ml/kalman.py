"""
This module contains code for Kalman filtering.


"""

from __future__ import absolute_import, division, print_function

import numpy as np


class KalmanFilter(object):
    """
    Kalman Filter Model

    Implementation of a linear dynamical system (LDS) aka Kalman filter.

    Parameters
    ----------
    A : numpy array
        Transition matrix of the LDS
    C : numpy array
        Observation matrix of the LDS
    Q : numpy array
        Covariance matrix of the Gaussian transition noise
    R : numpy array
        Covariance matrix of the Gaussian observation noise

    """

    def __init__(self, A, C, Q, R):
        self.A = A
        self.C = C
        self.Q = Q
        self.R = R
        self.state_dim = A.shape[0]

    def forward(self, x, P, y):
        """
        Compute forward path

        Parameters
        ----------
        x : numpy array
           hidden state means of the last time step
        P : numpy array
           hidden state covariances of the last time step
        y : numpy array
           observation

        Returns
        -------
        x : numpy array
           hidden state means of the current time step (filtering distribution)
        P : numpy array
           hidden state covariances of the current time step (filtering
           distribution)
        """
        # predict new means
        x = np.dot(self.A, x)
        # predict new covariances
        P = np.dot(np.dot(self.A, P), self.A.transpose()) + self.Q
        # predict new error covariances
        covE = np.dot(np.dot(self.C.transpose(), P), self.C) + self.R
        # compute Kalman gain
        K = np.dot(P, self.C.transpose()) / covE
        # Error between observation and prediction
        E = y - np.dot(self.C, x)
        # correct the means by weighted average of prediction and observation
        x = x + K * E
        # correct the covariances
        P = np.dot((np.eye(self.state_dim) - K[:, np.newaxis] * self.C), P)
        return x, P