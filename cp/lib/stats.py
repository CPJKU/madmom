"""
    Self-defined probability distributions
"""

__docformat__ = "restructuredtext en"

import numpy as np


class TanhProb:

    def __init__(self, i, phi, lmbda):
        self.k = 1.0 / phi
        self.d = -self.k * i
        self.lmbda = lmbda
        self.a = self.k / (self.k * self.lmbda +
                           np.log(np.cosh(self.k + self.d)) +
                           self.k - np.log(np.cosh(self.d)))

    def __call__(self, x):
        return self.a * (np.tanh(self.k * x + self.d) + 1 + self.lmbda)

    @staticmethod
    def pdf(x, i, phi, lmbda):
        """ PDF of the tanh probability distribution."""
        k = 1.0 / phi
        d = -k * i
        a = k / (k * lmbda + np.log(np.cosh(k + d)) + k - np.log(np.cosh(d)))
        return a * (np.tanh(k * x + d) + 1 + lmbda)
