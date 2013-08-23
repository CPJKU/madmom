"""
    Self-defined probability distributions
"""

__docformat__ = "restructuredtext en"

import numpy as np


class TanhProb:
    """
    Class for computing the Tanh probability function. It also contains
    a static method to compute the PDF. Some parameters are computed in
    advance when using the class, so if you need the pdf with the same
    parameters multiple times it should be faster to compute it using
    the class interface rather than the static function.
    """

    def __init__(self, i, phi, lmbda):
        """
        Initialises the parameters of the PDF.

        :Parameters:
            - `i`: Defines the transition point between the shelves
            - `phi`: Defines the steepness of the transition (the smaller, the
                     steeper)
            - `lmbda`: Defines the relative difference between the pdf values
                       at 0 and 1. A value of 0 means maximal difference.
        """
        self.k = 1.0 / phi
        self.d = -self.k * i
        self.lmbda = lmbda
        self.a = self.k / (self.k * self.lmbda +
                           np.log(np.cosh(self.k + self.d)) +
                           self.k - np.log(np.cosh(self.d)))

    def __call__(self, x):
        """
        Compute the PDF.

        :Parameters:
            - `x`: value or numpy array of values for which the pdf shall be
                   computed
        :Returns:
            PDF at the positions passed in `x`
        """
        return self.a * (np.tanh(self.k * x + self.d) + 1 + self.lmbda)

    @staticmethod
    def pdf(x, i, phi, lmbda):
        """
        PDF of the tanh probability distribution. Take a look at the
        documentation of the __init__ method for a description of the
        parameters.

        :Parameters:
            `i`: i-parameter of the tanh distribution
            `phi`: phi-parameter of the tanh distribution
            `lmbda`: lambda parameter of the tahn distribution
            `x`: Values for which the PDF shall be computed

        :Returns:
            PDF at the positions given in `x`
        """
        k = 1.0 / phi
        d = -k * i
        a = k / (k * lmbda + np.log(np.cosh(k + d)) + k - np.log(np.cosh(d)))
        return a * (np.tanh(k * x + d) + 1 + lmbda)
