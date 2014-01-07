#!/usr/bin/env python
# encoding: utf-8
"""
This file contains some statistical functionality.

@author: Filip Korzeniowski <filip.korzeniowski@jku.at>

"""

import numpy as np


class TanhProb(object):
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

        :param i:     defines the transition point between the shelves
        :param phi:   defines the steepness of the transition (the smaller, the
                      steeper)
        :param lmbda: defines the relative difference between the pdf values
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

        :param x: values for which the pdf shall be computed
        :returns: PDF at the positions passed in `x`
        """
        return self.a * (np.tanh(self.k * x + self.d) + 1 + self.lmbda)

    @staticmethod
    def pdf(x, i, phi, lmbda):
        """
        PDF of the tanh probability distribution. Take a look at the
        documentation of the __init__() method for a description of the
        parameters.

        :param i:     i-parameter of the tanh distribution
        :param phi:   phi-parameter of the tanh distribution
        :param lmbda: lambda parameter of the tahn distribution
        :param x:     values for which the PDF shall be computed
        :returns:     PDF at the positions given in `x`

        """
        k = 1.0 / phi
        d = -k * i
        a = k / (k * lmbda + np.log(np.cosh(k + d)) + k - np.log(np.cosh(d)))
        return a * (np.tanh(k * x + d) + 1 + lmbda)
