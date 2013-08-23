#!/usr/bin/env python
# encoding: utf-8
"""
Copyright (c) 2013 Filip Korzeniowski <filip.korzeniowski@jku.at>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

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
