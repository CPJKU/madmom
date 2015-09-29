#!/usr/bin/env python
# encoding: utf-8
"""
This file contains some statistical functionality.

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
        Initializes the parameters of the PDF.

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
        :return:  PDF at the positions passed in `x`
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
        :param lmbda: lambda parameter of the tanh distribution
        :param x:     values for which the PDF shall be computed
        :return:      PDF at the positions given in `x`

        """
        k = 1.0 / phi
        d = -k * i
        a = k / (k * lmbda + np.log(np.cosh(k + d)) + k - np.log(np.cosh(d)))
        return a * (np.tanh(k * x + d) + 1 + lmbda)


def mcnemar_test(test_1, test_2, significance=0.01):
    """
    Perform McNemar's statistical test.

    :param test_1:       Test 1 sample [numpy array]
    :param test_2:       Test 2 sample [numpy array]
    :param significance: significance level
    :return:             tuple (significance, p-value) [{-1, 0, +1}}, float]

    Please see: http://en.wikipedia.org/wiki/McNemar%27s_test


                    | Test 2 positive | Test 2 negative | Row total
    ----------------+-----------------+-----------------+----------
    Test 1 positive |      a          |      b          |   a + b
    Test 1 negative |      c          |      d          |   c + d
    ----------------+-----------------+-----------------+----------
    Column total    |    a + c        |    b + d        |     n

    """
    from scipy.stats import chi2
    # convert the tests to numpy arrays
    test_1 = np.asarray(test_1)
    test_2 = np.asarray(test_2)
    # both test must have the same length
    if not (test_1.size == test_2.size and test_1.shape == test_2.shape):
        raise ValueError("Both tests must have the same size and shape.")
    # calculate a, b, c, d
    # a = np.sum(test_1 * test_2)
    b = np.sum(test_1 > test_2)
    c = np.sum(test_1 < test_2)
    # d = np.sum(-test_1 * -test_2)
    # is the approximation ok?
    if b + c < 25:
        raise NotImplementedError("implement correct binomial distribution or "
                                  "use bigger sample sizes (b + c > 25)")
    # statistical test
    stat = (b - c) ** 2 / float(b + c)
    # test under chi square distribution
    p = chi2(1).sf(stat)
    # direction of significance
    sig = 0
    if p < significance:
        sig = 1 if b > c else -1
    return sig, p
