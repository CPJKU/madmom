# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments

"""
This file contains some statistical functionality.

"""

import numpy as np


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
