# encoding: utf-8
"""
This file contains tests for the madmom.audio.filters module.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""
# pylint: skip-file

import unittest

from madmom.audio.comb_filters import *

sig_1d = np.asarray([0, 0, 1, 0, 0, 1, 0, 0, 1], dtype=np.float)
sig_2d = np.asarray([[0, 0, 1, 0, 0, 1, 0, 0, 1],
                     [1, 0, 1, 0, 1, 0, 1, 0, 1]], dtype=np.float).T

res_1d_bw_2 = np.asarray([0, 0, 1, 0, 0.5, 1, 0.25, 0.5, 1.125])

res_1d_bw_3 = np.asarray([0, 0, 1, 0, 0, 1.5, 0, 0, 1.75])

res_2d_bw_2 = np.asarray([[0, 0, 1, 0, 0.5, 1, 0.25, 0.5, 1.125],
                          [1, 0, 1.5, 0, 1.75, 0, 1.875, 0, 1.9375]]).T

res_2d_bw_3 = np.asarray([[0, 0, 1, 0, 0, 1.5, 0, 0, 1.75],
                          [1, 0, 1, 0.5, 1, 0.5, 1.25, 0.5, 1.25]]).T


class TestFeedBackwardFilterFunction(unittest.TestCase):

    def test_values(self):
        result = feed_backward_comb_filter(sig_1d, 2, 0.5)
        self.assertTrue(np.allclose(result, res_1d_bw_2))
        result = feed_backward_comb_filter(sig_1d, 3, 0.5)
        self.assertTrue(np.allclose(result, res_1d_bw_3))
        result = feed_backward_comb_filter(sig_2d, 2, 0.5)
        self.assertTrue(np.allclose(result, res_2d_bw_2))
        result = feed_backward_comb_filter(sig_2d, 3, 0.5)
        self.assertTrue(np.allclose(result, res_2d_bw_3))


class TestFeedBackwardFilter1DFunction(unittest.TestCase):

    def test_types(self):
        result = feed_backward_comb_filter_1d(sig_1d, 2, 0.5)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(type(result), float)

    def test_values(self):
        with self.assertRaises(ValueError):
            feed_backward_comb_filter_1d(sig_1d, 0, 0.5)
        with self.assertRaises(ValueError):
            feed_backward_comb_filter_1d(sig_2d, 0, 0.5)
        result = feed_backward_comb_filter_1d(sig_1d, 2, 0.5)
        self.assertTrue(np.allclose(result, res_1d_bw_2))
        result = feed_backward_comb_filter_1d(sig_1d, 3, 0.5)
        self.assertTrue(np.allclose(result, res_1d_bw_3))


class TestFeedBackwardFilter2DFunction(unittest.TestCase):

    def test_types(self):
        result = feed_backward_comb_filter_2d(sig_2d, 2, 0.5)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(type(result), float)

    def test_values(self):
        with self.assertRaises(ValueError):
            feed_backward_comb_filter_2d(sig_2d, 0, 0.5)
        with self.assertRaises(ValueError):
            feed_backward_comb_filter_2d(sig_1d, 0, 0.5)
        result = feed_backward_comb_filter_2d(sig_2d, 2, 0.5)
        self.assertTrue(np.allclose(result, res_2d_bw_2))
        result = feed_backward_comb_filter_2d(sig_2d, 3, 0.5)
        self.assertTrue(np.allclose(result, res_2d_bw_3))
