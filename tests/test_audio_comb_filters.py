# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.audio.comb_filters module.

"""

from __future__ import absolute_import, division, print_function

import unittest

from madmom.audio.comb_filters import *

sig_1d = np.asarray([0, 0, 1, 0, 0, 1, 0, 0, 1], dtype=np.float)
sig_2d = np.asarray([[0, 0, 1, 0, 0, 1, 0, 0, 1],
                     [1, 0, 1, 0, 1, 0, 1, 0, 1]], dtype=np.float).T


# test forward filters
res_1d_fw_2 = np.asarray([0, 0, 1, 0, 0.5, 1, 0, 0.5, 1])
res_1d_fw_3 = np.asarray([0, 0, 1, 0, 0, 1.5, 0, 0, 1.5])
res_2d_fw_2 = np.asarray([[0, 0, 1, 0, 0.5, 1, 0, 0.5, 1],
                          [1, 0, 1.5, 0, 1.5, 0, 1.5, 0, 1.5]]).T
res_2d_fw_3 = np.asarray([[0, 0, 1, 0, 0, 1.5, 0, 0, 1.5],
                          [1, 0, 1, 0.5, 1, 0.5, 1, 0.5, 1]]).T


class TestFeedForwardFilterFunction(unittest.TestCase):

    def test_types(self):
        result = feed_forward_comb_filter(sig_1d, 2, 0.5)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(type(result), float)

    def test_values(self):
        result = feed_forward_comb_filter(sig_1d, 2, 0.5)
        self.assertTrue(np.allclose(result, res_1d_fw_2))
        result = feed_forward_comb_filter(sig_1d, 3, 0.5)
        self.assertTrue(np.allclose(result, res_1d_fw_3))
        result = feed_forward_comb_filter(sig_2d, 2, 0.5)
        self.assertTrue(np.allclose(result, res_2d_fw_2))
        result = feed_forward_comb_filter(sig_2d, 3, 0.5)
        self.assertTrue(np.allclose(result, res_2d_fw_3))


# test backward filters
res_1d_bw_2 = np.asarray([0, 0, 1, 0, 0.5, 1, 0.25, 0.5, 1.125])

res_1d_bw_3 = np.asarray([0, 0, 1, 0, 0, 1.5, 0, 0, 1.75])

res_2d_bw_2 = np.asarray([[0, 0, 1, 0, 0.5, 1, 0.25, 0.5, 1.125],
                          [1, 0, 1.5, 0, 1.75, 0, 1.875, 0, 1.9375]]).T

res_2d_bw_3 = np.asarray([[0, 0, 1, 0, 0, 1.5, 0, 0, 1.75],
                          [1, 0, 1, 0.5, 1, 0.5, 1.25, 0.5, 1.25]]).T


class TestFeedBackwardFilterFunction(unittest.TestCase):

    def test_types(self):
        result = feed_forward_comb_filter(sig_1d, 2, 0.5)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(type(result), float)
        result = feed_backward_comb_filter(sig_2d, 2, 0.5)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(type(result), float)

    def test_values(self):
        result = feed_backward_comb_filter(sig_1d, 2, 0.5)
        self.assertTrue(np.allclose(result, res_1d_bw_2))
        result = feed_backward_comb_filter(sig_1d, 3, 0.5)
        self.assertTrue(np.allclose(result, res_1d_bw_3))
        result = feed_backward_comb_filter(sig_2d, 2, 0.5)
        self.assertTrue(np.allclose(result, res_2d_bw_2))
        result = feed_backward_comb_filter(sig_2d, 3, 0.5)
        self.assertTrue(np.allclose(result, res_2d_bw_3))


# test other stuff
class TestCombFilterFunction(unittest.TestCase):

    def test_types(self):
        function = feed_backward_comb_filter
        result = comb_filter(sig_1d, function, [2, 3], [0.5, 0.5])
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(type(result), float)
        function = feed_forward_comb_filter
        result = comb_filter(sig_1d, function, [2, 3], [0.5, 0.5])
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(type(result), float)

    def test_values_backward(self):
        function = feed_backward_comb_filter
        result = comb_filter(sig_1d, function, [2, 3], [0.5, 0.5])
        self.assertTrue(np.allclose(result[:, 0], res_1d_bw_2))
        self.assertTrue(np.allclose(result[:, 1], res_1d_bw_3))
        result = comb_filter(sig_2d, function, [2, 3], [0.5, 0.5])
        self.assertTrue(np.allclose(result[:, :, 0], res_2d_bw_2))
        self.assertTrue(np.allclose(result[:, :, 1], res_2d_bw_3))

    def test_values_forward(self):
        function = feed_forward_comb_filter
        result = comb_filter(sig_1d, function, [2, 3], [0.5, 0.5])
        self.assertTrue(np.allclose(result[:, 0], res_1d_fw_2))
        self.assertTrue(np.allclose(result[:, 1], res_1d_fw_3))
        result = comb_filter(sig_2d, function, [2, 3], [0.5, 0.5])
        self.assertTrue(np.allclose(result[:, :, 0], res_2d_fw_2))
        self.assertTrue(np.allclose(result[:, :, 1], res_2d_fw_3))


class TestCombFilterbankClass(unittest.TestCase):

    def test_types(self):
        # backward function
        processor = CombFilterbankProcessor(feed_backward_comb_filter,
                                            [2, 3], [0.5, 0.5])
        self.assertIsInstance(processor, CombFilterbankProcessor)
        self.assertIsInstance(processor, Processor)
        self.assertTrue(processor.filter_function == feed_backward_comb_filter)
        processor = CombFilterbankProcessor('backward', [2, 3], [0.5, 0.5])
        self.assertTrue(processor.filter_function == feed_backward_comb_filter)
        # forward function
        processor = CombFilterbankProcessor(feed_forward_comb_filter,
                                            [2, 3], [0.5, 0.5])
        self.assertTrue(processor.filter_function == feed_forward_comb_filter)
        processor = CombFilterbankProcessor('forward', [2, 3], [0.5, 0.5])
        self.assertTrue(processor.filter_function == feed_forward_comb_filter)

    def test_errors(self):
        with self.assertRaises(ValueError):
            CombFilterbankProcessor('xyz', [2, 3], [0.5, 0.5])

    def test_values_backward(self):
        processor = CombFilterbankProcessor(feed_backward_comb_filter,
                                            [2, 3], [0.5, 0.5])
        result = processor.process(sig_1d)
        self.assertTrue(np.allclose(result[:, 0], res_1d_bw_2))
        self.assertTrue(np.allclose(result[:, 1], res_1d_bw_3))
        result = processor.process(sig_2d)
        self.assertTrue(np.allclose(result[:, :, 0], res_2d_bw_2))
        self.assertTrue(np.allclose(result[:, :, 1], res_2d_bw_3))

    def test_values_forward(self):
        processor = CombFilterbankProcessor(feed_forward_comb_filter,
                                            [2, 3], [0.5, 0.5])
        result = processor.process(sig_1d)
        self.assertTrue(np.allclose(result[:, 0], res_1d_fw_2))
        self.assertTrue(np.allclose(result[:, 1], res_1d_fw_3))
        result = processor.process(sig_2d)
        self.assertTrue(np.allclose(result[:, :, 0], res_2d_fw_2))
        self.assertTrue(np.allclose(result[:, :, 1], res_2d_fw_3))
