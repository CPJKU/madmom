# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.processors module.

"""

from __future__ import absolute_import, division, print_function
import tempfile
import unittest

from madmom.processors import *
from madmom.models import *
from madmom.ml.nn import NeuralNetwork

tmp_file = tempfile.NamedTemporaryFile(delete=False).name


class TestProcessor(unittest.TestCase):

    def test_unicode(self):
        if sys.version_info[0] == 2:
            # load from unicode string
            rnn = NeuralNetwork.load(unicode(ONSETS_RNN[0]))
            # save to unicode string
            rnn.dump(unicode(tmp_file))


class TestBufferProcessor(unittest.TestCase):

    def test_1d(self):
        buffer = BufferProcessor(5, init=np.zeros(5))
        self.assertTrue(np.allclose(buffer.buffer, 0))
        # shift in two new values
        result = buffer(np.arange(2))
        self.assertTrue(np.allclose(result, [0, 0, 0, 0, 1]))
        result = buffer(np.arange(2, 4))
        self.assertTrue(np.allclose(result, [0, 0, 1, 2, 3]))
        result = buffer(np.arange(4, 6))
        self.assertTrue(np.allclose(result, [1, 2, 3, 4, 5]))
        # shift in three new values
        result = buffer(np.arange(6, 9))
        self.assertTrue(np.allclose(result, [4, 5, 6, 7, 8]))

    def test_2d(self):
        buffer = BufferProcessor((5, 2), init=np.zeros((5, 2)))
        print(buffer.buffer)
        self.assertTrue(buffer.buffer.shape == (5, 2))
        self.assertTrue(np.allclose(buffer.buffer, 0))
        # shift in new values
        result = buffer(np.arange(2).reshape((1, -1)))
        self.assertTrue(result.shape == (5, 2))
        self.assertTrue(np.allclose(result[:4], 0))
        self.assertTrue(np.allclose(result[-1], [0, 1]))
        result = buffer(np.arange(2, 4).reshape((1, -1)))
        self.assertTrue(result.shape == (5, 2))
        self.assertTrue(np.allclose(result[:3], 0))
        self.assertTrue(np.allclose(result[-2], [0, 1]))
        self.assertTrue(np.allclose(result[-1], [2, 3]))
        # shift in two new values
        result = buffer(np.arange(4, 8).reshape((2, -1)))
        self.assertTrue(result.shape == (5, 2))
        self.assertTrue(np.allclose(result[0], 0))
        self.assertTrue(np.allclose(result[1], [0, 1]))
        self.assertTrue(np.allclose(result[2], [2, 3]))
        self.assertTrue(np.allclose(result[3], [4, 5]))
        self.assertTrue(np.allclose(result[4], [6, 7]))
        # shift in three new values
        result = buffer(np.arange(8, 14).reshape((3, -1)))
        self.assertTrue(result.shape == (5, 2))
        self.assertTrue(np.allclose(result.ravel(), np.arange(4, 14)))

    def test_reset(self):
        buffer = BufferProcessor(5, init=np.ones(5))
        self.assertTrue(np.allclose(buffer.buffer, 1))
        result = buffer(np.arange(2))
        self.assertTrue(np.allclose(result, [1, 1, 1, 0, 1]))
        buffer.reset()
        self.assertTrue(np.allclose(buffer.buffer, 1))


# clean up
def teardown():
    import os
    os.unlink(tmp_file)
