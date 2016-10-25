# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.processors module.

"""

from __future__ import absolute_import, division, print_function
import tempfile
import unittest
import sys

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


# clean up
def teardown():
    import os
    os.unlink(tmp_file)
