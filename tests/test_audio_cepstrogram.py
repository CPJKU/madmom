# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.audio.cepstrogram module.

"""

from __future__ import absolute_import, division, print_function

import unittest
from functools import partial
from os.path import join as pj

from madmom.audio.cepstrogram import MFCC, Cepstrogram
from madmom.audio.filters import MelFilterbank
from madmom.audio.spectrogram import *
from . import AUDIO_PATH

sample_file = pj(AUDIO_PATH, 'sample.wav')


class TestMFCCClass(unittest.TestCase):

    def setUp(self):
        self.mfcc = MFCC(sample_file)

    def test_types(self):
        self.assertIsInstance(self.mfcc, MFCC)
        self.assertIsInstance(self.mfcc, Cepstrogram)
        # attributes
        self.assertIsInstance(self.mfcc.filterbank, MelFilterbank)
        # properties
        self.assertIsInstance(self.mfcc.deltas, np.ndarray)
        self.assertIsInstance(self.mfcc.delta_deltas, np.ndarray)
        self.assertIsInstance(self.mfcc.num_bins, int)
        self.assertIsInstance(self.mfcc.num_frames, int)
        # wrong filterbank type
        with self.assertRaises(TypeError):
            FilteredSpectrogram(sample_file, filterbank='bla')

    def test_values(self):
        allclose = partial(np.allclose, rtol=1.e-3, atol=1.e-5)
        # values
        self.assertTrue(allclose(self.mfcc[0, :6],
                                 [-3.61102366, 6.81075716, 2.55457568,
                                  1.88377929, 1.04133379, 0.6382336]))
        self.assertTrue(allclose(self.mfcc[0, -6:],
                                 [-0.20386486, -0.18468723, -0.00233107,
                                  0.20703268, 0.21419463, 0.00598407]))
        # attributes
        self.assertTrue(self.mfcc.shape == (281, 30))
        # properties
        self.assertEqual(self.mfcc.num_bins, 30)
        self.assertEqual(self.mfcc.num_frames, 281)

    def test_deltas(self):
        allclose = partial(np.allclose, rtol=1.e-2, atol=1.e-4)
        # don't compare first element because it is dependent on the
        # padding used for filtering
        self.assertTrue(allclose(self.mfcc.deltas[1, :6],
                                 [-0.02286286, -0.11329014, 0.05381977,
                                  0.10438456, 0.04268386, -0.06839912]))
        self.assertTrue(allclose(self.mfcc.deltas[1, -6:],
                                 [-0.03156065, -0.019716, -0.03417692,
                                  -0.07768068, -0.05539324, -0.02616282]))
        # delta deltas
        self.assertTrue(allclose(self.mfcc.delta_deltas[1, :6],
                                 [-0.00804922, -0.009922, -0.00454391,
                                  0.0038989, 0.00254525, 0.0120557]))
        self.assertTrue(allclose(self.mfcc.delta_deltas[1, -6:],
                                 [0.0072148, 0.00094424, 0.00029913,
                                  0.00530994, 0.00184207, -0.00276511]))
