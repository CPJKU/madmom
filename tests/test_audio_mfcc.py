# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.audio.cepstrogram module.

"""

from __future__ import absolute_import, division, print_function

import unittest
from os.path import join as pj

from madmom.audio.cepstrogram import MFCC, Cepstrogram
from madmom.audio.filters import MelFilterbank
from madmom.audio.spectrogram import *
from . import AUDIO_PATH

sample_file = pj(AUDIO_PATH, 'sample.wav')
sample_file_22050 = pj(AUDIO_PATH, 'sample_22050.wav')


class TestMFCCClass(unittest.TestCase):
    def test_types(self):
        result = MFCC(sample_file)
        self.assertIsInstance(result, MFCC)
        self.assertIsInstance(result, Cepstrogram)
        # attributes
        self.assertIsInstance(result.filterbank, MelFilterbank)
        # properties
        self.assertIsInstance(result.deltas, np.ndarray)
        self.assertIsInstance(result.deltadeltas, np.ndarray)
        self.assertIsInstance(result.num_bins, int)
        self.assertIsInstance(result.num_frames, int)
        # wrong filterbank type
        with self.assertRaises(TypeError):
            FilteredSpectrogram(sample_file, filterbank='bla')

    def test_values(self):
        # from file
        result = MFCC(sample_file)
        self.assertTrue(np.allclose(result[0, :6],
                                    [-3.61102366, 6.81075716, 2.55457568,
                                     1.88377929, 1.04133379, 0.6382336]))
        self.assertTrue(np.allclose(result[0, -6:],
                                    [-0.20386486, -0.18468723, -0.00233107,
                                     0.20703268, 0.21419463, 0.00598407]))
        # attributes
        self.assertTrue(result.shape == (281, 30))

        # properties
        self.assertEqual(result.num_bins, 30)
        self.assertEqual(result.num_frames, 281)

    def test_deltas(self):
        # from file
        result = MFCC(sample_file)

        # don't compare first and last element because it is dependent on the
        # padding used for filtering
        self.assertTrue(np.allclose(result.deltas[0, 1:7],
                                    [-0.09853032, 0.05854281, 0.0971242,
                                     0.03878273, -0.07430606, -0.03955419]))
        self.assertTrue(np.allclose(result.deltas[0, -7:-1],
                                    [-0.0310398, -0.0324816, -0.02136466,
                                     -0.03697226, -0.07731059, -0.05689775]))

        self.assertTrue(np.allclose(result.deltadeltas[0, 1:7],
                                    [-0.00737991, -0.00236152, 0.00363018,
                                     0.00195056, 0.00295347, 0.0030891]))
        self.assertTrue(np.allclose(result.deltadeltas[0, -7:-1],
                                    [-0.00145328, 0.00046048, 0.00082433,
                                     0.00139767, -0.00018505, 0.00075226]))
