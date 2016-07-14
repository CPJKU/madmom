# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.audio.chroma module.

"""

from __future__ import absolute_import, division, print_function

import numpy as np
import unittest
from . import AUDIO_PATH, ACTIVATIONS_PATH
from madmom.audio.chroma import DeepChromaProcessor, CLPChroma
from madmom.features import Activations


sample_file = "%s/sample.wav" % AUDIO_PATH
sample_act_deep_chroma = Activations("%s/sample.deep_chroma.npz" %
                                     ACTIVATIONS_PATH)


class TestDeepChromaProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = DeepChromaProcessor()

    def test_process(self):
        chroma_act = self.processor(sample_file)
        self.assertTrue(np.allclose(chroma_act, sample_act_deep_chroma))


class TestCLPChromaClass(unittest.TestCase):

    def setUp(self):
        self.clp_50 = CLPChroma(sample_file, fps=50)
        self.clp_10 = CLPChroma(sample_file, fps=10)

    def test_process(self):
        # test with fps=50
        self.assertTrue(self.clp_50.bin_labels[0] == 'C')
        self.assertTrue(self.clp_50.fps == 50)
        # results
        self.assertTrue(self.clp_50.shape == (141, 12))
        self.assertTrue(np.allclose(self.clp_50[39, :],
                        np.array([0.2640026, 0.20764992, 0.28887013,
                                  0.3258862, 0.21502268, 0.24654484,
                                  0.1611352, 0.31651316, 0.4058146,
                                  0.3046548, 0.26724745, 0.36803654])))
        self.assertTrue(np.allclose(self.clp_50[100:111, 8],
                        np.array([0.63012157, 0.63953382, 0.64642419,
                                  0.63813059, 0.60423841, 0.56809763,
                                  0.50272806, 0.4193601, 0.3898431,
                                  0.41250395, 0.45836603])))
        # test with fps=10
        self.assertTrue(self.clp_10.bin_labels[0] == 'C')
        self.assertTrue(self.clp_10.fps == 10)
        # results
        self.assertTrue(self.clp_10.shape == (29, 12))
        self.assertTrue(np.allclose(self.clp_10[2:6, 7:9],
                        np.array([[0.23483211, 0.43080294],
                                  [0.2349911, 0.50033476],
                                  [0.2108299, 0.54075625],
                                  [0.21365665, 0.50447206]])))

    def test_compare_with_matlab_toolbox(self):
        # compare the results with the MATLAB chroma toolbox. There are
        # differences because of different resampling and filtering with
        # filtfilt, therefore we compare with higher tolerance
        # compare with MATLAB chroma toolbox
        self.assertTrue(np.allclose(self.clp_50[39, :],
                        np.array([0.28202948, 0.21473163, 0.29178235,
                                  0.31837119, 0.21773027, 0.24484771,
                                  0.16606759, 0.32054708, 0.39850856,
                                  0.30126012, 0.26116133, 0.36386101]),
                                    rtol=1e-01))
        self.assertTrue(np.allclose(self.clp_50[100:111, 8],
                        np.array([0.62898520, 0.63870508, 0.64272228,
                                  0.63746036, 0.60277398, 0.56819617,
                                  0.49709058, 0.40472238, 0.38589948,
                                  0.39892429, 0.43579584]),
                                    rtol=1e-01))
        self.assertTrue(np.allclose(self.clp_10[2:6, 7:9],
                        np.array([[0.22728066, 0.42691203],
                                  [0.23012684, 0.49625964],
                                  [0.20832211, 0.53284996],
                                  [0.21247771, 0.49591485]]), rtol=1e-01))
