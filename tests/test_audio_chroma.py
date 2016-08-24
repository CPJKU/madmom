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
from os.path import join as pj
from madmom.audio.signal import Signal

sample_files = [pj(AUDIO_PATH, sf) for sf in ['sample.wav', 'sample2.wav']]
sample_acts = [Activations(pj(ACTIVATIONS_PATH, af))
               for af in ['sample.deep_chroma.npz', 'sample2.deep_chroma.npz']]
sample_file = pj(AUDIO_PATH, 'sample.wav')
sample_file_22050 = pj(AUDIO_PATH, 'sample_22050.wav')
sample_act_deep_chroma = Activations(pj(ACTIVATIONS_PATH,
                                        'sample.deep_chroma.npz'))


class TestDeepChromaProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = DeepChromaProcessor()

    def test_process(self):
        for f, a in zip(sample_files, sample_acts):
            self.assertTrue(np.allclose(self.processor(f), a))


class TestCLPChromaClass(unittest.TestCase):

    def setUp(self):
        self.clp_50 = CLPChroma(sample_file, fps=50)
        self.clp_10 = CLPChroma(sample_file, fps=10)
        self.clp_22050 = CLPChroma(sample_file_22050, fps=50, fmin=2637,
                                   fmax=4200)
        data = Signal(sample_file)
        self.clp_10_from_signal = CLPChroma(data, fps=10)

    def test_process(self):
        # test with fps=50
        self.assertTrue(self.clp_50.bin_labels[0] == 'C')
        self.assertTrue(self.clp_50.fps == 50)
        # results
        self.assertTrue(self.clp_50.shape == (141, 12))
        tar = [0.28222724, 0.2145749, 0.29143909, 0.31838085, 0.21754939,
               0.24475572, 0.16546808, 0.32018109, 0.39918812, 0.30166908,
               0.26142349, 0.3635601]
        self.assertTrue(np.allclose(self.clp_50[39, :], tar, atol=1e-4))
        tar = [0.62827758, 0.63810707, 0.64559874, 0.63725388, 0.60231739,
               0.56549827, 0.49675867, 0.40509999, 0.38589308, 0.39961286,
               0.43776578]
        self.assertTrue(np.allclose(self.clp_50[100:111, 8], tar, atol=1e-5))
        # test with fps=10
        self.assertTrue(self.clp_10.bin_labels[0] == 'C')
        self.assertTrue(self.clp_10.fps == 10)
        # results
        self.assertTrue(self.clp_10.shape == (29, 12))
        tar = [[0.23144638, 0.42642003], [0.23364208, 0.49532055],
               [0.2099782, 0.53246478], [0.2120323, 0.49525887]]
        self.assertTrue(np.allclose(self.clp_10[2:6, 7:9], tar, atol=1e-4))
        # test clp from signal
        self.assertTrue(self.clp_10_from_signal.shape == (29, 12))
        self.assertTrue(np.allclose(self.clp_10_from_signal[2:6, 7:9],
                                    tar, atol=1e-4))
        # test 22050 Hz sampling rate. If we use only bands above 2637 Hz,
        # no resampling is necessary and we can therefore compare with
        # smaller tolerances.
        self.assertTrue(self.clp_22050.shape == (141, 12))
        tar = [[0.11270745, 0, 0, 0, 0.25741291, 0.58624929, 0.43997279,
                0.0999583, 0.21696206, 0.54994475, 0.05542545, 0.14558826]]
        self.assertTrue(np.allclose(self.clp_22050[140, :], tar))

    def test_compare_with_matlab_toolbox(self):
        # compare the results with the MATLAB chroma toolbox. There are
        # differences because of different resampling and filtering with
        # filtfilt, therefore we compare with higher tolerances
        tar = np.array([0.28202948, 0.21473163, 0.29178235, 0.31837119,
                        0.21773027, 0.24484771, 0.16606759, 0.32054708,
                        0.39850856, 0.30126012, 0.26116133, 0.36386101])
        self.assertTrue(np.allclose(self.clp_50[39, :], tar, rtol=1e-02))
        tar = np.array([0.62898520, 0.63870508, 0.64272228, 0.63746036,
                        0.60277398, 0.56819617, 0.49709058, 0.40472238,
                        0.38589948, 0.39892429, 0.43579584])
        self.assertTrue(np.allclose(self.clp_50[100:111, 8], tar, rtol=1e-02))
        tar = np.array([[0.22728066, 0.42691203], [0.23012684, 0.49625964],
                        [0.20832211, 0.53284996], [0.21247771, 0.49591485]])
        self.assertTrue(np.allclose(self.clp_10[2:6, 7:9], tar, rtol=2e-02))
