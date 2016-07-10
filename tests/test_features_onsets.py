# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.features.onsets module.

"""

from __future__ import absolute_import, division, print_function

import unittest
from os.path import join as pj

from . import AUDIO_PATH, ACTIVATIONS_PATH

from madmom.audio.spectrogram import (Spectrogram,
                                      LogarithmicFilteredSpectrogram)
from madmom.features import Activations
from madmom.features.onsets import *

sample_file = pj(AUDIO_PATH, 'sample.wav')
sample_spec = Spectrogram(sample_file, circular_shift=True)
sample_log_filt_spec = LogarithmicFilteredSpectrogram(
    sample_spec, num_bands=24, mul=1, add=1)
sample_rnn_act = Activations(pj(ACTIVATIONS_PATH, 'sample.onsets_rnn.npz'))
sample_brnn_act = Activations(pj(ACTIVATIONS_PATH, 'sample.onsets_brnn.npz'))
sample_superflux_act = Activations(pj(ACTIVATIONS_PATH,
                                      'sample.super_flux.npz'))


class TestHighFrequencyContentFunction(unittest.TestCase):

    def test_values(self):
        odf = high_frequency_content(sample_log_filt_spec)
        self.assertTrue(np.allclose(odf[:6], [8.97001563, 9.36399107,
                                              8.64144536, 8.34977449,
                                              8.21097918, 8.40412515]))


class TestFunction(unittest.TestCase):

    def test_values(self):
        odf = high_frequency_content(sample_log_filt_spec)
        self.assertTrue(np.allclose(odf[:6], [8.97001563, 9.36399107,
                                              8.64144536, 8.34977449,
                                              8.21097918, 8.40412515]))


class TestSpectralDiffFunction(unittest.TestCase):

    def test_values(self):
        odf = spectral_diff(sample_log_filt_spec)
        self.assertTrue(np.allclose(odf[:6], [0, 0.55715936, 0.64004618,
                                              0.0810971, 0.295396,
                                              0.16324584]))


class TestSpectralFluxFunction(unittest.TestCase):

    def test_values(self):
        odf = spectral_flux(sample_log_filt_spec)
        self.assertTrue(np.allclose(odf[:6], [0, 3.91207361, 2.91675663,
                                              1.38361311, 2.59582925,
                                              2.16986609]))


class TestSuperfluxFunction(unittest.TestCase):

    def test_values(self):
        odf = superflux(sample_log_filt_spec)
        self.assertTrue(np.allclose(odf[:6], [0, 2.08680153, 0.6411702,
                                              0.38634294, 0.40202433,
                                              0.63349575]))


class TestComplexFluxFunction(unittest.TestCase):

    def test_values(self):
        odf = complex_flux(sample_log_filt_spec)
        self.assertTrue(np.allclose(odf[:6], [0, 0.476213485, 0.0877621323,
                                              0.0593151376, 0.0654867291,
                                              0.0954693183]))


class TestModifiedKullbackLeiblerFunction(unittest.TestCase):

    def test_values(self):
        odf = modified_kullback_leibler(sample_log_filt_spec)
        self.assertTrue(np.allclose(odf[:6], [0, 0.71910584, 0.6664055,
                                              0.68092251, 0.69984031,
                                              0.71744561]))


class TestPhaseDeviationFunction(unittest.TestCase):

    def test_values(self):
        odf = phase_deviation(sample_log_filt_spec)
        self.assertTrue(np.allclose(odf[:6], [0, 0, 0.71957183, 0.91994524,
                                              0.9418999, 0.86083585]))


class TestWeightedPhaseDeviationFunction(unittest.TestCase):

    def test_values(self):
        odf = weighted_phase_deviation(sample_spec)
        self.assertTrue(np.allclose(odf[:6], [0, 0, 0.19568817, 0.20483065,
                                              0.17890805, 0.16970603]))

    def test_errors(self):
        with self.assertRaises(ValueError):
            weighted_phase_deviation(sample_log_filt_spec)


class TestNormalizesWeightedPhaseDeviationFunction(unittest.TestCase):

    def test_values(self):
        odf = normalized_weighted_phase_deviation(sample_spec)
        self.assertTrue(np.allclose(odf[:6], [0, 0, 0.46018526, 0.50193471,
                                              0.42031503, 0.40806249]))

    def test_errors(self):
        with self.assertRaises(ValueError):
            normalized_weighted_phase_deviation(sample_log_filt_spec)


class TestComplexDomainFunction(unittest.TestCase):

    def test_values(self):
        odf = complex_domain(sample_spec)
        self.assertTrue(np.allclose(odf[:6], [399.29980469, 585.9564209,
                                              262.08010864, 225.84718323,
                                              196.88954163, 200.32469177]))

    def test_errors(self):
        with self.assertRaises(ValueError):
            complex_domain(sample_log_filt_spec)


class TestRectifiedComplexDomainFunction(unittest.TestCase):

    def test_values(self):
        odf = rectified_complex_domain(sample_spec)
        self.assertTrue(np.allclose(odf[:6], [0, 394.165222, 119.79425,
                                              96.70564, 122.52311, 92.61698]))

    def test_errors(self):
        with self.assertRaises(ValueError):
            rectified_complex_domain(sample_log_filt_spec)


class TestSpectralOnsetProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = SpectralOnsetProcessor()

    def test_process(self):
        odf = self.processor(sample_log_filt_spec)
        self.assertTrue(np.allclose(odf[:6], [0, 2.0868, 0.64117,
                                              0.386343, 0.402024, 0.6335]))


class TestRNNOnsetProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = RNNOnsetProcessor()
        self.online_processor = RNNOnsetProcessor(online=True)

    def test_process(self):
        act = self.processor(sample_file)
        self.assertTrue(np.allclose(act, sample_brnn_act))
        act = self.online_processor(sample_file)
        self.assertTrue(np.allclose(act, sample_rnn_act))


class TestPeakPickingFunction(unittest.TestCase):

    def test_values(self):
        onsets = peak_picking(sample_superflux_act, 1.1)
        self.assertTrue(np.allclose(onsets[:6], [2, 10, 17, 48, 55, 80]))
        # smooth
        onsets = peak_picking(sample_superflux_act, 1.1, smooth=3)
        self.assertTrue(np.allclose(onsets[:6], [2, 10, 17, 24, 48, 55]))
        # default values
        onsets = peak_picking(sample_superflux_act, 1.1, pre_max=2,
                              post_max=10, pre_avg=30)
        self.assertTrue(np.allclose(onsets[:6], [2, 17, 55, 89, 122, 159]))


class TestPeakPickingProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = PeakPickingProcessor(threshold=1.1, pre_max=0.01,
                                              post_max=0.05, pre_avg=0.15,
                                              post_avg=0, combine=0.03,
                                              delay=0,
                                              fps=sample_superflux_act.fps)

    def test_process(self):
        onsets = self.processor(sample_superflux_act)
        self.assertTrue(np.allclose(onsets, [0.01, 0.085, 0.275, 0.445, 0.61,
                                             0.795, 0.98, 1.115, 1.365, 1.475,
                                             1.62, 1.795, 2.14, 2.33, 2.485,
                                             2.665]))
