# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.features.onsets module.

"""

from __future__ import absolute_import, division, print_function

import unittest
from os.path import join as pj

from . import AUDIO_PATH, ACTIVATIONS_PATH

from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.filters import LogarithmicFilterbank
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.spectrogram import (Spectrogram, SpectrogramProcessor,
                                      FilteredSpectrogramProcessor,
                                      LogarithmicFilteredSpectrogram,
                                      LogarithmicSpectrogramProcessor)
from madmom.features import Activations
from madmom.features.onsets import *

sample_file = pj(AUDIO_PATH, 'sample.wav')
sample_spec = Spectrogram(sample_file, circular_shift=True)
sample_log_filt_spec = LogarithmicFilteredSpectrogram(
    sample_spec, num_bands=24, mul=1, add=1)
sample_cnn_act = Activations(pj(ACTIVATIONS_PATH, 'sample.onsets_cnn.npz'))
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

    def test_processors(self):
        proc = SpectralOnsetProcessor()
        self.assertIsInstance(proc.processors[0], SignalProcessor)
        self.assertIsInstance(proc.processors[1], FramedSignalProcessor)
        self.assertIsInstance(proc.processors[2],
                              ShortTimeFourierTransformProcessor)
        self.assertIsInstance(proc.processors[3], SpectrogramProcessor)
        self.assertEqual(proc.processors[4], spectral_flux)

    def test_filterbank(self):
        # with filtering
        proc = SpectralOnsetProcessor(filterbank=LogarithmicFilterbank)
        self.assertIsInstance(proc.processors[4], FilteredSpectrogramProcessor)
        self.assertEqual(proc.processors[5], spectral_flux)

    def test_scaling(self):
        # with logarithmic scaling
        proc = SpectralOnsetProcessor(log=np.log10)
        self.assertIsInstance(proc.processors[4],
                              LogarithmicSpectrogramProcessor)
        self.assertEqual(proc.processors[5], spectral_flux)

    def test_filtered_scaling(self):
        # with both filtering and logarithmic scaling
        proc = SpectralOnsetProcessor(filterbank=LogarithmicFilterbank,
                                      log=np.log10)
        self.assertIsInstance(proc.processors[4], FilteredSpectrogramProcessor)
        self.assertIsInstance(proc.processors[5],
                              LogarithmicSpectrogramProcessor)
        self.assertEqual(proc.processors[6], spectral_flux)

    def test_circular_shift(self):
        # circular shift
        proc = SpectralOnsetProcessor(onset_method='phase_deviation')
        self.assertIsInstance(proc.processors[2],
                              ShortTimeFourierTransformProcessor)
        self.assertTrue(proc.processors[2].circular_shift)
        self.assertEqual(proc.processors[4], phase_deviation)

    def test_errors(self):
        with self.assertRaises(ValueError):
            SpectralOnsetProcessor(onset_method='nonexistent')

    def test_process(self):
        odf = self.processor(sample_file)
        self.assertTrue(np.allclose(odf[:6], [0., 100.90120697, 74.44419861,
                                              40.277565, 57.95736313,
                                              46.15561295]))


class TestRNNOnsetProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = RNNOnsetProcessor()
        self.online_processor = RNNOnsetProcessor(online=True, origin='online')

    def test_process(self):
        act = self.processor(sample_file)
        self.assertTrue(np.allclose(act, sample_brnn_act))
        act = self.online_processor(sample_file, reset=False)
        self.assertTrue(np.allclose(act, sample_rnn_act))


class TestCNNOnsetProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = CNNOnsetProcessor()

    def test_process(self):
        act = self.processor(sample_file)
        self.assertTrue(np.allclose(act, sample_cnn_act))


class TestPeakPickingFunction(unittest.TestCase):

    def test_values(self):
        onsets = peak_picking(sample_superflux_act, 1.1)
        self.assertTrue(np.allclose(onsets[:6], [2, 10, 17, 48, 55, 80]))
        self.assertTrue(len(onsets) == 35)
        # smooth
        onsets = peak_picking(sample_superflux_act, 1.1, smooth=3)
        self.assertTrue(np.allclose(onsets[:6], [2, 10, 17, 24, 48, 55]))
        # default values
        onsets = peak_picking(sample_superflux_act, 1.1, pre_max=2,
                              post_max=10, pre_avg=30)
        self.assertTrue(np.allclose(onsets[:6], [2, 17, 55, 89, 122, 159]))

    def test_online(self):
        onsets = peak_picking(sample_rnn_act, threshold=0.23, post_max=0)
        self.assertTrue(np.allclose(onsets,
                                    [1, 3, 10, 12, 29, 46, 62, 63, 77, 79,
                                     81, 99, 100, 113, 115, 148, 149, 164,
                                     181, 183, 216, 234, 250, 268]))
        self.assertTrue(len(onsets) == 24)


class TestOnsetPeakPickingProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = OnsetPeakPickingProcessor(
            threshold=1.1, pre_max=0.01, post_max=0.05, pre_avg=0.15,
            post_avg=0, combine=0.03, delay=0, fps=sample_superflux_act.fps)
        self.sample_superflux_result = [0.01, 0.085, 0.275, 0.445, 0.61, 0.795,
                                        0.98, 1.115, 1.365, 1.475, 1.62,
                                        1.795, 2.14, 2.33, 2.485, 2.665]
        self.online_processor = OnsetPeakPickingProcessor(
            threshold=0.23, online=True, fps=sample_rnn_act.fps)
        self.sample_rnn_result = [0.01, 0.1, 0.29, 0.46, 0.62, 0.77, 0.81,
                                  0.99, 1.13, 1.48, 1.64, 1.81, 2.16, 2.34,
                                  2.5, 2.68]

    def test_online_parameters(self):
        self.assertEqual(self.online_processor.smooth, 0)
        self.assertEqual(self.online_processor.post_avg, 0)
        self.assertEqual(self.online_processor.post_max, 0)

    def test_process(self):
        onsets = self.processor(sample_superflux_act)
        self.assertTrue(np.allclose(onsets, self.sample_superflux_result))

    def test_process_online(self):
        # process everything at once
        onsets = self.online_processor(sample_rnn_act)
        self.assertTrue(np.allclose(onsets, self.sample_rnn_result))
        # results must be the same if processed a second time
        onsets_1 = self.online_processor(sample_rnn_act)
        self.assertTrue(np.allclose(onsets_1, self.sample_rnn_result))
        # process frame by frame
        self.online_processor.reset()
        onsets_2 = np.hstack(
            [self.online_processor(np.atleast_1d(f), reset=False)
             for f in sample_rnn_act])
        self.assertTrue(np.allclose(onsets_2, self.sample_rnn_result))

    def test_delay(self):
        self.processor.delay = 1
        onsets = self.processor(sample_superflux_act)
        self.assertTrue(np.allclose(onsets - 1, self.sample_superflux_result))
