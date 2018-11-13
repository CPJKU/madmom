# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.audio.spectrogram module.

"""

from __future__ import absolute_import, division, print_function

import unittest
from os.path import join as pj

from . import AUDIO_PATH
from .test_audio_filters import FFT_FREQS_1024, LOG_FILTERBANK_CENTER_FREQS

from madmom.audio.spectrogram import *
from madmom.audio.filters import (Filterbank, LogarithmicFilterbank,
                                  MelFilterbank, BarkFilterbank)
from madmom.audio.stft import ShortTimeFourierTransform
from madmom.audio.signal import Signal

sample_file = pj(AUDIO_PATH, 'sample.wav')
sample_file_22050 = pj(AUDIO_PATH, 'sample_22050.wav')
sample_spec = Spectrogram(sample_file)


# test functions
class TestDftFunction(unittest.TestCase):

    def test_types(self):
        self.assertTrue(True)

    def test_values(self):
        self.assertTrue(True)


class TestStftFunction(unittest.TestCase):

    def test_types(self):
        self.assertTrue(True)

    def test_value(self):
        self.assertTrue(True)


class TestSpecFunction(unittest.TestCase):

    def test_types(self):
        result = spec(np.random.rand(10))
        self.assertTrue(result.dtype == np.float)
        self.assertTrue(result.shape == (10, ))
        result = spec(np.random.rand(10, 2))
        self.assertTrue(result.dtype == np.float)
        self.assertTrue(result.shape == (10, 2))
        # complex data
        data = np.random.rand(10) + 1j * np.random.rand(10)
        result = spec(data)
        self.assertTrue(result.dtype == np.float)
        self.assertTrue(result.shape == (10, ))
        data = np.random.rand(10, 2) + 1j * np.random.rand(10, 2)
        result = spec(data)
        self.assertTrue(result.dtype == np.float)
        self.assertTrue(result.shape == (10, 2))

    def test_values(self):
        data = np.random.rand(10) + 1j * np.random.rand(10)
        self.assertTrue(np.allclose(np.abs(data), spec(data)))
        data = np.random.rand(10, 2) + 1j * np.random.rand(10, 2)
        self.assertTrue(np.allclose(np.abs(data), spec(data)))


class TestSpectrogramClass(unittest.TestCase):

    def test_types(self):
        result = Spectrogram(sample_file)
        self.assertIsInstance(result, Spectrogram)
        # attributes
        self.assertIsInstance(result.stft, ShortTimeFourierTransform)
        self.assertIsInstance(result.bin_frequencies, np.ndarray)
        # properties
        self.assertIsInstance(result.num_bins, int)
        self.assertIsInstance(result.num_frames, int)

    def test_values(self):
        # from file
        result = Spectrogram(sample_file)
        self.assertTrue(np.allclose(result[0, :8],
                                    [3.15249, 4.00272, 5.66156, 6.30141,
                                     6.02199, 10.84909, 17.83130, 19.44511]))
        self.assertTrue(np.allclose(result[0, -8:],
                                    [0.0365325, 0.036513, 0.0364213, 0.0366203,
                                     0.036737, 0.036423, 0.036335, 0.0367054]))
        # attributes
        self.assertTrue(result.shape == (281, 1024))
        self.assertTrue(np.allclose(result.bin_frequencies, FFT_FREQS_1024))
        # properties
        self.assertTrue(result.num_frames == 281)
        self.assertTrue(result.num_bins == 1024)
        # from spec
        self.assertTrue(np.allclose(Spectrogram(result), result))
        # from stft
        stft = ShortTimeFourierTransform(sample_file)
        self.assertTrue(np.allclose(Spectrogram(stft), result))

    def test_methods(self):
        result = Spectrogram(sample_file)
        self.assertIsInstance(result.diff(), SpectrogramDifference)
        self.assertIsInstance(result.filter(), FilteredSpectrogram)
        self.assertIsInstance(result.log(), LogarithmicSpectrogram)


class TestSpectrogramProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = SpectrogramProcessor()

    def test_types(self):
        self.assertIsInstance(self.processor, SpectrogramProcessor)

    def test_process(self):
        result = self.processor.process(sample_file)
        self.assertIsInstance(result, Spectrogram)
        # attributes
        self.assertTrue(result.shape == (281, 1024))
        self.assertTrue(np.allclose(result.bin_frequencies, FFT_FREQS_1024))
        # properties
        self.assertTrue(result.num_frames == 281)
        self.assertTrue(result.num_bins == 1024)


class TestFilteredSpectrogramClass(unittest.TestCase):

    def test_types(self):
        result = FilteredSpectrogram(sample_file)
        self.assertIsInstance(result, FilteredSpectrogram)
        self.assertIsInstance(result, Spectrogram)
        # attributes
        self.assertIsInstance(result.stft, ShortTimeFourierTransform)
        self.assertIsInstance(result.filterbank, LogarithmicFilterbank)
        self.assertIsInstance(result.bin_frequencies, np.ndarray)
        # properties
        self.assertIsInstance(result.num_bins, int)
        self.assertIsInstance(result.num_frames, int)
        # wrong filterbank type
        with self.assertRaises(TypeError):
            FilteredSpectrogram(sample_file, filterbank='bla')

    def test_values(self):
        # from file
        result = FilteredSpectrogram(sample_file)
        self.assertTrue(np.allclose(result[0, :8],
                                    [5.661564, 6.30141, 6.02199, 10.84909,
                                     17.8313, 19.44511, 17.56456, 21.859523]))
        self.assertTrue(np.allclose(result[0, -8:],
                                    [0.123125, 0.119462, 0.137849, 0.1269156,
                                     0.110888, 0.083526, 0.05426, 0.064614]))
        # attributes
        self.assertTrue(result.shape == (281, 81))
        self.assertTrue(np.allclose(result.bin_frequencies,
                                    LOG_FILTERBANK_CENTER_FREQS))
        # properties
        self.assertTrue(result.num_bins == 81)
        self.assertTrue(result.num_frames == 281)
        # with given filterbank
        result = FilteredSpectrogram(sample_file,
                                     filterbank=result.filterbank)
        # attributes
        self.assertTrue(result.shape == (281, 81))
        self.assertTrue(np.allclose(result.bin_frequencies,
                                    LOG_FILTERBANK_CENTER_FREQS))
        # properties
        self.assertTrue(result.num_bins == 81)
        self.assertTrue(result.num_frames == 281)

    def test_filterbanks(self):
        # with Mel filterbank
        result = FilteredSpectrogram(sample_file,
                                     filterbank=MelFilterbank, num_bands=40)
        self.assertTrue(np.allclose(result[0, :6],
                                    [8.42887115, 17.98174477, 19.50165367,
                                     6.48194313, 2.96991181, 4.06280804]))
        self.assertTrue(result.shape == (281, 40))
        # with Bark filterbank
        result = FilteredSpectrogram(sample_file,
                                     filterbank=BarkFilterbank,
                                     num_bands='normal')
        self.assertTrue(np.allclose(result[0, :6],
                                    [16.42251968, 17.36715126, 2.81979132,
                                     4.27050114, 3.08699131, 1.50553513]))
        self.assertTrue(result.shape == (281, 23))

    def test_from_spec(self):
        spec = Spectrogram(AUDIO_PATH + '/sample.wav')
        result = FilteredSpectrogram(spec)
        # same results as above
        self.assertTrue(np.allclose(result[0, :8],
                                    [5.661564, 6.30141, 6.02199, 10.84909,
                                     17.8313, 19.44511, 17.56456, 21.859523]))
        # spec must not be altered
        self.assertTrue(np.allclose(spec[0, :8],
                                    [3.15249, 4.00272, 5.66156, 6.30141,
                                     6.02199, 10.84909, 17.83130, 19.44511]))

    def test_methods(self):
        result = FilteredSpectrogram(sample_file)
        self.assertIsInstance(result.diff(), SpectrogramDifference)
        # TODO: should we return a LogarithmicFilteredSpectrogram?
        self.assertIsInstance(result.log(), LogarithmicSpectrogram)


class TestFilteredSpectrogramProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = FilteredSpectrogramProcessor()

    def test_types(self):
        self.assertIsInstance(self.processor, FilteredSpectrogramProcessor)
        self.assertTrue(issubclass(self.processor.filterbank,
                                   LogarithmicFilterbank))
        self.assertIsInstance(self.processor.num_bands, int)
        self.assertIsInstance(self.processor.fmin, float)
        self.assertIsInstance(self.processor.fmax, float)
        self.assertIsInstance(self.processor.fref, float)
        self.assertIsInstance(self.processor.norm_filters, bool)
        self.assertIsInstance(self.processor.unique_filters, bool)

    def test_values(self):
        self.assertTrue(issubclass(self.processor.filterbank,
                                   LogarithmicFilterbank))
        self.assertTrue(self.processor.num_bands == 12)
        self.assertTrue(self.processor.fmin == 30)
        self.assertTrue(self.processor.fmax == 17000)
        self.assertTrue(self.processor.fref == 440)
        self.assertTrue(self.processor.norm_filters is True)
        self.assertTrue(self.processor.unique_filters is True)

    def test_process(self):
        # default values
        result = self.processor.process(sample_file)
        self.assertIsInstance(result, FilteredSpectrogram)
        # attributes
        self.assertTrue(result.shape == (281, 81))
        self.assertTrue(np.allclose(result.bin_frequencies,
                                    LOG_FILTERBANK_CENTER_FREQS))
        # properties
        self.assertTrue(result.num_bins == 81)
        self.assertTrue(result.num_frames == 281)

    def test_other_values(self):
        processor = FilteredSpectrogramProcessor(num_bands=6, fmin=300,
                                                 fmax=10000)
        result = processor.process(sample_file)
        self.assertIsInstance(result, FilteredSpectrogram)
        # attributes
        self.assertTrue(result.shape == (281, 29))
        self.assertTrue(np.allclose(result.bin_frequencies,
                                    [344.53125, 387.5976562, 430.6640625,
                                     495.2636718, 559.86328125, 624.4628906,
                                     689.0625, 775.1953125, 882.8613281,
                                     990.52734375, 1098.1933593, 1248.9257812,
                                     1399.6582031, 1571.9238281, 1765.7226562,
                                     1981.0546875, 2217.9199218, 2497.8515625,
                                     2799.3164062, 3143.84765625, 3509.912109,
                                     3940.5761718, 4435.8398437, 4974.1699218,
                                     5577.09960938, 6266.1621093, 7041.3574218,
                                     7902.6855468, 8871.6796875]))
        # properties
        self.assertTrue(result.num_bins == 29)
        self.assertTrue(result.num_frames == 281)


class TestLogarithmicSpectrogramClass(unittest.TestCase):

    def test_types(self):
        result = LogarithmicSpectrogram(sample_file)
        self.assertIsInstance(result, LogarithmicSpectrogram)
        self.assertIsInstance(result, Spectrogram)
        # attributes
        self.assertIsInstance(result.stft, ShortTimeFourierTransform)
        self.assertIsInstance(result.bin_frequencies, np.ndarray)
        self.assertIsInstance(result.mul, float)
        self.assertIsInstance(result.add, float)
        # properties
        self.assertIsInstance(result.num_frames, int)
        self.assertIsInstance(result.num_bins, int)

    def test_values(self):
        result = LogarithmicSpectrogram(sample_file)
        self.assertTrue(np.allclose(result[0, :8],
                                    [0.618309, 0.699206, 0.823576, 0.86341,
                                     0.84646, 1.073685, 1.27488, 1.310589]))
        self.assertTrue(np.allclose(result[0, -8:],
                                    [0.015583, 0.0155747, 0.0155363, 0.0156197,
                                     0.0156684, 0.015537, 0.0155003,
                                     0.01565535]))
        # attributes
        self.assertTrue(result.shape == (281, 1024))
        self.assertTrue(np.allclose(result.bin_frequencies,
                                    FFT_FREQS_1024))
        self.assertTrue(result.mul == 1)
        self.assertTrue(result.add == 1)
        # properties
        self.assertTrue(result.num_frames == 281)
        self.assertTrue(result.num_bins == 1024)
        # test other values
        result = LogarithmicSpectrogram(sample_file,
                                        mul=2, add=2)
        self.assertTrue(result.mul == 2)
        self.assertTrue(result.add == 2)

    def test_from_spec(self):
        spec = Spectrogram(AUDIO_PATH + '/sample.wav')
        result = LogarithmicSpectrogram(spec)
        # same results as above
        self.assertTrue(np.allclose(result[0, :8],
                                    [0.618309, 0.699206, 0.823576, 0.86341,
                                     0.84646, 1.073685, 1.27488, 1.310589]))
        # spec must not be altered
        self.assertTrue(np.allclose(spec[0, :8],
                                    [3.15249, 4.00272, 5.66156, 6.30141,
                                     6.02199, 10.84909, 17.83130, 19.44511]))

    def test_methods(self):
        result = LogarithmicSpectrogram(sample_file)
        self.assertIsInstance(result.diff(), SpectrogramDifference)
        self.assertIsInstance(result.filter(), FilteredSpectrogram)


class TestLogarithmicSpectrogramProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = LogarithmicSpectrogramProcessor()

    def test_types(self):
        self.assertIsInstance(self.processor, LogarithmicSpectrogramProcessor)
        self.assertIsInstance(self.processor.mul, float)
        self.assertIsInstance(self.processor.add, float)

    def test_values(self):
        self.assertTrue(self.processor.mul == 1)
        self.assertTrue(self.processor.add == 1)

    def test_process(self):
        result = self.processor.process(sample_file)
        self.assertIsInstance(result, LogarithmicSpectrogram)
        self.assertTrue(result.shape == (281, 1024))


class TestLogarithmicFilteredSpectrogramClass(unittest.TestCase):

    def test_types(self):
        result = LogarithmicFilteredSpectrogram(sample_file)
        self.assertIsInstance(result, LogarithmicFilteredSpectrogram)
        self.assertIsInstance(result, Spectrogram)
        # attributes
        self.assertIsInstance(result.stft, ShortTimeFourierTransform)
        self.assertIsInstance(result.filterbank, Filterbank)
        self.assertIsInstance(result.filterbank, LogarithmicFilterbank)
        self.assertIsInstance(result.bin_frequencies, np.ndarray)
        self.assertIsInstance(result.mul, float)
        self.assertIsInstance(result.add, float)
        # properties
        self.assertIsInstance(result.num_frames, int)
        self.assertIsInstance(result.num_bins, int)

    def test_values(self):
        result = LogarithmicFilteredSpectrogram(sample_file)
        self.assertTrue(np.allclose(result[0, :8],
                                    [0.8235762, 0.863407, 0.8464602, 1.073685,
                                     1.27488, 1.3105896, 1.2686847, 1.359067]))
        self.assertTrue(np.allclose(result[0, -8:],
                                    [0.05042794, 0.0490095, 0.05608485,
                                     0.05189138, 0.04567042, 0.03483925,
                                     0.02294769, 0.02719229]))
        # attributes
        self.assertTrue(result.shape == (281, 81))
        self.assertTrue(result.mul == 1)
        self.assertTrue(result.add == 1)
        self.assertTrue(np.allclose(result.bin_frequencies,
                                    LOG_FILTERBANK_CENTER_FREQS))
        # properties
        self.assertTrue(result.num_frames == 281)
        self.assertTrue(result.num_bins == 81)
        # test other values
        result = LogarithmicFilteredSpectrogram(sample_file, mul=2, add=2)
        self.assertTrue(result.mul == 2)
        self.assertTrue(result.add == 2)

    def test_from_spec(self):
        spec = Spectrogram(AUDIO_PATH + '/sample.wav')
        result = LogarithmicFilteredSpectrogram(spec)
        # same results as above
        self.assertTrue(result.shape == (281, 81))
        self.assertTrue(np.allclose(result[0, :8],
                                    [0.8235762, 0.863407, 0.8464602, 1.073685,
                                     1.27488, 1.3105896, 1.2686847, 1.359067]))
        # spec must not be altered
        self.assertTrue(spec.shape == (281, 1024))
        self.assertTrue(np.allclose(spec[0, :8],
                                    [3.15249, 4.00272, 5.66156, 6.30141,
                                     6.02199, 10.84909, 17.83130, 19.44511]))


class TestLogarithmicFilteredSpectrogramProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = LogarithmicFilteredSpectrogramProcessor()

    def test_types(self):

        self.assertIsInstance(self.processor,
                              LogarithmicFilteredSpectrogramProcessor)
        self.assertTrue(self.processor.filterbank == LogarithmicFilterbank)
        self.assertIsInstance(self.processor.num_bands, int)
        self.assertIsInstance(self.processor.fmin, float)
        self.assertIsInstance(self.processor.fmax, float)
        self.assertIsInstance(self.processor.fref, float)
        self.assertIsInstance(self.processor.norm_filters, bool)
        self.assertIsInstance(self.processor.unique_filters, bool)
        self.assertIsInstance(self.processor.mul, float)
        self.assertIsInstance(self.processor.add, float)

    def test_values(self):
        # filter stuff
        self.assertTrue(issubclass(self.processor.filterbank,
                                   LogarithmicFilterbank))
        self.assertTrue(self.processor.num_bands == 12)
        self.assertTrue(self.processor.fmin == 30)
        self.assertTrue(self.processor.fmax == 17000)
        self.assertTrue(self.processor.fref == 440)
        self.assertTrue(self.processor.norm_filters is True)
        self.assertTrue(self.processor.unique_filters is True)
        # log stuff
        self.assertTrue(self.processor.mul == 1)
        self.assertTrue(self.processor.add == 1)

    def test_process(self):
        result = self.processor.process(sample_file)
        self.assertIsInstance(result, LogarithmicFilteredSpectrogram)

        self.assertTrue(result.shape == (281, 81))


class TestSpectrogramDifferenceClass(unittest.TestCase):

    def test_types(self):
        result = SpectrogramDifference(sample_file)
        self.assertIsInstance(result, SpectrogramDifference)
        self.assertIsInstance(result, Spectrogram)
        # attributes
        self.assertIsInstance(result.spectrogram, Spectrogram)
        self.assertIsInstance(result.bin_frequencies, np.ndarray)
        self.assertIsInstance(result.diff_ratio, float)
        self.assertIsInstance(result.diff_frames, int)
        self.assertTrue(result.diff_max_bins is None)
        self.assertIsInstance(result.positive_diffs, bool)
        # properties
        self.assertIsInstance(result.num_frames, int)
        self.assertIsInstance(result.num_bins, int)

    def test_values(self):
        result = SpectrogramDifference(sample_file)
        self.assertTrue(np.allclose(result[1, :8],
                                    [1.13179708, -1.1511457, 2.7810955,
                                     2.39441729, -4.87367058, -0.90269375,
                                     3.48209763, 11.14723015]))
        self.assertTrue(np.allclose(result[1, -8:],
                                    [-0.01463442, -0.01408007, -0.01462659,
                                     -0.01431422, -0.01404046, -0.01457103,
                                     -0.01443923, -0.01443416]))
        # attributes
        self.assertTrue(result.shape == (281, 1024))
        self.assertTrue(np.allclose(result.bin_frequencies, FFT_FREQS_1024))
        self.assertTrue(result.diff_ratio == 0.5)
        self.assertTrue(result.diff_frames == 1)
        self.assertTrue(result.diff_max_bins is None)
        self.assertTrue(result.positive_diffs is False)
        # properties
        self.assertTrue(result.num_bins == 1024)
        self.assertTrue(result.num_frames == 281)
        # methods
        self.assertTrue(result.positive_diff().min() == 0)


class TestSpectrogramDifferenceProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = SpectrogramDifferenceProcessor()

    def test_types(self):
        self.assertIsInstance(self.processor, SpectrogramDifferenceProcessor)
        self.assertIsInstance(self.processor.diff_ratio, float)
        self.assertTrue(self.processor.diff_frames is None)
        self.assertTrue(self.processor.diff_max_bins is None)
        self.assertIsInstance(self.processor.positive_diffs, bool)
        self.assertTrue(self.processor.stack_diffs is None)

    def test_values(self):
        self.assertTrue(self.processor.diff_ratio == 0.5)
        self.assertTrue(self.processor.diff_frames is None)
        self.assertTrue(self.processor.diff_max_bins is None)
        self.assertTrue(self.processor.positive_diffs is False)

    def test_process(self):
        result = self.processor.process(sample_spec)
        self.assertTrue(result.shape == (281, 1024))
        self.assertTrue(np.sum(result[:1]) == 0)
        self.assertTrue(np.max(result[:2]) > 0)
        self.assertTrue(np.min(result) < 0)
        self.assertTrue(np.allclose(result[0], 0))
        self.assertTrue(np.allclose(result[1:], np.diff(sample_spec, axis=0)))
        # if called a second time, result must be the exact same
        result_1 = self.processor.process(sample_spec)
        self.assertTrue(np.allclose(result, result_1))
        # result must be the same if processed frame-by-frame
        self.processor.reset()
        result_2 = np.vstack([self.processor.process(np.atleast_2d(frame),
                                                     reset=False)
                              for frame in sample_spec])
        self.assertTrue(np.allclose(result_2, result))
        # result must be different without resetting (first frame != 0)
        result_3 = np.vstack([self.processor.process(np.atleast_2d(frame),
                                                     reset=False)
                              for frame in sample_spec])
        self.assertFalse(np.allclose(result_3, result))
        self.assertFalse(np.sum(result_3[0]) == 0)

    def test_diff_frames(self):
        # re-initialise the processor, because of the buffer
        self.processor = SpectrogramDifferenceProcessor(diff_frames=2)
        self.assertTrue(self.processor.diff_frames == 2)
        result = self.processor.process(sample_spec)
        self.assertTrue(result.shape == (281, 1024))
        self.assertTrue(np.sum(result[:2]) == 0)
        self.assertTrue(np.min(result) < 0)
        self.assertTrue(result.diff_frames == 2)
        # if called a second time, result must be the exact same
        result_1 = self.processor.process(sample_spec)
        self.assertTrue(np.allclose(result, result_1))
        # result must be the same if processed frame-by-frame
        self.processor.reset()
        self.assertTrue(self.processor.diff_frames == 2)
        result_2 = np.vstack([self.processor.process(np.atleast_2d(frame),
                                                     reset=False)
                              for frame in sample_spec])
        self.assertTrue(np.allclose(result_2, result))
        # result must be different without resetting (first 2 frame != 0)
        result_3 = np.vstack([self.processor.process(np.atleast_2d(frame),
                                                     reset=False)
                              for frame in sample_spec])
        self.assertFalse(np.allclose(result_3, result))
        self.assertFalse(np.sum(result_3[0]) == 0)
        self.assertFalse(np.sum(result_3[1]) == 0)

    def test_positive_diffs(self):
        # re-initialise the processor, because of the buffer
        self.processor = SpectrogramDifferenceProcessor(diff_frames=2)
        self.processor.positive_diffs = True
        result = self.processor.process(sample_spec)
        self.assertTrue(result.shape == (281, 1024))
        self.assertTrue(np.sum(result[:2]) == 0)
        self.assertTrue(np.min(result) >= 0)
        # change stacking
        self.processor.stack_diffs = np.hstack
        result = self.processor.process(sample_spec)
        self.assertTrue(result.shape == (281, 1024 * 2))
        self.assertTrue(np.min(result) >= 0)
        self.processor.stack_diffs = np.vstack
        result = self.processor.process(sample_spec)
        self.assertTrue(result.shape == (281 * 2, 1024))
        self.assertTrue(np.min(result) >= 0)
        self.processor.stack_diffs = np.dstack
        result = self.processor.process(sample_spec)
        self.assertTrue(result.shape == (281, 1024, 2))
        self.assertTrue(np.min(result) >= 0)


class TestSuperFluxProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = SuperFluxProcessor()

    def test_types(self):
        self.assertIsInstance(self.processor, SuperFluxProcessor)

    def test_values(self):
        result = self.processor.process(sample_file)
        self.assertTrue(np.allclose(result[1, :8],
                                    [0.11168772, 0.12317812, 0, 0, 0.03797626,
                                     0.18899226, 0, 0.0903399]))
        self.assertTrue(np.allclose(result[1, -8:],
                                    [0, 0, 0.01419619, 0, 0.02666602,
                                     0.04325962, 0.10899737, 0.06546581]))
        self.assertIsInstance(result, SpectrogramDifference)
        self.assertTrue(result.num_bins == 140)
        self.assertTrue(result.num_frames == 281)
        self.assertTrue(result.shape == (281, 140))
        # diff stuff
        self.assertTrue(result.diff_ratio == 0.5)
        self.assertTrue(result.diff_max_bins == 3)
        self.assertTrue(result.positive_diffs is True)


class TestMultiBandSpectrogramClass(unittest.TestCase):

    def test_types(self):
        result = MultiBandSpectrogram(sample_file, [200, 1000])
        self.assertIsInstance(result, MultiBandSpectrogram)
        # attributes
        self.assertIsInstance(result.bin_frequencies, np.ndarray)
        self.assertIsInstance(result.crossover_frequencies, list)
        # properties
        self.assertIsInstance(result.num_bins, int)
        self.assertIsInstance(result.num_frames, int)

    def test_values(self):
        result = MultiBandSpectrogram(sample_file, [200, 1000])
        self.assertTrue(np.allclose(result[:3],
                                    [[10.95971966, 4.23556566, 0.19092605],
                                     [11.38149452, 4.88609695, 0.21491699],
                                     [13.50860405, 4.48350096, 0.20132662]]))
        self.assertTrue(isinstance(result.filterbank, Filterbank))
        # attributes
        self.assertTrue(result.crossover_frequencies == [200, 1000])
        self.assertTrue(result.shape == (281, 3))
        # properties
        self.assertTrue(result.num_frames == 281)
        self.assertTrue(result.num_bins == 3)
        self.assertTrue(np.allclose(result.bin_frequencies,
                                    [86.1328125, 581.39648438, 8979.34570312]))


class TestMultiBandSpectrogramProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = MultiBandSpectrogramProcessor([200, 1000])

    def test_types(self):
        self.assertIsInstance(self.processor, MultiBandSpectrogramProcessor)
        self.assertIsInstance(self.processor, Processor)
        self.assertIsInstance(self.processor.crossover_frequencies, np.ndarray)
        self.assertIsInstance(self.processor.fmin, float)
        self.assertIsInstance(self.processor.fmax, float)
        self.assertIsInstance(self.processor.norm_filters, bool)
        self.assertIsInstance(self.processor.unique_filters, bool)

    def test_values(self):
        self.assertTrue(np.allclose(self.processor.crossover_frequencies,
                                    [200, 1000]))
        self.assertTrue(self.processor.fmin == 30)
        self.assertTrue(self.processor.fmax == 17000)
        self.assertTrue(self.processor.norm_filters is True)
        self.assertTrue(self.processor.unique_filters is True)

    def test_process(self):
        # default values
        result = self.processor.process(sample_file)
        self.assertTrue(np.allclose(result[:3],
                                    [[10.95971966, 4.23556566, 0.19092605],
                                     [11.38149452, 4.88609695, 0.21491699],
                                     [13.50860405, 4.48350096, 0.20132662]]))
        self.assertIsInstance(result, MultiBandSpectrogram)
        # attributes
        self.assertTrue(result.shape == (281, 3))
        self.assertTrue(np.allclose(result.crossover_frequencies, [200, 1000]))
        self.assertTrue(np.allclose(result.bin_frequencies,
                                    [86.1328, 581.3965, 8979.3457]))
        # properties
        self.assertTrue(result.num_bins == 3)
        self.assertTrue(result.num_frames == 281)
        # test 2 bands
        self.processor.crossover_frequencies = [500]
        result = self.processor.process(sample_file)
        self.assertTrue(np.allclose(result[:3],
                                    [[9.37507915, 0.23498698],
                                     [10.4683371, 0.26268598],
                                     [10.8684139, 0.24078195]]))
        self.assertIsInstance(result, MultiBandSpectrogram)
        self.assertTrue(result.shape == (281, 2))
        self.assertTrue(np.allclose(result.crossover_frequencies, [500]))
        self.assertTrue(np.allclose(result.bin_frequencies,
                                    [236.865, 8720.947]))
        self.assertTrue(np.allclose(np.max(result.filterbank, axis=0),
                                    [0.04545455, 0.00130548]))
        # properties
        self.assertTrue(result.num_bins == 2)
        self.assertTrue(result.num_frames == 281)
        # test without normalized filters
        self.processor.norm_filters = False
        result = self.processor.process(sample_file)
        self.assertTrue(np.allclose(result[:3],
                                    [[206.25172424, 180],
                                     [230.30342102, 201.21743774],
                                     [239.10510254, 184.43896484]]))
        self.assertIsInstance(result, MultiBandSpectrogram)
        self.assertTrue(result.shape == (281, 2))
        self.assertTrue(np.allclose(result.bin_frequencies,
                                    [236.865, 8720.947]))
        self.assertTrue(np.allclose(np.max(result.filterbank, axis=0), [1, 1]))


class TestSemitoneBandpassSpectrogramClass(unittest.TestCase):

    def setUp(self):
        self.sbs_50 = SemitoneBandpassSpectrogram(sample_file, fps=50)
        self.sbs_10 = SemitoneBandpassSpectrogram(sample_file, fps=10)
        self.sbs_22050 = SemitoneBandpassSpectrogram(
            sample_file_22050, fps=50, fmin=2637, fmax=4200)
        data = Signal(sample_file)
        self.sbs_10_from_signal = SemitoneBandpassSpectrogram(data, fps=10)

    def test_process(self):
        # test fps = 50
        self.assertTrue(self.sbs_50.fps == 50)
        # results
        self.assertTrue(self.sbs_50.shape == (141, 88))
        self.assertTrue(self.sbs_50.num_bins == 88)
        self.assertTrue(np.allclose(self.sbs_50[120:122, 50:55],
                                    [[0.00056659, 0.00274373, 0.00037994,
                                      0.00031497, 0.0063823],
                                     [0.00032294, 0.00285728, 0.00023723,
                                      0.00010553, 0.0069074]]))
        self.assertTrue(np.allclose(self.sbs_50[:10, 0],
                                    [0.00108844, 0.0020613, 0.00187792,
                                     0.00173228, 0.00163516, 0.00149813,
                                     0.0013027, 0.0010594, 0.00079916,
                                     0.00060871], atol=1e-04))
        self.assertTrue(np.allclose(self.sbs_50[:10, 29],
                                    [0.05326259, 0.10912816, 0.11616101,
                                     0.11595627, 0.11979639, 0.12206492,
                                     0.12836982, 0.12495992, 0.11759637,
                                     0.10559082]))
        # test fps = 10
        self.assertTrue(self.sbs_10.fps == 10)
        self.assertTrue(self.sbs_10.shape == (29, 88))
        sbs_10 = [[0.01951193, 0.01638364, 0.00384092, 0.00732366, 0.10310112],
                  [0.14484727, 0.032042, 0.00719009, 0.02043642, 0.06407038]]
        self.assertTrue(np.allclose(self.sbs_10[10:12, 50:55], sbs_10))
        # test computing SemitoneBandpassSpectrogram from signal
        self.assertTrue(self.sbs_10_from_signal.shape == (29, 88))
        self.assertTrue(np.allclose(self.sbs_10_from_signal[10:12, 50:55],
                                    sbs_10))
        # test 22050 Hz sampling rate. If we use only bands above 2637 Hz,
        # no resampling is necessary and we can therefore compare with
        # smaller tolerances.
        self.assertTrue(self.sbs_22050.shape == (141, 9))
        tar = [[0.06541425, 0.09758339, 0.09000319, 0.06654418, 0.06468658,
                0.05898506, 0.03190501, 0.04980498, 0.07482897],
               [0.07191198, 0.07706247, 0.05581443, 0.03765683, 0.04524021,
                0.03835757, 0.0295172, 0.04417975, 0.06682143]]
        self.assertTrue(np.allclose(self.sbs_22050[108:110, :], tar))
        # check end of signal
        tar = [9.44913489e-06, 2.15330783e-05, 1.61559697e-05, 3.66821812e-06,
               7.96367061e-06, 2.01982581e-05, 2.03380816e-06, 5.34317005e-06,
               4.13617626e-06]
        self.assertTrue(np.allclose(self.sbs_22050[140, :], tar))

    def test_compare_with_matlab_toolbox(self):
        # compare the results with the MATLAB chroma toolbox. There are
        # differences because of different resampling and filtering with
        # filtfilt, therefore we compare with higher tolerances.
        self.assertTrue(np.allclose(self.sbs_50[:10, 29],
                                    [0.054849, 0.114634, 0.115050, 0.119006,
                                     0.128422, 0.128793, 0.127636, 0.124041,
                                     0.113962, 0.103785], rtol=1e-01))
        self.assertTrue(np.allclose(self.sbs_10[10:12, 50:55],
                                    [[0.01951726, 0.01638535, 0.00384128,
                                      0.00732471, 0.10306561],
                                     [0.14487972, 0.03204085, 0.00718818,
                                      0.02043327, 0.06404668]], rtol=1e-03))
