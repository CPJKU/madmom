# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.audio.spectrogram module.

"""

from __future__ import absolute_import, division, print_function

import unittest

from . import AUDIO_PATH
from .test_audio_filters import FFT_FREQS_1024, LOG_FILTERBANK_CENTER_FREQS

from madmom.audio.spectrogram import *
from madmom.audio.filters import Filterbank, LogarithmicFilterbank
from madmom.audio.stft import ShortTimeFourierTransform


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


# noinspection PyArgumentList
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


class TestAdaptiveWhiteningFunction(unittest.TestCase):

    def test_errors(self):
        with self.assertRaises(NotImplementedError):
            adaptive_whitening(np.random.rand(10))


class TestStatisticalSpectrumDescriptorsFunction(unittest.TestCase):

    def test_types(self):
        result = statistical_spectrum_descriptors(np.random.rand(10))
        self.assertIsInstance(result, dict)
        self.assertTrue(result['mean'].dtype == np.float)
        self.assertTrue(result['median'].dtype == np.float)
        self.assertTrue(result['variance'].dtype == np.float)
        self.assertTrue(type(result['skewness']) == float)
        self.assertTrue(type(result['kurtosis']) == float)
        self.assertTrue(result['min'].dtype == np.float)
        self.assertTrue(result['max'].dtype == np.float)
        result = statistical_spectrum_descriptors(np.random.rand(10, 2))
        self.assertIsInstance(result, dict)
        self.assertTrue(result['mean'].dtype == np.float)
        self.assertTrue(result['median'].dtype == np.float)
        self.assertTrue(result['variance'].dtype == np.float)
        self.assertTrue(result['skewness'].dtype == np.float)
        self.assertTrue(result['kurtosis'].dtype == np.float)
        self.assertTrue(result['min'].dtype == np.float)
        self.assertTrue(result['max'].dtype == np.float)

    def test_values(self):
        from scipy.stats import skew, kurtosis
        data = np.random.rand(10)
        result = statistical_spectrum_descriptors(data)
        self.assertTrue(np.allclose(result['mean'], np.mean(data, axis=0)))
        self.assertTrue(np.allclose(result['median'], np.median(data, axis=0)))
        self.assertTrue(np.allclose(result['variance'], np.var(data, axis=0)))
        self.assertTrue(np.allclose(result['skewness'], skew(data)))
        self.assertTrue(np.allclose(result['kurtosis'], kurtosis(data)))
        self.assertTrue(np.allclose(result['min'], np.min(data, axis=0)))
        self.assertTrue(np.allclose(result['max'], np.max(data, axis=0)))
        data = np.random.rand(10, 2)
        result = statistical_spectrum_descriptors(data)
        self.assertTrue(np.allclose(result['mean'], np.mean(data, axis=0)))
        self.assertTrue(np.allclose(result['median'], np.median(data, axis=0)))
        self.assertTrue(np.allclose(result['variance'], np.var(data, axis=0)))
        self.assertTrue(np.allclose(result['skewness'], skew(data)))
        self.assertTrue(np.allclose(result['kurtosis'], kurtosis(data)))
        self.assertTrue(np.allclose(result['min'], np.min(data, axis=0)))
        self.assertTrue(np.allclose(result['max'], np.max(data, axis=0)))


class TestSpectrogramClass(unittest.TestCase):

    def test_types(self):
        result = Spectrogram(AUDIO_PATH + '/sample.wav')
        self.assertIsInstance(result, Spectrogram)
        # attributes
        self.assertIsInstance(result.stft, ShortTimeFourierTransform)
        self.assertIsInstance(result.bin_frequencies, np.ndarray)
        # properties
        self.assertIsInstance(result.num_bins, int)
        self.assertIsInstance(result.num_frames, int)
        # other faked attributes
        self.assertTrue(result.filterbank is None)
        self.assertTrue(result.mul is None)
        self.assertTrue(result.add is None)

    def test_values(self):
        # from file
        result = Spectrogram(AUDIO_PATH + '/sample.wav')
        # attributes
        self.assertTrue(result.shape == (281, 1024))
        self.assertTrue(np.allclose(result.bin_frequencies, FFT_FREQS_1024))
        # properties
        self.assertTrue(result.num_frames == 281)
        self.assertTrue(result.num_bins == 1024)
        # from spec
        self.assertTrue(np.allclose(Spectrogram(result), result))
        # from stft
        stft = ShortTimeFourierTransform(AUDIO_PATH + '/sample.wav')
        self.assertTrue(np.allclose(Spectrogram(stft), result))

    def test_methods(self):
        result = Spectrogram(AUDIO_PATH + '/sample.wav')
        self.assertIsInstance(result.diff(), SpectrogramDifference)
        self.assertIsInstance(result.filter(), FilteredSpectrogram)
        self.assertIsInstance(result.log(), LogarithmicSpectrogram)


class TestSpectrogramProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = SpectrogramProcessor()

    def test_types(self):
        self.assertIsInstance(self.processor, SpectrogramProcessor)

    def test_process(self):
        result = self.processor.process(AUDIO_PATH + '/sample.wav')
        self.assertIsInstance(result, Spectrogram)
        # attributes
        self.assertTrue(result.shape == (281, 1024))
        self.assertTrue(np.allclose(result.bin_frequencies, FFT_FREQS_1024))
        # properties
        self.assertTrue(result.num_frames == 281)
        self.assertTrue(result.num_bins == 1024)


class TestFilteredSpectrogramClass(unittest.TestCase):

    def test_types(self):
        result = FilteredSpectrogram(AUDIO_PATH + '/sample.wav')
        self.assertIsInstance(result, FilteredSpectrogram)
        self.assertIsInstance(result, Spectrogram)
        # attributes
        self.assertIsInstance(result.stft, ShortTimeFourierTransform)
        self.assertIsInstance(result.filterbank, LogarithmicFilterbank)
        self.assertIsInstance(result.bin_frequencies, np.ndarray)
        # properties
        self.assertIsInstance(result.num_bins, int)
        self.assertIsInstance(result.num_frames, int)
        # other faked attributes
        self.assertTrue(result.mul is None)
        self.assertTrue(result.add is None)
        # wrong filterbank type
        with self.assertRaises(TypeError):
            FilteredSpectrogram(AUDIO_PATH + '/sample.wav', filterbank='bla')

    def test_values(self):
        # from file
        result = FilteredSpectrogram(AUDIO_PATH + '/sample.wav')
        # attributes
        self.assertTrue(result.shape == (281, 81))
        self.assertTrue(np.allclose(result.bin_frequencies,
                                    LOG_FILTERBANK_CENTER_FREQS))
        # properties
        self.assertTrue(result.num_bins == 81)
        self.assertTrue(result.num_frames == 281)
        # with given filterbank
        result = FilteredSpectrogram(AUDIO_PATH + '/sample.wav',
                                     filterbank=result.filterbank)
        # attributes
        self.assertTrue(result.shape == (281, 81))
        self.assertTrue(np.allclose(result.bin_frequencies,
                                    LOG_FILTERBANK_CENTER_FREQS))
        # properties
        self.assertTrue(result.num_bins == 81)
        self.assertTrue(result.num_frames == 281)

    def test_methods(self):
        result = FilteredSpectrogram(AUDIO_PATH + '/sample.wav')
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
        result = self.processor.process(AUDIO_PATH + '/sample.wav')
        self.assertIsInstance(result, FilteredSpectrogram)
        # attributes
        self.assertTrue(result.shape == (281, 81))
        self.assertTrue(np.allclose(result.bin_frequencies,
                                    LOG_FILTERBANK_CENTER_FREQS))
        # properties
        self.assertTrue(result.num_bins == 81)
        self.assertTrue(result.num_frames == 281)
        # changed values
        self.processor.num_bands = 6
        self.processor.fmin = 300
        self.processor.fmax = 10000
        result = self.processor.process(AUDIO_PATH + '/sample.wav')
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
        result = LogarithmicSpectrogram(AUDIO_PATH + '/sample.wav')
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
        # other faked attributes
        self.assertTrue(result.filterbank is None)

    def test_values(self):
        result = LogarithmicSpectrogram(AUDIO_PATH + '/sample.wav')
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
        result = LogarithmicSpectrogram(AUDIO_PATH + '/sample.wav',
                                        mul=2, add=2)
        self.assertTrue(result.mul == 2)
        self.assertTrue(result.add == 2)

    def test_methods(self):
        result = LogarithmicSpectrogram(AUDIO_PATH + '/sample.wav')
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
        result = self.processor.process(AUDIO_PATH + '/sample.wav')
        self.assertIsInstance(result, LogarithmicSpectrogram)
        self.assertTrue(result.shape == (281, 1024))


class TestLogarithmicFilteredSpectrogramClass(unittest.TestCase):

    def test_types(self):
        result = LogarithmicFilteredSpectrogram(AUDIO_PATH + '/sample.wav')
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
        result = LogarithmicFilteredSpectrogram(AUDIO_PATH + '/sample.wav')
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
        result = LogarithmicFilteredSpectrogram(AUDIO_PATH + '/sample.wav',
                                                mul=2, add=2)
        self.assertTrue(result.mul == 2)
        self.assertTrue(result.add == 2)


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
        result = self.processor.process(AUDIO_PATH + '/sample.wav')
        self.assertIsInstance(result, LogarithmicFilteredSpectrogram)

        self.assertTrue(result.shape == (281, 81))


class TestSpectrogramDifferenceClass(unittest.TestCase):

    def test_types(self):
        result = SpectrogramDifference(AUDIO_PATH + '/sample.wav')
        self.assertIsInstance(result, SpectrogramDifference)
        self.assertIsInstance(result, Spectrogram)
        # attributes
        self.assertIsInstance(result.stft, ShortTimeFourierTransform)
        self.assertIsInstance(result.bin_frequencies, np.ndarray)
        self.assertIsInstance(result.diff_ratio, float)
        self.assertIsInstance(result.diff_frames, int)
        self.assertTrue(result.diff_max_bins is None)
        self.assertIsInstance(result.positive_diffs, bool)
        # properties
        self.assertIsInstance(result.num_frames, int)
        self.assertIsInstance(result.num_bins, int)
        # other faked attributes
        self.assertTrue(result.filterbank is None)
        self.assertTrue(result.mul is None)
        self.assertTrue(result.add is None)

    def test_values(self):
        result = SpectrogramDifference(AUDIO_PATH + '/sample.wav')
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

    def test_values(self):
        self.assertTrue(self.processor.diff_ratio == 0.5)
        self.assertTrue(self.processor.diff_frames is None)
        self.assertTrue(self.processor.diff_max_bins is None)
        self.assertTrue(self.processor.positive_diffs is False)

    def test_process(self):
        # default values
        result = self.processor.process(AUDIO_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 1024))
        self.assertTrue(np.sum(result[:1]) == 0)
        self.assertTrue(np.min(result) <= 0)
        # change diff frames
        self.processor.diff_frames = 2
        result = self.processor.process(AUDIO_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 1024))
        self.assertTrue(np.sum(result[:2]) == 0)
        self.assertTrue(np.min(result) <= 0)
        # change positive diffs
        self.processor.positive_diffs = True
        result = self.processor.process(AUDIO_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 1024))
        self.assertTrue(np.sum(result[:2]) == 0)
        self.assertTrue(np.min(result) <= 0)


class TestSuperFluxProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = SuperFluxProcessor()

    def test_types(self):
        self.assertIsInstance(self.processor, SuperFluxProcessor)

    def test_values(self):
        result = self.processor.process(AUDIO_PATH + '/sample.wav')
        self.assertIsInstance(result, SpectrogramDifference)
        self.assertTrue(result.num_bins == 140)
        self.assertTrue(result.num_frames == 281)
        self.assertTrue(result.shape == (281, 140))
        # filterbank stuff
        self.assertIsInstance(result.filterbank, LogarithmicFilterbank)
        self.assertTrue(result.filterbank.num_bands_per_octave == 24)
        # log stuff
        self.assertTrue(result.mul == 1)
        self.assertTrue(result.add == 1)
        # diff stuff
        self.assertTrue(result.diff_ratio == 0.5)
        self.assertTrue(result.diff_max_bins == 3)
        self.assertTrue(result.positive_diffs is True)


class TestMultiBandSpectrogramClass(unittest.TestCase):

    def test_types(self):
        result = MultiBandSpectrogram(AUDIO_PATH + '/sample.wav', [200, 1000])
        self.assertIsInstance(result, MultiBandSpectrogram)
        self.assertTrue(type(result.crossover_frequencies) == list)
        self.assertTrue(type(result.norm_bands) == bool)
        # properties
        self.assertIsInstance(result.num_frames, int)
        self.assertIsInstance(result.bin_frequencies, np.ndarray)
        self.assertIsInstance(result.num_bins, int)

    def test_values(self):
        result = MultiBandSpectrogram(AUDIO_PATH + '/sample.wav', [200, 1000])
        self.assertTrue(isinstance(result.filterbank, Filterbank))
        self.assertTrue(result.crossover_frequencies == [200, 1000])
        self.assertTrue(result.norm_bands is False)
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
        self.assertIsInstance(self.processor.norm_bands, bool)

    def test_values(self):
        self.assertTrue(np.allclose(self.processor.crossover_frequencies,
                                    [200, 1000]))
        self.assertTrue(self.processor.norm_bands is False)

    def test_process(self):
        # default values
        result = self.processor.process(AUDIO_PATH + '/sample.wav')
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
        result = self.processor.process(AUDIO_PATH + '/sample.wav')
        self.assertIsInstance(result, MultiBandSpectrogram)
        self.assertTrue(result.shape == (281, 2))
        self.assertTrue(np.allclose(result.crossover_frequencies, [500]))
        self.assertTrue(np.allclose(result.bin_frequencies,
                                    [236.865, 8720.947]))
        self.assertTrue(np.allclose(np.max(result.filterbank, axis=0), [1, 1]))
        # properties
        self.assertTrue(result.num_bins == 2)
        self.assertTrue(result.num_frames == 281)

        # test norm bands
        self.processor.norm_bands = True
        result = self.processor.process(AUDIO_PATH + '/sample.wav')
        self.assertIsInstance(result, MultiBandSpectrogram)
        self.assertTrue(result.shape == (281, 2))
        self.assertTrue(np.allclose(result.bin_frequencies,
                                    [236.865, 8720.947]))
        self.assertTrue(np.allclose(np.max(result.filterbank, axis=0),
                                    [0.04545455, 0.00130548]))


class TestStackedSpectrogramProcessorClass(unittest.TestCase):

    def test_types(self):
        frame_sizes = [512, 1024, 2048]
        spec_processor = SpectrogramProcessor()
        processor = StackedSpectrogramProcessor(frame_sizes, spec_processor)
        self.assertIsInstance(processor, StackedSpectrogramProcessor)
        self.assertIsInstance(processor, Processor)

    def test_stack_specs(self):
        # stack only the specs
        spec_processor = LogarithmicFilteredSpectrogramProcessor()
        processor = StackedSpectrogramProcessor([512], spec_processor)
        result = processor.process(AUDIO_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 58))
        processor = StackedSpectrogramProcessor([1024], spec_processor)
        result = processor.process(AUDIO_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 69))
        processor = StackedSpectrogramProcessor([2048], spec_processor)
        result = processor.process(AUDIO_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 81))
        processor = StackedSpectrogramProcessor([512, 1024, 2048],
                                                spec_processor)
        result = processor.process(AUDIO_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 58 + 69 + 81))

    def test_stack_diffs(self):
        # also include the differences
        spec_processor = LogarithmicFilteredSpectrogramProcessor()
        diff_processor = SpectrogramDifferenceProcessor()
        processor = StackedSpectrogramProcessor([512], spec_processor,
                                                diff_processor)
        result = processor.process(AUDIO_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 116))
        processor = StackedSpectrogramProcessor([1024], spec_processor,
                                                diff_processor)
        result = processor.process(AUDIO_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 138))
        processor = StackedSpectrogramProcessor([2048], spec_processor,
                                                diff_processor)
        result = processor.process(AUDIO_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 162))
        processor = StackedSpectrogramProcessor([512, 1024, 2048],
                                                spec_processor,
                                                diff_processor)
        result = processor.process(AUDIO_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 116 + 138 + 162))

    def test_stack_depth(self):
        # stack in depth
        spec_processor = LogarithmicFilteredSpectrogramProcessor(
            unique_filters=False)
        processor = StackedSpectrogramProcessor([512], spec_processor,
                                                stack='depth')
        result = processor.process(AUDIO_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 108, 1))
        processor = StackedSpectrogramProcessor([1024], spec_processor,
                                                stack='depth')
        result = processor.process(AUDIO_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 108, 1))
        processor = StackedSpectrogramProcessor([2048], spec_processor,
                                                stack='depth')
        result = processor.process(AUDIO_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 108, 1))
        processor = StackedSpectrogramProcessor([512, 1024, 2048],
                                                spec_processor,
                                                stack='depth')
        result = processor.process(AUDIO_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 108, 3))

    def test_stack_literals(self):
        spec_processor = LogarithmicFilteredSpectrogramProcessor()
        processor = StackedSpectrogramProcessor([512, 1024], spec_processor,
                                                stack='time')
        self.assertEqual(processor.stack, np.vstack)
        processor = StackedSpectrogramProcessor([512, 1024], spec_processor,
                                                stack='freq')
        self.assertEqual(processor.stack, np.hstack)
        processor = StackedSpectrogramProcessor([512, 1024], spec_processor,
                                                stack='frequency')
        self.assertEqual(processor.stack, np.hstack)
        processor = StackedSpectrogramProcessor([512, 1024], spec_processor,
                                                stack='depth')
        self.assertEqual(processor.stack, np.dstack)
