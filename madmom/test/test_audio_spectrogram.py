# encoding: utf-8
"""
This file contains tests for the madmom.audio.spectrogram module.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""
# pylint: skip-file

import unittest

from . import DATA_PATH

from madmom.audio.spectrogram import *
from madmom.audio.signal import FramedSignal
from madmom.audio.filters import Filterbank, LogarithmicFilterbank


# test functions
class TestBinFrequenciesFunction(unittest.TestCase):

    def test_num_arguments(self):
        # number of arguments arguments
        with self.assertRaises(TypeError):
            fft_frequencies()
        with self.assertRaises(TypeError):
            fft_frequencies(1)
        with self.assertRaises(TypeError):
            fft_frequencies(1, 2, 3)

    def test_types(self):
        result = fft_frequencies(5, 10)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float)

    def test_value(self):
        result = fft_frequencies(5, 10)
        self.assertTrue(np.allclose(result, [0, 1, 2, 3, 4]))


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


class TestPhaseFunction(unittest.TestCase):

    def test_types(self):
        result = phase(np.random.rand(10))
        self.assertTrue(result.dtype == np.float)
        self.assertTrue(result.shape == (10, ))
        result = phase(np.random.rand(10, 2))
        self.assertTrue(result.dtype == np.float)
        self.assertTrue(result.shape == (10, 2))
        # complex data
        data = np.random.rand(10) + 1j * np.random.rand(10)
        result = phase(data)
        self.assertTrue(result.dtype == np.float)
        self.assertTrue(result.shape == (10, ))
        data = np.random.rand(10, 2) + 1j * np.random.rand(10, 2)
        result = phase(data)
        self.assertTrue(result.dtype == np.float)
        self.assertTrue(result.shape == (10, 2))

    def test_values(self):
        data = np.random.rand(10) + 1j * np.random.rand(10)
        self.assertTrue(np.allclose(np.angle(data), phase(data)))
        data = np.random.rand(10, 2) + 1j * np.random.rand(10, 2)
        self.assertTrue(np.allclose(np.angle(data), phase(data)))


class TestLocalGroupDelayFunction(unittest.TestCase):

    def test_errors(self):
        self.assertTrue(True)


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


class TestTuningFrequencyFunction(unittest.TestCase):

    def test_errors(self):
        with self.assertRaises(NotImplementedError):
            adaptive_whitening(np.random.rand(10))


# test classes
class TestSpectrogramClass(unittest.TestCase):

    def test_types(self):
        result = Spectrogram(DATA_PATH + '/sample.wav')
        self.assertIsInstance(result, Spectrogram)
        self.assertIsInstance(result.frames, FramedSignal)
        self.assertIsInstance(result.window, np.ndarray)
        self.assertIsInstance(result.fft_size, int)
        self.assertIsInstance(result.fft_window, np.ndarray)
        self.assertIsInstance(result.block_size, int)
        self.assertIsInstance(result.filterbank, type(None))
        self.assertIsInstance(result.log, bool)
        self.assertIsInstance(result.mul, float)
        self.assertIsInstance(result.add, float)
        self.assertIsInstance(result.num_diff_frames, int)
        self.assertIsInstance(result.diff_max_bins, int)
        self.assertIsInstance(result.positive_diff, bool)
        # properties
        self.assertIsInstance(result.num_frames, int)
        self.assertIsInstance(result.fft_freqs, np.ndarray)
        self.assertIsInstance(result.num_fft_bins, int)
        self.assertIsInstance(result.num_bins, int)
        self.assertIsInstance(result.stft, np.ndarray)
        self.assertIsInstance(result.spec, np.ndarray)
        self.assertIsInstance(result.magnitude, np.ndarray)
        self.assertIsInstance(result.phase, np.ndarray)
        self.assertIsInstance(result.lgd, np.ndarray)
        self.assertIsInstance(result.diff, np.ndarray)

    def test_types_filterbank(self):
        result = Spectrogram(DATA_PATH + '/sample.wav',
                             filterbank=LogarithmicFilterbank)
        self.assertIsInstance(result, Spectrogram)
        self.assertIsInstance(result.filterbank, Filterbank)

    def test_values(self):
        result = Spectrogram(DATA_PATH + '/sample.wav')
        self.assertTrue(np.allclose(result.window, np.hanning(2048)))
        self.assertTrue(result.fft_size == 2048)
        self.assertTrue(np.allclose(result.fft_window,
                                    np.hanning(2048) / 32767))
        self.assertTrue(result.block_size == 2048)
        self.assertTrue(result.filterbank is None)
        self.assertTrue(result.log is False)
        self.assertTrue(result.mul == 1)
        self.assertTrue(result.add == 1)
        self.assertTrue(result.num_diff_frames == 1)
        self.assertTrue(result.diff_max_bins == 1)
        self.assertTrue(result.positive_diff is True)
        # properties
        self.assertTrue(result.num_frames == 281)
        self.assertTrue(np.allclose(result.fft_freqs,
                                    fft_frequencies(1024, 44100)))
        self.assertTrue(result.num_fft_bins == 1024)
        self.assertTrue(result.num_bins == 1024)
        self.assertTrue(result.stft.shape == (281, 1024))
        self.assertTrue(result.phase.shape == (281, 1024))
        self.assertTrue(result.lgd.shape == (281, 1024))
        self.assertTrue(result.spec.shape == (281, 1024))
        self.assertTrue(result.magnitude.shape == (281, 1024))
        self.assertTrue(result.diff.shape == (281, 1024))

    def test_values_filterbank(self):
        result = Spectrogram(DATA_PATH + '/sample.wav',
                             filterbank=LogarithmicFilterbank)
        self.assertTrue(np.allclose(result.window, np.hanning(2048)))
        self.assertTrue(result.fft_size == 2048)
        self.assertTrue(np.allclose(result.fft_window,
                                    np.hanning(2048) / 32767))
        self.assertTrue(np.allclose(result.filterbank,
                                    LogarithmicFilterbank(result.fft_freqs)))
        self.assertTrue(result.num_fft_bins == 1024)
        self.assertTrue(result.num_bins == 81)
        # these matrices are not filtered
        self.assertTrue(result.stft.shape == (281, 1024))
        self.assertTrue(result.phase.shape == (281, 1024))
        self.assertTrue(result.lgd.shape == (281, 1024))
        # these matrices are filtered
        self.assertTrue(result.spec.shape == (281, 81))
        self.assertTrue(result.magnitude.shape == (281, 81))
        self.assertTrue(result.diff.shape == (281, 81))

    def test_values_log(self):
        result = Spectrogram(DATA_PATH + '/sample.wav', log=True, mul=2, add=1)
        self.assertTrue(np.allclose(result.window, np.hanning(2048)))
        self.assertTrue(result.fft_size == 2048)
        self.assertTrue(np.allclose(result.fft_window,
                                    np.hanning(2048) / 32767))
        self.assertTrue(result.log is True)
        self.assertTrue(result.mul == 2)
        self.assertTrue(result.add == 1)


class TestSpectrogramProcessorClass(unittest.TestCase):

    def test_types(self):
        processor = SpectrogramProcessor()
        self.assertIsInstance(processor, SpectrogramProcessor)
        self.assertTrue(issubclass(processor.filterbank,
                                   LogarithmicFilterbank))
        self.assertIsInstance(processor.bands, int)
        self.assertIsInstance(processor.fmin, float)
        self.assertIsInstance(processor.fmax, float)
        self.assertIsInstance(processor.log, bool)
        self.assertIsInstance(processor.mul, float)
        self.assertIsInstance(processor.add, float)
        self.assertIsInstance(processor.diff_ratio, float)
        self.assertIsInstance(processor.diff_frames, type(None))
        self.assertIsInstance(processor.diff_max_bins, int)

    def test_types_filterbank(self):
        processor = SpectrogramProcessor(filterbank=True)
        self.assertTrue(issubclass(processor.filterbank,
                                   LogarithmicFilterbank))
        processor = SpectrogramProcessor(filterbank=False)
        self.assertIsInstance(processor.filterbank, type(None))

    def test_values(self):
        processor = SpectrogramProcessor()
        self.assertTrue(issubclass(processor.filterbank,
                                   LogarithmicFilterbank))
        self.assertTrue(processor.bands == 12)
        self.assertTrue(processor.fmin == 30)
        self.assertTrue(processor.fmax == 17000)
        self.assertTrue(processor.norm_filters is True)
        self.assertTrue(processor.log is True)
        self.assertTrue(processor.mul == 1)
        self.assertTrue(processor.add == 1)
        self.assertTrue(processor.diff_ratio == 0.5)
        self.assertTrue(processor.diff_frames is None)
        self.assertTrue(processor.diff_max_bins == 1)
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(result.stft.shape == (281, 1024))
        self.assertTrue(result.phase.shape == (281, 1024))
        self.assertTrue(result.lgd.shape == (281, 1024))
        self.assertTrue(result.spec.shape == (281, 81))
        self.assertTrue(result.diff.shape == (281, 81))
        self.assertTrue(result.diff.min() == 0)

    def test_values_no_filterbank(self):
        processor = SpectrogramProcessor(filterbank=None)
        self.assertTrue(processor.filterbank is None)
        self.assertTrue(processor.bands == 12)
        self.assertTrue(processor.fmin == 30)
        self.assertTrue(processor.fmax == 17000)
        self.assertTrue(processor.norm_filters is True)
        self.assertTrue(processor.log is True)
        self.assertTrue(processor.mul == 1)
        self.assertTrue(processor.add == 1)
        self.assertTrue(processor.diff_ratio == 0.5)
        self.assertTrue(processor.diff_frames is None)
        self.assertTrue(processor.diff_max_bins == 1)
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(result.stft.shape == (281, 1024))
        self.assertTrue(result.phase.shape == (281, 1024))
        self.assertTrue(result.lgd.shape == (281, 1024))
        self.assertTrue(result.spec.shape == (281, 1024))
        self.assertTrue(result.diff.shape == (281, 1024))
        self.assertTrue(result.diff.min() == 0)

    def test_values_others(self):
        processor = SpectrogramProcessor(log=True, mul=2, add=1)
        self.assertTrue(processor.log is True)
        self.assertTrue(processor.mul == 2)
        self.assertTrue(processor.add == 1)
        processor = SpectrogramProcessor(diff_ratio=0.25)
        self.assertTrue(processor.diff_ratio == 0.25)
        processor = SpectrogramProcessor(diff_frames=2)
        self.assertTrue(processor.diff_frames == 2)
        processor = SpectrogramProcessor(diff_max_bins=3)
        self.assertTrue(processor.diff_max_bins == 3)
        processor = SpectrogramProcessor(norm_filters=False)
        self.assertTrue(processor.norm_filters is False)


class TestSuperFluxProcessorClass(unittest.TestCase):

    def test_types(self):
        processor = SuperFluxProcessor()
        self.assertIsInstance(processor, SuperFluxProcessor)
        self.assertIsInstance(processor, SpectrogramProcessor)
        self.assertTrue(issubclass(processor.filterbank,
                                   LogarithmicFilterbank))

    def test_values(self):
        processor = SuperFluxProcessor()
        self.assertTrue(issubclass(processor.filterbank,
                                   LogarithmicFilterbank))
        self.assertTrue(processor.bands == 24)
        self.assertTrue(processor.fmin == 30)
        self.assertTrue(processor.fmax == 17000)
        self.assertTrue(processor.norm_filters is False)
        self.assertTrue(processor.log is True)
        self.assertTrue(processor.mul == 1)
        self.assertTrue(processor.add == 1)
        self.assertTrue(processor.diff_ratio == 0.5)
        self.assertTrue(processor.diff_frames is None)
        self.assertTrue(processor.diff_max_bins == 3)

        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(result.stft.shape == (281, 1024))
        self.assertTrue(result.phase.shape == (281, 1024))
        self.assertTrue(result.lgd.shape == (281, 1024))
        self.assertTrue(result.spec.shape == (281, 140))
        self.assertTrue(result.diff.shape == (281, 140))
        self.assertTrue(result.diff.min() == 0)


class TestMultiBandSpectrogramProcessorClass(unittest.TestCase):

    def test_types(self):
        processor = MultiBandSpectrogramProcessor([200, 1000])
        self.assertIsInstance(processor, MultiBandSpectrogramProcessor)
        self.assertIsInstance(processor, SpectrogramProcessor)
        self.assertTrue(issubclass(processor.filterbank,
                                   LogarithmicFilterbank))
        self.assertTrue(type(processor.crossover_frequencies) == list)
        self.assertTrue(type(processor.norm_bands) == bool)

    def test_values(self):
        processor = MultiBandSpectrogramProcessor([200, 1000])
        self.assertTrue(issubclass(processor.filterbank,
                                   LogarithmicFilterbank))
        self.assertTrue(processor.bands == 12)
        self.assertTrue(processor.fmin == 30)
        self.assertTrue(processor.fmax == 17000)
        self.assertTrue(processor.norm_filters is True)
        self.assertTrue(processor.log is True)
        self.assertTrue(processor.mul == 1)
        self.assertTrue(processor.add == 1)
        self.assertTrue(processor.diff_ratio == 0.5)
        self.assertTrue(processor.diff_frames is None)
        self.assertTrue(processor.diff_max_bins == 1)
        self.assertTrue(processor.crossover_frequencies == [200, 1000])
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 3))
        self.assertTrue(result.min() == 0)


class TestStackSpectrogramProcessorClass(unittest.TestCase):

    def test_types(self):
        processor = StackSpectrogramProcessor([512, 1024, 2048])
        self.assertIsInstance(processor, StackSpectrogramProcessor)
        self.assertIsInstance(processor, Processor)

    def test_stack_specs(self):
        # stack only the specs
        processor = StackSpectrogramProcessor([512])
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 58))
        processor = StackSpectrogramProcessor([1024])
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 69))
        processor = StackSpectrogramProcessor([2048])
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 81))
        processor = StackSpectrogramProcessor([512, 1024, 2048])
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 58 + 69 + 81))

    def test_stack_diffs(self):
        # also include the differences
        processor = StackSpectrogramProcessor([512], stack_diffs=True)
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 116))
        processor = StackSpectrogramProcessor([1024], stack_diffs=True)
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 138))
        processor = StackSpectrogramProcessor([2048], stack_diffs=True)
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 162))
        processor = StackSpectrogramProcessor([512, 1024, 2048],
                                              stack_diffs=True)
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 116 + 138 + 162))

    def test_stack_depth(self):
        # stack in depth
        processor = StackSpectrogramProcessor([512], stack='depth')
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 108, 1))
        processor = StackSpectrogramProcessor([1024], stack='depth')
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 108, 1))
        processor = StackSpectrogramProcessor([2048], stack='depth')
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 108, 1))
        processor = StackSpectrogramProcessor([512, 1024, 2048], stack='depth')
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 108, 3))
