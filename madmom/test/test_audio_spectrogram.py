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
        # properties
        self.assertIsInstance(result.num_frames, int)
        self.assertIsInstance(result.bin_frequencies, np.ndarray)
        self.assertIsInstance(result.num_bins, int)

    def test_values(self):
        result = Spectrogram(DATA_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 1024))
        self.assertTrue(result.num_frames == 281)
        self.assertTrue(result.num_bins == 1024)


class TestSpectrogramProcessorClass(unittest.TestCase):

    def test_types(self):
        processor = SpectrogramProcessor()
        self.assertIsInstance(processor, SpectrogramProcessor)

    def test_values(self):
        processor = SpectrogramProcessor()
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 1024))
        self.assertTrue(result.num_frames == 281)
        self.assertTrue(result.num_bins == 1024)


class ShortTimeFourierTransformClass(unittest.TestCase):

    def test_types(self):
        result = ShortTimeFourierTransform(DATA_PATH + '/sample.wav')
        self.assertIsInstance(result, ShortTimeFourierTransform)
        self.assertIsInstance(result.frames, FramedSignal)
        self.assertIsInstance(result.window, np.ndarray)
        self.assertIsInstance(result.fft_size, int)
        self.assertIsInstance(result.fft_window, np.ndarray)
        # properties
        self.assertIsInstance(result.num_frames, int)
        self.assertIsInstance(result.bin_frequencies, np.ndarray)
        self.assertIsInstance(result.num_bins, int)
        self.assertIsInstance(result, np.ndarray)

    def test_values(self):
        result = ShortTimeFourierTransform(DATA_PATH + '/sample.wav')
        self.assertTrue(np.allclose(result.window, np.hanning(2048)))
        self.assertTrue(result.fft_size == 2048)
        self.assertTrue(np.allclose(result.fft_window,
                                    np.hanning(2048) / 32767))
        # properties
        self.assertTrue(result.num_frames == 281)
        self.assertTrue(np.allclose(result.bin_frequencies,
                                    fft_frequencies(1024, 44100)))
        self.assertTrue(result.num_bins == 1024)
        self.assertTrue(result.shape == (281, 1024))


class ShortTimeFourierTransformProcessorClass(unittest.TestCase):

    def test_types(self):
        processor = ShortTimeFourierTransformProcessor()
        self.assertIsInstance(processor, ShortTimeFourierTransformProcessor)

    def test_values(self):
        processor = ShortTimeFourierTransformProcessor()
        self.assertTrue(processor.window == np.hanning)
        self.assertTrue(processor.fft_size is None)
        self.assertTrue(processor.circular_shift is False)
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertIsInstance(result, ShortTimeFourierTransform)
        self.assertTrue(result.fft_size == 2048)
        self.assertTrue(np.allclose(result.fft_window,
                                    np.hanning(2048) / 32767))
        # properties
        self.assertTrue(result.num_frames == 281)
        self.assertTrue(np.allclose(result.bin_frequencies,
                                    fft_frequencies(1024, 44100)))
        self.assertTrue(result.num_bins == 1024)
        self.assertTrue(result.shape == (281, 1024))


class TestFilteredSpectrogramClass(unittest.TestCase):

    def test_types(self):
        result = FilteredSpectrogram(DATA_PATH + '/sample.wav')
        self.assertIsInstance(result, FilteredSpectrogram)
        self.assertIsInstance(result.frames, FramedSignal)
        self.assertIsInstance(result.filterbank, LogarithmicFilterbank)
        # properties
        self.assertIsInstance(result.num_frames, int)
        self.assertIsInstance(result.bin_frequencies, np.ndarray)
        self.assertIsInstance(result.num_bins, int)

    def test_values(self):
        result = FilteredSpectrogram(DATA_PATH + '/sample.wav')
        self.assertTrue(result.num_bins == 81)
        self.assertTrue(result.num_frames == 281)


class TestFilteredSpectrogramProcessorClass(unittest.TestCase):

    def test_types(self):
        processor = FilteredSpectrogramProcessor()
        self.assertIsInstance(processor, FilteredSpectrogramProcessor)
        self.assertTrue(issubclass(processor.filterbank,
                                   LogarithmicFilterbank))
        self.assertIsInstance(processor.num_bands, int)
        self.assertIsInstance(processor.fmin, float)
        self.assertIsInstance(processor.fmax, float)
        self.assertIsInstance(processor.fref, float)

    def test_values(self):
        processor = FilteredSpectrogramProcessor()
        self.assertTrue(issubclass(processor.filterbank,
                                   LogarithmicFilterbank))
        self.assertTrue(processor.num_bands == 12)
        self.assertTrue(processor.fmin == 30)
        self.assertTrue(processor.fmax == 17000)
        self.assertTrue(processor.fref == 440)
        self.assertTrue(processor.norm_filters is True)
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 81))


class TestLogarithmicSpectrogramClass(unittest.TestCase):

    def test_types(self):
        result = LogarithmicSpectrogram(DATA_PATH + '/sample.wav')
        self.assertIsInstance(result, LogarithmicSpectrogram)
        self.assertIsInstance(result.frames, FramedSignal)
        self.assertIsInstance(result.mul, float)
        self.assertIsInstance(result.add, float)
        # properties
        self.assertIsInstance(result.num_frames, int)
        self.assertIsInstance(result.bin_frequencies, np.ndarray)
        self.assertIsInstance(result.num_bins, int)

    def test_values(self):
        result = LogarithmicSpectrogram(DATA_PATH + '/sample.wav')
        self.assertTrue(result.mul == 1)
        self.assertTrue(result.add == 1)
        # properties
        self.assertTrue(result.num_frames == 281)
        self.assertTrue(result.num_bins == 1024)
        self.assertTrue(result.shape == (281, 1024))
        # test other values
        result = LogarithmicSpectrogram(DATA_PATH + '/sample.wav',
                                        mul=2, add=2)
        self.assertTrue(result.mul == 2)
        self.assertTrue(result.add == 2)


class TestLogarithmicSpectrogramProcessorClass(unittest.TestCase):

    def test_types(self):
        processor = LogarithmicSpectrogramProcessor()
        self.assertIsInstance(processor, LogarithmicSpectrogramProcessor)
        self.assertIsInstance(processor.mul, float)
        self.assertIsInstance(processor.add, float)

    def test_values(self):
        processor = LogarithmicSpectrogramProcessor()
        self.assertTrue(processor.mul == 1)
        self.assertTrue(processor.add == 1)
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 1024))


class TestSuperFluxProcessorClass(unittest.TestCase):

    def test_types(self):
        processor = SuperFluxProcessor()
        self.assertIsInstance(processor, SuperFluxProcessor)

    def test_values(self):
        processor = SuperFluxProcessor()
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertIsInstance(result, SpectrogramDifference)
        self.assertTrue(result.num_bins == 140)
        self.assertTrue(result.num_frames == 281)
        self.assertTrue(result.shape == (281, 140))


class TestMultiBandSpectrogramClass(unittest.TestCase):

    def test_types(self):
        result = MultiBandSpectrogram(DATA_PATH + '/sample.wav', [200, 1000])
        self.assertIsInstance(result, MultiBandSpectrogram)
        self.assertTrue(type(result.crossover_frequencies) == list)
        self.assertTrue(type(result.norm_bands) == bool)
        # properties
        self.assertIsInstance(result.num_frames, int)
        self.assertIsInstance(result.bin_frequencies, np.ndarray)
        self.assertIsInstance(result.num_bins, int)

    def test_values(self):
        result = MultiBandSpectrogram(DATA_PATH + '/sample.wav', [200, 1000])
        self.assertTrue(isinstance(result.filterbank, Filterbank))
        self.assertTrue(result.crossover_frequencies == [200, 1000])
        self.assertTrue(result.norm_bands is False)
        self.assertTrue(result.shape == (281, 3))
        # properties
        self.assertTrue(result.num_frames == 281)
        self.assertTrue(result.num_bins == 3)
        # self.assertTrue(result.bin_frequencies == [])


class TestMultiBandSpectrogramProcessorClass(unittest.TestCase):

    def test_types(self):
        processor = MultiBandSpectrogramProcessor([200, 1000])
        self.assertIsInstance(processor, MultiBandSpectrogramProcessor)
        self.assertIsInstance(processor, Processor)
        self.assertTrue(type(processor.crossover_frequencies) == list)
        self.assertTrue(type(processor.norm_bands) == bool)

    def test_values(self):
        processor = MultiBandSpectrogramProcessor([200, 1000], norm_bands=True)
        self.assertTrue(processor.crossover_frequencies == [200, 1000])
        self.assertTrue(processor.norm_bands is True)
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertIsInstance(result, MultiBandSpectrogram)
        self.assertTrue(result.shape == (281, 3))
        # properties
        self.assertTrue(result.num_frames == 281)
        self.assertTrue(result.num_bins == 3)
        # test other values
        processor = MultiBandSpectrogramProcessor([500])
        self.assertTrue(processor.crossover_frequencies == [500])
        self.assertTrue(processor.norm_bands is False)
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertIsInstance(result, MultiBandSpectrogram)
        self.assertTrue(result.shape == (281, 2))
        # properties
        self.assertTrue(result.num_frames == 281)
        self.assertTrue(result.num_bins == 2)


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
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 58))
        processor = StackedSpectrogramProcessor([1024], spec_processor)
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 69))
        processor = StackedSpectrogramProcessor([2048], spec_processor)
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 81))
        processor = StackedSpectrogramProcessor([512, 1024, 2048],
                                                spec_processor)
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 58 + 69 + 81))

    def test_stack_diffs(self):
        # also include the differences
        spec_processor = LogarithmicFilteredSpectrogramProcessor()
        processor = StackedSpectrogramProcessor([512], spec_processor,
                                                stack_diffs=True)
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 116))
        processor = StackedSpectrogramProcessor([1024], spec_processor,
                                                stack_diffs=True)
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 138))
        processor = StackedSpectrogramProcessor([2048], spec_processor,
                                                stack_diffs=True)
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 162))
        processor = StackedSpectrogramProcessor([512, 1024, 2048],
                                                spec_processor,
                                                stack_diffs=True)
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 116 + 138 + 162))

    def test_stack_depth(self):
        # stack in depth
        spec_processor = LogarithmicFilteredSpectrogramProcessor(
            duplicate_filters=True)
        processor = StackedSpectrogramProcessor([512], spec_processor,
                                                stack='depth')
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 108, 1))
        processor = StackedSpectrogramProcessor([1024], spec_processor,
                                                stack='depth')
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 108, 1))
        processor = StackedSpectrogramProcessor([2048], spec_processor,
                                                stack='depth')
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 108, 1))
        processor = StackedSpectrogramProcessor([512, 1024, 2048],
                                                spec_processor,
                                                stack='depth')
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 108, 3))
