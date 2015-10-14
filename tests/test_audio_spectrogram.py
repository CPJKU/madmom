# encoding: utf-8
"""
This file contains tests for the madmom.audio.spectrogram module.

"""
# pylint: skip-file

import unittest
import cPickle

from . import DATA_PATH
from madmom.audio.spectrogram import *
from madmom.audio.signal import FramedSignal
from madmom.audio.filters import Filterbank, LogarithmicFilterbank


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


class TestSpectrogramClass(unittest.TestCase):

    def test_types(self):
        result = Spectrogram(DATA_PATH + '/sample.wav')
        self.assertIsInstance(result, Spectrogram)
        self.assertIsInstance(result.frames, FramedSignal)
        self.assertIsInstance(result.stft, ShortTimeFourierTransform)
        # properties
        self.assertIsInstance(result.num_frames, int)
        self.assertIsInstance(result.bin_frequencies, np.ndarray)
        self.assertIsInstance(result.num_bins, int)
        # other faked attributes
        self.assertTrue(result.filterbank is None)
        self.assertTrue(result.mul is None)
        self.assertTrue(result.add is None)

    def test_values(self):
        # from file
        result = Spectrogram(DATA_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 1024))
        self.assertTrue(result.num_frames == 281)
        self.assertTrue(result.num_bins == 1024)
        # from spec
        self.assertTrue(np.allclose(Spectrogram(result), result))
        # from stft
        stft = ShortTimeFourierTransform(DATA_PATH + '/sample.wav')
        self.assertTrue(np.allclose(Spectrogram(stft), result))

    def test_pickle(self):
        result = Spectrogram(DATA_PATH + '/sample.wav')
        dump = cPickle.dumps(result, protocol=cPickle.HIGHEST_PROTOCOL)
        dump = cPickle.loads(dump)
        self.assertTrue(np.allclose(result, dump))

    def test_methods(self):
        result = Spectrogram(DATA_PATH + '/sample.wav')
        self.assertIsInstance(result.diff(), SpectrogramDifference)
        self.assertIsInstance(result.filter(), FilteredSpectrogram)
        self.assertIsInstance(result.log(), LogarithmicSpectrogram)


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


class TestFilteredSpectrogramClass(unittest.TestCase):

    def test_types(self):
        result = FilteredSpectrogram(DATA_PATH + '/sample.wav')
        self.assertIsInstance(result, FilteredSpectrogram)
        self.assertIsInstance(result, Spectrogram)
        self.assertIsInstance(result.stft, ShortTimeFourierTransform)
        self.assertIsInstance(result.frames, FramedSignal)
        self.assertIsInstance(result.filterbank, LogarithmicFilterbank)
        # properties
        self.assertIsInstance(result.num_frames, int)
        self.assertIsInstance(result.bin_frequencies, np.ndarray)
        self.assertIsInstance(result.num_bins, int)
        # other faked attributes
        self.assertTrue(result.mul is None)
        self.assertTrue(result.add is None)
        # wrong filterbank type
        with self.assertRaises(TypeError):
            FilteredSpectrogram(DATA_PATH + '/sample.wav', filterbank='bla')

    def test_values(self):
        # from file
        result = FilteredSpectrogram(DATA_PATH + '/sample.wav')
        self.assertTrue(result.num_bins == 81)
        self.assertTrue(result.num_frames == 281)
        # with given filterbank
        result = FilteredSpectrogram(DATA_PATH + '/sample.wav',
                                     filterbank=result.filterbank)
        self.assertTrue(result.num_bins == 81)
        self.assertTrue(result.num_frames == 281)

    def test_pickle(self):
        # test with non-default values
        from madmom.audio.filters import MelFilterbank
        result = FilteredSpectrogram(DATA_PATH + '/sample.wav',
                                     filterbank=MelFilterbank)
        dump = cPickle.dumps(result, protocol=cPickle.HIGHEST_PROTOCOL)
        dump = cPickle.loads(dump)
        self.assertTrue(np.allclose(result, dump))
        # additional attributes
        self.assertTrue(np.allclose(result.filterbank, dump.filterbank))

    def test_methods(self):
        result = FilteredSpectrogram(DATA_PATH + '/sample.wav')
        self.assertIsInstance(result.diff(), SpectrogramDifference)
        # TODO: should we return a LogarithmicFilteredSpectrogram?
        self.assertIsInstance(result.log(), LogarithmicSpectrogram)


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
        self.assertIsInstance(result, Spectrogram)
        self.assertIsInstance(result.stft, ShortTimeFourierTransform)
        self.assertIsInstance(result.frames, FramedSignal)
        self.assertIsInstance(result.mul, float)
        self.assertIsInstance(result.add, float)
        # properties
        self.assertIsInstance(result.num_frames, int)
        self.assertIsInstance(result.bin_frequencies, np.ndarray)
        self.assertIsInstance(result.num_bins, int)
        # other faked attributes
        self.assertTrue(result.filterbank is None)

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

    def test_pickle(self):
        # test with non-default values
        result = LogarithmicSpectrogram(DATA_PATH + '/sample.wav',
                                        mul=2, add=2)
        dump = cPickle.dumps(result, protocol=cPickle.HIGHEST_PROTOCOL)
        dump = cPickle.loads(dump)
        self.assertTrue(np.allclose(result, dump))
        self.assertTrue(result.mul == dump.mul)
        self.assertTrue(result.add == dump.add)

    def test_methods(self):
        result = LogarithmicSpectrogram(DATA_PATH + '/sample.wav')
        self.assertIsInstance(result.diff(), SpectrogramDifference)
        self.assertIsInstance(result.filter(), FilteredSpectrogram)


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


class TestLogarithmicFilteredSpectrogramClass(unittest.TestCase):

    def test_types(self):
        result = LogarithmicFilteredSpectrogram(DATA_PATH + '/sample.wav')
        self.assertIsInstance(result, LogarithmicFilteredSpectrogram)
        self.assertIsInstance(result, Spectrogram)
        self.assertIsInstance(result.stft, ShortTimeFourierTransform)
        self.assertIsInstance(result.frames, FramedSignal)
        self.assertIsInstance(result.filterbank, Filterbank)
        self.assertIsInstance(result.filterbank, LogarithmicFilterbank)
        self.assertIsInstance(result.mul, float)
        self.assertIsInstance(result.add, float)
        # properties
        self.assertIsInstance(result.num_frames, int)
        self.assertIsInstance(result.bin_frequencies, np.ndarray)
        self.assertIsInstance(result.num_bins, int)

    def test_values(self):
        result = LogarithmicFilteredSpectrogram(DATA_PATH + '/sample.wav')
        self.assertTrue(result.mul == 1)
        self.assertTrue(result.add == 1)
        # properties
        self.assertTrue(result.num_frames == 281)
        self.assertTrue(result.num_bins == 81)
        self.assertTrue(result.shape == (281, 81))
        # test other values
        result = LogarithmicFilteredSpectrogram(DATA_PATH + '/sample.wav',
                                                mul=2, add=2)
        self.assertTrue(result.mul == 2)
        self.assertTrue(result.add == 2)

    def test_pickle(self):
        # test with non-default values
        result = LogarithmicFilteredSpectrogram(DATA_PATH + '/sample.wav',
                                                mul=2, add=2)
        dump = cPickle.dumps(result, protocol=cPickle.HIGHEST_PROTOCOL)
        dump = cPickle.loads(dump)
        self.assertTrue(np.allclose(result, dump))
        self.assertTrue(np.allclose(result.filterbank, dump.filterbank))
        self.assertTrue(result.mul == dump.mul)
        self.assertTrue(result.add == dump.add)


class TestLogarithmicFilteredSpectrogramProcessorClass(unittest.TestCase):

    def test_types(self):
        processor = LogarithmicFilteredSpectrogramProcessor()
        self.assertIsInstance(processor,
                              LogarithmicFilteredSpectrogramProcessor)
        self.assertTrue(processor.filterbank == LogarithmicFilterbank)
        self.assertIsInstance(processor.num_bands, int)
        self.assertIsInstance(processor.fmin, float)
        self.assertIsInstance(processor.fmax, float)
        self.assertIsInstance(processor.fref, float)
        self.assertIsInstance(processor.norm_filters, bool)
        self.assertIsInstance(processor.unique_filters, bool)
        self.assertIsInstance(processor.mul, float)
        self.assertIsInstance(processor.add, float)

    def test_values(self):
        processor = LogarithmicFilteredSpectrogramProcessor()
        self.assertTrue(processor.mul == 1)
        self.assertTrue(processor.add == 1)
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 81))


class TestSpectrogramDifferenceClass(unittest.TestCase):

    def test_types(self):
        result = SpectrogramDifference(DATA_PATH + '/sample.wav')
        self.assertIsInstance(result, SpectrogramDifference)
        self.assertIsInstance(result, Spectrogram)
        self.assertIsInstance(result.stft, ShortTimeFourierTransform)
        self.assertIsInstance(result.frames, FramedSignal)
        self.assertIsInstance(result.diff_ratio, float)
        self.assertIsInstance(result.diff_frames, int)
        self.assertTrue(result.diff_max_bins is None)
        self.assertIsInstance(result.positive_diffs, bool)
        # properties
        self.assertIsInstance(result.num_frames, int)
        self.assertIsInstance(result.bin_frequencies, np.ndarray)
        self.assertIsInstance(result.num_bins, int)
        # other faked attributes
        self.assertTrue(result.filterbank is None)
        self.assertTrue(result.mul is None)
        self.assertTrue(result.add is None)

    def test_values(self):
        result = SpectrogramDifference(DATA_PATH + '/sample.wav')
        self.assertTrue(result.diff_ratio == 0.5)
        self.assertTrue(result.diff_frames == 1)
        self.assertTrue(result.diff_max_bins is None)
        self.assertTrue(result.positive_diffs is False)
        # properties
        self.assertTrue(result.num_frames == 281)
        self.assertTrue(result.num_bins == 1024)
        self.assertTrue(result.shape == (281, 1024))
        # methods
        self.assertTrue(result.positive_diff().min() == 0)

    def test_pickle(self):
        # test with non-default values
        result = SpectrogramDifference(DATA_PATH + '/sample.wav',
                                       diff_ratio=0.7, diff_frames=3,
                                       diff_max_bins=2, positive_diffs=True)
        dump = cPickle.dumps(result, protocol=cPickle.HIGHEST_PROTOCOL)
        dump = cPickle.loads(dump)
        self.assertTrue(np.allclose(result, dump))
        self.assertTrue(result.diff_ratio == dump.diff_ratio)
        self.assertTrue(result.diff_frames == dump.diff_frames)
        self.assertTrue(result.diff_max_bins == dump.diff_max_bins)
        self.assertTrue(result.positive_diffs == dump.positive_diffs)


class TestSpectrogramDifferenceProcessorClass(unittest.TestCase):

    def test_types(self):
        processor = SpectrogramDifferenceProcessor()
        self.assertIsInstance(processor, SpectrogramDifferenceProcessor)
        self.assertIsInstance(processor.diff_ratio, float)
        self.assertTrue(processor.diff_frames is None)
        self.assertTrue(processor.diff_max_bins is None)
        self.assertIsInstance(processor.positive_diffs, bool)

    def test_values(self):
        processor = SpectrogramDifferenceProcessor(diff_frames=2,
                                                   diff_max_bins=3)
        self.assertTrue(processor.diff_ratio == 0.5)
        self.assertTrue(processor.diff_frames == 2)
        self.assertTrue(processor.diff_max_bins == 3)
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 1024))
        self.assertTrue(np.sum(result[:2]) == 0)
        self.assertTrue(np.min(result) <= 0)
        # positive diffs
        processor = SpectrogramDifferenceProcessor(positive_diffs=True)
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(np.min(result) >= 0)


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

    def test_pickle(self):
        # test with non-default values
        result = MultiBandSpectrogram(DATA_PATH + '/sample.wav', [200, 1000])
        dump = cPickle.dumps(result, protocol=cPickle.HIGHEST_PROTOCOL)
        dump = cPickle.loads(dump)
        self.assertTrue(np.allclose(result, dump))
        self.assertTrue(result.crossover_frequencies ==
                        dump.crossover_frequencies)
        self.assertTrue(np.allclose(result.filterbank, dump.filterbank))
        self.assertTrue(result.norm_bands == dump.norm_bands)


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
        diff_processor = SpectrogramDifferenceProcessor()
        processor = StackedSpectrogramProcessor([512], spec_processor,
                                                diff_processor)
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 116))
        processor = StackedSpectrogramProcessor([1024], spec_processor,
                                                diff_processor)
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 138))
        processor = StackedSpectrogramProcessor([2048], spec_processor,
                                                diff_processor)
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 162))
        processor = StackedSpectrogramProcessor([512, 1024, 2048],
                                                spec_processor,
                                                diff_processor)
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(result.shape == (281, 116 + 138 + 162))

    def test_stack_depth(self):
        # stack in depth
        spec_processor = LogarithmicFilteredSpectrogramProcessor(
            unique_filters=False)
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
