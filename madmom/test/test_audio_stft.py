# encoding: utf-8
"""
This file contains tests for the madmom.audio.spectrogram module.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""
# pylint: skip-file

import unittest
import cPickle

from . import DATA_PATH
from madmom.audio.stft import *
from madmom.audio.spectrogram import Spectrogram
from madmom.audio.signal import FramedSignal


# test functions

# noinspection PyArgumentList
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


# noinspection PyArgumentList,PyArgumentList,PyArgumentList
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


# test classes
class ShortTimeFourierTransformClass(unittest.TestCase):

    def test_types(self):
        result = ShortTimeFourierTransform(DATA_PATH + '/sample.wav')
        self.assertIsInstance(result, ShortTimeFourierTransform)
        self.assertIsInstance(result, np.ndarray)
        self.assertIsInstance(result.frames, FramedSignal)
        self.assertIsInstance(result.window, np.ndarray)
        self.assertIsInstance(result.fft_window, np.ndarray)
        self.assertIsInstance(result.fft_size, int)
        self.assertIsInstance(result.circular_shift, bool)
        # properties
        self.assertIsInstance(result.num_frames, int)
        self.assertIsInstance(result.bin_frequencies, np.ndarray)
        self.assertIsInstance(result.num_bins, int)

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

    def test_pickle(self):
        # test with non-default values
        result = ShortTimeFourierTransform(DATA_PATH + '/sample.wav',
                                           window=np.hamming, fft_size=4096,
                                           circular_shift=True)
        dump = cPickle.dumps(result, protocol=cPickle.HIGHEST_PROTOCOL)
        dump = cPickle.loads(dump)
        self.assertTrue(np.allclose(result, dump))
        # additional attributes
        self.assertTrue(np.allclose(result.window, dump.window))
        self.assertTrue(np.allclose(result.fft_window, dump.fft_window))
        self.assertTrue(result.fft_size == dump.fft_size)
        self.assertTrue(result.circular_shift == dump.circular_shift)

    def test_methods(self):
        result = ShortTimeFourierTransform(DATA_PATH + '/sample.wav')
        self.assertIsInstance(result.spec(), Spectrogram)
        self.assertIsInstance(result.phase(), Phase)


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


class PhaseClass(unittest.TestCase):

    def test_types(self):
        result = Phase(DATA_PATH + '/sample.wav')
        self.assertIsInstance(result, Phase)
        self.assertIsInstance(result, np.ndarray)
        self.assertIsInstance(result.stft, ShortTimeFourierTransform)
        self.assertIsInstance(result.frames, FramedSignal)
        # properties
        self.assertIsInstance(result.num_frames, int)
        self.assertIsInstance(result.bin_frequencies, np.ndarray)
        self.assertIsInstance(result.num_bins, int)

    def test_values(self):
        result = Phase(DATA_PATH + '/sample.wav')
        # properties
        self.assertTrue(result.num_frames == 281)
        self.assertTrue(np.allclose(result.bin_frequencies,
                                    fft_frequencies(1024, 44100)))
        self.assertTrue(result.num_bins == 1024)
        self.assertTrue(result.shape == (281, 1024))

    def test_pickle(self):
        result = Phase(DATA_PATH + '/sample.wav')
        dump = cPickle.dumps(result, protocol=cPickle.HIGHEST_PROTOCOL)
        dump = cPickle.loads(dump)
        self.assertTrue(np.allclose(result, dump))

    def test_methods(self):
        result = Phase(DATA_PATH + '/sample.wav')
        self.assertIsInstance(result.local_group_delay(), LocalGroupDelay)
        self.assertIsInstance(result.lgd(), LocalGroupDelay)

    def test_warnings(self):
        # TODO: write a test which catches the warning about the circular_shift
        pass


class LocalGroupDelayClass(unittest.TestCase):

    def test_types(self):
        result = LocalGroupDelay(DATA_PATH + '/sample.wav')
        self.assertIsInstance(result, LocalGroupDelay)
        self.assertIsInstance(result, np.ndarray)
        self.assertIsInstance(result.phase, Phase)
        self.assertIsInstance(result.stft, ShortTimeFourierTransform)
        self.assertIsInstance(result.frames, FramedSignal)
        # properties
        self.assertIsInstance(result.num_frames, int)
        self.assertIsInstance(result.bin_frequencies, np.ndarray)
        self.assertIsInstance(result.num_bins, int)

    def test_values(self):
        result = LocalGroupDelay(DATA_PATH + '/sample.wav')
        # properties
        self.assertTrue(result.num_frames == 281)
        self.assertTrue(np.allclose(result.bin_frequencies,
                                    fft_frequencies(1024, 44100)))
        self.assertTrue(result.num_bins == 1024)
        self.assertTrue(result.shape == (281, 1024))

    def test_pickle(self):
        result = LocalGroupDelay(DATA_PATH + '/sample.wav')
        dump = cPickle.dumps(result, protocol=cPickle.HIGHEST_PROTOCOL)
        dump = cPickle.loads(dump)
        self.assertTrue(np.allclose(result, dump))
