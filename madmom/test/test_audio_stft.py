# encoding: utf-8
"""
This file contains tests for the madmom.audio.stft module.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""
# pylint: skip-file

import unittest
import cPickle

from . import DATA_PATH
from madmom.audio.stft import *
from madmom.audio.spectrogram import Spectrogram
from madmom.audio.signal import FramedSignal

sig_2d = np.array([[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                   [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                   [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]])


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


class TestStftFunction(unittest.TestCase):

    def test_types(self):
        result = stft(np.arange(10).reshape(5, 2))
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.complex64)

    def test_dimensionality(self):
        # input must be 2D
        with self.assertRaises(ValueError):
            stft(np.arange(10))
        # like this:
        result = stft(np.arange(10).reshape(5, 2))
        self.assertEqual(result.shape, (5, 1))
        # window size must match frame size
        with self.assertRaises(ValueError):
            stft(np.arange(10).reshape(5, 2), window=[1, 2, 3])
        # fft size must be greater or equal frame size
        with self.assertRaises(ValueError):
            stft(np.arange(10).reshape(5, 2), fft_size=1)

    def test_value(self):
        result = stft(sig_2d)
        # signal length and FFT size = 12
        # fft_freqs: 0, 1/12, 2/12, 3/12, 4/12, 5/12
        # [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0] every 4th bin => 3/12
        res = [3.+0.j, 0.+0.j, 0.-0.j, 3+0.j, 0.+0.j, 0.+0.j]
        self.assertTrue(np.allclose(result[0], res))
        # [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0] every erd bin => 4/12
        res = [4.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 4.+0.j, 0.+0.j]
        self.assertTrue(np.allclose(result[1], res))
        # [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0] every 2nd bin => 6/12
        # can't resolve any more
        res = [6.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]
        self.assertTrue(np.allclose(result[2], res))

    def test_circular_shift(self):
        result = stft(sig_2d, circular_shift=True)
        # signal length and FFT size = 12
        # fft_freqs: 0, 1/12, 2/12, 3/12, 4/12, 5/12
        # [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0] every 4th bin => 3/12
        res = [3.+0.j, 0.+0.j, 0.+0j, -3.+0.j, 0.+0.j, 0.+0.j]
        self.assertTrue(np.allclose(result[0], res))
        # [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0] every erd bin => 4/12
        res = [4.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 4.+0.j, 0.+0.j]
        self.assertTrue(np.allclose(result[1], res))
        # [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0] every 2nd bin => 6/12
        # can't resolve any more
        res = [6.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]
        self.assertTrue(np.allclose(result[2], res))


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

    def test_types(self):
        result = local_group_delay(np.random.rand(10, 2))
        self.assertTrue(result.dtype == np.float)
        self.assertTrue(result.shape == (10, 2))
        with self.assertRaises(ValueError):
            local_group_delay(np.arange(10))
        with self.assertRaises(ValueError):
            local_group_delay(np.arange(20).reshape(5, 2, 2))

    def test_values(self):
        data = np.arange(20).reshape(10, 2) * 2
        correct = np.tile([-2, 0], 10).reshape(10, 2)
        self.assertTrue(np.allclose(correct, local_group_delay(data)))
        data = np.arange(20).reshape(10, 2) * 4
        correct = np.tile([2.28318531, 0], 10).reshape(10, 2)
        self.assertTrue(np.allclose(correct, local_group_delay(data)))


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
        # from STFT
        self.assertTrue(np.allclose(ShortTimeFourierTransform(result), result))

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

    def test_fft_window(self):
        # use a signal
        from madmom.audio.signal import Signal
        signal = Signal(DATA_PATH + '/sample.wav')
        # scale the signal to float and range -1..1
        scaling = float(np.iinfo(signal.dtype).max)
        scaled_signal = signal / scaling
        # calculate the STFTs of both signals
        result = ShortTimeFourierTransform(signal)
        scaled_result = ShortTimeFourierTransform(scaled_signal)
        # both STFTs must be the same
        self.assertTrue(np.allclose(result, scaled_result))
        # if now window is given, a uniformly distributed one should be used
        result = ShortTimeFourierTransform(signal, window=None)
        self.assertTrue(np.allclose(result.fft_window,
                                    np.ones(2048, dtype=float) / scaling))
        scaled_result = ShortTimeFourierTransform(scaled_signal, window=None)
        self.assertTrue(scaled_result.fft_window is None)


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
