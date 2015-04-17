# encoding: utf-8
"""
This file contains tests for the madmom.audio.signal module.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""
# pylint: skip-file

import unittest
import __builtin__

from madmom.audio.signal import *

from . import DATA_PATH
from .test_audio_comb_filters import sig_1d, sig_2d


# test signal functions
class TestSmoothFunction(unittest.TestCase):

    def test_types_1d(self):
        result = smooth(sig_1d, None)
        self.assertTrue(type(result) == type(sig_1d))
        self.assertTrue(len(result) == len(sig_1d))
        self.assertTrue(result.shape == sig_1d.shape)
        result = smooth(sig_1d, 3)
        self.assertTrue(type(result) == type(sig_1d))
        self.assertTrue(len(result) == len(sig_1d))
        self.assertTrue(result.shape == sig_1d.shape)

    def test_types_2d(self):
        result = smooth(sig_2d, None)
        self.assertTrue(type(result) == type(sig_2d))
        self.assertTrue(len(result) == len(sig_2d))
        self.assertTrue(result.shape == sig_2d.shape)
        result = smooth(sig_2d, 3)
        self.assertTrue(type(result) == type(sig_2d))
        self.assertTrue(len(result) == len(sig_2d))
        self.assertTrue(result.shape == sig_2d.shape)

    def test_errors(self):
        with self.assertRaises(ValueError):
            smooth(np.zeros(9).reshape(3, 3), 'bla')

    def test_values_1d(self):
        result = smooth(sig_1d, None)
        self.assertTrue(np.allclose(result, sig_1d))
        result = smooth(sig_1d, 0)
        self.assertTrue(np.allclose(result, sig_1d))
        result = smooth(sig_1d, 3)
        result_3 = [0, 0.08, 1, 0.08, 0.08, 1, 0.08, 0.08, 1]
        self.assertTrue(np.allclose(result, result_3))
        result = smooth(sig_1d, 5)
        result_5 = [0.08, 0.54, 1, 0.62, 0.62, 1, 0.62, 0.62, 1]
        self.assertTrue(np.allclose(result, result_5))
        result = smooth(sig_1d, 7)
        result_7 = [0.31, 0.77, 1.08, 1.08, 1.08, 1.16, 1.08, 1.08, 1.08]
        self.assertTrue(np.allclose(result, result_7))
        result = smooth(sig_1d, np.ones(3))
        result_3_ones = [0, 1, 1, 1, 1, 1, 1, 1, 1]
        self.assertTrue(np.allclose(result, result_3_ones))
        result = smooth(sig_1d, np.ones(4))
        result_4_ones = [0, 1, 1, 1, 2, 1, 1, 2, 1]
        self.assertTrue(np.allclose(result, result_4_ones))

    def test_values_2d(self):
        result = smooth(sig_2d, None)
        self.assertTrue(np.allclose(result, sig_2d))
        result = smooth(sig_2d, 3)
        result_3 = [[0, 0.08, 1, 0.08, 0.08, 1, 0.08, 0.08, 1],
                    [1, 0.16, 1, 0.16, 1, 0.16, 1, 0.16, 1]]
        self.assertTrue(np.allclose(result, np.asarray(result_3).T))
        result = smooth(sig_2d, 5)
        result_5 = [[0.08, 0.54, 1, 0.62, 0.62, 1, 0.62, 0.62, 1],
                    [1.08, 1.08, 1.16, 1.08, 1.16, 1.08, 1.16, 1.08, 1.08]]
        self.assertTrue(np.allclose(result, np.asarray(result_5).T))
        result = smooth(sig_2d, 7)
        result_7 = [[0.31, 0.77, 1.08, 1.08, 1.08, 1.16, 1.08, 1.08, 1.08],
                    [1.31, 1.62, 1.62, 1.7, 1.62, 1.7, 1.62, 1.62, 1.31]]
        self.assertTrue(np.allclose(result, np.asarray(result_7).T))


class TestAttenuateFunction(unittest.TestCase):

    def test_types_1d(self):
        result = attenuate(sig_1d, 0)
        self.assertTrue(type(result) == type(sig_1d))
        self.assertTrue(len(result) == len(sig_1d))
        self.assertTrue(result.shape == sig_1d.shape)
        self.assertTrue(result.dtype == sig_1d.dtype)

    def test_types_2d(self):
        result = attenuate(sig_2d, 0)
        self.assertTrue(type(result) == type(sig_2d))
        self.assertTrue(len(result) == len(sig_2d))
        self.assertTrue(result.shape == sig_2d.shape)
        self.assertTrue(result.dtype == sig_2d.dtype)

    def test_types_1d_int_dtype(self):
        result = attenuate(sig_1d.astype(np.int16), 0)
        self.assertTrue(len(result) == len(sig_1d))
        self.assertTrue(result.shape == sig_1d.shape)
        self.assertTrue(result.dtype == np.int16)

    def test_types_2d_int_dtype(self):
        result = attenuate(sig_2d.astype(np.int), 0)
        self.assertTrue(len(result) == len(sig_2d))
        self.assertTrue(result.shape == sig_2d.shape)
        self.assertTrue(result.dtype == np.int)

    def test_types_signal(self):
        signal = Signal(DATA_PATH + '/sample.wav')
        result = attenuate(signal, 0)
        self.assertIsInstance(result, Signal)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(result.dtype == np.int16)

    def test_values_1d(self):
        result = attenuate(sig_1d, 0)
        self.assertTrue(np.allclose(result, sig_1d))
        result = attenuate(sig_1d, 10)
        self.assertTrue(np.allclose(result, 0.31622777 * sig_1d))

    def test_values_2d(self):
        result = attenuate(sig_2d, 0)
        self.assertTrue(np.allclose(result, sig_2d))
        result = attenuate(sig_2d, 3)
        self.assertTrue(np.allclose(result, 0.70794578 * sig_2d))

    def test_values_1d_int_dtype(self):
        result = attenuate(sig_1d.astype(np.int), 0)
        self.assertTrue(np.allclose(result, sig_1d.astype(np.int)))
        result = attenuate(sig_1d.astype(np.int), 5)
        self.assertTrue(np.allclose(result, 0 * sig_1d))

    def test_values_2d_int_dtype(self):
        result = attenuate(sig_2d.astype(np.int), 0)
        self.assertTrue(np.allclose(result, sig_2d))
        result = attenuate(sig_2d.astype(np.int), 1)
        self.assertTrue(np.allclose(result, 0 * sig_2d))


class TestNormalizeFunction(unittest.TestCase):

    def test_types_1d(self):
        result = normalize(sig_1d)
        self.assertTrue(result.dtype == float)
        self.assertTrue(len(result) == len(sig_1d))
        self.assertTrue(result.shape == sig_1d.shape)
        self.assertTrue(result.dtype == sig_1d.dtype)

    def test_types_2d(self):
        result = normalize(sig_2d)
        self.assertTrue(result.dtype == float)
        self.assertTrue(len(result) == len(sig_2d))
        self.assertTrue(result.shape == sig_2d.shape)
        self.assertTrue(result.dtype == sig_2d.dtype)

    def test_types_1d_int_dtype(self):
        result = normalize(sig_1d.astype(np.int16))
        self.assertTrue(len(result) == len(sig_1d))
        self.assertTrue(result.shape == sig_1d.shape)
        self.assertTrue(result.dtype == np.float)

    def test_types_2d_int_dtype(self):
        result = normalize(sig_2d.astype(np.int))
        self.assertTrue(len(result) == len(sig_2d))
        self.assertTrue(result.shape == sig_2d.shape)
        self.assertTrue(result.dtype == np.float)

    def test_types_signal(self):
        signal = Signal(DATA_PATH + '/sample.wav')
        result = normalize(signal)
        self.assertIsInstance(result, Signal)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(result.dtype == np.float)

    def test_values_1d(self):
        result = normalize(sig_1d)
        self.assertTrue(np.allclose(result, sig_1d))
        result = normalize(sig_1d * 0.5)
        self.assertTrue(np.allclose(result, sig_1d))

    def test_values_2d(self):
        result = normalize(sig_2d)
        self.assertTrue(np.allclose(result, sig_2d))
        result = normalize(sig_2d * 0.5)
        self.assertTrue(np.allclose(result, sig_2d))

    def test_values_1d_int_dtype(self):
        result = normalize(10 * sig_1d.astype(np.int16))
        self.assertTrue(np.allclose(result, sig_1d))

    def test_values_2d_int_dtype(self):
        result = normalize(3 * sig_2d.astype(np.int))
        self.assertTrue(np.allclose(result, sig_2d))


class TestMixFunction(unittest.TestCase):

    mono_2d = np.asarray([0.5, 0, 1, 0, 0.5, 0.5, 0.5, 0, 1], dtype=np.float)

    def test_types_1d(self):
        result = remix(sig_1d, 1)
        self.assertTrue(len(result) == len(sig_1d))
        self.assertTrue(result.shape == sig_1d.shape)
        self.assertTrue(result.dtype == sig_1d.dtype)
        result = remix(sig_1d, 2)
        self.assertTrue(len(result) == len(sig_1d))
        self.assertTrue(result.shape == (len(sig_1d), 2))
        self.assertTrue(result.dtype == sig_1d.dtype)
        result = remix(sig_1d, 3)
        self.assertTrue(len(result) == len(sig_1d))
        self.assertTrue(result.shape == (len(sig_1d), 3))
        self.assertTrue(result.dtype == sig_1d.dtype)

    def test_types_2d(self):
        result = remix(sig_2d, 1)
        self.assertTrue(len(result) == len(sig_2d))
        self.assertTrue(result.shape == sig_1d.shape)
        self.assertTrue(result.dtype == sig_2d.dtype)
        result = remix(sig_2d, 2)
        self.assertTrue(len(result) == len(sig_2d))
        self.assertTrue(result.shape == sig_2d.shape)
        self.assertTrue(result.dtype == sig_2d.dtype)
        with self.assertRaises(NotImplementedError):
            remix(sig_2d, 3)

    def test_types_1d_int_dtype(self):
        result = remix(sig_1d.astype(np.int), 1)
        self.assertTrue(len(result) == len(sig_1d))
        self.assertTrue(result.shape == sig_1d.shape)
        self.assertTrue(result.dtype == np.int)
        result = remix(sig_1d.astype(np.int), 2)
        self.assertTrue(len(result) == len(sig_1d))
        self.assertTrue(result.shape == (len(sig_1d), 2))
        self.assertTrue(result.dtype == np.int)

    def test_types_2d_int_dtype(self):
        result = remix(sig_2d.astype(np.int), 1)
        self.assertTrue(len(result) == len(sig_2d))
        self.assertTrue(result.shape == sig_1d.shape)
        self.assertTrue(result.dtype == np.int)
        result = remix(sig_2d.astype(np.int), 2)
        self.assertTrue(len(result) == len(sig_2d))
        self.assertTrue(result.shape == sig_2d.shape)
        self.assertTrue(result.dtype == np.int)

    def test_types_signal(self):
        signal = Signal(DATA_PATH + '/sample.wav')
        result = remix(signal, 1)
        self.assertTrue(isinstance(result, Signal))
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertTrue(result.dtype == np.int16)

    def test_types_stereo_signal(self):
        signal = Signal(DATA_PATH + '/stereo_sample.wav')
        result = remix(signal, 1)
        self.assertTrue(isinstance(result, Signal))
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertTrue(result.dtype == np.int16)

    def test_values_1d(self):
        result = remix(sig_1d, 1)
        self.assertTrue(np.allclose(result, sig_1d))

    def test_values_2d(self):
        result = remix(sig_2d, 1)
        self.assertTrue(np.allclose(result, self.mono_2d))

    def test_values_2d_int_dtype(self):
        result = remix(sig_2d.astype(np.int), 1)
        self.assertTrue(np.allclose(result, self.mono_2d.astype(np.int)))

    def test_values_2d_double_range(self):
        result = remix(2 * sig_2d.astype(np.int), 1)
        self.assertTrue(np.allclose(result, 2 * self.mono_2d))


class TestTrimFunction(unittest.TestCase):

    def test_types_1d(self):
        result = trim(sig_1d)
        self.assertTrue(type(result) == type(sig_1d))
        self.assertTrue(len(result) == len(sig_1d) - 2)
        self.assertTrue(result.ndim == sig_1d.ndim)

    def test_types_signal(self):
        signal = Signal(DATA_PATH + '/sample.wav')
        result = trim(signal)
        self.assertIsInstance(result, Signal)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(result.dtype == np.int16)

    def test_values_1d(self):
        result = trim(sig_1d)
        trimmed_1d = [1, 0, 0, 1, 0, 0, 1]
        self.assertTrue(np.allclose(result, trimmed_1d))

    def test_types_2d(self):
        with self.assertRaises(NotImplementedError):
            trim(sig_2d)

    def test_values_2d(self):
        with self.assertRaises(NotImplementedError):
            trim(sig_2d)


class TestRootMeanSquareFunction(unittest.TestCase):

    def test_types_1d(self):
        result = root_mean_square(sig_1d)
        self.assertIsInstance(result, float)

    def test_values_1d(self):
        result = root_mean_square(sig_1d)
        rms_1d = 0.57735026919
        self.assertTrue(np.allclose(result, rms_1d))
        result = root_mean_square(np.zeros(100))
        self.assertTrue(np.allclose(result, 0))

    def test_types_2d(self):
        with self.assertRaises(NotImplementedError):
            root_mean_square(sig_2d)

    def test_values_2d(self):
        with self.assertRaises(NotImplementedError):
            root_mean_square(sig_2d)


class TestSoundPressureLevelFunction(unittest.TestCase):

    def test_types_1d(self):
        result = sound_pressure_level(sig_1d)
        self.assertIsInstance(result, float)

    def test_values_1d(self):
        result = sound_pressure_level(sig_1d)
        spl_1d = -4.7712125472
        self.assertTrue(np.allclose(result, spl_1d))
        result = sound_pressure_level(np.zeros(100))
        self.assertTrue(np.allclose(result, -np.finfo(float).max))

    def test_types_2d(self):
        with self.assertRaises(NotImplementedError):
            sound_pressure_level(sig_2d)

    def test_values_2d(self):
        with self.assertRaises(NotImplementedError):
            sound_pressure_level(sig_2d)


class TestLoadAudioFileFunction(unittest.TestCase):

    def test_types(self):
        signal, sample_rate = load_audio_file(DATA_PATH + '/sample.wav')
        self.assertIsInstance(signal, np.ndarray)
        self.assertTrue(signal.dtype == np.int16)
        self.assertTrue(type(sample_rate) == int)

    def test_file_handle(self):
        file_handle = __builtin__.open(DATA_PATH + '/sample.wav')
        signal, sample_rate = load_audio_file(file_handle)
        self.assertIsInstance(signal, np.ndarray)
        self.assertTrue(signal.dtype == np.int16)
        self.assertTrue(type(sample_rate) == int)
        file_handle.close()

        signal, sample_rate = load_audio_file(DATA_PATH + '/sample.wav')
        self.assertIsInstance(signal, np.ndarray)
        self.assertTrue(signal.dtype == np.int16)
        self.assertTrue(type(sample_rate) == int)

    def test_values(self):
        signal, sample_rate = load_audio_file(DATA_PATH + '/sample.wav')
        self.assertTrue(np.allclose(signal[:5],
                                    [-2494, -2510, -2484, -2678, -2833]))
        self.assertTrue(len(signal) == 123481)
        self.assertTrue(sample_rate == 44100)

    def test_stereo(self):
        signal, sample_rate = load_audio_file(DATA_PATH + '/stereo_sample.flac')
        self.assertTrue(np.allclose(signal[:4],
                                    [[33, 38], [35, 36], [29, 34], [36, 31]]))
        self.assertTrue(len(signal) == 182919)
        self.assertTrue(sample_rate == 44100)
        self.assertTrue(signal.shape == (182919, 2))

    def test_stereo_downmix_wav(self):
        signal, sample_rate = load_audio_file(DATA_PATH + '/stereo_sample.wav',
                                              num_channels=1)
        # TODO: is it a problemm that the results are rounded differently?
        self.assertTrue(np.allclose(signal[:5], [35, 35, 31, 33, 33]))
        self.assertTrue(len(signal) == 182919)
        self.assertTrue(sample_rate == 44100)
        self.assertTrue(signal.shape == (182919, ))

    def test_stereo_two_channels_wav(self):
        f = DATA_PATH + '/stereo_sample.wav'
        signal, sample_rate = load_audio_file(f, num_channels=2)
        self.assertTrue(np.allclose(signal[:4],
                                    [[33, 38], [35, 36], [29, 34], [36, 31]]))
        self.assertTrue(len(signal) == 182919)
        self.assertTrue(sample_rate == 44100)
        self.assertTrue(signal.shape == (182919, 2))

    def test_stereo_downmix_flac(self):
        f = DATA_PATH + '/stereo_sample.flac'
        signal, sample_rate = load_audio_file(f, num_channels=1)
        # TODO: is it a problemm that the results are rounded differently?
        self.assertTrue(np.allclose(signal[:5], [36, 36, 32, 34, 34]))
        self.assertTrue(len(signal) == 182919)
        self.assertTrue(sample_rate == 44100)
        self.assertTrue(signal.shape == (182919, ))

    def test_stereo_resample_downmix_wav(self):
        f = DATA_PATH + '/stereo_sample.wav'
        signal, sample_rate = load_audio_file(f, sample_rate=22050,
                                              num_channels=1)
        self.assertTrue(np.allclose(signal[:5], [36, 33, 34, 35, 33]))
        self.assertTrue(len(signal) == 91460)
        self.assertTrue(sample_rate == 22050)
        self.assertTrue(signal.shape == (91460, ))


# signal classes
class TestSignalClass(unittest.TestCase):

    def test_types_array(self):
        result = Signal(sig_1d)
        self.assertIsInstance(result, Signal)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(result.dtype == np.float)
        self.assertIsInstance(result.num_samples, int)
        self.assertIsInstance(result.sample_rate, type(None))
        self.assertIsInstance(result.num_channels, int)
        self.assertIsInstance(result.length, type(None))

    def test_types_array_with_sample_rate(self):
        result = Signal(sig_1d, 1)
        self.assertIsInstance(result, Signal)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(result.dtype == np.float)
        self.assertIsInstance(result.num_samples, int)
        self.assertIsInstance(result.sample_rate, float)
        self.assertIsInstance(result.num_channels, int)
        self.assertIsInstance(result.length, float)

    def test_types_file(self):
        result = Signal(DATA_PATH + '/sample.wav')
        self.assertIsInstance(result, Signal)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(result.dtype == np.int16)

    def test_values_1d(self):
        result = Signal(sig_1d, 1)
        self.assertTrue(np.allclose(result, sig_1d))
        self.assertTrue(len(result) == 9)
        self.assertTrue(result.num_samples == 9)
        self.assertTrue(result.sample_rate == 1)
        self.assertTrue(result.num_channels == 1)
        self.assertTrue(result.length == 9)

    def test_values_1d_no_sample_rate(self):
        result = Signal(sig_1d)
        self.assertTrue(np.allclose(result, sig_1d))
        self.assertTrue(len(result) == 9)
        self.assertTrue(result.num_samples == 9)
        self.assertTrue(result.sample_rate is None)
        self.assertTrue(result.num_channels == 1)
        self.assertTrue(result.length is None)

    def test_values_2d(self):
        result = Signal(sig_2d, 12.3)
        self.assertTrue(np.allclose(result, sig_2d))
        self.assertTrue(len(result) == 9)
        self.assertTrue(result.num_samples == 9)
        self.assertTrue(result.sample_rate == 12.3)
        self.assertTrue(result.num_channels == 2)
        self.assertTrue(result.length == 9 / 12.3)

    def test_values_file(self):
        result = Signal(DATA_PATH + '/sample.wav')
        self.assertTrue(np.allclose(result[:5],
                                    [-2494, -2510, -2484, -2678, -2833]))
        self.assertTrue(len(result) == 123481)
        self.assertTrue(result.num_samples == 123481)
        self.assertTrue(result.sample_rate == 44100)
        self.assertTrue(result.num_channels == 1)
        self.assertTrue(np.allclose(result.length, 2.8))


class TestSignalProcessorClass(unittest.TestCase):

    def test_types(self):
        processor = SignalProcessor()
        self.assertIsInstance(processor, SignalProcessor)
        self.assertIsInstance(processor, Processor)
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertIsInstance(result, Signal)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(result.dtype == np.int16)

    def test_types_mono(self):
        processor = SignalProcessor(num_channels=1)
        self.assertIsInstance(processor, SignalProcessor)
        self.assertIsInstance(processor, Processor)
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertIsInstance(result, Signal)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(result.dtype == np.int16)
        result = processor.process(DATA_PATH + '/stereo_sample.wav')
        self.assertIsInstance(result, Signal)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(result.dtype == np.int16)

    def test_types_norm(self):
        processor = SignalProcessor(norm=True)
        self.assertIsInstance(processor, SignalProcessor)
        self.assertIsInstance(processor, Processor)
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertIsInstance(result, Signal)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(result.dtype == np.float)

    def test_types_att(self):
        processor = SignalProcessor(att=10)
        self.assertIsInstance(processor, SignalProcessor)
        self.assertIsInstance(processor, Processor)
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertIsInstance(result, Signal)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(result.dtype == np.int16)

    def test_constant_types(self):
        self.assertIsInstance(SignalProcessor.SAMPLE_RATE, type(None))
        self.assertIsInstance(SignalProcessor.NUM_CHANNELS, type(None))
        self.assertIsInstance(SignalProcessor.NORM, bool)
        self.assertIsInstance(SignalProcessor.ATT, float)

    def test_constant_values(self):
        self.assertEqual(SignalProcessor.SAMPLE_RATE, None)
        self.assertEqual(SignalProcessor.NUM_CHANNELS, None)
        self.assertEqual(SignalProcessor.NORM, False)
        self.assertEqual(SignalProcessor.ATT, 0)

    def test_values_file(self):
        processor = SignalProcessor()

        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(np.allclose(result[:5],
                                    [-2494, -2510, -2484, -2678, -2833]))
        self.assertTrue(len(result) == 123481)
        self.assertTrue(result.num_samples == 123481)
        self.assertTrue(result.sample_rate == 44100)
        self.assertTrue(result.num_channels == 1)
        self.assertTrue(np.allclose(result.length, 2.8))

    def test_rewrite_values(self):
        processor = SignalProcessor()
        self.assertTrue(processor.num_channels is None)
        self.assertTrue(processor.norm is False)
        self.assertTrue(processor.att == 0.)
        processor.num_channels = 1
        processor.norm = True
        processor.att = 10
        self.assertTrue(processor.num_channels == 1)
        self.assertTrue(processor.norm is True)
        self.assertTrue(processor.att == 10.)

    def test_values_file_norm(self):
        processor = SignalProcessor(norm=True)
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(np.allclose(result[:5],
                                    [-0.138733, -0.139623, -0.138177,
                                     -0.148968, -0.157590]))
        self.assertTrue(len(result) == 123481)
        self.assertTrue(result.num_samples == 123481)
        self.assertTrue(result.sample_rate == 44100)
        self.assertTrue(result.num_channels == 1)
        self.assertTrue(np.allclose(result.length, 2.8))

    def test_values_file_att(self):
        processor = SignalProcessor(att=10)
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(np.allclose(result[:5],
                                    [-788, -793, -785, -846, -895]))
        self.assertTrue(len(result) == 123481)
        self.assertTrue(result.num_samples == 123481)
        self.assertTrue(result.sample_rate == 44100)
        self.assertTrue(result.num_channels == 1)
        self.assertTrue(np.allclose(result.length, 2.8))


# framing functions
class TestSignalFrameFunction(unittest.TestCase):

    def test_types(self):
        result = signal_frame(np.arange(10), 0, 4, 2)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(result.dtype == np.int)
        result = signal_frame(np.arange(10, dtype=np.float), 0, 4, 2)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(result.dtype == np.float)
        signal = Signal(DATA_PATH + '/sample.wav')
        result = signal_frame(signal, 0, 4, 2)
        self.assertIsInstance(result, Signal)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(result.dtype == np.int16)

    def test_short_input_length(self):
        result = signal_frame(np.arange(4), 0, 10, 5)
        self.assertTrue(np.allclose(result, [0, 0, 0, 0, 0, 0, 1, 2, 3, 0]))
        result = signal_frame(np.arange(4), 1, 10, 5)
        self.assertTrue(np.allclose(result, [0, 1, 2, 3, 0, 0, 0, 0, 0, 0]))

    def test_values(self):
        result = signal_frame(np.arange(10), 0, 4, 2)
        self.assertTrue(np.allclose(result, [0, 0, 0, 1]))
        result = signal_frame(np.arange(10), 1, 4, 2)
        self.assertTrue(np.allclose(result, [0, 1, 2, 3]))
        result = signal_frame(np.arange(10), 2, 4, 2)
        self.assertTrue(np.allclose(result, [2, 3, 4, 5]))
        result = signal_frame(np.arange(10), 3, 4, 2)
        self.assertTrue(np.allclose(result, [4, 5, 6, 7]))
        result = signal_frame(np.arange(10), 4, 4, 2)
        self.assertTrue(np.allclose(result, [6, 7, 8, 9]))
        result = signal_frame(np.arange(10), 5, 4, 2)
        self.assertTrue(np.allclose(result, [8, 9, 0, 0]))

    def test_float_hop_size(self):
        result = signal_frame(np.arange(10), 0, 3.5, 2)
        self.assertTrue(np.allclose(result, [0, 0, 1]))
        result = signal_frame(np.arange(10), 1, 3.5, 2)
        self.assertTrue(np.allclose(result, [1, 2, 3]))
        result = signal_frame(np.arange(10), 2, 3.5, 2)
        self.assertTrue(np.allclose(result, [3, 4, 5]))

    def test_origin(self):
        result = signal_frame(np.arange(10), 0, 4, 2, 1)
        self.assertTrue(np.allclose(result, [0, 0, 1, 2]))
        result = signal_frame(np.arange(10), 0, 4, 2, -1)
        self.assertTrue(np.allclose(result, [0, 0, 0, 0]))
        result = signal_frame(np.arange(10), 0, 4, 2, 2)
        self.assertTrue(np.allclose(result, [0, 1, 2, 3]))
        result = signal_frame(np.arange(10), 0, 4, 2, 4)
        self.assertTrue(np.allclose(result, [2, 3, 4, 5]))
        # test with float origin with half the size of the frame size
        result = signal_frame(np.arange(10), 0, 5, 2, 2.5)
        self.assertTrue(np.allclose(result, [0, 1, 2, 3, 4]))


class TestSegmentAxisFunction(unittest.TestCase):

    def test_types(self):
        result = segment_axis(np.arange(10), 4, 2)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(result.dtype == np.int)
        result = segment_axis(np.arange(10, dtype=np.float), 4, 2)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(result.dtype == np.float)
        signal = Signal(DATA_PATH + '/sample.wav')
        result = segment_axis(signal, 4, 2)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(result.dtype == np.int16)

    def test_errors(self):
        with self.assertRaises(ValueError):
            segment_axis(np.arange(10), 4, 2, axis=1)

    def test_values(self):
        result = segment_axis(np.arange(10), 4, 2)
        self.assertTrue(np.allclose(result, [[0, 1, 2, 3], [2, 3, 4, 5],
                                             [4, 5, 6, 7], [6, 7, 8, 9]]))
        result = segment_axis(np.arange(10), 4, 3, end='pad')
        self.assertTrue(np.allclose(result, [[0, 1, 2, 3], [3, 4, 5, 6],
                                             [6, 7, 8, 9]]))
        result = segment_axis(np.arange(11), 4, 3, end='pad')
        self.assertTrue(np.allclose(result, [[0, 1, 2, 3], [3, 4, 5, 6],
                                             [6, 7, 8, 9], [9, 10, 0, 0]]))
        result = segment_axis(np.arange(11), 4, 3, end='pad', end_value=1)
        self.assertTrue(np.allclose(result, [[0, 1, 2, 3], [3, 4, 5, 6],
                                             [6, 7, 8, 9], [9, 10, 1, 1]]))
        result = segment_axis(np.arange(11), 4, 3, end='wrap')
        self.assertTrue(np.allclose(result, [[0, 1, 2, 3], [3, 4, 5, 6],
                                             [6, 7, 8, 9], [9, 10, 0, 1]]))
        result = segment_axis(np.arange(11), 4, 3, end='cut')
        self.assertTrue(np.allclose(result, [[0, 1, 2, 3], [3, 4, 5, 6],
                                             [6, 7, 8, 9]]))
        result = segment_axis(np.arange(11), 4, 3, axis=0)
        self.assertTrue(np.allclose(result, [[0, 1, 2, 3], [3, 4, 5, 6],
                                             [6, 7, 8, 9]]))


# framing classes
class TestFramedSignalClass(unittest.TestCase):

    def test_types(self):
        result = FramedSignal(np.arange(10), 4, 2)
        self.assertIsInstance(result, FramedSignal)
        self.assertIsInstance(result.signal, Signal)
        self.assertIsInstance(result.frame_size, int)
        self.assertIsInstance(result.hop_size, float)
        self.assertIsInstance(result.origin, int)
        self.assertIsInstance(result.start, int)
        self.assertIsInstance(result.num_frames, int)
        self.assertIsInstance(result[0], Signal)
        # properties
        self.assertIsInstance(result.frame_rate, type(None))
        self.assertIsInstance(result.fps, type(None))
        self.assertIsInstance(result.overlap_factor, float)
        self.assertIsInstance(result.shape, tuple)

    def test_types_slice(self):
        # get a slice of a FramedSignal
        result = FramedSignal(np.arange(10), 4, 2)[:2]
        self.assertIsInstance(result, FramedSignal)
        self.assertIsInstance(result.signal, Signal)
        self.assertIsInstance(result.frame_size, int)
        self.assertIsInstance(result.hop_size, float)
        self.assertIsInstance(result.origin, int)
        self.assertIsInstance(result.start, int)
        self.assertIsInstance(result.num_frames, int)
        self.assertIsInstance(result[0], Signal)
        # properties
        self.assertIsInstance(result.frame_rate, type(None))
        self.assertIsInstance(result.fps, type(None))
        self.assertIsInstance(result.overlap_factor, float)
        self.assertIsInstance(result.shape, tuple)

    def test_types_with_sample_rate(self):
        result = FramedSignal(np.arange(10), 4, 2, sample_rate=1)
        self.assertIsInstance(result, FramedSignal)
        self.assertIsInstance(result.signal, Signal)
        self.assertIsInstance(result.frame_size, int)
        self.assertIsInstance(result.hop_size, float)
        self.assertIsInstance(result.origin, int)
        self.assertIsInstance(result.start, int)
        self.assertIsInstance(result.num_frames, int)
        self.assertIsInstance(result[0], Signal)
        # properties
        self.assertIsInstance(result.frame_rate, float)
        self.assertIsInstance(result.fps, float)
        self.assertIsInstance(result.overlap_factor, float)
        self.assertIsInstance(result.shape, tuple)

    def test_types_signal(self):
        signal = Signal(DATA_PATH + '/sample.wav')
        result = FramedSignal(signal)
        self.assertIsInstance(result, FramedSignal)
        self.assertIsInstance(result.signal, Signal)
        self.assertIsInstance(result.frame_size, int)
        self.assertIsInstance(result.hop_size, float)
        self.assertIsInstance(result.origin, int)
        self.assertIsInstance(result.start, int)
        self.assertIsInstance(result.num_frames, int)
        self.assertIsInstance(result[0], Signal)
        # properties
        self.assertIsInstance(result.frame_rate, float)
        self.assertIsInstance(result.fps, float)
        self.assertIsInstance(result.overlap_factor, float)
        self.assertIsInstance(result.shape, tuple)

    def test_types_file(self):
        result = FramedSignal(DATA_PATH + '/sample.wav')
        self.assertIsInstance(result, FramedSignal)
        self.assertIsInstance(result.signal, Signal)
        self.assertIsInstance(result.frame_size, int)
        self.assertIsInstance(result.hop_size, float)
        self.assertIsInstance(result.origin, int)
        self.assertIsInstance(result.start, int)
        self.assertIsInstance(result.num_frames, int)
        self.assertIsInstance(result[0], Signal)
        # properties
        self.assertIsInstance(result.frame_rate, float)
        self.assertIsInstance(result.fps, float)
        self.assertIsInstance(result.overlap_factor, float)
        self.assertIsInstance(result.shape, tuple)

    def test_values_array(self):
        result = FramedSignal(np.arange(10), 4, 2)
        self.assertTrue(result.frame_size == 4)
        self.assertTrue(result.hop_size == 2.)
        self.assertTrue(result.origin == 0)
        self.assertTrue(result.start == 0)
        self.assertTrue(result.num_frames == 6)
        self.assertTrue(np.allclose(result[0], [0, 0, 0, 1]))
        self.assertTrue(result.frame_rate is None)
        self.assertTrue(result.fps is None)
        self.assertTrue(result.overlap_factor == 0.5)
        self.assertTrue(result.shape == (6, 4))

    def test_values_array_end(self):
        result = FramedSignal(np.arange(10), 4, 2, end='extend')
        self.assertTrue(result.num_frames == 6)
        result = FramedSignal(np.arange(10), 4, 2, end='normal')
        self.assertTrue(result.num_frames == 5)

    def test_values_array_with_sample_rate(self):
        result = FramedSignal(np.arange(10), 4, 2, sample_rate=4)
        self.assertTrue(result.frame_size == 4)
        self.assertTrue(result.hop_size == 2.)
        self.assertTrue(result.origin == 0)
        self.assertTrue(result.start == 0)
        self.assertTrue(result.num_frames == 6)
        self.assertTrue(np.allclose(result[0], [0, 0, 0, 1]))
        self.assertTrue(result.frame_rate == 2)
        self.assertTrue(result.fps == 2)
        self.assertTrue(result.overlap_factor == 0.5)
        self.assertTrue(result.shape == (6, 4))

    def test_values_file(self):
        signal = Signal(DATA_PATH + '/sample.wav')
        result = FramedSignal(DATA_PATH + '/sample.wav')
        self.assertTrue(result.frame_size == 2048)
        self.assertTrue(result.hop_size == 441.)
        self.assertTrue(result.origin == 0)
        self.assertTrue(result.start == 0)
        self.assertTrue(result.num_frames == 281)
        self.assertTrue(np.allclose(result[0][:5], [0, 0, 0, 0, 0]))
        # 3rd frame should start at 3 * 441 - 2048 / 2 = 299
        self.assertTrue(np.allclose(result[3], signal[299: 299 + 2048]))
        self.assertTrue(result.frame_rate == 100)
        self.assertTrue(result.fps == 100)

    def test_values_file_origin(self):
        signal = Signal(DATA_PATH + '/sample.wav')
        # literal origin
        result = FramedSignal(DATA_PATH + '/sample.wav', origin='online')
        self.assertTrue(result.origin == 1023)
        self.assertTrue(result.num_frames == 281)
        # 6th frame should start at 6 * 441 - 2048 + 1 (ref sample) = 599
        self.assertTrue(np.allclose(result[6], signal[599: 599 + 2048]))
        # literal left origin
        result = FramedSignal(DATA_PATH + '/sample.wav', origin='left')
        self.assertTrue(result.origin == 1023)
        # positive origin shifts the window to the left
        result = FramedSignal(DATA_PATH + '/sample.wav', origin=10)
        self.assertTrue(result.origin == 10)
        # literal offline origin
        result = FramedSignal(DATA_PATH + '/sample.wav', origin='offline')
        self.assertTrue(result.origin == 0)
        # literal center origin
        result = FramedSignal(DATA_PATH + '/sample.wav', origin='center')
        self.assertTrue(result.origin == 0)
        # literal right origin
        result = FramedSignal(DATA_PATH + '/sample.wav', origin='right')
        self.assertTrue(result.origin == -1024)
        # literal future origin
        result = FramedSignal(DATA_PATH + '/sample.wav', origin='future')
        self.assertTrue(result.origin == -1024)

    def test_values_file_start(self):
        signal = Signal(DATA_PATH + '/sample.wav')
        result = FramedSignal(DATA_PATH + '/sample.wav', start=10)
        # start sample shifted to the right
        self.assertTrue(result.origin == 0)
        self.assertTrue(result.start == 10)
        self.assertTrue(result.num_frames == 281)
        # 3rd frame should start at 3 * 441 - 2048 / 2 + 10 = 309
        self.assertTrue(np.allclose(result[3], signal[309: 309 + 2048]))

    def test_values_file_fps(self):
        result = FramedSignal(DATA_PATH + '/sample.wav', fps=200)
        self.assertTrue(result.frame_size == 2048)
        self.assertTrue(result.hop_size == 220.5)
        result = FramedSignal(DATA_PATH + '/sample.wav', fps=50)
        self.assertTrue(result.frame_size == 2048)
        self.assertTrue(result.hop_size == 882.)


class TestFramedSignalProcessorClass(unittest.TestCase):

    def test_types(self):
        processor = FramedSignalProcessor()
        self.assertIsInstance(processor, FramedSignalProcessor)
        self.assertIsInstance(processor, Processor)
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertIsInstance(result, FramedSignal)

    def test_values(self):
        processor = FramedSignalProcessor()
        self.assertTrue(processor.frame_size == 2048)
        self.assertTrue(processor.hop_size == 441.)
        self.assertTrue(processor.fps is None)
        self.assertTrue(processor.online is False)
        self.assertTrue(processor.end == 'extend')
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(np.allclose(result[0][:1023], np.zeros(1023)))
        self.assertTrue(np.allclose(result[0][1024], -2494))
        self.assertTrue(len(result) == 281)
        self.assertTrue(result.num_frames == 281)

    def test_rewrite_values(self):
        processor = FramedSignalProcessor()
        processor.frame_size = 100
        processor.hop_size = 44.5
        processor.fps = 20
        processor.online = True
        processor.end = 'bogus'
        self.assertTrue(processor.frame_size == 100)
        self.assertTrue(processor.hop_size == 44.5)
        self.assertTrue(processor.fps == 20)
        self.assertTrue(processor.online is True)
        self.assertTrue(processor.end == 'bogus')

    def test_values_online(self):
        processor = FramedSignalProcessor(online=True)
        self.assertTrue(processor.frame_size == 2048)
        self.assertTrue(processor.hop_size == 441.)
        self.assertTrue(processor.fps is None)
        self.assertTrue(processor.online is True)
        self.assertTrue(processor.end == 'extend')
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(np.allclose(result[0][-1], -2494))
        self.assertTrue(len(result) == 281)
        self.assertTrue(result.num_frames == 281)

    def test_values_fps(self):
        processor = FramedSignalProcessor(fps=200.)
        self.assertTrue(processor.frame_size == 2048)
        self.assertTrue(processor.hop_size == 441.)
        self.assertTrue(processor.fps == 200)
        self.assertTrue(processor.online is False)
        self.assertTrue(processor.end == 'extend')
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(np.allclose(result[0][:1023], np.zeros(1023)))
        self.assertTrue(np.allclose(result[0][1024], -2494))
        self.assertTrue(len(result) == 561)
        self.assertTrue(result.num_frames == 561)

    def test_values_end(self):
        processor = FramedSignalProcessor(end='normal')
        self.assertTrue(processor.frame_size == 2048)
        self.assertTrue(processor.hop_size == 441.)
        self.assertTrue(processor.fps is None)
        self.assertTrue(processor.online is False)
        self.assertTrue(processor.end == 'normal')
        # test with a file
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(np.allclose(result[0][:1023], np.zeros(1023)))
        self.assertTrue(np.allclose(result[0][1024], -2494))
        self.assertTrue(len(result) == 281)
        self.assertTrue(result.num_frames == 281)
        # test with an array
        processor.frame_size = 10
        processor.hop_size = 6
        result = processor.process(np.arange(18))
        self.assertTrue(len(result) == 3)
        self.assertTrue(result.num_frames == 3)
        # rewrite the end
        processor.end = 'extend'
        result = processor.process(np.arange(18))
        self.assertTrue(len(result) == 4)
        self.assertTrue(result.num_frames == 4)
        # test with incorrect end value
        with self.assertRaises(ValueError):
            processor = FramedSignalProcessor(end='bla')
            processor.process(DATA_PATH + '/sample.wav')

    def test_constant_types(self):
        self.assertIsInstance(FramedSignalProcessor.FRAME_SIZE, int)
        self.assertIsInstance(FramedSignalProcessor.HOP_SIZE, float)
        self.assertIsInstance(FramedSignalProcessor.FPS, float)
        self.assertIsInstance(FramedSignalProcessor.ONLINE, bool)
        self.assertIsInstance(FramedSignalProcessor.START, int)
        self.assertIsInstance(FramedSignalProcessor.END_OF_SIGNAL, str)

    def test_constant_values(self):
        self.assertEqual(FramedSignalProcessor.FRAME_SIZE, 2048)
        self.assertEqual(FramedSignalProcessor.HOP_SIZE, 441.)
        self.assertEqual(FramedSignalProcessor.FPS, 100.)
        self.assertEqual(FramedSignalProcessor.ONLINE, False)
        self.assertEqual(FramedSignalProcessor.START, 0)
        self.assertEqual(FramedSignalProcessor.END_OF_SIGNAL, 'extend')

    def test_values_file(self):
        processor = FramedSignalProcessor()
        result = processor.process(DATA_PATH + '/sample.wav')
        self.assertTrue(np.allclose(result[0][:100], np.zeros(100)))
        self.assertTrue(len(result) == 281)
        self.assertTrue(result.num_frames == 281)
        self.assertTrue(result.frame_size == 2048)
        self.assertTrue(result.hop_size == 441.)
