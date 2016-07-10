# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.audio.signal module.

"""

from __future__ import absolute_import, division, print_function

import unittest
from os.path import join as pj

from . import AUDIO_PATH, DATA_PATH
from .test_audio_comb_filters import sig_1d, sig_2d
from madmom.audio.signal import *


sample_file = pj(AUDIO_PATH, 'sample.wav')
stereo_sample_file = pj(AUDIO_PATH, 'stereo_sample.wav')


# test signal functions
class TestSmoothFunction(unittest.TestCase):

    def test_types(self):
        # mono signals
        result = smooth(sig_1d, None)
        self.assertTrue(type(result) == type(sig_1d))
        self.assertTrue(len(result) == len(sig_1d))
        self.assertTrue(result.shape == sig_1d.shape)
        result = smooth(sig_1d, 3)
        self.assertTrue(type(result) == type(sig_1d))
        self.assertTrue(len(result) == len(sig_1d))
        self.assertTrue(result.shape == sig_1d.shape)
        # multi-channel signals
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
            smooth(np.zeros(9).reshape(3, 3), -1)
        with self.assertRaises(ValueError):
            smooth(np.zeros(9).reshape(3, 3), 'bla')
        with self.assertRaises(ValueError):
            smooth(np.zeros(18).reshape(3, 3, 2), 4)

    def test_values(self):
        # mono signals
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
        # multi-channel signals
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


class TestAdjustGainFunction(unittest.TestCase):

    def test_types(self):
        # mono signals
        result = adjust_gain(sig_1d, 0)
        self.assertTrue(type(result) == type(sig_1d))
        self.assertTrue(len(result) == len(sig_1d))
        self.assertTrue(result.shape == sig_1d.shape)
        self.assertTrue(result.dtype == sig_1d.dtype)
        # same with int16 dtype
        result = adjust_gain(sig_1d.astype(np.int16), 0)
        self.assertTrue(len(result) == len(sig_1d))
        self.assertTrue(result.shape == sig_1d.shape)
        self.assertTrue(result.dtype == np.int16)
        # from file
        signal = Signal(sample_file)
        result = adjust_gain(signal, 0)
        self.assertIsInstance(result, Signal)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(result.dtype == signal.dtype)
        # multi-channel signals
        result = adjust_gain(sig_2d, 0)
        self.assertTrue(type(result) == type(sig_2d))
        self.assertTrue(len(result) == len(sig_2d))
        self.assertTrue(result.shape == sig_2d.shape)
        self.assertTrue(result.dtype == sig_2d.dtype)
        # same with int dtype
        result = adjust_gain(sig_2d.astype(np.int), 0)
        self.assertTrue(len(result) == len(sig_2d))
        self.assertTrue(result.shape == sig_2d.shape)
        self.assertTrue(result.dtype == np.int)

    def test_values(self):
        # mono signals
        result = adjust_gain(sig_1d, 0)
        self.assertTrue(np.allclose(result, sig_1d))
        result = adjust_gain(sig_1d, -10)
        self.assertTrue(np.allclose(result, 0.31622777 * sig_1d))
        result = adjust_gain(sig_1d, 10)
        self.assertTrue(np.allclose(result, 3.1622777 * sig_1d))
        # same with int dtype
        result = adjust_gain(sig_1d.astype(np.int), 0)
        self.assertTrue(np.allclose(result, sig_1d.astype(np.int)))
        result = adjust_gain(sig_1d.astype(np.int), -5)
        self.assertTrue(np.allclose(result, 0 * sig_1d))
        # multi-channel signals
        result = adjust_gain(sig_2d, 0)
        self.assertTrue(np.allclose(result, sig_2d))
        result = adjust_gain(sig_2d, -3)
        self.assertTrue(np.allclose(result, 0.70794578 * sig_2d))
        # same with int16 dtype
        result = adjust_gain(sig_2d.astype(np.int16), 0)
        self.assertTrue(np.allclose(result, sig_2d))
        result = adjust_gain(sig_2d.astype(np.int16), -1)
        self.assertTrue(np.allclose(result, 0 * sig_2d))

    def test_errors(self):
        with self.assertRaises(ValueError):
            adjust_gain(sig_2d.astype(np.int16), +60)


class TestAttenuateFunction(unittest.TestCase):

    def test_types(self):
        # mono signals
        result = attenuate(sig_1d, 0)
        self.assertTrue(type(result) == type(sig_1d))
        self.assertTrue(len(result) == len(sig_1d))
        self.assertTrue(result.shape == sig_1d.shape)
        self.assertTrue(result.dtype == sig_1d.dtype)
        # same as int16 dtype
        result = attenuate(sig_1d.astype(np.int16), 0)
        self.assertTrue(len(result) == len(sig_1d))
        self.assertTrue(result.shape == sig_1d.shape)
        self.assertTrue(result.dtype == np.int16)
        # from file
        signal = Signal(sample_file)
        result = attenuate(signal, 0)
        self.assertIsInstance(result, Signal)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(result.dtype == np.int16)
        # multi-channel signals
        result = attenuate(sig_2d, 0)
        self.assertTrue(type(result) == type(sig_2d))
        self.assertTrue(len(result) == len(sig_2d))
        self.assertTrue(result.shape == sig_2d.shape)
        self.assertTrue(result.dtype == sig_2d.dtype)
        # same as int dtype
        result = attenuate(sig_2d.astype(np.int), 0)
        self.assertTrue(len(result) == len(sig_2d))
        self.assertTrue(result.shape == sig_2d.shape)
        self.assertTrue(result.dtype == np.int)

    def test_values(self):
        # mono signals
        result = attenuate(sig_1d, 0)
        self.assertTrue(np.allclose(result, sig_1d))
        result = attenuate(sig_1d, 10)
        self.assertTrue(np.allclose(result, 0.31622777 * sig_1d))
        result = attenuate(sig_1d, -10)
        self.assertTrue(np.allclose(result, 3.1622777 * sig_1d))
        # same with int dtype
        result = attenuate(sig_1d.astype(np.int), 0)
        self.assertTrue(np.allclose(result, sig_1d.astype(np.int)))
        result = attenuate(sig_1d.astype(np.int), 5)
        self.assertTrue(np.allclose(result, 0 * sig_1d))
        # multi-channel signals
        result = attenuate(sig_2d, 0)
        self.assertTrue(np.allclose(result, sig_2d))
        result = attenuate(sig_2d, 3)
        self.assertTrue(np.allclose(result, 0.70794578 * sig_2d))
        # same with int16 dtype
        result = attenuate(sig_2d.astype(np.int16), 0)
        self.assertTrue(np.allclose(result, sig_2d))
        result = attenuate(sig_2d.astype(np.int16), 1)
        self.assertTrue(np.allclose(result, 0 * sig_2d))

    def test_errors(self):
        with self.assertRaises(ValueError):
            attenuate(sig_2d.astype(np.int16), -10)


class TestNormalizeFunction(unittest.TestCase):

    def test_types(self):
        # mono signals
        result = normalize(sig_1d)
        self.assertTrue(len(result) == len(sig_1d))
        self.assertTrue(result.shape == sig_1d.shape)
        self.assertTrue(result.dtype == sig_1d.dtype)
        # same as int16 dtype
        result = normalize(sig_1d.astype(np.int16))
        self.assertTrue(len(result) == len(sig_1d))
        self.assertTrue(result.shape == sig_1d.shape)
        self.assertTrue(result.dtype == np.int16)
        # from file
        signal = Signal(sample_file)
        result = normalize(signal)
        self.assertIsInstance(result, Signal)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(result.dtype == np.int16)
        # multi-channel signals
        result = normalize(sig_2d)
        self.assertTrue(len(result) == len(sig_2d))
        self.assertTrue(result.shape == sig_2d.shape)
        self.assertTrue(result.dtype == sig_2d.dtype)
        # same as int32 dtype
        result = normalize(sig_2d.astype(np.int32))
        self.assertTrue(len(result) == len(sig_2d))
        self.assertTrue(result.shape == sig_2d.shape)
        self.assertTrue(result.dtype == np.int32)

    def test_values(self):
        # mono signals
        result = normalize(sig_1d)
        self.assertTrue(np.allclose(result, sig_1d))
        result = normalize(sig_1d * 0.5)
        self.assertTrue(np.allclose(result, sig_1d))
        self.assertTrue(np.max(result) == 1)
        # same as int16 dtype
        result = normalize(10 * sig_1d.astype(np.int16))
        self.assertTrue(np.allclose(result, sig_1d * 32767))
        self.assertTrue(np.max(result) == 32767)
        # multi-channel signals
        result = normalize(sig_2d)
        self.assertTrue(np.allclose(result, sig_2d))
        self.assertTrue(np.max(result) == 1)
        # negative values
        result = normalize(sig_2d * 4 - 2)
        self.assertTrue(np.allclose(result, sig_2d * 2 - 1))
        self.assertTrue(result.max() == 1)
        self.assertTrue(result.min() == -1)
        # same as int32 dtype
        result = normalize(3 * sig_2d.astype(np.int32))
        self.assertTrue(np.allclose(result, sig_2d * 2147483647))
        self.assertTrue(np.max(result) == 2147483647)

    def test_errors(self):
        with self.assertRaises(ValueError):
            normalize(sig_2d.astype(np.int64))


class TestMixFunction(unittest.TestCase):

    mono_2d = np.asarray([0.5, 0, 1, 0, 0.5, 0.5, 0.5, 0, 1], dtype=np.float)

    def test_types(self):
        # mono signals
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
        # same as int dtype
        result = remix(sig_1d.astype(np.int), 1)
        self.assertTrue(len(result) == len(sig_1d))
        self.assertTrue(result.shape == sig_1d.shape)
        self.assertTrue(result.dtype == np.int)
        result = remix(sig_1d.astype(np.int), 2)
        self.assertTrue(len(result) == len(sig_1d))
        self.assertTrue(result.shape == (len(sig_1d), 2))
        self.assertTrue(result.dtype == np.int)
        # from file
        signal = Signal(sample_file)
        result = remix(signal, 1)
        self.assertTrue(isinstance(result, Signal))
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertTrue(result.dtype == np.int16)
        # multi-channel signals
        result = remix(sig_2d, 1)
        self.assertTrue(len(result) == len(sig_2d))
        self.assertTrue(result.shape == sig_1d.shape)
        self.assertTrue(result.dtype == sig_2d.dtype)
        result = remix(sig_2d, 2)
        self.assertTrue(len(result) == len(sig_2d))
        self.assertTrue(result.shape == sig_2d.shape)
        self.assertTrue(result.dtype == sig_2d.dtype)
        # from file
        signal = Signal(stereo_sample_file)
        result = remix(signal, 1)
        self.assertTrue(isinstance(result, Signal))
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertTrue(result.dtype == np.int16)
        result = remix(signal, 2)
        self.assertTrue(isinstance(result, Signal))
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertTrue(result.dtype == np.int16)
        with self.assertRaises(NotImplementedError):
            remix(sig_2d, 3)
        # same as int dtype
        result = remix(sig_2d.astype(np.int), 1)
        self.assertTrue(len(result) == len(sig_2d))
        self.assertTrue(result.shape == sig_1d.shape)
        self.assertTrue(result.dtype == np.int)
        result = remix(sig_2d.astype(np.int), 2)
        self.assertTrue(len(result) == len(sig_2d))
        self.assertTrue(result.shape == sig_2d.shape)
        self.assertTrue(result.dtype == np.int)

    def test_values(self):
        # mono signals
        result = remix(sig_1d, 1)
        self.assertTrue(np.allclose(result, sig_1d))
        # same as int dtype
        result = remix(sig_2d.astype(np.int), 1)
        self.assertTrue(np.allclose(result, self.mono_2d.astype(np.int)))
        # multi-channel signals
        result = remix(sig_2d, 1)
        self.assertTrue(np.allclose(result, self.mono_2d))
        # same as int dtype
        result = remix(2 * sig_2d.astype(np.int), 1)
        self.assertTrue(np.allclose(result, 2 * self.mono_2d))


class TestRescaleFunction(unittest.TestCase):

    def test_types(self):
        # mono signals
        result = rescale(sig_1d, np.float)
        self.assertTrue(len(result) == len(sig_1d))
        self.assertTrue(result.shape == sig_1d.shape)
        self.assertTrue(result.dtype == np.float)
        # from file
        signal = Signal(sample_file)
        result = rescale(signal)
        self.assertTrue(isinstance(result, Signal))
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertTrue(result.dtype == np.float32)
        # multi-channel signals
        result = rescale(sig_2d, np.float16)
        self.assertTrue(len(result) == len(sig_2d))
        self.assertTrue(result.shape == sig_2d.shape)
        self.assertTrue(result.dtype == np.float16)
        # from file
        signal = Signal(stereo_sample_file)
        result = rescale(signal, np.float)
        self.assertTrue(isinstance(result, Signal))
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertTrue(result.dtype == np.float)

    def test_errors(self):
        with self.assertRaises(ValueError):
            rescale(sig_2d, np.complex)
        with self.assertRaises(ValueError):
            rescale(sig_2d, np.int)

    def test_values(self):
        # mono signals
        result = rescale(sig_1d, np.float)
        self.assertTrue(np.allclose(result, sig_1d))
        # from file
        signal = Signal(sample_file)
        result = rescale(signal)
        self.assertTrue(np.allclose(result[:6],
                                    [-0.07611316, -0.07660146, -0.07580798,
                                     -0.08172857, -0.08645894, -0.08212531]))
        # multi-channel signals
        result = rescale(sig_2d, np.float16)
        self.assertTrue(np.allclose(result, sig_2d))
        # from file
        signal = Signal(stereo_sample_file)
        result = rescale(signal, np.float)
        self.assertTrue(np.allclose(result[:6], [[0.00100711, 0.0011597],
                                                 [0.00106815, 0.00109867],
                                                 [0.00088504, 0.00103763],
                                                 [0.00109867, 0.00094607],
                                                 [0.00112918, 0.00091556],
                                                 [0.00109867, 0.00103763]]))


class TestTrimFunction(unittest.TestCase):

    def test_types(self):
        # mono signals
        result = trim(sig_1d)
        self.assertTrue(type(result) == type(sig_1d))
        self.assertTrue(result.ndim == sig_1d.ndim)
        self.assertTrue(result.dtype == sig_1d.dtype)
        signal = Signal(sample_file)
        result = trim(signal)
        self.assertIsInstance(result, Signal)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(result.ndim == signal.ndim)
        self.assertTrue(result.dtype == np.int16)
        # multi-channel signals
        result = trim(sig_2d)
        self.assertTrue(type(result) == type(sig_2d))
        self.assertTrue(len(result) == len(sig_2d))
        self.assertTrue(result.ndim == sig_2d.ndim)
        self.assertTrue(result.dtype == sig_2d.dtype)
        signal = Signal(stereo_sample_file)
        result = trim(signal)
        self.assertIsInstance(result, Signal)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(result.ndim == signal.ndim)
        self.assertTrue(result.dtype == np.int16)

    def test_values(self):
        # mono signals
        result = trim(sig_1d)
        trimmed_1d = [1, 0, 0, 1, 0, 0, 1]
        self.assertTrue(np.allclose(result, trimmed_1d))
        self.assertTrue(len(result) == len(sig_1d) - 2)
        # multi-channel signals
        # signal has leading zeros only in one channel
        result = trim(sig_2d)
        self.assertTrue(result.shape == sig_2d.shape)
        signal = Signal(stereo_sample_file)
        result = trim(signal)
        self.assertTrue(result.shape == signal.shape)
        # signal with leading zeros only in both channels
        signal = np.tile(np.arange(5), 2).reshape(2, 5).T
        result = trim(signal)
        self.assertTrue(result.shape == (4, 2))
        self.assertTrue(np.allclose(result[:, 0], np.arange(1, 5)))
        self.assertTrue(np.allclose(result[:, 1], np.arange(1, 5)))


class TestRootMeanSquareFunction(unittest.TestCase):

    def test_types(self):
        # mono signals
        result = root_mean_square(sig_1d)
        self.assertIsInstance(result, float)
        # multi-channel signals
        result = root_mean_square(sig_2d)
        self.assertIsInstance(result, float)

    def test_values(self):
        # mono signals
        result = root_mean_square(sig_1d)
        self.assertTrue(np.allclose(result, 0.57735026919))
        result = root_mean_square(np.zeros(100))
        self.assertTrue(np.allclose(result, 0))
        # multi-channel signals
        result = root_mean_square(sig_2d)
        self.assertTrue(np.allclose(result, 2. / 3))
        result = root_mean_square(np.zeros(100).reshape(-1, 2))
        self.assertTrue(np.allclose(result, 0))


class TestSoundPressureLevelFunction(unittest.TestCase):

    def test_types(self):
        # mono signals
        result = sound_pressure_level(sig_1d)
        self.assertIsInstance(result, float)
        # multi-channel signals
        result = sound_pressure_level(sig_2d)
        self.assertIsInstance(result, float)

    def test_values(self):
        # mono signals
        result = sound_pressure_level(sig_1d)
        self.assertTrue(np.allclose(result, -4.7712125472))
        # silence
        result = sound_pressure_level(np.zeros(100))
        self.assertTrue(np.allclose(result, -np.finfo(float).max))
        # maximum float amplitude, alternating between -1 and 1
        sinus = np.cos(np.linspace(0, 2 * np.pi * 100, 2 * 100 + 1))
        result = sound_pressure_level(sinus)
        self.assertTrue(np.allclose(result, 0.))
        # maximum int16 amplitude, alternating between -1 and 1
        sinus_int16 = (sinus * np.iinfo(np.int16).max).astype(np.int16)
        result = sound_pressure_level(sinus_int16)
        self.assertTrue(np.allclose(result, 0.))

        # multi-channel signals
        result = sound_pressure_level(sig_2d)
        self.assertTrue(np.allclose(result, -3.52182518111))
        # silence
        result = sound_pressure_level(np.zeros(100).reshape(-1, 2))
        self.assertTrue(np.allclose(result, -np.finfo(float).max))
        # maximum float amplitude, alternating between -1 and 1
        sig = remix(sinus, 2)
        result = sound_pressure_level(sig)
        self.assertTrue(np.allclose(result, 0.))
        # maximum int16 amplitude, alternating between -1 and 1
        sig = remix(sinus_int16, 2)
        result = sound_pressure_level(sig)
        self.assertTrue(np.allclose(result, 0.))


class TestLoadWaveFileFunction(unittest.TestCase):

    def test_types(self):
        signal, sample_rate = load_wave_file(sample_file)
        self.assertIsInstance(signal, np.ndarray)
        self.assertTrue(signal.dtype == np.int16)
        self.assertTrue(type(sample_rate) == int)

    def test_file_handle(self):
        # open file handle
        file_handle = open(sample_file, 'rb')
        signal, sample_rate = load_wave_file(file_handle)
        self.assertIsInstance(signal, np.ndarray)
        self.assertTrue(signal.dtype == np.int16)
        self.assertTrue(type(sample_rate) == int)

    def test_values(self):
        # test wave loader
        signal, sample_rate = load_wave_file(sample_file)
        self.assertTrue(np.allclose(signal[:5],
                                    [-2494, -2510, -2484, -2678, -2833]))
        self.assertTrue(len(signal) == 123481)
        self.assertTrue(sample_rate == 44100)
        self.assertTrue(signal.shape == (123481,))
        # stereo
        signal, sample_rate = load_wave_file(stereo_sample_file)
        self.assertTrue(np.allclose(signal[:4],
                                    [[33, 38], [35, 36], [29, 34], [36, 31]]))
        self.assertTrue(len(signal) == 182919)
        self.assertTrue(sample_rate == 44100)
        self.assertTrue(signal.shape == (182919, 2))

    def test_start_stop(self):
        # test wave loader
        signal, sample_rate = load_wave_file(sample_file, start=1. / 44100,
                                             stop=5. / 44100)
        self.assertTrue(np.allclose(signal, [-2510, -2484, -2678, -2833]))
        self.assertTrue(len(signal) == 4)
        self.assertTrue(sample_rate == 44100)

    def test_downmix(self):
        # test wave loader
        signal, sample_rate = load_wave_file(stereo_sample_file,
                                             num_channels=1)
        self.assertTrue(np.allclose(signal[:5], [35, 35, 31, 33, 33]))

        self.assertTrue(len(signal) == 182919)
        self.assertTrue(sample_rate == 44100)
        self.assertTrue(signal.shape == (182919, ))

    def test_upmix(self):
        signal, sample_rate = load_wave_file(sample_file, num_channels=2)
        self.assertTrue(np.allclose(signal[:5],
                                    [[-2494, -2494], [-2510, -2510],
                                     [-2484, -2484], [-2678, -2678],
                                     [-2833, -2833]]))
        self.assertTrue(len(signal) == 123481)
        self.assertTrue(sample_rate == 44100)
        self.assertTrue(signal.shape == (123481, 2))

    def test_errors(self):
        # resampling of wav not supported
        with self.assertRaises(ValueError):
            load_wave_file(sample_file, sample_rate=22050)
        # resampling of wav not supported
        with self.assertRaises(ValueError):
            load_wave_file(sample_file, dtype=np.float)
        # file not found
        with self.assertRaises(IOError):
            load_wave_file(pj(AUDIO_PATH, 'foo_bar.wav'))
        # not an audio file
        with self.assertRaises(ValueError):
            load_wave_file(pj(DATA_PATH, 'README'))
        # closed file handle
        with self.assertRaises(ValueError):
            file_handle = open(sample_file, 'rb')
            file_handle.close()
            load_wave_file(file_handle)


class TestLoadAudioFileFunction(unittest.TestCase):

    # this tests both the madmom.audio.signal.load_wave_file() and the
    # madmom.audio.ffmpeg.load_ffmpeg_file() functions via the universal
    # load_audio_file() function

    def test_types(self):
        # test wave loader
        signal, sample_rate = load_audio_file(sample_file)
        self.assertIsInstance(signal, np.ndarray)
        self.assertTrue(signal.dtype == np.int16)
        self.assertTrue(type(sample_rate) == int)
        # test ffmpeg loader
        signal, sample_rate = load_audio_file(stereo_sample_file)
        self.assertIsInstance(signal, np.ndarray)
        self.assertTrue(signal.dtype == np.int16)
        self.assertTrue(type(sample_rate) == int)

    def test_file_handle(self):
        # test wave loader
        file_handle = open(sample_file)
        signal, sample_rate = load_audio_file(file_handle)
        self.assertIsInstance(signal, np.ndarray)
        self.assertTrue(signal.dtype == np.int16)
        self.assertTrue(type(sample_rate) == int)
        file_handle.close()
        # closed file handle
        signal, sample_rate = load_audio_file(file_handle)
        self.assertIsInstance(signal, np.ndarray)
        self.assertTrue(signal.dtype == np.int16)
        self.assertTrue(type(sample_rate) == int)
        # test ffmpeg loader
        file_handle = open(sample_file)
        signal, sample_rate = load_audio_file(file_handle)
        self.assertIsInstance(signal, np.ndarray)
        self.assertTrue(signal.dtype == np.int16)
        self.assertTrue(type(sample_rate) == int)
        file_handle.close()
        # closed file handle
        signal, sample_rate = load_audio_file(file_handle)
        self.assertIsInstance(signal, np.ndarray)
        self.assertTrue(signal.dtype == np.int16)
        self.assertTrue(type(sample_rate) == int)

    def test_values(self):
        # test wave loader
        signal, sample_rate = load_audio_file(sample_file)
        self.assertTrue(np.allclose(signal[:5],
                                    [-2494, -2510, -2484, -2678, -2833]))
        self.assertTrue(len(signal) == 123481)
        self.assertTrue(sample_rate == 44100)
        self.assertTrue(signal.shape == (123481,))
        # stereo
        signal, sample_rate = load_audio_file(stereo_sample_file)
        self.assertTrue(np.allclose(signal[:4],
                                    [[33, 38], [35, 36], [29, 34], [36, 31]]))
        self.assertTrue(len(signal) == 182919)
        self.assertTrue(sample_rate == 44100)
        self.assertTrue(signal.shape == (182919, 2))
        # test ffmpeg loader
        signal, sample_rate = load_audio_file(stereo_sample_file)
        self.assertTrue(np.allclose(signal[:4],
                                    [[33, 38], [35, 36], [29, 34], [36, 31]]))
        self.assertTrue(len(signal) == 182919)
        self.assertTrue(sample_rate == 44100)
        self.assertTrue(signal.shape == (182919, 2))

    def test_start_stop(self):
        # test wave loader
        signal, sample_rate = load_audio_file(sample_file, start=1. / 44100,
                                              stop=5. / 44100)
        self.assertTrue(np.allclose(signal, [-2510, -2484, -2678, -2833]))
        self.assertTrue(len(signal) == 4)
        self.assertTrue(sample_rate == 44100)
        # test ffmpeg loader
        signal, sample_rate = load_audio_file(stereo_sample_file,
                                              start=1. / 44100,
                                              stop=4. / 44100)
        self.assertTrue(np.allclose(signal, [[35, 36], [29, 34], [36, 31]]))
        self.assertTrue(len(signal) == 3)
        self.assertTrue(sample_rate == 44100)

    def test_downmix(self):
        # test wave loader
        signal, sample_rate = load_audio_file(stereo_sample_file,
                                              num_channels=1)
        self.assertTrue(np.allclose(signal[:5],
                                    [35, 35, 31, 33, 33]))
        self.assertTrue(len(signal) == 182919)
        self.assertTrue(sample_rate == 44100)
        self.assertTrue(signal.shape == (182919, ))
        # test ffmpeg loader
        signal, sample_rate = load_audio_file(stereo_sample_file,
                                              num_channels=1)
        # results are rounded differently, thus allow atol=1
        self.assertTrue(np.allclose(signal[:5], [35, 35, 31, 33, 33], atol=1))
        # avconv results in a different length of 182909 samples
        self.assertTrue(np.allclose(len(signal), 182919, atol=10))
        self.assertTrue(sample_rate == 44100)
        # test clipping
        f = pj(AUDIO_PATH, 'stereo_chirp.wav')
        signal, _ = load_audio_file(f, num_channels=1)
        signal_stereo, _ = load_audio_file(f)
        # sanity checks
        self.assertTrue(np.allclose(signal_stereo[:, 0], signal_stereo[:, 1]))
        self.assertTrue(np.allclose(signal_stereo.mean(axis=1),
                                    signal_stereo[:, 0]))
        # check clipping
        self.assertTrue(np.allclose(signal, signal_stereo[:, 1]))

    def test_upmix(self):
        signal, sample_rate = load_audio_file(sample_file, num_channels=2)
        self.assertTrue(np.allclose(signal[:5],
                                    [[-2494, -2494], [-2510, -2510],
                                     [-2484, -2484], [-2678, -2678],
                                     [-2833, -2833]]))
        self.assertTrue(len(signal) == 123481)
        self.assertTrue(sample_rate == 44100)
        self.assertTrue(signal.shape == (123481, 2))

    def test_resample(self):
        # method must chose ffmpeg loader
        signal, sample_rate = load_audio_file(stereo_sample_file,
                                              sample_rate=22050)
        self.assertTrue(sample_rate == 22050)
        # avconv does round differently, thus allow atol=1
        # result: [[33, 38], [33, 33], [36, 31], [35, 35], [32, 35]]
        self.assertTrue(np.allclose(signal[:5], [[34, 38], [32, 33], [37, 31],
                                                 [35, 35], [32, 34]], atol=1))
        # also downmix
        signal, sample_rate = load_audio_file(stereo_sample_file,
                                              sample_rate=22050,
                                              num_channels=1)
        self.assertTrue(np.allclose(signal[:5], [36, 33, 34, 35, 33], atol=1))
        # avconv results in a different length of 91450 samples
        self.assertTrue(np.allclose(len(signal), 91460, atol=10))

    def test_errors(self):
        # file not found
        with self.assertRaises(IOError):
            load_audio_file(pj(AUDIO_PATH, 'sample.flac'))
        # not an audio file
        with self.assertRaises(LoadAudioFileError):
            load_audio_file(pj(DATA_PATH, 'README'))


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
        self.assertIsInstance(result.ndim, int)

    def test_types_array_with_sample_rate(self):
        result = Signal(sig_1d, 1)
        self.assertIsInstance(result, Signal)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(result.dtype == np.float)
        self.assertIsInstance(result.num_samples, int)
        self.assertIsInstance(result.sample_rate, int)
        self.assertIsInstance(result.num_channels, int)
        self.assertIsInstance(result.length, float)
        self.assertIsInstance(result.ndim, int)

    def test_types_file(self):
        result = Signal(sample_file)
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
        # not officially supported, but Signal can handle float sample rates
        self.assertTrue(result.sample_rate == 12.3)
        self.assertTrue(result.num_channels == 2)
        self.assertTrue(result.length == 9 / 12.3)
        self.assertTrue(result.ndim == 2)

    def test_values_file(self):
        result = Signal(sample_file)
        self.assertTrue(np.allclose(result[:5],
                                    [-2494, -2510, -2484, -2678, -2833]))
        self.assertTrue(len(result) == 123481)
        self.assertTrue(result.num_samples == 123481)
        self.assertTrue(result.sample_rate == 44100)
        self.assertTrue(result.num_channels == 1)
        self.assertTrue(result.ndim == 1)
        self.assertTrue(np.allclose(result.length, 2.8))


class TestSignalProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = SignalProcessor()

    def test_types(self):
        self.assertIsInstance(self.processor, SignalProcessor)
        self.assertIsInstance(self.processor, Processor)
        # attributes
        self.assertTrue(self.processor.sample_rate is None)
        self.assertTrue(self.processor.num_channels is None)
        self.assertTrue(self.processor.start is None)
        self.assertTrue(self.processor.stop is None)
        self.assertIsInstance(self.processor.norm, bool)
        self.assertIsInstance(self.processor.gain, float)

    def test_values(self):
        # attributes
        self.assertTrue(self.processor.sample_rate is None)
        self.assertTrue(self.processor.num_channels is None)
        self.assertTrue(self.processor.start is None)
        self.assertTrue(self.processor.stop is None)
        self.assertTrue(self.processor.norm is False)
        self.assertTrue(self.processor.gain == 0)

    def test_process(self):
        result = self.processor.process(sample_file)
        self.assertIsInstance(result, Signal)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(result.dtype == np.int16)
        self.assertTrue(np.allclose(result[:5],
                                    [-2494, -2510, -2484, -2678, -2833]))
        self.assertTrue(len(result) == 123481)
        self.assertTrue(result.min() == -20603)
        self.assertTrue(result.max() == 17977)
        self.assertTrue(result.mean() == -172.88385257650975)
        # attributes
        self.assertTrue(result.sample_rate == 44100)
        # properties
        self.assertTrue(result.num_samples == 123481)
        self.assertTrue(result.num_channels == 1)
        self.assertTrue(np.allclose(result.length, 2.8))

    def test_process_stereo(self):
        self.processor.num_channels = 1
        self.assertTrue(self.processor.num_channels == 1)
        result = self.processor.process(stereo_sample_file)
        self.assertIsInstance(result, Signal)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(result.dtype == np.int16)

    def test_process_norm(self):
        self.processor.norm = True
        self.assertTrue(self.processor.norm is True)
        result = self.processor.process(sample_file)
        self.assertIsInstance(result, Signal)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(result.dtype == np.int16)
        self.assertTrue(np.allclose(result[:5],
                                    [-3966, -3991, -3950, -4259, -4505]))
        self.assertTrue(len(result) == 123481)
        self.assertTrue(result.min() == -32767)
        self.assertTrue(result.max() == 28590)
        self.assertTrue(result.mean() == -274.92599671204476)
        # attributes
        self.assertTrue(result.sample_rate == 44100)
        # properties
        self.assertTrue(result.num_samples == 123481)
        self.assertTrue(result.num_channels == 1)
        self.assertTrue(np.allclose(result.length, 2.8))

    def test_process_gain(self):
        self.processor.gain = -10
        self.assertTrue(self.processor.gain == -10.)
        result = self.processor.process(sample_file)
        self.assertIsInstance(result, Signal)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(result.dtype == np.int16)
        self.assertTrue(np.allclose(result[:5],
                                    [-788, -793, -785, -846, -895]))
        self.assertTrue(len(result) == 123481)
        # attributes
        self.assertTrue(result.sample_rate == 44100)
        # properties
        self.assertTrue(result.num_samples == 123481)
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
        signal = Signal(sample_file)
        result = signal_frame(signal, 0, 4, 2)
        self.assertIsInstance(result, Signal)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(result.dtype == np.int16)
        result = signal_frame(signal, 2000, 400, 200)
        self.assertIsInstance(result, Signal)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(result.dtype == np.int16)
        result = signal_frame(signal, -10, 400, 200)
        self.assertIsInstance(result, Signal)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(result.dtype == np.int16)

    def test_short_input_length(self):
        result = signal_frame(np.arange(4), 0, 10, 5)
        self.assertTrue(np.allclose(result, [0, 0, 0, 0, 0, 0, 1, 2, 3, 0]))
        result = signal_frame(np.arange(4), 1, 10, 5)
        self.assertTrue(np.allclose(result, [0, 1, 2, 3, 0, 0, 0, 0, 0, 0]))
        result = signal_frame(np.arange(4), 2, 10, 5)
        self.assertTrue(np.allclose(result, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        result = signal_frame(np.arange(4), -2, 10, 5)
        self.assertTrue(np.allclose(result, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

    def test_values(self):
        result = signal_frame(np.arange(10), -1, 4, 2)
        self.assertTrue(np.allclose(result, [0, 0, 0, 0]))
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

    def test_stereo_values(self):
        signal = np.tile(np.arange(10)[:, np.newaxis], 2)
        result = signal_frame(signal, 0, 4, 2)
        self.assertTrue(np.allclose(result, [[0, 0], [0, 0], [0, 0], [1, 1]]))
        result = signal_frame(signal, 1, 4, 2)
        self.assertTrue(np.allclose(result, [[0, 0], [1, 1], [2, 2], [3, 3]]))
        result = signal_frame(signal, 2, 4, 2)
        self.assertTrue(np.allclose(result, [[2, 2], [3, 3], [4, 4], [5, 5]]))
        result = signal_frame(signal, 3, 4, 2)
        self.assertTrue(np.allclose(result, [[4, 4], [5, 5], [6, 6], [7, 7]]))
        result = signal_frame(signal, 4, 4, 2)
        self.assertTrue(np.allclose(result, [[6, 6], [7, 7], [8, 8], [9, 9]]))
        result = signal_frame(signal, 5, 4, 2)
        self.assertTrue(np.allclose(result, [[8, 8], [9, 9], [0, 0], [0, 0]]))
        result = signal_frame(signal, 6, 4, 2)
        self.assertTrue(np.allclose(result, [[0, 0], [0, 0], [0, 0], [0, 0]]))

    def test_float_hop_size(self):
        result = signal_frame(np.arange(10), 0, 3.5, 2)
        self.assertTrue(np.allclose(result, [0, 0, 1]))
        result = signal_frame(np.arange(10), 1, 3.5, 2)
        self.assertTrue(np.allclose(result, [1, 2, 3]))
        result = signal_frame(np.arange(10), 2, 3.5, 2)
        self.assertTrue(np.allclose(result, [3, 4, 5]))

    def test_origin(self):
        result = signal_frame(np.arange(10), 0, 4, 2)
        self.assertTrue(np.allclose(result, [0, 0, 0, 1]))
        # positive values shift to the left
        result = signal_frame(np.arange(10), 0, 4, 2, 1)
        self.assertTrue(np.allclose(result, [0, 0, 0, 0]))
        # negative values shift to the right
        result = signal_frame(np.arange(10), 0, 4, 2, -1)
        self.assertTrue(np.allclose(result, [0, 0, 1, 2]))
        result = signal_frame(np.arange(10), 0, 4, 2, -2)
        self.assertTrue(np.allclose(result, [0, 1, 2, 3]))
        result = signal_frame(np.arange(10), 0, 4, 2, -4)
        self.assertTrue(np.allclose(result, [2, 3, 4, 5]))
        # test with float origin with half the size of the frame size
        result = signal_frame(np.arange(10), 0, 5, 2, -2.5)
        self.assertTrue(np.allclose(result, [0, 1, 2, 3, 4]))


# framing classes
class TestFramedSignalClass(unittest.TestCase):

    def test_types(self):
        result = FramedSignal(np.arange(10), 4, 2)
        self.assertIsInstance(result, FramedSignal)
        # attributes
        self.assertIsInstance(result.signal, Signal)
        self.assertIsInstance(result.frame_size, int)
        self.assertIsInstance(result.hop_size, float)
        self.assertIsInstance(result.origin, int)
        self.assertIsInstance(result.num_frames, int)
        # get item
        self.assertIsInstance(result[0], Signal)
        # get slice
        self.assertIsInstance(result[:5], FramedSignal)
        self.assertIsInstance(result[1:2], FramedSignal)
        # properties
        self.assertIsInstance(len(result), int)
        self.assertIsInstance(result.frame_rate, type(None))
        self.assertIsInstance(result.fps, type(None))
        self.assertIsInstance(result.overlap_factor, float)
        self.assertIsInstance(result.shape, tuple)
        self.assertIsInstance(result.ndim, int)

    def test_types_slice(self):
        # get a slice of a FramedSignal
        result = FramedSignal(np.arange(10), 4, 2)[:5]
        self.assertIsInstance(result, FramedSignal)
        # attributes
        self.assertIsInstance(result.signal, Signal)
        self.assertIsInstance(result.frame_size, int)
        self.assertIsInstance(result.hop_size, float)
        self.assertIsInstance(result.origin, int)
        self.assertIsInstance(result.num_frames, int)
        # get item
        self.assertIsInstance(result[0], Signal)
        # get slice
        self.assertIsInstance(result[:2], FramedSignal)
        self.assertIsInstance(result[5:6], FramedSignal)
        # properties
        self.assertIsInstance(len(result), int)
        self.assertIsInstance(result.frame_rate, type(None))
        self.assertIsInstance(result.fps, type(None))
        self.assertIsInstance(result.overlap_factor, float)
        self.assertIsInstance(result.shape, tuple)
        self.assertIsInstance(result.ndim, int)

    def test_types_with_sample_rate(self):
        result = FramedSignal(np.arange(10), 4, 2, sample_rate=1)
        self.assertIsInstance(result, FramedSignal)
        # attributes
        self.assertIsInstance(result.signal, Signal)
        self.assertIsInstance(result.frame_size, int)
        self.assertIsInstance(result.hop_size, float)
        self.assertIsInstance(result.origin, int)
        self.assertIsInstance(result.num_frames, int)
        self.assertIsInstance(result[0], Signal)
        # properties
        self.assertIsInstance(len(result), int)
        self.assertIsInstance(result.frame_rate, float)
        self.assertIsInstance(result.fps, float)
        self.assertIsInstance(result.overlap_factor, float)
        self.assertIsInstance(result.shape, tuple)
        self.assertIsInstance(result.ndim, int)

    def test_types_signal(self):
        signal = Signal(sample_file)
        result = FramedSignal(signal)
        self.assertIsInstance(result, FramedSignal)
        # attributes
        self.assertIsInstance(result.signal, Signal)
        self.assertIsInstance(result.frame_size, int)
        self.assertIsInstance(result.hop_size, float)
        self.assertIsInstance(result.origin, int)
        self.assertIsInstance(result.num_frames, int)
        self.assertIsInstance(result[0], Signal)
        # properties
        self.assertIsInstance(len(result), int)
        self.assertIsInstance(result.frame_rate, float)
        self.assertIsInstance(result.fps, float)
        self.assertIsInstance(result.overlap_factor, float)
        self.assertIsInstance(result.shape, tuple)
        self.assertIsInstance(result.ndim, int)

    def test_types_file(self):
        result = FramedSignal(sample_file)
        self.assertIsInstance(result, FramedSignal)
        # attributes
        self.assertIsInstance(result.signal, Signal)
        self.assertIsInstance(result.frame_size, int)
        self.assertIsInstance(result.hop_size, float)
        self.assertIsInstance(result.origin, int)
        self.assertIsInstance(result.num_frames, int)
        self.assertIsInstance(result[0], Signal)
        # properties
        self.assertIsInstance(len(result), int)
        self.assertIsInstance(result.frame_rate, float)
        self.assertIsInstance(result.fps, float)
        self.assertIsInstance(result.overlap_factor, float)
        self.assertIsInstance(result.shape, tuple)
        self.assertIsInstance(result.ndim, int)

    def test_values_array(self):
        result = FramedSignal(np.arange(10), 4, 2)
        self.assertTrue(np.allclose(result[0], [0, 0, 0, 1]))
        # attributes
        self.assertTrue(result.frame_size == 4)
        self.assertTrue(result.hop_size == 2.)
        self.assertTrue(result.origin == 0)
        self.assertTrue(result.num_frames == 5)
        # properties
        self.assertTrue(len(result) == 5)
        self.assertTrue(result.frame_rate is None)
        self.assertTrue(result.fps is None)
        self.assertTrue(result.overlap_factor == 0.5)
        self.assertTrue(result.shape == (5, 4))
        self.assertTrue(result.ndim == 2)

    def test_values_array_end(self):
        result = FramedSignal(np.arange(10), 4, 2)
        self.assertTrue(result.num_frames == 5)
        result = FramedSignal(np.arange(10), 4, 2, end='extend')
        self.assertTrue(result.num_frames == 6)

    def test_values_array_with_sample_rate(self):
        result = FramedSignal(np.arange(10), 4, 2, sample_rate=4)
        self.assertTrue(np.allclose(result[0], [0, 0, 0, 1]))
        self.assertTrue(np.allclose(result[1], [0, 1, 2, 3]))
        self.assertTrue(np.allclose(result[2], [2, 3, 4, 5]))
        self.assertTrue(np.allclose(result[3], [4, 5, 6, 7]))
        self.assertTrue(np.allclose(result[4], [6, 7, 8, 9]))
        self.assertTrue(np.allclose(result[-1], [6, 7, 8, 9]))
        with self.assertRaises(IndexError):
            result[5]
        # attributes
        self.assertTrue(result.frame_size == 4)
        self.assertTrue(result.hop_size == 2.)
        self.assertTrue(result.origin == 0)
        self.assertTrue(result.num_frames == 5)
        # properties
        self.assertTrue(len(result) == 5)
        self.assertTrue(result.frame_rate == 2)
        self.assertTrue(result.fps == 2)
        self.assertTrue(result.overlap_factor == 0.5)
        self.assertTrue(result.shape == (5, 4))
        self.assertTrue(result.ndim == 2)

    def test_values_slicing(self):
        result = FramedSignal(np.arange(10), 4, 2, sample_rate=4)[1:]
        self.assertTrue(np.allclose(result[0], [0, 1, 2, 3]))
        self.assertTrue(np.allclose(result[1], [2, 3, 4, 5]))
        self.assertTrue(np.allclose(result[2], [4, 5, 6, 7]))
        self.assertTrue(np.allclose(result[-2], [4, 5, 6, 7]))
        self.assertTrue(np.allclose(result[3], [6, 7, 8, 9]))
        self.assertTrue(np.allclose(result[-1], [6, 7, 8, 9]))
        with self.assertRaises(IndexError):
            result[4]
        # attributes
        self.assertTrue(result.frame_size == 4)
        self.assertTrue(result.hop_size == 2.)
        self.assertTrue(result.origin == -2)
        self.assertTrue(result.num_frames == 4)
        # properties
        self.assertTrue(len(result) == 4)
        self.assertTrue(result.shape == (4, 4))
        self.assertTrue(result.frame_rate == 2)
        self.assertTrue(result.fps == 2)
        self.assertTrue(result.overlap_factor == 0.5)
        # other slice
        result = FramedSignal(np.arange(10), 4, 2, sample_rate=4)[2:4]
        self.assertTrue(result.shape == (2, 4))
        self.assertTrue(np.allclose(result[0], [2, 3, 4, 5]))
        self.assertTrue(np.allclose(result[1], [4, 5, 6, 7]))
        with self.assertRaises(IndexError):
            result[2]
        # slices with steps != 1
        with self.assertRaises(ValueError):
            FramedSignal(np.arange(10), 4, 2, sample_rate=4)[2:4:2]
        # only slices with integers should work
        with self.assertRaises(TypeError):
            FramedSignal(np.arange(10), 4, 2, sample_rate=4)['foo':'bar']
        # only slices or integers should work
        with self.assertRaises(TypeError):
            FramedSignal(np.arange(10), 4, 2, sample_rate=4)['bar']

    def test_values_file(self):
        signal = Signal(sample_file)
        result = FramedSignal(sample_file)
        self.assertTrue(np.allclose(result[0][:5], [0, 0, 0, 0, 0]))
        # 3rd frame should start at 3 * 441 - 2048 / 2 = 299
        self.assertTrue(np.allclose(result[3], signal[299: 299 + 2048]))
        # attributes
        self.assertTrue(result.frame_size == 2048)
        self.assertTrue(result.hop_size == 441.)
        self.assertTrue(result.origin == 0)
        self.assertTrue(result.num_frames == 281)
        # properties
        self.assertTrue(len(result) == 281)
        self.assertTrue(result.shape == (281, 2048))
        self.assertTrue(result.frame_rate == 100.)
        self.assertTrue(result.fps == 100.)
        self.assertTrue(result.overlap_factor == 0.78466796875)
        self.assertTrue(result.ndim == 2)

    def test_values_stereo_file(self):
        signal = Signal(stereo_sample_file)
        result = FramedSignal(stereo_sample_file)
        self.assertTrue(np.allclose(result[0][:3], [[0, 0], [0, 0], [0, 0]]))
        # 3rd frame should start at 3 * 441 - 2048 / 2 = 299
        self.assertTrue(np.allclose(result[3], signal[299: 299 + 2048]))
        # attributes
        self.assertTrue(result.frame_size == 2048)
        self.assertTrue(result.hop_size == 441.)
        self.assertTrue(result.origin == 0)
        self.assertTrue(result.num_frames == 415)
        # properties
        self.assertTrue(len(result) == 415)
        self.assertTrue(result.frame_rate == 100)
        self.assertTrue(result.fps == 100)
        self.assertTrue(result.overlap_factor == 0.78466796875)
        self.assertTrue(result.shape == (415, 2048, 2))
        self.assertTrue(result.ndim == 3)

    def test_values_file_origin(self):
        signal = Signal(sample_file)
        # literal origin
        result = FramedSignal(sample_file, origin='online')
        self.assertTrue(result.origin == 1023)
        self.assertTrue(result.num_frames == 281)
        # 6th frame should start at 6 * 441 - 2048 + 1 (ref sample) = 599
        self.assertTrue(np.allclose(result[6], signal[599: 599 + 2048]))
        # literal left origin
        result = FramedSignal(sample_file, origin='left')
        self.assertTrue(result.origin == 1023)
        # positive origin shifts the window to the left
        result = FramedSignal(sample_file, origin=10)
        self.assertTrue(result.origin == 10)
        # literal offline origin
        result = FramedSignal(sample_file, origin='offline')
        self.assertTrue(result.origin == 0)
        # literal center origin
        result = FramedSignal(sample_file, origin='center')
        self.assertTrue(result.origin == 0)
        # literal right origin
        result = FramedSignal(sample_file, origin='right')
        self.assertTrue(result.origin == -1024)
        # literal future origin
        result = FramedSignal(sample_file, origin='future')
        self.assertTrue(result.origin == -1024)

    def test_values_file_start(self):
        signal = Signal(sample_file)
        result = FramedSignal(sample_file, origin=-10)
        # start sample shifted to the right
        self.assertTrue(result.origin == -10)
        self.assertTrue(result.num_frames == 281)
        # 3rd frame should start at 3 * 441 - 2048 / 2 + 10 = 309
        self.assertTrue(np.allclose(result[3], signal[309: 309 + 2048]))

    def test_values_file_fps(self):
        result = FramedSignal(sample_file, fps=200)
        self.assertTrue(result.frame_size == 2048)
        self.assertTrue(result.hop_size == 220.5)
        result = FramedSignal(sample_file, fps=50)
        self.assertTrue(result.frame_size == 2048)
        self.assertTrue(result.hop_size == 882.)


class TestFramedSignalProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = FramedSignalProcessor()

    def test_types(self):
        self.assertIsInstance(self.processor, FramedSignalProcessor)
        self.assertIsInstance(self.processor, Processor)
        result = self.processor.process(sample_file)
        self.assertIsInstance(result, FramedSignal)

    def test_values(self):
        self.assertTrue(self.processor.frame_size == 2048)
        self.assertTrue(self.processor.hop_size == 441.)
        self.assertTrue(self.processor.fps is None)
        self.assertTrue(self.processor.online is False)
        self.assertTrue(self.processor.end == 'normal')

    def test_process(self):
        result = self.processor.process(sample_file)
        self.assertTrue(np.allclose(result[0][:1023], np.zeros(1023)))
        self.assertTrue(np.allclose(result[0][1024], -2494))
        # attributes
        self.assertTrue(result.frame_size == 2048)
        self.assertTrue(result.hop_size == 441.)
        self.assertTrue(result.origin == 0)
        self.assertTrue(result.num_frames == 281)
        # properties
        self.assertTrue(len(result) == 281.)
        self.assertTrue(result.fps == 100.)
        self.assertTrue(result.frame_rate == 100.)
        self.assertTrue(result.overlap_factor == 0.78466796875)
        self.assertTrue(result.shape == (281, 2048))
        self.assertTrue(result.ndim == 2)

    def test_rewrite_values(self):
        self.processor.end = 'bogus'
        self.assertTrue(self.processor.end == 'bogus')

    def test_process_online(self):
        # set online
        self.processor.online = True
        self.assertTrue(self.processor.online)
        result = self.processor.process(sample_file)
        self.assertTrue(np.allclose(result[0][-1], -2494))
        self.assertTrue(len(result) == 281)
        self.assertTrue(result.num_frames == 281)
        # reset online
        self.processor.online = False
        self.assertFalse(self.processor.online)

    def test_process_fps(self):
        # set fps
        self.processor.fps = 200.
        self.assertTrue(self.processor.fps == 200)
        result = self.processor.process(sample_file)
        self.assertTrue(np.allclose(result[0][:1023], np.zeros(1023)))
        self.assertTrue(np.allclose(result[0][1024], -2494))
        self.assertTrue(len(result) == 561)
        self.assertTrue(result.num_frames == 561)
        # reset fps
        self.processor.fps = 100.
        self.assertTrue(self.processor.fps == 100)

    def test_process_end(self):
        # set end
        self.processor.end = 'normal'
        self.assertTrue(self.processor.end == 'normal')
        # test with a file
        result = self.processor.process(sample_file)
        self.assertTrue(np.allclose(result[0][:1023], np.zeros(1023)))
        self.assertTrue(np.allclose(result[0][1024], -2494))
        # properties
        self.assertTrue(result.num_frames == 281)
        # test with an array
        self.processor.frame_size = 10
        self.processor.hop_size = 6
        result = self.processor.process(np.arange(18))
        self.assertTrue(len(result) == 3)
        self.assertTrue(result.num_frames == 3)
        # rewrite the end
        self.processor.end = 'extend'
        result = self.processor.process(np.arange(18))
        self.assertTrue(len(result) == 4)
        self.assertTrue(result.num_frames == 4)
        # test with incorrect end value
        with self.assertRaises(ValueError):
            processor = FramedSignalProcessor(end='bla')
            processor.process(sample_file)
        # reset end
        self.processor.end = 'normal'
        self.assertTrue(self.processor.end == 'normal')

    def test_constant_types(self):
        self.assertIsInstance(FramedSignalProcessor.FRAME_SIZE, int)
        self.assertIsInstance(FramedSignalProcessor.HOP_SIZE, float)
        self.assertIsInstance(FramedSignalProcessor.FPS, float)
        self.assertIsInstance(FramedSignalProcessor.START, int)
        self.assertIsInstance(FramedSignalProcessor.END_OF_SIGNAL, str)

    def test_constant_values(self):
        self.assertTrue(FramedSignalProcessor.FRAME_SIZE == 2048)
        self.assertTrue(FramedSignalProcessor.HOP_SIZE == 441.)
        self.assertTrue(FramedSignalProcessor.FPS == 100.)
        self.assertTrue(FramedSignalProcessor.START == 0)
        self.assertTrue(FramedSignalProcessor.END_OF_SIGNAL == 'normal')
