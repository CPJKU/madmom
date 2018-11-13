# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.audio.signal module.

"""

from __future__ import absolute_import, division, print_function

import os
import sys
import tempfile
import unittest
from os.path import join as pj

from madmom.io.audio import *
from . import AUDIO_PATH, DATA_PATH
from .test_audio_comb_filters import sig_1d, sig_2d

sample_file = pj(AUDIO_PATH, 'sample.wav')
sample_file_22k = pj(AUDIO_PATH, 'sample_22050.wav')
stereo_sample_file = pj(AUDIO_PATH, 'stereo_sample.wav')
tmp_file = tempfile.NamedTemporaryFile(delete=False).name


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


class TestWriteWaveFileFunction(unittest.TestCase):

    def setUp(self):
        self.signal = Signal(sample_file)
        write_wave_file(self.signal, tmp_file)
        self.result = Signal(sample_file)

    def test_types(self):
        self.assertIsInstance(self.result, Signal)
        self.assertIsInstance(self.result, np.ndarray)
        self.assertTrue(self.result.dtype == np.int16)
        self.assertTrue(type(self.result.sample_rate) == int)

    def test_values(self):
        # test wave loader
        self.assertTrue(np.allclose(self.signal, self.result))
        self.assertTrue(self.result.sample_rate == 44100)
        self.assertTrue(self.result.shape == (123481,))


class TestLoadAudioFileFunction(unittest.TestCase):

    # this tests both madmom.io.audio.load_wave_file() and
    # madmom.io.audio.load_ffmpeg_file() functions via the universal
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
        if sys.version_info[0] == 2:
            # test unicode string type (Python 2 only)
            signal, sample_rate = load_audio_file(unicode(sample_file))

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


# clean up
def teardown_module():
    os.unlink(tmp_file)
