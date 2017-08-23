# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.features.drums module.

"""

from __future__ import absolute_import, division, print_function

import unittest
from os.path import join as pj

from . import AUDIO_PATH, ACTIVATIONS_PATH

from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.spectrogram import (
    LogarithmicFilteredSpectrogramProcessor,
    SpectrogramDifferenceProcessor)

from madmom.ml.nn import NeuralNetwork
from madmom.features import Activations
from madmom.features.drums import *
from madmom.features.drums import _crnn_drum_processor_pad

sample_file = pj(AUDIO_PATH, 'sample.wav')
sample_drum_gru = Activations(pj(ACTIVATIONS_PATH, 'sample.drums_rnn_gru.npz'))


class TestSpectralOnsetProcessorClass(unittest.TestCase):
    def setUp(self):
        self.processor = CRNNDrumProcessor()

    def test_processors(self):
        proc = CRNNDrumProcessor()
        self.assertIsInstance(proc.processors[0].processors[0],
                              SignalProcessor)
        self.assertIsInstance(proc.processors[0].processors[1],
                              FramedSignalProcessor)
        self.assertIsInstance(proc.processors[0].processors[2],
                              ShortTimeFourierTransformProcessor)
        self.assertIsInstance(proc.processors[0].processors[3],
                              LogarithmicFilteredSpectrogramProcessor)
        self.assertIsInstance(proc.processors[0].processors[4],
                              SpectrogramDifferenceProcessor)
        self.assertEqual(proc.processors[0].processors[5],
                         _crnn_drum_processor_pad)
        self.assertIsInstance(proc.processors[1], NeuralNetwork)

    def test_process(self):
        act = self.processor(sample_file)
        self.assertTrue(np.allclose(act, sample_drum_gru))


class TestDrumPeakPickingProcessorClass(unittest.TestCase):
    def setUp(self):
        self.processor = DrumPeakPickingProcessor(
            fps=sample_drum_gru.fps)
        self.online_processor = DrumPeakPickingProcessor(
            online=True,
            fps=sample_drum_gru.fps)
        self.sample_rnn_result = [[0.13, 0.],
                                  [0.13, 2.],
                                  [0.48, 2.],
                                  [0.65, 0.],
                                  [0.8, 0.],
                                  [1.16, 0.],
                                  [1.16, 2.],
                                  [1.52, 0.],
                                  [1.66, 1.],
                                  [1.84, 0.],
                                  [1.84, 2.],
                                  [2.18, 1.],
                                  [2.7, 0.]]

    def test_online_parameters(self):
        self.assertEqual(self.online_processor.smooth, 0)
        self.assertEqual(self.online_processor.post_avg, 0)
        self.assertEqual(self.online_processor.post_max, 0)

    def test_process(self):
        onsets = self.processor(sample_drum_gru)
        self.assertTrue(np.allclose(onsets, self.sample_rnn_result))

    def test_process_online(self):
        # process everything at once
        drums = self.online_processor(sample_drum_gru)
        self.assertTrue(np.allclose(drums, self.sample_rnn_result))
        # results must be the same if processed a second time
        onsets_1 = self.online_processor(sample_drum_gru)
        self.assertTrue(np.allclose(onsets_1, self.sample_rnn_result))
