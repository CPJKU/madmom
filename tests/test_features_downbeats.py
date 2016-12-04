# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.features.beats module.

"""

from __future__ import absolute_import, division, print_function

import unittest
from . import AUDIO_PATH, ANNOTATIONS_PATH
from madmom.audio.chroma import CLPChroma
from madmom.features.downbeats import *
from os.path import join as pj


sample_file = pj(AUDIO_PATH, "sample.wav")
sample_beats = pj(ANNOTATIONS_PATH, "sample.beats")


class TestBeatSyncProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = BeatSyncProcessor(beat_subdivision=2,
                                           sum_func=np.mean, fps=100)

    def test_process(self):
        data = list([CLPChroma(sample_file, fps=100)])
        data.append(np.loadtxt(sample_beats))
        feat_sync = self.processor.process(data)
        print(feat_sync[0, :])
        target = [0.28231065, 0.14807641, 0.22790557, 0.41458403, 0.15966462,
                  0.22294236, 0.1429988, 0.16661506, 0.5978227, 0.24039252,
                  0.23444982, 0.21910049, 0.25676728, 0.13382165, 0.19957431,
                  0.47225753, 0.18936998, 0.17014103, 0.14079712, 0.18317944,
                  0.60692955, 0.20016842, 0.17619181, 0.24408179]
        self.assertTrue(np.allclose(feat_sync[0, :], target, rtol=1e-3))


class TestRNNBarTrackingProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = RNNBarProcessor()

    def test_process(self):
        # check RNN activations and feature computation
        params = {'load_beats': True, 'beat_files': list([sample_beats]),
                  'beat_suffix': '.beats'}
        activations, beats = self.processor.process(sample_file, params)
        print(activations)
        target = [0.4819403, 0.1262536, 0.1980488]
        self.assertTrue(np.allclose(activations, target, rtol=1e-4))


class TestDBNBarTrackingProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = DBNBarTrackingProcessor()

    def test_process(self):
        # check DBN output
        in_act = np.array([0.48208269, 0.12524545, 0.1998145])
        beats = np.array([0.0913, 0.7997, 1.4806, 2.1478])
        downbeats = self.processor.process(list([in_act, beats]))
        target = np.array([[0.0913, 1.],
                           [0.7997, 2.],
                           [1.4806, 3.],
                           [2.1478, 1.]])
        self.assertTrue(np.allclose(downbeats, target))
        path, log = self.processor.hmm.viterbi(in_act)
        self.assertTrue(np.allclose(log, -12.222513053716115))

