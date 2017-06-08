# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.features.tempo module.

"""

from __future__ import absolute_import, division, print_function

import unittest
from os.path import join as pj

from . import ACTIVATIONS_PATH
from madmom.features.tempo import *

act_file = np.load(pj(ACTIVATIONS_PATH, "sample.beats_blstm.npz"))
act = act_file['activations'].astype(np.float)
fps = float(act_file['fps'])

COMB_TEMPI = np.array([[176.470, 0.475], [117.647, 0.177],
                       [240.0, 0.154], [68.966, 0.099], [82.192, 0.096]])

HIST = interval_histogram_comb(act, 0.79, min_tau=24, max_tau=150)


class TestIntervalHistogramAcfFunction(unittest.TestCase):

    def test_values(self):
        hist = interval_histogram_acf(act, min_tau=24, max_tau=150)
        self.assertTrue(np.allclose(hist[0][:6], [0.10034907, 0.10061631,
                                                  0.11078519, 0.13461014,
                                                  0.17694432, 0.24372872]))
        self.assertTrue(np.allclose(hist[1], np.arange(24, 151)))


class TestIntervalHistogramCombFunction(unittest.TestCase):

    def setUp(self):
        self.hist_0_6 = np.array([3.16024358, 2.00690881, 2.52592621,
                                  2.00221429, 1.73527979, 1.47528936])

    def test_values(self):
        hist = interval_histogram_comb(act, 0.79, min_tau=24, max_tau=150)
        self.assertTrue(np.allclose(hist[0][:6], self.hist_0_6))
        self.assertTrue(np.allclose(hist[1], np.arange(24, 151)))

    def test_values_2d(self):
        act_2d = np.vstack((act, act)).T
        hist = interval_histogram_comb(act_2d, 0.79, min_tau=24, max_tau=150)
        # test both channels individually
        self.assertTrue(np.allclose(hist[0][0, :6], self.hist_0_6))
        # 2nd channel is the same
        self.assertTrue(np.allclose(hist[0][1, :6], self.hist_0_6))
        self.assertTrue(np.allclose(hist[1], np.arange(24, 151)))


class TestSmoothHistogramFunction(unittest.TestCase):

    def test_values(self):
        hist = smooth_histogram(HIST, 3)
        self.assertTrue(np.allclose(hist[0][:6], [3.32079628, 2.46180239,
                                                  2.84665606, 2.34311077,
                                                  2.01348008, 1.73241048]))
        self.assertTrue(np.allclose(hist[1], HIST[1]))


class TestDominantIntervalFunction(unittest.TestCase):

    def test_values(self):
        result = dominant_interval(HIST)
        self.assertTrue(result == 34)


class TestDetectTempoFunction(unittest.TestCase):

    def test_values(self):
        result = detect_tempo(HIST, fps)
        self.assertTrue(np.allclose(result[:5], [[176.47, 0.169],
                                                 [117.65, 0.064],
                                                 [250.0, 0.051],
                                                 [230.77, 0.041],
                                                 [105.26, 0.040]], atol=0.1))


class TestTempoEstimationProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = TempoEstimationProcessor(fps=fps)

    def test_types(self):
        self.assertIsInstance(self.processor.method, str)
        self.assertIsInstance(self.processor.min_bpm, float)
        self.assertIsInstance(self.processor.max_bpm, float)
        self.assertIsInstance(self.processor.act_smooth, float)
        self.assertIsInstance(self.processor.hist_smooth, int)
        self.assertIsInstance(self.processor.alpha, float)
        self.assertIsInstance(self.processor.fps, float)
        # properties
        self.assertIsInstance(self.processor.min_interval, int)
        self.assertIsInstance(self.processor.max_interval, int)

    def test_values(self):
        self.assertTrue(self.processor.method == 'comb')
        self.assertTrue(self.processor.min_bpm == 40)
        self.assertTrue(self.processor.max_bpm == 250)
        self.assertTrue(self.processor.act_smooth == 0.14)
        self.assertTrue(self.processor.hist_smooth == 9)
        self.assertTrue(self.processor.alpha == 0.79)
        self.assertTrue(self.processor.fps == 100)
        self.assertTrue(self.processor.min_interval == 24)
        self.assertTrue(self.processor.max_interval == 150)

    def test_process(self):
        tempi = self.processor(act)
        self.assertTrue(np.allclose(tempi, COMB_TEMPI, atol=0.01))


class TestCombFilterTempoEstimationProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = CombFilterTempoEstimationProcessor(fps=fps)

    def test_types(self):
        self.assertIsInstance(self.processor.min_bpm, float)
        self.assertIsInstance(self.processor.max_bpm, float)
        self.assertIsInstance(self.processor.act_smooth, float)
        self.assertIsInstance(self.processor.hist_smooth, int)
        self.assertIsInstance(self.processor.alpha, float)
        self.assertIsInstance(self.processor.fps, float)
        # properties
        self.assertIsInstance(self.processor.min_interval, int)
        self.assertIsInstance(self.processor.max_interval, int)

    def test_values(self):
        self.assertTrue(self.processor.min_bpm == 40)
        self.assertTrue(self.processor.max_bpm == 250)
        self.assertTrue(self.processor.act_smooth == 0.14)
        self.assertTrue(self.processor.hist_smooth == 9)
        self.assertTrue(self.processor.alpha == 0.79)
        self.assertTrue(self.processor.fps == 100)
        self.assertTrue(self.processor.min_interval == 24)
        self.assertTrue(self.processor.max_interval == 150)

    def test_process(self):
        tempi = self.processor(act)
        self.assertTrue(np.allclose(tempi, COMB_TEMPI, atol=0.01))

    def test_process_online(self):
        processor = CombFilterTempoEstimationProcessor(fps=fps, online=True)
        tempi = [processor.process_online(np.atleast_1d(a), reset=False)
                 for a in act]
        self.assertTrue(np.allclose(tempi[-1][0:3],
                                    [[176.47058824, 0.289414],
                                     [115.38461538, 0.124638],
                                     [230.76923076, 0.091837]]))
        # with resetting results are the same
        processor.reset()
        tempi = [processor.process_online(np.atleast_1d(a), reset=False)
                 for a in act]
        self.assertTrue(np.allclose(tempi[-1][0:3],
                                    [[176.47058824, 0.289414],
                                     [115.38461538, 0.124638],
                                     [230.76923076, 0.091837]]))
        # without resetting results are different
        tempi = [processor.process_online(np.atleast_1d(a), reset=False)
                 for a in act]
        self.assertTrue(np.allclose(tempi[-1][0:3],
                                    [[176.470588, 0.31322337],
                                     [85.7142857, 0.11437361],
                                     [115.384615, 0.10919612]]))


class TestWriteTempoFunction(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.out_file = tempfile.SpooledTemporaryFile()

    def test_types(self):
        # must work with 2d arrays
        write_tempo(COMB_TEMPI[:1], self.out_file)
        # but also with 1d arrays
        write_tempo(COMB_TEMPI[0], self.out_file)

    def test_values(self):
        # only one tempo given (>68 bpm)
        result = write_tempo(COMB_TEMPI[0], self.out_file)
        self.assertTrue(np.allclose(result, [176.47, 88.235, 1], atol=0.001))
        # only one tempo given (<68 bpm)
        result = write_tempo(COMB_TEMPI[3] / 2, self.out_file)
        self.assertTrue(np.allclose(result, [34.483, 68.966, 1], atol=0.01))
        # multiple tempi given
        result = write_tempo(COMB_TEMPI, self.out_file)
        self.assertTrue(np.allclose(result, [176.47, 117.647, 0.728],
                                    atol=0.001))

    def test_values_mirex(self):
        # multiple tempi given
        result = write_tempo(COMB_TEMPI, self.out_file, mirex=True)
        self.assertTrue(np.allclose(result, [117.647, 176.47, 0.271],
                                    atol=0.001))
