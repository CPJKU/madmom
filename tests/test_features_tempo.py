# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.ml.hmm module.

"""

from __future__ import absolute_import, division, print_function

import unittest
from . import ACTIVATIONS_PATH, DETECTIONS_PATH
from madmom.features.tempo import *

act_file = np.load("%s/sample.beats_blstm_2013.npz" % ACTIVATIONS_PATH)
act = act_file['activations'].astype(np.float)
fps = float(act_file['fps'])

COMB_TEMPI = np.array([[176.47, 0.348308], [117.65, 0.160887],
                       [82.19, 0.107172], [240, 0.106155],
                       [93.75, 0.102080], [52.17, 0.087847],
                       [67.42, 0.087548]])

HIST = interval_histogram_comb(act, 0.79, min_tau=24, max_tau=150)


class TestIntervalHistogramAcfFunction(unittest.TestCase):

    def test_values(self):
        hist = interval_histogram_acf(act, min_tau=24, max_tau=150)
        self.assertTrue(np.allclose(hist[0][:6], [0.0131512, 0.01230626,
                                                  0.01338983, 0.01715117,
                                                  0.02519087, 0.03996497]))
        self.assertTrue(np.allclose(hist[1], np.arange(24, 151)))


class TestIntervalHistogramCombFunction(unittest.TestCase):

    def test_values(self):
        hist = interval_histogram_comb(act.astype(np.float), 0.79,
                                       min_tau=24, max_tau=150)
        self.assertTrue(np.allclose(hist[0][:6], [1.42615775, 1.0374281,
                                                  1.2080798, 1.19360007,
                                                  1.19332424, 1.03763841]))
        self.assertTrue(np.allclose(hist[1], np.arange(24, 151)))

# class TestIntervalHistogramDbnFunction(unittest.TestCase):
#
#     def test_values(self):
#         result = interval_histogram_dbn()
#         self.assertTrue(result, [])


class TestSmoothHistogramFunction(unittest.TestCase):

    def test_values(self):
        hist = smooth_histogram(HIST, 3)
        self.assertTrue(np.allclose(hist[0][:6], [1.509152, 1.2481671,
                                                  1.38656206, 1.38571239,
                                                  1.37182331, 1.21616335]))
        self.assertTrue(np.allclose(hist[1], HIST[1]))


class TestDominantIntervalFunction(unittest.TestCase):

    def test_values(self):
        result = dominant_interval(HIST)
        self.assertTrue(result == 34)


class TestDetectTempoFunction(unittest.TestCase):

    def test_values(self):
        result = detect_tempo(HIST, fps)
        self.assertTrue(np.allclose(result[:6], [[176.47, 0.072],
                                                 [115.38, 0.041],
                                                 [86.96, 0.034],
                                                 [58.25, 0.033],
                                                 [89.55, 0.031],
                                                 [60, 0.028]], atol=0.1))


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
        self.assertTrue(self.processor.hist_smooth == 7)
        self.assertTrue(self.processor.alpha == 0.79)
        self.assertTrue(self.processor.fps == 100)
        self.assertTrue(self.processor.min_interval == 24)
        self.assertTrue(self.processor.max_interval == 150)

    def test_process(self):
        tempi = self.processor(act)
        self.assertTrue(np.allclose(tempi, COMB_TEMPI, atol=0.01))


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
        result = write_tempo(COMB_TEMPI[5], self.out_file)
        self.assertTrue(np.allclose(result, [52.17, 104.34, 1], atol=0.001))
        # multiple tempi given
        result = write_tempo(COMB_TEMPI, DETECTIONS_PATH + 'sample.tempo.txt')
        self.assertTrue(np.allclose(result, [176.47, 117.65, 0.684],
                                    atol=0.001))

    def test_values_mirex(self):
        # multiple tempi given
        result = write_tempo(COMB_TEMPI, self.out_file, mirex=True)
        print(result)
        self.assertTrue(np.allclose(result, [117.65, 176.47, 0.316],
                                    atol=0.001))
