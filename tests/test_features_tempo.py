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
COMB_TEMPI_ONLINE = [[176.470588, 0.289414003], [115.384615, 0.124638601],
                     [230.769231, 0.0918372569], [84.5070423, 0.0903815502],
                     [75.0000000, 0.0713704506], [53.5714286, 0.0701783497],
                     [65.9340659, 0.0696296514], [49.1803279, 0.0676349815],
                     [61.2244898, 0.0646209647], [40.8163265, 0.0602941909]]
ACF_TEMPI = np.array([[176.470, 0.246], [86.956, 0.226], [58.823, 0.181],
                      [43.795, 0.137], [115.384, 0.081], [70.588, 0.067],
                      [50.847, 0.058]])
ACF_TEMPI_ONLINE = [[176.470588, 0.253116038], [88.2352941, 0.231203195],
                    [58.8235294, 0.187827698], [43.7956204, 0.139373027],
                    [115.384615, 0.0749783568], [69.7674419, 0.0599632291],
                    [50.4201681, 0.0535384559]]
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
        self.assertIsInstance(self.processor.fps, float)
        self.assertIsInstance(self.processor.histogram_processor,
                              TempoHistogramProcessor)

    def test_values(self):
        self.assertTrue(self.processor.method == 'comb')
        self.assertTrue(self.processor.min_bpm == 40)
        self.assertTrue(self.processor.max_bpm == 250)
        self.assertTrue(self.processor.act_smooth == 0.14)
        self.assertTrue(self.processor.hist_smooth == 9)
        self.assertTrue(self.processor.fps == 100)
        # test default values of the histogram processor
        self.assertTrue(self.processor.histogram_processor.alpha == 0.79)
        self.assertTrue(self.processor.histogram_processor.min_interval == 24)
        self.assertTrue(self.processor.histogram_processor.max_interval == 150)

    def test_process(self):
        tempi = self.processor(act)
        self.assertTrue(np.allclose(tempi, COMB_TEMPI, atol=0.01))

    def test_process_online(self):
        processor = TempoEstimationProcessor(fps=fps, online=True)
        tempi = [processor.process_online(np.atleast_1d(a), reset=False)
                 for a in act]
        self.assertTrue(np.allclose(tempi[-1], COMB_TEMPI_ONLINE))
        # with resetting results are the same
        processor.reset()
        tempi = [processor.process_online(np.atleast_1d(a), reset=False)
                 for a in act]
        self.assertTrue(np.allclose(tempi[-1], COMB_TEMPI_ONLINE))
        # without resetting results are different
        tempi = [processor.process_online(np.atleast_1d(a), reset=False)
                 for a in act]
        self.assertTrue(np.allclose(tempi[-1][:3], [[176.470588, 0.31322337],
                                                    [85.7142857, 0.11437361],
                                                    [115.384615, 0.10919612]]))


class TestCombFilterTempoHistogramProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = CombFilterTempoHistogramProcessor(fps=fps)
        self.online_processor = CombFilterTempoHistogramProcessor(fps=fps,
                                                                  online=True)

    def test_types(self):
        self.assertIsInstance(self.processor.min_bpm, float)
        self.assertIsInstance(self.processor.max_bpm, float)
        self.assertIsInstance(self.processor.alpha, float)
        self.assertIsInstance(self.processor.fps, float)
        # properties
        self.assertIsInstance(self.processor.min_interval, int)
        self.assertIsInstance(self.processor.max_interval, int)

    def test_values(self):
        self.assertTrue(self.processor.min_bpm == 40)
        self.assertTrue(self.processor.max_bpm == 250)
        self.assertTrue(self.processor.alpha == 0.79)
        self.assertTrue(self.processor.fps == 100)
        self.assertTrue(self.processor.min_interval == 24)
        self.assertTrue(self.processor.max_interval == 150)

    def test_tempo(self):
        tempo_processor = TempoEstimationProcessor(
            histogram_processor=self.processor, fps=fps)
        tempi = tempo_processor(act)
        self.assertTrue(np.allclose(tempi, COMB_TEMPI, atol=0.01))

    def test_tempo_online(self):
        tempo_processor = TempoEstimationProcessor(
            histogram_processor=self.online_processor, fps=fps, online=True)
        tempi = [tempo_processor.process_online(np.atleast_1d(a), reset=False)
                 for a in act]
        self.assertTrue(np.allclose(tempi[-1], COMB_TEMPI_ONLINE))
        # with resetting results are the same
        tempo_processor.reset()
        tempi = [tempo_processor.process_online(np.atleast_1d(a), reset=False)
                 for a in act]
        self.assertTrue(np.allclose(tempi[-1], COMB_TEMPI_ONLINE))
        # without resetting results are different
        tempi = [tempo_processor.process_online(np.atleast_1d(a), reset=False)
                 for a in act]
        self.assertTrue(np.allclose(tempi[-1][:3], [[176.470588, 0.31322337],
                                                    [85.7142857, 0.11437361],
                                                    [115.384615, 0.10919612]]))


class TestACFTempoHistogramProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = ACFTempoHistogramProcessor(fps=fps)
        self.online_processor = ACFTempoHistogramProcessor(fps=fps,
                                                           online=True)

    def test_types(self):
        self.assertIsInstance(self.processor.min_bpm, float)
        self.assertIsInstance(self.processor.max_bpm, float)
        self.assertIsInstance(self.processor.fps, float)
        # properties
        self.assertIsInstance(self.processor.min_interval, int)
        self.assertIsInstance(self.processor.max_interval, int)

    def test_values(self):
        self.assertTrue(self.processor.min_bpm == 40)
        self.assertTrue(self.processor.max_bpm == 250)
        self.assertTrue(self.processor.fps == 100)
        self.assertTrue(self.processor.min_interval == 24)
        self.assertTrue(self.processor.max_interval == 150)

    def test_tempo(self):
        tempo_processor = TempoEstimationProcessor(
            histogram_processor=self.processor, fps=fps)
        tempi = tempo_processor(act)
        self.assertTrue(np.allclose(tempi, ACF_TEMPI, atol=0.01))

    def test_tempo_online(self):
        tempo_processor = TempoEstimationProcessor(
            histogram_processor=self.online_processor, fps=fps, online=True)
        tempi = [tempo_processor.process_online(np.atleast_1d(a), reset=False)
                 for a in act]
        self.assertTrue(np.allclose(tempi[-1], ACF_TEMPI_ONLINE))
        # with resetting results are the same
        tempo_processor.reset()
        tempi = [tempo_processor.process_online(np.atleast_1d(a), reset=False)
                 for a in act]
        self.assertTrue(np.allclose(tempi[-1], ACF_TEMPI_ONLINE))
        # without resetting results are different
        tempi = [tempo_processor.process_online(np.atleast_1d(a), reset=False)
                 for a in act]
        self.assertTrue(np.allclose(tempi[-1][:3], [[176.4705882, 0.2414368],
                                                    [86.95652174, 0.2248635],
                                                    [58.25242718, 0.1878183]]))


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
