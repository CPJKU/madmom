# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.features.tempo module.

"""

from __future__ import absolute_import, division, print_function

import unittest
from os.path import join as pj

from madmom.features.tempo import *
from madmom.io import write_tempo, load_tempo
from . import ACTIVATIONS_PATH

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
DBN_TEMPI = np.array([[176.470, 1]])
DBN_TEMPI_ONLINE = [[176.470588, 0.580877380], [86.9565217, 0.244729904],
                    [74.0740741, 0.127887992], [40.8163265, 0.0232523621],
                    [250.000000, 0.0232523621]]
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
        self.online_processor = TempoEstimationProcessor(fps=fps, online=True)

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
        # process all activations at once
        tempi = self.online_processor(act, reset=False)
        self.assertTrue(np.allclose(tempi, COMB_TEMPI_ONLINE))
        # process frame by frame; with resetting results are the same
        self.online_processor.reset()
        tempi = [self.online_processor(np.atleast_1d(a), reset=False)
                 for a in act]
        self.assertTrue(np.allclose(tempi[-1], COMB_TEMPI_ONLINE))
        # without resetting results are different
        tempi = [self.online_processor(np.atleast_1d(a), reset=False)
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
        # process all activations at once
        tempi = tempo_processor(act, reset=False)
        self.assertTrue(np.allclose(tempi, COMB_TEMPI_ONLINE))
        # process frame by frame; with resetting results are the same
        tempo_processor.reset()
        tempi = [tempo_processor(np.atleast_1d(a), reset=False) for a in act]
        self.assertTrue(np.allclose(tempi[-1], COMB_TEMPI_ONLINE))
        # without resetting results are different
        tempi = [tempo_processor(np.atleast_1d(a), reset=False) for a in act]
        self.assertTrue(np.allclose(tempi[-1][:3], [[176.470588, 0.31322337],
                                                    [85.7142857, 0.11437361],
                                                    [115.384615, 0.10919612]]))

    def test_process(self):
        hist, delays = self.processor(act)
        self.assertTrue(np.allclose(delays, np.arange(24, 151)))
        self.assertTrue(np.allclose(hist.max(), 10.5064280455))
        self.assertTrue(np.allclose(hist.min(), 1.23250838113))
        self.assertTrue(np.allclose(hist.argmax(), 10))
        self.assertTrue(np.allclose(hist.argmin(), 44))
        self.assertTrue(np.allclose(np.sum(hist), 231.568316445))
        self.assertTrue(np.allclose(np.mean(hist), 1.82337257043))
        self.assertTrue(np.allclose(np.median(hist), 1.48112542203))

    def test_process_online(self):
        # offline results
        hist_offline, delays_offline = self.processor(act)
        # calling with all activations at once
        hist, delays = self.online_processor(act)
        # result must be the same as for offline processing
        self.assertTrue(np.allclose(hist, hist_offline))
        self.assertTrue(np.allclose(delays, delays_offline))
        # calling frame by frame after resetting
        self.online_processor.reset()
        result = [self.online_processor(np.atleast_1d(a), reset=False)
                  for a in act]
        # the final result must be the same as for offline processing
        hist, delays = result[-1]
        hist_, delays_ = self.processor(act)
        self.assertTrue(np.allclose(hist, hist_))
        self.assertTrue(np.allclose(delays, delays_))
        # result after 100 frames
        hist, delays = result[99]
        self.assertTrue(np.allclose(hist.max(), 2.03108930086))
        self.assertTrue(np.allclose(hist.min(), 1.23250838113))
        self.assertTrue(np.allclose(hist.argmax(), 12))
        self.assertTrue(np.allclose(hist.argmin(), 44))
        self.assertTrue(np.allclose(np.sum(hist), 175.034206851))
        self.assertTrue(np.allclose(np.mean(hist), 1.37822210119))
        self.assertTrue(np.allclose(np.median(hist), 1.23250838113))
        # the final result must be the same as for offline processing
        hist, delays = result[-1]
        self.assertTrue(np.allclose(hist, hist_offline))
        self.assertTrue(np.allclose(delays, delays_offline))
        # results must be different without resetting
        result = [self.online_processor(np.atleast_1d(a), reset=False)
                  for a in act]
        hist, delays = result[-1]
        self.assertTrue(np.allclose(hist.max(), 18.1385269354))
        self.assertTrue(np.allclose(hist.min(), 1.23250838113))
        self.assertTrue(np.allclose(hist.argmax(), 11))
        self.assertTrue(np.allclose(hist.argmin(), 72))
        self.assertTrue(np.allclose(np.sum(hist), 332.668525522))
        self.assertTrue(np.allclose(np.mean(hist), 2.61943720884))
        self.assertTrue(np.allclose(np.median(hist), 1.96220625848))


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
        # process all activations at once
        tempi = tempo_processor(act, reset=False)
        self.assertTrue(np.allclose(tempi, ACF_TEMPI_ONLINE))
        # process frame by frame; with resetting results are the same
        tempo_processor.reset()
        tempi = [tempo_processor(np.atleast_1d(a), reset=False) for a in act]
        self.assertTrue(np.allclose(tempi[-1], ACF_TEMPI_ONLINE))
        # without resetting results are different
        tempi = [tempo_processor(np.atleast_1d(a), reset=False) for a in act]
        self.assertTrue(np.allclose(tempi[-1][:3], [[176.4705882, 0.2414368],
                                                    [86.95652174, 0.2248635],
                                                    [58.25242718, 0.1878183]]))

    def test_process(self):
        hist, delays = self.processor(act)
        self.assertTrue(np.allclose(delays, np.arange(24, 151)))
        self.assertTrue(np.allclose(hist.max(), 0.772242703961))
        self.assertTrue(np.allclose(hist.min(), 0.0550745515184))
        self.assertTrue(np.allclose(hist.argmax(), 11))
        self.assertTrue(np.allclose(hist.argmin(), 103))
        self.assertTrue(np.allclose(np.sum(hist), 28.4273056042))
        self.assertTrue(np.allclose(np.mean(hist), 0.223837052001))
        self.assertTrue(np.allclose(np.median(hist), 0.147368463433))

    def test_process_online(self):
        # offline results
        hist_offline, delays_offline = self.processor(act)
        # calling with all activations at once
        hist, delays = self.online_processor(act)
        # result must be the same as for offline processing
        self.assertTrue(np.allclose(hist, hist_offline))
        self.assertTrue(np.allclose(delays, delays_offline))
        # calling frame by frame after resetting
        self.online_processor.reset()
        result = [self.online_processor(np.atleast_1d(a), reset=False)
                  for a in act]
        # the final result must be the same as for offline processing
        hist, delays = result[-1]
        hist_, delays_ = self.processor(act)
        self.assertTrue(np.allclose(hist, hist_))
        self.assertTrue(np.allclose(delays, delays_))
        # result after 100 frames
        hist, delays = result[99]
        self.assertTrue(np.allclose(hist.max(), 0.19544739526))
        self.assertTrue(np.allclose(hist.min(), 0))
        self.assertTrue(np.allclose(hist.argmax(), 46))
        self.assertTrue(np.allclose(hist.argmin(), 76))
        self.assertTrue(np.allclose(np.sum(hist), 3.58546628975))
        self.assertTrue(np.allclose(np.mean(hist), 0.0282320180295))
        self.assertTrue(np.allclose(np.median(hist), 0.00471735456373))
        # the final result must be the same as for offline processing
        hist, delays = result[-1]
        self.assertTrue(np.allclose(hist, hist_offline))
        self.assertTrue(np.allclose(delays, delays_offline))


class TestDBNTempoHistogramProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = DBNTempoHistogramProcessor(fps=fps)
        self.online_processor = DBNTempoHistogramProcessor(fps=fps,
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
        self.assertTrue(np.allclose(tempi, DBN_TEMPI, atol=0.01))

    def test_tempo_online(self):
        tempo_processor = TempoEstimationProcessor(
            histogram_processor=self.online_processor, fps=fps, online=True)
        # process all activations at once
        tempi = tempo_processor(act, reset=False)
        self.assertTrue(np.allclose(tempi, DBN_TEMPI_ONLINE))
        # process frame by frame; with resetting results are the same
        tempo_processor.reset()
        tempo_processor.reset()
        tempi = [tempo_processor(np.atleast_1d(a), reset=False) for a in act]
        self.assertTrue(np.allclose(tempi[-1], DBN_TEMPI_ONLINE))
        # without resetting results are different
        tempi = [tempo_processor(np.atleast_1d(a), reset=False) for a in act]
        self.assertTrue(np.allclose(tempi[-1][:3],
                                    [[176.4705882, 0.472499032],
                                     [84.5070423, 0.432130320],
                                     [74.0740741, 0.0699384753]]))

    def test_process(self):
        hist, delays = self.processor(act)
        self.assertTrue(np.allclose(delays, np.arange(24, 151)))
        self.assertTrue(np.allclose(hist.max(), 281))
        self.assertTrue(np.allclose(hist.min(), 0))
        self.assertTrue(np.allclose(hist.argmax(), 10))
        self.assertTrue(np.allclose(hist.argmin(), 0))
        self.assertTrue(np.allclose(np.sum(hist), 281))
        self.assertTrue(np.allclose(np.mean(hist), 2.2125984252))
        self.assertTrue(np.allclose(np.median(hist), 0))

    def test_process_online(self):
        hist, delays = self.online_processor(act)
        self.assertTrue(np.allclose(delays, np.arange(24, 151)))
        self.assertTrue(np.allclose(hist.max(), 106))
        self.assertTrue(np.allclose(hist.min(), 0))
        self.assertTrue(np.allclose(hist.argmax(), 10))
        self.assertTrue(np.allclose(hist.argmin(), 1))
        self.assertTrue(np.allclose(np.sum(hist), 281))
        self.assertTrue(np.allclose(np.mean(hist), 2.2125984252))
        self.assertTrue(np.allclose(np.median(hist), 0))


class TestWriteTempoFunction(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.out_file = tempfile.NamedTemporaryFile(delete=False).name

    def test_types(self):
        # must work with 2d arrays
        write_tempo(COMB_TEMPI[:1], self.out_file)
        # but also with 1d arrays
        write_tempo(COMB_TEMPI[0], self.out_file)

    def test_values(self):
        # only one tempo given (>68 bpm)
        write_tempo(COMB_TEMPI[0], self.out_file)
        result = load_tempo(self.out_file)
        self.assertTrue(np.allclose(result, [[176.47, 1]],
                                    atol=1e-4, equal_nan=True))
        # only one tempo given (<68 bpm)
        write_tempo(COMB_TEMPI[3] / 2, self.out_file)
        result = load_tempo(self.out_file)
        self.assertTrue(np.allclose(result, [[34.48, 1]],
                                    atol=1e-4, equal_nan=True))
        # multiple tempi given
        write_tempo(COMB_TEMPI, self.out_file)
        result = load_tempo(self.out_file)
        self.assertTrue(np.allclose(result, [[176.47, 0.73],
                                             [117.65, 0.27]], atol=1e-4))

    def test_values_mirex(self):
        # multiple tempi given
        write_tempo(COMB_TEMPI, self.out_file, mirex=True)
        result = load_tempo(self.out_file)
        self.assertTrue(np.allclose(result, [[117.65, 0.27],
                                             [176.47, 0.73]], atol=1e-4))

    def tearDown(self):
        import os
        os.unlink(self.out_file)
