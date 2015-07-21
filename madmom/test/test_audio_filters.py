# encoding: utf-8
"""
This file contains tests for the madmom.audio.filters module.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""
# pylint: skip-file

import unittest
import types
import cPickle
import tempfile

from madmom.audio.filters import *

# Mel frequency scale
HZ = np.asarray([20, 258.7484, 576.6645, 1000])
MEL = np.asarray([31.749, 354.5, 677.25, 1000])

FFT_FREQS_1024 = np.fft.fftfreq(2048, 1. / 44100)[:1024]


class TestHz2MelFunction(unittest.TestCase):

    def test_types(self):
        # array
        result = hz2mel(HZ)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float)
        # list
        result = hz2mel([1, 2.])
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float)
        # single value
        result = hz2mel(1.1)
        self.assertIsInstance(result, float)
        self.assertEqual(result.dtype, np.float)

    def test_value(self):
        self.assertTrue(np.allclose(hz2mel(HZ), MEL))


class TestMel2HzFunction(unittest.TestCase):

    def test_types(self):
        # array
        result = mel2hz(HZ)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float)
        # list
        result = mel2hz([1, 2.])
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float)
        # single value
        result = mel2hz(1.1)
        self.assertIsInstance(result, float)
        self.assertEqual(result.dtype, np.float)

    def test_values(self):
        self.assertTrue(np.allclose(mel2hz(MEL), HZ))


class TestMelFrequenciesFunction(unittest.TestCase):

    def test_types(self):
        result = mel_frequencies(4, 20, 1000)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float)

    def test_values(self):
        # HZ is already mel-scaled, so use this for comparison
        result = mel_frequencies(4, 20, 1000)
        self.assertTrue(np.allclose(result, HZ))
        self.assertEqual(len(result), 4)


# Bark frequency scale
BARK = np.array([20, 100, 200, 300, 400, 510, 630, 770, 920, 1080,
                 1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700,
                 4400, 5300, 6400, 7700, 9500, 12000, 15500])

BARK_DOUBLE = np.array([20, 50, 100, 150, 200, 250, 300, 350, 400, 450,
                        510, 570, 630, 700, 770, 840, 920, 1000, 1080,
                        1170, 1270, 1370, 1480, 1600, 1720, 1850, 2000,
                        2150, 2320, 2500, 2700, 2900, 3150, 3400, 3700,
                        4000, 4400, 4800, 5300, 5800, 6400, 7000, 7700,
                        8500, 9500, 10500, 12000, 13500, 15500])


class TestHz2BarkFunction(unittest.TestCase):

    def test_raises_warning(self):
        with self.assertRaises(NotImplementedError):
            hz2bark(HZ)


class TestBark2HzFunction(unittest.TestCase):

    def test_raises_warning(self):
        with self.assertRaises(NotImplementedError):
            bark2hz(BARK)


class TestBarkFrequenciesFunction(unittest.TestCase):

    def test_types(self):
        # array
        result = bark_frequencies()
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.int)

    def test_values(self):
        # all values
        self.assertTrue(np.allclose(bark_frequencies(), BARK))
        # limit range
        result = bark_frequencies(20, 18000)
        self.assertTrue(np.allclose(result, BARK))
        result = bark_frequencies(40, 18000)
        self.assertTrue(np.allclose(result, BARK[1:]))
        result = bark_frequencies(40, 12000)
        self.assertTrue(np.allclose(result, BARK[1:-1]))
        result = bark_frequencies(40, 11999)
        self.assertTrue(np.allclose(result, BARK[1:-2]))


class TestBarkDoubleFrequenciesFunction(unittest.TestCase):

    def test_types(self):
        # array
        result = bark_double_frequencies()
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.int)

    def test_values(self):
        # all values
        self.assertTrue(np.allclose(bark_double_frequencies(), BARK_DOUBLE))
        # limit range
        result = bark_double_frequencies(20, 18000)
        self.assertTrue(np.allclose(result, BARK_DOUBLE))
        result = bark_double_frequencies(40, 18000)
        self.assertTrue(np.allclose(result, BARK_DOUBLE[1:]))
        result = bark_double_frequencies(40, 12000)
        self.assertTrue(np.allclose(result, BARK_DOUBLE[1:-2]))
        result = bark_double_frequencies(40, 11999)
        self.assertTrue(np.allclose(result, BARK_DOUBLE[1:-3]))


# logarithmic frequency scale stuff
class TestLogFrequenciesFunction(unittest.TestCase):

    def test_num_arguments(self):
        # number of arguments arguments
        with self.assertRaises(TypeError):
            log_frequencies()
        with self.assertRaises(TypeError):
            log_frequencies(1)
        with self.assertRaises(TypeError):
            log_frequencies(1, 2)
        with self.assertRaises(TypeError):
            log_frequencies(1, 2, 3, 4, 5)

    def test_types(self):
        # return types
        result = log_frequencies(12, 20, 20000)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float)

    def test_values(self):
        # all values
        result = log_frequencies(6, 30, 17000)
        self.assertTrue(len(result) == 55)
        # 1 band per octave
        result = log_frequencies(1, 100, 1000)
        self.assertTrue(np.allclose(result, [110., 220., 440., 880.]))
        # different reference frequency
        result = log_frequencies(1, 100, 1000, 441)
        self.assertTrue(np.allclose(result, [110.25, 220.5, 441., 882.]))


class TestSemitoneFrequenciesFunction(unittest.TestCase):

    def test_num_arguments(self):
        # number of arguments
        with self.assertRaises(TypeError):
            semitone_frequencies()
        with self.assertRaises(TypeError):
            semitone_frequencies(1)
        with self.assertRaises(TypeError):
            semitone_frequencies(1, 2, 3, 4)

    def test_types(self):
        # return types
        result = semitone_frequencies(20, 20000)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float)

    def test_values(self):
        # all values
        result = semitone_frequencies(30, 17000)
        self.assertTrue(len(result) == 110)
        # 12 bands per octave
        result = semitone_frequencies(250, 500)
        result_ = [261.6255653, 277.18263098, 293.66476792, 311.12698372,
                   329.62755691, 349.22823143, 369.99442271, 391.99543598,
                   415.30469758, 440, 466.16376152, 493.88330126]
        self.assertTrue(len(result) == 12)
        self.assertTrue(np.allclose(result, result_))
        # different reference frequency
        result = semitone_frequencies(441, 500, 441)
        self.assertTrue(np.allclose(result, [441, 467.22322461, 495.0057633]))


# MIDI


# ERB
ERB = np.asarray([0.77873163, 7.03051042, 11.69607601, 15.62144971])


class TestHz2ErbFunction(unittest.TestCase):

    def test_types(self):
        # array
        result = hz2erb(HZ)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float)
        # list
        result = hz2erb([1, 2.])
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float)
        # single value
        result = hz2erb(1.1)
        self.assertIsInstance(result, float)
        self.assertEqual(result.dtype, np.float)

    def test_value(self):
        self.assertTrue(np.allclose(hz2erb(HZ), ERB))


class TestErb2HzFunction(unittest.TestCase):

    def test_types(self):
        # array
        result = erb2hz(ERB)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float)
        # list
        result = erb2hz([1, 2.])
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float)
        # single value
        result = erb2hz(1.1)
        self.assertIsInstance(result, float)
        self.assertEqual(result.dtype, np.float)

    def test_value(self):
        self.assertTrue(np.allclose(erb2hz(ERB), HZ))


# helper functions for filter creation
class TestFrequencies2BinsFunction(unittest.TestCase):

    freqs = np.asarray([0, 1, 2, 3, 4])

    def test_num_arguments(self):
        # number of arguments arguments
        with self.assertRaises(TypeError):
            frequencies2bins()
        with self.assertRaises(TypeError):
            frequencies2bins(1)
        with self.assertRaises(TypeError):
            frequencies2bins(1, 2, 3)

    def test_types(self):
        result = frequencies2bins([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.int)

    def test_value(self):
        # normal frequencies
        result = frequencies2bins([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
        self.assertTrue(np.allclose(result, [0, 1, 2, 3, 4]))
        # double mapping frequencies
        result = frequencies2bins([0, 1, 1.1, 1.2, 2, 3, 3.5], [0, 1, 2, 3, 4])
        self.assertTrue(np.allclose(result, [0, 1, 1, 1, 2, 3, 4]))
        # higher frequencies should be mapped to the last bin
        result = frequencies2bins([0, 1, 2, 3, 11], [0, 1, 2, 3, 4])
        self.assertTrue(np.allclose(result, [0, 1, 2, 3, 4]))
        # lower frequencies should be mapped to the first bin
        result = frequencies2bins([-1, 0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
        self.assertTrue(np.allclose(result, [0, 0, 1, 2, 3, 4]))


class TestBins2FrequenciesFunction(unittest.TestCase):

    def test_num_arguments(self):
        # number of arguments arguments
        with self.assertRaises(TypeError):
            bins2frequencies()
        with self.assertRaises(TypeError):
            bins2frequencies(1)
        with self.assertRaises(TypeError):
            bins2frequencies(1, 2, 3)

    def test_types(self):
        result = bins2frequencies([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float)

    def test_value(self):
        result = bins2frequencies([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
        self.assertTrue(np.allclose(result, [0, 1, 2, 3, 4]))
        result = bins2frequencies([0, 1, 2, 3, 4], [0, 1, 1.1, 1.2, 2, 3, 4])
        self.assertTrue(np.allclose(result, [0, 1, 1.1, 1.2, 2]))
        result = bins2frequencies([-1, 0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
        self.assertTrue(np.allclose(result, [4, 0, 1, 2, 3, 4]))


# test classes
class TestFilterClass(unittest.TestCase):

    def test_types(self):
        filt = Filter(np.arange(5))
        self.assertIsInstance(filt, Filter)
        self.assertIsInstance(filt.start, int)
        self.assertIsInstance(filt.stop, int)

    def test_conversion(self):
        with self.assertRaises(TypeError):
            Filter([0, 1, 2, 3, 4])
        with self.assertRaises(TypeError):
            Filter(0, 1)
        with self.assertRaises(TypeError):
            Filter(np.arange(5), [0, 1])
        with self.assertRaises(NotImplementedError):
            Filter(np.zeros((10, 2)), 1)

    def test_value(self):
        filt = Filter(np.arange(5))
        self.assertTrue(np.allclose(filt, [0, 1, 2, 3, 4]))
        self.assertTrue(filt.start == 0)
        self.assertTrue(filt.stop == 5)
        filt = Filter(np.arange(5), 1)
        self.assertTrue(np.allclose(filt, [0, 1, 2, 3, 4]))
        self.assertTrue(filt.start == 1)
        self.assertTrue(filt.stop == 6)

    def test_normalization(self):
        filt = Filter(np.arange(5), norm=True)
        self.assertTrue(np.allclose(filt, [0, 0.1, 0.2, 0.3, 0.4]))
        self.assertTrue(filt.start == 0)
        self.assertTrue(filt.stop == 5)

    def test_pickling(self):
        f, filename = tempfile.mkstemp()
        filt = Filter(np.arange(5))
        cPickle.dump(filt, open(filename, 'w'),
                     protocol=cPickle.HIGHEST_PROTOCOL)
        filt_ = cPickle.load(open(filename))
        self.assertTrue(np.allclose(filt, filt_))
        self.assertTrue(np.allclose(filt.start, filt_.start))

    def test_filters_method(self):
        with self.assertRaises(NotImplementedError):
            Filter(np.arange(5)).filters(3)


class TestTriangularFilterClass(unittest.TestCase):

    bins = np.asarray([0, 0, 1, 1, 2, 3, 4, 6, 9])

    def test_types(self):
        filt = TriangularFilter(0, 1, 2, False)
        self.assertIsInstance(filt, TriangularFilter)
        self.assertTrue(filt.dtype == FILTER_DTYPE)
        self.assertIsInstance(filt.band_bins(np.arange(8)),
                              types.GeneratorType)

    def test_errors(self):
        # integers bin numbers
        with self.assertRaises(ValueError):
            TriangularFilter(0, 1.1, 2, False)
        # filter bins ascending order
        with self.assertRaises(ValueError):
            TriangularFilter(0, 2, 1, False)

    def test_values(self):
        filt = TriangularFilter(0, 1, 2, False)
        self.assertTrue(np.allclose(filt, [0, 1]))
        self.assertTrue(filt.start == 0)
        self.assertTrue(filt.center == 1)
        self.assertTrue(filt.stop == 2)
        filt = TriangularFilter(1, 2, 3, True)
        self.assertTrue(np.allclose(filt, [0, 1]))
        self.assertTrue(filt.start == 1)
        self.assertTrue(filt.center == 2)
        self.assertTrue(filt.stop == 3)
        filt = TriangularFilter(1, 2, 4, False)
        self.assertTrue(np.allclose(filt, [0, 1, 0.5]))
        self.assertTrue(filt.start == 1)
        self.assertTrue(filt.center == 2)
        self.assertTrue(filt.stop == 4)
        filt = TriangularFilter(1, 2, 4, True)
        self.assertTrue(np.allclose(filt, [0, 0.66667, 0.33333]))
        self.assertTrue(filt.start == 1)
        self.assertTrue(filt.center == 2)
        self.assertTrue(filt.stop == 4)
        filt = TriangularFilter(4, 6, 9, True)
        self.assertTrue(np.allclose(filt, [0, 0.2, 0.4, 0.266667, 0.133333]))
        self.assertTrue(filt.start == 4)
        self.assertTrue(filt.center == 6)
        self.assertTrue(filt.stop == 9)
        # test small filters (these can occur if a `Filterbank` uses
        # `TriangularFilter`s and has `duplicate_filters` set to 'True'
        filt = TriangularFilter(0, 0, 1, True)
        self.assertTrue(np.allclose(filt, [1]))
        self.assertTrue(filt.start == 0)
        self.assertTrue(filt.center == 0)
        self.assertTrue(filt.stop == 1)
        filt = TriangularFilter(0, 0, 1, False)
        self.assertTrue(np.allclose(filt, [1]))
        self.assertTrue(filt.start == 0)
        self.assertTrue(filt.center == 0)
        self.assertTrue(filt.stop == 1)

    def test_pickling(self):
        f, filename = tempfile.mkstemp()
        filt = TriangularFilter(1, 4, 10)
        cPickle.dump(filt, open(filename, 'w'),
                     protocol=cPickle.HIGHEST_PROTOCOL)
        filt_ = cPickle.load(open(filename))
        self.assertTrue(np.allclose(filt, filt_))
        self.assertTrue(np.allclose(filt.start, filt_.start))
        self.assertTrue(np.allclose(filt.stop, filt_.stop))
        self.assertTrue(np.allclose(filt.center, filt_.center))

    def test_band_bins_method_too_few_bins(self):
        with self.assertRaises(ValueError):
            result = TriangularFilter.band_bins(np.arange(2))
            result.next()

    def test_band_bins_method_overlap(self):
        # test overlapping
        result = TriangularFilter.band_bins(self.bins)
        self.assertTrue(result.next() == (0, 1, 2, True))
        self.assertTrue(result.next() == (1, 2, 3, True))
        self.assertTrue(result.next() == (2, 3, 4, True))
        self.assertTrue(result.next() == (3, 4, 6, True))
        self.assertTrue(result.next() == (4, 6, 9, True))
        with self.assertRaises(StopIteration):
            result.next()

    def test_band_bins_method_overlap_no_norm(self):
        # test overlapping without normalization
        result = TriangularFilter.band_bins(self.bins, norm=False)
        self.assertTrue(result.next() == (0, 1, 2, False))
        self.assertTrue(result.next() == (1, 2, 3, False))
        self.assertTrue(result.next() == (2, 3, 4, False))
        self.assertTrue(result.next() == (3, 4, 6, False))
        self.assertTrue(result.next() == (4, 6, 9, False))
        with self.assertRaises(StopIteration):
            result.next()

    def test_band_bins_method_non_overlap(self):
        # test non-overlapping
        result = TriangularFilter.band_bins(self.bins, overlap=False)
        self.assertTrue(result.next() == (1, 1, 2, True))
        self.assertTrue(result.next() == (2, 2, 3, True))
        self.assertTrue(result.next() == (3, 3, 4, True))
        self.assertTrue(result.next() == (4, 4, 5, True))
        self.assertTrue(result.next() == (5, 6, 8, True))
        with self.assertRaises(StopIteration):
            result.next()

    def test_band_bins_method_overlap_duplicates(self):
        # test duplicate & overlapping
        result = TriangularFilter.band_bins(self.bins, duplicates=True)
        self.assertTrue(result.next() == (0, 0, 1, True))
        # if filters are too small, always return start=centre
        self.assertTrue(result.next() == (0, 0, 1, True))
        self.assertTrue(result.next() == (1, 1, 2, True))
        self.assertTrue(result.next() == (1, 2, 3, True))
        self.assertTrue(result.next() == (2, 3, 4, True))
        self.assertTrue(result.next() == (3, 4, 6, True))
        self.assertTrue(result.next() == (4, 6, 9, True))
        with self.assertRaises(StopIteration):
            result.next()

    def test_band_bins_method_non_overlap_duplicates(self):
        # test duplicate & non-overlapping
        result = TriangularFilter.band_bins(self.bins, overlap=False,
                                            duplicates=True)
        self.assertTrue(result.next() == (0, 0, 1, True))
        self.assertTrue(result.next() == (1, 1, 2, True))
        # if filters are too small, always return start=centre
        self.assertTrue(result.next() == (1, 1, 2, True))
        self.assertTrue(result.next() == (2, 2, 3, True))
        self.assertTrue(result.next() == (3, 3, 4, True))
        self.assertTrue(result.next() == (4, 4, 5, True))
        self.assertTrue(result.next() == (5, 6, 8, True))
        with self.assertRaises(StopIteration):
            result.next()

    def test_filters_method_normalized(self):
        # normalized filters
        result = TriangularFilter.filters(self.bins)
        filters = np.asarray([[0, 1], [0, 1], [0, 1], [0, 0.66667, 0.33333],
                              [0, 0.2, 0.4, 0.266667, 0.133333]])
        starts = [0, 1, 2, 3, 4]
        stops = [2, 3, 4, 6, 9]
        # test the values of the filters itself
        for i, res in enumerate(result):
            self.assertTrue(np.allclose(res, filters[i]))
        # test start positions
        for i, res in enumerate(result):
            self.assertTrue(np.allclose(res.start, starts[i]))
        # test stop positions
        for i, res in enumerate(result):
            self.assertTrue(np.allclose(res.stop, stops[i]))

    def test_filters_method_non_normalized(self):
        # non-normalized filters
        result = TriangularFilter.filters(self.bins, norm=False)
        filters = np.asarray([[0, 1], [0, 1], [0, 1], [0, 1, 0.5],
                              [0, 0.5, 1, 0.66667, 0.33333]])
        starts = [0, 1, 2, 3, 4]
        stops = [2, 3, 4, 6, 9]
        # test the values of the filters itself
        for i, res in enumerate(result):
            self.assertTrue(np.allclose(res, filters[i]))
        # test start positions
        for i, res in enumerate(result):
            self.assertTrue(np.allclose(res.start, starts[i]))
        # test stop positions
        for i, res in enumerate(result):
            self.assertTrue(np.allclose(res.stop, stops[i]))


class TestRectangularFilterClass(unittest.TestCase):

    bins = np.asarray([0, 0, 1, 1, 2, 3, 4, 6, 9])

    def test_types(self):
        filt = RectangularFilter(0, 1, False)
        self.assertIsInstance(filt, RectangularFilter)
        self.assertTrue(filt.dtype == FILTER_DTYPE)
        self.assertIsInstance(filt.band_bins(np.arange(8)),
                              types.GeneratorType)

    def test_errors(self):
        # TODO: why is this error not raised? it does not really matter, though
        # # integer bin numbers
        # with self.assertRaises(ValueError):
        #     RectangularFilter(0, 1.1, False)
        # filter bins ascending order
        with self.assertRaises(ValueError):
            RectangularFilter(2, 1, False)

    def test_values(self):
        filt = RectangularFilter(0, 1, False)
        self.assertTrue(np.allclose(filt, [1]))
        self.assertTrue(filt.start == 0)
        self.assertTrue(filt.stop == 1)
        filt = RectangularFilter(1, 3, True)
        self.assertTrue(np.allclose(filt, [0.5, 0.5]))
        self.assertTrue(filt.start == 1)
        self.assertTrue(filt.stop == 3)
        filt = RectangularFilter(1, 4, False)
        self.assertTrue(np.allclose(filt, [1, 1, 1]))
        self.assertTrue(filt.start == 1)
        self.assertTrue(filt.stop == 4)
        filt = RectangularFilter(1, 4, True)
        self.assertTrue(np.allclose(filt, [0.33333, 0.33333, 0.33333]))
        self.assertTrue(filt.start == 1)
        self.assertTrue(filt.stop == 4)
        filt = RectangularFilter(4, 9, True)
        self.assertTrue(np.allclose(filt, [0.2, 0.2, 0.2, 0.2, 0.2]))
        self.assertTrue(filt.start == 4)
        self.assertTrue(filt.stop == 9)

    def test_pickling(self):
        f, filename = tempfile.mkstemp()
        filt = RectangularFilter(5, 10)
        cPickle.dump(filt, open(filename, 'w'),
                     protocol=cPickle.HIGHEST_PROTOCOL)
        filt_ = cPickle.load(open(filename))
        self.assertTrue(np.allclose(filt, filt_))
        self.assertTrue(np.allclose(filt.start, filt_.start))
        self.assertTrue(np.allclose(filt.stop, filt_.stop))

    def test_band_bins_method_too_few_bins(self):
        with self.assertRaises(ValueError):
            result = RectangularFilter.band_bins(np.arange(1))
            result.next()

    def test_band_bins_method_overlap(self):
        result = RectangularFilter.band_bins(self.bins, overlap=True)
        with self.assertRaises(NotImplementedError):
            result.next()

    def test_band_bins_method_norm(self):
        result = RectangularFilter.band_bins(self.bins)
        self.assertTrue(result.next() == (0, 1, True))
        self.assertTrue(result.next() == (1, 2, True))
        self.assertTrue(result.next() == (2, 3, True))
        self.assertTrue(result.next() == (3, 4, True))
        self.assertTrue(result.next() == (4, 6, True))
        self.assertTrue(result.next() == (6, 9, True))
        with self.assertRaises(StopIteration):
            result.next()

    def test_band_bins_method_no_norm(self):
        result = RectangularFilter.band_bins(self.bins, norm=False)
        self.assertTrue(result.next() == (0, 1, False))
        self.assertTrue(result.next() == (1, 2, False))
        self.assertTrue(result.next() == (2, 3, False))
        self.assertTrue(result.next() == (3, 4, False))
        self.assertTrue(result.next() == (4, 6, False))
        self.assertTrue(result.next() == (6, 9, False))
        with self.assertRaises(StopIteration):
            result.next()

    def test_band_bins_method_duplicates(self):
        result = RectangularFilter.band_bins(self.bins, duplicates=True)
        self.assertTrue(result.next() == (0, 0, True))
        self.assertTrue(result.next() == (0, 1, True))
        self.assertTrue(result.next() == (1, 1, True))
        self.assertTrue(result.next() == (1, 2, True))
        self.assertTrue(result.next() == (2, 3, True))
        self.assertTrue(result.next() == (3, 4, True))
        self.assertTrue(result.next() == (4, 6, True))
        self.assertTrue(result.next() == (6, 9, True))
        with self.assertRaises(StopIteration):
            result.next()

    def test_filters_method_normalized(self):
        # normalized filters
        # resulting bins: [0, 1, 2, 3, 4, 6, 9]
        result = RectangularFilter.filters(self.bins)
        filters = np.asarray([[1], [1], [1], [1], [0.5, 0.5],
                              [0.33333, 0.33333, 0.33333]])
        starts = [0, 1, 2, 3, 4, 6]
        stops = [1, 2, 3, 4, 6, 9]
        # test the values of the filters itself
        for i, res in enumerate(result):
            self.assertTrue(np.allclose(res, filters[i]))
        # test start positions
        for i, res in enumerate(result):
            self.assertTrue(np.allclose(res.start, starts[i]))
        # test stop positions
        for i, res in enumerate(result):
            self.assertTrue(np.allclose(res.stop, stops[i]))

    def test_filters_method_non_normalized(self):
        # non-normalized filters
        # resulting bins: [0, 0, 1, 1, 2, 3, 4, 6, 9]
        result = RectangularFilter.filters(self.bins, norm=False)
        filters = np.asarray([[1], [1], [1], [1], [1, 1], [1, 1, 1]])
        starts = [0, 1, 2, 3, 4, 6]
        stops = [1, 2, 3, 4, 6, 9]
        # test the values of the filters itself
        for i, res in enumerate(result):
            self.assertTrue(np.allclose(res, filters[i]))
        # test start positions
        for i, res in enumerate(result):
            self.assertTrue(np.allclose(res.start, starts[i]))
        # test stop positions
        for i, res in enumerate(result):
            self.assertTrue(np.allclose(res.stop, stops[i]))


class TestConstantsClass(unittest.TestCase):

    def test_types(self):
        self.assertIsInstance(FMIN, float)
        self.assertIsInstance(FMAX, float)
        self.assertIsInstance(BANDS, int)
        self.assertIsInstance(NORM_FILTERS, bool)
        self.assertIsInstance(DUPLICATE_FILTERS, bool)
        self.assertIsInstance(OVERLAP_FILTERS, bool)

    def test_values(self):
        self.assertEqual(FMIN, 30.)
        self.assertEqual(FMAX, 17000.)
        self.assertEqual(BANDS, 12)
        self.assertEqual(NORM_FILTERS, True)
        self.assertEqual(DUPLICATE_FILTERS, False)
        self.assertEqual(OVERLAP_FILTERS, True)


class TestFilterbankClass(unittest.TestCase):

    rect_filters = [RectangularFilter(0, 10),
                    RectangularFilter(10, 25),
                    RectangularFilter(25, 50),
                    RectangularFilter(50, 100)]

    triang_filters = [TriangularFilter(0, 6, 15),
                      TriangularFilter(6, 15, 25),
                      TriangularFilter(15, 25, 50),
                      TriangularFilter(25, 50, 70)]

    def test_types(self):
        filt = Filterbank(np.zeros((100, 10)), np.arange(100))
        self.assertIsInstance(filt, Filterbank)
        self.assertTrue(filt.dtype == FILTER_DTYPE)
        self.assertTrue(filt.bin_frequencies.dtype == np.float)

    def test_errors(self):
        with self.assertRaises(TypeError):
            Filterbank(np.zeros((100, 10)))
        with self.assertRaises(TypeError):
            Filterbank(np.zeros(10), np.arange(10))
        with self.assertRaises(ValueError):
            Filterbank(np.zeros((100, 10)), np.arange(10))

    def test_put_filter_function(self):
        # normal filter placement
        filt = np.ones(50) * 0.5
        Filterbank._put_filter(RectangularFilter(10, 25), filt)
        self.assertTrue(np.allclose(filt[:10], np.ones(10) * 0.5))
        self.assertTrue(np.allclose(filt[10:25], np.ones(15)))
        self.assertTrue(np.allclose(filt[25:50], np.ones(25) * 0.5))
        # out of range end
        filt = np.zeros(20)
        Filterbank._put_filter(RectangularFilter(10, 25), filt)
        self.assertTrue(np.allclose(filt[:10], np.zeros(10)))
        self.assertTrue(np.allclose(filt[10:], np.ones(10)))
        # out of range start
        filt = np.zeros(20)
        Filterbank._put_filter(RectangularFilter(-5, 10), filt)
        self.assertTrue(np.allclose(filt[:10], np.ones(10)))
        self.assertTrue(np.allclose(filt[10:], np.zeros(10)))

    def test_from_filters_function(self):
        filt = Filterbank.from_filters(self.rect_filters, np.arange(100))
        self.assertIsInstance(filt, Filterbank)
        self.assertTrue(filt.dtype == FILTER_DTYPE)
        self.assertTrue(filt.bin_frequencies.dtype == np.float)

    def test_values_rectangular(self):
        filt = Filterbank.from_filters(self.rect_filters, np.arange(100))
        self.assertTrue(filt.num_bands == 4)
        self.assertTrue(filt.num_bins == 100)
        self.assertTrue(filt.fmin == 0)
        self.assertTrue(filt.fmax == 99)
        self.assertTrue(np.allclose(filt.bin_frequencies, np.arange(100)))
        self.assertTrue(np.allclose(filt.corner_frequencies,
                                    [[0, 9], [10, 24], [25, 49], [50, 99]]))
        self.assertTrue(np.allclose(filt.center_frequencies,
                                    [4, 17, 37, 74]))

    def test_values_triangular(self):
        filt = Filterbank.from_filters(self.triang_filters, np.arange(100))
        self.assertTrue(filt.num_bands == 4)
        self.assertTrue(filt.num_bins == 100)
        self.assertTrue(filt.fmin == 1)
        self.assertTrue(filt.fmax == 69)
        self.assertTrue(np.allclose(filt.bin_frequencies, np.arange(100)))
        self.assertTrue(np.allclose(filt.corner_frequencies,
                                    [[1, 14], [7, 24], [16, 49], [26, 69]]))
        self.assertTrue(np.allclose(filt.center_frequencies,
                                    [6, 15, 25, 50]))

    def test_pickling(self):
        filt = Filterbank.from_filters(self.triang_filters, np.arange(100))
        f, filename = tempfile.mkstemp()
        cPickle.dump(filt, open(filename, 'w'),
                     protocol=cPickle.HIGHEST_PROTOCOL)
        filt_ = cPickle.load(open(filename))
        self.assertTrue(np.allclose(filt, filt_))
        self.assertTrue(np.allclose(filt.bin_frequencies,
                                    filt_.bin_frequencies))


class TestMelFilterbankClass(unittest.TestCase):

    def test_types(self):
        filt = MelFilterbank(np.arange(20000))
        self.assertIsInstance(filt, MelFilterbank)
        self.assertTrue(filt.dtype == FILTER_DTYPE)
        self.assertTrue(filt.bin_frequencies.dtype == np.float)

    def test_constant_types(self):
        self.assertIsInstance(MelFilterbank.FMIN, float)
        self.assertIsInstance(MelFilterbank.FMAX, float)
        self.assertIsInstance(MelFilterbank.BANDS, int)
        self.assertIsInstance(MelFilterbank.NORM_FILTERS, bool)
        self.assertIsInstance(MelFilterbank.DUPLICATE_FILTERS, bool)

    def test_constant_values(self):
        self.assertEqual(MelFilterbank.FMIN, 20.)
        self.assertEqual(MelFilterbank.FMAX, 17000.)
        self.assertEqual(MelFilterbank.BANDS, 40)
        self.assertEqual(MelFilterbank.NORM_FILTERS, True)
        self.assertEqual(MelFilterbank.DUPLICATE_FILTERS, False)

    def test_values(self):
        filt = MelFilterbank(np.arange(1000) * 20, 10)
        self.assertTrue(filt.num_bands == 10)
        self.assertTrue(filt.num_bins == 1000)
        self.assertTrue(filt.fmin == 40)
        self.assertTrue(filt.fmax == 16980)
        self.assertTrue(np.allclose(filt.center_frequencies,
                                    [260, 580, 1020, 1600, 2380, 3420, 4820,
                                     6700, 9180, 12520]))
        self.assertTrue(np.allclose(filt.corner_frequencies,
                                    [[40, 560], [280, 1000], [600, 1580],
                                     [1040, 2360], [1620, 3400], [2400, 4800],
                                     [3440, 6680], [4840, 9160], [6720, 12500],
                                     [9200, 16980]]))

    def test_pickling(self):
        filt = MelFilterbank(np.arange(1000) * 20, 10)
        f, filename = tempfile.mkstemp()
        cPickle.dump(filt, open(filename, 'w'),
                     protocol=cPickle.HIGHEST_PROTOCOL)
        filt_ = cPickle.load(open(filename))
        self.assertTrue(np.allclose(filt, filt_))
        self.assertTrue(np.allclose(filt.bin_frequencies,
                                    filt_.bin_frequencies))

    def test_default_values(self):
        filt = MelFilterbank(np.fft.fftfreq(2048, 1. / 44100)[:1024])
        center = [86.132812, 150.732422, 215.332031, 279.931640,
                  366.064453, 452.197266, 538.330078, 645.996094,
                  753.662109, 882.861328, 990.527344, 1141.259766,
                  1291.992187, 1442.724609, 1614.990234, 1808.789063,
                  2024.121093, 2239.453125, 2476.318359, 2734.716797,
                  3014.648437, 3316.113281, 3639.111328, 3983.642578,
                  4371.240234, 4780.371094, 5232.568359, 5706.298828,
                  6223.095703, 6804.492187, 7407.421875, 8053.417969,
                  8785.546875, 9539.208984, 10379.003906, 11283.398437,
                  12252.392578, 13307.519531, 14448.779297, 15676.171875]
        corner = [[43.066406, 129.199218], [107.666015, 193.798828],
                  [172.265625, 258.398437], [236.865234, 344.531250],
                  [301.464843, 430.664062], [387.597656, 516.796875],
                  [473.730468, 624.462890], [559.863281, 732.128906],
                  [667.529296, 861.328125], [775.195312, 968.994140],
                  [904.394531, 1119.72656], [1012.060546, 1270.458984],
                  [1162.792968, 1421.191406], [1313.525390, 1593.457031],
                  [1464.257812, 1787.255859], [1636.523437, 2002.587890],
                  [1830.322265, 2217.919921], [2045.654296, 2454.785156],
                  [2260.986328, 2713.183593], [2497.851562, 2993.115234],
                  [2756.250000, 3294.580078], [3036.181640, 3617.578125],
                  [3337.646484, 3962.109375], [3660.644531, 4349.707031],
                  [4005.175781, 4758.837890], [4392.773437, 5211.035156],
                  [4801.904296, 5684.765625], [5254.101562, 6201.562500],
                  [5727.832031, 6782.958984], [6244.628906, 7385.888671],
                  [6826.025390, 8031.884765], [7428.955078, 8764.013671],
                  [8074.951171, 9517.675781], [8807.080078, 10357.470702],
                  [9560.742187, 11261.865238], [10400.537109, 12230.859375],
                  [11304.931640, 13285.986328], [12273.925781, 14427.246093],
                  [13329.052734, 15654.638671], [14470.312500, 16968.164062]]
        self.assertTrue(np.allclose(filt.center_frequencies, center))
        self.assertTrue(np.allclose(filt.corner_frequencies, corner))


class TestBarkFilterbankClass(unittest.TestCase):

    def test_types(self):
        filt = BarkFilterbank(np.arange(20000))
        self.assertIsInstance(filt, BarkFilterbank)
        self.assertTrue(filt.dtype == FILTER_DTYPE)
        self.assertTrue(filt.bin_frequencies.dtype == np.float)

    def test_constant_types(self):
        self.assertIsInstance(BarkFilterbank.FMIN, float)
        self.assertIsInstance(BarkFilterbank.FMAX, float)
        self.assertIsInstance(BarkFilterbank.BANDS, str)
        self.assertIsInstance(BarkFilterbank.NORM_FILTERS, bool)
        self.assertIsInstance(BarkFilterbank.DUPLICATE_FILTERS, bool)

    def test_constant_values(self):
        self.assertEqual(BarkFilterbank.FMIN, 20.)
        self.assertEqual(BarkFilterbank.FMAX, 15500.)
        self.assertEqual(BarkFilterbank.BANDS, 'normal')
        self.assertEqual(BarkFilterbank.NORM_FILTERS, True)
        self.assertEqual(BarkFilterbank.DUPLICATE_FILTERS, False)

    def test_pickling(self):
        filt = BarkFilterbank(FFT_FREQS_1024)
        f, filename = tempfile.mkstemp()
        cPickle.dump(filt, open(filename, 'w'),
                     protocol=cPickle.HIGHEST_PROTOCOL)
        filt_ = cPickle.load(open(filename))
        self.assertTrue(np.allclose(filt, filt_))
        self.assertTrue(np.allclose(filt.bin_frequencies,
                                    filt_.bin_frequencies))

    def test_default_values(self):
        filt = BarkFilterbank(FFT_FREQS_1024)
        center = [43.066406, 129.199218, 236.865234, 344.53125, 452.197265,
                  559.863281, 689.0625, 839.794921, 990.527343, 1162.792968,
                  1356.591796, 1593.457031, 1851.855468, 2153.320312,
                  2497.851562, 2906.982421, 3402.246093, 4026.708984,
                  4823.4375, 5835.498046, 7041.357421, 8591.748046,
                  10723.535156, 13738.183593]
        corner = [[21.533203, 86.132812], [107.666015, 172.265625],
                  [193.798828, 279.931640], [301.464843, 387.597656],
                  [409.130859, 495.263671], [516.796875, 602.929687],
                  [624.462890, 753.662109], [775.195312, 904.394531],
                  [925.927734, 1055.126953], [1076.660156, 1248.925781],
                  [1270.458984, 1464.257812], [1485.791015, 1701.123046],
                  [1722.656250, 1981.054687], [2002.587890, 2304.052734],
                  [2325.585937, 2670.117187], [2691.650390, 3122.314453],
                  [3143.847656, 3682.177734], [3703.710937, 4371.240234],
                  [4392.773437, 5275.634765], [5297.167968, 6373.828125],
                  [6395.361328, 7687.353515], [7708.886718, 9474.609375],
                  [9496.142578, 11972.460937], [11993.994140, 15482.373046]]
        self.assertTrue(filt.num_bands == 24)
        self.assertTrue(filt.num_bins == 1024)
        self.assertTrue(np.allclose(filt.fmin, 21.5332031))
        self.assertTrue(np.allclose(filt.fmax, 15482.3730468))
        self.assertTrue(np.allclose(filt.center_frequencies, center))
        self.assertTrue(np.allclose(filt.corner_frequencies, corner))


class TestLogarithmicFilterbankClass(unittest.TestCase):

    def test_types(self):
        filt = LogarithmicFilterbank(np.arange(20000))
        self.assertIsInstance(filt, LogarithmicFilterbank)
        self.assertTrue(filt.dtype == FILTER_DTYPE)
        self.assertTrue(filt.bin_frequencies.dtype == np.float)

    def test_constant_types(self):
        # TODO: why can't we test the inherited constants? it does not matter
        # self.assertIsInstance(LogarithmicFilterbank.FMIN, float))
        # self.assertIsInstance(LogarithmicFilterbank.FMAX, float))
        self.assertIsInstance(LogarithmicFilterbank.BANDS_PER_OCTAVE, int)
        # self.assertIsInstance(LogarithmicFilterbank.NORM_FILTERS, bool))
        # self.assertIsInstance(LogarithmicFilterbank.DUPLICATE_FILTERS, bool))

    def test_constant_values(self):
        # self.assertEqual(LogarithmicFilterbank.FMIN, 30.)
        # self.assertEqual(LogarithmicFilterbank.FMAX, 17000.)
        self.assertEqual(LogarithmicFilterbank.BANDS_PER_OCTAVE, 12)
        # self.assertEqual(LogarithmicFilterbank.NORM_FILTERS, True)
        # self.assertEqual(LogarithmicFilterbank.DUPLICATE_FILTERS, False)

    def test_pickling(self):
        filt = LogarithmicFilterbank(FFT_FREQS_1024)
        f, filename = tempfile.mkstemp()
        cPickle.dump(filt, open(filename, 'w'),
                     protocol=cPickle.HIGHEST_PROTOCOL)
        filt_ = cPickle.load(open(filename))
        self.assertTrue(np.allclose(filt, filt_))
        self.assertTrue(np.allclose(filt.bin_frequencies,
                                    filt_.bin_frequencies))
        self.assertTrue(filt.bands_per_octave == filt_.bands_per_octave)
        self.assertTrue(filt.fref == filt_.fref)

    def test_values_duplicate_filters(self):
        filt = LogarithmicFilterbank(np.arange(0, 20000, 20), num_bands=12)
        self.assertTrue(np.allclose(filt.max(), 1))
        self.assertEqual(filt.shape, (1000, 81))
        filt = LogarithmicFilterbank(np.arange(0, 20000, 20), num_bands=12,
                                     duplicate_filters=True)
        self.assertTrue(np.allclose(filt.max(), 1))
        self.assertEqual(filt.shape, (1000, 108))

    def test_default_values(self):
        filt = LogarithmicFilterbank(FFT_FREQS_1024)
        center = [43.066406, 64.5996093, 86.1328125, 107.6660156, 129.199218,
                  150.732421, 172.265625, 193.798828, 215.332031, 236.865234,
                  258.398437, 279.931640, 301.464843, 322.998046, 344.531250,
                  366.064453, 387.597656, 409.130859, 430.664062, 452.197265,
                  495.263671, 516.796875, 538.330078, 581.396484, 624.462890,
                  645.996093, 689.062500, 732.128906, 775.195312, 839.794921,
                  882.861328, 925.927734, 990.527343, 1055.126953, 1098.193359,
                  1184.326171, 1248.925781, 1313.525390, 1399.658203,
                  1485.791015, 1571.923828, 1658.056640, 1765.722656,
                  1873.388671, 1981.054687, 2088.720703, 2217.919921,
                  2347.119140, 2497.851562, 2627.050781, 2799.316406,
                  2950.048828, 3143.847656, 3316.113281, 3509.912109,
                  3725.244140, 3940.576171, 4177.441406, 4435.839843,
                  4694.238281, 4974.169921, 5275.634765, 5577.099609,
                  5921.630859, 6266.162109, 6653.759765, 7041.357421,
                  7450.488281, 7902.685546, 8376.416015, 8871.679687,
                  9388.476562, 9948.339843, 10551.269531, 11175.732421,
                  11843.261718, 12553.857421, 13285.986328, 14082.714843,
                  14922.509765, 15805.37109375]
        corner = [[43.066406, 43.066406], [64.599609, 64.599609],
                  [86.132812, 86.132812], [107.666015, 107.666015],
                  [129.199218, 129.199218], [150.732421, 150.732421],
                  [172.265625, 172.265625], [193.798828, 193.798828],
                  [215.332031, 215.332031], [236.865234, 236.865234],
                  [258.398437, 258.398437], [279.931640, 279.931640],
                  [301.464843, 301.464843], [322.998046, 322.998046],
                  [344.531250, 344.531250], [366.064453, 366.064453],
                  [387.597656, 387.597656], [409.130859, 409.130859],
                  [430.664062, 452.197265], [452.197265, 473.730468],
                  [495.263671, 495.263671], [516.796875, 538.330078],
                  [538.330078, 559.863281], [581.396484, 602.929687],
                  [602.929687, 645.996093], [645.996093, 667.529296],
                  [689.062500, 710.595703], [710.595703, 753.662109],
                  [753.662109, 818.261718], [796.728515, 861.328125],
                  [861.328125, 904.394531], [904.394531, 968.994140],
                  [947.460937, 1033.59375], [1012.060546, 1076.660156],
                  [1076.660156, 1162.792968], [1119.726562, 1227.392578],
                  [1205.859375, 1291.992187], [1270.458984, 1378.125000],
                  [1335.058593, 1464.257812], [1421.191406, 1550.390625],
                  [1507.324218, 1636.523437], [1593.457031, 1744.189453],
                  [1679.589843, 1851.855468], [1787.255859, 1959.521484],
                  [1894.921875, 2067.187500], [2002.587890, 2196.386718],
                  [2110.253906, 2325.585937], [2239.453125, 2476.318359],
                  [2368.652343, 2605.517578], [2519.384765, 2777.783203],
                  [2648.583984, 2928.515625], [2820.849609, 3122.314453],
                  [2971.582031, 3294.580078], [3165.380859, 3488.378906],
                  [3337.646484, 3703.710937], [3531.445312, 3919.042968],
                  [3746.777343, 4155.908203], [3962.109375, 4414.306640],
                  [4198.974609, 4672.705078], [4457.373046, 4952.636718],
                  [4715.771484, 5254.101562], [4995.703125, 5555.566406],
                  [5297.167968, 5900.097656], [5598.632812, 6244.628906],
                  [5943.164062, 6632.226562], [6287.695312, 7019.824218],
                  [6675.292968, 7428.955078], [7062.890625, 7881.152343],
                  [7472.021484, 8354.882812], [7924.218750, 8850.146484],
                  [8397.949218, 9366.943359], [8893.212890, 9926.806640],
                  [9410.009765, 10529.736328], [9969.873046, 11154.199218],
                  [10572.802734, 11821.728515], [11197.265625, 12532.324218],
                  [11864.794921, 13264.453125], [12575.390625, 14061.181640],
                  [13307.519531, 14900.976562], [14104.248046, 15783.837890],
                  [14944.042968, 16731.298828]]
        self.assertTrue(filt.num_bands == 81)
        self.assertTrue(filt.bands_per_octave == 12)
        self.assertTrue(filt.num_bins == 1024)
        self.assertTrue(np.allclose(filt.fmin, 43.066406))
        self.assertTrue(np.allclose(filt.fmax, 16731.298828))
        self.assertTrue(np.allclose(filt.center_frequencies, center))
        self.assertTrue(np.allclose(filt.corner_frequencies, corner))


class TestSimpleChromaFilterbankClass(unittest.TestCase):

    def test_error(self):
        with self.assertRaises(NotImplementedError):
            SimpleChromaFilterbank(FFT_FREQS_1024)


class TestHarmonicFilterbankClass(unittest.TestCase):

    def test_error(self):
        with self.assertRaises(NotImplementedError):
            SimpleChromaFilterbank(FFT_FREQS_1024)


class TestPitchClassProfileFilterbankClass(unittest.TestCase):

    def test_error(self):
        with self.assertRaises(NotImplementedError):
            PitchClassProfileFilterbank(FFT_FREQS_1024)


class TestHarmonicPitchClassProfileFilterbankClass(unittest.TestCase):

    def test_error(self):
        with self.assertRaises(NotImplementedError):
            HarmonicPitchClassProfileFilterbank(FFT_FREQS_1024)


# comb filter test stuff
from .test_audio_comb_filters import (sig_1d, sig_2d, res_1d_bw_2, res_1d_bw_3,
                                      res_2d_bw_2, res_2d_bw_3)

res_1d_fw_2 = np.asarray([0, 0, 1, 0, 0.5, 1, 0, 0.5, 1])
res_1d_fw_3 = np.asarray([0, 0, 1, 0, 0, 1.5, 0, 0, 1.5])
res_2d_fw_2 = np.asarray([[0, 0, 1, 0, 0.5, 1, 0, 0.5, 1],
                          [1, 0, 1.5, 0, 1.5, 0, 1.5, 0, 1.5]]).T
res_2d_fw_3 = np.asarray([[0, 0, 1, 0, 0, 1.5, 0, 0, 1.5],
                          [1, 0, 1, 0.5, 1, 0.5, 1, 0.5, 1]]).T


class TestFeedForwardFilterFunction(unittest.TestCase):

    def test_types(self):
        result = feed_forward_comb_filter(sig_1d, 2, 0.5)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(type(result), float)

    def test_values(self):
        result = feed_forward_comb_filter(sig_1d, 2, 0.5)
        self.assertTrue(np.allclose(result, res_1d_fw_2))
        result = feed_forward_comb_filter(sig_1d, 3, 0.5)
        self.assertTrue(np.allclose(result, res_1d_fw_3))
        result = feed_forward_comb_filter(sig_2d, 2, 0.5)
        self.assertTrue(np.allclose(result, res_2d_fw_2))
        result = feed_forward_comb_filter(sig_2d, 3, 0.5)
        self.assertTrue(np.allclose(result, res_2d_fw_3))


class TestFeedBackwardFilterFunction(unittest.TestCase):

    def test_types(self):
        result = feed_forward_comb_filter(sig_1d, 2, 0.5)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(type(result), float)

    def test_values(self):
        result = feed_backward_comb_filter(sig_1d, 2, 0.5)
        self.assertTrue(np.allclose(result, res_1d_bw_2))
        result = feed_backward_comb_filter(sig_1d, 3, 0.5)
        self.assertTrue(np.allclose(result, res_1d_bw_3))
        result = feed_backward_comb_filter(sig_2d, 2, 0.5)
        self.assertTrue(np.allclose(result, res_2d_bw_2))
        result = feed_backward_comb_filter(sig_2d, 3, 0.5)
        self.assertTrue(np.allclose(result, res_2d_bw_3))


class TestFeedBackwardFilterFallbackFunction(unittest.TestCase):

    def test_types(self):
        result = feed_forward_comb_filter(sig_1d, 2, 0.5)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(type(result), float)

    def test_values(self):
        from madmom.audio.filters import _feed_backward_comb_filter
        result = _feed_backward_comb_filter(sig_1d, 2, 0.5)
        self.assertTrue(np.allclose(result, res_1d_bw_2))
        result = _feed_backward_comb_filter(sig_1d, 3, 0.5)
        self.assertTrue(np.allclose(result, res_1d_bw_3))
        result = _feed_backward_comb_filter(sig_2d, 2, 0.5)
        self.assertTrue(np.allclose(result, res_2d_bw_2))
        result = _feed_backward_comb_filter(sig_2d, 3, 0.5)
        self.assertTrue(np.allclose(result, res_2d_bw_3))


class TestCombFilterFunction(unittest.TestCase):

    def test_types(self):
        function = feed_backward_comb_filter
        result = comb_filter(sig_1d, function, [2, 3], [0.5, 0.5])
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(type(result), float)
        function = feed_forward_comb_filter
        result = comb_filter(sig_1d, function, [2, 3], [0.5, 0.5])
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(type(result), float)

    def test_values_backward(self):
        function = feed_backward_comb_filter
        result = comb_filter(sig_1d, function, [2, 3], [0.5, 0.5])
        self.assertTrue(np.allclose(result, [res_1d_bw_2, res_1d_bw_3]))
        result = comb_filter(sig_2d, function, [2, 3], [0.5, 0.5])
        self.assertTrue(np.allclose(result, [res_2d_bw_2, res_2d_bw_3]))

    def test_values_forward(self):
        function = feed_forward_comb_filter
        result = comb_filter(sig_1d, function, [2, 3], [0.5, 0.5])
        self.assertTrue(np.allclose(result, [res_1d_fw_2, res_1d_fw_3]))
        result = comb_filter(sig_2d, function, [2, 3], [0.5, 0.5])
        self.assertTrue(np.allclose(result, [res_2d_fw_2, res_2d_fw_3]))


class TestCombFilterbankClass(unittest.TestCase):

    def test_types(self):
        # backward function
        processor = CombFilterbankProcessor(feed_backward_comb_filter,
                                            [2, 3], [0.5, 0.5])
        self.assertIsInstance(processor, CombFilterbankProcessor)
        self.assertIsInstance(processor, Processor)
        self.assertTrue(processor.comb_filter_function ==
                        feed_backward_comb_filter)
        processor = CombFilterbankProcessor('backward', [2, 3], [0.5, 0.5])
        self.assertTrue(processor.comb_filter_function ==
                        feed_backward_comb_filter)
        # forward function
        processor = CombFilterbankProcessor(feed_forward_comb_filter,
                                            [2, 3], [0.5, 0.5])
        self.assertTrue(processor.comb_filter_function ==
                        feed_forward_comb_filter)
        processor = CombFilterbankProcessor('forward', [2, 3], [0.5, 0.5])
        self.assertTrue(processor.comb_filter_function ==
                        feed_forward_comb_filter)

    def test_errors(self):
        with self.assertRaises(ValueError):
            CombFilterbankProcessor('xyz', [2, 3], [0.5, 0.5])

    def test_values_backward(self):
        processor = CombFilterbankProcessor(feed_backward_comb_filter,
                                            [2, 3], [0.5, 0.5])
        result = processor.process(sig_1d)
        self.assertTrue(np.allclose(result, [res_1d_bw_2, res_1d_bw_3]))
        result = processor.process(sig_2d)
        self.assertTrue(np.allclose(result, [res_2d_bw_2, res_2d_bw_3]))

    def test_values_forward(self):
        processor = CombFilterbankProcessor(feed_forward_comb_filter,
                                            [2, 3], [0.5, 0.5])
        result = processor.process(sig_1d)
        self.assertTrue(np.allclose(result, [res_1d_fw_2, res_1d_fw_3]))
        result = processor.process(sig_2d)
        self.assertTrue(np.allclose(result, [res_2d_fw_2, res_2d_fw_3]))
