# encoding: utf-8
"""
This file contains beat evaluation tests.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import unittest

from madmom.evaluation.beats import *

TARGETS = np.asarray([1., 2, 3, 4, 5, 6, 7, 8, 9, 10])
DOUBLE_TARGETS = np.asarray([1., 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5,
                             7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5])
HALF_TARGETS_EVEN = np.asarray([1., 3, 5, 7, 9])
HALF_TARGETS_ODD = np.asarray([2., 4, 6, 8, 10])
DETECTIONS = np.asarray([1.01, 2, 2.95, 4, 6, 7, 8, 9.1, 10, 11])


# test functions
class TestCalcInterval(unittest.TestCase):

    def test_types(self):
        intervals = calc_intervals(TARGETS)
        self.assertIsInstance(intervals, np.ndarray)
        # events must be correct type
        intervals = calc_intervals([1, 2])
        self.assertIsInstance(intervals, np.ndarray)

    def test_values(self):
        # empty sequences should return 0
        intervals = calc_intervals([])
        self.assertTrue(np.allclose(intervals, []))
        # test targets backwards
        intervals = calc_intervals(TARGETS)
        correct = np.asarray([1., 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.assertTrue(np.allclose(intervals, correct))
        # test detections backwards
        intervals = calc_intervals(DETECTIONS)
        correct = [0.99, 0.99, 0.95, 1.05, 2, 1, 1, 1.1, 0.9, 1]
        self.assertTrue(np.allclose(intervals, correct))
        # test targets forwards
        intervals = calc_intervals(TARGETS, fwd=True)
        correct = np.asarray([1., 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.assertTrue(np.allclose(intervals, correct))
        # test detections forwards
        intervals = calc_intervals(DETECTIONS, fwd=True)
        correct = [0.99, 0.95, 1.05, 2, 1, 1, 1.1, 0.9, 1, 1]
        self.assertTrue(np.allclose(intervals, correct))
        # TODO: same tests with matches given


class TestFindClosestInterval(unittest.TestCase):

    def test_types(self):
        intervals = find_closest_intervals(DETECTIONS, TARGETS)
        self.assertIsInstance(intervals, np.ndarray)
        # events must be correct type
        with self.assertRaises(AttributeError):
            find_closest_intervals([1.5], [1, 2])
        with self.assertRaises(TypeError):
            find_closest_intervals(None, TARGETS)
        with self.assertRaises(TypeError):
            find_closest_intervals(DETECTIONS, None)

    def test_values(self):
        # empty detections should return an empty result
        intervals = find_closest_intervals([], TARGETS)
        self.assertTrue(np.allclose(intervals, []))
        # less than 2 targets should return an empty result
        intervals = find_closest_intervals(DETECTIONS, [])
        self.assertTrue(np.allclose(intervals, []))
        intervals = find_closest_intervals(DETECTIONS, [1.])
        self.assertTrue(np.allclose(intervals, []))
        # test detections w.r.t. targets
        intervals = find_closest_intervals(DETECTIONS, TARGETS)
        correct = np.asarray([1., 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.assertTrue(np.allclose(intervals, correct))
        # intervals = find_closest_intervals(DETECTIONS, TARGETS)
        # correct = np.asarray([1., 1, 1, 1, 1, 1, 1, 1, 1, 1])
        # test targets w.r.t. detections
        intervals = find_closest_intervals(TARGETS, DETECTIONS)
        correct = [0.99, 0.99, 1.05, 1.05, 2, 2, 1, 1, 1.1, 0.9]
        self.assertTrue(np.allclose(intervals, correct))
        # TODO: same tests with matches given


class TestCalcRelativeErrorsFunction(unittest.TestCase):

    def test_types(self):
        rel_errors = calc_relative_errors(DETECTIONS, TARGETS)
        self.assertIsInstance(rel_errors, np.ndarray)
        # events must be correct type
        with self.assertRaises(AttributeError):
            calc_relative_errors([1.5], [1, 2])
        with self.assertRaises(TypeError):
            calc_relative_errors(None, TARGETS)
        with self.assertRaises(TypeError):
            calc_relative_errors(DETECTIONS, None)

    def test_values(self):
        # empty detections should return an empty result
        errors = calc_relative_errors([], TARGETS)
        self.assertTrue(np.allclose(errors, []))
        # less than 2 targets should return an empty result
        errors = calc_relative_errors(DETECTIONS, [])
        self.assertTrue(np.allclose(errors, []))
        errors = calc_relative_errors(DETECTIONS, [1.])
        self.assertTrue(np.allclose(errors, []))
        # test detections w.r.t. targets
        errors = calc_relative_errors(DETECTIONS, TARGETS)
        # det: [1.01, 2, 2.95, 4,    6, 7, 8, 9.1, 10, 11]
        # tar: [1,    2, 3,    4, 5, 6, 7, 8, 9,   10]
        correct = [0.01, 0, -0.05, 0, 0, 0, 0, 0.1, 0, 1]
        # all intervals are 1, so need for division
        self.assertTrue(np.allclose(errors, correct))
        # test targets w.r.t. detections
        errors = calc_relative_errors(TARGETS, DETECTIONS)
        # tar: [1,    2, 3,    4, 5, 6, 7, 8, 9,   10]
        # det: [1.01, 2, 2.95, 4,    6, 7, 8, 9.1, 10, 11]
        errors_ = np.asarray([-0.01, 0, 0.05, 0, -1, 0, 0, 0, -0.1, 0])
        intervals_ = np.asarray([0.99, 0.99, 1.05, 1.05, 2, 2, 1, 1, 1.1, 0.9])
        self.assertTrue(np.allclose(errors, errors_ / intervals_))
        # TODO: same tests with matches given


class TestPscoreFunction(unittest.TestCase):

    def test_types(self):
        score = pscore(DETECTIONS, TARGETS, 0.2)
        self.assertIsInstance(score, float)
        # detections / targets must be correct type
        score = pscore([], [], 0.2)
        self.assertIsInstance(score, float)
        score = pscore({}, {}, 0.2)
        self.assertIsInstance(score, float)
        with self.assertRaises(AttributeError):
            pscore(DETECTIONS.tolist(), TARGETS.tolist(), 0.2)
        with self.assertRaises(TypeError):
            pscore(None, TARGETS, 0.2)
        with self.assertRaises(TypeError):
            pscore(DETECTIONS, None, 0.2)
        # tolerance must be correct type
        score = pscore(DETECTIONS, TARGETS, int(1.2))
        self.assertIsInstance(score, float)
        with self.assertRaises(ValueError):
            pscore(DETECTIONS, TARGETS, [])
        with self.assertRaises(TypeError):
            pscore(DETECTIONS, TARGETS, {})

    def test_values(self):
        # tolerance must be > 0
        with self.assertRaises(ValueError):
            pscore(DETECTIONS, TARGETS, 0)
        with self.assertRaises(ValueError):
            pscore(DETECTIONS, TARGETS, None)
        # empty sequences should return 0
        score = pscore([], [], 0.2)
        self.assertEqual(score, 0)
        # score relies on intervals, hence at least 2 targets must be given
        score = pscore(DETECTIONS, [1], 0.2)
        self.assertEqual(score, 0)
        # no detections should return 0
        score = pscore([], TARGETS, 0.2)
        self.assertEqual(score, 0)
        # no targets should return 0
        score = pscore(DETECTIONS, [], 0.2)
        self.assertEqual(score, 0)
        # normal calculation
        score = pscore(DETECTIONS, TARGETS, 0.2)
        self.assertEqual(score, 0.9)


class TestCemgilFunction(unittest.TestCase):

    def test_types(self):
        score = cemgil(DETECTIONS, TARGETS, 0.04)
        self.assertIsInstance(score, float)
        # detections / targets must be correct type
        score = cemgil([], [], 0.04)
        self.assertIsInstance(score, float)
        score = cemgil({}, {}, 0.04)
        self.assertIsInstance(score, float)
        with self.assertRaises(AttributeError):
            cemgil(DETECTIONS.tolist(), TARGETS.tolist(), 0.04)
        with self.assertRaises(TypeError):
            cemgil(None, TARGETS, 0.04)
        with self.assertRaises(TypeError):
            cemgil(DETECTIONS, None, 0.04)
        # tolerance must be correct type
        score = cemgil(DETECTIONS, TARGETS, int(1))
        self.assertIsInstance(score, float)
        with self.assertRaises(TypeError):
            cemgil(DETECTIONS, TARGETS, [0.04])
        with self.assertRaises(TypeError):
            cemgil(DETECTIONS, TARGETS, {0: 0.04})
        with self.assertRaises(TypeError):
            cemgil(DETECTIONS, TARGETS, {0.04: 0})

    def test_values(self):
        # sigma must be greater than 0
        with self.assertRaises(ValueError):
            cemgil(DETECTIONS, TARGETS, 0)
        with self.assertRaises(ValueError):
            cemgil(DETECTIONS, TARGETS, None)
        # empty sequences should return 0
        score = cemgil([], [], 0.04)
        self.assertEqual(score, 0)
        # no detections should return 0
        score = cemgil([], TARGETS, 0.04)
        self.assertEqual(score, 0)
        # no targets should return 0
        score = cemgil(DETECTIONS, [], 0.04)
        self.assertEqual(score, 0)
        # normal calculation
        score = cemgil(DETECTIONS, TARGETS, 0.04)
        self.assertEqual(score, 0.74710035298713695)


class TestCmlFunction(unittest.TestCase):

    def test_types(self):
        cmlc, cmlt = cml(DETECTIONS, TARGETS, 0.175, 0.175)
        self.assertIsInstance(cmlc, float)
        self.assertIsInstance(cmlt, float)
        # detections / targets must be correct type
        cmlc, cmlt = cml([], [], 0.175, 0.175)
        self.assertIsInstance(cmlc, float)
        self.assertIsInstance(cmlt, float)
        cmlc, cmlt = cml({}, {}, 0.175, 0.175)
        self.assertIsInstance(cmlc, float)
        self.assertIsInstance(cmlt, float)
        with self.assertRaises(AttributeError):
            cml(DETECTIONS.tolist(), TARGETS.tolist(), 0.175, 0.175)
        with self.assertRaises(TypeError):
            cml(None, TARGETS, 0.175, 0.175)
        with self.assertRaises(TypeError):
            cml(DETECTIONS, None, 0.175, 0.175)
        # tolerances must be correct type
        cmlc, cmlt = cml(DETECTIONS, TARGETS, int(1), int(1))
        self.assertIsInstance(cmlc, float)
        self.assertIsInstance(cmlt, float)
        cmlc, cmlt = cml(DETECTIONS, TARGETS, [0.175], [0.175])
        self.assertIsInstance(cmlc, float)
        self.assertIsInstance(cmlt, float)
        with self.assertRaises(TypeError):
            cml(DETECTIONS, TARGETS, {}, {})

    def test_values(self):
        # tolerances must be greater than 0
        with self.assertRaises(ValueError):
            cml(DETECTIONS, TARGETS, 0, None)
        with self.assertRaises(ValueError):
            cml(DETECTIONS, TARGETS, None, 0)
        # empty sequences should return 0
        scores = cml([], [], 0.175, 0.175)
        self.assertEqual(scores, (0, 0))
        # no detections should return 0
        scores = cml([], TARGETS, 0.175, 0.175)
        self.assertEqual(scores, (0, 0))
        # less than 2 targets should return 0
        scores = cml(DETECTIONS, [], 0.175, 0.175)
        self.assertEqual(scores, (0, 0))
        scores = cml(DETECTIONS, [1.], 0.175, 0.175)
        self.assertEqual(scores, (0, 0))
        # normal calculation
        scores = cml(DETECTIONS, TARGETS, 0.175, 0.175)
        self.assertEqual(scores, (0.4, 0.8))


class TestContinuityFunction(unittest.TestCase):

    def test_types(self):
        cmlc, cmlt, amlc, amlt = continuity(DETECTIONS, TARGETS, 0.175, 0.175)
        self.assertIsInstance(cmlc, float)
        self.assertIsInstance(cmlt, float)
        self.assertIsInstance(amlc, float)
        self.assertIsInstance(amlt, float)
        # detections / targets must be correct type
        cmlc, cmlt, amlc, amlt = continuity([], [], 0.175, 0.175)
        self.assertIsInstance(cmlc, float)
        self.assertIsInstance(cmlt, float)
        self.assertIsInstance(amlc, float)
        self.assertIsInstance(amlt, float)
        cmlc, cmlt, amlc, amlt = continuity({}, {}, 0.175, 0.175)
        self.assertIsInstance(cmlc, float)
        self.assertIsInstance(cmlt, float)
        self.assertIsInstance(amlc, float)
        self.assertIsInstance(amlt, float)
        with self.assertRaises(AttributeError):
            continuity(DETECTIONS.tolist(), TARGETS.tolist(), 0.175, 0.175)
        with self.assertRaises(TypeError):
            continuity(None, TARGETS, 0.175, 0.175)
        with self.assertRaises(TypeError):
            continuity(DETECTIONS, None, 0.175, 0.175)
        # tolerances must be correct type
        cmlc, cmlt, amlc, amlt = continuity(DETECTIONS, TARGETS, int(1), int(1))
        self.assertIsInstance(cmlc, float)
        self.assertIsInstance(cmlt, float)
        self.assertIsInstance(amlc, float)
        self.assertIsInstance(amlt, float)
        cmlc, cmlt, amlc, amlt = continuity(DETECTIONS, TARGETS, [0.175], [0.175])
        self.assertIsInstance(cmlc, float)
        self.assertIsInstance(cmlt, float)
        self.assertIsInstance(amlc, float)
        self.assertIsInstance(amlt, float)
        with self.assertRaises(TypeError):
            continuity(DETECTIONS, TARGETS, {}, {})

    def test_values(self):
        # tolerances must be greater than 0
        with self.assertRaises(ValueError):
            continuity(DETECTIONS, TARGETS, 0, None)
        with self.assertRaises(ValueError):
            continuity(DETECTIONS, TARGETS, None, 0)
        # empty sequences should return 0
        scores = continuity([], [], 0.175, 0.175)
        self.assertEqual(scores, (0, 0, 0, 0))
        # no detections should return 0
        scores = continuity([], TARGETS, 0.175, 0.175)
        self.assertEqual(scores, (0, 0, 0, 0))
        # less than 2 targets should return 0
        scores = continuity(DETECTIONS, [], 0.175, 0.175)
        self.assertEqual(scores, (0, 0, 0, 0))
        scores = continuity(DETECTIONS, [1.], 0.175, 0.175)
        self.assertEqual(scores, (0, 0, 0, 0))
        # normal calculation
        scores = continuity(DETECTIONS, TARGETS, 0.175, 0.175)
        self.assertEqual(scores, (0.4, 0.8, 0.4, 0.8))
        # double tempo targets
        scores = continuity(DETECTIONS, DOUBLE_TARGETS, 0.175, 0.175)
        self.assertEqual(scores, (0., 0., 0.4, 0.8))
        # half tempo targets (even beats)
        scores = continuity(DETECTIONS, DOUBLE_TARGETS, 0.175, 0.175)
        self.assertEqual(scores, (0., 0., 0.4, 0.8))
        # half tempo targets (odd beats)
        scores = continuity(DETECTIONS, DOUBLE_TARGETS, 0.175, 0.175)
        self.assertEqual(scores, (0., 0., 0.4, 0.8))


class TestInformationGainFunction(unittest.TestCase):

    def test_types(self):
        ig, histogram = information_gain(DETECTIONS, TARGETS, 40)
        self.assertIsInstance(ig, float)
        self.assertIsInstance(histogram, np.ndarray)
        # detections / targets must be correct type
        ig, histogram = information_gain([], [], 40)
        self.assertIsInstance(ig, float)
        self.assertIsInstance(histogram, np.ndarray)
        ig, histogram = information_gain({}, {}, 40)
        self.assertIsInstance(ig, float)
        self.assertIsInstance(histogram, np.ndarray)
        with self.assertRaises(AttributeError):
            information_gain(DETECTIONS.tolist(), TARGETS.tolist(), 40)
        with self.assertRaises(TypeError):
            information_gain(None, TARGETS, 40)
        with self.assertRaises(TypeError):
            information_gain(DETECTIONS, None, 40)
        # tolerances must be correct type
        ig, histogram = information_gain(DETECTIONS, TARGETS, 40)
        self.assertIsInstance(ig, float)
        self.assertIsInstance(histogram, np.ndarray)
        ig, histogram = information_gain(DETECTIONS, TARGETS, 40)
        self.assertIsInstance(ig, float)
        self.assertIsInstance(histogram, np.ndarray)

    def test_values(self):
        # bins must be even and greater or equal than 2, independently of the
        # length of detections
        with self.assertRaises(ValueError):
            information_gain(DETECTIONS, TARGETS, 1)
        with self.assertRaises(ValueError):
            information_gain([], [], 1)
        with self.assertRaises(ValueError):
            information_gain(DETECTIONS, TARGETS, 2.1)
        with self.assertRaises(ValueError):
            information_gain(DETECTIONS, TARGETS, 5)
        # empty sequences should return 0 and a uniform histogram
        ig, histogram = information_gain([], [], 4)
        self.assertEqual(ig, 0)
        self.assertTrue(np.allclose(histogram, np.ones(4) / 4.))
        # less than 2 detections should return 0 and a uniform histogram
        ig, histogram = information_gain([], TARGETS, 4)
        self.assertEqual(ig, 0)
        self.assertTrue(np.allclose(histogram, np.ones(4) / 4.))
        ig, histogram = information_gain([1.], TARGETS, 4)
        self.assertEqual(ig, 0)
        self.assertTrue(np.allclose(histogram, np.ones(4) / 4.))
        # less than 2 targets should return 0 and a uniform histogram
        ig, histogram = information_gain(DETECTIONS, [], 4)
        self.assertEqual(ig, 0)
        self.assertTrue(np.allclose(histogram, np.ones(4) / 4.))
        ig, histogram = information_gain(DETECTIONS, [1.], 4)
        self.assertEqual(ig, 0)
        self.assertTrue(np.allclose(histogram, np.ones(4) / 4.))
        # normal calculation
        ig, histogram = information_gain(DETECTIONS, TARGETS, 4)
        # tar: [1,    2, 3,    4, 5, 6, 7, 8, 9,   10]
        # det: [1.01, 2, 2.95, 4,    6, 7, 8, 9.1, 10, 11]
        # errors: [-0.01, 0, 0.05, 0, -1, 0, 0, 0, -0.1, 0]
        # intervals: [0.99, 0.99, 1.05, 1.05, 2, 2, 1, 1, 1.1, 0.9]
        # rel. err.: [-0.01010101, 0, 0.04761905, 0, -0.5, 0, 0, 0,
        #             -0.09090909, 0]
        # bin edges: [-0.625 -0.375 -0.125  0.125  0.375  0.625]
        # bin count: [1, 0, 9, 0]
        self.assertTrue(np.allclose(histogram, [1, 0, 9, 0]))
        histogram_ = np.asarray([1, 0, 9, 0], np.float)
        histogram_ /= np.sum(histogram_)
        histogram_[histogram_ == 0] = 1
        entropy = - np.sum(histogram_ * np.log2(histogram_))
        self.assertEqual(ig, np.log2(4) - entropy)


# test evaluation class
class TestBeatEvaluationClass(unittest.TestCase):

    def test_types(self):
        e = BeatEvaluation(DETECTIONS, TARGETS)
        # from OnsetEvaluation
        self.assertIsInstance(e.num_tp, int)
        self.assertIsInstance(e.num_fp, int)
        self.assertIsInstance(e.num_tn, int)
        self.assertIsInstance(e.num_fn, int)
        self.assertIsInstance(e.precision, float)
        self.assertIsInstance(e.recall, float)
        self.assertIsInstance(e.fmeasure, float)
        self.assertIsInstance(e.accuracy, float)
        self.assertIsInstance(e.errors, np.ndarray)
        self.assertIsInstance(e.mean_error, float)
        self.assertIsInstance(e.std_error, float)
        # additional beat score types
        self.assertIsInstance(e.pscore, float)
        self.assertIsInstance(e.cemgil, float)
        self.assertIsInstance(e.cmlc, float)
        self.assertIsInstance(e.cmlt, float)
        self.assertIsInstance(e.amlc, float)
        self.assertIsInstance(e.amlt, float)
        self.assertIsInstance(e.information_gain, float)
        self.assertIsInstance(e.global_information_gain, float)
        self.assertIsInstance(e.error_histogram, np.ndarray)

    def test_conversion(self):
        # conversion from list should work
        e = BeatEvaluation([], [])
        self.assertIsInstance(e.tp, np.ndarray)
        self.assertIsInstance(e.fp, np.ndarray)
        self.assertIsInstance(e.tn, np.ndarray)
        self.assertIsInstance(e.fn, np.ndarray)
        # conversion from dict should work as well
        e = BeatEvaluation({}, {})
        self.assertIsInstance(e.tp, np.ndarray)
        self.assertIsInstance(e.fp, np.ndarray)
        self.assertIsInstance(e.tn, np.ndarray)
        self.assertIsInstance(e.fn, np.ndarray)
        # others should fail
        self.assertRaises(TypeError, BeatEvaluation, float(0), float(0))
        self.assertRaises(TypeError, BeatEvaluation, int(0), int(0))

    def test_results_empty(self):
        e = BeatEvaluation([], [])
        self.assertEqual(e.fmeasure, 1)
        self.assertEqual(e.pscore, 0)
        self.assertEqual(e.cemgil, 0)
        self.assertEqual(e.cmlc, 0)
        self.assertEqual(e.cmlt, 0)
        self.assertEqual(e.amlc, 0)
        self.assertEqual(e.amlt, 0)
        self.assertEqual(e.information_gain, 0)
        self.assertEqual(e.global_information_gain, 0)
        self.assertTrue(np.allclose(e.error_histogram, np.ones(40) / 40.))

    def test_results(self):
        e = BeatEvaluation(DETECTIONS, TARGETS)
        # tar: [1,    2, 3,    4, 5, 6, 7, 8, 9,   10]
        # det: [1.01, 2, 2.95, 4,    6, 7, 8, 9.1, 10, 11]
        # WINDOW = 0.07
        # TOLERANCE = 0.2
        # SIGMA = 0.04
        # TEMPO_TOLERANCE = 0.175
        # PHASE_TOLERANCE = 0.175
        # BINS = 40
        self.assertEqual(e.tp.tolist(), [1.01, 2, 2.95, 4, 6, 7, 8, 10])
        self.assertEqual(e.fp.tolist(), [9.1, 11])
        self.assertEqual(e.tn.tolist(), [])
        self.assertEqual(e.fn.tolist(), [5, 9])
        self.assertEqual(e.num_tp, 8)
        self.assertEqual(e.num_fp, 2)
        self.assertEqual(e.num_tn, 0)
        self.assertEqual(e.num_fn, 2)
        self.assertEqual(e.precision, 8. / 10.)
        self.assertEqual(e.recall, 8. / 10.)
        f = 2 * (8. / 10.) * (8. / 10.) / ((8. / 10.) + (8. / 10.))
        self.assertEqual(e.fmeasure, f)
        self.assertEqual(e.accuracy, (8. + 0) / (8 + 2 + 0 + 2))
        # pscore: delta <= tolerance * median(inter beat interval)
        self.assertEqual(e.pscore, 9. / 10.)
        # cemgil:
        self.assertEqual(e.cemgil, 0.74710035298713695)
        self.assertEqual(e.cmlc, 0.4)
        self.assertEqual(e.cmlt, 0.8)
        self.assertEqual(e.amlc, 0.4)
        self.assertEqual(e.amlt, 0.8)
        self.assertEqual(e.information_gain, 3.965148445440323)
        self.assertEqual(e.global_information_gain, 3.965148445440323)
        error_histogram_ = np.zeros(40)
        error_histogram_[0] = 1
        error_histogram_[16] = 1
        error_histogram_[20] = 7
        error_histogram_[22] = 1
        self.assertTrue(np.allclose(e.error_histogram, error_histogram_))


class TestMeanBeatEvaluationClass(unittest.TestCase):

    def test_types(self):
        e = MeanBeatEvaluation()
        # scores
        self.assertIsInstance(e.fmeasure, float)
        self.assertIsInstance(e.pscore, float)
        self.assertIsInstance(e.cemgil, float)
        self.assertIsInstance(e.cmlc, float)
        self.assertIsInstance(e.cmlt, float)
        self.assertIsInstance(e.amlc, float)
        self.assertIsInstance(e.amlt, float)
        self.assertIsInstance(e.information_gain, float)
        self.assertIsInstance(e.global_information_gain, float)
        # error histogram is initially None
        self.assertIsNone(e.error_histogram)

    def test_append(self):
        e = MeanBeatEvaluation()
        e.append(BeatEvaluation(DETECTIONS, TARGETS))
        self.assertIsInstance(e, MeanBeatEvaluation)
        # appending something valid should not return anything
        self.assertEqual(e.append(BeatEvaluation(DETECTIONS, TARGETS)), None)
        # appending something else should not work

    def test_append_types(self):
        e = MeanBeatEvaluation()
        e.append(BeatEvaluation(DETECTIONS, TARGETS))
        # scores
        self.assertIsInstance(e.fmeasure, float)
        self.assertIsInstance(e.pscore, float)
        self.assertIsInstance(e.cemgil, float)
        self.assertIsInstance(e.cmlc, float)
        self.assertIsInstance(e.cmlt, float)
        self.assertIsInstance(e.amlc, float)
        self.assertIsInstance(e.amlt, float)
        self.assertIsInstance(e.information_gain, float)
        self.assertIsInstance(e.global_information_gain, float)
        # error histogram
        self.assertIsInstance(e.error_histogram, np.ndarray)

    def test_results_empty(self):
        e = MeanBeatEvaluation()
        e.append(BeatEvaluation([], []))
        self.assertEqual(e.fmeasure, 1)
        self.assertEqual(e.pscore, 0)
        self.assertEqual(e.cemgil, 0)
        self.assertEqual(e.cmlc, 0)
        self.assertEqual(e.cmlt, 0)
        self.assertEqual(e.amlc, 0)
        self.assertEqual(e.amlt, 0)
        self.assertEqual(e.information_gain, 0)
        self.assertEqual(e.global_information_gain, 0)
        self.assertTrue(np.allclose(e.error_histogram, np.ones(40) / 40.))

    def test_results(self):
        e = MeanBeatEvaluation()
        e.append(BeatEvaluation(DETECTIONS, TARGETS))
        f = 2 * (8. / 10.) * (8. / 10.) / ((8. / 10.) + (8. / 10.))
        self.assertEqual(e.fmeasure, f)
        self.assertEqual(e.pscore, 9. / 10.)
        self.assertEqual(e.cemgil, 0.74710035298713695)
        self.assertEqual(e.cmlc, 0.4)
        self.assertEqual(e.cmlt, 0.8)
        self.assertEqual(e.amlc, 0.4)
        self.assertEqual(e.amlt, 0.8)
        self.assertEqual(e.information_gain, 3.965148445440323)
        self.assertEqual(e.global_information_gain, 3.965148445440323)
        error_histogram_ = np.zeros(40)
        error_histogram_[0] = 1
        error_histogram_[16] = 1
        error_histogram_[20] = 7
        error_histogram_[22] = 1
        self.assertTrue(np.allclose(e.error_histogram, error_histogram_))
