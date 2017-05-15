# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the programs in the /bin directory.

"""

from __future__ import absolute_import, division, print_function

import unittest
import os
import imp
import sys
import tempfile
from os.path import join as pj

try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

import numpy as np

from madmom.features import Activations
from madmom.features.chords import load_chords

from . import AUDIO_PATH, ACTIVATIONS_PATH, DETECTIONS_PATH


tmp_act = tempfile.NamedTemporaryFile(delete=False).name
tmp_result = tempfile.NamedTemporaryFile(delete=False).name
sample_file = pj(AUDIO_PATH, 'sample.wav')
sample2_file = pj(AUDIO_PATH, 'sample2.wav')
stereo_sample_file = pj(AUDIO_PATH, 'stereo_sample.wav')
program_path = os.path.dirname(os.path.realpath(__file__)) + '/../bin/'

# prevent writing compiled Python files to disk
sys.dont_write_bytecode = True


def run_program(program):
    # import module, capture stdout
    test = imp.load_source('test', program[0])
    sys.argv = program
    backup = sys.stdout
    sys.stdout = StringIO()
    # run the program
    data = test.main()
    # close stdout, restore environment
    sys.stdout.getvalue()
    sys.stdout.close()
    sys.stdout = backup
    return data


def run_help(program):
    test = imp.load_source('test', program)
    sys.argv = [program, '-h']
    try:
        test.main()
    except SystemExit:
        return True
    return False


# TODO: parametrize tests, don't know how to do with nose, should be simple
#       with pytest: http://pytest.org/latest/parametrize.html

# TODO: can we speed up these tests?


class TestBeatDetectorProgram(unittest.TestCase):
    def setUp(self):
        self.bin = pj(program_path, "BeatDetector")
        self.activations = Activations(
            pj(ACTIVATIONS_PATH, "sample.beats_blstm.npz"))
        self.result = np.loadtxt(
            pj(DETECTIONS_PATH, "sample.beat_detector.txt"))

    def test_help(self):
        self.assertTrue(run_help(self.bin))

    def test_binary(self):
        # save activations as binary file
        run_program([self.bin, '--save', 'single', sample_file, '-o', tmp_act])
        act = Activations(tmp_act)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        self.assertEqual(act.fps, self.activations.fps)
        # reload from file
        run_program([self.bin, '--load', 'single', tmp_act, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_txt(self):
        # save activations as txt file
        run_program([self.bin, '--save', '--sep', ' ', 'single', sample_file,
                     '-o', tmp_act])
        act = Activations(tmp_act, sep=' ', fps=100)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        # reload from file
        run_program([self.bin, '--load', '--sep', ' ', 'single', tmp_act,
                     '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_run(self):
        run_program([self.bin, 'single', sample_file, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))


class TestBeatTrackerProgram(unittest.TestCase):
    def setUp(self):
        self.bin = pj(program_path, "BeatTracker")
        self.activations = Activations(
            pj(ACTIVATIONS_PATH, "sample.beats_blstm.npz"))
        self.result = np.loadtxt(
            pj(DETECTIONS_PATH, "sample.beat_tracker.txt"))

    def test_help(self):
        self.assertTrue(run_help(self.bin))

    def test_binary(self):
        # save activations as binary file
        run_program([self.bin, '--save', 'single', sample_file, '-o', tmp_act])
        act = Activations(tmp_act)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        self.assertEqual(act.fps, self.activations.fps)
        # reload from file
        run_program([self.bin, '--load', 'single', tmp_act, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_txt(self):
        # save activations as txt file
        run_program([self.bin, '--save', '--sep', ' ', 'single', sample_file,
                     '-o', tmp_act])
        act = Activations(tmp_act, sep=' ', fps=100)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        # reload from file
        run_program([self.bin, '--load', '--sep', ' ', 'single', tmp_act, '-o',
                     tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_run(self):
        run_program([self.bin, 'single', sample_file, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))


class TestCNNChordRecognition(unittest.TestCase):
    def setUp(self):
        self.bin = pj(program_path, "CNNChordRecognition")
        self.activations = [
            Activations(pj(ACTIVATIONS_PATH, af))
            for af in ['sample.cnn_chord_features.npz',
                       'sample2.cnn_chord_features.npz']
        ]
        self.results = [
            load_chords(pj(DETECTIONS_PATH, df))
            for df in ['sample.cnn_chord_recognition.txt',
                       'sample2.cnn_chord_recognition.txt']
        ]

    def _check_results(self, result, true_result):
        self.assertTrue(np.allclose(result['start'], true_result['start']))
        self.assertTrue(np.allclose(result['end'], true_result['end']))
        self.assertTrue((result['label'] == true_result['label']).all())

    def test_help(self):
        self.assertTrue(run_help(self.bin))

    def test_binary(self):
        for sf, true_act, true_res in zip([sample_file, sample2_file],
                                          self.activations,
                                          self.results):
            # save activations as binary file
            run_program([self.bin, '--save', 'single', sf, '-o', tmp_act])
            act = Activations(tmp_act)
            self.assertTrue(np.allclose(act, true_act, atol=1e-5))
            self.assertEqual(act.fps, true_act.fps)
            # reload from file
            run_program([self.bin, '--load', 'single', tmp_act,
                         '-o', tmp_result])
            self._check_results(load_chords(tmp_result), true_res)

    def test_txt(self):
        for sf, true_act, true_res in zip([sample_file, sample2_file],
                                          self.activations,
                                          self.results):
            # save activations as txt file
            run_program([self.bin, '--save', '--sep', ' ', 'single', sf,
                         '-o', tmp_act])
            act = Activations(tmp_act, sep=' ', fps=100)
            self.assertTrue(np.allclose(act, true_act, atol=1e-5))
            # reload from file
            run_program([self.bin, '--load', '--sep', ' ', 'single', tmp_act,
                         '-o', tmp_result])
            self._check_results(load_chords(tmp_result), true_res)

    def test_run(self):
        for sf, true_res in zip([sample_file, sample2_file], self.results):
            run_program([self.bin, 'single', sf, '-o', tmp_result])
            self._check_results(load_chords(tmp_result), true_res)


class TestComplexFluxProgram(unittest.TestCase):
    def setUp(self):
        self.bin = pj(program_path, "ComplexFlux")
        self.activations = Activations(
            pj(ACTIVATIONS_PATH, "sample.complex_flux.npz"))
        self.result = np.loadtxt(
            pj(DETECTIONS_PATH, "sample.complex_flux.txt"))

    def test_help(self):
        self.assertTrue(run_help(self.bin))

    def test_binary(self):
        # save activations as binary file
        run_program([self.bin, '--save', 'single', sample_file, '-o', tmp_act])
        act = Activations(tmp_act)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        self.assertEqual(act.fps, self.activations.fps)
        # reload from file
        run_program([self.bin, '--load', 'single', tmp_act, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_txt(self):
        # save activations as txt file
        run_program([self.bin, '--save', '--sep', ' ', 'single', sample_file,
                     '-o', tmp_act])
        act = Activations(tmp_act, sep=' ', fps=100)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        # reload from file
        run_program([self.bin, '--load', '--sep', ' ', 'single', tmp_act,
                     '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_run(self):
        run_program([self.bin, 'single', sample_file, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))


class TestCNNOnsetDetectorProgram(unittest.TestCase):
    def setUp(self):
        self.bin = pj(program_path, "CNNOnsetDetector")
        self.activations = Activations(
            pj(ACTIVATIONS_PATH, "sample.onsets_cnn.npz"))
        self.result = np.loadtxt(
            pj(DETECTIONS_PATH, "sample.cnn_onset_detector.txt"))

    def test_help(self):
        self.assertTrue(run_help(self.bin))

    def test_binary(self):
        # save activations as binary file
        run_program([self.bin, '--save', 'single', sample_file, '-o', tmp_act])
        act = Activations(tmp_act)
        self.assertTrue(np.allclose(act, self.activations))
        self.assertEqual(act.fps, self.activations.fps)
        # reload from file
        run_program([self.bin, '--load', 'single', tmp_act, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result))

    def test_txt(self):
        # save activations as txt file
        run_program([self.bin, '--save', '--sep', ' ', 'single', sample_file,
                     '-o', tmp_act])
        act = Activations(tmp_act, sep=' ', fps=100)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        # reload from file
        run_program([self.bin, '--load', '--sep', ' ', 'single', tmp_act,
                     '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result))

    def test_run(self):
        run_program([self.bin, 'single', sample_file, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result))


class TestCRFBeatDetectorProgram(unittest.TestCase):
    def setUp(self):
        self.bin = pj(program_path, "CRFBeatDetector")
        self.activations = Activations(
            pj(ACTIVATIONS_PATH, "sample.beats_blstm.npz"))
        self.result = np.loadtxt(
            pj(DETECTIONS_PATH, "sample.crf_beat_detector.txt"))

    def test_help(self):
        self.assertTrue(run_help(self.bin))

    def test_binary(self):
        # save activations as binary file
        run_program([self.bin, '--save', 'single', sample_file, '-o', tmp_act])
        act = Activations(tmp_act)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        self.assertEqual(act.fps, self.activations.fps)
        # reload from file
        run_program([self.bin, '--load', 'single', tmp_act, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_txt(self):
        # save activations as txt file
        run_program([self.bin, '--save', '--sep', ' ', 'single', sample_file,
                     '-o', tmp_act])
        act = Activations(tmp_act, sep=' ', fps=100)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        # reload from file
        run_program([self.bin, '--load', '--sep', ' ', 'single', tmp_act,
                     '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_run(self):
        run_program([self.bin, 'single', sample_file, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))


class TestDBNBeatTrackerProgram(unittest.TestCase):
    def setUp(self):
        self.bin = pj(program_path, "DBNBeatTracker")
        self.activations = Activations(
            pj(ACTIVATIONS_PATH, "sample.beats_blstm.npz"))
        self.result = np.loadtxt(
            pj(DETECTIONS_PATH, "sample.dbn_beat_tracker.txt"))
        self.online_results = [0.47, 0.79, 1.48, 2.16, 2.5]

    def test_help(self):
        self.assertTrue(run_help(self.bin))

    def test_binary(self):
        # save activations as binary file
        run_program([self.bin, '--save', 'single', sample_file, '-o', tmp_act])
        act = Activations(tmp_act)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        self.assertEqual(act.fps, self.activations.fps)
        # reload from file
        run_program([self.bin, '--load', 'single', tmp_act, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_txt(self):
        # save activations as txt file
        run_program([self.bin, '--save', '--sep', ' ', 'single', sample_file,
                     '-o', tmp_act])
        act = Activations(tmp_act, sep=' ', fps=100)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        # reload from file
        run_program([self.bin, '--load', '--sep', ' ', 'single', tmp_act,
                     '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_run(self):
        run_program([self.bin, 'single', sample_file, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_online(self):
        run_program([self.bin, 'online', sample_file, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.online_results))
        run_program([self.bin, 'single', '--online', sample_file, '-o',
                     tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.online_results))


class TestDBNDownBeatTrackerProgram(unittest.TestCase):
    def setUp(self):
        self.bin = pj(program_path, "DBNDownBeatTracker")
        self.activations = Activations(
            pj(ACTIVATIONS_PATH, "sample.downbeats_blstm.npz"))
        self.result = np.loadtxt(
            pj(DETECTIONS_PATH, "sample.dbn_downbeat_tracker.txt"))
        self.downbeat_result = self.result[self.result[:, 1] == 1][:, 0]

    def test_help(self):
        self.assertTrue(run_help(self.bin))

    def test_binary(self):
        # save activations as binary file
        run_program([self.bin, '--save', 'single', sample_file, '-o', tmp_act])
        act = Activations(tmp_act)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        self.assertEqual(act.fps, self.activations.fps)
        # reload from file
        run_program([self.bin, '--load', 'single', tmp_act, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_txt(self):
        # save activations as txt file
        run_program([self.bin, '--save', '--sep', ' ', 'single', sample_file,
                     '-o', tmp_act])
        act = Activations(tmp_act, sep=' ', fps=100)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        # reload from file
        run_program([self.bin, '--load', '--sep', ' ', 'single', tmp_act,
                     '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_run(self):
        run_program([self.bin, 'single', sample_file, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_run_downbeats(self):
        run_program([self.bin, '--downbeats', 'single', sample_file,
                     '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.downbeat_result, atol=1e-5))


class TestDCChordRecognition(unittest.TestCase):
    def setUp(self):
        self.bin = pj(program_path, "DCChordRecognition")
        self.activations = [
            Activations(pj(ACTIVATIONS_PATH, af))
            for af in ['sample.deep_chroma.npz', 'sample2.deep_chroma.npz']
        ]
        self.results = [
            load_chords(pj(DETECTIONS_PATH, df))
            for df in ['sample.dc_chord_recognition.txt',
                       'sample2.dc_chord_recognition.txt']
        ]

    def _check_results(self, result, true_result):
        self.assertTrue(np.allclose(result['start'], true_result['start']))
        self.assertTrue(np.allclose(result['end'], true_result['end']))
        self.assertTrue((result['label'] == true_result['label']).all())

    def test_help(self):
        self.assertTrue(run_help(self.bin))

    def test_binary(self):
        for sf, true_act, true_res in zip([sample_file, sample2_file],
                                          self.activations,
                                          self.results):
            # save activations as binary file
            run_program([self.bin, '--save', 'single', sf, '-o', tmp_act])
            act = Activations(tmp_act)
            self.assertTrue(np.allclose(act, true_act, atol=1e-5))
            self.assertEqual(act.fps, true_act.fps)
            # reload from file
            run_program([self.bin, '--load', 'single', tmp_act,
                         '-o', tmp_result])
            self._check_results(load_chords(tmp_result), true_res)

    def test_txt(self):
        for sf, true_act, true_res in zip([sample_file, sample2_file],
                                          self.activations,
                                          self.results):
            # save activations as txt file
            run_program([self.bin, '--save', '--sep', ' ', 'single', sf,
                         '-o', tmp_act])
            act = Activations(tmp_act, sep=' ', fps=100)
            self.assertTrue(np.allclose(act, true_act, atol=1e-5))
            # reload from file
            run_program([self.bin, '--load', '--sep', ' ', 'single', tmp_act,
                         '-o', tmp_result])
            self._check_results(load_chords(tmp_result), true_res)

    def test_run(self):
        for sf, true_res in zip([sample_file, sample2_file], self.results):
            run_program([self.bin, 'single', sf, '-o', tmp_result])
            self._check_results(load_chords(tmp_result), true_res)


class TestGMMPatternTrackerProgram(unittest.TestCase):
    def setUp(self):
        self.bin = pj(program_path, "GMMPatternTracker")
        self.activations = Activations(
            pj(ACTIVATIONS_PATH, "sample.gmm_pattern_tracker.npz"))
        self.result = np.loadtxt(
            pj(DETECTIONS_PATH, "sample.gmm_pattern_tracker.txt"))
        self.downbeat_result = self.result[self.result[:, 1] == 1][:, 0]

    def test_help(self):
        self.assertTrue(run_help(self.bin))

    def test_binary(self):
        # save activations as binary file
        run_program([self.bin, '--save', 'single', sample_file, '-o', tmp_act])
        act = Activations(tmp_act)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        self.assertEqual(act.fps, self.activations.fps)
        # reload from file
        run_program([self.bin, '--load', 'single', tmp_act, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_txt(self):
        # save activations as txt file
        run_program([self.bin, '--save', '--sep', ' ', 'single', sample_file,
                     '-o', tmp_act])
        act = Activations(tmp_act, sep=' ', fps=50)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        # reload from file
        run_program([self.bin, '--load', '--sep', ' ', 'single', tmp_act,
                     '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_run(self):
        run_program([self.bin, 'single', sample_file, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_run_downbeats(self):
        run_program([self.bin, '--downbeats', 'single', sample_file,
                     '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.downbeat_result, atol=1e-5))


class TestLogFiltSpecFluxProgram(unittest.TestCase):
    def setUp(self):
        self.bin = pj(program_path, "LogFiltSpecFlux")
        self.activations = Activations(
            pj(ACTIVATIONS_PATH, "sample.log_filt_spec_flux.npz"))
        self.result = np.loadtxt(
            pj(DETECTIONS_PATH, "sample.log_filt_spec_flux.txt"))

    def test_help(self):
        self.assertTrue(run_help(self.bin))

    def test_binary(self):
        # save activations as binary file
        run_program([self.bin, '--save', 'single', sample_file, '-o', tmp_act])
        act = Activations(tmp_act)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        self.assertEqual(act.fps, self.activations.fps)
        # reload from file
        run_program([self.bin, '--load', 'single', tmp_act, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_txt(self):
        # save activations as txt file
        run_program([self.bin, '--save', '--sep', ' ', 'single', sample_file,
                     '-o', tmp_act])
        act = Activations(tmp_act, sep=' ', fps=100)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        # reload from file
        run_program([self.bin, '--load', '--sep', ' ', 'single', tmp_act,
                     '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_run(self):
        run_program([self.bin, 'single', sample_file, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))


class TestMMBeatTrackerProgram(unittest.TestCase):
    def setUp(self):
        self.bin = pj(program_path, "MMBeatTracker")
        self.activations = Activations(
            pj(ACTIVATIONS_PATH, "sample.beats_blstm_mm.npz"))
        self.result = np.loadtxt(
            pj(DETECTIONS_PATH, "sample.mm_beat_tracker.txt"))

    def test_help(self):
        self.assertTrue(run_help(self.bin))

    def test_binary(self):
        # save activations as binary file
        run_program([self.bin, '--save', 'single', sample_file, '-o', tmp_act])
        act = Activations(tmp_act)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        self.assertEqual(act.fps, self.activations.fps)
        # reload from file
        run_program([self.bin, '--load', 'single', tmp_act, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_txt(self):
        # save activations as txt file
        run_program([self.bin, '--save', '--sep', ' ', 'single', sample_file,
                     '-o', tmp_act])
        act = Activations(tmp_act, sep=' ', fps=100)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        # reload from file
        run_program([self.bin, '--load', '--sep', ' ', 'single', tmp_act,
                     '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_run(self):
        run_program([self.bin, 'single', sample_file, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))


class TestOnsetDetectorProgram(unittest.TestCase):
    def setUp(self):
        self.bin = pj(program_path, "OnsetDetector")
        self.activations = Activations(
            pj(ACTIVATIONS_PATH, "sample.onsets_brnn.npz"))
        self.result = np.loadtxt(
            pj(DETECTIONS_PATH, "sample.onset_detector.txt"))

    def test_help(self):
        self.assertTrue(run_help(self.bin))

    def test_binary(self):
        # save activations as binary file
        run_program([self.bin, '--save', 'single', sample_file, '-o', tmp_act])
        act = Activations(tmp_act)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        self.assertEqual(act.fps, self.activations.fps)
        # reload from file
        run_program([self.bin, '--load', 'single', tmp_act, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_txt(self):
        # save activations as txt file
        run_program([self.bin, '--save', '--sep', ' ', 'single', sample_file,
                     '-o', tmp_act])
        act = Activations(tmp_act, sep=' ', fps=100)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        # reload from file
        run_program([self.bin, '--load', '--sep', ' ', 'single', tmp_act,
                     '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_run(self):
        run_program([self.bin, 'single', sample_file, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))


class TestOnsetDetectorLLProgram(unittest.TestCase):
    def setUp(self):
        self.bin = pj(program_path, "OnsetDetectorLL")
        self.activations = Activations(
            pj(ACTIVATIONS_PATH, "sample.onsets_rnn.npz"))
        self.result = np.loadtxt(
            pj(DETECTIONS_PATH, "sample.onset_detector_ll.txt"))

    def test_help(self):
        self.assertTrue(run_help(self.bin))

    def test_binary(self):
        # save activations as binary file
        run_program([self.bin, '--save', 'single', sample_file, '-o', tmp_act])
        act = Activations(tmp_act)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        self.assertEqual(act.fps, self.activations.fps)
        # reload from file
        run_program([self.bin, '--load', 'single', tmp_act, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))
        # reload from file
        run_program([self.bin, '--load', 'single', '--online', tmp_act, '-o',
                     tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_txt(self):
        # save activations as txt file
        run_program([self.bin, '--save', '--sep', ' ', 'single', sample_file,
                     '-o', tmp_act])
        act = Activations(tmp_act, sep=' ', fps=100)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        # reload from file
        run_program([self.bin, '--load', '--sep', ' ', 'single', tmp_act,
                     '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_run(self):
        run_program([self.bin, 'single', sample_file, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_online(self):
        run_program([self.bin, 'single', sample_file, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result))
        run_program([self.bin, 'single', '--online', sample_file, '-o',
                     tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result))
        run_program([self.bin, 'online', sample_file, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result))


class TestPianoTranscriptorProgram(unittest.TestCase):
    def setUp(self):
        self.bin = pj(program_path, "PianoTranscriptor")
        self.activations = Activations(
            pj(ACTIVATIONS_PATH, "stereo_sample.notes_brnn.npz"))
        self.result = np.loadtxt(
            pj(DETECTIONS_PATH, "stereo_sample.piano_transcriptor.txt"))

    def test_help(self):
        self.assertTrue(run_help(self.bin))

    def test_binary(self):
        # save activations as binary file
        run_program([self.bin, '--save', 'single', stereo_sample_file,
                     '-o', tmp_act])
        act = Activations(tmp_act)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        self.assertEqual(act.fps, self.activations.fps)
        # reload from file
        run_program([self.bin, '--load', 'single', tmp_act, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_txt(self):
        # save activations as txt file
        run_program([self.bin, '--save', '--sep', ' ', 'single',
                     stereo_sample_file, '-o', tmp_act])
        act = Activations(tmp_act, sep=' ', fps=100)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        # reload from file
        run_program([self.bin, '--load', '--sep', ' ', 'single', tmp_act,
                     '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_run(self):
        run_program([self.bin, 'single', stereo_sample_file, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))


class TestSpectralOnsetDetectionProgram(unittest.TestCase):
    def setUp(self):
        self.bin = pj(program_path, "SpectralOnsetDetection")
        self.activations = Activations(
            pj(ACTIVATIONS_PATH, "sample.spectral_flux.npz"))
        self.result = np.loadtxt(
            pj(DETECTIONS_PATH, "sample.spectral_flux.txt"))

    def test_help(self):
        self.assertTrue(run_help(self.bin))

    def test_binary(self):
        # save activations as binary file
        run_program([self.bin, '--save', 'single', sample_file, '-o', tmp_act])
        act = Activations(tmp_act)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        self.assertEqual(act.fps, self.activations.fps)
        # reload from file
        run_program([self.bin, '--load', 'single', tmp_act, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_txt(self):
        # save activations as txt file
        run_program([self.bin, '--save', '--sep', ' ', 'single', sample_file,
                     '-o', tmp_act])
        act = Activations(tmp_act, sep=' ', fps=100)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        # reload from file
        run_program([self.bin, '--load', '--sep', ' ', 'single', tmp_act,
                     '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_run(self):
        run_program([self.bin, 'single', sample_file, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))


class TestSuperFluxProgram(unittest.TestCase):
    def setUp(self):
        self.bin = pj(program_path, "SuperFlux")
        self.activations = Activations(
            pj(ACTIVATIONS_PATH, "sample.super_flux.npz"))
        self.result = np.loadtxt(pj(DETECTIONS_PATH, "sample.super_flux.txt"))

    def test_help(self):
        self.assertTrue(run_help(self.bin))

    def test_binary(self):
        # save activations as binary file
        run_program([self.bin, '--save', 'single', sample_file, '-o', tmp_act])
        act = Activations(tmp_act)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        self.assertEqual(act.fps, self.activations.fps)
        # reload from file
        run_program([self.bin, '--load', 'single', tmp_act, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_txt(self):
        # save activations as txt file
        run_program([self.bin, '--save', '--sep', ' ', 'single', sample_file,
                     '-o', tmp_act])
        act = Activations(tmp_act, sep=' ', fps=200)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        # reload from file
        run_program([self.bin, '--load', '--sep', ' ', 'single', tmp_act,
                     '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_run(self):
        run_program([self.bin, 'single', sample_file, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))


class TestSuperFluxNNProgram(unittest.TestCase):
    def setUp(self):
        self.bin = pj(program_path, "SuperFluxNN")
        self.activations = Activations(
            pj(ACTIVATIONS_PATH, "sample.super_flux_nn.npz"))
        self.result = np.loadtxt(
            pj(DETECTIONS_PATH, "sample.super_flux_nn.txt"))

    def test_help(self):
        self.assertTrue(run_help(self.bin))

    def test_binary(self):
        # save activations as binary file
        run_program([self.bin, '--save', 'single', sample_file, '-o', tmp_act])
        act = Activations(tmp_act)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        self.assertEqual(act.fps, self.activations.fps)
        # reload from file
        run_program([self.bin, '--load', 'single', tmp_act, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_txt(self):
        # save activations as txt file
        run_program([self.bin, '--save', '--sep', ' ', 'single', sample_file,
                     '-o', tmp_act])
        act = Activations(tmp_act, sep=' ', fps=100)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        # reload from file
        run_program([self.bin, '--load', '--sep', ' ', 'single', tmp_act,
                     '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_run(self):
        run_program([self.bin, 'single', sample_file, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))


class TestTempoDetectorProgram(unittest.TestCase):
    def setUp(self):
        self.bin = pj(program_path, "TempoDetector")
        self.activations = Activations(
            pj(ACTIVATIONS_PATH, "sample.beats_blstm.npz"))
        self.result = np.loadtxt(
            pj(DETECTIONS_PATH, "sample.tempo_detector.txt"))
        self.online_results = np.array([176.47, 88.24, 0.58])

    def test_help(self):
        self.assertTrue(run_help(self.bin))

    def test_binary(self):
        # save activations as binary file
        run_program([self.bin, '--save', 'single', sample_file, '-o', tmp_act])
        act = Activations(tmp_act)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        self.assertEqual(act.fps, self.activations.fps)
        # reload from file
        run_program([self.bin, '--load', 'single', tmp_act, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_txt(self):
        # save activations as txt file
        run_program(
            [self.bin, '--save', '--sep', ' ', 'single', sample_file, '-o',
             tmp_act])
        act = Activations(tmp_act, sep=' ', fps=100)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        # reload from file
        run_program([self.bin, '--load', '--sep', ' ', 'single', tmp_act,
                     '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_run(self):
        run_program([self.bin, 'single', sample_file, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_online(self):
        run_program([self.bin, 'online', sample_file, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result[-1], self.online_results))
        run_program([self.bin, 'single', '--online', sample_file, '-o',
                     tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.online_results))


# clean up
def teardown():
    os.unlink(tmp_act)
    os.unlink(tmp_result)
