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

from . import AUDIO_PATH, ANNOTATIONS_PATH, ACTIVATIONS_PATH, DETECTIONS_PATH

tmp_act = tempfile.NamedTemporaryFile(delete=False).name
tmp_result = tempfile.NamedTemporaryFile(delete=False).name
tmp_dir = tempfile.mkdtemp()
sample_file = pj(AUDIO_PATH, 'sample.wav')
sample2_file = pj(AUDIO_PATH, 'sample2.wav')
sample_file_22050 = pj(AUDIO_PATH, 'sample_22050.wav')
sample_beats = pj(ANNOTATIONS_PATH, 'sample.beats')
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
        print(result)
        print(true_result)
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


class TestBarTrackerProgram(unittest.TestCase):
    def setUp(self):
        self.bin = os.path.join(program_path, 'BarTracker')
        self.result_beat_ann = np.array([[0.091, 1], [0.8, 2], [1.481, 3],
                                         [2.148, 1]])
        self.result_beat_det = np.array([[0.1, 1], [0.45, 2], [0.8, 3],
                                         [1.12, 1], [1.48, 2], [1.8, 3],
                                         [2.15, 1], [2.49, 2]])
        self.result_sample2 = [[0.140, 1], [0.900, 2], [1.650, 3], [2.380, 1],
                               [3.120, 2], [3.880, 3]]
        self.downbeat_result = self.result_beat_ann[
                                   self.result_beat_ann[:, 1] == 1][:, 0]

    def test_run(self):
        # run with beat tracker
        run_program([self.bin, 'single', sample_file, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result_beat_det))
        # check 22050 Hz sample rate
        run_program([self.bin, 'single', sample_file_22050, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result_beat_det))
        # check sample2
        run_program([self.bin, 'single', sample2_file, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result_sample2))
        # run using beat annotations
        run_program([self.bin, '--load_beats', 'batch',
                     '-o', tmp_dir, sample_file, sample_beats])
        result = np.loadtxt(pj(tmp_dir, 'sample.beats.txt'))
        self.assertTrue(np.allclose(result, self.result_beat_ann))

    def test_output_downbeats(self):
        run_program([self.bin, '--downbeats', '--load_beats', 'batch',
                     '-o', tmp_dir, sample_file, sample_beats])
        result = np.loadtxt(pj(tmp_dir, 'sample.beats.txt'))
        self.assertTrue(np.allclose(result, self.downbeat_result))

    def save_load_rnn_activations(self):
        # save RNN activations and beat times
        run_program([self.bin, '--save', 'single', '-o', tmp_act,
                     sample_file])
        data = np.load(tmp_act)
        acts = [0.37781784, 0.18954057, 0.11194395, 0.32766795, 0.2700946,
                0.18147217, 0.16246837]
        self.assertTrue(np.allclose(data['activations'], acts))
        beats = [0.1, 0.45, 0.8, 1.12, 1.48, 1.8, 2.15, 2.49]
        self.assertTrue(np.allclose(data['beats'], beats))
        # load RNN activations and beat times
        run_program([self.bin, '--load', 'single', '-o', tmp_result, tmp_act])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result_beat_det))


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


# clean up
def teardown():
    os.unlink(tmp_act)
    os.unlink(tmp_result)
