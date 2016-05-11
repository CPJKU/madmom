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

try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

import numpy as np

from madmom.features import Activations

from . import AUDIO_PATH, ACTIVATIONS_PATH, DETECTIONS_PATH

tmp_act = tempfile.NamedTemporaryFile().name
tmp_result = tempfile.NamedTemporaryFile().name
sample_file = '%s/sample.wav' % AUDIO_PATH
stereo_sample_file = '%s/stereo_sample.wav' % AUDIO_PATH
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
        self.bin = "%s/BeatDetector" % program_path
        self.activations = Activations(
            "%s/sample.beats_blstm_2013.npz" % ACTIVATIONS_PATH)
        self.result = np.loadtxt(
            "%s/sample.beat_detector.txt" % DETECTIONS_PATH)

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


class TestBeatTrackerProgram(unittest.TestCase):
    def setUp(self):
        self.bin = "%s/BeatTracker" % program_path
        self.activations = Activations(
            "%s/sample.beats_blstm_2013.npz" % ACTIVATIONS_PATH)
        self.result = np.loadtxt(
            "%s/sample.beat_tracker.txt" % DETECTIONS_PATH)

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
        run_program([self.bin, '--load', '--sep', ' ', 'single', tmp_act, '-o',
                     tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result))

    def test_run(self):
        run_program([self.bin, 'single', sample_file, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result))


class TestComplexFluxProgram(unittest.TestCase):
    def setUp(self):
        self.bin = "%s/ComplexFlux" % program_path
        self.activations = Activations(
            "%s/sample.complex_flux.npz" % ACTIVATIONS_PATH)
        self.result = np.loadtxt(
            "%s/sample.complex_flux.txt" % DETECTIONS_PATH)

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
        self.bin = "%s/CRFBeatDetector" % program_path
        self.activations = Activations(
            "%s/sample.beats_blstm_2013.npz" % ACTIVATIONS_PATH)
        self.result = np.loadtxt(
            "%s/sample.crf_beat_detector.txt" % DETECTIONS_PATH)

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


class TestDBNBeatTrackerProgram(unittest.TestCase):
    def setUp(self):
        self.bin = "%s/DBNBeatTracker" % program_path
        self.activations = Activations(
            "%s/sample.beats_blstm_2013.npz" % ACTIVATIONS_PATH)
        self.result = np.loadtxt(
            "%s/sample.dbn_beat_tracker.txt" % DETECTIONS_PATH)

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


class TestDBNDownBeatTrackerProgram(unittest.TestCase):
    def setUp(self):
        self.bin = "%s/DBNDownBeatTracker" % program_path
        self.activations = Activations(
            "%s/sample.downbeats_blstm_2016.npz" % ACTIVATIONS_PATH)
        self.result = np.loadtxt(
            "%s/sample.dbn_downbeat_tracker.txt" % DETECTIONS_PATH)
        self.downbeat_result = self.result[self.result[:, 1] == 1][:, 0]

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

    def test_run_downbeats(self):
        run_program([self.bin, '--downbeats', 'single', sample_file,
                     '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.downbeat_result))


class TestGMMPatternTrackerProgram(unittest.TestCase):
    def setUp(self):
        self.bin = "%s/GMMPatternTracker" % program_path
        self.activations = Activations(
            "%s/sample.gmm_pattern_tracker.npz" % ACTIVATIONS_PATH)
        self.result = np.loadtxt(
            "%s/sample.gmm_pattern_tracker.txt" % DETECTIONS_PATH)
        self.downbeat_result = self.result[self.result[:, 1] == 1][:, 0]

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
        act = Activations(tmp_act, sep=' ', fps=50)
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

    def test_run_downbeats(self):
        run_program([self.bin, '--downbeats', 'single', sample_file,
                     '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.downbeat_result))


class TestLogFiltSpecFluxProgram(unittest.TestCase):
    def setUp(self):
        self.bin = "%s/LogFiltSpecFlux" % program_path
        self.activations = Activations(
            "%s/sample.log_filt_spec_flux.npz" % ACTIVATIONS_PATH)
        self.result = np.loadtxt(
            "%s/sample.log_filt_spec_flux.txt" % DETECTIONS_PATH)

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


class TestMMBeatTrackerProgram(unittest.TestCase):
    def setUp(self):
        self.bin = "%s/MMBeatTracker" % program_path
        self.activations = Activations(
            "%s/sample.beats_blstm_mm_2013.npz" % ACTIVATIONS_PATH)
        self.result = np.loadtxt(
            "%s/sample.mm_beat_tracker.txt" % DETECTIONS_PATH)

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


class TestOnsetDetectorProgram(unittest.TestCase):
    def setUp(self):
        self.bin = "%s/OnsetDetector" % program_path
        self.activations = Activations(
            "%s/sample.onsets_brnn_2013.npz" % ACTIVATIONS_PATH)
        self.result = np.loadtxt(
            "%s/sample.onset_detector.txt" % DETECTIONS_PATH)

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


class TestOnsetDetectorLLProgram(unittest.TestCase):
    def setUp(self):
        self.bin = "%s/OnsetDetectorLL" % program_path
        self.activations = Activations(
            "%s/sample.onsets_rnn_2013.npz" % ACTIVATIONS_PATH)
        self.result = np.loadtxt(
            "%s/sample.onset_detector_ll.txt" % DETECTIONS_PATH)

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


class TestPianoTranscriptorProgram(unittest.TestCase):
    def setUp(self):
        self.bin = "%s/PianoTranscriptor" % program_path
        self.activations = Activations(
            "%s/stereo_sample.notes_brnn_2013.npz" % ACTIVATIONS_PATH)
        self.result = np.loadtxt(
            "%s/stereo_sample.pianot_ranscriptor.txt" % DETECTIONS_PATH)

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
        self.assertTrue(np.allclose(result, self.result))

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
        self.assertTrue(np.allclose(result, self.result))

    def test_run(self):
        run_program([self.bin, 'single', stereo_sample_file, '-o', tmp_result])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result))


class TestSpectralOnsetDetectionProgram(unittest.TestCase):
    def setUp(self):
        self.bin = "%s/SpectralOnsetDetection" % program_path
        self.activations = Activations(
            "%s/sample.spectral_flux.npz" % ACTIVATIONS_PATH)
        self.result = np.loadtxt(
            "%s/sample.spectral_flux.txt" % DETECTIONS_PATH)

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


class TestSuperFluxProgram(unittest.TestCase):
    def setUp(self):
        self.bin = "%s/SuperFlux" % program_path
        self.activations = Activations(
            "%s/sample.super_flux.npz" % ACTIVATIONS_PATH)
        self.result = np.loadtxt("%s/sample.super_flux.txt" % DETECTIONS_PATH)

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
        act = Activations(tmp_act, sep=' ', fps=200)
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


class TestSuperFluxNNProgram(unittest.TestCase):
    def setUp(self):
        self.bin = "%s/SuperFluxNN" % program_path
        self.activations = Activations(
            "%s/sample.super_flux_nn.npz" % ACTIVATIONS_PATH)
        self.result = np.loadtxt(
            "%s/sample.super_flux_nn.txt" % DETECTIONS_PATH)

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


class TestTempoDetectorProgram(unittest.TestCase):
    def setUp(self):
        self.bin = "%s/TempoDetector" % program_path
        self.activations = Activations(
            "%s/sample.beats_blstm_2013.npz" % ACTIVATIONS_PATH)
        self.result = np.loadtxt(
            "%s/sample.tempo_detector.txt" % DETECTIONS_PATH)

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
        run_program(
            [self.bin, '--save', '--sep', ' ', 'single', sample_file, '-o',
             tmp_act])
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
