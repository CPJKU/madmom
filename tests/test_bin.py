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

from . import AUDIO_PATH, ACTIVATIONS_PATH, DETECTIONS_PATH

tmp_act = tempfile.NamedTemporaryFile().name
tmp_result = tempfile.NamedTemporaryFile().name
sample_file = pj(AUDIO_PATH, 'sample.wav')
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
        self.bin = "%s/BeatDetector" % program_path
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
        self.bin = "%s/BeatTracker" % program_path
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


class TestComplexFluxProgram(unittest.TestCase):
    def setUp(self):
        self.bin = "%s/ComplexFlux" % program_path
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


class TestCRFBeatDetectorProgram(unittest.TestCase):
    def setUp(self):
        self.bin = "%s/CRFBeatDetector" % program_path
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
        self.bin = "%s/DBNBeatTracker" % program_path
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
        self.bin = "%s/DBNDownBeatTracker" % program_path
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


class TestGMMPatternTrackerProgram(unittest.TestCase):
    def setUp(self):
        self.bin = "%s/GMMPatternTracker" % program_path
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
        self.bin = "%s/LogFiltSpecFlux" % program_path
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
        self.bin = "%s/MMBeatTracker" % program_path
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
        self.bin = "%s/OnsetDetector" % program_path
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
        self.bin = "%s/OnsetDetectorLL" % program_path
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
        self.bin = "%s/PianoTranscriptor" % program_path
        self.activations = Activations(
            pj(ACTIVATIONS_PATH, "stereo_sample.notes_brnn.npz"))
        self.result = np.loadtxt(
            pj(DETECTIONS_PATH, "stereo_sample.pianot_ranscriptor.txt"))

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
        self.bin = "%s/SpectralOnsetDetection" % program_path
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
        self.bin = "%s/SuperFlux" % program_path
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
        self.bin = "%s/SuperFluxNN" % program_path
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
        self.bin = "%s/TempoDetector" % program_path
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
