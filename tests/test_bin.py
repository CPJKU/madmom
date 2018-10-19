# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the programs in the /bin directory.

"""

from __future__ import absolute_import, division, print_function

import imp
import os
import sys
import tempfile
import unittest
from os.path import join as pj

try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

import numpy as np

from madmom.features import Activations
from madmom.evaluation.key import load_key
from madmom.io import load_chords, midi

from . import AUDIO_PATH, ACTIVATIONS_PATH, ANNOTATIONS_PATH, DETECTIONS_PATH

tmp_act = tempfile.NamedTemporaryFile(delete=False).name
tmp_result = tempfile.NamedTemporaryFile(delete=False).name
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


def run_batch(program, infiles, outdir=None, args=None):
    argv = [program]
    if args:
        argv.extend(args)
    argv.extend(['batch', '-j', '1'])
    argv.extend(infiles)
    if outdir:
        argv.extend(['-o', outdir])
    run_program(argv)


def run_single(program, infile, outfile, online=False, args=None):
    argv = [program]
    if args:
        argv.extend(args)
    argv.extend(['single', '-j', '1'])
    if online:
        argv.append('--online')
    argv.extend([infile, '-o', outfile])
    run_program(argv)


def run_online(program, infile, outfile):
    argv = [program, 'online', '-j', '1', infile, '-o', outfile]
    run_program(argv)


def run_save(program, infile, outfile, args=None):
    argv = [program, '--save']
    if args:
        argv.extend(args)
    argv.extend(['single', '-j', '1', infile, '-o', outfile])
    run_program(argv)


def run_load(program, infile, outfile, online=False, args=None):
    argv = [program, '--load']
    if args:
        argv.extend(args)
    argv.extend(['single', '-j', '1'])
    if online:
        argv.append('--online')
    argv.extend([infile, '-o', outfile])
    run_program(argv)


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


class TestBarTrackerProgram(unittest.TestCase):
    def setUp(self):
        self.bin = pj(program_path, "BarTracker")
        self.act = np.load(pj(ACTIVATIONS_PATH, 'sample.bar_tracker.npz'))
        self.beats = [[0.091, 1], [0.8, 2], [1.481, 3], [2.148, 1]]
        self.downbeats = [0.091, 2.148]

    def test_run(self):
        # with beat annotations
        run_single(self.bin, sample_file, tmp_result,
                   args=['--beats', sample_beats])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.beats))
        # check 22050 Hz sample rate
        run_single(self.bin, sample_file_22050, tmp_result,
                   args=['--beats', sample_beats])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.beats))

    # TODO: investigate why this fails on Windows
    @unittest.skipIf(sys.platform.startswith('win'), "fails on Windows")
    def test_batch(self):
        # run using beat detections in batch mode
        run_batch(self.bin, [sample_file, sample_beats],
                  args=['--beats_suffix', '.beats'])
        # detections got stored into the audio folder, thus remove that file
        result_file = pj(AUDIO_PATH, 'sample.beats.txt')
        result = np.loadtxt(result_file)
        os.unlink(result_file)
        self.assertTrue(np.allclose(result, self.beats))

    def test_output_downbeats(self):
        run_single(self.bin, sample_file, tmp_result,
                   args=['--down', '--beats', sample_beats])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.downbeats))

    def test_save_load_activations(self):
        # save RNN bar activations
        run_save(self.bin, sample_file, tmp_act,
                 args=['--beats', sample_beats])
        act = np.load(tmp_act)['activations']
        self.assertTrue(np.allclose(act, self.act['activations'], rtol=1e-3,
                                    equal_nan=True))
        # load RNN bar activations
        run_load(self.bin, tmp_act, tmp_result)
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.beats))


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
        run_save(self.bin, sample_file, tmp_act)
        act = Activations(tmp_act)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        self.assertEqual(act.fps, self.activations.fps)
        # reload from file
        run_load(self.bin, tmp_act, tmp_result)
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_txt(self):
        # save activations as txt file
        run_save(self.bin, sample_file, tmp_act, args=['--sep', ' '])
        act = Activations(tmp_act, sep=' ', fps=100)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        # reload from file
        run_load(self.bin, tmp_act, tmp_result, args=['--sep', ' '])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_run(self):
        run_single(self.bin, sample_file, tmp_result)
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
        run_save(self.bin, sample_file, tmp_act)
        act = Activations(tmp_act)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        self.assertEqual(act.fps, self.activations.fps)
        # reload from file
        run_load(self.bin, tmp_act, tmp_result)
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_txt(self):
        # save activations as txt file
        run_save(self.bin, sample_file, tmp_act, args=['--sep', ' '])
        act = Activations(tmp_act, sep=' ', fps=100)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        # reload from file
        run_load(self.bin, tmp_act, tmp_result, args=['--sep', ' '])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_run(self):
        run_single(self.bin, sample_file, tmp_result)
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
                                          self.activations, self.results):
            # save activations as binary file
            run_save(self.bin, sf, tmp_act)
            act = Activations(tmp_act)
            self.assertTrue(np.allclose(act, true_act, atol=1e-5))
            self.assertEqual(act.fps, true_act.fps)
            # reload from file
            run_load(self.bin, tmp_act, tmp_result)
            self._check_results(load_chords(tmp_result), true_res)

    def test_txt(self):
        for sf, true_act, true_res in zip([sample_file, sample2_file],
                                          self.activations, self.results):
            # save activations as txt file
            run_save(self.bin, sf, tmp_act, args=['--sep', ' '])
            act = Activations(tmp_act, sep=' ', fps=100)
            self.assertTrue(np.allclose(act, true_act, atol=1e-5))
            # reload from file
            run_load(self.bin, tmp_act, tmp_result, args=['--sep', ' '])
            self._check_results(load_chords(tmp_result), true_res)

    def test_run(self):
        for sf, true_res in zip([sample_file, sample2_file], self.results):
            run_single(self.bin, sf, tmp_result)
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
        run_save(self.bin, sample_file, tmp_act)
        act = Activations(tmp_act)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        self.assertEqual(act.fps, self.activations.fps)
        # reload from file
        run_load(self.bin, tmp_act, tmp_result)
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_txt(self):
        # save activations as txt file
        run_save(self.bin, sample_file, tmp_act, args=['--sep', ' '])
        act = Activations(tmp_act, sep=' ', fps=100)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        # reload from file
        run_load(self.bin, tmp_act, tmp_result, args=['--sep', ' '])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_run(self):
        run_single(self.bin, sample_file, tmp_result)
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
        run_save(self.bin, sample_file, tmp_act)
        act = Activations(tmp_act)
        self.assertTrue(np.allclose(act, self.activations))
        self.assertEqual(act.fps, self.activations.fps)
        # reload from file
        run_load(self.bin, tmp_act, tmp_result)
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result))

    def test_txt(self):
        # save activations as txt file
        run_save(self.bin, sample_file, tmp_act, args=['--sep', ' '])
        act = Activations(tmp_act, sep=' ', fps=100)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        # reload from file
        run_load(self.bin, tmp_act, tmp_result, args=['--sep', ' '])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result))

    def test_run(self):
        run_single(self.bin, sample_file, tmp_result)
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
        run_save(self.bin, sample_file, tmp_act)
        act = Activations(tmp_act)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        self.assertEqual(act.fps, self.activations.fps)
        # reload from file
        run_load(self.bin, tmp_act, tmp_result)
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_txt(self):
        # save activations as txt file
        run_save(self.bin, sample_file, tmp_act, args=['--sep', ' '])
        act = Activations(tmp_act, sep=' ', fps=100)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        # reload from file
        run_load(self.bin, tmp_act, tmp_result, args=['--sep', ' '])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_run(self):
        run_single(self.bin, sample_file, tmp_result)
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
        run_save(self.bin, sample_file, tmp_act)
        act = Activations(tmp_act)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        self.assertEqual(act.fps, self.activations.fps)
        # reload from file
        run_load(self.bin, tmp_act, tmp_result)
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_txt(self):
        # save activations as txt file
        run_save(self.bin, sample_file, tmp_act, args=['--sep', ' '])
        act = Activations(tmp_act, sep=' ', fps=100)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        # reload from file
        run_load(self.bin, tmp_act, tmp_result, args=['--sep', ' '])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_run(self):
        run_single(self.bin, sample_file, tmp_result)
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_online(self):
        run_online(self.bin, sample_file, tmp_result)
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.online_results))
        run_single(self.bin, sample_file, tmp_result, online=True)
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
        run_save(self.bin, sample_file, tmp_act)
        act = Activations(tmp_act)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        self.assertEqual(act.fps, self.activations.fps)
        # reload from file
        run_load(self.bin, tmp_act, tmp_result)
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_txt(self):
        # save activations as txt file
        run_save(self.bin, sample_file, tmp_act, args=['--sep', ' '])
        act = Activations(tmp_act, sep=' ', fps=100)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        # reload from file
        run_load(self.bin, tmp_act, tmp_result, args=['--sep', ' '])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_run(self):
        run_single(self.bin, sample_file, tmp_result)
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_run_downbeats(self):
        run_single(self.bin, sample_file, tmp_result, args=['--downbeats'])
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
                                          self.activations, self.results):
            # save activations as binary file
            run_save(self.bin, sf, tmp_act)
            act = Activations(tmp_act)
            self.assertTrue(np.allclose(act, true_act, atol=1e-5))
            self.assertEqual(act.fps, true_act.fps)
            # reload from file
            run_load(self.bin, tmp_act, tmp_result)
            self._check_results(load_chords(tmp_result), true_res)

    def test_txt(self):
        for sf, true_act, true_res in zip([sample_file, sample2_file],
                                          self.activations, self.results):
            # save activations as txt file
            run_save(self.bin, sf, tmp_act, args=['--sep', ' '])
            act = Activations(tmp_act, sep=' ', fps=100)
            self.assertTrue(np.allclose(act, true_act, atol=1e-5))
            # reload from file
            run_load(self.bin, tmp_act, tmp_result, args=['--sep', ' '])
            self._check_results(load_chords(tmp_result), true_res)

    def test_run(self):
        for sf, true_res in zip([sample_file, sample2_file], self.results):
            run_single(self.bin, sf, tmp_result)
            self._check_results(load_chords(tmp_result), true_res)


class TestKeyRecognitionProgram(unittest.TestCase):
    def setUp(self):
        self.bin = pj(program_path, 'KeyRecognition')
        self.activations = [
            Activations(pj(ACTIVATIONS_PATH, af))
            for af in ['sample.key_cnn.npz', 'sample2.key_cnn.npz']
        ]
        self.results = [
            load_key(pj(DETECTIONS_PATH, df))
            for df in ['sample.key_recognition.txt',
                       'sample2.key_recognition.txt']
        ]

    def test_help(self):
        self.assertTrue(run_help(self.bin))

    def test_binary(self):
        for sf, true_act, true_res in zip([sample_file, sample2_file],
                                          self.activations, self.results):
            # save activations as binary file
            run_save(self.bin, sf, tmp_act)
            act = Activations(tmp_act)
            self.assertTrue(np.allclose(act, true_act, atol=1e-5))
            self.assertEqual(act.fps, true_act.fps)
            # reload from file
            run_load(self.bin, tmp_act, tmp_result)
            self.assertEqual(load_key(tmp_result), true_res)

    def test_txt(self):
        for sf, true_act, true_res in zip([sample_file, sample2_file],
                                          self.activations, self.results):
            # save activations as txt file
            run_save(self.bin, sf, tmp_act, args=['--sep', ' '])
            act = Activations(tmp_act, sep=' ', fps=0)
            self.assertTrue(np.allclose(act, true_act, atol=1e-5))
            # reload from file
            run_load(self.bin, tmp_act, tmp_result, args=['--sep', ' '])
            self.assertEqual(load_key(tmp_result), true_res)

    def test_run(self):
        for sf, true_res in zip([sample_file, sample2_file], self.results):
            run_single(self.bin, sf, tmp_result)
            self.assertEqual(load_key(tmp_result), true_res)


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
        run_save(self.bin, sample_file, tmp_act)
        act = Activations(tmp_act)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        self.assertEqual(act.fps, self.activations.fps)
        # reload from file
        run_load(self.bin, tmp_act, tmp_result)
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_txt(self):
        # save activations as txt file
        run_save(self.bin, sample_file, tmp_act, args=['--sep', ' '])
        act = Activations(tmp_act, sep=' ', fps=50)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        # reload from file
        run_load(self.bin, tmp_act, tmp_result, args=['--sep', ' '])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_run(self):
        run_single(self.bin, sample_file, tmp_result)
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_run_downbeats(self):
        run_single(self.bin, sample_file, tmp_result, args=['--downbeats'])
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
        run_save(self.bin, sample_file, tmp_act)
        act = Activations(tmp_act)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        self.assertEqual(act.fps, self.activations.fps)
        # reload from file
        run_load(self.bin, tmp_act, tmp_result)
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_txt(self):
        # save activations as txt file
        run_save(self.bin, sample_file, tmp_act, args=['--sep', ' '])
        act = Activations(tmp_act, sep=' ', fps=100)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        # reload from file
        run_load(self.bin, tmp_act, tmp_result, args=['--sep', ' '])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_run(self):
        run_single(self.bin, sample_file, tmp_result)
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
        run_save(self.bin, sample_file, tmp_act)
        act = Activations(tmp_act)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        self.assertEqual(act.fps, self.activations.fps)
        # reload from file
        run_load(self.bin, tmp_act, tmp_result)
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_txt(self):
        # save activations as txt file
        run_save(self.bin, sample_file, tmp_act, args=['--sep', ' '])
        act = Activations(tmp_act, sep=' ', fps=100)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        # reload from file
        run_load(self.bin, tmp_act, tmp_result, args=['--sep', ' '])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_run(self):
        run_single(self.bin, sample_file, tmp_result)
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
        run_save(self.bin, sample_file, tmp_act)
        act = Activations(tmp_act)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        self.assertEqual(act.fps, self.activations.fps)
        # reload from file
        run_load(self.bin, tmp_act, tmp_result)
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_txt(self):
        # save activations as txt file
        run_save(self.bin, sample_file, tmp_act, args=['--sep', ' '])
        act = Activations(tmp_act, sep=' ', fps=100)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        # reload from file
        run_load(self.bin, tmp_act, tmp_result, args=['--sep', ' '])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_run(self):
        run_single(self.bin, sample_file, tmp_result)
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
        run_save(self.bin, sample_file, tmp_act)
        act = Activations(tmp_act)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        self.assertEqual(act.fps, self.activations.fps)
        # reload from file
        run_load(self.bin, tmp_act, tmp_result)
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))
        # reload from file
        run_load(self.bin, tmp_act, tmp_result, online=True)
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_txt(self):
        # save activations as txt file
        run_save(self.bin, sample_file, tmp_act, args=['--sep', ' '])
        act = Activations(tmp_act, sep=' ', fps=100)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        # reload from file
        run_load(self.bin, tmp_act, tmp_result, args=['--sep', ' '])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_run(self):
        run_single(self.bin, sample_file, tmp_result)
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_online(self):
        run_single(self.bin, sample_file, tmp_result)
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result))
        run_single(self.bin, sample_file, tmp_result, online=True)
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result))
        run_online(self.bin, sample_file, tmp_result)
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
        run_save(self.bin, stereo_sample_file, tmp_act)
        act = Activations(tmp_act)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        self.assertEqual(act.fps, self.activations.fps)
        # reload from file
        run_load(self.bin, tmp_act, tmp_result)
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_txt(self):
        # save activations as txt file
        run_save(self.bin, stereo_sample_file, tmp_act, args=['--sep', ' '])
        act = Activations(tmp_act, sep=' ', fps=100)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        # reload from file
        run_load(self.bin, tmp_act, tmp_result, args=['--sep', ' '])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_run(self):
        run_single(self.bin, stereo_sample_file, tmp_result)
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_midi(self):
        run_single(self.bin, stereo_sample_file, tmp_result, args=['--midi'])
        result = midi.MIDIFile(tmp_result).notes
        self.assertTrue(np.allclose(result[:, :2], self.result, atol=1e-3))

    def test_mirex(self):
        run_single(self.bin, stereo_sample_file, tmp_result, args=['--mirex'])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result[:, 0], self.result[:, 0]))
        self.assertTrue(np.allclose(result[:, 2], [523.3, 87.3, 698.5, 622.3]))


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
        run_save(self.bin, sample_file, tmp_act)
        act = Activations(tmp_act)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        self.assertEqual(act.fps, self.activations.fps)
        # reload from file
        run_load(self.bin, tmp_act, tmp_result)
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_txt(self):
        # save activations as txt file
        run_save(self.bin, sample_file, tmp_act, args=['--sep', ' '])
        act = Activations(tmp_act, sep=' ', fps=100)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        # reload from file
        run_load(self.bin, tmp_act, tmp_result, args=['--sep', ' '])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_run(self):
        run_single(self.bin, sample_file, tmp_result)
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
        run_save(self.bin, sample_file, tmp_act)
        act = Activations(tmp_act)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        self.assertEqual(act.fps, self.activations.fps)
        # reload from file
        run_load(self.bin, tmp_act, tmp_result)
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_txt(self):
        # save activations as txt file
        run_save(self.bin, sample_file, tmp_act, args=['--sep', ' '])
        act = Activations(tmp_act, sep=' ', fps=200)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        # reload from file
        run_load(self.bin, tmp_act, tmp_result, args=['--sep', ' '])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_run(self):
        run_single(self.bin, sample_file, tmp_result)
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
        run_save(self.bin, sample_file, tmp_act)
        act = Activations(tmp_act)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        self.assertEqual(act.fps, self.activations.fps)
        # reload from file
        run_load(self.bin, tmp_act, tmp_result)
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_txt(self):
        # save activations as txt file
        run_save(self.bin, sample_file, tmp_act, args=['--sep', ' '])
        act = Activations(tmp_act, sep=' ', fps=100)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        # reload from file
        run_load(self.bin, tmp_act, tmp_result, args=['--sep', ' '])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_run(self):
        run_single(self.bin, sample_file, tmp_result)
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
        run_save(self.bin, sample_file, tmp_act)
        act = Activations(tmp_act)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        self.assertEqual(act.fps, self.activations.fps)
        # reload from file
        run_load(self.bin, tmp_act, tmp_result)
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_txt(self):
        # save activations as txt file
        run_save(self.bin, sample_file, tmp_act, args=['--sep', ' '])
        act = Activations(tmp_act, sep=' ', fps=100)
        self.assertTrue(np.allclose(act, self.activations, atol=1e-5))
        # reload from file
        run_load(self.bin, tmp_act, tmp_result, args=['--sep', ' '])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_run(self):
        run_single(self.bin, sample_file, tmp_result)
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.result, atol=1e-5))

    def test_online(self):
        run_online(self.bin, sample_file, tmp_result)
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result[-1], self.online_results))
        run_single(self.bin, sample_file, tmp_result, online=True)
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, self.online_results))

    def test_mirex(self):
        run_single(self.bin, sample_file, tmp_result, args=['--mirex'])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(result, [117.65, 176.47, 0.27]))

    def test_all_tempi(self):
        run_single(self.bin, sample_file, tmp_result, args=['--all'])
        result = np.loadtxt(tmp_result)
        self.assertTrue(np.allclose(
            result, [[176.47, 0.475], [117.65, 0.177], [240.00, 0.154],
                     [68.97, 0.099], [82.19, 0.096]]))


# clean up
def teardown_module():
    os.unlink(tmp_act)
    os.unlink(tmp_result)
