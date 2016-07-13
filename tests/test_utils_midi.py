# encoding: utf-8
# pylint: skip-file
"""
This file contains test functions for the madmom.utils.midi module.

"""

from __future__ import absolute_import, division, print_function

import unittest
import tempfile
from os.path import join as pj

from madmom.utils.midi import *

from . import ANNOTATIONS_PATH


class TestMIDIFileClass(unittest.TestCase):

    def test_notes(self):
        # read a MIDI file
        midi = MIDIFile.from_file(pj(ANNOTATIONS_PATH, 'stereo_sample.mid'))
        notes = np.loadtxt(pj(ANNOTATIONS_PATH, 'stereo_sample.notes'))
        self.assertTrue(np.allclose(midi.notes(), notes, atol=1e-3))

    def test_recreate_midi(self):
        notes = np.loadtxt(pj(ANNOTATIONS_PATH, 'stereo_sample.notes'))
        # create a MIDI file from the notes
        midi = MIDIFile.from_notes(notes)
        self.assertTrue(np.allclose(midi.notes(), notes, atol=1e-3))
        # write to a temporary file
        tmp_file = tempfile.NamedTemporaryFile().name
        midi.write(tmp_file)
        # FIXME: re-read this file and compare the notes
        tmp_midi = MIDIFile.from_file(tmp_file)
        self.assertTrue(np.allclose(tmp_midi.notes(), notes, atol=1e-3))

    def test_notes_in_beats(self):
        # read a MIDI file
        midi = MIDIFile.from_file(pj(ANNOTATIONS_PATH, 'piano_sample.mid'))
        notes = np.loadtxt(pj(ANNOTATIONS_PATH, 'piano_sample.notes_in_beats'))
        self.assertTrue(np.allclose(midi.notes(note_time_unit='b'), notes))
