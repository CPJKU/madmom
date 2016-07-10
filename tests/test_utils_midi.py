# encoding: utf-8
# pylint: skip-file
"""
This file contains test functions for the madmom.utils.midi module.

"""

from __future__ import absolute_import, division, print_function

import unittest
from os.path import join as pj

from madmom.utils.midi import *

from . import ANNOTATIONS_PATH


class TestMIDIFileClass(unittest.TestCase):

    def test_notes(self):
        # poor man's test to make sure we can read the MIDI file
        midi = MIDIFile.from_file(pj(ANNOTATIONS_PATH, 'stereo_sample.mid'))
        notes = np.loadtxt(pj(ANNOTATIONS_PATH, 'stereo_sample.notes'))
        self.assertTrue(np.allclose(midi.notes(), notes, atol=1e-3))

        midi = MIDIFile.from_file(pj(ANNOTATIONS_PATH, 'piano_sample.mid'))
        notes = np.loadtxt(pj(ANNOTATIONS_PATH, 'piano_sample.notes_in_beats'))
        self.assertTrue(np.allclose(midi.notes(note_time_unit='b'), notes))
