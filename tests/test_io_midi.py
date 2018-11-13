# encoding: utf-8
# pylint: skip-file
"""
This file contains test functions for the madmom.utils.midi module.

"""

from __future__ import absolute_import, division, print_function

import os
import unittest
import tempfile
from os.path import join as pj

from madmom.io.midi import *

from . import ANNOTATIONS_PATH

tmp_file = tempfile.NamedTemporaryFile(delete=False).name


class TestMIDIFileClass(unittest.TestCase):

    def test_notes(self):
        # read a MIDI file
        midi = MIDIFile(pj(ANNOTATIONS_PATH, 'stereo_sample.mid'))
        notes = np.loadtxt(pj(ANNOTATIONS_PATH, 'stereo_sample.notes'))
        self.assertTrue(np.allclose(notes, midi.notes[:, :4], atol=1e-3))

    def test_recreate_midi(self):
        notes = np.loadtxt(pj(ANNOTATIONS_PATH, 'stereo_sample.notes'))
        # create a MIDI file from the notes
        midi = MIDIFile.from_notes(notes)
        self.assertTrue(np.allclose(notes, midi.notes[:, :4], atol=1e-3))
        # write to a temporary file
        midi.save(tmp_file)
        tmp_midi = MIDIFile(tmp_file)
        self.assertTrue(np.allclose(notes, tmp_midi.notes[:, :4], atol=1e-3))

    def test_notes_in_beats(self):
        # read a MIDI file
        midi = MIDIFile(pj(ANNOTATIONS_PATH, 'piano_sample.mid'))
        midi.unit = 'b'
        notes = np.loadtxt(pj(ANNOTATIONS_PATH, 'piano_sample.notes_in_beats'))
        self.assertTrue(np.allclose(notes, midi.notes[:, :4]))

    def test_multitrack(self):
        # read a multi-track MIDI file
        midi = MIDIFile(pj(ANNOTATIONS_PATH, 'multitrack.mid'))
        self.assertTrue(np.allclose(midi.notes[:4],
                                    [[0, 60, 0.2272725, 90, 2],
                                     [0, 72, 0.90814303, 90, 1],
                                     [0.2272725, 67, 0.22632553, 90, 2],
                                     [0.45359803, 64, 0.22821947, 90, 2]]))
        midi.unit = 'b'
        self.assertTrue(np.allclose(midi.notes[:4], [[0, 60, 0.5, 90, 2],
                                                     [0, 72, 2, 90, 1],
                                                     [0.5, 67, 0.5, 90, 2],
                                                     [1, 64, 0.5, 90, 2]],
                                    atol=1e-2))


# clean up
def teardown_module():
    os.unlink(tmp_file)
