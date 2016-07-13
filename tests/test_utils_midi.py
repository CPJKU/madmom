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


class TestEventsClass(unittest.TestCase):

    def test_sort_events(self):
        e1 = NoteOnEvent(tick=100, pitch=50, channel=1)
        e2 = NoteOffEvent(tick=300, pitch=50, channel=1)
        e3 = NoteOffEvent(tick=200, pitch=50, channel=1)
        self.assertTrue(sorted([e1, e2, e3]) == [e1, e3, e2])
        # TODO: add test case if note on and note off occur at the same tick
        #       note on must come first then


class TestMIDIFileClass(unittest.TestCase):

    def test_notes(self):
        # read a MIDI file
        midi = MIDIFile.from_file(pj(ANNOTATIONS_PATH, 'stereo_sample.mid'))
        notes = np.loadtxt(pj(ANNOTATIONS_PATH, 'stereo_sample.notes'))
        notes_ = midi.notes()[:, :4]
        self.assertTrue(np.allclose(notes, notes_, atol=1e-3))

    def test_recreate_midi(self):
        notes = np.loadtxt(pj(ANNOTATIONS_PATH, 'stereo_sample.notes'))
        # create a MIDI file from the notes
        midi = MIDIFile.from_notes(notes)
        notes_ = midi.notes()[:, :4]
        self.assertTrue(np.allclose(notes, notes_, atol=1e-3))
        # write to a temporary file
        tmp_file = tempfile.NamedTemporaryFile().name
        midi.write(tmp_file)
        # FIXME: re-read this file and compare the notes
        tmp_midi = MIDIFile.from_file(tmp_file)
        notes_ = tmp_midi.notes()[:, :4]
        self.assertTrue(np.allclose(notes, notes_, atol=1e-3))

    def test_notes_in_beats(self):
        # read a MIDI file
        midi = MIDIFile.from_file(pj(ANNOTATIONS_PATH, 'piano_sample.mid'))
        notes = np.loadtxt(pj(ANNOTATIONS_PATH, 'piano_sample.notes_in_beats'))
        notes_ = midi.notes(note_time_unit='b')[:, :4]
        self.assertTrue(np.allclose(notes, notes_))
