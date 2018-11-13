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

from madmom.utils.midi import *

from . import ANNOTATIONS_PATH

tmp_file = tempfile.NamedTemporaryFile(delete=False).name


class TestEventsClass(unittest.TestCase):

    def setUp(self):
        self.e1 = NoteOnEvent(tick=100, pitch=50, velocity=60)
        self.e2 = NoteOffEvent(tick=300, pitch=50)
        self.e3 = NoteOffEvent(tick=200, pitch=50)
        self.e4 = NoteOnEvent(tick=300, pitch=50, velocity=60)

    def test_equality(self):
        self.assertEqual(
            self.e1, NoteOnEvent(tick=100, pitch=50, velocity=60))
        self.assertNotEqual(
            self.e1, NoteOnEvent(tick=101, pitch=50, velocity=60))
        self.assertNotEqual(
            self.e1, NoteOnEvent(tick=100, pitch=51, velocity=60))
        self.assertNotEqual(
            self.e1, NoteOnEvent(tick=100, pitch=50, velocity=61))
        self.assertNotEqual(
            self.e1, NoteOnEvent(tick=100, pitch=50, velocity=60, channel=1))
        self.assertNotEqual(
            self.e1, NoteOffEvent(tick=100, pitch=50))

    def test_comparison(self):
        self.assertTrue(self.e1 < self.e2)
        self.assertTrue(self.e1 < self.e3)
        self.assertTrue(self.e1 < self.e4)
        self.assertTrue(self.e4 < self.e2)

    def test_sort_events(self):
        events = sorted([self.e1, self.e2, self.e3, self.e4])
        self.assertTrue(events == [self.e1, self.e3, self.e4, self.e2])
        # MIDITrack should sort the events before writing the MIDI file
        track = MIDITrack([self.e1, self.e2, self.e3, self.e4])
        midi = MIDIFile(track)
        midi.write(tmp_file)
        events = MIDIFile.from_file(tmp_file).tracks[0].events
        self.assertTrue(events == [self.e1, self.e3, self.e4, self.e2])


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
        midi.write(tmp_file)
        # FIXME: re-read this file and compare the notes
        tmp_midi = MIDIFile.from_file(tmp_file)
        notes_ = tmp_midi.notes()[:, :4]
        self.assertTrue(np.allclose(notes, notes_, atol=1e-3))

    def test_notes_in_beats(self):
        # read a MIDI file
        midi = MIDIFile.from_file(pj(ANNOTATIONS_PATH, 'piano_sample.mid'))
        notes = np.loadtxt(pj(ANNOTATIONS_PATH, 'piano_sample.notes_in_beats'))
        notes_ = midi.notes(unit='b')[:, :4]
        self.assertTrue(np.allclose(notes, notes_))

    def test_multitrack(self):
        # read a multi-track MIDI file
        midi = MIDIFile.from_file(pj(ANNOTATIONS_PATH, 'multitrack.mid'))
        notes = midi.notes(unit='b')
        self.assertTrue(np.allclose(notes[:4], [[0, 60, 0.5, 90, 2],
                                                [0, 72, 2, 90, 1],
                                                [0.5, 67, 0.5, 90, 2],
                                                [1, 64, 0.5, 90, 2]],
                                    atol=1e-2))
        notes = midi.notes(unit='s')
        self.assertTrue(np.allclose(notes[:4],
                                    [[0, 60, 0.2272725, 90, 2],
                                     [0, 72, 0.90814303, 90, 1],
                                     [0.2272725, 67, 0.22632553, 90, 2],
                                     [0.45359803, 64, 0.22821947, 90, 2]]))


# clean up
def teardown_module():
    os.unlink(tmp_file)
