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
        midi = MIDIFile.from_notes(notes, tempo=120)
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

    def test_notes_in_ticks(self):
        # read a MIDI file
        midi = MIDIFile(pj(ANNOTATIONS_PATH, 'piano_sample.mid'))
        midi.unit = 't'
        note_times = [0, 240, 480, 720, 960, 1200, 1440, 1680, 1920, 2160,
                      2400, 2640, 2880, 3120, 3360, 3600, 3840, 3840, 3840,
                      4320, 4800, 4800, 5280]
        self.assertTrue(np.allclose(note_times, midi.notes[:, 0]))

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

    def test_sustain(self):
        # use an existing file as a start
        midi = MIDIFile(pj(ANNOTATIONS_PATH, 'stereo_sample_sustained.mid'))
        # obtain all sustain messages from that file
        sustain_msgs = []
        for msg in midi:
            if msg.type == 'control_change' and msg.control == 64:
                sustain_msgs.append(msg)
        # there's only a single sustain message
        self.assertTrue(len(sustain_msgs) == 4)
        # the logic adds a another sustain OFF message (i.e. value of 0)
        self.assertTrue(len(midi.sustain_messages) == 5)
        self.assertTrue(midi.sustain_messages[-1].value == 0)
        self.assertTrue(midi.sustain_messages[0] == sustain_msgs[0])
        # check notes with and without sustain information
        self.assertTrue(np.allclose(midi.notes,
                                    [[0.146875, 72., 3.32291667, 63., 0.],
                                     [1.56666667, 41., 0.22291667, 29., 0.],
                                     [2.525, 77., 0.93020833, 72., 0.],
                                     [2.54895833, 60., 0.21041667, 28., 0.],
                                     [2.5625, 65., 0.20208333, 34., 0.],
                                     [2.57604167, 56., 0.234375, 31., 0.],
                                     [3.36875, 75., 0.78020833, 64., 0.],
                                     [3.44895833, 43., 0.271875, 35., 0.]]))
        self.assertTrue(np.allclose(midi.sustained_notes,
                                    [[0.146875, 72., 4.00208333, 63., 0.],
                                     [1.56666667, 41., 0.22291667, 29., 0.],
                                     [2.525, 77., 1.62395833, 72., 0.],
                                     [2.54895833, 60., 0.21041667, 28., 0.],
                                     [2.5625, 65., 0.20208333, 34., 0.],
                                     [2.57604167, 56., 0.234375, 31., 0.],
                                     [3.36875, 75., 0.78020833, 64., 0.],
                                     [3.44895833, 43., 0.7, 35., 0.]]))

    def test_time_signature(self):
        midi = MIDIFile(pj(ANNOTATIONS_PATH, 'stereo_sample.mid'))
        self.assertTrue(np.allclose(midi.time_signatures, [[0, 4, 4]]))

    def test_tempi(self):
        midi = MIDIFile(pj(ANNOTATIONS_PATH, 'stereo_sample.mid'))
        self.assertTrue(np.allclose(midi.tempi, [[0, 500000]]))


class TestWriteMidiFunction(unittest.TestCase):

    def test_write_midi(self):
        notes = np.loadtxt(pj(ANNOTATIONS_PATH, 'stereo_sample.notes'))
        # write to a temporary file
        write_midi(notes, tmp_file)
        # read in that file and compare notes
        tmp_midi = MIDIFile(tmp_file)
        self.assertTrue(np.allclose(notes, tmp_midi.notes[:, :4], atol=1e-3))


class TestLoadMidiFunction(unittest.TestCase):

    def test_load_midi(self):
        notes = load_midi(pj(ANNOTATIONS_PATH, 'stereo_sample.mid'))
        notes_txt = np.loadtxt(pj(ANNOTATIONS_PATH, 'stereo_sample.notes'))
        self.assertTrue(np.allclose(notes[:, :4], notes_txt, atol=1e-3))

    def test_load_midi_sustained(self):
        notes = load_midi(pj(ANNOTATIONS_PATH, 'stereo_sample_sustained.mid'))
        self.assertTrue(np.allclose(notes[0, 2], 3.32291667))
        notes = load_midi(pj(ANNOTATIONS_PATH, 'stereo_sample_sustained.mid'),
                          sustain=True)
        self.assertTrue(np.allclose(notes[0, 2], 4.00208333))


# clean up
def teardown_module():
    os.unlink(tmp_file)
