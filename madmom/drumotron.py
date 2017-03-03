# encoding: utf-8
"""
This file contains functions to control drumotron, a drum robot.

"""

from __future__ import absolute_import, division, print_function
import numpy as np
import pickle
from os.path import join
from madmom.audio.signal import Signal
from madmom.processors import Processor
from pyaudio import PyAudio, paInt16, paContinue

SAMPLE_RATE = 44100


class DrumotronHardwareProcessor(Processor):

    def __init__(self, arduino=False):
        if arduino:
            import serial
            self.ser = serial.Serial('/dev/ttyACM0', 9600)
        self.arduino = arduino

    def process(self, cmd):
        if self.arduino:
            self.ser.write(cmd)
        # print('command ', cmd)


class DrumotronSamplePlayer(Processor):

    def __init__(self, sample_folder, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
        # load samples
        bd = Signal(join(sample_folder, 'bd.wav'))
        sn = Signal(join(sample_folder, 'sn.wav'))
        hh = Signal(join(sample_folder, 'hh.wav'))
        chunk_size = np.max([len(bd), len(sn), len(hh)])

        self.pa = PyAudio()
        self.stream = self.pa.open(format=paInt16,
                                   frames_per_buffer=chunk_size,
                                   channels=1,
                                   rate=self.sample_rate,
                                   output=True)

        # the sound needs to be longer (otherwise it doesn't play)
        out = np.zeros(chunk_size)
        out[:len(bd)] = bd
        # converting in int16
        self.bd = out.astype(np.int16).tostring()

        # the sound needs to be longer (otherwise it doesn't play)
        out = np.zeros(chunk_size)
        out[:len(sn)] = sn
        # converting in int16
        self.sn = out.astype(np.int16).tostring()

        # the sound needs to be longer (otherwise it doesn't play)
        out = np.zeros(chunk_size)
        out[:len(hh)] = hh
        # converting in int16
        self.hh = out.astype(np.int16).tostring()

    def process(self, cmd):
        if cmd == '1':
            self.stream.write(self.bd)
        elif cmd == '2':
            self.stream.write(self.sn)
        # elif cmd == '3':
        #     self.stream.write(self.hh)

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()


class DrumotronControlProcessor(Processor):
    """
    Plays drums.

    Parameters
    ----------
    grid : int, optional
        Number of grid points per beat.
    """
    def __init__(self, pattern_files, delay=0, smooth_win_len=0,
                 grid=4, out=print):
        # store parameters
        self.delay = delay
        self.smooth_win_len = smooth_win_len
        self.grid = grid
        # load patterns
        self.patterns = [dict(np.load(pf)) for pf in pattern_files]
        # apply discretisation
        for p in range(len(self.patterns)):
            self.patterns[p]['hh'] = [int(np.round(float(h) * float(grid)))
                                      for h in self.patterns[p]['hh']]
            self.patterns[p]['sn'] = [int(np.round(float(h) * float(grid)))
                                      for h in self.patterns[p]['sn']]
            self.patterns[p]['bd'] = [int(np.round(float(h) * float(grid)))
                                      for h in self.patterns[p]['bd']]
        self.num_beats = [int(p['num_beats']) for p in self.patterns]
        # variables for storing intermediate data
        self.last_position = None
        # counts the frames since the last beat
        self.frame_counter = None
        if smooth_win_len > 0:
            self.beat_periods = [None] * smooth_win_len
        else:
            self.beat_periods = [None] * 3
        self.beat_period = None
        # beat count, e.g., 1, 2, 3, 4 for a 4/4 meter
        self.beat_count = None
        self.pattern_id = None
        # divide a bar into discrete cells
        self.beat_grid = None
        self.last_played_position = None
        self.out = out
        self.frame_count = 0
        self.beats_since_sync = 0

    def process(self, data):
        """
        Play drums

        Parameters
        ----------
        data : tuple (beat_period, beat_count, pattern_id)

        Returns
        -------
        hit : None or numpy array
            Defines if and which drum to hit

        """
        self.frame_count += 1
        if data is None:
            is_beat = False
        else:
            (beat_period, beat_count, pattern_id) = data
            is_beat = beat_count is not None
        if is_beat:
            # store new information about pattern_id, beat_count, beat_period
            self.pattern_id = pattern_id
            self.smooth_beat_period(beat_period)
            self.beat_count = beat_count
            self.frame_counter = self.delay
            # create bins to relate the frame counter to the beat grid
            self.beat_grid = np.linspace(
                0, self.beat_period, self.grid + 1)[:-1]
            self.beats_since_sync = 0
        if self.frame_counter is None:
            return None
        if (is_beat is False) and (self.frame_counter >
                                   self.beat_period - 1):
            # trigger new beat without observing an external one
            self.beats_since_sync += 1
            self.frame_counter = 0
            # increase beat counter
            self.beat_count = int(self.beat_count % self.patterns[
                self.pattern_id]['num_beats'] + 1)
        # map the beat_frame_counter to the pattern grid
        current_position = int(np.digitize(
            self.frame_counter, self.beat_grid) - 1 + \
                               (self.beat_count - 1) * self.grid)

        if self.last_position != current_position:
            if current_position in self.patterns[self.pattern_id]['hh']:
                self.out('3')
                # print('hh at %i' % self.frame_count)
            if current_position in self.patterns[self.pattern_id]['sn']:
                self.out('2')
                # print('sn at %i' % self.frame_count)
            if current_position in self.patterns[self.pattern_id]['bd']:
                self.out('1')
                # print('bd at %i' % self.frame_count)
        # update state variables
        self.frame_counter += 1
        self.last_position = current_position
        if (self.beat_periods[-1] is None) or (self.beats_since_sync > 2):
            # stop tracking if no beat has been observed for 2 beat periods
            self.frame_counter = None

    def smooth_beat_period(self, beat_period):
        # shift entries to the left
        self.beat_periods[:-1] = self.beat_periods[1:]
        # append new beat period
        self.beat_periods[-1] = beat_period
        if self.smooth_win_len > 0:
            if None not in self.beat_periods:
                beat_period = np.median(self.beat_periods)
        self.beat_period = beat_period
