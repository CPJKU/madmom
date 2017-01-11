# encoding: utf-8
"""
This file contains functions to control drumotron, a drum robot.

"""

from __future__ import absolute_import, division, print_function
import numpy as np
import pickle
from madmom.processors import Processor


class DrumotronHardwareProcessor(Processor):

    def __init__(self, arduino=False):
        if arduino:
            import serial
            self.ser = serial.Serial('/dev/ttyACM0', 9600)
        self.arduino = arduino

    def process(self, cmd):
        if self.arduino:
            self.ser.write(cmd)
            print('a command ', cmd)


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
        self.beat_frame_counter_int = None
        self.beat_periods = [None] * smooth_win_len
        self.current_beat_period = None
        self.beat_count = None
        self.pattern_id = None
        self.beat_grid = None
        self.beat_frame_counter_ext = None
        self.last_played_position = None
        self.out = out

    def process(self, data):
        """
        Play drums

        Parameters
        ----------
        data : tuple (beat_interval, beat_count, pattern_id)

        Returns
        -------
        hit : None or numpy array
            Defines if and which drum to hit

        """
        # print('======= %s ======' % str(self.beat_frame_counter_int))
        if data is None:
            beat_count = None
        else:
            (beat_interval, beat_count, pattern_id) = data
        is_beat = beat_count is not None
        if is_beat:
            # print('-- new ext beat', beat_count)
            self.pattern_id = pattern_id
            if self.smooth_win_len > 0:
                # shift entries to the left
                self.beat_periods[:-1] = self.beat_periods[1:]
                # append new beat period
                self.beat_periods[-1] = beat_interval
                if None not in self.beat_periods:
                    beat_interval = np.median(self.beat_periods)
            self.current_beat_period = beat_interval
            # print('--- beat (period %i) ---' % beat_interval)
            self.beat_count = beat_count
            self.beat_frame_counter_int = self.delay
            self.beat_frame_counter_ext = self.beat_frame_counter_int
            # create bins to relate the frame counter to the beat grid
            self.beat_grid = np.linspace(0, beat_interval, self.grid + 1)[:-1]
        if self.beat_frame_counter_int is None:
            return None
        if (is_beat is False) and (self.beat_frame_counter_int >
                                   self.current_beat_period - 1):
            # start new bar
            self.beat_frame_counter_int = 0
            # increase beat counter
            self.beat_count = int(self.beat_count % self.patterns[
                self.pattern_id]['num_beats'] + 1)
            # print('-- new int beat', self.beat_count)
        current_position = int(np.digitize(
            self.beat_frame_counter_int, self.beat_grid) - 1 + \
            (self.beat_count - 1) * self.grid)
        # if is_beat or self.beat_frame_counter_int == 0:
            # print('cp =', current_position)

        if self.last_position != current_position:
            self.last_position = current_position
            if current_position != self.last_played_position:
                if current_position in self.patterns[self.pattern_id]['hh']:
                    self.out('2')
                    self.last_played_position = current_position
                    self.out('1')
                if current_position in self.patterns[self.pattern_id]['sn']:
                    self.out('4')
                    self.last_played_position = current_position
                    self.out('1')
                if current_position in self.patterns[self.pattern_id]['bd']:
                    # self.out('3')
                    self.last_played_position = current_position
        # update state variables
        self.beat_frame_counter_int += 1
        self.beat_frame_counter_ext += 1
        self.last_position = current_position
