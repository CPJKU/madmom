# encoding: utf-8
# pylint: disable=no-member
"""
This module contains MIDI functionality.

"""

from __future__ import absolute_import, division, print_function

import numpy as np
import mido
import warnings

DEFAULT_TEMPO = 500000  # microseconds per quarter note (i.e. 120 bpm in 4/4)
DEFAULT_TICKS_PER_BEAT = 480  # ticks per quarter note
DEFAULT_TIME_SIGNATURE = (4, 4)


# TODO: remove these unit conversion functions after upstream PR is merged
#       https://github.com/olemb/mido/pull/114
def tick2second(tick, ticks_per_beat=DEFAULT_TICKS_PER_BEAT,
                tempo=DEFAULT_TEMPO):
    """
    Convert absolute time in ticks to seconds.

    Returns absolute time in seconds for a chosen MIDI file time resolution
    (ticks/pulses per quarter note, also called PPQN) and tempo (microseconds
    per quarter note).

    """
    # Note: both tempo (microseconds) and ticks are per quarter note
    #       thus the time signature is irrelevant
    scale = tempo * 1e-6 / ticks_per_beat
    return tick * scale


def second2tick(second, ticks_per_beat=DEFAULT_TICKS_PER_BEAT,
                tempo=DEFAULT_TEMPO):
    """
    Convert absolute time in seconds to ticks.

    Returns absolute time in ticks for a chosen MIDI file time resolution
    (ticks/pulses per quarter note, also called PPQN) and tempo (microseconds
    per quarter note).

    """
    # Note: both tempo (microseconds) and ticks are per quarter note
    #       thus the time signature is irrelevant
    scale = tempo * 1e-6 / ticks_per_beat
    return int(round(second / scale))


def bpm2tempo(bpm, time_signature=DEFAULT_TIME_SIGNATURE):
    """
    Convert BPM (beats per minute) to MIDI file tempo (microseconds per
    quarter note).

    Depending on the chosen time signature a bar contains a different number of
    beats. These beats are multiples/fractions of a quarter note, thus the
    returned BPM depend on the time signature.

    """
    return int(round(60 * 1e6 / bpm * time_signature[1] / 4.))


def tempo2bpm(tempo, time_signature=DEFAULT_TIME_SIGNATURE):
    """
    Convert MIDI file tempo (microseconds per quarter note) to BPM (beats per
    minute).

    Depending on the chosen time signature a bar contains a different number of
    beats. These beats are multiples/fractions of a quarter note, thus the
    returned tempo depends on the time signature.

    """
    return 60 * 1e6 / tempo * time_signature[1] / 4.


def tick2beat(tick, ticks_per_beat=DEFAULT_TICKS_PER_BEAT,
              time_signature=DEFAULT_TIME_SIGNATURE):
    """
    Convert ticks to beats.

    Returns beats for a chosen MIDI file time resolution (ticks/pulses per
    quarter note, also called PPQN) and time signature.

    """
    return tick / (4. * ticks_per_beat / time_signature[1])


def beat2tick(beat, ticks_per_beat=DEFAULT_TICKS_PER_BEAT,
              time_signature=DEFAULT_TIME_SIGNATURE):
    """
    Convert beats to ticks.

    Returns ticks for a chosen MIDI file time resolution (ticks/pulses per
    quarter note, also called PPQN) and time signature.

    """
    return int(round(beat * 4. * ticks_per_beat / time_signature[1]))


class MIDIFile(mido.MidiFile):
    """
    MIDI File.

    Parameters
    ----------
    filename : str
        MIDI file name.
    file_format : int, optional
        MIDI file format (0, 1, 2).
    ticks_per_beat : int, optional
        Resolution (i.e. ticks per quarter note) of the MIDI file.
    unit : str, optional
        Unit of all MIDI messages, can be one of the following:

        - 'ticks', 't': use native MIDI ticks as unit,
        - 'seconds', 's': use seconds as unit,
        - 'beats', 'b' : use beats as unit.

    timing : str, optional
        Timing of all MIDI messages, can be one of the following:

        - 'absolute', 'abs', 'a': use absolute timing.
        - 'relative', 'rel', 'r': use relative timing, i.e. delta to
          previous message.

    Examples
    --------
    Create a MIDI file from an array with notes. The format of the note array
    is: 'onset time', 'pitch', 'duration', 'velocity', 'channel'. The last
    column can be omitted, assuming channel 0.

    >>> notes = np.array([[0, 50, 1, 60], [0.5, 62, 0.5, 90]])
    >>> m = MIDIFile.from_notes(notes)
    >>> m  # doctest: +ELLIPSIS
    <madmom.io.midi.MIDIFile object at 0x...>

    The notes can be accessed as a numpy array in various formats (default is
    seconds):

    >>> m.notes
    array([[ 0. , 50. ,  1. , 60. ,  0. ],
           [ 0.5, 62. ,  0.5, 90. ,  0. ]])
    >>> m.unit ='ticks'
    >>> m.notes
    array([[  0.,  50., 960.,  60.,   0.],
           [480.,  62., 480.,  90.,   0.]])
    >>> m.unit = 'seconds'
    >>> m.notes
    array([[ 0. , 50. ,  1. , 60. ,  0. ],
           [ 0.5, 62. ,  0.5, 90. ,  0. ]])
    >>> m.unit = 'beats'
    >>> m.notes
    array([[ 0., 50.,  2., 60.,  0.],
           [ 1., 62.,  1., 90.,  0.]])

    >>> m = MIDIFile.from_notes(notes, tempo=60)
    >>> m.notes
    array([[ 0. , 50. ,  1. , 60. ,  0. ],
           [ 0.5, 62. ,  0.5, 90. ,  0. ]])
    >>> m.unit = 'ticks'
    >>> m.notes
    array([[  0.,  50., 480.,  60.,   0.],
           [240.,  62., 240.,  90.,   0.]])
    >>> m.unit = 'beats'
    >>> m.notes
    array([[ 0. , 50. ,  1. , 60. ,  0. ],
           [ 0.5, 62. ,  0.5, 90. ,  0. ]])

    >>> m = MIDIFile.from_notes(notes, time_signature=(2, 2))
    >>> m.notes
    array([[ 0. , 50. ,  1. , 60. ,  0. ],
           [ 0.5, 62. ,  0.5, 90. ,  0. ]])
    >>> m.unit = 'ticks'
    >>> m.notes
    array([[   0.,   50., 1920.,   60.,    0.],
           [ 960.,   62.,  960.,   90.,    0.]])
    >>> m.unit = 'beats'
    >>> m.notes
    array([[ 0., 50.,  2., 60.,  0.],
           [ 1., 62.,  1., 90.,  0.]])

    >>> m = MIDIFile.from_notes(notes, tempo=60, time_signature=(2, 2))
    >>> m.notes
    array([[ 0. , 50. ,  1. , 60. ,  0. ],
           [ 0.5, 62. ,  0.5, 90. ,  0. ]])
    >>> m.unit = 'ticks'
    >>> m.notes
    array([[  0.,  50., 960.,  60.,   0.],
           [480.,  62., 480.,  90.,   0.]])
    >>> m.unit = 'beats'
    >>> m.notes
    array([[ 0. , 50. ,  1. , 60. ,  0. ],
           [ 0.5, 62. ,  0.5, 90. ,  0. ]])

    >>> m = MIDIFile.from_notes(notes, tempo=240, time_signature=(3, 8))
    >>> m.notes
    array([[ 0. , 50. ,  1. , 60. ,  0. ],
           [ 0.5, 62. ,  0.5, 90. ,  0. ]])
    >>> m.unit = 'ticks'
    >>> m.notes
    array([[  0.,  50., 960.,  60.,   0.],
           [480.,  62., 480.,  90.,   0.]])
    >>> m.unit = 'beats'
    >>> m.notes
    array([[ 0., 50.,  4., 60.,  0.],
           [ 2., 62.,  2., 90.,  0.]])

    """

    def __init__(self, filename=None, file_format=0,
                 ticks_per_beat=DEFAULT_TICKS_PER_BEAT, unit='seconds',
                 timing='absolute', **kwargs):
        # instantiate a MIDIFile
        super(MIDIFile, self).__init__(filename=filename, type=file_format,
                                       ticks_per_beat=ticks_per_beat, **kwargs)
        # add attributes for unit conversion
        self.unit = unit
        self.timing = timing

    # TODO: remove this method after upstream PR is merged
    #       https://github.com/olemb/mido/pull/115
    def __iter__(self):
        # The tracks of type 2 files are not in sync, so they can
        # not be played back like this.
        if self.type == 2:
            raise TypeError("can't merge tracks in type 2 (asynchronous) file")

        tempo = DEFAULT_TEMPO
        time_signature = DEFAULT_TIME_SIGNATURE
        cum_delta = 0
        for msg in mido.merge_tracks(self.tracks):
            # Convert relative message time to desired unit
            if msg.time > 0:
                if self.unit.lower() in ('t', 'ticks'):
                    delta = msg.time
                elif self.unit.lower() in ('s', 'sec', 'seconds'):
                    delta = tick2second(msg.time, self.ticks_per_beat, tempo)
                elif self.unit.lower() in ('b', 'beats'):
                    delta = tick2beat(msg.time, self.ticks_per_beat,
                                      time_signature)
                else:
                    raise ValueError("`unit` must be either 'ticks', 't', "
                                     "'seconds', 's', 'beats', 'b', not %s." %
                                     self.unit)
            else:
                delta = 0
            # Convert relative time to absolute values if needed
            if self.timing.lower() in ('a', 'abs', 'absolute'):
                cum_delta += delta
            elif self.timing.lower() in ('r', 'rel', 'relative'):
                cum_delta = delta
            else:
                raise ValueError("`timing` must be either 'relative', 'rel', "
                                 "'r', or 'absolute', 'abs', 'a', not %s." %
                                 self.timing)

            yield msg.copy(time=cum_delta)

            if msg.type == 'set_tempo':
                tempo = msg.tempo
            elif msg.type == 'time_signature':
                time_signature = (msg.numerator, msg.denominator)

    def __repr__(self):
        return object.__repr__(self)

    @property
    def tempi(self):
        """
        Tempi (microseconds per quarter note) of the MIDI file.

        Returns
        -------
        tempi : numpy array
            Array with tempi (time, tempo).

        Notes
        -----
        The time will be given in the unit set by `unit`.

        """
        # list for all tempi
        tempi = []
        # process all events
        for msg in self:
            if msg.type == 'set_tempo':
                tempi.append((msg.time, msg.tempo))
        # make sure a tempo is set (and occurs at time 0)
        if not tempi or tempi[0][0] > 0:
            tempi.insert(0, (0, DEFAULT_TEMPO))
        # tempo is given in microseconds per quarter note
        # TODO: add option to return in BPM
        return np.asarray(tempi, np.float)

    @property
    def time_signatures(self):
        """
        Time signatures of the MIDI file.

        Returns
        -------
        time_signatures : numpy array
            Array with time signatures (time, numerator, denominator).

        Notes
        -----
        The time will be given in the unit set by `unit`.

        """
        # list for all time signature
        signatures = []
        # process all events
        for msg in self:
            if msg.type == 'time_signature':
                signatures.append((msg.time, msg.numerator, msg.denominator))
        # make sure a signatures is set (and occurs at time 0)
        if not signatures or signatures[0][0] > 0:
            signatures.insert(0, (0, DEFAULT_TIME_SIGNATURE[0],
                                  DEFAULT_TIME_SIGNATURE[1]))
        # return time signatures
        return np.asarray(signatures, dtype=np.float)

    @property
    def notes(self):
        """
        Notes of the MIDI file.

        Returns
        -------
        notes : numpy array
            Array with notes (onset time, pitch, duration, velocity, channel).

        """
        # lists to collect notes and sustain messages
        notes = []
        # dictionary for storing the last onset time and velocity for each
        # individual note (i.e. same pitch and channel)
        sounding_notes = {}

        # as key for the dict use channel * 128 (max number of pitches) + pitch
        def note_hash(channel, pitch):
            """Generate a note hash."""
            return channel * 128 + pitch

        # process all events
        for msg in self:
            # use only note on or note off events
            note_on = msg.type == 'note_on'
            note_off = msg.type == 'note_off'
            if not (note_on or note_off):
                continue
            # hash sounding note
            note = note_hash(msg.channel, msg.note)
            # start note if it's a 'note on' event with velocity > 0
            if note_on and msg.velocity > 0:
                # save the onset time and velocity
                sounding_notes[note] = (msg.time, msg.velocity)
            # end note if it's a 'note off' event or 'note on' with velocity 0
            elif note_off or (note_on and msg.velocity == 0):
                if note not in sounding_notes:
                    warnings.warn('ignoring MIDI message %s' % msg)
                    continue
                # append the note to the list
                notes.append((sounding_notes[note][0], msg.note,
                              msg.time - sounding_notes[note][0],
                              sounding_notes[note][1], msg.channel))
                # remove hash from dict
                del sounding_notes[note]

        # sort the notes and convert to numpy array
        return np.asarray(sorted(notes), dtype=np.float)

    @property
    def sustain_messages(self):
        """
        Sustain messages of the MIDI file.

        Returns
        -------
        sustain_messages : list
            List with MIDI sustain messages.

        Notes
        -----
        If the last sustain message is a 'sustain on' message (i.e. it has a
        value >= 64), an artificial sustain message with a value of 0 and the
        timing of the last MIDI message is appended to the list.

        """
        sustain_msgs = []
        last_msg_time = None
        for msg in self:
            last_msg_time = msg.time
            # keep track of sustain information
            if msg.type == 'control_change' and msg.control == 64:
                sustain_msgs.append(msg)
        # if the last sustain message is 'sustain on', append a fake sustain
        # message to end sustain with the last note
        if sustain_msgs and sustain_msgs[-1].value >= 64:
            msg = sustain_msgs[-1].copy()
            msg.time = last_msg_time
            msg.value = 0
            sustain_msgs.append(msg)
        return sustain_msgs

    @property
    def sustained_notes(self):
        """
        Notes of the MIDI file with applied sustain information.

        Returns
        -------
        notes : numpy array
            Array with notes (onset time, pitch, duration, velocity, channel).

        """
        notes = np.copy(self.notes)
        # apply sustain information
        # keep track of sustain start times (channel = key)
        sustain_starts = {}
        note_offsets = notes[:, 0] + notes[:, 2]
        for msg in self.sustain_messages:
            # remember sustain start
            if msg.value >= 64:
                if msg.channel in sustain_starts:
                    # sustain is ON already, ignoring
                    continue
                sustain_starts[msg.channel] = msg.time
            # expand all notes in this channel until sustain end
            else:
                if msg.channel not in sustain_starts:
                    # sustain is OFF already, ignoring
                    continue
                # end all notes with i) offsets between sustain start and end
                sustained = np.logical_and(
                    note_offsets >= sustain_starts[msg.channel],
                    note_offsets <= msg.time)
                # and ii) same channel
                sustained = np.logical_and(sustained,
                                           notes[:, 4] == msg.channel)
                # update duration of notes (sustain end time - onset time)
                notes[sustained, 2] = msg.time - notes[sustained, 0]
                # remove sustain start time for this channel
                del sustain_starts[msg.channel]
        # end all notes latest when next note (of same pitch) starts
        for pitch in np.unique(notes[:, 1]):
            note_idx = np.nonzero(notes[:, 1] == pitch)[0]
            max_duration = np.diff(notes[note_idx, 0])
            notes[note_idx[:-1], 2] = np.minimum(notes[note_idx[:-1], 2],
                                                 max_duration)
        # finally return notes
        return notes

    @classmethod
    def from_notes(cls, notes, unit='seconds', tempo=DEFAULT_TEMPO,
                   time_signature=DEFAULT_TIME_SIGNATURE,
                   ticks_per_beat=DEFAULT_TICKS_PER_BEAT):
        """
        Create a MIDIFile from the given notes.

        Parameters
        ----------
        notes : numpy array
            Array with notes, one per row. The columns are defined as:
            (onset time, pitch, duration, velocity, [channel]).
        unit : str, optional
            Unit of `notes`, can be one of the following:

            - 'seconds', 's': use seconds as unit,
            - 'ticks', 't': use native MIDI ticks as unit,
            - 'beats', 'b' : use beats as unit.

        tempo : float, optional
            Tempo of the MIDI track, given in bpm or microseconds per quarter
            note. The unit is determined automatically by the value:

            - `tempo` <= 1000: bpm
            - `tempo` > 1000: microseconds per quarter note

        time_signature : tuple, optional
            Time signature of the track, e.g. (4, 4) for 4/4.
        ticks_per_beat : int, optional
            Resolution (i.e. ticks per quarter note) of the MIDI file.

        Returns
        -------
        :class:`MIDIFile` instance
            :class:`MIDIFile` instance with all notes collected in one track.

        Notes
        -----
        All note events (including the generated tempo and time signature
        events) are written into a single track (i.e. MIDI file format 0).

        """
        # create new MIDI file
        midi_file = cls(file_format=0, ticks_per_beat=ticks_per_beat,
                        unit=unit, timing='absolute')
        # convert tempo
        if tempo <= 1000:
            # convert from bpm to tempo
            tempo = bpm2tempo(tempo, time_signature)
        else:
            # tempo given in ticks per quarter note
            # i.e. we have to adjust according to the time signature
            tempo = int(tempo * time_signature[1] / 4)
        # create new track and add tempo and time signature information
        track = midi_file.add_track()
        track.append(mido.MetaMessage('set_tempo', tempo=tempo))
        track.append(mido.MetaMessage('time_signature',
                                      numerator=time_signature[0],
                                      denominator=time_signature[1]))
        # create note on/off messages with absolute timing
        messages = []
        for note in notes:
            try:
                onset, pitch, duration, velocity, channel = note
            except ValueError:
                onset, pitch, duration, velocity = note
                channel = 0
            pitch = int(pitch)
            velocity = int(velocity)
            channel = int(channel)
            offset = onset + duration
            # create MIDI messages
            onset = second2tick(onset, ticks_per_beat, tempo)
            note_on = mido.Message('note_on', time=onset, note=pitch,
                                   velocity=velocity, channel=channel)
            offset = second2tick(offset, ticks_per_beat, tempo)
            note_off = mido.Message('note_off', time=offset, note=pitch,
                                    channel=channel)
            # append to list
            messages.extend([note_on, note_off])
        # sort them, convert to relative timing and append to track
        messages.sort(key=lambda msg: msg.time)
        messages = mido.midifiles.tracks._to_reltime(messages)
        track.extend(messages)
        # return MIDI file
        return midi_file

    def save(self, filename):
        """
        Save to MIDI file.

        Parameters
        ----------
        filename : str or open file handle
            The MIDI file name.

        """
        from . import open_file
        # write the MIDI stream
        with open_file(filename, 'wb') as f:
            self._save(f)


def load_midi(filename):
    """
    Load notes from a MIDI file.

    Parameters
    ----------
    filename: str
        MIDI file.

    Returns
    -------
    numpy array
        Notes ('onset time' 'note number' 'duration' 'velocity' 'channel')

    """
    return MIDIFile(filename).notes


def write_midi(notes, filename, duration=0.6, velocity=100):
    """
    Write notes to a MIDI file.

    Parameters
    ----------
    notes : numpy array, shape (num_notes, 2)
        Notes, one per row (column definition see notes).
    filename : str
        Output MIDI file.
    duration : float, optional
        Note duration if not defined by `notes`.
    velocity : int, optional
        Note velocity if not defined by `notes`.

    Returns
    -------
    numpy array
        Notes (including note length, velocity and channel).

    Notes
    -----
    The note columns format must be (duration, velocity and channel optional):

    'onset time' 'note number' ['duration' ['velocity' ['channel']]]

    """
    from ..utils import expand_notes
    # expand the array to have a default duration and velocity
    notes = expand_notes(notes, duration, velocity)
    # write the notes to the file and return them
    MIDIFile.from_notes(notes).save(filename)
