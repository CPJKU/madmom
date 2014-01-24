#!/usr/bin/env python
# encoding: utf-8
"""
This file contains MIDI functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

Almost all code is taken from Giles Hall's python-midi package:
https://github.com/vishnubob/python-midi

The last merged commit is 3053fefe8cd829ff891ac4fe58dc230744fce0e6

The MIT License (MIT)
Copyright (c) 2013 Giles F. Hall

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import math
import struct

# constants
OCTAVE_MAX_VALUE = 12
OCTAVE_VALUES = range(OCTAVE_MAX_VALUE)

NOTE_NAMES = ['C', 'Cs', 'D', 'Ds', 'E', 'F', 'Fs', 'G', 'Gs', 'A', 'As', 'B']
WHITE_KEYS = [0, 2, 4, 5, 7, 9, 11]
BLACK_KEYS = [1, 3, 6, 8, 10]
NOTE_PER_OCTAVE = len(NOTE_NAMES)
NOTE_VALUES = range(OCTAVE_MAX_VALUE * NOTE_PER_OCTAVE)
NOTE_NAME_MAP_FLAT = {}
NOTE_VALUE_MAP_FLAT = []
NOTE_NAME_MAP_SHARP = {}
NOTE_VALUE_MAP_SHARP = []

for index in range(128):
    note_idx = index % NOTE_PER_OCTAVE
    oct_idx = index / OCTAVE_MAX_VALUE
    name = NOTE_NAMES[note_idx]
    if len(name) == 2:
        # sharp note
        flat = NOTE_NAMES[note_idx + 1] + 'b'
        NOTE_NAME_MAP_FLAT['%s_%d' % (flat, oct_idx)] = index
        NOTE_NAME_MAP_SHARP['%s_%d' % (name, oct_idx)] = index
        NOTE_VALUE_MAP_FLAT.append('%s_%d' % (flat, oct_idx))
        NOTE_VALUE_MAP_SHARP.append('%s_%d' % (name, oct_idx))
        globals()['%s_%d' % (name[0] + 's', oct_idx)] = index
        globals()['%s_%d' % (flat, oct_idx)] = index
    else:
        NOTE_NAME_MAP_FLAT['%s_%d' % (name, oct_idx)] = index
        NOTE_NAME_MAP_SHARP['%s_%d' % (name, oct_idx)] = index
        NOTE_VALUE_MAP_FLAT.append('%s_%d' % (name, oct_idx))
        NOTE_VALUE_MAP_SHARP.append('%s_%d' % (name, oct_idx))
        globals()['%s_%d' % (name, oct_idx)] = index

BEAT_NAMES = ['whole', 'half', 'quarter', 'eighth', 'sixteenth',
              'thirty-second', 'sixty-fourth']
BEAT_VALUES = [4, 2, 1, .5, .25, .125, .0625]
WHOLE = 0
HALF = 1
QUARTER = 2
EIGHTH = 3
SIXTEENTH = 4
THIRTY_SECOND = 5
SIXTY_FOURTH = 6

DEFAULT_MIDI_HEADER_SIZE = 14


# functions for packing / unpacking variable length data
def read_variable_length(data):
    """
    Read a variable length variable from the given data.

    :param data: data
    :return:     length in bytes

    """
    next_byte = 1
    value = 0
    while next_byte:
        next_value = ord(data.next())
        # is the hi-bit set?
        if not (next_value & 0x80):
            # no next BYTE
            next_byte = 0
            # mask out the 8th bit
        next_value &= 0x7f
        # shift last value up 7 bits
        value <<= 7
        # add new value
        value += next_value
    return value


def write_variable_length(value):
    """

    :param value:
    :return:
    """
    chr1 = chr(value & 0x7F)
    value >>= 7
    if value:
        chr2 = chr((value & 0x7F) | 0x80)
        value >>= 7
        if value:
            chr3 = chr((value & 0x7F) | 0x80)
            value >>= 7
            if value:
                chr4 = chr((value & 0x7F) | 0x80)
                result = chr4 + chr3 + chr2 + chr1
            else:
                result = chr3 + chr2 + chr1
        else:
            result = chr2 + chr1
    else:
        result = chr1
    return result


# Event classes
class EventRegistry(object):
    """
    Event Registry.

    """
    Events = {}
    MetaEvents = {}

    @classmethod
    def register_event(cls, event, bases):
        """
        Registers an event in the registry.
        :param event:      the event to register
        :param bases:      the base class
        :raise ValueError: for unknown events

        """
        if (Event in bases) or (NoteEvent in bases):
            assert event.status_msg not in cls.Events, \
                "Event %s already registered" % event.name
            cls.Events[event.status_msg] = event
        elif (MetaEvent in bases) or (MetaEventWithText in bases):
            assert event.meta_command not in cls.MetaEvents, \
                "Event %s already registered" % event.name
            cls.MetaEvents[event.meta_command] = event
        else:
            raise ValueError("Unknown bases class in event type: %s" %
                             event.name)


class AbstractEvent(object):
    """
    Abstract Event.

    """
    __slots__ = ['tick', 'data']
    name = "Generic MIDI Event"
    length = 0
    status_msg = 0x0

    class __metaclass__(type):
        """
        Class factory class.

        """
        def __init__(cls, name, bases, dict):
            if name not in ['AbstractEvent', 'Event', 'MetaEvent', 'NoteEvent',
                            'MetaEventWithText']:
                EventRegistry.register_event(cls, bases)

    def __init__(self, **kwargs):
        if type(self.length) == int:
            data = [0] * self.length
        else:
            data = []
        self.tick = 0
        self.data = data
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def __cmp__(self, other):
        if self.tick < other.tick:
            return -1
        elif self.tick > other.tick:
            return 1
        return cmp(self.data, other.data)

    def __baserepr__(self, keys=[]):
        keys = ['tick'] + keys + ['data']
        body = []
        for key in keys:
            val = getattr(self, key)
            key_val = "%s=%r" % (key, val)
            body.append(key_val)
        body = str.join(', ', body)
        return "midi.%s(%s)" % (self.__class__.__name__, body)

    def __repr__(self):
        return self.__baserepr__()


class Event(AbstractEvent):
    """
    Event.

    """
    __slots__ = ['channel']
    name = 'Event'

    def __init__(self, **kwargs):
        if 'channel' not in kwargs:
            kwargs = kwargs.copy()
            kwargs['channel'] = 0
        super(Event, self).__init__(**kwargs)

    def copy(self, **kwargs):
        """
        Copy an event.

        :param kwargs: dictionary with all attributes to copy.

        """
        raise ValueError("please remove the TODO as it seems needed")
        # TODO: can this method be removed?
        _kwargs = {'channel': self.channel, 'tick': self.tick,
                   'data': self.data}
        _kwargs.update(kwargs)
        return self.__class__(**_kwargs)

    def __cmp__(self, other):
        if self.tick < other.tick:
            return -1
        elif self.tick > other.tick:
            return 1
        return 0

    def __repr__(self):
        return self.__baserepr__(['channel'])

    @classmethod
    def is_event(cls, status_msg):
        """
        Indicates whether the given status message belongs to this event.

        :param status_msg: status message
        :return:           boolean

        """
        return cls.status_msg == (status_msg & 0xF0)


class MetaEvent(AbstractEvent):
    """
    MetaEvent is a special subclass of Event that is not meant to be used as a
    concrete class. It defines a subset of Events known as the Meta events.

    """
    status_msg = 0xFF
    meta_command = 0x0
    name = 'Meta Event'

    @classmethod
    def is_event(cls, status_msg):
        """
        Indicates whether the given status message belongs to this event.

        :param status_msg: status message
        :return:           boolean

        """
        return cls.status_msg == status_msg


class NoteEvent(Event):
    """
    NoteEvent is a special subclass of Event that is not meant to be used as a
    concrete class. It defines the generalities of NoteOn and NoteOff events.

    """
    __slots__ = ['pitch', 'velocity']
    length = 2

    @property
    def pitch(self):
        """
        Pitch of the note event.

        """
        return self.data[0]

    @pitch.setter
    def pitch(self, pitch):
        """
        Set the pitch of the note event.

        :param pitch: pitch of the note.

        """
        self.data[0] = pitch

    @property
    def velocity(self):
        """
        Velocity of the note event.

        """
        return self.data[1]

    @velocity.setter
    def velocity(self, velocity):
        """
        Set the velocity of the note event.

        :param velocity: velocity of the note.

        """
        self.data[1] = velocity


class NoteOnEvent(NoteEvent):
    """
    Note On Event.

    """
    status_msg = 0x90
    name = 'Note On'


class NoteOffEvent(NoteEvent):
    """
    Note Off Event.

    """
    status_msg = 0x80
    name = 'Note Off'


class AfterTouchEvent(Event):
    """
    After Touch Event.

    """
    status_msg = 0xA0
    length = 2
    name = 'After Touch'


class ControlChangeEvent(Event):
    """
    Control Change Event.

    """
    __slots__ = ['control', 'value']
    status_msg = 0xB0
    length = 2
    name = 'Control Change'

    @property
    def control(self):
        """
        Control ID.

        """
        return self.data[0]

    @control.setter
    def control(self, control):
        """
        Set control ID.

        :param control: control ID

        """
        self.data[0] = control

    @property
    def value(self):
        """
        Value of the controller.

        """
        return self.data[1]

    @value.setter
    def value(self, value):
        """
        Set the value of the controller.

        :param value: value

        """
        self.data[1] = value


class ProgramChangeEvent(Event):
    """
    Program Change Event.

    """
    __slots__ = ['value']
    status_msg = 0xC0
    length = 1
    name = 'Program Change'

    @property
    def value(self):
        """
        Value of the Program Change Event.

        """
        return self.data[0]

    @value.setter
    def value(self, value):
        """
        Set the value of the Program Change Event.

        :param value: value
        """
        self.data[0] = value


class ChannelAfterTouchEvent(Event):
    """
    Channel After Touch Event.

    """
    __slots__ = ['value']
    status_msg = 0xD0
    length = 1
    name = 'Channel After Touch'

    @property
    def value(self):
        """
        Value of the Channel After Touch Event.

        """
        return self.data[0]

    @value.setter
    def value(self, value):
        """
        Set the value of the Channel After Touch Event.

        :param value: value
        """
        self.data[0] = value


class PitchWheelEvent(Event):
    """
    Pitch Wheel Event.

    """
    __slots__ = ['pitch']
    status_msg = 0xE0
    length = 2
    name = 'Pitch Wheel'

    @property
    def pitch(self):
        """
        Pitch of the Pitch Wheel Event.

        """
        return ((self.data[1] << 7) | self.data[0]) - 0x2000

    @pitch.setter
    def pitch(self, pitch):
        """
        Set the pitch of the note event.

        :param pitch: pitch of the note.

        """
        value = pitch + 0x2000
        self.data[0] = value & 0x7F
        self.data[1] = (value >> 7) & 0x7F


class SysExEvent(Event):
    """
    System Exclusive Event.

    """
    status_msg = 0xF0
    name = 'SysEx'
    length = 'variable'

    @classmethod
    def is_event(cls, status_msg):
        """
        Indicates whether the given status message belongs to this event.

        :param status_msg: status message
        :return:           boolean

        """
        return cls.status_msg == status_msg


class SequenceNumberMetaEvent(MetaEvent):
    """
    Sequence Number Meta Event.

    """
    name = 'Sequence Number'
    meta_command = 0x00
    length = 2


class MetaEventWithText(MetaEvent):
    """
    Meta Event With Text.

    """
    def __init__(self, **kwargs):
        super(MetaEventWithText, self).__init__(**kwargs)
        if 'text' not in kwargs:
            self.text = ''.join(chr(datum) for datum in self.data)

    def __repr__(self):
        return self.__baserepr__(['text'])


class TextMetaEvent(MetaEventWithText):
    """
    Text Meta Event.

    """
    name = 'Text'
    meta_command = 0x01
    length = 'variable'


class CopyrightMetaEvent(MetaEventWithText):
    """
    Copyright Meta Event.

    """
    name = 'Copyright Notice'
    meta_command = 0x02
    length = 'variable'


class TrackNameEvent(MetaEventWithText):
    """
    Track Name Event.

    """
    name = 'Track Name'
    meta_command = 0x03
    length = 'variable'


class InstrumentNameEvent(MetaEventWithText):
    """
    Instrument Name Event.

    """
    name = 'Instrument Name'
    meta_command = 0x04
    length = 'variable'


class LyricsEvent(MetaEventWithText):
    """
    Lyrics Event.

    """
    name = 'Lyrics'
    meta_command = 0x05
    length = 'variable'


class MarkerEvent(MetaEventWithText):
    """
    Marker Event.

    """
    name = 'Marker'
    meta_command = 0x06
    length = 'variable'


class CuePointEvent(MetaEventWithText):
    """
    Cue Point Event.

    """
    name = 'Cue Point'
    meta_command = 0x07
    length = 'variable'


class SomethingEvent(MetaEvent):
    """
    Something Event.

    """
    name = 'Something'
    meta_command = 0x09


class ChannelPrefixEvent(MetaEvent):
    """
    Channel Prefix Event.

    """
    name = 'Channel Prefix'
    meta_command = 0x20
    length = 1


class PortEvent(MetaEvent):
    """
    Port Event.

    """
    name = 'MIDI Port/Cable'
    meta_command = 0x21


class TrackLoopEvent(MetaEvent):
    """
    Track Loop Event.

    """
    name = 'Track Loop'
    meta_command = 0x2E


class EndOfTrackEvent(MetaEvent):
    """
    End Of Track Event.

    """
    name = 'End of Track'
    meta_command = 0x2F


class SetTempoEvent(MetaEvent):
    """
    Set Tempo Event.

    """
    __slots__ = ['bpm', 'ms_per_quarter_note']
    name = 'Set Tempo'
    meta_command = 0x51
    length = 3

    @property
    def bpm(self):
        """
        Tempo in beats per minute.

        """
        return float(6e7) / self.ms_per_quarter_note

    @bpm.setter
    def bpm(self, bpm):
        """
        Set the tempo in beats per minute.

        :param bpm: beats per minute

        """
        self.ms_per_quarter_note = int(float(6e7) / bpm)

    @property
    def ms_per_quarter_note(self):
        """
        Milliseconds per quarter note.

        """
        assert len(self.data) == 3
        values = [self.data[x] << (16 - (8 * x)) for x in xrange(3)]
        return sum(values)

    @ms_per_quarter_note.setter
    def ms_per_quarter_note(self, ms):
        """
        Set milliseconds per quarter note.

        :param ms: milliseconds

        """
        self.data = [(ms >> (16 - (8 * x)) & 0xFF) for x in range(3)]


class SmpteOffsetEvent(MetaEvent):
    """
    SMPTE Offset Event.

    """
    name = 'SMPTE Offset'
    meta_command = 0x54


class TimeSignatureEvent(MetaEvent):
    """
    Time Signature Event.

    """
    __slots__ = ['numerator', 'denominator', 'metronome', 'thirty_seconds']
    name = 'Time Signature'
    meta_command = 0x58
    length = 4

    @property
    def numerator(self):
        """
        Numerator.

        """
        return self.data[0]

    @numerator.setter
    def numerator(self, numerator):
        """
        Set numerator.

        :param numerator: numerator
        """
        self.data[0] = numerator

    @property
    def denominator(self):
        """
        Denominator.

        """
        return 2 ** self.data[1]

    @denominator.setter
    def denominator(self, denominator):
        """
        Set denominator.

        :param denominator: denominator
        """
        self.data[1] = int(math.log(denominator, 2))

    @property
    def metronome(self):
        """
        Metronome.

        """
        return self.data[2]

    @metronome.setter
    def metronome(self, metronome):
        """
        Set metronome.

        :param metronome:
        """
        self.data[2] = metronome

    @property
    def thirty_seconds(self):
        """
        Thirty-seconds.

        """
        return self.data[3]

    @thirty_seconds.setter
    def thirty_seconds(self, thirty_seconds):
        """
        Set thirty-seconds.

        :param thirty_seconds: thirty-seconds
        """
        self.data[3] = thirty_seconds


class KeySignatureEvent(MetaEvent):
    """
    Key Signature Event.

    """
    __slots__ = ['alternatives', 'minor']
    name = 'Key Signature'
    meta_command = 0x59
    length = 2

    @property
    def alternatives(self):
        """
        Alternatives.

        """
        return self.data[0] - 256 if self.data[0] > 127 else self.data[0]

    @alternatives.setter
    def alternatives(self, alternatives):
        """
        Set alternatives.

        :param alternatives: alternatives

        """
        self.data[0] = 256 + alternatives if alternatives < 0 else alternatives

    @property
    def minor(self):
        """
        Major / minor.

        """
        return self.data[1]

    @minor.setter
    def minor(self, val):
        """
        Major / minor.

        :param val: value

        """
        self.data[1] = val


class SequencerSpecificEvent(MetaEvent):
    """
    Sequencer Specific Event.

    """
    name = 'Sequencer Specific'
    meta_command = 0x7F


# MIDI Track
class MidiTrack(object):
    """
    MIDI Track.

    """
    def __init__(self):
        """
        Instantiate a new MIDI track object instance.

        """
        self.events = []
        self._relative_timing = True
        self._status_msg = None  # needed for correct SysEx event handling

    def make_ticks_abs(self):
        """
        Make the track's timing information absolute.

        """
        if self._relative_timing:
            running_tick = 0
            for event in self.events:
                event.tick += running_tick
                running_tick = event.tick
            self._relative_timing = False

    def make_ticks_rel(self):
        """
        Make the track's timing information relative.

        """
        if not self._relative_timing:
            running_tick = 0
            for event in self.events:
                event.tick -= running_tick
                running_tick += event.tick
            self._relative_timing = True

    def read(self, midi_file):
        """
        Read the MIDI track data from a file.

        :param midi_file: open file handle
        :return:          the MidiTrack object

        """
        # reset the status
        self._status_msg = None
        # first four bytes are Track header
        chunk = midi_file.read(4)
        if chunk != 'MTrk':
            raise TypeError("Bad track header in MIDI file: %s" % chunk)
        # next four bytes are track size
        track_size = struct.unpack(">L", midi_file.read(4))[0]
        track_data = iter(midi_file.read(track_size))
        # read in all events
        while True:
            try:
                # event, last_status_msg = _parse_midi_event(track_data,
                #                                            last_status_msg)
                # # append the events to the event list
                # self.events.append(event)
                # first datum is variable length representing the delta-time
                tick = read_variable_length(track_data)
                # next byte is status message
                status_msg = ord(track_data.next())
                # is the event a MetaEvent?
                if MetaEvent.is_event(status_msg):
                    cmd = ord(track_data.next())
                    if cmd not in EventRegistry.MetaEvents:
                        raise Warning("Unknown Meta MIDI Event: %s" % cmd)
                    cls = EventRegistry.MetaEvents[cmd]
                    data_len = read_variable_length(track_data)
                    data = [ord(track_data.next()) for _ in range(data_len)]
                    # create an event and append it to the list
                    self.events.append(cls(tick=tick, data=data))
                # is this event a SysEx Event?
                elif SysExEvent.is_event(status_msg):
                    data = []
                    while True:
                        datum = ord(track_data.next())
                        if datum == 0xF7:
                            break
                        data.append(datum)
                    # create an event and append it to the list
                    self.events.append(SysExEvent(tick=tick, data=data))
                # not a meta or SysEx event, must be a general MIDI event
                else:
                    key = status_msg & 0xF0
                    if key not in EventRegistry.Events:

                        assert self._status_msg, "Bad byte value"
                        data = []
                        key = self._status_msg & 0xF0
                        cls = EventRegistry.Events[key]
                        channel = self._status_msg & 0x0F
                        data.append(status_msg)
                        data += [ord(track_data.next()) for _ in
                                 range(cls.length - 1)]
                        # create an event and append it to the list
                        self.events.append(cls(tick=tick, channel=channel,
                                               data=data))
                    else:
                        self._status_msg = status_msg
                        cls = EventRegistry.Events[key]
                        channel = self._status_msg & 0x0F
                        data = [ord(track_data.next()) for _ in
                                range(cls.length)]
                        # create an event and append it to the list
                        self.events.append(cls(tick=tick, channel=channel,
                                               data=data))
            # no more events to be processed
            except StopIteration:
                break
        # return the track
        return self

    def write(self, midi_file):
        """
        Write the MIDI track to file

        :param midi_file: open file handle

        """
        # first make sure the timing information is relative
        self.make_ticks_rel()
        # and the last status message is unset
        self._status_msg = None
        # then encode all events of the track
        track_data = ''
        for event in self.events:
            # encode the event data, first the timing information
            track_data += write_variable_length(event.tick)
            # is the event a MetaEvent?
            if isinstance(event, MetaEvent):
                track_data += chr(event.status_msg)
                track_data += chr(event.meta_command)
                track_data += write_variable_length(len(event.data))
                track_data += str.join('', map(chr, event.data))
            # is this event a SysEx Event?
            elif isinstance(event, SysExEvent):
                track_data += chr(0xF0)
                track_data += str.join('', map(chr, event.data))
                track_data += chr(0xF7)
            # not a meta or SysEx event, must be a general message
            elif isinstance(event, Event):
                if not self._status_msg or \
                        self._status_msg.status_msg != event.status_msg or \
                        self._status_msg.channel != event.channel:
                    self._status_msg = event
                    track_data += chr(event.status_msg | event.channel)
                track_data += str.join('', map(chr, event.data))
            else:
                raise ValueError("Unknown MIDI Event: " + str(event))
        # prepend track header
        track_header = 'MTrk%s' % struct.pack(">L", len(track_data))
        track_data = track_header + track_data
        # and write the track
        midi_file.write(track_data)


# File I/O classes
class MidiFile(object):
    """
    MIDI File.

    """
    def __init__(self, resolution=960, format=1):
        # init variables
        self.format = format
        self.resolution = resolution
        self.tracks = []

    def make_ticks_abs(self):
        """
        Make the timing information of all tracks absolute.

        """
        for track in self.tracks:
            track.make_ticks_abs()

    def make_ticks_rel(self):
        """
        Make the timing information of all tracks relative.

        """
        for track in self.tracks:
            track.make_ticks_rel()

    def read(self, midi_file):
        """
        Read in a midi file.

        :param midi_file: the midi file name

        """
        close_file = False
        # open file if needed
        if isinstance(midi_file, basestring):
            midi_file = open(midi_file, 'rb')
            close_file = True
        try:
            # read in file header
            # self.format, num_tracks, self.resolution = parse_midi_header(midi_file)
            # first four bytes are MIDI header
            chunk = midi_file.read(4)
            if chunk != 'MThd':
                raise TypeError("Bad header in MIDI file.")
            # next four bytes are header size
            # next two bytes specify the format version
            # next two bytes specify the number of tracks
            # next two bytes specify the resolution/PPQ/Parts Per Quarter
            # (in other words, how many ticks per quarter note)
            data = struct.unpack(">LHHH", midi_file.read(10))
            header_size = data[0]
            self.format = data[1]
            num_tracks = data[2]
            self.resolution = data[3]
            # skip the remaining part of the header
            if header_size > DEFAULT_MIDI_HEADER_SIZE:
                midi_file.read(header_size - DEFAULT_MIDI_HEADER_SIZE)
            # read in all tracks
            for _ in range(num_tracks):
                # read in one track and append it to the tracks list
                track = MidiTrack().read(midi_file)
                self.tracks.append(track)
        finally:
            # close file if needed
            if close_file:
                midi_file.close()
        # return the object
        return self

    # methods for writing MIDI stuff
    def write(self, midi_file):
        """
        Write a midi file.

        :param midi_file: the MIDI file handle

        """
        close_file = False
        # open file if needed
        if isinstance(midi_file, basestring):
            midi_file = open(midi_file, 'wb')
            close_file = True
        try:
            # write a MIDI header
            header_data = struct.pack(">LHHH", 6, self.format,
                                      len(self.tracks), self.resolution)
            midi_file.write('MThd%s' % header_data)
            # write all tracks
            for track in self.tracks:
                # write each track to file
                track.write(midi_file)
        # close file if needed
        finally:
            if close_file:
                midi_file.close()
