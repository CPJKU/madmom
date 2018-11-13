# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-few-public-methods
"""
This module contains MIDI functionality, but is deprecated as of version 0.16.
Please use madmom.io.midi instead. This module will be removed in version 0.18.

Almost all code is taken from Giles Hall's python-midi package:
https://github.com/vishnubob/python-midi

It combines the complete package in a single file, to make it easier to
distribute. Most notable changes are `MIDITrack` and `MIDIFile` classes which
handle all data i/o and provide a interface which allows to read/display all
notes as simple numpy arrays. Also, the EventRegistry is handled differently.

The last merged commit is 3053fefe.

Since then the following commits have been added functionality-wise:

- 0964c0b (prevent multiple tick conversions)
- c43bf37 (add pitch and value properties to AfterTouchEvent)
- 40111c6 (add 0x08 MetaEvent: ProgramNameEvent)
- 43de818 (handle unknown MIDI meta events gracefully)

Additionally, the module has been updated to work with Python3.

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

from __future__ import absolute_import, division, print_function

import sys
import math
import struct
import numpy as np
import warnings


# constants
OCTAVE_MAX_VALUE = 12
OCTAVE_VALUES = list(range(OCTAVE_MAX_VALUE))

NOTE_NAMES = ['C', 'Cs', 'D', 'Ds', 'E', 'F', 'Fs', 'G', 'Gs', 'A', 'As', 'B']
WHITE_KEYS = [0, 2, 4, 5, 7, 9, 11]
BLACK_KEYS = [1, 3, 6, 8, 10]
NOTE_PER_OCTAVE = len(NOTE_NAMES)
NOTE_VALUES = list(range(OCTAVE_MAX_VALUE * NOTE_PER_OCTAVE))
NOTE_NAME_MAP_FLAT = {}
NOTE_VALUE_MAP_FLAT = []
NOTE_NAME_MAP_SHARP = {}
NOTE_VALUE_MAP_SHARP = []

for index in range(128):
    note_idx = index % NOTE_PER_OCTAVE
    oct_idx = index / OCTAVE_MAX_VALUE
    note_name = NOTE_NAMES[note_idx]
    if len(note_name) == 2:
        # sharp note
        flat = NOTE_NAMES[note_idx + 1] + 'b'
        NOTE_NAME_MAP_FLAT['%s_%d' % (flat, oct_idx)] = index
        NOTE_NAME_MAP_SHARP['%s_%d' % (note_name, oct_idx)] = index
        NOTE_VALUE_MAP_FLAT.append('%s_%d' % (flat, oct_idx))
        NOTE_VALUE_MAP_SHARP.append('%s_%d' % (note_name, oct_idx))
        globals()['%s_%d' % (note_name[0] + 's', oct_idx)] = index
        globals()['%s_%d' % (flat, oct_idx)] = index
    else:
        NOTE_NAME_MAP_FLAT['%s_%d' % (note_name, oct_idx)] = index
        NOTE_NAME_MAP_SHARP['%s_%d' % (note_name, oct_idx)] = index
        NOTE_VALUE_MAP_FLAT.append('%s_%d' % (note_name, oct_idx))
        NOTE_VALUE_MAP_SHARP.append('%s_%d' % (note_name, oct_idx))
        globals()['%s_%d' % (note_name, oct_idx)] = index

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

HEADER_SIZE = 14

RESOLUTION = 480  # ticks per quarter note
TEMPO = 120
TIME_SIGNATURE_NUMERATOR = 4
TIME_SIGNATURE_DENOMINATOR = 4
TIME_SIGNATURE = (TIME_SIGNATURE_NUMERATOR, TIME_SIGNATURE_DENOMINATOR)
SECONDS_PER_QUARTER_NOTE = 60. / TEMPO
SECONDS_PER_TICK = SECONDS_PER_QUARTER_NOTE / RESOLUTION

warnings.warn('Deprecated as of version 0.16. Please use madmom.io.midi '
              'instead. This module will be removed in version 0.18.')

# Ensure Python2/3 compatibility when reading bytes from MIDI files
if sys.version_info[0] == 2:
    int2byte = chr

    def byte2int(byte):
        """Convert a byte-character to an integer."""
        return ord(byte)
else:
    int2byte = struct.Struct(">B").pack

    def byte2int(byte):
        """Convert a byte-character to an integer."""
        return byte


# functions for packing / unpacking variable length data
def read_variable_length(data):
    """
    Read a variable length variable from the given data.

    Parameters
    ----------
    data : bytearray
        Data of variable length.

    Returns
    -------
    length : int
        Length in bytes.

    """
    next_byte = 1
    value = 0
    while next_byte:
        next_value = byte2int(next(data))
        # is the hi-bit set?
        if not next_value & 0x80:
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
    Write a variable length variable.

    Parameters
    ----------
    value : bytearray
        Value to be encoded as a variable of variable length.

    Returns
    -------
    bytearray
        Variable with variable length.

    """
    result = bytearray()
    result.insert(0, value & 0x7F)
    value >>= 7
    if value:
        result.insert(0, (value & 0x7F) | 0x80)
        value >>= 7
        if value:
            result.insert(0, (value & 0x7F) | 0x80)
            value >>= 7
            if value:
                result.insert(0, (value & 0x7F) | 0x80)
    return result


# class for dynamically registering event classes
class EventRegistry(object):
    """
    Class for registering Events.

    Event classes should be registered manually by calling
    EventRegistry.register_event(EventClass) after the class definition.

    Normal events are registered in the `events` dictionary and use the event's
    `status_msg` as a key; meta events are registered in the `meta_events`
    dictionary and use their `meta_command` as key.

    """
    events = {}
    meta_events = {}

    @classmethod
    def register_event(cls, event):
        """
        Registers an event in the registry.

        Parameters
        ----------
        event : :class:`Event` instance
            Event to be registered.

        """
        # normal events
        if any(b in (Event, ChannelEvent, NoteEvent) for b in event.__bases__):
            # raise an error if the event class is registered already
            if event.status_msg in cls.events:
                raise AssertionError("Event %s already registered" %
                                     event.name)
            # register the Event
            cls.events[event.status_msg] = event
        # meta events
        elif any(b in (MetaEvent, MetaEventWithText) for b in event.__bases__):
            # raise an error if the meta event class is registered already
            if event.meta_command in EventRegistry.meta_events:
                raise AssertionError("Event %s already registered" %
                                     event.name)
            # register the MetaEvent
            cls.meta_events[event.meta_command] = event
        # unknown events
        else:
            # raise an error
            raise AssertionError("Unknown base class in event type: %s" %
                                 event.__bases__)


class Event(object):
    """
    Generic MIDI Event.

    """
    name = "Generic MIDI Event"
    length = 0
    status_msg = 0x0
    # sort is a float value used for sorting events occurring at the same tick
    sort = 0.

    def __init__(self, **kwargs):
        if isinstance(self.length, int):
            data = [0] * self.length
        else:
            data = []
        self.tick = 0
        self.data = data
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def __eq__(self, other):
        return (
            self.tick == other.tick and self.data == other.data and
            self.status_msg == other.status_msg)

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        if self.tick < other.tick:
            return True
        elif self.tick == other.tick and self.sort < other.sort:
            return True
        return False

    def __le__(self, other):
        return NotImplementedError

    def __gt__(self, other):
        if self.tick > other.tick:
            return True
        elif self.tick == other.tick and self.sort > other.sort:
            return True
        return False

    def __ge__(self, other):
        return NotImplementedError

    def __str__(self):
        return "%s: tick: %s data: %s" % (
            self.__class__.__name__, self.tick, self.data)


class ChannelEvent(Event):
    """
    Event with a channel number.

    """
    name = 'ChannelEvent'

    def __init__(self, **kwargs):
        super(ChannelEvent, self).__init__(**kwargs)
        self.channel = kwargs.get('channel', 0)

    def __eq__(self, other):
        return (
            self.tick == other.tick and self.channel == other.channel and
            self.data == other.data and self.status_msg == other.status_msg)

    def __str__(self):
        return "%s: tick: %s channel: %s data: %s" % (
            self.__class__.__name__, self.tick, self.channel, self.data)


class NoteEvent(ChannelEvent):
    """
    NoteEvent is a special subclass of Event that is not meant to be used as a
    concrete class. It defines the generalities of NoteOn and NoteOff events.

    """
    length = 2

    def __str__(self):
        return "%s: tick: %s channel: %s pitch: %s velocity: %s" % (
            self.__class__.__name__, self.tick, self.channel, self.pitch,
            self.velocity)

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

        Parameters
        ----------
        pitch : int
            Pitch of the note.

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

        Parameters
        ----------
        velocity : int
            Velocity of the note.

        """
        self.data[1] = velocity


class NoteOnEvent(NoteEvent):
    """
    Note On Event.

    """
    status_msg = 0x90
    name = 'Note On'
    sort = .1  # make sure it is sorted before NoteOffEvent

EventRegistry.register_event(NoteOnEvent)


class NoteOffEvent(NoteEvent):
    """
    Note Off Event.

    """
    status_msg = 0x80
    name = 'Note Off'
    sort = .2  # make sure it is sorted after NoteOnEvent

EventRegistry.register_event(NoteOffEvent)


class AfterTouchEvent(ChannelEvent):
    """
    After Touch Event.

    """
    status_msg = 0xA0
    length = 2
    name = 'After Touch'

    def __str__(self):
        return "%s: tick: %s channel: %s pitch: %s value: %s" % (
            self.__class__.__name__, self.tick, self.channel, self.pitch,
            self.value)

    @property
    def pitch(self):
        """
        Pitch of the after touch event.

        """
        return self.data[0]

    @pitch.setter
    def pitch(self, pitch):
        """
        Set the pitch of the after touch event.

        Parameters
        ----------
        pitch : int
            Pitch of the after touch event.

        """
        self.data[0] = pitch

    @property
    def value(self):
        """
        Value of the after touch event.

        """
        return self.data[1]

    @value.setter
    def value(self, value):
        """
        Set the value of the after touch event.

        Parameters
        ----------
        value : int
            Value of the after touch event.

        """
        self.data[1] = value

EventRegistry.register_event(AfterTouchEvent)


class ControlChangeEvent(ChannelEvent):
    """
    Control Change Event.

    """
    status_msg = 0xB0
    length = 2
    name = 'Control Change'

    def __str__(self):
        return "%s: tick: %s channel: %s control: %s value: %s" % (
            self.__class__.__name__, self.tick, self.channel, self.control,
            self.value)

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

        Parameters
        ----------
        control : int
            Control ID.

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

        Parameters
        ----------
        value : int
            Value of the controller.

        """
        self.data[1] = value

EventRegistry.register_event(ControlChangeEvent)


class ProgramChangeEvent(ChannelEvent):
    """
    Program Change Event.

    """
    status_msg = 0xC0
    length = 1
    name = 'Program Change'

    def __str__(self):
        return "%s: tick: %s channel: %s value: %s" % (
            self.__class__.__name__, self.tick, self.channel, self.value)

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

        Parameters
        ----------
        value : int
            Value of the Program Change Event.

        """
        self.data[0] = value

EventRegistry.register_event(ProgramChangeEvent)


class ChannelAfterTouchEvent(ChannelEvent):
    """
    Channel After Touch Event.

    """
    status_msg = 0xD0
    length = 1
    name = 'Channel After Touch'

    def __str__(self):
        return "%s: tick: %s channel: %s value: %s" % (
            self.__class__.__name__, self.tick, self.channel, self.value)

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

        Parameters
        ----------
        value : int
            Value of the Channel After Touch Event.

        """
        self.data[0] = value

EventRegistry.register_event(ChannelAfterTouchEvent)


class PitchWheelEvent(ChannelEvent):
    """
    Pitch Wheel Event.

    """
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
        Set the pitch of the Pitch Wheel Event.

        Parameters
        ----------
        pitch : int
            Pitch of the Pitch Wheel Event.

        """
        value = pitch + 0x2000
        self.data[0] = value & 0x7F
        self.data[1] = (value >> 7) & 0x7F

EventRegistry.register_event(PitchWheelEvent)


class SysExEvent(Event):
    """
    System Exclusive Event.

    """
    status_msg = 0xF0
    length = 'variable'
    name = 'SysEx'

EventRegistry.register_event(SysExEvent)


class MetaEvent(Event):
    """
    MetaEvent is a special subclass of Event that is not meant to be used as a
    concrete class. It defines a subset of Events known as the Meta events.

    """
    status_msg = 0xFF
    meta_command = 0x0
    name = 'Meta Event'

    def __eq__(self, other):
        return (
            self.tick == other.tick and self.data == other.data and
            self.status_msg == other.status_msg and
            self.meta_command == other.meta_command)


class MetaEventWithText(MetaEvent):
    """
    Meta Event With Text.

    """
    def __init__(self, **kwargs):
        super(MetaEventWithText, self).__init__(**kwargs)
        if 'text' not in kwargs:
            self.text = ''.join(chr(datum) for datum in self.data)

    def __str__(self):
        return "%s: %s" % (self.__class__.__name__, self.text)


class SequenceNumberMetaEvent(MetaEvent):
    """
    Sequence Number Meta Event.

    """
    meta_command = 0x00
    length = 2
    name = 'Sequence Number'

EventRegistry.register_event(SequenceNumberMetaEvent)


class TextMetaEvent(MetaEventWithText):
    """
    Text Meta Event.

    """
    meta_command = 0x01
    length = 'variable'
    name = 'Text'

EventRegistry.register_event(TextMetaEvent)


class CopyrightMetaEvent(MetaEventWithText):
    """
    Copyright Meta Event.

    """
    meta_command = 0x02
    length = 'variable'
    name = 'Copyright Notice'

EventRegistry.register_event(CopyrightMetaEvent)


class TrackNameEvent(MetaEventWithText):
    """
    Track Name Event.

    """
    meta_command = 0x03
    length = 'variable'
    name = 'Track Name'

EventRegistry.register_event(TrackNameEvent)


class InstrumentNameEvent(MetaEventWithText):
    """
    Instrument Name Event.

    """
    meta_command = 0x04
    length = 'variable'
    name = 'Instrument Name'

EventRegistry.register_event(InstrumentNameEvent)


class LyricsEvent(MetaEventWithText):
    """
    Lyrics Event.

    """
    meta_command = 0x05
    length = 'variable'
    name = 'Lyrics'

EventRegistry.register_event(LyricsEvent)


class MarkerEvent(MetaEventWithText):
    """
    Marker Event.

    """
    meta_command = 0x06
    length = 'variable'
    name = 'Marker'

EventRegistry.register_event(MarkerEvent)


class CuePointEvent(MetaEventWithText):
    """
    Cue Point Event.

    """
    meta_command = 0x07
    length = 'variable'
    name = 'Cue Point'

EventRegistry.register_event(CuePointEvent)


class ProgramNameEvent(MetaEventWithText):
    """
    Program Name Event.

    """
    meta_command = 0x08
    length = 'variable'
    name = 'Program Name'

EventRegistry.register_event(ProgramNameEvent)


class UnknownMetaEvent(MetaEvent):
    """
    Unknown Meta Event.

    Parameters
    ----------
    meta_command : int
        Value of the meta command.

    """
    meta_command = None
    name = 'Unknown'

    def __init__(self, **kwargs):
        super(UnknownMetaEvent, self).__init__(**kwargs)
        # TODO: is this needed, should be handled by Event already
        self.meta_command = kwargs['meta_command']

EventRegistry.register_event(UnknownMetaEvent)


class ChannelPrefixEvent(MetaEvent):
    """
    Channel Prefix Event.

    """
    meta_command = 0x20
    length = 1
    name = 'Channel Prefix'

EventRegistry.register_event(ChannelPrefixEvent)


class PortEvent(MetaEvent):
    """
    Port Event.

    """
    meta_command = 0x21
    name = 'MIDI Port/Cable'

EventRegistry.register_event(PortEvent)


class TrackLoopEvent(MetaEvent):
    """
    Track Loop Event.

    """
    meta_command = 0x2E
    name = 'Track Loop'

EventRegistry.register_event(TrackLoopEvent)


class EndOfTrackEvent(MetaEvent):
    """
    End Of Track Event.

    """
    meta_command = 0x2F
    name = 'End of Track'
    sort = .99  # should always come last

EventRegistry.register_event(EndOfTrackEvent)


class SetTempoEvent(MetaEvent):
    """
    Set Tempo Event.

    """
    meta_command = 0x51
    length = 3
    name = 'Set Tempo'

    def __str__(self):
        return "%s: tick: %s microseconds per quarter note: %s" % (
            self.__class__.__name__, self.tick,
            self.microseconds_per_quarter_note)

    @property
    def microseconds_per_quarter_note(self):
        """
        Microseconds per quarter note.

        """
        assert len(self.data) == 3
        values = [self.data[x] << (16 - (8 * x)) for x in range(3)]
        return sum(values)

    @microseconds_per_quarter_note.setter
    def microseconds_per_quarter_note(self, microseconds):
        """
        Set microseconds per quarter note.

        Parameters
        ----------
        microseconds : int
            Microseconds per quarter note.

        """
        self.data = [(microseconds >> (16 - (8 * x)) & 0xFF) for x in range(3)]

EventRegistry.register_event(SetTempoEvent)


class SmpteOffsetEvent(MetaEvent):
    """
    SMPTE Offset Event.

    """
    meta_command = 0x54
    name = 'SMPTE Offset'

EventRegistry.register_event(SmpteOffsetEvent)


class TimeSignatureEvent(MetaEvent):
    """
    Time Signature Event.

    """
    meta_command = 0x58
    length = 4
    name = 'Time Signature'

    @property
    def numerator(self):
        """
        Numerator of the time signature.

        """
        return self.data[0]

    @numerator.setter
    def numerator(self, numerator):
        """
        Set numerator of the time signature.

        Parameters
        ----------
        numerator : int
            Numerator of the time signature.
        """
        self.data[0] = numerator

    @property
    def denominator(self):
        """
        Denominator of the time signature.

        """
        return 2 ** self.data[1]

    @denominator.setter
    def denominator(self, denominator):
        """
        Set denominator of the time signature.

        Parameters
        ----------
        denominator : int
            Denominator of the time signature.

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
        Set metronome of the time signature.

        Parameters
        ----------
        metronome : int
            Metronome of the time signature.

        """
        self.data[2] = metronome

    @property
    def thirty_seconds(self):
        """
        Thirty-seconds of the time signature.

        """
        return self.data[3]

    @thirty_seconds.setter
    def thirty_seconds(self, thirty_seconds):
        """
        Set thirty-seconds of the time signature.

        Parameters
        ----------
        thirty_seconds : int
            Thirty-seconds of the time signature.

        """
        self.data[3] = thirty_seconds

EventRegistry.register_event(TimeSignatureEvent)


class KeySignatureEvent(MetaEvent):
    """
    Key Signature Event.

    """
    meta_command = 0x59
    length = 2
    name = 'Key Signature'

    @property
    def alternatives(self):
        """
        Alternatives of the key signature.

        """
        return self.data[0] - 256 if self.data[0] > 127 else self.data[0]

    @alternatives.setter
    def alternatives(self, alternatives):
        """
        Set alternatives of the key signature.

        Parameters
        ----------
        alternatives : int
            Alternatives of the key signature.

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
        Set major / minor.

        Parameters
        ----------
        val : int
            Major / minor.

        """
        self.data[1] = val

EventRegistry.register_event(KeySignatureEvent)


class SequencerSpecificEvent(MetaEvent):
    """
    Sequencer Specific Event.

    """
    meta_command = 0x7F
    name = 'Sequencer Specific'

EventRegistry.register_event(SequencerSpecificEvent)


def _add_channel(notes, channel=0):
    """
    Adds a default channel to the notes if missing.

    Parameters
    ----------
    notes : numpy array, shape (num_notes, 2)
        Notes, one per row (column definition see notes).
    channel : int, optional
        Note channel if not defined by `notes`.

    Returns
    -------
    numpy array
        Notes (including note channel).

    Notes
    -----
    The note columns format must be (channel being optional):

    'onset' 'pitch' 'duration' 'velocity' ['channel']

    """
    if not notes.ndim == 2:
        raise ValueError('unknown format for `notes`')
    rows, columns = notes.shape
    if columns == 5:
        return notes
    elif columns == 4:
        channels = np.ones((rows, 1)) * channel
        return np.hstack((notes, channels))
    raise ValueError('unable to handle `notes` with %d columns' % columns)


# MIDI Track
class MIDITrack(object):
    """
    MIDI Track.

    Parameters
    ----------
    events : list
        MIDI events.

    Notes
    -----
    All events are stored with timing information in absolute ticks.
    The events must be sorted. Consider using `from_notes()` method.

    Examples
    --------

    Create a MIDI track from a list of events. Please note that the events must
    be sorted.

    >>> e1 = NoteOnEvent(tick=100, pitch=50, velocity=60)
    >>> e2 = NoteOffEvent(tick=300, pitch=50)
    >>> e3 = NoteOnEvent(tick=200, pitch=62, velocity=90)
    >>> e4 = NoteOffEvent(tick=600, pitch=62)
    >>> t = MIDITrack(sorted([e1, e2, e3, e4]))
    >>> t  # doctest: +ELLIPSIS
    <madmom.utils.midi.MIDITrack object at 0x...>
    >>> t.events  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [<madmom.utils.midi.NoteOnEvent object at 0x...>,
     <madmom.utils.midi.NoteOnEvent object at 0x...>,
     <madmom.utils.midi.NoteOffEvent object at 0x...>,
     <madmom.utils.midi.NoteOffEvent object at 0x...>]

    It can also be created from an array containing the notes. The `from_notes`
    method also takes care of creating tempo and time signature events.

    >>> notes = np.array([[0.1, 50, 0.3, 60], [0.2, 62, 0.4, 90]])
    >>> t = MIDITrack.from_notes(notes)
    >>> t  # doctest: +ELLIPSIS
    <madmom.utils.midi.MIDITrack object at 0x...>
    >>> t.events  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [<madmom.utils.midi.SetTempoEvent object at 0x...>,
     <madmom.utils.midi.TimeSignatureEvent object at 0...>,
     <madmom.utils.midi.NoteOnEvent object at 0x...>,
     <madmom.utils.midi.NoteOnEvent object at 0x...>,
     <madmom.utils.midi.NoteOffEvent object at 0x...>,
     <madmom.utils.midi.NoteOffEvent object at 0x...>]

    """

    def __init__(self, events=None):
        if events is None:
            self.events = []
        else:
            # do not sort the events, since they can have relative timing!
            self.events = events

    def _make_ticks_abs(self):
        """Make the track's events timing information absolute."""
        running_tick = 0
        for event in self.events:
            event.tick += running_tick
            running_tick = event.tick

    def _make_ticks_rel(self):
        """Make the track's events timing information relative."""
        running_tick = 0
        for event in self.events:
            event.tick -= running_tick
            running_tick += event.tick

    @property
    def data_stream(self):
        """
        MIDI data stream representation of the track.

        """
        # sort the events
        self.events.sort()
        # first make sure the timing information is relative
        self._make_ticks_rel()
        # and unset the status message
        status = None
        # then encode all events of the track
        track_data = bytearray()
        for event in self.events:
            # encode the event data, first the timing information
            track_data.extend(write_variable_length(event.tick))
            # is the event a MetaEvent?
            if isinstance(event, MetaEvent):
                track_data.append(event.status_msg)
                track_data.append(event.meta_command)
                track_data.extend(write_variable_length(len(event.data)))
                track_data.extend(event.data)
            # is this event a SysEx Event?
            elif isinstance(event, SysExEvent):
                track_data.append(0xF0)
                track_data.extend(event.data)
                track_data.append(0xF7)
            # not a meta or SysEx event, must be a general message
            elif isinstance(event, Event):
                if not status or status.status_msg != event.status_msg or \
                        status.channel != event.channel:
                    status = event
                    track_data.append(event.status_msg | event.channel)
                track_data.extend(event.data)
            else:
                raise ValueError("Unknown MIDI Event: " + str(event))
        # TODO: should we add a EndOfTrackEvent?
        # convert events back to absolute ticks
        self._make_ticks_abs()
        # prepare the data
        data = bytearray()
        # generate a MIDI header
        data.extend(b'MTrk')
        data.extend(struct.pack(">L", len(track_data)))
        # append the track data
        data.extend(track_data)
        # return the track data
        return data

    @classmethod
    def from_stream(cls, midi_stream):
        """
        Create a MIDI track by reading the data from a stream.

        Parameters
        ----------
        midi_stream : open file handle
            MIDI file stream (e.g. open MIDI file handle)

        Returns
        -------
        :class:`MIDITrack` instance
            :class:`MIDITrack` instance

        """
        events = []
        # reset the status
        status = None
        # first four bytes are Track header
        chunk = midi_stream.read(4)
        if chunk != b'MTrk':
            raise TypeError("Bad track header in MIDI file: %s" % chunk)
        # next four bytes are track size
        track_size = struct.unpack(">L", midi_stream.read(4))[0]
        track_data = iter(midi_stream.read(track_size))
        # read in all events
        while True:
            try:
                # first datum is variable length representing the delta-time
                tick = read_variable_length(track_data)
                # next byte is status message
                status_msg = byte2int(next(track_data))
                # is the event a MetaEvent?
                if MetaEvent.status_msg == status_msg:
                    meta_cmd = byte2int(next(track_data))
                    if meta_cmd not in EventRegistry.meta_events:
                        import warnings
                        warnings.warn("Unknown Meta MIDI Event: %s" % meta_cmd)
                        event_cls = UnknownMetaEvent
                    else:
                        event_cls = EventRegistry.meta_events[meta_cmd]
                    data_len = read_variable_length(track_data)
                    data = [byte2int(next(track_data)) for _ in
                            range(data_len)]
                    # create an event and append it to the list
                    events.append(event_cls(tick=tick, data=data,
                                            meta_command=meta_cmd))
                # is this event a SysEx Event?
                elif SysExEvent.status_msg == status_msg:
                    data = []
                    while True:
                        datum = byte2int(next(track_data))
                        if datum == 0xF7:
                            break
                        data.append(datum)
                    # create an event and append it to the list
                    events.append(SysExEvent(tick=tick, data=data))
                # not a meta or SysEx event, must be a general MIDI event
                else:
                    key = status_msg & 0xF0
                    if key not in EventRegistry.events:
                        assert status, "Bad byte value"
                        data = []
                        key = status & 0xF0
                        event_cls = EventRegistry.events[key]
                        channel = status & 0x0F
                        data.append(status_msg)
                        data += [byte2int(next(track_data)) for _ in
                                 range(event_cls.length - 1)]
                        # create an event and append it to the list
                        events.append(event_cls(tick=tick, channel=channel,
                                                data=data))
                    else:
                        status = status_msg
                        event_cls = EventRegistry.events[key]
                        channel = status & 0x0F
                        data = [byte2int(next(track_data)) for _ in
                                range(event_cls.length)]
                        # create an event and append it to the list
                        events.append(event_cls(tick=tick, channel=channel,
                                                data=data))
            # no more events to be processed
            except StopIteration:
                break
        # create a new track
        track = cls(events)
        # make the timing of the events (i.e. the ticks) absolute
        track._make_ticks_abs()
        # return this track
        return track

    @classmethod
    def from_notes(cls, notes, tempo=TEMPO, time_signature=TIME_SIGNATURE,
                   resolution=RESOLUTION):
        """
        Create a MIDI track from the given notes.

        Parameters
        ----------
        notes : numpy array
            Array with the notes, one per row. The columns must be:
            (onset time, pitch, duration, velocity, [channel]).
        tempo : float, optional
            Tempo of the MIDI track, given in beats per minute (bpm).
        time_signature : tuple, optional
            Time signature of the track, e.g. (4, 4) for 4/4.
        resolution : int
            Resolution (i.e. ticks per quarter note) of the MIDI track.

        Returns
        -------
        :class:`MIDITrack` instance
            :class:`MIDITrack` instance

        Notes
        -----
        All events including the generated tempo and time signature events is
        included in the returned track (i.e. as defined in MIDI format 0).

        """
        # add a default channel if needed
        notes = _add_channel(notes)

        # set time signature
        sig = TimeSignatureEvent(tick=0)
        sig.numerator, sig.denominator = time_signature

        # length of a quarter note (seconds)
        quarter_note_length = 60. / tempo * sig.denominator / 4
        # quarter notes per second
        quarter_notes_per_second = 1 / quarter_note_length
        # ticks per second
        ticks_per_second = resolution * quarter_notes_per_second

        # set tempo
        tempo = SetTempoEvent(tick=0)
        tempo.microseconds_per_quarter_note = int(quarter_note_length * 1e6)

        # list for events (ticks in absolute timing)
        events = []

        # add the notes
        for note in notes:
            onset, pitch, duration, velocity, channel = note
            # add NoteOn
            e_on = NoteOnEvent()
            e_on.tick = int(onset * ticks_per_second)
            e_on.pitch = int(pitch)
            e_on.velocity = int(velocity)
            e_on.channel = int(channel)
            # and NoteOff
            e_off = NoteOffEvent()
            e_off.tick = int((onset + duration) * ticks_per_second)
            e_off.pitch = int(pitch)
            e_off.channel = int(channel)
            events.append(e_on)
            events.append(e_off)
        # sort the events and prepend the tempo and time signature events
        events = sorted(events)
        events.insert(0, sig)
        events.insert(0, tempo)
        # create a track from the events
        return cls(events)


# File I/O classes
class MIDIFile(object):
    """
    MIDI File.

    Parameters
    ----------
    tracks : list
        List of :class:`MIDITrack` instances.
    resolution : int, optional
        Resolution (i.e. microseconds per quarter note).
    file_format : int, optional
        Format of the MIDI file.

    Notes
    -----
    Writing a MIDI file assumes a tempo of 120 beats per minute (bpm) and a 4/4
    time signature and writes all events into a single track (i.e. MIDI format
    0).

    Examples
    --------
    Create a MIDI file from an array with notes. The format of the note array
    is: 'onset time', 'pitch', 'duration', 'velocity', 'channel'. The last
    column can be omitted, assuming channel 0.

    >>> notes = np.array([[0, 50, 1, 60], [0.5, 62, 0.5, 90]])
    >>> m = MIDIFile.from_notes(notes)
    >>> m  # doctest: +ELLIPSIS
    <madmom.utils.midi.MIDIFile object at 0x...>

    The notes can be accessed as a numpy array in various formats (default is
    seconds):

    >>> m.notes()
    array([[ 0. , 50. ,  1. , 60. ,  0. ],
           [ 0.5, 62. ,  0.5, 90. ,  0. ]])
    >>> m.notes(unit='ticks')
    array([[  0.,  50., 960.,  60.,   0.],
           [480.,  62., 480.,  90.,   0.]])
    >>> m.notes(unit='beats')
    array([[ 0., 50.,  2., 60.,  0.],
           [ 1., 62.,  1., 90.,  0.]])

    >>> m = MIDIFile.from_notes(notes, tempo=60)
    >>> m.notes(unit='ticks')
    array([[  0.,  50., 480.,  60.,   0.],
           [240.,  62., 240.,  90.,   0.]])
    >>> m.notes(unit='beats')
    array([[ 0. , 50. ,  1. , 60. ,  0. ],
           [ 0.5, 62. ,  0.5, 90. ,  0. ]])

    >>> m = MIDIFile.from_notes(notes, tempo=60, time_signature=(2, 2))
    >>> m.notes(unit='ticks')
    array([[  0.,  50., 960.,  60.,   0.],
           [480.,  62., 480.,  90.,   0.]])
    >>> m.notes(unit='beats')
    array([[ 0. , 50. ,  1. , 60. ,  0. ],
           [ 0.5, 62. ,  0.5, 90. ,  0. ]])

    >>> m = MIDIFile.from_notes(notes, tempo=240, time_signature=(3, 8))
    >>> m.notes(unit='ticks')
    array([[  0.,  50., 960.,  60.,   0.],
           [480.,  62., 480.,  90.,   0.]])
    >>> m.notes(unit='beats')
    array([[ 0., 50.,  4., 60.,  0.],
           [ 2., 62.,  2., 90.,  0.]])

    """

    def __init__(self, tracks=None, resolution=RESOLUTION, file_format=0):
        # init variables
        if tracks is None:
            self.tracks = []
        elif isinstance(tracks, MIDITrack):
            self.tracks = [tracks]
        elif isinstance(tracks, list):
            # TODO: test if the items of the list are of type MIDITrack
            self.tracks = tracks
        else:
            raise ValueError('file_format of `tracks` not supported.')
        self.resolution = resolution  # i.e. ticks per quarter note
        # format 0 stores all information in 1 track
        # format 1 has multiple tracks but plays them back simultaneously
        # TODO: format 2 has multiple tracks but plays them back one after
        #       another. This type is not supported (yet).
        if file_format > 1:
            raise ValueError('Only MIDI file formats 0 and 1 supported.')
        self.format = file_format

    @property
    def ticks_per_quarter_note(self):
        """
        Number of ticks per quarter note.

        """
        return self.resolution

    def tempi(self, suppress_warnings=False):
        """
        Tempi of the MIDI file.

        Returns
        -------
        tempi : numpy array
            Array with tempi (tick, seconds per tick, cumulative time).

        """
        if not suppress_warnings:
            import warnings
            warnings.warn('this method will be removed soon, do not rely on '
                          'its output, rather fix issue #192 ;)')
        # create an empty tempo list
        tempo_events = []
        for i, track in enumerate(self.tracks):
            # get a list with tempo events
            track_tempo_events = [e for e in track.events if
                                  isinstance(e, SetTempoEvent)]
            # tempo events should be only in the first track of a MIDI file
            if track_tempo_events and i > 0:
                raise ValueError('SetTempoEvents should be only in the first '
                                 'track of a MIDI file.')
            tempo_events.extend(track_tempo_events)

        # convert to desired format (tick, microseconds per tick)
        tempi = [(e.tick, e.microseconds_per_quarter_note /
                  (1e6 * self.resolution)) for e in tempo_events]
        # make sure a tempo is set and the first tempo occurs at tick 0
        if not tempi or tempi[0][0] > 0:
            tempi.insert(0, (0, SECONDS_PER_TICK))
        # sort (just to be sure)
        tempi.sort()
        # re-iterate over the list to calculate the cumulative time
        for i, _ in enumerate(tempi):
            if i == 0:
                tempi[i] = (tempi[i][0], tempi[i][1], 0)
            else:
                ticks = tempi[i][0] - tempi[i - 1][0]
                cum_time = tempi[i - 1][2] + ticks * tempi[i - 1][1]
                tempi[i] = (tempi[i][0], tempi[i][1], cum_time)
        # return tempo
        return np.asarray(tempi, np.float)

    def time_signatures(self, suppress_warnings=False):
        """
        Time signatures of the MIDI file.

        Returns
        -------
        time_signatures : numpy array
            Array with time signatures (tick, numerator, denominator).

        """
        if not suppress_warnings:
            import warnings
            warnings.warn('this method will be removed soon, do not rely on '
                          'its output, rather fix issue #192 ;)')
        signatures = None
        for track in self.tracks:
            # get a list with time signature events
            time_signature_events = [e for e in track.events if
                                     isinstance(e, TimeSignatureEvent)]
            if signatures is None and len(time_signature_events) > 0:
                # convert to desired format
                signatures = [(e.tick, e.numerator, e.denominator)
                              for e in time_signature_events]
            elif signatures is not None and len(time_signature_events) > 0:
                # time signature events should be contained only in the first
                # track of a MIDI file, thus raise an error
                raise ValueError('TimeSignatureEvent should be only in the '
                                 'first track of a MIDI file.')
        # make sure a time signature is set and the first one occurs at tick 0
        if signatures is None:
            signatures = [(0, TIME_SIGNATURE)]
        if signatures[0][0] > 0:
            signatures.insert(0, (0, TIME_SIGNATURE))
        # return time signatures
        return np.asarray(signatures, dtype=np.float)

    def notes(self, unit='s'):
        """
        Notes of the MIDI file.

        Parameters
        ----------
        unit : {'s', 'seconds', 'b', 'beats', 't', 'ticks'}
            Time unit for notes, seconds ('s') beats ('b') or ticks ('t')

        Returns
        -------
        notes : numpy array
            Array with notes (onset time, pitch, duration, velocity, channel).

        """
        # list for all notes
        notes = []
        # dictionary for storing the last onset and velocity for each
        # individual note (i.e. same pitch and channel)
        sounding_notes = {}

        # as key for the dict use channel * 128 (max number of pitches) + pitch
        def note_hash(channel, pitch):
            """Generate a note hash."""
            return channel * 128 + pitch

        for track in self.tracks:
            # get a list with note events
            note_events = [e for e in track.events if isinstance(e, NoteEvent)]
            # process all events
            tick = 0
            for e in note_events:
                if tick > e.tick:
                    raise AssertionError('note events must be sorted!')
                n = note_hash(e.channel, e.pitch)
                is_note_on = isinstance(e, NoteOnEvent)
                is_note_off = isinstance(e, NoteOffEvent)
                # if it's a note on event with a velocity > 0,
                if is_note_on and e.velocity > 0:
                    # save the onset time and velocity
                    sounding_notes[n] = (e.tick, e.velocity)
                # if it's a note off event or a note on with a velocity of 0,
                elif is_note_off or (is_note_on and e.velocity == 0):
                    if n not in sounding_notes:
                        import warnings
                        warnings.warn("ignoring %s" % e)
                        continue
                    if sounding_notes[n][0] > e.tick:
                        raise AssertionError('note duration must be positive')
                    if sounding_notes[n][1] <= 0:
                        raise AssertionError('note velocity must be positive')
                    # append the note to the list
                    notes.append((sounding_notes[n][0], e.pitch,
                                  e.tick - sounding_notes[n][0],
                                  sounding_notes[n][1], e.channel))
                    # remove hash from dict
                    del sounding_notes[n]
                else:
                    raise TypeError('unexpected NoteEvent', e)
                tick = e.tick

        # sort the notes and convert to numpy array
        notes = np.asarray(sorted(notes), dtype=np.float)

        # convert onset times and durations from ticks to the requested unit
        # and return the notes
        if unit.lower() in ('t', 'ticks'):
            return notes
        elif unit.lower() in ('s', 'seconds'):
            return self._notes_in_seconds(notes)
        elif unit.lower() in ('b', 'beats'):
            return self._notes_in_beats(notes)
        else:
            raise ValueError("`unit` must be either 'seconds', 's', 'beats', "
                             "'b', 'ticks', or 't' not %s." % unit)

    def _notes_in_beats(self, notes):
        """
        Converts onsets and offsets of notes from ticks to beats.

        Parameters
        ----------
        notes : numpy array or list of tuples
            Notes (onset, pitch, offset, velocity).

        Returns
        -------
        notes : numpy array
            Notes with onsets and offsets in beats.

        """
        tpq = self.ticks_per_quarter_note
        time_signatures = self.time_signatures(suppress_warnings=True)

        # change the second column of time_signatures to beat position of the
        # signature change, the first column is now the tick position, the
        # second column the beat position and the third column the new beat
        # unit after the signature change
        time_signatures[0, 1] = 0

        # quarter notes between time signature changes
        qnbtsc = np.diff(time_signatures[:, 0]) / tpq
        # beats between time signature changes
        bbtsc = qnbtsc * (time_signatures[:-1, 2] / 4.0)
        # compute beat position of each time signature change
        time_signatures[1:, 1] = bbtsc.cumsum()

        # iterate over all notes
        for note in notes:
            onset, _, offset, _, _ = note
            # get info about last time signature change
            tsc = time_signatures[np.argmax(time_signatures[:, 0] > onset) - 1]
            # adjust onset
            onset_ticks_since_tsc = onset - tsc[0]
            note[0] = tsc[1] + (onset_ticks_since_tsc / tpq) * (tsc[2] / 4.)
            # adjust offsets
            offset_ticks_since_tsc = offset - tsc[0]
            note[2] = tsc[1] + (offset_ticks_since_tsc / tpq) * (tsc[2] / 4.)
        # return notes
        return notes

    def _notes_in_seconds(self, notes):
        """
        Converts onsets and offsets of notes from ticks to seconds.

        Parameters
        ----------
        notes : numpy array or list of tuples
            Notes (onset, pitch, offset, velocity).

        Returns
        -------
        notes : numpy array
            Notes with onset and offset times in seconds.

        """
        # cache tempo
        tempi = self.tempi(suppress_warnings=True)
        # iterate over all notes
        for note in notes:
            onset, _, offset, _, _ = note
            # get last tempo for the onset and offset
            t_on = tempi[np.argmax(tempi[:, 0] > onset) - 1]
            t_off = tempi[np.argmax(tempi[:, 0] > offset) - 1]
            # adjust the note onset and offset
            note[0] = (onset - t_on[0]) * t_on[1] + t_on[2]
            note[2] = (offset - t_off[0]) * t_off[1] + t_off[2]
        # return notes
        return notes

    # methods for writing MIDI stuff
    @property
    def data_stream(self):
        """
        MIDI data stream representation of the MIDI file.

        """
        # prepare data
        data = bytearray()
        # generate a MIDI header
        data.extend(b'MThd')
        data.extend(struct.pack(">LHHH", 6, self.format, len(self.tracks),
                                self.resolution))
        # append the tracks
        for track in self.tracks:
            data.extend(track.data_stream)
        # return the data
        return data

    def write(self, midi_file):
        """
        Write a MIDI file.

        Parameters
        ----------
        midi_file : str
            The MIDI file name.

        """
        # if we get a filename, open the file
        if not hasattr(midi_file, 'write'):
            midi_file = open(midi_file, 'wb')
        # write the MIDI stream
        midi_file.write(self.data_stream)
        # close the file
        midi_file.close()

    @classmethod
    def from_file(cls, midi_file):
        """
        Create a MIDI file instance from a .mid file.

        Parameters
        ----------
        midi_file : str
            Name of the .mid file to load.

        Returns
        -------
        :class:`MIDIFile` instance
            :class:`MIDIFile` instance

        """
        tracks = []
        resolution = None
        midi_format = None
        with open(midi_file, 'rb') as midi_file:
            # read in file header
            # first four bytes are MIDI header
            chunk = midi_file.read(4)
            if chunk != b'MThd':
                raise TypeError("Bad header in MIDI file: %s", chunk)
            # next four bytes are header size
            # next two bytes specify the format version
            # next two bytes specify the number of tracks
            # next two bytes specify the resolution/PPQ/Parts Per Quarter
            # (in other words, how many ticks per quarter note)
            data = struct.unpack(">LHHH", midi_file.read(10))
            header_size = data[0]
            midi_format = data[1]
            num_tracks = data[2]
            resolution = data[3]
            # if the top bit of the resolution word is 0, the following 15 bits
            # describe the time division in ticks per beat
            if resolution & 0x8000 == 0:
                resolution = resolution
            # otherwise the following 15 bits describe the time division in
            # frames per second
            else:
                # from http://www.sonicspot.com/guide/midifiles.html:
                # Frames per second is defined by breaking the remaining 15
                # bytes into two values. The top 7 bits (bit mask 0x7F00)
                # define a value for the number of SMPTE frames and can be
                # 24, 25, 29 (for 29.97 fps) or 30. The remaining byte
                # (bit mask 0x00FF) defines how many clock ticks or track delta
                # positions there are per frame. So a time division example of
                # 0x9978 could be broken down into it's three parts: the top
                # bit is one, so it is in SMPTE frames per second format, the
                # following 7 bits have a value of 25 (0x19) and the bottom
                # byte has a value of 120 (0x78). This means the example plays
                # at 24(?) frames per second SMPTE time and has 120 ticks per
                # frame.
                raise NotImplementedError("SMPTE resolution not implemented.")
            # skip the remaining part of the header
            if header_size > HEADER_SIZE:
                midi_file.read(header_size - HEADER_SIZE)
            # read in all tracks
            for _ in range(num_tracks):
                # read in one track and append it to the tracks list
                track = MIDITrack.from_stream(midi_file)
                tracks.append(track)
        if resolution is None or midi_format is None:
            raise IOError('unable to read MIDI file %s.' % midi_file)
        # return a newly created object
        return cls(tracks=tracks, resolution=resolution,
                   file_format=midi_format)

    @classmethod
    def from_notes(cls, notes, tempo=TEMPO, time_signature=TIME_SIGNATURE,
                   resolution=RESOLUTION):
        """
        Create a MIDIFile from the given notes.

        Parameters
        ----------
        notes : numpy array
            Array with the notes, one per row. The columns must be:
            (onset time, pitch, duration, velocity, [channel]).
        tempo : float, optional
            Tempo of the MIDI track, given in beats per minute (bpm).
        time_signature : tuple, optional
            Time signature of the track, e.g. (4, 4) for 4/4.
        resolution : int
            Resolution (i.e. ticks per quarter note) of the MIDI track.

        Returns
        -------
        :class:`MIDIFile` instance
            :class:`MIDIFile` instance with all notes collected in one track.

        Notes
        -----
        All note events (including the generated tempo and time signature
        events) are written into a single track (i.e. MIDI file format 0).

        """
        # create a new track from the notes and then a MIDIFile instance
        return cls(MIDITrack.from_notes(notes, tempo, time_signature,
                                        resolution))

    @staticmethod
    def add_arguments(parser, length=None, velocity=None, channel=None):
        """
        Add MIDI related arguments to an existing parser object.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        length : float, optional
            Default length of the notes [seconds].
        velocity : int, optional
            Default velocity of the notes.
        channel : int, optional
            Default channel of the notes.

        Returns
        -------
        argparse argument group
            MIDI argument parser group object.

        """
        # add MIDI related options to the existing parser
        g = parser.add_argument_group('MIDI arguments')
        g.add_argument('--midi', action='store_true', help='save as MIDI')
        if length is not None:
            g.add_argument('--note_length', action='store', type=float,
                           default=length,
                           help='set the note length [default=%(default).2f]')
        if velocity is not None:
            g.add_argument('--note_velocity', action='store', type=int,
                           default=velocity,
                           help='set the note velocity [default=%(default)i]')
        if channel is not None:
            g.add_argument('--note_channel', action='store', type=int,
                           default=channel,
                           help='set the note channel [default=%(default)i]')
        # return the argument group so it can be modified if needed
        return g


def process_notes(data, output=None):
    """
    This is a simple processing function. It either loads the notes from a MIDI
    file and or writes the notes to a file.

    The behaviour depends on the presence of the `output` argument, if 'None'
    is given, the notes are read, otherwise the notes are written to file.

    Parameters
    ----------
    data : str or numpy array
        MIDI file to be loaded (if `output` is 'None') / notes to be written.
    output : str, optional
        Output file name. If set, the notes given by `data` are written.

    Returns
    -------
    notes : numpy array
        Notes read/written.

    """
    if output is None:
        # load the notes
        return MIDIFile.from_file(data).notes()
    MIDIFile.from_notes(data).write(output)
    return data
