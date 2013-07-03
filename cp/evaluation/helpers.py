#!/usr/bin/env python
# encoding: utf-8
"""
Copyright (c) 2012-2013 Sebastian Böck <sebastian.boeck@jku.at>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""


def load_events(filename):
    """
    Load a list of events from file.

    :param filename: name of the file
    :return: list of events

    """
    # array for events
    events = []
    # try to read in the onsets from the file
    with open(filename, 'rb') as f:
        # read in each line of the file
        for line in f:
            # append the event (1st column) to the list, ignore the rest
            events.append(float(line.split()[0]))
    # return
    return events


def write_events(events, filename):
    """
    Write the detected onsets to the given file.

    :param events: list of events [seconds]
    :param filename: output file name

    """
    with open(filename, 'w') as f:
        for e in events:
            f.write(str(e) + '\n')


def combine_events(events, delta):
    """
    Combine all events within a certain range.

    :param events: list of events [seconds]
    :param delta: combination length [seconds]
    :return: list of combined events

    """
    # sort the events
    events.sort()
    events_length = len(events)
    events_index = 0
    # array for combined events
    comb = []
    # iterate over all events
    while events_index < events_length - 1:
        # get the first event
        first = events[events_index]
        # always increase the events index
        events_index += 1
        # get the second event
        second = events[events_index]
        # combine the two events?
        if second - first <= delta:
            # two events within the combination window, combine them and replace
            # the second event in the original list with the mean of the events
            events[events_index] = (first + second) / 2.
        else:
            # the two events can not be combined,
            # store the first event in the new list
            comb.append(first)
    # always append the last element of the list
    comb.append(events[-1])
    # return the combined onsets
    return comb
