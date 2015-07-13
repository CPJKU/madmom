#!/usr/bin/env python
# encoding: utf-8
"""
@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import argparse

from madmom.processors import IOProcessor, io_arguments
from madmom.features import ActivationsProcessor
from madmom.features.notes import (RNNNoteProcessor, write_midi, write_notes,
                                   write_frequencies)
from madmom.features.onsets import PeakPickingProcessor


def main():
    """PianoTranscriptor.2014"""

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    The software detects all notes (onsets) in an audio file with the algorithm
    described in:

    "Polyphonic Piano Note Transcription with Recurrent Neural Networks"
    Sebastian Böck and Markus Schedl.
    Proceedings of the 37th International Conference on Acoustics, Speech and
    Signal Processing (ICASSP), 2012.

    Instead of 'LSTM' units, the current version uses 'tanh' units.

    ''')
    # version
    p.add_argument('--version', action='version',
                   version='PianoTranscriptor.2014')
    # add arguments
    io_arguments(p, suffix='.notes.txt')
    ActivationsProcessor.add_arguments(p)
    RNNNoteProcessor.add_arguments(p)
    PeakPickingProcessor.add_arguments(p, threshold=0.35, smooth=0.09,
                                       combine=0.05)
    # midi arguments
    # import madmom.utils.midi as midi
    # midi.MIDIFile.add_arguments(p, length=0.6, velocity=100)
    p.add_argument('--midi', dest='output_format', action='store_const',
                   const='midi', help='save as MIDI')
    # mirex stuff
    p.add_argument('--mirex', dest='output_format', action='store_const',
                   const='mirex', help='use the MIREX output format')

    # parse arguments
    args = p.parse_args()
    # set the suffix for midi files
    if args.output_format == 'midi':
        args.output_suffix = '.mid'
    # print arguments
    if args.verbose:
        print args

    # TODO: remove this hack!
    args.fps = 100

    # input processor
    if args.load:
        # load the activations from file
        in_processor = ActivationsProcessor(mode='r', **vars(args))
    else:
        # use the RNN Beat processor
        in_processor = RNNNoteProcessor(**vars(args))

    # output processor
    if args.save:
        # save the RNN note activations to file
        out_processor = ActivationsProcessor(mode='w', **vars(args))
    else:
        # perform peak picking of the detection function
        peak_picking = PeakPickingProcessor(pre_max=1. / args.fps,
                                            post_max=1. / args.fps,
                                            **vars(args))
        # output everything in the right format
        if args.output_format is None:
            output = write_notes
        elif args.output_format == 'midi':
            output = write_midi
        elif args.output_format == 'mirex':
            output = write_frequencies
        else:
            raise ValueError('unknown output format: %s' % args.output_format)
        out_processor = [peak_picking, output]

    # create an IOProcessor
    processor = IOProcessor(in_processor, out_processor)

    # and call the processing function
    args.func(processor, **vars(args))


if __name__ == '__main__':
    main()
