#!/usr/bin/env python
# encoding: utf-8
"""
@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import argparse

from madmom import IOProcessor, io_arguments
from madmom.features import ActivationsProcessor
from madmom.features.beats import RNNBeatProcessor
from madmom.features.tempo import TempoEstimationProcessor, write_tempo


# wrapper function to be used as output of TempoEstimation
from functools import partial
write_tempo_mirex = partial(write_tempo, mirex=True)
write_tempo_mirex.__doc__ = 'write_tempo(tempo, filename, mirex=True)'


def main():
    """TempoDetector.2014"""

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    The software detects the dominant tempi in an audio file by inferring it
    with comb filters from the beat activations produced by the algorithm
    described in:

    "Enhanced Beat Tracking with Context-Aware Neural Networks"
    Sebastian Böck and Markus Schedl
    Proceedings of the 14th International Conference on Digital Audio Effects
    (DAFx), 2011.

    Instead of using the originally proposed auto-correlation method to build
    a tempo histogram, a new method based on comb filters is used:

    TODO: add reference!

    The old behaviour can be restored by using the `--method acf` switch.

    ''')
    # version
    p.add_argument('--version', action='version', version='TempoDetector.2014')
    # add arguments
    io_arguments(p, suffix='.bpm.txt')
    ActivationsProcessor.add_arguments(p)
    RNNBeatProcessor.add_arguments(p)
    TempoEstimationProcessor.add_arguments(p)
    # mirex stuff
    g = p.add_mutually_exclusive_group()
    g.add_argument('--mirex', dest='tempo_format',
                   action='store_const', const='mirex',
                   help='use the MIREX output format (lower tempo first)')
    g.add_argument('--all', dest='tempo_format',
                   action='store_const', const='all',
                   help='output all detected tempi in raw format')

    # parse arguments
    args = p.parse_args()
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
        in_processor = RNNBeatProcessor(**vars(args))
    # output processor
    if args.save:
        # save the RNN beat activations to file
        out_processor = ActivationsProcessor(mode='w', **vars(args))
    else:
        # tempo estimation based on the beat activation function
        tempo_estimator = TempoEstimationProcessor(**vars(args))
        # output handler
        if args.tempo_format == 'mirex':
            # output in the MIREX format (i.e. slower tempo first)
            writer = write_tempo_mirex
        elif args.tempo_format in ('raw', 'all'):
            # borrow the note writer for outputting multiple values
            from madmom.features.notes import write_notes as writer
        else:
            # normal output
            writer = write_tempo
        # sequentially process them
        out_processor = [tempo_estimator, writer]
    # create an IOProcessor
    processor = IOProcessor(in_processor, out_processor)

    # finally call the processing function (single/batch processing)
    args.func(processor, **vars(args))


if __name__ == '__main__':
    main()
