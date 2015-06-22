#!/usr/bin/env python
# encoding: utf-8
"""
@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import argparse

from madmom import IOProcessor, io_arguments
from madmom.features import ActivationsProcessor
from madmom.features.beats import RNNBeatProcessor, BeatDetectionProcessor


def main():
    """BeatDetector.2014"""

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    The software detects all beats in an audio file; it assumes a constant
    tempo throughout the whole piece.

    "Enhanced Beat Tracking with Context-Aware Neural Networks"
    Sebastian Böck and Markus Schedl
    Proceedings of the 14th International Conference on Digital Audio Effects
    (DAFx), 2011.

    A new comb filter method is used for tempo estimation (instead of the old
    auto-correlation based one).

    ''')
    # version
    p.add_argument('--version', action='version', version='BeatDetector.2014')
    # add arguments
    io_arguments(p, suffix='.beats.txt')
    ActivationsProcessor.add_arguments(p)
    RNNBeatProcessor.add_arguments(p)
    BeatDetectionProcessor.add_tempo_arguments(p)
    BeatDetectionProcessor.add_arguments(p, look_ahead=None)
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
        # process the signal with a RNN tp predict the beats
        in_processor = RNNBeatProcessor(**vars(args))
    # output processor
    if args.save:
        # save the RNN beat activations to file
        out_processor = ActivationsProcessor(mode='w', **vars(args))
    else:
        # detect the beats in the activation function
        beat_processor = BeatDetectionProcessor(**vars(args))
        # output handler
        from madmom.utils import write_events as writer
        # sequentially process them
        out_processor = [beat_processor, writer]
    # create an IOProcessor
    processor = IOProcessor(in_processor, out_processor)

    # and call the processing function
    args.func(processor, **vars(args))


if __name__ == "__main__":
    main()
