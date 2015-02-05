#!/usr/bin/env python
# encoding: utf-8
"""
@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

from madmom import IOProcessor
from madmom.utils import io_arguments
from madmom.features import ActivationsProcessor
from madmom.features.beats import RNNBeatProcessor, BeatTrackingProcessor


def parser():
    """
    Create a parser and parse the arguments.

    :return: the parsed arguments

    """
    import argparse

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    If invoked without any parameters, the software detects all beats in the
    given input (file) and writes them to the output (file). The algorithm can
    follow tempo changes.

    "Enhanced Beat Tracking with Context-Aware Neural Networks"
    Sebastian Böck and Markus Schedl
    Proceedings of the 14th International Conference on Digital Audio Effects
    (DAFx-11), 2011.

    A new comb filter method is used for tempo estimation (instead of the old
    auto-correlation based one).

    ''')

    # add arguments
    io_arguments(p)
    ActivationsProcessor.add_arguments(p)
    RNNBeatProcessor.add_arguments(p)
    BeatTrackingProcessor.add_tempo_arguments(p)
    BeatTrackingProcessor.add_arguments(p, look_aside=0.2, look_ahead=10)
    # version
    p.add_argument('--version', action='version', version='BeatTracker.2014')
    # parse arguments
    args = p.parse_args()
    # print arguments
    if args.verbose:
        print args
    # return
    return args


def main():
    """BeatTracker.2014"""

    # parse arguments
    args = parser()
    args.fps = 100

    # load or create beat activations
    if args.load:
        in_processor = ActivationsProcessor(mode='r', **vars(args))
    else:
        in_processor = RNNBeatProcessor(**vars(args))

    # save beat activations or detect beats
    if args.save:
        out_processor = ActivationsProcessor(mode='w', **vars(args))
    else:
        out_processor = BeatTrackingProcessor(**vars(args))

    # process everything
    IOProcessor(in_processor, out_processor).process(args.input, args.output)


if __name__ == '__main__':
    main()
