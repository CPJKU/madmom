#!/usr/bin/env python
# encoding: utf-8
"""
@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

from madmom.utils import io_arguments
from madmom.features.beats import RNNBeatTracking, BeatTracking


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
    RNNBeatTracking.add_activation_arguments(p)
    RNNBeatTracking.add_rnn_arguments(p)
    BeatTracking.add_tempo_arguments(p)
    BeatTracking.add_arguments(p, look_ahead=10)
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

    # create an processor
    processor = RNNBeatTracking(beat_method='BeatTracking', **vars(args))
    # pickle the processor if needed
    if args.pickle is not None:
        processor.dump(args.pickle)
    # process everything
    processor.process(args.input, args.output)


if __name__ == '__main__':
    main()
