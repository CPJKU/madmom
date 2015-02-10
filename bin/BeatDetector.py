#!/usr/bin/env python
# encoding: utf-8
"""
@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

from madmom.utils import io_arguments
from madmom.features.beats import RNNBeatTracking, BeatDetection


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
    given input (file) and writes them to the output (file). The algorithm
    assumes a constant tempo throughout the whole piece.

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
    BeatDetection.add_tempo_arguments(p)
    BeatDetection.add_arguments(p, look_ahead=None)
    # version
    p.add_argument('--version', action='version', version='BeatDetector.2014')
    # parse arguments
    args = p.parse_args()
    # print arguments
    if args.verbose:
        print args
    # return
    return args


def main():
    """BeatDetector.2014"""

    # parse arguments
    args = parser()

    # create an processor
    processor = RNNBeatTracking(beat_method='BeatDetection', **vars(args))
    # process everything
    processor.process(args.input, args.output)


if __name__ == "__main__":
    main()
