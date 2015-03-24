#!/usr/bin/env python
# encoding: utf-8
"""
@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import argparse

from madmom.utils import io_arguments
from madmom.features.beats import RNNBeatTracking, BeatDetection


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
    io_arguments(p)
    RNNBeatTracking.add_activations_arguments(p)
    RNNBeatTracking.add_rnn_arguments(p)
    BeatDetection.add_tempo_arguments(p)
    BeatDetection.add_arguments(p, look_ahead=None)
    # parse arguments
    args = p.parse_args()
    # print arguments
    if args.verbose:
        print args

    # create a processor
    processor = RNNBeatTracking(beat_method='BeatDetection', **vars(args))
    # and call the processing function
    args.func(processor, **vars(args))


if __name__ == "__main__":
    main()
