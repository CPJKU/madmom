#!/usr/bin/env python
# encoding: utf-8
"""
@author: Sebastian Böck <sebastian.boeck@jku.at>

"""
import argparse

from madmom.utils import io_arguments
from madmom.features.beats import (RNNBeatTrackingProcessor,
                                   BeatTrackingProcessor)


def main():
    """BeatTracker.2014"""

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    The software detects all beats in an audio file; it can follow tempo
    changes.

    "Enhanced Beat Tracking with Context-Aware Neural Networks"
    Sebastian Böck and Markus Schedl
    Proceedings of the 14th International Conference on Digital Audio Effects
    (DAFx), 2011.

    A new comb filter method is used for tempo estimation (instead of the old
    auto-correlation based one).

    ''')
    # version
    p.add_argument('--version', action='version', version='BeatTracker.2014')
    # add arguments
    io_arguments(p, suffix='.beats.txt')
    RNNBeatTrackingProcessor.add_activation_arguments(p)
    RNNBeatTrackingProcessor.add_rnn_arguments(p)
    BeatTrackingProcessor.add_tempo_arguments(p)
    BeatTrackingProcessor.add_arguments(p, look_ahead=10)
    # parse arguments
    args = p.parse_args()
    # print arguments
    if args.verbose:
        print args

    # create a processor
    processor = RNNBeatTrackingProcessor(beat_method='BeatTracking',
                                         **vars(args))
    # and call the processing function
    args.func(processor, **vars(args))


if __name__ == '__main__':
    main()
