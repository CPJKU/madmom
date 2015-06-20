#!/usr/bin/env python
# encoding: utf-8
"""
@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import argparse

from madmom.utils import io_arguments
from madmom.features.tempo import RNNTempoEstimationProcessor


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
    RNNTempoEstimationProcessor.add_activation_arguments(p)
    RNNTempoEstimationProcessor.add_rnn_arguments(p)
    RNNTempoEstimationProcessor.add_arguments(p)
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

    # create a processor
    processor = RNNTempoEstimationProcessor(**vars(args))
    # and call the processing function
    args.func(processor, **vars(args))


if __name__ == '__main__':
    main()
