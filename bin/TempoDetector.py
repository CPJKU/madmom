#!/usr/bin/env python
# encoding: utf-8
"""
@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

from madmom.utils import io_arguments
from madmom.features.tempo import RNNTempoEstimation


def parser():
    """
    Create a parser and parse the arguments.

    :return: the parsed arguments

    """
    import argparse

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    The software detects the dominant tempi in an audio file by inferring it
    with comb filters from the beat activations produced by the algorithm
    described in:

    "Enhanced Beat Tracking with Context-Aware Neural Networks"
    Sebastian Böck and Markus Schedl
    Proceedings of the 14th International Conference on Digital Audio Effects
    (DAFx-11), 2011.

    ''')
    # add arguments
    io_arguments(p)
    RNNTempoEstimation.add_activation_arguments(p)
    RNNTempoEstimation.add_rnn_arguments(p)
    RNNTempoEstimation.add_arguments(p)
    # mirex stuff
    p.add_argument('--mirex', action='store_true', default=False,
                   help='use the MIREX output format (lower tempo first)')
    # version
    p.add_argument('--version', action='version', version='TempoDetector.2014')
    # parse arguments
    args = p.parse_args()
    # print arguments
    if args.verbose:
        print args
    # return
    return args


def main():
    """TempoDetector.2014"""

    # parse arguments
    args = parser()

    # create a processor
    processor = RNNTempoEstimation(**vars(args))
    # and call the processing function
    args.func(processor, **vars(args))


if __name__ == '__main__':
    main()
