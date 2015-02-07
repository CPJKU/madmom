#!/usr/bin/env python
# encoding: utf-8
"""
@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

from madmom.utils import io_arguments
from madmom.features import ActivationsProcessor
from madmom.features.tempo import TempoEstimationProcessor


def parser():
    """
    Create a parser and parse the arguments.

    :return: the parsed arguments

    """
    import argparse

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    If invoked without any parameters, the software detects the dominant tempi
    in the given input (file) and writes them to the output (file).

    The tempo is inferred with comb filters from the beat activations produced
    by the algorithm described in:

    "Enhanced Beat Tracking with Context-Aware Neural Networks"
    Sebastian Böck and Markus Schedl
    Proceedings of the 14th International Conference on Digital Audio Effects
    (DAFx-11), 2011.

    ''')
    # add arguments
    io_arguments(p)
    ActivationsProcessor.add_arguments(p)
    TempoEstimationProcessor.add_arguments(p)
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

    # create an processor
    processor = TempoEstimationProcessor(**vars(args))
    # swap in/out processors if needed
    if args.load:
        processor.in_processor = ActivationsProcessor(mode='r', **vars(args))
    if args.save:
        processor.out_processor = ActivationsProcessor(mode='w', **vars(args))

    # process everything
    processor.process(args.input, args.output)


if __name__ == '__main__':
    main()
