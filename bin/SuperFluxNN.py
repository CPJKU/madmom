#!/usr/bin/env python
# encoding: utf-8
"""
SuperFlux with neural network based peak picking onset detection algorithm.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

from madmom import IOProcessor
from madmom.utils import io_arguments
from madmom.features import ActivationsProcessor
from madmom.features.onsets import (SpectralOnsetProcessor,
                                    NNOnsetDetectionProcessor)


def parser():
    """
    Create a parser and parse the arguments.

    :return: the parsed arguments

    """
    import argparse

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    If invoked without any parameters, the software detects all onsets in the
    given input file and writes them to the output file with the algorithm
    introduced in:

    "Enhanced peak picking for onset detection with recurrent neural networks"
    Sebastian Böck, Jan Schlüter and Gerhard Widmer
    Proceedings of the 6th International Workshop on Machine Learning and
    Music (MML), 2013.

    Please note that this implementation uses 100 frames per second (instead
    of 200), because it is faster and produces highly comparable results.

    ''')
    # add arguments
    io_arguments(p)
    SpectralOnsetProcessor.add_arguments(p)
    NNOnsetDetectionProcessor.add_arguments(p)
    ActivationsProcessor.add_arguments(p)
    # version
    p.add_argument('--version', action='version', version='SuperFluxNN')
    # parse arguments
    args = p.parse_args()
    # print arguments
    if args.verbose:
        print args
    # return
    return args


def main():
    """SuperFluxNN"""

    # parse arguments
    args = parser()

    # load or create beat activations
    if args.load:
        in_processor = ActivationsProcessor(mode='r', **vars(args))
    else:
        in_processor = SpectralOnsetProcessor(**vars(args))

    # save onset activations or detect onsets
    if args.save:
        out_processor = ActivationsProcessor(mode='w', **vars(args))
    else:
        out_processor = NNOnsetDetectionProcessor(**vars(args))

    # process everything
    IOProcessor(in_processor, out_processor).process(args.input, args.output)

if __name__ == '__main__':
    main()
