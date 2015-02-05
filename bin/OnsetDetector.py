#!/usr/bin/env python
# encoding: utf-8
"""
@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

from madmom import IOProcessor
from madmom.utils import io_arguments
from madmom.features import ActivationsProcessor
from madmom.features.onsets import RNNOnsetProcessor, OnsetDetectionProcessor


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
    given input (file) and writes them to the output (file).
    ''')

    # input/output options
    io_arguments(p)
    ActivationsProcessor.add_arguments(p)
    RNNOnsetProcessor.add_arguments(p)
    OnsetDetectionProcessor.add_arguments(p, threshold=0.35, smooth=0.07)
    # version
    p.add_argument('--version', action='version', version='OnsetDetector.2013')
    # parse arguments
    args = p.parse_args()
    # print arguments
    if args.verbose:
        print args
    # return
    return args


def main():
    """OnsetDetector.2013"""

    # parse arguments
    args = parser()
    args.fps = 100

    # load or create beat activations
    if args.load:
        in_processor = ActivationsProcessor(mode='r', **vars(args))
    else:
        in_processor = RNNOnsetProcessor(**vars(args))

    # save beat activations or detect beats
    if args.save:
        out_processor = ActivationsProcessor(mode='w', **vars(args))
    else:
        out_processor = OnsetDetectionProcessor(**vars(args))

    # process everything
    IOProcessor(in_processor, out_processor).process(args.input, args.output)


if __name__ == '__main__':
    main()
