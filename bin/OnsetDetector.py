#!/usr/bin/env python
# encoding: utf-8
"""
@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

from madmom.utils import io_arguments
from madmom.features.onsets import RNNOnsetDetection


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
    RNNOnsetDetection.add_arguments(p)
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

    # create an processor
    processor = RNNOnsetDetection(**vars(args))
    # pickle the processor if needed
    if args.pickle is not None:
        processor.dump(args.pickle)
    # process everything
    processor.process(args.input, args.output)


if __name__ == '__main__':
    main()
