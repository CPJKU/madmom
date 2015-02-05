#!/usr/bin/env python
# encoding: utf-8
"""
SuperFlux onset detection algorithm.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

from madmom import IOProcessor
from madmom.utils import io_arguments
from madmom.features import ActivationsProcessor
from madmom.features.onsets import (SpectralOnsetProcessor,
                                    OnsetDetectionProcessor)


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
    given input file and writes them to the output file with the SuperFlux
    algorithm introduced in:

    "Maximum Filter Vibrato Suppression for Onset Detection"
    Sebastian Böck and Gerhard Widmer
    Proceedings of the 16th International Conference on Digital Audio Effects
    (DAFx-13), 2013.

    ''')
    # add arguments
    io_arguments(p)
    SpectralOnsetProcessor.add_arguments(p)
    OnsetDetectionProcessor.add_arguments(p, threshold=1.1, pre_max=0.01,
                                          post_max=0.05, pre_avg=0.15,
                                          post_avg=0, combine=0.03, delay=0)
    ActivationsProcessor.add_arguments(p)
    # version
    p.add_argument('--version', action='version', version='SuperFlux.2014')
    # parse arguments
    args = p.parse_args()
    # switch to offline mode
    if args.norm:
        args.online = False
    # print arguments
    if args.verbose:
        print args
    # return
    return args


def main():
    """SuperFlux.2014"""

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
        out_processor = OnsetDetectionProcessor(**vars(args))

    # process everything
    IOProcessor(in_processor, out_processor).process(args.input, args.output)

if __name__ == '__main__':
    main()
