#!/usr/bin/env python
# encoding: utf-8
"""
@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import argparse

from madmom.utils import io_arguments
from madmom.features.onsets import RNNOnsetDetectionProcessor


def main():
    """OnsetDetector.2013"""

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    The software detects all onsets in an audio file with a recurrent neural
    network.
    ''')
    # version
    p.add_argument('--version', action='version', version='OnsetDetector.2013')
    # input/output options
    io_arguments(p, suffix='.onsets.txt')
    RNNOnsetDetectionProcessor.add_arguments(p)
    # parse arguments
    args = p.parse_args()
    # print arguments
    if args.verbose:
        print args

    # create a processor
    processor = RNNOnsetDetectionProcessor(**vars(args))
    # and call the processing function
    args.func(processor, **vars(args))


if __name__ == '__main__':
    main()
