#!/usr/bin/env python
# encoding: utf-8
"""
@author: Sebastian Böck <sebastian.boeck@jku.at>

"""
import argparse

from madmom.utils import io_arguments
from madmom.features.beats import DownbeatTracking, SpectralBeatTracking


def main():
    """DownBeatTracker"""

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    The software detects all down-beats in an audio file according to the
    method described in:

    "Rhythmic Pattern Modelling For Beat and Downbeat Tracking in Musical
     Audio"
    Florian Krebs, Sebastian Böck, and Gerhard Widmer
    Proceedings of the 14th International Society for Music Information
    Retrieval Conference (ISMIR), 2013.

    ''')
    # version
    p.add_argument('--version', action='version', version='DownBeatTracker')
    # add arguments
    io_arguments(p)
    DownbeatTracking.add_arguments(p)
    # parse arguments
    args = p.parse_args()
    # print arguments
    if args.verbose:
        print args

    # create a processor
    processor = SpectralBeatTracking(**vars(args))
    # and call the processing function
    args.func(processor, **vars(args))


if __name__ == '__main__':
    main()
