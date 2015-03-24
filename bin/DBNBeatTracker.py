#!/usr/bin/env python
# encoding: utf-8
"""
@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import argparse

from madmom.utils import io_arguments
from madmom.features.beats import RNNBeatTracking, DBNBeatTracking


def main():
    """DBNBeatTracker"""

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    The software detects all beats in an audio file according to the method
    described in:

    "A multi-model approach to beat tracking considering heterogeneous music
     styles"
    Sebastian Böck, Florian Krebs and Gerhard Widmer
    Proceedings of the 15th International Society for Music Information
    Retrieval Conference (ISMIR), 2014.

    It does not use the multi-model (Section 2.2.) and selection stage (Section
    2.3), i.e. this version corresponds to the pure DBN version of the
    algorithm for which results are given in Table 2.

    Instead of the originally proposed transition model for the DBN, the
    following is used:

    TODO: add reference!

    ''')
    # version
    p.add_argument('--version', action='version',
                   version='DBNBeatTracker.2015')
    # add arguments
    io_arguments(p)
    RNNBeatTracking.add_activations_arguments(p)
    RNNBeatTracking.add_rnn_arguments(p)
    DBNBeatTracking.add_arguments(p)
    # parse arguments
    args = p.parse_args()
    # print arguments
    if args.verbose:
        print args

    # create a processor
    processor = RNNBeatTracking(beat_method='DBNBeatTracking', **vars(args))
    # and call the processing function
    args.func(processor, **vars(args))


if __name__ == '__main__':
    main()
