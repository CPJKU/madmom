#!/usr/bin/env python
# encoding: utf-8
"""
@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

from madmom.utils import io_arguments
from madmom.features.beats import RNNBeatTracking, DBNBeatTracking


def parser():
    """
    Create a parser and parse the arguments.

    :return: the parsed arguments

    """
    import argparse

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    If invoked without any parameters, the software detects all beats in the
    given input (file) and writes them to the output (file) according to the
    method described in:

    "A multi-model approach to beat tracking considering heterogeneous music
     styles"
    Sebastian Böck, Florian Krebs and Gerhard Widmer
    Proceedings of the 15th International Society for Music Information
    Retrieval Conference (ISMIR 2014), 2014.

    ''')

    # add arguments
    io_arguments(p)
    RNNBeatTracking.add_activation_arguments(p)
    RNNBeatTracking.add_rnn_arguments(p)
    DBNBeatTracking.add_arguments(p)
    # version
    p.add_argument('--version', action='version', version='MMBeatTracker')
    # parse arguments
    args = p.parse_args()
    # print arguments
    if args.verbose:
        print args
    # return
    return args


def main():
    """MMBeatTracker"""

    # parse arguments
    args = parser()
    # create an processor
    processor = RNNBeatTracking(beat_method='DBNBeatTracking',
                                multi_model=True, **vars(args))
    # pickle the processor if needed
    if args.pickle is not None:
        processor.dump(args.pickle)
    # process everything
    processor.process(args.input, args.output)

if __name__ == '__main__':
    main()
