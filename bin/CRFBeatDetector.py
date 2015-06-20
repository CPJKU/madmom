#!/usr/bin/env python
# encoding: utf-8
"""
@author: Filip Korzeniowski <filip.korzeniowski@jku.at>

"""

import argparse

from madmom.utils import io_arguments
from madmom.features.beats import (RNNBeatTrackingProcessor,
                                   CRFBeatDetectionProcessor)


def main():
    """CRFBeatDetector"""

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    The software detects all beats in an audio file according to the method
    described in:

    "Probabilistic extraction of beat positions from a beat activation
     function"
    Filip Korzeniowski, Sebastian Böck and Gerhard Widmer
    In Proceedings of the 15th International Society for Music Information
    Retrieval Conference (ISMIR), 2014.

    ''')
    # version
    p.add_argument('--version', action='version', version='CRFBeatDetector')
    # add arguments
    io_arguments(p, suffix='.beats.txt')
    RNNBeatTrackingProcessor.add_activation_arguments(p)
    RNNBeatTrackingProcessor.add_rnn_arguments(p)
    CRFBeatDetectionProcessor.add_tempo_arguments(p)
    CRFBeatDetectionProcessor.add_arguments(p)

    # parse arguments
    args = p.parse_args()
    # print arguments
    if args.verbose:
        print args

    # create a processor
    processor = RNNBeatTrackingProcessor(beat_method='CRFBeatDetection',
                                         **vars(args))
    # and call the processing function
    args.func(processor, **vars(args))


if __name__ == "__main__":
    main()
