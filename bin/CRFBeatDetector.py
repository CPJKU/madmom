#!/usr/bin/env python
# encoding: utf-8
"""
@author: Filip Korzeniowski <filip.korzeniowski@jku.at>

"""

import argparse

from madmom.utils import io_arguments
from madmom.features.beats import RNNBeatTracking, CRFBeatDetection


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

    Instead of using the auto-correlation method to determine the dominant
    interval, a new method based on comb filters is used to get multiple tempo
    hypotheses.

    "Accurate Tempo Estimation based on Recurrent Neural Networks and
     Resonating Comb Filters"
    Sebastian Böck, Florian Krebs and Gerhard Widmer
    Proceedings of the 16th International Society for Music Information
    Retrieval Conference (ISMIR), 2015.

    ''')
    # version
    p.add_argument('--version', action='version', version='CRFBeatDetector')
    # add arguments
    io_arguments(p, suffix='.beats.txt')
    RNNBeatTracking.add_activation_arguments(p)
    RNNBeatTracking.add_rnn_arguments(p)
    CRFBeatDetection.add_tempo_arguments(p)
    CRFBeatDetection.add_arguments(p)

    # parse arguments
    args = p.parse_args()
    # print arguments
    if args.verbose:
        print args

    # create a processor
    processor = RNNBeatTracking(beat_method='CRFBeatDetection', **vars(args))
    # and call the processing function
    args.func(processor, **vars(args))


if __name__ == "__main__":
    main()
