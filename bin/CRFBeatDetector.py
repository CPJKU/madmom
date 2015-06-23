#!/usr/bin/env python
# encoding: utf-8
"""
@author: Filip Korzeniowski <filip.korzeniowski@jku.at>

"""

import argparse

from madmom import IOProcessor, io_arguments
from madmom.features import ActivationsProcessor
from madmom.features.beats import RNNBeatProcessor, CRFBeatDetectionProcessor


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
    p.add_argument('--version', action='version',
                   version='CRFBeatDetector.2015')
    # add arguments
    io_arguments(p, suffix='.beats.txt')
    ActivationsProcessor.add_arguments(p)
    RNNBeatProcessor.add_arguments(p)
    CRFBeatDetectionProcessor.add_tempo_arguments(p)
    CRFBeatDetectionProcessor.add_arguments(p)

    # parse arguments
    args = p.parse_args()
    # print arguments
    if args.verbose:
        print args

    # TODO: remove this hack!
    args.fps = 100

    # input processor
    if args.load:
        # load the activations from file
        in_processor = ActivationsProcessor(mode='r', **vars(args))
    else:
        # process the signal with a RNN tp predict the beats
        in_processor = RNNBeatProcessor(**vars(args))

    # output processor
    if args.save:
        # save the RNN beat activations to file
        out_processor = ActivationsProcessor(mode='w', **vars(args))
    else:
        # detect the beats with a CRF
        beat_processor = CRFBeatDetectionProcessor(**vars(args))
        # output handler
        from madmom.utils import write_events as writer
        # sequentially process them
        out_processor = [beat_processor, writer]

    # create an IOProcessor
    processor = IOProcessor(in_processor, out_processor)

    # and call the processing function
    args.func(processor, **vars(args))


if __name__ == "__main__":
    main()
