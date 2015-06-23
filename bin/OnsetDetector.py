#!/usr/bin/env python
# encoding: utf-8
"""
@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import argparse

from madmom import IOProcessor, io_arguments
from madmom.features import ActivationsProcessor
from madmom.features.onsets import RNNOnsetProcessor, PeakPickingProcessor


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
    ActivationsProcessor.add_arguments(p)
    RNNOnsetProcessor.add_arguments(p)
    PeakPickingProcessor.add_arguments(p, threshold=0.3, smooth=0.07)
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
        # use the RNN Beat processor
        in_processor = RNNOnsetProcessor(**vars(args))

    # output processor
    if args.save:
        # save the RNN onset activations to file
        out_processor = ActivationsProcessor(mode='w', **vars(args))
    else:
        # perform peak picking on the onset activations
        peak_picking = PeakPickingProcessor(pre_max=1. / args.fps,
                                            post_max=1. / args.fps,
                                            **vars(args))
        # output handler
        from madmom.utils import write_events as writer
        # sequentially process them
        out_processor = [peak_picking, writer]

    # create an IOProcessor
    processor = IOProcessor(in_processor, out_processor)

    # and call the processing function
    args.func(processor, **vars(args))


if __name__ == '__main__':
    main()
