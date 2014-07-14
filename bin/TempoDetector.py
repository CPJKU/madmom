#!/usr/bin/env python
# encoding: utf-8
"""
Copyright (c) Sebastian Böck <sebastian.boeck@jku.at>

Redistribution in any form is not permitted!

"""

from madmom.audio.signal import Signal
from madmom.features.tempo import TempoEstimation


def parser():
    """
    Create a parser and parse the arguments.

    :return: the parsed arguments

    """
    import argparse
    import madmom.utils

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    If invoked without any parameters, the software detects the dominant tempi
    in the given input (file) and writes them to the output (file).
    ''')
    # input/output options
    madmom.utils.io_arguments(p)
    # signal arguments
    Signal.add_arguments(p, norm=False)
    # rnn onset detection arguments
    TempoEstimation.add_arguments(p)
    # mirex stuff
    p.add_argument('--mirex', action='store_true', default=False,
                   help='report the lower tempo first (as required by MIREX)')
    # version
    p.add_argument('--version', action='version', version='TempoDetector.2014')
    # parse arguments
    args = p.parse_args()
    # set some defaults
    args.num_threads = min(len(args.nn_files), max(1, args.num_threads))
    # print arguments
    if args.verbose:
        print args
    # return
    return args


def main():
    """TempoDetector.2014"""

    # parse arguments
    args = parser()

    # load or create onset activations
    if args.load:
        # load activations
        t = TempoEstimation.from_activations(args.input, sep=args.sep)
    else:
        # exit if no NN files are given
        if not args.nn_files:
            raise SystemExit('no NN model(s) given')

        # create a Signal object
        s = Signal(args.input, mono=True, norm=args.norm, att=args.att)
        # create a RNNBeatDetection object from the signal and given NN files
        t = TempoEstimation(s, nn_files=args.nn_files,
                            num_threads=args.num_threads)

    # save activations or detect tempo
    if args.save:
        # save activations
        t.activations.save(args.output, sep=args.sep)
    else:
        # save detections
        t.write(args.output, args.mirex)

if __name__ == '__main__':
    main()
