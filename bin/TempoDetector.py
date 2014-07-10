#!/usr/bin/env python
# encoding: utf-8
"""
Copyright (c) Sebastian Böck <sebastian.boeck@jku.at>

Redistribution in any form is not permitted!

"""

from madmom.audio.signal import Signal
from madmom.features.tempo import TempoEstimator


def parser():
    """
    Create a parser and parse the arguments.

    :return: the parsed arguments
    """
    import argparse
    import madmom.utils.params

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    If invoked without any parameters, the software detects the dominant tempi
    in the given input (file) and writes them to the output (file).
    ''')
    # input/output options
    madmom.utils.params.io(p)
    # add tempo estimation arguments
    TempoEstimator.add_arguments(p)
    # add other argument groups
    madmom.utils.params.audio(p, fps=None, norm=False, online=None,
                              window=None)
    madmom.utils.params.save_load(p)
    # version
    p.add_argument('--mirex', action='store_true', default=False,
                   help='report the lower tempo first (as required by MIREX)')
    p.add_argument('--version', action='version',
                   version='TempoDetector.2013v2')
    # parse arguments
    args = p.parse_args()
    # set some defaults
    args.threads = min(len(args.nn_files), max(1, args.threads))
    # print arguments
    if args.verbose:
        print args
    # return
    return args


def main():
    """TempoDetector.2013"""

    # parse arguments
    args = parser()

    # load or create onset activations
    if args.load:
        # load activations
        t = TempoEstimator(args.input, args.fps, sep=args.sep)
    else:
        # exit if no NN files are given
        if not args.nn_files:
            raise SystemExit('no NN model(s) given')

        # create a Wav object
        w = Signal(args.input, mono=True, norm=args.norm, att=args.att)
        # create an Beat object with the activations
        t = TempoEstimator(w, **vars(args))

    # save activations or detect tempo
    if args.save:
        # save activations
        t.save_activations(args.output, sep=args.sep)
    else:
        # for MIREX, the lower tempo must be given first
        if args.mirex:
            t1, t2, strength = t.detections
            if t1 > t2:
                t.detections = t1, t2, 1. - strength

        # save detections
        t.save_detections(args.output)

if __name__ == '__main__':
    main()
