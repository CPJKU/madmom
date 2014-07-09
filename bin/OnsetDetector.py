#!/usr/bin/env python
# encoding: utf-8
"""
Copyright (c) Sebastian Böck <sebastian.boeck@jku.at>

Redistribution in any form is not permitted!

"""

from madmom.audio.signal import Signal
from madmom.features.onsets import RnnOnsetDetector
from madmom.ml.rnn import RecurrentNeuralNetwork


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
    If invoked without any parameters, the software detects all onsets in the
    given input (file) and writes them to the output (file).
    ''')
    # input/output options
    madmom.utils.params.io(p)
    # rnn onset detection arguments
    RnnOnsetDetector.add_arguments(p)
    # add other argument groups
    madmom.utils.params.audio(p, fps=None, norm=False, online=None,
                              window=None)
    madmom.utils.params.save_load(p)
    # version
    p.add_argument('--version', action='version', version='OnsetDetector.2013')
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
    """OnsetDetector.2013"""

    # parse arguments
    args = parser()

    # load or create onset activations
    if args.load:
        # load activations
        o = Onset(args.input, **vars(args))
    else:
        # exit if no NN files are given
        if not args.nn_files:
            raise SystemExit('no NN model(s) given')

        w = Signal(args.input, mono=True, norm=args.norm, att=args.att)
        # create an Onset object with the activations
        o = RnnOnsetDetector(w, **vars(args))

    # save onset activations or detect onsets
    if args.save:
        # save activations
        o.save_activations(args.output, sep=args.sep)
    else:
        # save detections
        o.save_detections(args.output)


if __name__ == '__main__':
    main()
