#!/usr/bin/env python
# encoding: utf-8
"""
Copyright (c) Sebastian Böck <sebastian.boeck@jku.at>

Redistribution in any form is not permitted!

"""

import warnings
with warnings.catch_warnings():
    # import in this block to avoid warnings about missing compiled modules
    warnings.filterwarnings("ignore")
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

    The tempo is inferred with comb filters from the beat activations produced
    by the algorithm described in:

    "Enhanced Beat Tracking with Context-Aware Neural Networks"
    Sebastian Böck and Markus Schedl
    Proceedings of the 14th International Conference on Digital Audio Effects
    (DAFx-11), 2011.

    ''')
    # input/output options
    madmom.utils.io_arguments(p)
    # signal arguments
    Signal.add_arguments(p, norm=False)
    # rnn onset detection arguments
    TempoEstimation.add_arguments(p)
    # mirex stuff
    p.add_argument('--mirex', action='store_true', default=False,
                   help='use the MIREX output format (lower tempo first)')
    # version
    p.add_argument('--version', action='version', version='TempoDetector.2014')
    # parse arguments
    args = p.parse_args()
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
        t = TempoEstimation.from_activations(args.input, fps=100, sep=args.sep)
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
        # detect tempo
        t.detect(method=args.method, min_bpm=args.min_bpm,
                 max_bpm=args.max_bpm, act_smooth=args.act_smooth,
                 hist_smooth=args.hist_smooth, alpha=args.alpha)
        # save detections
        t.write(args.output, mirex=args.mirex)

if __name__ == '__main__':
    main()
