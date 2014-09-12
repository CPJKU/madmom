#!/usr/bin/env python
# encoding: utf-8
"""
Copyright (c) Sebastian Böck <sebastian.boeck@jku.at>

Redistribution in any form is not permitted!

"""

from madmom.audio.signal import Signal
from madmom.features.beats import RNNBeatTracking


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
    If invoked without any parameters, the software detects all beats in the
    given input (file) and writes them to the output (file). The algorithm can
    follow tempo changes.

    "Enhanced Beat Tracking with Context-Aware Neural Networks"
    Sebastian Böck and Markus Schedl
    Proceedings of the 14th International Conference on Digital Audio Effects
    (DAFx-11), 2011.

    A new comb filter method is used for tempo estimation (instead of the old
    auto-correlation based one).

    ''')

    # input/output options
    madmom.utils.io_arguments(p)
    # signal arguments
    Signal.add_arguments(p, norm=False)
    # beat tracking arguments
    RNNBeatTracking.add_arguments(p)
    # version
    p.add_argument('--version', action='version', version='BeatTracker.2014')
    # parse arguments
    args = p.parse_args()
    # print arguments
    if args.verbose:
        print args
    # return
    return args


def main():
    """BeatTracker.2014"""
    # parse arguments
    args = parser()

    # load or create onset activations
    if args.load:
        # load activations
        b = RNNBeatTracking.from_activations(args.input, fps=100)
    else:
        # exit if no NN files are given
        if not args.nn_files:
            raise SystemExit('no NN model(s) given')

        # create a Signal object
        s = Signal(args.input, mono=True, norm=args.norm, att=args.att)
        # create an RNNBeatTracking object
        b = RNNBeatTracking(s, nn_files=args.nn_files,
                            num_threads=args.num_threads)

    # save beat activations or detect beats
    if args.save:
        # save activations
        b.activations.save(args.output, sep=args.sep)
    else:
        # detect the beats
        b.detect(min_bpm=args.min_bpm, max_bpm=args.max_bpm,
                 act_smooth=args.act_smooth, hist_smooth=args.hist_smooth,
                 alpha=args.alpha, look_aside=args.look_aside,
                 look_ahead=args.look_ahead)
        # save detections
        b.write(args.output)

if __name__ == '__main__':
    main()
