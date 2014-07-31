#!/usr/bin/env python
# encoding: utf-8
"""
Copyright (c) Filip Korzeniowski <filip.korzeniowski@jku.at>

Redistribution in any form is not permitted!

"""

from madmom.audio.signal import Signal
from madmom.features.beats import CRFBeatDetection


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
    given input (file) and writes them to the output (file) according to the
    method described in:

    "Probabilistic extraction of beat positions from a beat activation
     function"
    Filip Korzeniowski, Sebastian Böck and Gerhard Widmer
    In Proceedings of the 15th International Society for Music Information
    Retrieval Conference (ISMIR 2014), Taipeh, Taiwan, November 2014.

    ''')

    # input/output options
    madmom.utils.io_arguments(p)
    # signal arguments
    Signal.add_arguments(p, norm=False)
    # rnn beat detection arguments
    CRFBeatDetection.add_arguments(p)
    # version
    p.add_argument('--version', action='version', version='CRFBeatDetector')
    # parse arguments
    args = p.parse_args()
    # print arguments
    if args.verbose:
        print args
    # return
    return args


def main():
    """CRFBeatDetector."""

    # parse arguments
    args = parser()

    # load or create onset activations
    if args.load:
        # load activations
        b = CRFBeatDetection.from_activations(args.input, fps=100)
        # set the number of threads, since the detection works multi-threaded
        b.num_threads = args.num_threads
    else:
        # exit if no NN files are given
        if not args.nn_files:
            raise SystemExit('no NN model(s) given')

        # create a Signal object
        s = Signal(args.input, mono=True, norm=args.norm, att=args.att)
        # create an CRFBeatDetection object
        b = CRFBeatDetection(s, nn_files=args.nn_files,
                             num_threads=args.num_threads)

    # save beat activations or detect beats
    if args.save:
        # save activations
        b.activations.save(args.output, sep=args.sep)
    else:
        # detect the beats
        b.detect(smooth=args.smooth, min_bpm=args.min_bpm,
                 max_bpm=args.max_bpm, interval_sigma=args.interval_sigma,
                 factors=args.factors)
        # save detections
        b.write(args.output)


if __name__ == "__main__":
    main()
