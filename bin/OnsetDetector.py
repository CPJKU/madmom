#!/usr/bin/env python
# encoding: utf-8
"""
@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

from madmom.audio.signal import Signal
from madmom.features.onsets import RNNOnsetDetection


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
    If invoked without any parameters, the software detects all onsets in the
    given input (file) and writes them to the output (file).
    ''')
    # input/output options
    madmom.utils.io_arguments(p)
    # signal arguments
    Signal.add_arguments(p, norm=False)
    # rnn onset detection arguments
    RNNOnsetDetection.add_arguments(p)
    # version
    p.add_argument('--version', action='version', version='OnsetDetector.2013')
    # parse arguments
    args = p.parse_args()
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
        # instantiate OnsetDetection object from activations
        o = RNNOnsetDetection.from_activations(args.input, fps=100)
    else:
        # exit if no NN files are given
        if not args.nn_files:
            raise SystemExit('no NN model(s) given')

        s = Signal(args.input, mono=True, norm=args.norm, att=args.att)
        # create an RNNOnsetDetection object
        o = RNNOnsetDetection(s, nn_files=args.nn_files,
                              num_threads=args.num_threads)

    # save onset activations or detect onsets
    if args.save:
        # save activations
        o.activations.save(args.output, sep=args.sep)
    else:
        # detect onsets
        o.detect(args.threshold, combine=args.combine, delay=args.delay,
                 smooth=args.smooth)
        # save detections
        o.write(args.output)


if __name__ == '__main__':
    main()
