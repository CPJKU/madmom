#!/usr/bin/env python
# encoding: utf-8
"""
Copyright (c) Sebastian Böck <sebastian.boeck@jku.at>

Redistribution in any form is not permitted!

"""

from madmom.audio.signal import Signal
from madmom.features.onsets import RnnOnsetDetectorLL


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
    given input (file) and writes them to the output (file) with the algorithm
    introduced in:

    "Online Real-time Onset Detection with Recurrent Neural Networks"
    Sebastian Böck, Andreas Arzt, Florian Krebs and Markus Schedl
    Proceedings of the 15th International Conference on Digital Audio Effects
    (DAFx-12), York, UK, September 2012

    The paper includes an error in Section 2.2.1, 2nd paragraph:
    The targets of the training examples have been annotated 1 frame shifted to
    the future, thus the results given in Table 2 are off by 10ms. Given the
    fact that the delayed reporting (as outlined in Section 2.3) is not
    needed, an extra shift of 5ms needs to be added to the mean errors given in
    Table 2.

    This implementation takes care of this error is is modified in this way:
    - a logarithmic frequency spacing is used for the spectrograms instead of
      the Bark scale
    - targets are annotated at the next frame for neural network training
    - post processing reports the onset instantaneously instead of delayed.

    ''')
    # mirex options
    madmom.utils.params.io(p)
    # rnn ll onset detection arguments
    RnnOnsetDetectorLL.add_arguments(p)
    # add other argument groups
    madmom.utils.params.audio(p, fps=None, norm=False, online=None,
                              window=None)
    madmom.utils.params.save_load(p)
    # version
    p.add_argument('--version', action='version',
                   version='OnsetDetectorLL.2013')
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
    """OnsetDetectorLL.2013"""

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
        o = RnnOnsetDetectorLL(w, **vars(args))

    # save onset activations or detect onsets
    if args.save:
        # save activations
        o.save_activations(args.output, sep=args.sep)
    else:
        # save detections
        o.save_detections(args.output)

if __name__ == '__main__':
    main()
