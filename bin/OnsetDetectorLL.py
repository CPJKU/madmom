#!/usr/bin/env python
# encoding: utf-8
"""
Copyright (c) Sebastian Böck <sebastian.boeck@jku.at>

Redistribution in any form is not permitted!

"""

import os
import glob
from madmom.audio.signal import Signal
from madmom.features.onsets import RNNOnsetDetection

# set the path to saved neural networks and generate lists of NN files
NN_PATH = '%s/../madmom/ml/data' % (os.path.dirname(__file__))
NN_FILES = glob.glob("%s/onsets_rnn*npz" % NN_PATH)


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
    # input/output options
    madmom.utils.io_arguments(p)
    # signal arguments
    Signal.add_arguments(p, norm=None)
    # rnn onset detection arguments
    RNNOnsetDetection.add_arguments(p, nn_files=NN_FILES, threshold=0.2,
                                    combine=0.02, smooth=None)
    # version
    p.add_argument('--version', action='version',
                   version='OnsetDetectorLL.2013')
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
    """OnsetDetectorLL.2013"""

    # parse arguments
    args = parser()

    # load or create onset activations
    if args.load:
        # load activations
        o = RNNOnsetDetection.from_activations(args.input, fps=100)
    else:
        # exit if no NN files are given
        if not args.nn_files:
            raise SystemExit('no NN model(s) given')

        s = Signal(args.input, mono=True, att=args.att)
        # create an Onset object with the activations
        o = RNNOnsetDetection(s, nn_files=args.nn_files,
                              num_threads=args.num_threads)
        o.pre_process(frame_sizes=[512, 1024, 2048], origin='online')

    # save onset activations or detect onsets
    if args.save:
        # save activations
        o.activations.save(args.output, sep=args.sep)
    else:
        # detect onsets
        o.detect(args.threshold, combine=args.combine, delay=args.delay,
                 smooth=0, pre_avg=args.pre_avg,
                 post_avg=0, pre_max=args.pre_max,
                 post_max=0)
        # save detections
        o.write(args.output)

if __name__ == '__main__':
    main()
