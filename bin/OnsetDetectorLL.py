#!/usr/bin/env python
# encoding: utf-8
"""
Copyright (c) Sebastian Böck <sebastian.boeck@jku.at>

Redistribution in any form is not permitted!

"""

import os
import glob
from madmom.audio.signal import Signal
from madmom.features.onsets import RnnOnsetDetector


# set the path to saved neural networks and generate lists of NN files
NN_PATH = '%s/../madmom/ml/data' % (os.path.dirname(__file__))
NN_FILES = glob.glob("%s/onsets_rnn*npz" % NN_PATH)

# TODO: this information should be included/extracted in/from the NN files
FPS = 100
BANDS_PER_OCTAVE = 6
MUL = 5
ADD = 1
NORM_FILTERS = True

## TODO: these do not seem to be used in the original implementation
FMIN = 30
FMAX = 17000
RATIO = 0.5

ONLINE = True
WINDOW_SIZES = [512, 1024, 2048]
THRESHOLD = 0.2
COMBINE = 0.02
SMOOTH = 0.0
PRE_AVG = 0
POST_AVG = 0
PRE_MAX = 1. / FPS
POST_MAX = 0


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
    RnnOnsetDetector.add_arguments(p, nn_files=NN_FILES, fps=FPS,
                                   bands_per_octave=BANDS_PER_OCTAVE, mul=MUL,
                                   add=ADD, norm_filters=NORM_FILTERS,
                                   online=ONLINE, window_sizes=WINDOW_SIZES,
                                   threshold=THRESHOLD, combine=COMBINE,
                                   smooth=SMOOTH, pre_avg=PRE_AVG,
                                   post_avg=POST_AVG, pre_max=PRE_MAX,
                                   post_max=POST_MAX)

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
        o = RnnOnsetDetector(args.input, **vars(args))
    else:
        # exit if no NN files are given
        if not args.nn_files:
            raise SystemExit('no NN model(s) given')

        w = Signal(args.input, mono=True, norm=args.norm, att=args.att)
        o = RnnOnsetDetector(w, window_sizes=WINDOW_SIZES, **vars(args))

    # save onset activations or detect onsets
    if args.save:
        # save activations
        o.save_activations(args.output, sep=args.sep)
    else:
        # save detections
        o.save_detections(args.output)

if __name__ == '__main__':
    main()
