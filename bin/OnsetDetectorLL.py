#!/usr/bin/env python
# encoding: utf-8
"""
@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import argparse
import glob

from madmom.utils import write_events, io_arguments
from madmom import SequentialProcessor, IOProcessor
from madmom.audio.signal import SignalProcessor
from madmom.audio.spectrogram import StackSpectrogramProcessor
from madmom.ml.rnn import RNNProcessor
from madmom.features.peak_picking import PeakPickingProcessor
from madmom.features import ActivationsProcessor
from madmom import MODELS_PATH

# set the path to saved neural networks and generate lists of NN files
NN_FILES = glob.glob("%s/onsets_rnn_[1-8].npz" % MODELS_PATH)


def parser():
    """
    Create a parser and parse the arguments.

    :return: the parsed arguments

    """
    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    If invoked without any parameters, the software detects all onsets in the
    given input (file) and writes them to the output (file) with the algorithm
    introduced in:

    "Online Real-time Onset Detection with Recurrent Neural Networks"
    Sebastian Böck, Andreas Arzt, Florian Krebs and Markus Schedl
    Proceedings of the 15th International Conference on Digital Audio Effects
    (DAFx-12), 2012.

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
    io_arguments(p)
    # add other argument groups
    SignalProcessor.add_arguments(p, att=0)
    # rnn onset detection arguments
    RNNProcessor.add_arguments(p, nn_files=NN_FILES)
    PeakPickingProcessor.add_arguments(p, threshold=0.2, combine=0.03, delay=0)
    ActivationsProcessor.add_arguments(p)
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
    """OnsetDetectorLL.2013"""

    # parse arguments
    args = parser()
    # set the frame rate
    args.fps = 100

    # load or create onset activations
    if args.load:
        # load the activations
        act = ActivationsProcessor(mode='r', **vars(args))
        in_processor = SequentialProcessor([act])
    else:
        # signal handling processor
        sig = SignalProcessor(**vars(args))
        # parallel specs + stacking processor
        stack = StackSpectrogramProcessor(frame_sizes=[512, 1024, 2048],
                                          fps=args.fps, online=True, bands=6,
                                          norm_filters=True, mul=5, add=1,
                                          diff_ratio=0.25)
        # multiple RNN processor
        rnn = RNNProcessor(nn_files=args.nn_files,
                           num_threads=args.num_threads)
        # sequentially process everything
        in_processor = SequentialProcessor([sig, stack, rnn])

    # save onset activations or detect onsets
    if args.save:
        # save activations
        out_processor = ActivationsProcessor(mode='w', **vars(args))
    else:
        # detect the onsets
        pp = PeakPickingProcessor(pre_max=0.01, **vars(args))
        in_processor.append(pp)
        out_processor = write_events

    # process everything
    IOProcessor(in_processor, out_processor).process(args.input, args.output)

if __name__ == '__main__':
    main()
