#!/usr/bin/env python
# encoding: utf-8
"""
"""

import glob
import argparse

from madmom import MODELS_PATH
from madmom.processors import IOProcessor, io_arguments
from madmom.audio.signal import SignalProcessor
from madmom.audio.spectrogram import (LogarithmicFilteredSpectrogramProcessor,
                                      SpectrogramDifferenceProcessor,
                                      StackedSpectrogramProcessor)
from madmom.ml.rnn import RNNProcessor, average_predictions
from madmom.features import ActivationsProcessor
from madmom.features.onsets import PeakPickingProcessor


def main():
    """OnsetDetectorLL.2013"""

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    The software detects all onsets in an audio file with the algorithm
    described in:

    "Online Real-time Onset Detection with Recurrent Neural Networks"
    Sebastian Böck, Andreas Arzt, Florian Krebs and Markus Schedl
    Proceedings of the 15th International Conference on Digital Audio Effects
    (DAFx), 2012.

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
    # version
    p.add_argument('--version', action='version',
                   version='OnsetDetectorLL.2013')
    # input/output options
    io_arguments(p, output_suffix='.onsets.txt')
    ActivationsProcessor.add_arguments(p)
    # signal processing arguments
    SignalProcessor.add_arguments(p, norm=False, att=0)
    # peak picking arguments
    PeakPickingProcessor.add_arguments(p, threshold=0.23)

    # parse arguments
    args = p.parse_args()

    # set immutable defaults
    args.num_channels = 1
    args.sample_rate = 44100
    args.online = True
    args.fps = 100
    args.frame_size = [512, 1024, 2048]
    args.num_bands = 6
    args.fmin = 30
    args.fmax = 17000
    args.norm_filters = True
    args.log = True
    args.mul = 5
    args.add = 1
    args.diff_ratio = 0.25
    args.positive_diffs = True
    args.nn_files = glob.glob("%s/onsets_rnn_[1-8].npz" % MODELS_PATH)
    args.pre_max = 1. / args.fps
    args.post_max = 0
    args.post_avg = 0

    # print arguments
    if args.verbose:
        print args

    # input processor
    if args.load:
        # load the activations from file
        in_processor = ActivationsProcessor(mode='r', **vars(args))
    else:
        # define processing chain
        sig = SignalProcessor(**vars(args))
        # we need to define how specs and diffs should be stacked
        spec = LogarithmicFilteredSpectrogramProcessor(**vars(args))
        diff = SpectrogramDifferenceProcessor(**vars(args))
        stack = StackedSpectrogramProcessor(spectrogram=spec, difference=diff,
                                            **vars(args))
        # process everything with an RNN and average the predictions
        rnn = RNNProcessor(**vars(args))
        avg = average_predictions
        # sequentially process everything
        in_processor = [sig, stack, rnn, avg]

    # output processor
    if args.save:
        # save the RNN onset activations to file
        out_processor = ActivationsProcessor(mode='w', **vars(args))
    else:
        # perform peak picking on the onset activations
        peak_picking = PeakPickingProcessor(**vars(args))
        # output handler
        from madmom.utils import write_events as writer
        # sequentially process them
        out_processor = [peak_picking, writer]

    # create an IOProcessor
    processor = IOProcessor(in_processor, out_processor)

    # and call the processing function
    args.func(processor, **vars(args))


if __name__ == '__main__':
    main()
