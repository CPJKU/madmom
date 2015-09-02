#!/usr/bin/env python
# encoding: utf-8
"""
@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import glob
import argparse

from madmom import MODELS_PATH
from madmom.processors import IOProcessor, io_arguments
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.spectrogram import (FilteredSpectrogramProcessor,
                                      LogarithmicSpectrogramProcessor,
                                      LogarithmicFilteredSpectrogramProcessor,
                                      SpectrogramDifferenceProcessor,
                                      StackedSpectrogramProcessor)
from madmom.ml.rnn import RNNProcessor, average_predictions
from madmom.features import ActivationsProcessor
from madmom.features.onsets import PeakPickingProcessor


NN_FILES = glob.glob("%s/onsets_rnn_[1-8].npz" % MODELS_PATH)


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
    FramedSignalProcessor.add_arguments(p, fps=100,
                                        frame_size=[512, 1024, 2048])
    FilteredSpectrogramProcessor.add_arguments(p, num_bands=6, fmin=30,
                                               fmax=17000, norm_filters=True)
    # TODO: make sure newer models are trained with mul=1
    LogarithmicSpectrogramProcessor.add_arguments(p, log=True, mul=5, add=1)
    # TODO: make sure newer models are trained with diff_ratio=0.5
    SpectrogramDifferenceProcessor.add_arguments(p, diff_ratio=0.25,
                                                 positive_diffs=True)
    # RNN processing arguments
    RNNProcessor.add_arguments(p, nn_files=NN_FILES)
    # peak picking arguments
    PeakPickingProcessor.add_arguments(p, threshold=0.23)
    # parse arguments
    args = p.parse_args()
    # print arguments
    if args.verbose:
        print args

    # input processor
    if args.load:
        # load the activations from file
        in_processor = ActivationsProcessor(mode='r', **vars(args))
    else:
        # define processing chain
        sig = SignalProcessor(num_channels=1, sample_rate=44100, **vars(args))
        # we need to define how specs and diffs should be stacked
        spec = LogarithmicFilteredSpectrogramProcessor(**vars(args))
        diff = SpectrogramDifferenceProcessor(**vars(args))
        stack = StackedSpectrogramProcessor(spectrogram=spec, difference=diff,
                                            online=True, **vars(args))
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
        peak_picking = PeakPickingProcessor(pre_max=1. / args.fps, post_max=0,
                                            post_avg=0, **vars(args))
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
