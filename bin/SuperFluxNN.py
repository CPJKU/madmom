#!/usr/bin/env python
# encoding: utf-8
"""
SuperFlux with neural network based peak picking onset detection algorithm.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import argparse
import glob

from madmom.utils import write_events, io_arguments
from madmom import SequentialProcessor, IOProcessor
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.spectrogram import SpectrogramProcessor
from madmom.features.onsets import SpectralOnsetDetectionProcessor
from madmom.features.peak_picking import NNPeakPickingProcessor
from madmom.features import ActivationsProcessor
from madmom import MODELS_PATH

# define NN files
NN_FILES = glob.glob("%s/onsets_brnn_peak_picking_[1-8].npz" % MODELS_PATH)


def parser():
    """
    Create a parser and parse the arguments.

    :return: the parsed arguments

    """
    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    If invoked without any parameters, the software detects all onsets in the
    given input file and writes them to the output file with the algorithm
    introduced in:

    "Enhanced peak picking for onset detection with recurrent neural networks"
    Sebastian Böck, Jan Schlüter and Gerhard Widmer
    Proceedings of the 6th International Workshop on Machine Learning and
    Music (MML), 2013.

    Please note that this implementation uses 100 frames per second (instead
    of 200), because it is faster and produces highly comparable results.

    ''')
    # input/output options
    io_arguments(p)
    # add other argument groups
    SignalProcessor.add_arguments(p, att=0, norm=False)
    FramedSignalProcessor.add_arguments(p)
    SpectrogramProcessor.add_filter_arguments(p, bands=24, fmin=30, fmax=17000,
                                              norm_filters=False)
    SpectrogramProcessor.add_log_arguments(p, log=True, mul=1, add=1)
    SpectrogramProcessor.add_diff_arguments(p, diff_ratio=0.5, diff_max_bins=3)
    NNPeakPickingProcessor.add_arguments(p, nn_files=NN_FILES, threshold=0.4,
                                         smooth=0.07, combine=0.03, delay=0)
    ActivationsProcessor.add_arguments(p)
    # version
    p.add_argument('--version', action='version', version='SuperFluxNN')
    # parse arguments
    args = p.parse_args()
    # print arguments
    if args.verbose:
        print args
    # return
    return args


def main():
    """SuperFluxNN"""

    # parse arguments
    args = parser()

    # load or create onset activations
    if args.load:
        # load the activations
        act = ActivationsProcessor(mode='r', **vars(args))
        in_processor = SequentialProcessor([act])
    else:
        # create processors
        p1 = SignalProcessor(mono=True, **vars(args))
        p2 = FramedSignalProcessor(**vars(args))
        p3 = SpectrogramProcessor(**vars(args))
        p4 = SpectralOnsetDetectionProcessor(odf='superflux', **vars(args))
        # sequentially process everything
        in_processor = SequentialProcessor([p1, p2, p3, p4])

    # save onset activations or detect onsets
    if args.save:
        # save activations
        out_processor = ActivationsProcessor(mode='w', **vars(args))
    else:
        # peak-picking & output processor
        in_processor.append(NNPeakPickingProcessor(**vars(args)))
        out_processor = write_events

    # process everything
    IOProcessor(in_processor, out_processor).process(args.input, args.output)

if __name__ == '__main__':
    main()
