#!/usr/bin/env python
# encoding: utf-8
"""
TempoDetector tempo estimation algorithm.

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
from madmom.features.tempo import TempoEstimationProcessor, write_tempo


def main():
    """TempoDetector"""

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    The software detects the dominant tempi in an audio file by inferring it
    with comb filters from the beat activations produced by the algorithm
    described in:

    "Enhanced Beat Tracking with Context-Aware Neural Networks"
    Sebastian Böck and Markus Schedl
    Proceedings of the 14th International Conference on Digital Audio Effects
    (DAFx), 2011.

    Instead of using the originally proposed auto-correlation method to build
    a tempo histogram, a new method based on comb filters is used:

    "Accurate Tempo Estimation based on Recurrent Neural Networks and
     Resonating Comb Filters"
    Sebastian Böck, Florian Krebs and Gerhard Widmer
    Proceedings of the 16th International Society for Music Information
    Retrieval Conference (ISMIR), 2015.

    The old behaviour can be restored by using the `--method acf` switch.

    ''')
    # version
    p.add_argument('--version', action='version', version='TempoDetector')
    # input/output options
    io_arguments(p, output_suffix='.bpm.txt')
    ActivationsProcessor.add_arguments(p)
    # signal processing arguments
    SignalProcessor.add_arguments(p, norm=False, att=0)
    # tempo arguments
    TempoEstimationProcessor.add_arguments(p)
    # mirex stuff
    g = p.add_mutually_exclusive_group()
    g.add_argument('--mirex', dest='tempo_format',
                   action='store_const', const='mirex',
                   help='use the MIREX output format (lower tempo first)')
    g.add_argument('--all', dest='tempo_format',
                   action='store_const', const='all',
                   help='output all detected tempi in raw format')

    # parse arguments
    args = p.parse_args()

    # set immutable defaults
    args.num_channels = 1
    args.sample_rate = 44100
    args.fps = 100
    args.frame_size = [1024, 2048, 4096]
    args.num_bands = 3
    args.fmin = 30
    args.fmax = 17000
    args.norm_filters = True
    args.log = True
    args.mul = 1
    args.add = 1
    args.diff_ratio = 0.5
    args.positive_diffs = True
    args.nn_files = glob.glob("%s/beats/2013/beats_blstm_[1-8].npz" %
                              MODELS_PATH)

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
        # process everything with a RNN and average the predictions
        rnn = RNNProcessor(**vars(args))
        avg = average_predictions
        # sequentially process everything
        in_processor = [sig, stack, rnn, avg]

    # output processor
    if args.save:
        # save the RNN beat activations to file
        out_processor = ActivationsProcessor(mode='w', **vars(args))
    else:
        # tempo estimation based on the beat activation function
        tempo_estimator = TempoEstimationProcessor(**vars(args))
        # output handler
        if args.tempo_format == 'mirex':
            # output in the MIREX format (i.e. slower tempo first)
            from functools import partial
            writer = partial(write_tempo, mirex=True)
        elif args.tempo_format in ('raw', 'all'):
            # borrow the note writer for outputting multiple values
            from madmom.features.notes import write_notes as writer
        else:
            # normal output
            writer = write_tempo
        # sequentially process them
        out_processor = [tempo_estimator, writer]

    # create an IOProcessor
    processor = IOProcessor(in_processor, out_processor)

    # finally call the processing function (single/batch processing)
    args.func(processor, **vars(args))


if __name__ == '__main__':
    main()
