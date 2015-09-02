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
from madmom.ml.rnn import RNNProcessor
from madmom.features import ActivationsProcessor
from madmom.features.beats import (DBNBeatTrackingProcessor,
                                   MultiModelSelectionProcessor)

NN_FILES = glob.glob("%s/beats_blstm_[1-8].npz" % MODELS_PATH)
NN_REF_FILES = None


def main():
    """MMBeatTracker"""

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    The software detects all beats in an audio file according to the method
    described in:

    "A multi-model approach to beat tracking considering heterogeneous music
     styles"
    Sebastian Böck, Florian Krebs and Gerhard Widmer
    Proceedings of the 15th International Society for Music Information
    Retrieval Conference (ISMIR), 2014.

    Instead of the originally proposed transition model for the DBN, the
    following is used:

    "An efficient state space model for joint tempo and meter tracking"
    Florian Krebs, Sebastian Böck and Gerhard Widmer
    Proceedings of the 16th International Society for Music Information
    Retrieval Conference (ISMIR), 2015.

    ''')
    # version
    p.add_argument('--version', action='version', version='MMBeatTracker.2015')
    # input/output arguments
    io_arguments(p, output_suffix='.beats.txt')
    ActivationsProcessor.add_arguments(p)
    # signal processing arguments
    SignalProcessor.add_arguments(p, norm=False, att=0)
    FramedSignalProcessor.add_arguments(p, fps=100,
                                        frame_size=[1024, 2048, 4096])
    FilteredSpectrogramProcessor.add_arguments(p, num_bands=3, fmin=30,
                                               fmax=17000, norm_filters=True)
    LogarithmicSpectrogramProcessor.add_arguments(p, log=True, mul=1, add=1)
    SpectrogramDifferenceProcessor.add_arguments(p, diff_ratio=0.5,
                                                 positive_diffs=True)
    # RNN processing arguments (including option for reference files)
    g = RNNProcessor.add_arguments(p, nn_files=NN_FILES)
    g.add_argument('--nn_ref_files', action='append', type=str,
                   default=NN_REF_FILES,
                   help='Compare the predictions to these pre-trained '
                        'neural networks (multiple files can be'
                        'given, one file per argument) and choose the '
                        'most suitable one accordingly (i.e. the one '
                        'with the least deviation form the reference '
                        'model). If multiple reference files are'
                        'given, the predictions of the networks are '
                        'averaged first.')
    # beat tracking arguments
    DBNBeatTrackingProcessor.add_arguments(p)
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
                                            online=False, **vars(args))
        # process everything with an RNN and select the best predictions
        rnn = RNNProcessor(**vars(args))
        if args.nn_ref_files is None:
            # set the nn_ref_files the same as the nn_files, i.e. average them
            args.nn_ref_files = args.nn_files
        if args.nn_ref_files == args.nn_files:
            # if we don't have nn_ref_files given or they are the same as
            # the nn_files, set num_ref_predictions to 0
            num_ref_predictions = 0
        else:
            # set the number of reference files according to the length
            num_ref_predictions = len(args.nn_ref_files)
            # redefine the list of files to be tested
            args.nn_files = args.nn_ref_files + args.nn_files
        # define the selector
        selector = MultiModelSelectionProcessor(num_ref_predictions)
        # sequentially process everything
        in_processor = [sig, stack, rnn, selector]

    # output processor
    if args.save:
        # save the RNN beat activations to file
        out_processor = ActivationsProcessor(mode='w', **vars(args))
    else:
        # track the beats with a DBN
        beat_processor = DBNBeatTrackingProcessor(**vars(args))
        # output handler
        from madmom.utils import write_events as writer
        # sequentially process them
        out_processor = [beat_processor, writer]

    # create an IOProcessor
    processor = IOProcessor(in_processor, out_processor)

    # and call the processing function
    args.func(processor, **vars(args))


if __name__ == '__main__':
    main()
