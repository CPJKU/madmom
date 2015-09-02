#!/usr/bin/env python
# encoding: utf-8
"""
@author: Filip Korzeniowski <filip.korzeniowski@jku.at>

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
from madmom.features.beats import CRFBeatDetectionProcessor


NN_FILES = glob.glob("%s/beats_blstm_[1-8].npz" % MODELS_PATH)


def main():
    """CRFBeatDetector"""

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    The software detects all beats in an audio file according to the method
    described in:

    "Probabilistic extraction of beat positions from a beat activation
     function"
    Filip Korzeniowski, Sebastian Böck and Gerhard Widmer
    In Proceedings of the 15th International Society for Music Information
    Retrieval Conference (ISMIR), 2014.

    Instead of using the auto-correlation method to determine the dominant
    interval, a new method based on comb filters is used to get multiple tempo
    hypotheses.

    "Accurate Tempo Estimation based on Recurrent Neural Networks and
     Resonating Comb Filters"
    Sebastian Böck, Florian Krebs and Gerhard Widmer
    Proceedings of the 16th International Society for Music Information
    Retrieval Conference (ISMIR), 2015.

    ''')
    # version
    p.add_argument('--version', action='version',
                   version='CRFBeatDetector.2015')
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
    # RNN processing arguments
    RNNProcessor.add_arguments(p, nn_files=NN_FILES)
    # beat tracking arguments
    CRFBeatDetectionProcessor.add_tempo_arguments(p)
    CRFBeatDetectionProcessor.add_arguments(p)

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
        # detect the beats with a CRF
        beat_processor = CRFBeatDetectionProcessor(**vars(args))
        # output handler
        from madmom.utils import write_events as writer
        # sequentially process them
        out_processor = [beat_processor, writer]

    # create an IOProcessor
    processor = IOProcessor(in_processor, out_processor)

    # and call the processing function
    args.func(processor, **vars(args))


if __name__ == "__main__":
    main()
