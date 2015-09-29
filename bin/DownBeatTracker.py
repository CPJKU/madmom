#!/usr/bin/env python
# encoding: utf-8
"""
"""

import glob
import argparse

from madmom import MODELS_PATH
from madmom.processors import IOProcessor, io_arguments
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.spectrogram import (FilteredSpectrogramProcessor,
                                      LogarithmicSpectrogramProcessor,
                                      SpectrogramDifferenceProcessor,
                                      MultiBandSpectrogramProcessor)
from madmom.features import ActivationsProcessor
from madmom.features.beats import DownbeatTrackingProcessor


GMM_FILE = glob.glob("%s/downbeat_ismir2013.pkl" % MODELS_PATH)[0]


def main():
    """DownBeatTracker"""

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    The software detects all (down-)beats in an audio file according to the
    method described in:

    "Rhythmic Pattern Modelling For Beat and Downbeat Tracking in Musical
     Audio"
    Florian Krebs, Sebastian Böck, and Gerhard Widmer
    Proceedings of the 14th International Society for Music Information
    Retrieval Conference (ISMIR), 2013.

    Instead of the originally proposed state space and transition model for the
    DBN, the following is used:

    "An efficient state space model for joint tempo and meter tracking"
    Florian Krebs, Sebastian Böck and Gerhard Widmer
    Proceedings of the 16th International Society for Music Information
    Retrieval Conference (ISMIR), 2015.

    ''')
    # version
    p.add_argument('--version', action='version',
                   version='DownBeatTracker.2015')
    # add arguments
    io_arguments(p, output_suffix='.beats.txt')
    ActivationsProcessor.add_arguments(p)
    SignalProcessor.add_arguments(p, norm=False, att=0)
    FramedSignalProcessor.add_arguments(p, fps=50, online=False)
    FilteredSpectrogramProcessor.add_arguments(p, num_bands=12, fmin=30,
                                               fmax=17000, norm_filters=False)
    LogarithmicSpectrogramProcessor.add_arguments(p, log=True, mul=1, add=1)
    SpectrogramDifferenceProcessor.add_arguments(p, diff_ratio=0.5,
                                                 positive_diffs=True)
    MultiBandSpectrogramProcessor.add_arguments(p, crossover_frequencies=[270])
    DownbeatTrackingProcessor.add_arguments(p, gmm_file=GMM_FILE,
                                            num_beats=None)
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
        # define an input processor
        sig = SignalProcessor(num_channels=1, **vars(args))
        frames = FramedSignalProcessor(**vars(args))
        filt = FilteredSpectrogramProcessor(**vars(args))
        log = LogarithmicSpectrogramProcessor(**vars(args))
        diff = SpectrogramDifferenceProcessor(**vars(args))
        mb = MultiBandSpectrogramProcessor(**vars(args))
        in_processor = [sig, frames, filt, log, diff, mb]

    # output processor
    if args.save:
        # save the RNN beat activations to file
        out_processor = ActivationsProcessor(mode='w', **vars(args))
    else:
        # downbeat processor
        downbeat_processor = DownbeatTrackingProcessor(**vars(args))
        if args.downbeats:
            # simply write the timestamps
            from madmom.utils import write_events as writer
        else:
            # borrow the note writer for outputting timestamps + beat numbers
            from madmom.features.notes import write_notes as writer
        # sequentially process them
        out_processor = [downbeat_processor, writer]

    # create an IOProcessor
    processor = IOProcessor(in_processor, out_processor)

    # and call the processing function
    args.func(processor, **vars(args))


if __name__ == '__main__':
    main()
