#!/usr/bin/env python
# encoding: utf-8
"""
DownBeatTracker (down-)beat tracking algorithm.

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


def main():
    """DownBeatTracker"""

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    The software detects all (down-)beats in an audio file according to the
    method described in:

    "Rhythmic Pattern Modelling for Beat and Downbeat Tracking in Musical
     Audio"
    Florian Krebs, Sebastian Böck and Gerhard Widmer
    Proceedings of the 14th International Society for Music Information
    Retrieval Conference (ISMIR), 2013.

    Instead of the originally proposed state space and transition model for the
    DBN, the following is used:

    "An Efficient State Space Model for Joint Tempo and Meter Tracking"
    Florian Krebs, Sebastian Böck and Gerhard Widmer
    Proceedings of the 16th International Society for Music Information
    Retrieval Conference (ISMIR), 2015.

    In its default setting, this script uses only two rhythmical patterns and
    allows tempo changes only at bar boundaries.

    ''')
    # version
    p.add_argument('--version', action='version', version='DownBeatTracker')
    # add arguments
    io_arguments(p, output_suffix='.beats.txt')
    ActivationsProcessor.add_arguments(p)
    SignalProcessor.add_arguments(p, norm=False, att=0)
    DownbeatTrackingProcessor.add_arguments(p)

    # parse arguments
    args = p.parse_args()

    # set immutable defaults
    args.num_channels = 1
    args.sample_rate = 44100
    args.fps = 50
    args.num_bands = 12
    args.fmin = 30
    args.fmax = 17000
    args.norm_filters = False
    args.log = True
    args.mul = 1
    args.add = 1
    args.diff_ratio = 0.5
    args.positive_diffs = True
    args.crossover_frequencies = [270]
    args.pattern_files = glob.glob("%s/downbeats/2013/*.pkl" % MODELS_PATH)

    # print arguments
    if args.verbose:
        print args

    # input processor
    if args.load:
        # load the activations from file
        in_processor = ActivationsProcessor(mode='r', **vars(args))
    else:
        # define an input processor
        sig = SignalProcessor(**vars(args))
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
