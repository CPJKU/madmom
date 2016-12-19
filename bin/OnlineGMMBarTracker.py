#!/usr/bin/env python
# encoding: utf-8
"""
GMMBarTracker beat tracking algorithm.

"""

from __future__ import absolute_import, division, print_function

import argparse
from madmom.processors import IOProcessor, io_arguments
from madmom.audio.signal import SignalProcessor
from madmom.features import ActivationsProcessor
from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
from madmom.features.downbeats import (DownbeatFeatureProcessor,
                                       BeatSyncProcessor, GMMBarProcessor)
#     OnlineDBNBarTrackingProcessor, GMMBarProcessor,
#     BarTrackerActivationsProcessor)
from madmom.processors import ParallelProcessor, SequentialProcessor
from madmom.utils import write_output as writer
from madmom.models import DOWNBEATS_GMM


def main():
    """DBNBeatTracker"""

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    The GMMBarTracker program detects all beats and downbeats in an audio file
    according to
    the method described in:

    ''')
    # version
    p.add_argument('--version', action='version',
                   version='GMMBarTracker.2016')
    # input/output options
    io_arguments(p, output_suffix='.beats.txt', online=True)
    ActivationsProcessor.add_arguments(p)
    # signal processing arguments
    SignalProcessor.add_arguments(p, norm=False, gain=0)
    # peak picking arguments
    DBNBeatTrackingProcessor.add_arguments(p)
    GMMBarProcessor.add_arguments(p)
    # OnlineDBNBarTrackingProcessor.add_arguments(p)

    # parse arguments
    args = p.parse_args()

    # set immutable arguments
    args.fps = 100

    # print arguments
    if args.verbose:
        print(args)

    # input processor
    downbeats_feats = DownbeatFeatureProcessor(fps=args.fps, num_bands=12)
    beats_rnn = RNNBeatProcessor(**vars(args))
    beats_dbn = DBNBeatTrackingProcessor(**vars(args))
    beat_processor = SequentialProcessor([beats_rnn, beats_dbn])
    feat_processor = ParallelProcessor([beat_processor, downbeats_feats])
    beat_sync = SequentialProcessor([feat_processor, BeatSyncProcessor(
        beat_subdivisions=args.beat_div, feat_dim=1, fps=args.fps)])
    # Process beat-synchronous features with a GMM
    # TODO: read num_div from model file
    gmm_bar_processor = SequentialProcessor([beat_sync, GMMBarProcessor(
        pattern_files=DOWNBEATS_GMM, pattern_change_prob=0.001, **vars(args))])
    # output processor
    # sequentially process everything
    # out_processor = [beat_processor, writer]

    # create an IOProcessor
    processor = IOProcessor(gmm_bar_processor, writer)

    # and call the processing function
    args.func(processor, **vars(args))


if __name__ == '__main__':
    main()
