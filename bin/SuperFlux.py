#!/usr/bin/env python
# encoding: utf-8
"""
SuperFlux onset detection algorithm.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import argparse

from madmom import IOProcessor, io_arguments
from madmom.features import ActivationsProcessor
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.spectrogram import SpectrogramProcessor
from madmom.features.onsets import SpectralOnsetProcessor, PeakPickingProcessor


def main():
    """SuperFlux.2014"""

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    The software detects all onsets in an audio file with the SuperFlux
    algorithm described in:

    "Maximum Filter Vibrato Suppression for Onset Detection"
    Sebastian Böck and Gerhard Widmer
    Proceedings of the 16th International Conference on Digital Audio Effects
    (DAFx), 2013.

    ''')
    # version
    p.add_argument('--version', action='version', version='SuperFlux.2014')
    # add arguments
    io_arguments(p, suffix='.onsets.txt')
    ActivationsProcessor.add_arguments(p)
    SignalProcessor.add_arguments(p, norm=False, att=0)
    FramedSignalProcessor.add_arguments(p, fps=200, online=False)
    SpectrogramProcessor.add_filter_arguments(p, bands=24, fmin=30, fmax=17000,
                                              norm_filters=False)
    SpectrogramProcessor.add_log_arguments(p, log=True, mul=1, add=1)
    SpectrogramProcessor.add_diff_arguments(p, diff_ratio=0.5, diff_max_bins=3)
    PeakPickingProcessor.add_arguments(p, threshold=1.1, pre_max=0.01,
                                       post_max=0.05, pre_avg=0.15, post_avg=0,
                                       combine=0.03, delay=0)
    # parse arguments
    args = p.parse_args()
    # switch to offline mode
    if args.norm:
        args.online = False
    # print arguments
    if args.verbose:
        print args

    # input processor
    if args.load:
        # load the activations from file
        in_processor = ActivationsProcessor(mode='r', **vars(args))
    else:
        # define processing chain
        sig = SignalProcessor(num_channels=1, **vars(args))
        frames = FramedSignalProcessor(**vars(args))
        spec = SpectrogramProcessor(**vars(args))
        odf = SpectralOnsetProcessor(onset_method='superflux', **vars(args))
        in_processor = [sig, frames, spec, odf]

    # output processor
    if args.save:
        # save the onset activations to file
        out_processor = ActivationsProcessor(mode='w', **vars(args))
    else:
        # perform peak picking of the onset function
        peak_picking = PeakPickingProcessor(**vars(args))
        from madmom.utils import write_events as writer
        # sequentially process them
        out_processor = [peak_picking, writer]

    # create an IOProcessor
    processor = IOProcessor(in_processor, out_processor)

    # and call the processing function
    args.func(processor, **vars(args))


if __name__ == '__main__':
    main()
