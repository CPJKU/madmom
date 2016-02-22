#!/usr/bin/env python
# encoding: utf-8
"""
ComplexFlux onset detection algorithm.

"""

from __future__ import absolute_import, division, print_function

import argparse

from madmom.processors import IOProcessor, io_arguments
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.filters import FilterbankProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.spectrogram import (LogarithmicSpectrogramProcessor,
                                      SpectrogramDifferenceProcessor,
                                      LogarithmicFilteredSpectrogramProcessor)
from madmom.features import ActivationsProcessor
from madmom.features.onsets import SpectralOnsetProcessor, PeakPickingProcessor


def main():
    """ComplexFlux"""

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    The ComplexFlux program detects all onsets in an audio file according to
    the method described in:

    "Local Group Delay based Vibrato and Tremolo Suppression for Onset
     Detection"
    Sebastian Böck and Gerhard Widmer.
    Proceedings of the 13th International Society for Music Information
    Retrieval Conference (ISMIR), 2013.

    This program can be run in 'single' file mode to process a single audio
    file and write the detected onsets to STDOUT or the given output file.

    $ ComplexFlux single INFILE [-o OUTFILE]

    If multiple audio files should be processed, the program can also be run
    in 'batch' mode to save the detected onsets to files with the given suffix.

    $ ComplexFlux batch [-o OUTPUT_DIR] [-s OUTPUT_SUFFIX] LIST OF FILES

    If no output directory is given, the program writes the files with the
    detected onsets to same location as the audio files.

    The 'pickle' mode can be used to store the used parameters to be able to
    exactly reproduce experiments.

    ''')
    # version
    p.add_argument('--version', action='version', version='ComplexFlux.2014')
    # add arguments
    io_arguments(p, output_suffix='.onsets.txt')
    ActivationsProcessor.add_arguments(p)
    SignalProcessor.add_arguments(p, norm=False, gain=0)
    FramedSignalProcessor.add_arguments(p, fps=200, online=False)
    FilterbankProcessor.add_arguments(p, num_bands=24, fmin=30, fmax=17000,
                                      norm_filters=False)
    LogarithmicSpectrogramProcessor.add_arguments(p, log=True, mul=1, add=1)
    SpectrogramDifferenceProcessor.add_arguments(p, diff_ratio=0.5,
                                                 diff_max_bins=3)
    PeakPickingProcessor.add_arguments(p, threshold=0.25, pre_max=0.01,
                                       post_max=0.05, pre_avg=0.15, post_avg=0,
                                       combine=0.03, delay=0)
    # parse arguments
    args = p.parse_args()
    # switch to offline mode
    if args.norm:
        args.online = False
    # print arguments
    if args.verbose:
        print(args)

    # input processor
    if args.load:
        # load the activations from file
        in_processor = ActivationsProcessor(mode='r', **vars(args))
    else:
        # define processing chain
        sig = SignalProcessor(num_channels=1, **vars(args))
        frames = FramedSignalProcessor(**vars(args))
        # add a STFT processor so that we can set the circular shift needed for
        # correct phase and local group delay
        stft = ShortTimeFourierTransformProcessor(circular_shift=True,
                                                  **vars(args))
        spec = LogarithmicFilteredSpectrogramProcessor(**vars(args))
        odf = SpectralOnsetProcessor(onset_method='complex_flux', **vars(args))
        in_processor = [sig, frames, stft, spec, odf]

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
