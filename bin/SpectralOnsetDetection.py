#!/usr/bin/env python
# encoding: utf-8
"""
Spectral onset detection script.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import argparse

from madmom.processors import IOProcessor, io_arguments
from madmom.features import ActivationsProcessor
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.spectrogram import (ShortTimeFourierTransformProcessor,
                                      SpectrogramProcessor,
                                      FilteredSpectrogramProcessor,
                                      LogarithmicSpectrogramProcessor,
                                      SpectrogramDifferenceProcessor)
from madmom.features.onsets import SpectralOnsetProcessor, PeakPickingProcessor


def main():
    """Spectral onset detection script."""

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    The software detects all onsets in an audio file with a selectable
    algorithms. The parameters have to be set accordingly.

    ''')
    # add arguments
    io_arguments(p, output_suffix='.onsets.txt')
    ActivationsProcessor.add_arguments(p)
    SignalProcessor.add_arguments(p, norm=False, att=0)
    FramedSignalProcessor.add_arguments(p, fps=100, online=False)
    FilteredSpectrogramProcessor.add_arguments(p, num_bands=12, fmin=30,
                                               fmax=17000, norm_filters=False)
    LogarithmicSpectrogramProcessor.add_arguments(p, log=True, mul=1, add=1)
    SpectrogramDifferenceProcessor.add_arguments(p, diff_ratio=0.5,
                                                 positive_diffs=True)
    SpectralOnsetProcessor.add_arguments(p, onset_method='spectral_flux')
    PeakPickingProcessor.add_arguments(p, threshold=1.6, pre_max=0.01,
                                       post_max=0.05, pre_avg=0.15, post_avg=0,
                                       combine=0.03, delay=0)
    # parse arguments
    args = p.parse_args()
    # switch to offline mode
    if args.norm:
        args.online = False
    # add circular shift for correct phase and remove filterbank if needed
    if args.onset_method in ('phase_deviation', 'weighted_phase_deviation',
                             'normalized_weighted_phase_deviation',
                             'complex_domain', 'rectified_complex_domain'):
        args.circular_shift = True
        args.filterbank = None
    if args.onset_method in ('superflux', 'complex_flux'):
        raise SystemExit('Please use the dedicated onset detection script for '
                         '%s.' % args.onset_method)
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
        stft = ShortTimeFourierTransformProcessor(**vars(args))
        spec = SpectrogramProcessor(**vars(args))
        in_processor = [sig, frames, stft, spec]
        # append additional processors as needed
        if args.filterbank:
            filt = FilteredSpectrogramProcessor(**vars(args))
            in_processor.append(filt)
        if args.log:
            log = LogarithmicSpectrogramProcessor(**vars(args))
            in_processor.append(log)
        # define a spectral onset processor
        odf = SpectralOnsetProcessor(**vars(args))
        in_processor.append(odf)

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
