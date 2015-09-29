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
                                      LogarithmicFilteredSpectrogramProcessor,
                                      SpectrogramDifferenceProcessor,
                                      StackedSpectrogramProcessor)
from madmom.ml.rnn import RNNProcessor, average_predictions
from madmom.features import ActivationsProcessor
from madmom.features.onsets import PeakPickingProcessor
from madmom.features.notes import (write_midi, write_notes, write_frequencies,
                                   note_reshaper)


def main():
    """PianoTranscriptor.2014"""

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    The software detects all notes (onsets) in an audio file with the algorithm
    described in:

    "Polyphonic Piano Note Transcription with Recurrent Neural Networks"
    Sebastian Böck and Markus Schedl.
    Proceedings of the 37th International Conference on Acoustics, Speech and
    Signal Processing (ICASSP), 2012.

    Instead of 'LSTM' units, the current version uses 'tanh' units.

    ''')
    # version
    p.add_argument('--version', action='version',
                   version='PianoTranscriptor.2014')
    # input/output arguments
    io_arguments(p, output_suffix='.notes.txt')
    ActivationsProcessor.add_arguments(p)
    # signal processing arguments
    SignalProcessor.add_arguments(p, norm=False, att=0, start=True, stop=True)
    # peak picking arguments
    PeakPickingProcessor.add_arguments(p, threshold=0.35, smooth=0.09,
                                       combine=0.05)
    # midi arguments
    # import madmom.utils.midi as midi
    # midi.MIDIFile.add_arguments(p, length=0.6, velocity=100)
    p.add_argument('--midi', dest='output_format', action='store_const',
                   const='midi', help='save as MIDI')
    # mirex stuff
    p.add_argument('--mirex', dest='output_format', action='store_const',
                   const='mirex', help='use the MIREX output format')

    # parse arguments
    args = p.parse_args()

    # set immutable defaults
    args.num_channels = 1
    args.sample_rate = 44100
    args.online = True
    args.fps = 100
    args.frame_size = [1024, 2048, 4096]
    args.num_bands = 12
    args.fmin = 30
    args.fmax = 17000
    args.norm_filters = True
    args.log = True
    args.mul = 5
    args.add = 1
    args.diff_ratio = 0.5
    args.positive_diffs = True
    args.nn_files = glob.glob("%s/notes_brnn*npz" % MODELS_PATH)
    args.pre_max = 1. / args.fps
    args.post_max = 1. / args.fps

    # set the suffix for midi files
    if args.output_format == 'midi':
        args.output_suffix = '.mid'

    # print arguments
    if args.verbose:
        print args

    # TODO: remove this hack!
    args.fps = 100

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
        reshape = note_reshaper
        # sequentially process everything
        in_processor = [sig, stack, rnn, avg, reshape]

    # output processor
    if args.save:
        # save the RNN note activations to file
        out_processor = ActivationsProcessor(mode='w', **vars(args))
    else:
        # perform peak picking of the detection function
        peak_picking = PeakPickingProcessor(**vars(args))
        # output everything in the right format
        if args.output_format is None:
            output = write_notes
        elif args.output_format == 'midi':
            output = write_midi
        elif args.output_format == 'mirex':
            output = write_frequencies
        else:
            raise ValueError('unknown output format: %s' % args.output_format)
        out_processor = [peak_picking, output]

    # create an IOProcessor
    processor = IOProcessor(in_processor, out_processor)

    # and call the processing function
    args.func(processor, **vars(args))


if __name__ == '__main__':
    main()
