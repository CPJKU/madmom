#!/usr/bin/env python
# encoding: utf-8
"""
The DownBeatTracker program detects all (down-)beats in an audio file.
It needs beat-synchronised features (or features + beats) as input and
uses GMMs to get an observation likelihood function.

"""

from __future__ import absolute_import, division, print_function

import glob
import argparse
import warnings
import os.path
import numpy as np
from madmom.processors import (IOProcessor, io_arguments)
from madmom.features.downbeats import BeatSyncProcessor_gmm, LoadBeatsProcessor
from madmom.utils import search_files, match_file
from madmom.audio.spectrogram import (FilteredSpectrogramProcessor,
                                      LogarithmicSpectrogramProcessor,
                                      SpectrogramDifferenceProcessor,
                                      MultiBandSpectrogramProcessor)
from madmom.features import ActivationsProcessor
from madmom.features.downbeats import GMMBarProcessor
BEAT_DIV = 4
MODEL_PATH = '/home/flokadillo/diss/src/python/madmom/madmom/models/downbeats/2016'


def match_files(files, input_suffix, beat_suffix):
    """
    Find all matching pairs of audio/feature files and beat file

    :param files:               list of filenames
    :param input_suffix:        suffix of input files
    :param beat_suffix:         suffix of beat files
    :return matched_input_files:list of input files
    :return matched_beat_files: list of beat files

    """
    matched_input_files = []
    matched_beat_files = []
    input_files = search_files(files, input_suffix)
    beat_files = search_files(files, beat_suffix)
    # check if each input file has a match in beat_files
    for num_file, in_file in enumerate(input_files):
        matches = match_file(in_file, beat_files, input_suffix, beat_suffix)
        if len(matches) > 1:
            # exit if multiple detections were found
            raise SystemExit("multiple beat annotations for %s "
                             "found" % in_file)
        elif len(matches) == 0:
            # output a warning if no detections were found
            warnings.warn(" can't find beat detections for %s" % in_file)
            continue
        else:
            # use the first (and only) matched detection file
            matched_input_files.append(in_file)
            matched_beat_files.append(matches[0])
    return matched_input_files, matched_beat_files


def main():
    """DownBeatTracker"""

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    The DownBeatTracker program detects all (down-)beats in an audio file.
    It needs beat-synchronised features (or features + beats) as input and
    uses GMMs to get an observation likelihood function.

    This program can be run in 'single' file mode to process a single audio
    file and write the detected beats to STDOUT or the given output file.

    $ DownBeatTracker single INFILE [-o OUTFILE]

    If multiple audio files should be processed, the program can also be run
    in 'batch' mode to save the detected beats to files with the given suffix.

    $ DownBeatTracker batch [-o OUTPUT_DIR] [-s OUTPUT_SUFFIX] LIST OF FILES

    If no output directory is given, the program writes the files with the
    detected beats to same location as the audio files.

    The 'pickle' mode can be used to store the used parameters to be able to
    exactly reproduce experiments.

    ''')
    # version
    p.add_argument('--version', action='version',
                   version='DownBeatTracker2.2015')
    p.add_argument('-lb', dest='load_beats', action='store_true',
                   default=True, help='load beats from file [default=%('
                                      'default)s]')
    p.add_argument('--save_raw', action='store_true', default=False,
                   help='save the (raw, non beat-syncronized) activations '
                   'to file')
    p.add_argument('-bs', dest='beat_suffix', type=str, default='.beats',
                   help='suffix of beat annotation files '
                   '[default=%(default)s]', action='store')
    p.add_argument('--div', dest='beat_div', type=int, default=BEAT_DIV,
                   help='number of beat subdivisions '
                   '[default=%(default)d]', action='store')
    p.add_argument('-is', dest='input_suffix', type=str, default='.npz',
                   help='suffix of input files [default=%(default)s]',
                   action='store')
    p.add_argument('-mop', dest='model_path', type=str, default=MODEL_PATH,
                   help='path where models (*.pkl) are located [default=%('
                        'default)s]',
                   action='store')
    p.add_argument('--load_hmm_in', action='store_true', default=False,
                   help='load the hmm input activations from file')

    # add processor arguments
    io_arguments(p, output_suffix='.beats.txt')
    ActivationsProcessor.add_arguments(p)
    GMMBarProcessor.add_arguments(p)
    # parse arguments
    args = p.parse_args()
    # set immutable defaults
    args.fps = 100
    args.pattern_files = glob.glob(os.path.join(args.model_path, "gmm*.pkl"))
    # print arguments
    if args.verbose:
        print(args)

    # input downbeat feature processor
    if args.load:
        # load the features from file
        frame_feature_extractor = [ActivationsProcessor(
            mode='r', **vars(args))]
    else:
        # define an input processor
        filt = FilteredSpectrogramProcessor(num_bands=12, **vars(args))
        log = LogarithmicSpectrogramProcessor(**vars(args))
        diff = SpectrogramDifferenceProcessor(**vars(args))
        mb = MultiBandSpectrogramProcessor(crossover_frequencies=[270],
                                           **vars(args))
        frame_feature_extractor = [filt, log, diff, mb]

    # input beat times processor
    if args.load_beats is None:
        # TODO: define an input processor to detect beats
        raise NotImplementedError('implement feature extraction')
    else:
        # load the beats from file
        # divide files into input and beat files
        args.files, beat_files = match_files(args.files, args.input_suffix,
                                             args.beat_suffix)
        beat_extractor = LoadBeatsProcessor(beat_files, args.beat_suffix)

    # input processor for downbeat HMM
    if args.load_hmm_in:
        # load the activations from file
        input_hmm = ActivationsProcessor(mode='r', **vars(args))
    else:
        input_hmm = BeatSyncProcessor_gmm(frame_feature_extractor,
                                          beat_extractor,
                                          beat_subdivision=args.beat_div,
                                          sum_func=np.mean, fps=args.fps)

    # output processor
    if args.save:
        # save the RNN/GMM downbeat activations to file
        out_processor = ActivationsProcessor(mode='w', **vars(args))
    else:
        # downbeat processor
        downbeat_processor = GMMBarProcessor(
            pattern_change_prob=0., **vars(args))
        if args.downbeats:
            # simply write the timestamps
            from madmom.utils import write_events as writer
        else:
            # borrow the note writer for outputting timestamps + beat numbers
            from madmom.features.notes import write_notes as writer
        # sequentially process them
        out_processor = [downbeat_processor, writer]

    # create an IOProcessor
    processor = IOProcessor(input_hmm, out_processor)

    # and call the processing function
    args.func(processor, **vars(args))


if __name__ == '__main__':
    main()
