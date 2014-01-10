#!/usr/bin/env python
# encoding: utf-8
"""
Script for creating .nc files for use with RNNLIB.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import numpy as np
from madmom.ml.rnnlib import create_nc_file


def parser():
    """
    Create a parser and parse the arguments.

    :return: the parsed arguments
    """
    import argparse
    import madmom.utils.params

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    This module creates .nc files to be used by RNNLIB.

    """)
    # general options
    p.add_argument('files', nargs='+', help='files to be processed')
    p.add_argument('-v', dest='verbose', action='count',
                   help='increase verbosity level')
    p.add_argument('-o', dest='output', default=None,
                   help='output directory')
    p.add_argument('-p', dest='path', default=None,
                   help='path for audio files')
    annotations = ['.onsets', '.beats', '.notes']
    p.add_argument('-a', dest='annotations', default=annotations,
                   help='annotations to use [default=%s]' % annotations)
    p.add_argument('--spec', dest='specs', default=None, type=int,
                   action='append', help='spectrogram size(s) to use')
    p.add_argument('--split', default=None, type=float,
                   help='split files every N seconds')
    # add onset detection related options to the existing parser
    madmom.utils.params.add_audio_arguments(p, fps=100, norm=False,
                                            window=None)
    madmom.utils.params.add_spec_arguments(p)
    madmom.utils.params.add_filter_arguments(p, bands=12)
    madmom.utils.params.add_log_arguments(p, log=True, mul=5, add=1)
    # parse arguments
    args = p.parse_args()
    if args.specs is None:
        args.specs = [1024, 2048, 4096]
    # print arguments
    if args.verbose >= 2:
        print args
    # return
    return args


def main():
    """Example script for generating .nc files."""
    # parse arguments
    args = parser()

    import os
    from madmom.audio.wav import Wav
    from madmom.audio.spectrogram import LogFiltSpec
    from madmom.utils.helpers import (files, match_file, load_events,
                                      quantize_events)

    # treat all files as annotation files and try to create .nc files
    for f in args.files:
        # split the extension of the input file
        annotation = os.path.splitext(f)[1]
        # continue with next file if annotation is not in the list
        if annotation not in args.annotations:
            continue
        # ok, a valid annotations file
        if args.verbose:
            print f
        # get a list of wav files
        if args.path:
            # search the given path
            wav_files = files(args.path, '*.wav')
        else:
            # search for the wav files in the same path as the input file
            wav_files = files(os.path.dirname(os.path.abspath(f)), '*.wav')
        # get the matching wav file to the input file
        wav_files = match_file(f, wav_files, annotation, '.wav')
        # no wav file found
        if len(wav_files) < 1:
            continue
        # create a Wav object
        w = Wav(wav_files[0], mono=True, norm=args.norm)
        # spec
        nc_data = None
        for spec in args.specs:
            s = LogFiltSpec(w, frame_size=spec, fps=args.fps,
                            bands_per_octave=args.bands, fmin=args.fmin,
                            fmax=args.fmax, mul=args.mul, add=args.add,
                            ratio=args.ratio, norm_filters=args.norm_filters)
            if nc_data is None:
                nc_data = np.hstack((s.spec, s.pos_diff))
            else:
                nc_data = np.hstack((nc_data, s.spec, s.pos_diff))
        # targets
        if f.endswith('.notes'):
            # load notes
            from madmom.features.notes import load_notes
            notes = load_notes(f)
            targets = np.zeros((s.num_frames, 88))
            notes[:, 0] *= args.fps
            notes[:, 2] -= 21
            for note in notes:
                try:
                    targets[int(note[0]), int(note[2])] = 1
                except IndexError:
                    pass
        else:
            # load event (onset/beat)
            targets = load_events(f)
            targets = quantize_events(targets, args.fps, length=s.num_frames)
        # tags
        tags = "file=%s | fps=%s | specs=%s | bands=%s | fmin=%s | fmax=%s |" \
               "norm_filter=%s | log=%s | mul=%s | add=%s | ratio=%s" %\
               (f, args.fps, args.specs, args.bands, args.fmin, args.fmax,
                args.norm_filters, args.log, args.mul, args.add, args.ratio)
        # .nc file name
        if args.output:
            nc_file = "%s/%s" % (args.output, f)
        else:
            nc_file = "%s" % os.path.abspath(f)
        # split files
        if args.split is None:
            # create a .nc file
            create_nc_file(nc_file + '.nc', nc_data, targets, tags)
        else:
            # length of one part
            length = int(args.split * args.fps)
            # number of parts
            parts = int(np.ceil(s.num_frames / float(length)))
            digits = int(np.ceil(np.log10(parts + 1)))
            if digits > 4:
                raise ValueError('please chose longer splits')
            for i in range(parts):
                nc_part_file = "%s.part%04d.nc" % (nc_file, i)
                start = i * length
                stop = start + length
                if stop > s.num_frames:
                    stop = s.num_frames
                part_tags = "%s | part=%s | start=%s | stop=%s" %\
                            (tags, i, start, stop - 1)
                create_nc_file(nc_part_file, nc_data[start:stop],
                               targets[start:stop], part_tags)

if __name__ == '__main__':
    main()
