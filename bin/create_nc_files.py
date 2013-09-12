#!/usr/bin/env python
# encoding: utf-8
"""
Script for creating .nc files for use with RNNLIB.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import numpy as np
from cp.utils.rnnlib import create_nc_file


def parser():
    import argparse
    import cp.utils.params

    # define parser
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    This module creates .nc files or tests .save files used / produced by RNNLIB.

    """)
    # general options
    p.add_argument('files', nargs='+', help='files to be processed')
    p.add_argument('-v', dest='verbose', action='count', help='increase verbosity level')
    p.add_argument('-o', dest='output', default=None, help='output directory')
    p.add_argument('-p', dest='path', default=None, help='path for audio files')
    annotations = ['.onsets', '.beats']
    p.add_argument('-a', dest='annotations', default=annotations, help='annotations to use [default=%s]' % annotations)
    # add onset detection related options to the existing parser
    cp.utils.params.add_audio_arguments(p, fps=100, norm=False)
    cp.utils.params.add_spec_arguments(p)
    cp.utils.params.add_filter_arguments(p, bands=12)
    cp.utils.params.add_log_arguments(p, log=True, mul=5, add=1)
    # parse arguments
    args = p.parse_args()
    # print arguments
    if args.verbose >= 2:
        print args
    # return
    return args


def main():
    # parse arguments
    args = parser()

    import os
    from cp.audio.wav import Wav
    from cp.audio.spectrogram import LogFiltSpec
    from cp.utils.helpers import files, match_file, load_events, quantize_events

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
        w = Wav(wav_files[0], fps=args.fps, mono=True, norm=args.norm)
        # 1st spec
        w.frame_size = 1024
        s = LogFiltSpec(w, bands_per_octave=args.bands, fmin=args.fmin, fmax=args.fmax, mul=args.mul, add=args.add, ratio=args.ratio, norm_filter=args.norm_filter)
        nc_data = np.hstack((s.spec, s.pos_diff))
        # 2nd spec
        w.frame_size = 2048
        s = LogFiltSpec(w, bands_per_octave=args.bands, fmin=args.fmin, fmax=args.fmax, mul=args.mul, add=args.add, ratio=args.ratio, norm_filter=args.norm_filter)
        nc_data = np.hstack((nc_data, s.spec, s.pos_diff))
        # 3rd spec
        w.frame_size = 4096
        s = LogFiltSpec(w, bands_per_octave=args.bands, fmin=args.fmin, fmax=args.fmax, mul=args.mul, add=args.add, ratio=args.ratio, norm_filter=args.norm_filter)
        nc_data = np.hstack((nc_data, s.spec, s.pos_diff))
        # targets
        targets = load_events(f)
        targets = quantize_events(targets, args.fps, length=w.num_frames)
        # tags
        tags = tags = "file=%s | fps=%s | specs=%s | bands=%s | fmin=%s | fmax=%s | norm_filter=%s | log=%s | mul=%s | add=%s | ratio=%s" % (f, args.fps, [1024, 2048, 4096], args.bands, args.fmin, args.fmax, args.norm_filter, args.log, args.mul, args.add, args.ratio)
        # .nc file name
        if args.output:
            nc_file = "%s/%s.nc" % (args.output, f)
        else:
            nc_file = "%s.nc" % os.path.abspath(f)
        # create a .nc file
        create_nc_file(nc_file, nc_data, targets, tags)

if __name__ == '__main__':
    main()
