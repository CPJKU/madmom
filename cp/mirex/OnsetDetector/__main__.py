#!/usr/bin/env python
# encoding: utf-8
"""
Copyright (c) 2012-2013 Sebastian Böck <sebastian.boeck@jku.at>

Redistribution in any form is not permitted!
"""

import os
import tempfile
import numpy as np

FPS = 100
BANDS_PER_OCTAVE = 6
MUL = 1
ADD = 1


def parser():
    import argparse
    import cp.utils.params

    # define parser
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    If invoked without any parameters, the software detects all onsets in
    the given input (file) and writes them to the output (file).
    ''')
    # mirex options
    cp.utils.params.add_mirex_io(p)
    # add onset detection related options to the existing parser
    nn = p.add_argument_group('neural network arguments')
    nn.add_argument('--threads', action='store', type=int, default=2, help='number of threads [default=2]')
    nn.add_argument('--configs', action='store', type=str, default='%s/configs' % os.path.dirname(__file__), help='configuration directory [default="configs"]')
    nn.add_argument('--nc_file', action='store', type=str, default=tempfile.mkstemp()[1], help='temporary file')
    # add other argument groups
    cp.utils.params.add_audio_arguments(p, norm=False)
    cp.utils.params.add_onset_arguments(p, io=True, threshold=0.3, combine=0.04, smooth=0.07, pre_avg=0, post_avg=0, pre_max=1. / FPS, post_max=1. / FPS)
    # version
    p.add_argument('--version', action='version', version='OnsetDetector.2013')
    # parse arguments
    args = p.parse_args()
    # set some defaults
    args.fps = FPS
    args.online = False
    # print arguments
    if args.verbose:
        print args
    # return
    return args


def main():
    """OnsetDetector.2013"""
    from cp.audio.wav import Wav
    from cp.audio.spectrogram import LogFiltSpec
    from cp.audio.onset_detection import Onset
    from cp.utils.helpers import files
    from cp.utils.rnnlib import create_nc_file, test_nc_files

    # parse arguments
    args = parser()

    # load or create onset activations
    if args.load:
        # load activations
        o = Onset(args.input, args.fps, args.online, args.sep)
    else:
        # create a Wav object
        w = Wav(args.input, fps=args.fps, mono=True, norm=args.norm)
        # 1st spec
        w.frame_size = 1024
        s = LogFiltSpec(w, bands_per_octave=BANDS_PER_OCTAVE, mul=MUL, add=ADD)
        nc_data = np.hstack((s.spec, s.pos_diff))
        # 2nd spec
        w.frame_size = 2048
        s = LogFiltSpec(w, bands_per_octave=BANDS_PER_OCTAVE, mul=MUL, add=ADD)
        nc_data = np.hstack((nc_data, s.spec, s.pos_diff))
        # 3rd spec
        w.frame_size = 4096
        s = LogFiltSpec(w, bands_per_octave=BANDS_PER_OCTAVE, mul=MUL, add=ADD)
        nc_data = np.hstack((nc_data, s.spec, s.pos_diff))
        # create a fake onset vector
        nc_targets = np.zeros(w.num_frames)
        nc_targets[0] = 1
        # create a .nc file
        create_nc_file(args.nc_file, nc_data, nc_targets)
        # test the file against all saved neural nets
        nn_files = files(args.configs, '.save')
        # Note: test_nc_files() always expects a list of .nc_files
        acts = test_nc_files([args.nc_file], nn_files, threads=args.threads, verbose=(args.verbose >= 2))
        # create an Onset object with the first activations of the list
        o = Onset(acts[0], args.fps, args.online)

    # save onset activations or detect onsets
    if args.save:
        # save activations
        o.save_activations(args.output, sep=args.sep)
    else:
        # detect the onsets
        o.detect(args.threshold, combine=args.combine, delay=args.delay, smooth=args.smooth,
                 pre_avg=args.pre_avg, post_avg=args.post_avg, pre_max=args.pre_max, post_max=args.post_max)
        # write the onsets to output
        o.write(args.output)

    # clean up
    os.remove(args.nc_file)

if __name__ == '__main__':
    main()
