#!/usr/bin/env python
# encoding: utf-8
"""
SuperFlux onset detection algorithm.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""


def parser():
    """
    Create a parser and parse the arguments.

    :return: the parsed arguments
    """
    import argparse
    import madmom.utils.params

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    If invoked without any parameters, the software detects all onsets in the
    given input file and writes them to the output file with the SuperFlux
    algorithm introduced in:

    "Maximum Filter Vibrato Suppression for Onset Detection"
    by Sebastian Böck and Gerhard Widmer
    Proceedings of the 16th International Conference on Digital Audio Effects
    (DAFx-13), Maynooth, Ireland, September 2013

    ''')
    # general options
    madmom.utils.params.add_mirex_io(p)
    # add other argument groups
    madmom.utils.params.add_audio_arguments(p, fps=200, online=False)
    madmom.utils.params.add_spec_arguments(p)
    madmom.utils.params.add_filter_arguments(p, bands=24, norm_filters=False)
    madmom.utils.params.add_log_arguments(p, mul=1, add=1)
    madmom.utils.params.add_spectral_odf_arguments(p)
    madmom.utils.params.add_onset_arguments(p)
    madmom.utils.params.add_io_arguments(p)
    # version
    p.add_argument('--version', action='version', version='SuperFlux.2013')
    # parse arguments
    args = p.parse_args()
    # switch to offline mode
    if args.norm:
        args.online = False
        args.post_avg = 0
        args.post_max = 0
    # translate online/offline mode
    if args.online:
        args.origin = 'online'
    else:
        args.origin = 'offline'
    # print arguments
    if args.verbose:
        print args
    # return
    return args


def main():
    """SuperFlux.2013"""
    from madmom.audio.wav import Wav
    from madmom.audio.spectrogram import LogFiltSpec
    from madmom.features.onsets import SpectralOnsetDetection, Onset

    # parse arguments
    args = parser()

    # load or create onset activations
    if args.load:
        # load activations
        o = Onset(args.input, args.fps, args.online, args.sep)
    else:
        # create a Wav object
        w = Wav(args.input, mono=True, norm=args.norm, att=args.att)
        # create a Spectrogram object
        s = LogFiltSpec(w, frame_size=args.window, origin=args.origin,
                        fps=args.fps, mul=args.mul, add=args.add,
                        norm_filters=args.norm_filters)
        # create an SpectralOnsetDetection object
        # and perform detection function on the object
        act = SpectralOnsetDetection(s).superflux()
        # create an Onset object with the activations
        o = Onset(act, args.fps, args.online)

    # save onset activations or detect onsets
    if args.save:
        # save activations
        o.save_activations(args.output, sep=args.sep)
    else:
        # detect the onsets
        o.detect(args.threshold, combine=args.combine, delay=args.delay,
                 smooth=args.smooth, pre_avg=args.pre_avg,
                 post_avg=args.post_avg, pre_max=args.pre_max,
                 post_max=args.post_max)
        # write the onsets to output
        o.write(args.output)

if __name__ == '__main__':
    main()
