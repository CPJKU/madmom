#!/usr/bin/env python
# encoding: utf-8
"""
LogFiltSpecFlux onset detection algorithm.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""


def parser():
    import argparse
    import cp.utils.params

    # define parser
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    If invoked without any parameters, the software detects all onsets in
    the given input file and writes them to the output file with the algorithm
    described in:

    "Evaluating the Online Capabilities of Onset Detection Methods"
    by Sebastian Böck, Florian Krebs and Markus Schedl
    in Proceedings of the 13th International Society for
    Music Information Retrieval Conference (ISMIR), 2012

    ''')
    # general options
    cp.utils.params.add_mirex_io(p)
    # add other argument groups
    cp.utils.params.add_audio_arguments(p, fps=100)
    cp.utils.params.add_spec_arguments(p)
    cp.utils.params.add_filter_arguments(p, bands=12, norm_filter=False)
    cp.utils.params.add_log_arguments(p, mul=1, add=1)
    cp.utils.params.add_onset_arguments(p, threshold=2.75)
    # version
    p.add_argument('--version', action='version', version='LogFiltSpecFlux.2013')
    # parse arguments
    args = p.parse_args()
    # switch to offline mode
    if args.norm:
        args.online = False
        args.post_avg = 0
        args.post_max = 0
    # print arguments
    if args.verbose:
        print args
    # return
    return args


def main():
    from cp.audio.wav import Wav
    from cp.audio.spectrogram import LogarithmicFilteredSpectrogram
    from cp.audio.onset_detection import SpectralOnsetDetection, Onset

    # parse arguments
    args = parser()

    # create a Wav object
    w = Wav(args.input, frame_size=args.window, online=args.online, mono=True, norm=args.norm, att=args.att, fps=args.fps)
    # create a Spectrogram object
    s = LogarithmicFilteredSpectrogram(w, mul=args.mul, add=args.add)
    # create an SpectralOnsetDetection object and perform detection function on the object
    act = SpectralOnsetDetection(s).sf()
    # create an Onset object with the activations
    o = Onset(act, args.fps, args.online)
    # detect the onsets
    o.detect(args.threshold, combine=args.combine, delay=args.delay, smooth=args.smooth,
             pre_avg=args.pre_avg, post_avg=args.post_avg, pre_max=args.pre_max, post_max=args.post_max)
    # write the onsets to output
    o.write(args.output)

if __name__ == '__main__':
    main()
