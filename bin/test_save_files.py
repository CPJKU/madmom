#!/usr/bin/env python
# encoding: utf-8
"""
Script for testing RNNLIB .save files.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import os
import argparse
from madmom.utils.helpers import files, combine_activations
from madmom.utils.rnnlib import RnnConfig


def parser():
    # define parser
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    Tests .save files produced by RNNLIB.

    """)
    # general options
    p.add_argument('files', nargs='+', help='files to be processed')
    p.add_argument('-v', dest='verbose', action='count', help='increase verbosity level')
    p.add_argument('-o', dest='output', default=None, help='output directory')
    p.add_argument('--sep', action='store', default='', help='separator for saving/loading the activation functions [default=\'\' (numpy binary format)]')
    p.add_argument('--ext', action='store', default='.activations', help='separator for saving/loading the activation functions [default=\'\' (numpy binary format)]')
    p.add_argument('--threads', action='store', type=int, default=2, help='number of threads [default=2]')
    p.add_argument('--set', action='store', type=str, default='test', help='use this set (train, val, test) [default=test]')
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

    # test all .save files
    test_dirs = []
    for nn_file in files(args.files, '.save'):
        # create a RnnConfig object
        nn = RnnConfig(nn_file)
        # test the given set
        out_dir = nn.test(file_set=args.set, threads=args.threads, sep=args.sep)
        # append the output directory to the list
        test_dirs.append(out_dir)
    # overwrite the input files with the directories to be tested
    if test_dirs:
        args.files = test_dirs

    # combine the activations
    if args.output:
        # create output directory
        try:
            os.mkdir(args.output)
        except OSError:
            # directory exists already
            pass
        combine_activations(files(args.files, args.ext), args.output, ext=args.ext, sep=args.sep)


if __name__ == '__main__':
    main()
