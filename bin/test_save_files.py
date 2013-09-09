#!/usr/bin/env python
# encoding: utf-8
"""
Script for testing RNNLIB .save files.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""
import argparse

from cp.utils.helpers import files
from cp.utils.rnnlib import test_save_files


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
    nn_files = files(args.files, '.save')
    test_save_files(nn_files, out_dir=args.output, file_set=args.set, threads=args.threads, sep=args.sep)

if __name__ == '__main__':
    main()
