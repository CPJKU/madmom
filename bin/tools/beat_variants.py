#!/usr/bin/env python
# encoding: utf-8
"""
Script for creating variations of the given beat sequence.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import argparse
import os

from madmom.utils import load_events, write_events
from madmom.evaluation.beats import variations


def main():
    """
    Simple tool to create beat variations of the given sequence.

    """
    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    The script creates different versions of beat sequences read from .beats
    annotation files.

    """)
    # files to be processed
    p.add_argument('files', nargs='+', help='files to be processed')
    # output directory
    p.add_argument('-o', dest='output', default=None,
                   help='output directory')
    # parse arguments
    args = p.parse_args()

    # convert all files
    for f in args.files:
        # read in the annotations
        annotations = load_events(f)
        # variation names
        names = ['offbeat', 'double', 'half_odd', 'half_even', 'triple',
                 'third_first', 'third_second', 'third_third']
        # create the variations
        for i, var in enumerate(variations(annotations)):
            # determine the output file name
            if args.output:
                outfile = "%s/%s.%s" % (args.output, os.path.basename(f),
                                        names[i])
            else:
                outfile = "%s.%s" % (f, names[i])
            # write the new output file
            write_events(var, outfile)

if __name__ == '__main__':
    main()
