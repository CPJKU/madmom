#!/usr/bin/env python
# encoding: utf-8
"""
Script for altering (ground truth) annotations.

"""

import numpy as np
import argparse

from madmom.utils import search_files, match_file, load_events


def main():
    """
    Simple tool to alter annotations.

    """
    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    The script alters (ground truth) annotations.

    """)
    # files used for evaluation
    p.add_argument('files', nargs='*',
                   help='files (or folder) to be corrected')
    p.add_argument('--suffix', default=None,
                   help='file suffix of the ground-truth files')
    # output directory
    p.add_argument('-o', dest='output', default=None,
                   help='output directory')

    # annotations correction methods
    g = p.add_argument_group('timestamp correction methods')
    g.add_argument('--smooth', default=None, type=float,
                   help='smooth the annotations [seconds]')
    g.add_argument('--quantize', default=None,
                   help='quantize the annotations to this resolution (given '
                        'in seconds), or to the time instants given in this '
                        'file')
    g.add_argument('--offset', default=None,
                   help='suffix of the offsets files (shift + stretch)')
    g.add_argument('--shift', default=None, type=float,
                   help='shift the annotations [seconds]')
    g.add_argument('--stretch', default=None, type=float,
                   help='stretch the annotations [factor]')
    # verbose
    p.add_argument('-v', dest='verbose', action='count',
                   help='increase verbosity level')
    # parse the arguments
    args = p.parse_args()
    # print the args
    if args.verbose >= 2:
        print args

    # correct all files
    for infile in search_files(args.files, args.suffix):
        if args.verbose:
            print infile

        # offset
        if args.offset:
            if isinstance(args.offset, basestring):
                # get the offset from a file
                correct = match_file(infile, args.files, suffix=args.suffix,
                                     match_suffix=args.offset)[0]
                with open(correct, 'rb') as cf:
                    for l in cf:
                        # sample line: 0.0122817938+0.9999976816*T
                        shift, stretch = l.split('+')
                        args.shift = float(shift)
                        args.stretch = float(stretch.split('*')[0])
        # smooth
        if args.smooth:
            raise NotImplementedError
        # quantize
        quantized = None
        if args.quantize:
            try:
                # load quantisation timestamps
                quantized = load_events(args.quantize)
            except IOError:
                quantized = float(args.quantize)

        # write the corrected file
        with open("%s.corrected" % infile, 'wb') as o:
            # process all events in the ground-truth file
            with open(infile) as i:
                for l in i:
                    # strip line
                    l.strip()
                    # skip comments
                    if l.startswith('#'):
                        # copy comments as is
                        o.write('%s\n' % l)
                    # alter the first column
                    # TODO: extend to alter all timestamp columns
                    else:
                        rest = None
                        try:
                            # extract the timestamp
                            timestamp, rest = l.split()
                            timestamp = float(timestamp)
                        except ValueError:
                            # only a timestamp given
                            timestamp = float(l)
                        # stretch
                        if args.stretch:
                            timestamp *= args.stretch
                        # shift
                        if args.shift:
                            timestamp += args.shift
                        # quantize
                        if isinstance(quantized, np.ndarray):
                            # get the closest match
                            timestamp = quantized[np.argmin(np.abs(quantized -
                                                                   timestamp))]
                        elif isinstance(quantized, float):
                            # set to the grid with the given resolution
                            timestamp /= quantized
                            timestamp = np.round(timestamp)
                            timestamp *= quantized

                        # skip negative timestamps
                        if timestamp < 0:
                            continue

                        # write the new timestamp
                        if rest:
                            o.write('%s\t%s\n' % (timestamp, rest))
                        else:
                            o.write('%s\n' % timestamp)

if __name__ == '__main__':
    main()
