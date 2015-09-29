#!/usr/bin/env python
# encoding: utf-8
"""
Script for renumbering beat annotation files.

"""

import argparse
import os
import math


def main():
    """
    Simple beat renumbering tool.

    """
    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    The script converts .beats annotation files without bar.beat information
    into the desired format.

    """)
    # files to be processed
    p.add_argument('files', nargs='+', help='files to be processed')
    # output directory
    p.add_argument('-o', dest='output', default=None,
                   help='output directory')
    p.add_argument('--ext', dest='extension', default=None,
                   help='append the given file extension')
    p.add_argument('--bar', default=1, type=int,
                   help='manually set the start bar counter')
    p.add_argument('--beat', default=1, type=int,
                   help='manually set the start beat counter')
    p.add_argument('--meter', default=None, type=int,
                   help='manually set the meter (bar length in beats)')
    # parse arguments
    args = p.parse_args()

    # never overwrite the annotation files
    if args.extension is None and args.output is None:
        raise SystemExit('Can not overwrite the annotation files. Please chose'
                         ' either an extension or another output directory.')

    # convert all files
    for f in args.files:
        # determine the output file name
        if args.output:
            outfile = "%s/%s" % (args.output, os.path.basename(f))
        else:
            outfile = f
        # add the extension if needed
        if args.extension:
            outfile += args.extension
        # write the new output file
        with open(outfile, 'wb') as o:
            # set the start bar and beat counter
            bar = 0
            beat_counter = args.beat
            with open(f, 'rb') as i:
                for l in i:
                    # copy comments as is
                    if l.startswith('#'):
                        o.write('%s\n' % l)
                    # modify data before writing
                    else:
                        try:
                            # modify the second column (if it exists)
                            onset, beat = l.split()
                            # skip if already in bar.beat format
                            if '.' in beat:
                                o.write('%s' % l)
                                continue
                            # ok, beat is in int format
                            onset = float(onset)
                            beat = int(beat)
                            if beat == 1:
                                bar += 1
                        except ValueError:
                            # no beat present, use the meter information
                            if args.meter is None:
                                raise SystemExit('Please specify starting bar'
                                                 '& beat and the meter.')
                            # onset
                            onset = float(l)
                            # bar
                            bar = math.ceil(float(beat_counter) /
                                            args.meter) - 1
                            bar += args.bar
                            # beat
                            beat = (beat_counter % args.meter)
                            if beat == 0:
                                beat = args.meter
                            # increase the beat counter
                            beat_counter += 1
                        # write the beat
                        o.write('%s\t%d.%d\n' % (onset, bar, beat))


if __name__ == '__main__':
    main()
