#!/usr/bin/env python
# encoding: utf-8
"""
Script for converting ground truth annotations into the GiantSteps format.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import argparse
import os


# TODO: code a proper definition file parser
CONVERTER = {'onsets': '#@onset\ttimestamp',
             'beats': '#@beat\ttimestamp\tbar.beat',
             'notes': '#@note\ttimestamp\tpitch\tduration\tvelocity'}


def main():
    """
    Simple GiantSteps format converter.

    """
    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    The script converts ground-truth annotation files into format used by the
    GiantSteps project.

    TODO: include a quick overview here when definition is ready.

    """)
    # files to be processed
    p.add_argument('files', nargs='+', help='files to be processed')
    # output directory
    p.add_argument('-o', dest='output', default=None,
                   help='output directory')
    p.add_argument('--ext', dest='extension', default=None,
                   help='append the given file extension')
    # parse arguments
    args = p.parse_args()

    # never overwrite the annotation files
    if args.extension is None and args.output is None:
        raise SystemExit('Can not overwrite the annotation files. Please chose'
                         ' either an extension or another output directory.')

    # convert all files
    for f in args.files:
        # split extension form file
        ext = os.path.splitext(f)[1]
        # determine the output file name
        if args.output:
            outfile = "%s/%s" % (args.output, os.path.basename(f))
        else:
            outfile = f
        # add the extension if needed
        if args.extension:
            outfile += args.extension
        # determine header (key = extension without the leading dot)
        header = CONVERTER[ext[1:]]
        # determine prefix (= first value without trailing '#@')
        prefix = header.split('\t')[0][2:]
        # write the new output file
        with open(outfile, 'wb') as o:
            o.write('%s\n' % header)
            with open(f, 'rb') as i:
                for l in i:
                    # skip our format lines
                    if l.startswith('#@'):
                        continue
                    # copy comments as is
                    elif l.startswith('#'):
                        o.write('%s\n' % l)
                    # add first column
                    else:
                        o.write('%s\t%s' % (prefix, l))

if __name__ == '__main__':
    main()
