#!/usr/bin/env python
# encoding: utf-8
"""
Script for converting ground truth annotations into the GiantSteps format.

Version history:
0.1 initial version
0.2 added #@format: prefix
0.3 preserve directory structure
0.4 added tempo format
0.5 changed .tempo to .bpm extension

"""

import argparse
import os


# TODO: code a proper definition file parser or add other definitions
CONVERTER = {'onsets': '#@format: onset\ttimestamp',
             'beats': '#@format: beat\ttimestamp\tbar.beat',
             'notes': '#@format: note\ttimestamp\tpitch\tduration\tvelocity',
             'bpm': '#@format: tempo\ttimestamp\tbpm'}


def main():
    """
    Simple GiantSteps format converter.

    """
    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    The script converts ground-truth annotation files into format used by the
    GiantSteps project. The files are supposed tho have extensions like:
    - .onsets
    - .beats
    - .notes
    - .bpm

    TODO: include a quick overview here when definition is ready.

    """)
    # files to be processed
    p.add_argument('files', nargs='+', help='files to be processed')
    # output directory
    p.add_argument('-o', dest='output', default=None,
                   help='output directory')
    p.add_argument('--ext', dest='extension', default=None,
                   help='append the given file extension')
    p.add_argument('--version', action='version', version='0.5')
    # parse arguments
    args = p.parse_args()

    # never overwrite the annotation files
    if args.extension is None and args.output is None:
        raise SystemExit('Can not overwrite the annotation files. Please chose'
                         ' either an extension or another output directory.')

    # convert all files
    for in_file in args.files:
        # split extension form file
        ext = os.path.splitext(in_file)[1]
        # determine the output file name
        if args.output:
            out_file = "%s/%s/%s" % (args.output, os.path.dirname(in_file),
                                     os.path.basename(in_file))
            out_path = os.path.dirname(out_file)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
        else:
            out_file = in_file
        # add an extension if needed
        if args.extension:
            out_file += args.extension
        # determine header (key = extension without the leading dot)
        header = CONVERTER[ext[1:]]
        # determine prefix (= first value without trailing
        # whitespace after '#@format:')
        prefix = header.split('\t')[0].split(':')[1].strip()
        # write the new output file
        with open(out_file, 'wb') as o:
            o.write('%s\n' % header)
            with open(in_file, 'rb') as i:
                for l in i:
                    # skip our format lines
                    if l.startswith('#@'):
                        continue
                    # copy comments as is
                    elif l.startswith('#'):
                        o.write('%s' % l)
                    # add first column
                    else:
                        # we need to add a timestamp 0 for the tempo
                        if prefix == 'tempo':
                            o.write('%s\t0\t%s' % (prefix, l))
                        else:
                            o.write('%s\t%s' % (prefix, l))

if __name__ == '__main__':
    main()
