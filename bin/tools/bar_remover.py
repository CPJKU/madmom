#!/usr/bin/env python
# encoding: utf-8
"""
This file loads a beat file and translates the old downbeat format
(bar.beat) into the new one (beat).

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""
import os
import argparse
import numpy as np


def main():
    """
    Simple beat annotation correction tool.

    """
    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    The script removes the bar part of beat annotations.

    """)
    # files used for evaluation
    p.add_argument('files', nargs='*',
                   help='files (or folder) to be corrected')
    p.add_argument('--ext', dest='extension', default=None,
                   help='use this new extension')
    p.add_argument('-o', dest='output', default=None,
                   help='output directory')
    p.add_argument('--downbeats', dest='downbeats', action='store_true',
                   help='also write new .downbeat files')

    # parse the  arguments
    args = p.parse_args()

    # never overwrite the annotation files
    if args.extension is None and args.output is None:
        raise SystemExit('Can not overwrite the annotation files. Please chose'
                         ' either an extension or another output directory.')

    # convert all files
    for in_file in args.files:
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
            out_file = os.path.splitext(out_file)[0] + args.extension
        # read in the annotations
        annotations = np.loadtxt(in_file)
        # check if the second column contains a dot
        if (annotations[0, 1] % 1) > 0:
            # remove the bar. part
            annotations[:, 1] = np.around((annotations[:, 1] % 1) * 10)
        # write it back (with a new quantization)
        np.savetxt(out_file, annotations, fmt=['%.3f', '%d'], delimiter='\t')
        # also write a downbeat file?
        if args.downbeats:
            downbeat_file = os.path.splitext(out_file)[0] + '.downbeats'
            downbeats = annotations[annotations[:, 1] == 1][:, [0]]
            np.savetxt(downbeat_file, downbeats, fmt='%.3f')

if __name__ == '__main__':
    main()
