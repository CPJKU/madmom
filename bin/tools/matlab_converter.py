#!/usr/bin/env python
# encoding: utf-8
"""
Script for converting .npy and .npz files in .mat files.

"""

import os
import numpy as np
import scipy.io as sio
import argparse


def main():
    """Simple numpy to MATLAB file converter."""
    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    The script converts numpy .npy and .npz files in .mat files.

    If the numpy file is a single array (i.e. .npy) the array is stored in the
    MATLAB dictionary with 'data' as key.

    If the numpy file contains several arrays (i.e. .npz) the names of the
    individual arrays are used as keys for the MATLAB dictionary.

    """)
    # files used for evaluation
    p.add_argument('files', nargs='*', help='files to be converted.')
    # parse the arguments
    args = p.parse_args()
    # convert all files
    for f in args.files:
        # create a dictionary
        data = np.load(f)
        if isinstance(data, np.ndarray):
            # wrap in a dictionary
            data = {'data': data}
        # the data must be in a dictionary
        if isinstance(data, (dict, np.lib.npyio.NpzFile)):
            # write the .mat file
            sio.savemat(os.path.splitext(f)[0], data, appendmat=True)
        else:
            # otherwise just exit
            raise SystemExit('%s not in the right format.' % f)

if __name__ == '__main__':
    main()
