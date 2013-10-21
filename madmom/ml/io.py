#!/usr/bin/env python
# encoding: utf-8
"""
This file contains functionality needed for the conversion of from the universal
.h5 format to the .npz format understood by madmom.ml.rnn.

The .h5 files must conform to this format:

The `model` group contains just attributes.
  - `type`:    the type of the model
  - `comment`: free text for comments (optional)

The `layer` group contains a subgroup for each layer.

The subgroups are numbered consecutively, starting at index zero.

Each layer subgroup contains the following attributes:
  - `type`: the type of the layer
Each layer subgroup can contain the following datasets:
  - `bias`:              bias of the layer
  - `weights`:           weights of the layer
  - `recurrent_weights`: recurrent weights (optional for recurrent layers)
  - `peephole_weights`:  peephole weights (optional for LSTM layers)

Each of the previous layer subgroups can contain the same named datasets with a
'reverse_' prefix to indicate that they belong to the reverse/backward layer of
bidirectional layers.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import os
import numpy as np


def convert_model(infile, outfile=None, compressed=False):
    """
    Convert a neural network model from .h5 to .npz format.

    :param infile:  file with the saved model (.h5 format)
    :param outfile: file to write the model (.npz format)

    """
    import h5py
    npz = {}
    # read in modele
    with h5py.File(infile, 'r') as h5:
        # model attributes
        for attr in h5['model'].attrs.keys():
            npz['model_%s' % attr] = h5['model'].attrs[attr]
        # layers
        for l in h5['layer'].keys():
            layer = h5['layer'][l]
            # each layer has some attributes
            for attr in layer.attrs.keys():
                npz['layer_%s_%s' % (l, attr)] = layer.attrs[attr]
            # and some datasets
            for data in layer.keys():
                npz['layer_%s_%s' % (l, data)] = layer[data].value
    # save the model to .npz format
    if outfile is None:
        outfile = os.path.splitext(infile)[0]
    if compressed:
        np.savez_compressed(outfile, **npz)
    np.savez(outfile, **npz)
