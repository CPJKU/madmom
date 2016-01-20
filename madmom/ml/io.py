# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains functionality needed for the conversion of from the
universal .h5 format to the .npz format understood by madmom.ml.rnn.

The .h5 files must conform to this format:

The `model` group contains just attributes.

- `type`:
  The type of the model.
- `comment`:
  Free text for comments (optional).

The `layer` group contains a subgroup for each layer.

The subgroups are numbered consecutively, starting with zero.

Each layer subgroup contains the following attributes:

- `type`:
  The type of the layer (e.g. FeedForward, Recurrent, LSTM; basically any Layer
  of madmom.ml.rnn).
- `transfer_fn`:
  The transfer/activation function of the layer.
  Each layer subgroup contains the following data sets:

  - `bias`:
    Bias of the layer.
  - `weights`:
    Weights of the layer.
  - `recurrent_weights`:
    Recurrent weights (optional for recurrent layers).
  - `peephole_weights`:
    Peephole weights (optional for LSTM layers).

Each of the previous layer subgroups data sets and attributes can contain the
same named data sets with a `reverse_` prefix to indicate that they belong to
the reverse/backward layer of bidirectional layers.

"""

from __future__ import absolute_import, division, print_function

import os
import numpy as np


def convert_model(infile, outfile=None, compressed=False):
    """
    Convert a neural network model from .h5 to .npz format.

    Parameters
    ----------
    infile : str
        File with the saved model (.h5 format).
    outfile : str, optional
        File to write the model (.npz format).
    compressed : bool, optional
        Compress the resulting .npz file?

    """
    import h5py
    npz = {}
    # read in model
    with h5py.File(infile, 'r') as h5:
        # model attributes
        for attr in list(h5['model'].attrs.keys()):
            npz['model_{0!s}'.format(attr)] = h5['model'].attrs[attr]
        # layers
        for l in list(h5['layer'].keys()):
            layer = h5['layer'][l]
            # each layer has some attributes
            for attr in list(layer.attrs.keys()):
                npz['layer_{0!s}_{1!s}'.format(l, attr)] = layer.attrs[attr]
            # and some data sets (i.e. different weights)
            for data in list(layer.keys()):
                npz['layer_{0!s}_{1!s}'.format(l, data)] = layer[data].value
    # save the model to .npz format
    if outfile is None:
        outfile = os.path.splitext(infile)[0]
    if compressed:
        np.savez_compressed(outfile, **npz)
    else:
        np.savez(outfile, **npz)
