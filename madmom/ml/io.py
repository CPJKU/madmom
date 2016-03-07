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


def load(filename, file_type=None):
    if file_type is None:
        ext = os.path.splitext(filename)[1][1:]
        if ext in ['hdf', 'h5', 'hdf5', 'he5']:
            file_type = 'hdf5'
        elif ext == 'npz':
            file_type = 'npz'
        else:
            raise ValueError('Cannot determine file type for '
                             'file "{}".'.format(filename))
    if file_type == 'hdf5':
        return load_hdf5(filename)
    elif file_type == 'npz':
        raise NotImplementedError('NPZ model loading not implemented yet!')
    else:
        raise ValueError('Unknown file type "".'.format(file_type))


def load_hdf5(filename):
    import h5py

    def load_item(item):
        if type(item) == h5py.Dataset:
            return item.value
        elif type(item) == h5py.Group:
            return load_group(item)
        else:
            raise RuntimeError('Unknown entity: {}'.format(item))

    def load_list(grp):
        # sort contents according to 'id' attribute, return a list
        # with loaded items
        group_contents = sorted(
            grp.itervalues(),
            cmp=lambda x, y: cmp(x.attrs['id'], y.attrs['id'])
        )

        return [load_item(item) for item in group_contents]

    def load_group(grp):
        import madmom

        cls_name = grp.attrs['type']

        if cls_name == 'list':
            return load_list(grp)

        # load the class or function. getattr(madmom, 'ml.nn.xxx') does not
        # work, we need to load each thing individually, like
        # ml = getattr(madmom, 'ml'); rnn = getattr(ml, 'nn'), and so on
        # this code line does this.
        cls = reduce(lambda module, item: getattr(module, item),
                     cls_name.split('.')[1:], madmom)

        if not grp.attrs.get('instantiate', True):
            return cls

        params = {name: load_item(item) for name, item in grp.iteritems()}
        return cls(**params)

    f = h5py.File(filename, 'r')
    return {name: load_item(grp) for name, grp in f.iteritems()}



# def convert_model(infile, outfile=None, compressed=False):
#     """
#     Convert a neural network model from .h5 to .npz format.
#
#     Parameters
#     ----------
#     infile : str
#         File with the saved model (.h5 format).
#     outfile : str, optional
#         File to write the model (.npz format).
#     compressed : bool, optional
#         Compress the resulting .npz file?
#
#     """
#     import h5py
#     npz = {}
#     # read in model
#     with h5py.File(infile, 'r') as h5:
#         # model attributes
#         for attr in list(h5['model'].attrs.keys()):
#             npz['model_%s' % attr] = h5['model'].attrs[attr]
#         # layers
#         for l in list(h5['layer'].keys()):
#             layer = h5['layer'][l]
#             # each layer has some attributes
#             for attr in list(layer.attrs.keys()):
#                 npz['layer_%s_%s' % (l, attr)] = layer.attrs[attr]
#             # and some data sets (i.e. different weights)
#             for data in list(layer.keys()):
#                 npz['layer_%s_%s' % (l, data)] = layer[data].value
#     # save the model to .npz format
#     if outfile is None:
#         outfile = os.path.splitext(infile)[0]
#     if compressed:
#         np.savez_compressed(outfile, **npz)
#     else:
#         np.savez(outfile, **npz)
