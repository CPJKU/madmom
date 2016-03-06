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

# TODO: Documentation and comments!

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
        return load_npz(filename)
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
            raise TypeError('Unknown entity type: {}'.format(item))

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


def hdf5_to_npz(hdf5_filename, npz_filename):
    import h5py

    def convert_attrs(item, name):
        name += '/attrs/'
        for attr_name, attr_val in item.attrs.iteritems():
            arrs[name + attr_name] = attr_val

    def convert(item, name):
        convert_attrs(item, name)
        if type(item) == h5py.Dataset:
            arrs[name] = item.value
        elif type(item) == h5py.Group:
            for sub_item_name, sub_item in item.iteritems():
                convert(sub_item, name + '/' + sub_item_name)
        else:
            raise TypeError('Unknown entity type: {}'.format(item))

    f = h5py.File(hdf5_filename, 'r')
    arrs = {}
    for grp_name, grp in f.iteritems():
        convert(grp, grp_name)

    np.savez(npz_filename, **arrs)


def _npz_to_dict_tree(npz_file):
    # This 'magic' creates a hierarchical representation (tree) of the file's
    # content, represented by a python dictionary. It looks at every element
    # in the npz file, makes sure that the dictionary structure exists to
    # store the value (this is what the reduce operation does!), and stores
    # the value in there.

    def add_leaf(subtree, leaf_name):
        # add a new leaf if it does not exist already
        subtree[leaf_name] = subtree.get(leaf_name, {})
        return subtree[leaf_name]

    tree = {}
    for flat_key, v in npz_file.iteritems():
        ks = flat_key.split('/')
        reduce(add_leaf, ks[:-1], tree)[ks[-1]] = v

    return tree


def load_npz(filename):

    def load_item(item):
        if type(item) == np.ndarray:
            return item
        elif type(item) == dict:
            return load_group(item)
        else:
            raise TypeError('Unknown entity type: {}'.format(item))

    def load_list(grp):
        # sort contents according to 'id' attribute, return a list
        # with loaded items
        group_contents = sorted(
            (v for k, v in grp.iteritems() if k != 'attrs'),
            cmp=lambda x, y: cmp(x['attrs']['id'], y['attrs']['id'])
        )

        return [load_item(item) for item in group_contents]

    def load_group(grp):
        import madmom

        cls_name = str(grp['attrs']['type'])

        if cls_name == 'list':
            return load_list(grp)

        # load the class or function. getattr(madmom, 'ml.nn.xxx') does not
        # work, we need to load each thing individually, like
        # ml = getattr(madmom, 'ml'); rnn = getattr(ml, 'nn'), and so on
        # this code line does this.
        cls = reduce(lambda module, item: getattr(module, item),
                     cls_name.split('.')[1:], madmom)

        if not grp['attrs'].get('instantiate', True):
            return cls

        params = {name: load_item(item)
                  for name, item in grp.iteritems()
                  if name != 'attrs'}

        return cls(**params)

    model_tree = _npz_to_dict_tree(np.load(filename))
    return {name: load_item(grp) for name, grp in model_tree.iteritems()}

