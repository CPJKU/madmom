# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
TODO: Rewrite this docstring!

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
    """
    Loads a model from an HDF5 or NPZ file.

    Parameters
    ----------
    filename : string
        Name of the model file
    file_type : string, optional
        Type of file, 'npz' or 'hdf5'. If not given, inferred from the file
        name, if possible.

    Returns
    -------
    dict
        Dictionary containing all the models found in the file

    Notes
    -----
    Needs h5py if loading an HDF5 file
    """
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
    """
    Loads a model from an HDF5 file.

    Parameters
    ----------
    filename : string
        Name of the model file

    Returns
    -------
    dict
        Dictionary containing all the models found in the file

    Notes
    -----
    Needs h5py
    """
    import h5py

    def load_item(item):
        """Processes an item found in the HDF5 file"""
        if type(item) == h5py.Dataset:
            return item.value
        elif type(item) == h5py.Group:
            return load_group(item)
        else:
            raise TypeError('Unknown entity type: {}'.format(item))

    def load_list(grp):
        """Processes a group of type 'list'"""

        # sort contents according to 'id' attribute, return a list
        # with loaded items
        group_contents = sorted(
            grp.itervalues(),
            cmp=lambda x, y: cmp(x.attrs['id'], y.attrs['id'])
        )

        return [load_item(item) for item in group_contents]

    def load_group(grp):
        """Processes a group object found in the HDF5 file"""
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

        # if we do not need to instantiate the class/function represented
        # by this group, return it immediately
        if not grp.attrs.get('instantiate', True):
            return cls

        # collect the parameters and return the instantiated/called class
        # or function
        params = {name: load_item(item) for name, item in grp.iteritems()}
        return cls(**params)

    # load all the models found in the file
    f = h5py.File(filename, 'r')
    return {name: load_item(grp) for name, grp in f.iteritems()}


def hdf5_to_npz(hdf5_filename, npz_filename):
    """
    Converts an HDF5 model file to an NPZ model file.

    Parameters
    ----------
    hdf5_filename : string
        Name of the source HDF5 model file
    npz_filename : string
        Name of the destination NPZ model file

    Notes
    -----
    Needs h5py to load the HDF5 file

    """
    import h5py

    # note that the following functions are closures and use
    # the 'arrs' variable from the hdf5_to_npz scope to store
    # the converted HDF5 models found

    def convert_attrs(item, name):
        """Stores the attributes of an HDF5 item"""
        name += '/attrs/'
        for attr_name, attr_val in item.attrs.iteritems():
            arrs[name + attr_name] = attr_val

    def convert(item, name):
        """Converts an HDF5 item to a flat representation for the NPZ format"""
        convert_attrs(item, name)
        if type(item) == h5py.Dataset:
            arrs[name] = item.value
        elif type(item) == h5py.Group:
            # store all the sub-parts of this group
            for sub_item_name, sub_item in item.iteritems():
                convert(sub_item, name + '/' + sub_item_name)
        else:
            raise TypeError('Unknown entity type: {}'.format(item))

    f = h5py.File(hdf5_filename, 'r')

    # collect all models from the hdf5 file
    arrs = {}
    for grp_name, grp in f.iteritems():
        convert(grp, grp_name)

    np.savez(npz_filename, **arrs)


def _npz_to_dict_tree(npz_file):
    """
    Converts a flat model representation as found in the npz_file to a
    hierarchical representation (tree), stored in a Python dictionary.
    It looks at every element in the npz file, makes sure that the dictionary
    structure (the sub-tree) exists to store the value (this is what the
    reduce operation does!), and stores the value in there.
    """

    def add_leaf(subtree, leaf_name):
        """
        Adds a leaf (empty dict) to a sub-tree if it does not exist already
        """
        subtree[leaf_name] = subtree.get(leaf_name, {})
        return subtree[leaf_name]

    # start with an empty tree
    tree = {}

    # For every element in the npz_file, create the sub-tree using the
    # reduce function, and store the corresponding value as a leaf
    for flat_key, v in npz_file.iteritems():
        ks = flat_key.split('/')
        reduce(add_leaf, ks[:-1], tree)[ks[-1]] = v

    return tree


def load_npz(filename):
    """
    Loads a model from an NPZ file.

    Parameters
    ----------
    filename : string
        Name of the model file

    Returns
    -------
    dict
        Dictionary containing all the models found in the file
    """

    def load_item(item):
        """Processes an item in the tree representation of the model"""
        if type(item) == np.ndarray:
            return item
        elif type(item) == dict:
            return load_group(item)
        else:
            raise TypeError('Unknown entity type: {}'.format(item))

    def load_list(grp):
        """Processes a group of type 'list' found in the model tree"""

        # sort contents according to 'id' attribute, return a list
        # with loaded items
        group_contents = sorted(
            (v for k, v in grp.iteritems() if k != 'attrs'),  # skip attributes
            cmp=lambda x, y: cmp(x['attrs']['id'], y['attrs']['id'])
        )

        return [load_item(item) for item in group_contents]

    def load_group(grp):
        """Processes a group object found in the model tree"""
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

        # if we do not need to instantiate the class/function represented
        # by this group, return it immediately
        if not grp['attrs'].get('instantiate', True):
            return cls

        # collect the parameters and return the instantiated/called class
        # or function. skip the attributes.
        params = {name: load_item(item)
                  for name, item in grp.iteritems()
                  if name != 'attrs'}
        return cls(**params)

    # convert the flat representation found in the NPZ file to a tree
    # representation for easier parsing
    model_tree = _npz_to_dict_tree(np.load(filename))
    # Load all the models found in the file
    return {name: load_item(grp) for name, grp in model_tree.iteritems()}
