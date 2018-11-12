# encoding: utf-8
"""
This files contains pytest fixtures and other test related stuff.

"""
from __future__ import absolute_import, division, print_function

import numpy as np
import pytest

# save numpy's current print options
_NP_PRINT_OPTIONS = np.get_printoptions()


@pytest.fixture(autouse=True)
def setup_doctest(request):
    """Set up the environment for doctests (when run through pytest)."""
    np.set_printoptions(precision=5, edgeitems=2, suppress=True)

    def fin():
        """Restore the environment after doctests (when run through pytest)."""
        np.set_printoptions(**_NP_PRINT_OPTIONS)

    request.addfinalizer(fin)
