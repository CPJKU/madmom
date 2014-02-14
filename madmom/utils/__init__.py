# encoding: utf-8
"""
Utility package.

"""
import helpers

import io
from contextlib import contextmanager


@contextmanager
def open(filename):
    # check if we need to open the file
    if isinstance(filename, basestring):
        f = fid = io.open(filename, 'r')
    else:
        f = filename
        fid = None
    # yield an open file handle
    yield f
    # close the file if needed
    if fid:
        fid.close()
