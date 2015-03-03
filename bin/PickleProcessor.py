#!/usr/bin/env python
# encoding: utf-8
"""
@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import cPickle
import argparse

from madmom.utils import io_arguments


def main():
    """Generic Pickle Processor."""
    # define parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    This script runs previously pickled Processors in either batch or single
    file mode.

    To obtain a pickled Processor, either run the respective script with the
    'pickle' option or call the 'dump()' method of the Processor.

    ''')
    # add options
    parser.add_argument('processor', type=argparse.FileType('r'),
                        help='pickled processor')
    parser.add_argument('--version', action='version',
                        version='PickleProcessor v0.1')
    io_arguments(parser)

    # parse arguments
    args = parser.parse_args()
    kwargs = vars(args)

    # create a processor
    processor = cPickle.load(kwargs.pop('processor'))
    # and call the processing function
    args.func(processor, **kwargs)


if __name__ == '__main__':
    main()
