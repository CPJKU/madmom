#!/usr/bin/env python
# encoding: utf-8
"""
Abstract:
A wrapper for the standard 'ArgumentParser' class (module argparse) to be able
to override values coming from a YAML configuration file conveniently from the
command line

Description:
This wrapper reads in values from different sections of a YAML configuration
file, whose name is specifiable on the command line via '--config'.

It then adds each of the values as an optional(!) command line parameter,
- allowing them to be overridden by specifying them on the command line
- merging them with additionally specified (mostly positional) parameters

If you ever find yourself in the situation that you have lots of different
command line scripts, sharing common parameters (encoding of audio-streams, FFT
parameters, filter sizes, etc...) this is the ArgumentParser-wrapper to use.


YAML-Format is as follows:
<section_name>:
  <parameter_name>: [<value>, <help-text>]


An example:

-----------------------------------------------------------------------
config.yaml:
sectionA:
  param1: [0, 'this is the help text for parameter 1']
  param2: ['something', 'this is the help for parameter 2']

sectionB:
  ...
  ...

-----------------------------------------------------------------------
test_overparams.py:
import madmom.utils.overparams as overparams
parser = overparams.OverridableParameters(description='description',
                                          section_names=['sectionA'])

parser.add_argument('positional', help='some positional arguments')
args = parser.parse_args()

print args

-----------------------------------------------------------------------
Calling this script from the command line yields:

$ python test_overparams.py aloha
Namespace(config='config.yaml',
          param1=0,
          param2='something',
          positional='aloha')

python testop.py aloha --param1 23 --param2 "something completely different"
Namespace(config='config.yaml',
          param1=23,
          param2='something completely different',
          positional='aloha')


"""

import argparse
import os
import sys
import yaml.constructor
from collections import OrderedDict


# retrieved from https://gist.github.com/enaeseth/844388
class OrderedDictYAMLLoader(yaml.Loader):
    """
    A YAML loader that loads mappings into ordered dictionaries.

    """

    def __init__(self, *args, **kwargs):
        yaml.Loader.__init__(self, *args, **kwargs)

        self.add_constructor(u'tag:yaml.org,2002:map',
                             type(self).construct_yaml_map)
        self.add_constructor(u'tag:yaml.org,2002:omap',
                             type(self).construct_yaml_map)

    def construct_yaml_map(self, node):
        data = OrderedDict()
        yield data
        value = self.construct_mapping(node)
        data.update(value)

    def construct_mapping(self, node, deep=False):
        if isinstance(node, yaml.MappingNode):
            self.flatten_mapping(node)
        else:
            raise yaml.constructor.ConstructorError(
                None, None, 'expected a mapping node, but found %s' % node.id,
                node.start_mark)

        mapping = OrderedDict()
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                hash(key)
            except TypeError, exc:
                raise yaml.constructor.ConstructorError(
                    'while constructing a mapping', node.start_mark,
                    'found unacceptable key (%s)' % exc, key_node.start_mark)
            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value
        return mapping


class OverridableParameters(object):
    """
    Wrapper class for argparse.ArgumentParser. See module docs.

    :param config_filename: the default config_filename
    :param section_names:   which sections from the YAML file to include as
                            parameters
    :param description:     optional description of the argument parser

    """

    def __init__(self, config_filename='config.yaml', section_names=None,
                 description=''):

        if not section_names:
            section_names = []
        self.section_names = section_names

        if len(sys.argv) >= 3 and sys.argv[1] == '--config':
            config_filename = sys.argv[2]

        # create the parser
        self.parser = argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        def existing_config_file(cfn):
            if os.path.isfile(cfn):
                return cfn
            else:
                raise argparse.ArgumentTypeError('config file does not exist')

        self.parser.add_argument('--config',
                                 default=config_filename,
                                 type=existing_config_file,
                                 help='the name of the config file'
                                 ' supplying all additional parameters')

        if os.path.isfile(config_filename):
            config = yaml.load(open(config_filename, 'r'),
                               OrderedDictYAMLLoader)

            for section in self.section_names:
                if section in config:
                    group = self.parser.add_argument_group(section)
                    for option_name, option in config[section].items():
                        option_value = option[0]
                        option_help_txt = ''
                        if len(option) > 1:
                            option_help_txt = option[1]

                        group.add_argument('--%s' % option_name,
                                           type=type(option_value),
                                           default=option_value,
                                           help=option_help_txt)
                else:
                    raise ValueError('invalid section name encountered: "%s"' %
                                     section)

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def parse_args(self, args=None, namespace=None):
        return self.parser.parse_args(args, namespace)
