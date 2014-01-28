#!/usr/bin/env python
# encoding: utf-8
"""
Abstract:
convenience wrapper to be able to override values from configuration file
from the commandline

Description:
this wrapper reads in values from different sections of a YAML/JSON
configuration file, whose name is specifiable on the commandline via '--config'
it then adds each of the values as a commandline parameter,
- allowing them to be overridden by specifying them on the commandline
- merging them with additionally specified (mostly positional) parameters

@author: Rainer Kelz <rainer.kelz@jku.at>

"""

import argparse
import os
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
    def __init__(self,
                 configfilename='config.yaml',
                 sectionnames=[],
                 description=''):

        self.sectionnames = sectionnames

        add_help = False if os.path.isfile(configfilename) else True

        def existing_config_file(cf):
            if os.path.isfile(cf):
                return cf
            else:
                raise argparse.ArgumentTypeError('config file does not exist')

        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description=description, add_help=add_help)

        self.parser.add_argument('--config',
                                 dest='actualconfigfilename',
                                 default=configfilename,
                                 type=existing_config_file,
                                 help='the name of the config file '
                                      'supplying all additional parameters')

        pre_args, _ = self.parser.parse_known_args()

        # re-new the parser
        self.parser = argparse.ArgumentParser(description=description,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        config = yaml.load(open(pre_args.actualconfigfilename),
                           OrderedDictYAMLLoader)

        for section in self.sectionnames:
            if section in config:
                group = self.parser.add_argument_group(section)
                for optionname, option in config[section].items():
                    optionvalue = option[0]
                    optionhelptxt = option[1]

                    group.add_argument('--%s' % optionname,
                                       type=type(optionvalue),
                                       default=optionvalue,
                                       help=optionhelptxt)
            else:
                raise ValueError("invalid section name: %s" % section)

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def parse_args(self):
        # ignore unknown stuff
        args, _ = self.parser.parse_known_args()
        return args

    def _convert(self, yamlobj):
        if isinstance(yamlobj, dict):
            return {self._convert(key): self._convert(value)
                    for key, value in yamlobj.iteritems()}
        elif isinstance(yamlobj, list):
            return [self._convert(element) for element in yamlobj]
        elif isinstance(yamlobj, unicode):
            return str(yamlobj)
        else:
            return yamlobj
