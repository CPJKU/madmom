import argparse
import yaml
import yaml_ordered_dict
import os

#######################################
# abstract:
# convenience wrapper to be able to override values from configuration file
# from the commandline
#
# description:
# this wrapper reads in values from different sections of a YAML/JSON configuration file,
# whose name is specifiable on the commandline via '--config'
# it then adds each of the values as a commandline parameter,
# - allowing them to be overridden by specifying them on the commandline
# - merging them with additionally specified (mostly positional) parameters
#
# author: rainer.kelz@jku.at
#
###


class OverridableParameters():
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

        self.parser = argparse.ArgumentParser(description=description,
                                              formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                              add_help=add_help)

        self.parser.add_argument('--config',
                                 dest='actualconfigfilename',
                                 default=configfilename,
                                 type=existing_config_file,
                                 help='the name of the config file'
                                 ' supplying all additional parameters')

        pre_args, _ = self.parser.parse_known_args()

        # re-new the parser
        self.parser = argparse.ArgumentParser(description=description,
                                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        config = yaml.load(open(pre_args.actualconfigfilename),
                           yaml_ordered_dict.OrderedDictYAMLLoader)
        
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
                raise ValueError('invalid section name encountered "' + section + '"')


    def add_argument(self, *args, **kwords):
        self.parser.add_argument(*args, **kwords)

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
