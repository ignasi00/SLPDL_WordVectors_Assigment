
from docopt import docopt


# TODO: default values as "CONSTANTS"


# TODO: only --name=<n> parameters because wandb agents does not like options
DOCTEXT = f"""
Usage:
  #

Options:
  #

"""


def parse_args(argv):
    opts = docopt(DOCTEXT, argv=argv, help=True, version=None, options_first=False)
    
    # TODO : str2class

    args = () # TODO: fill main parameters list (usage: main(*args))

    return args
