#!/usr/bin/python

"""
This will convert SFS to SFS2c
"""
import argparse

# noinspection PyUnresolvedReferences
import Demography
#from Simulation.msprime import Demography
# noinspection PyUnresolvedReferences
import Class
from Classes import Misc
from _version import __version__

import argparse

# noinspection PyUnresolvedReferences
from Class import SFS_to_SFS2c
# from _version import __version__
##input argument is done
parser = argparse.ArgumentParser(description='To convert a multi-dimensional SFS csv file to cross population SFS csv'
                                             'file')
parser.add_argument('-v', '--version', action='version',
                    version='ABC-DLS {version}'.format(version=__version__))
parser.add_argument('sfs',
                    help='The path of the multi-dimensional SFS csv file with proper header coming from either '
                         'Run_Prior2SFS.py or Run_VCF2SFS.py. Can be zipped')
parser.add_argument('--params',
                    help="The number of columns with parameters, the first few columns can be parameteres if it is "
                         "coming from Run_Prior2SFS.py and rest are SFS. default is 0, meaning no parameters (for "
                         "example coming from Run_VCF2SFS.py",
                    type=int,default=0)
parser.add_argument('--chunksize',
                    help='If too big for the memory use chunk size. relatively slow but no problem with ram',
                    type=float, default=1e4)
args = parser.parse_args()

chunks=SFS_to_SFS2c.main(file=args.sfs,params=args.params, chunksize=int(args.chunksize))
[print(chunk,end="") for chunk in chunks]
