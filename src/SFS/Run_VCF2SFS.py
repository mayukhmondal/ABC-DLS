#!/usr/bin/python

"""
This file to convert vcf file to sfs (csv) format
"""
import argparse

import pandas

# noinspection PyUnresolvedReferences
from Class import VCF2SFS
from Classes import Misc
from _version import __version__

##input argument is done
parser = argparse.ArgumentParser(description='To convert a vcf file to sfs (csv) format which can be used in ABC')
parser.add_argument('-v', '--version', action='version',
                    version='ABC-DLS {version}'.format(version=__version__))
parser.add_argument('vcffile',
                    help='the path of the vcf file. can be zipped', type=lambda x: Misc.args_valid_file(parser, x))
parser.add_argument('--popfile',
                    help="The file where population format is written. first column is individual, second column is "
                         "population", type=lambda x: Misc.args_valid_file(parser, x),
                    required=True)
parser.add_argument('--sfs_pop', help="the name of pop. important for the order. example: pop1,pop2,pop3 ",
                    required=True)
parser.add_argument('--chunksize',
                    help='If too big for the memory use chunk size. relatively slow but no problem with ram. default is'
                         '1e6 (1 million)',
                    type=float, default=1e6)
parser.add_argument('--outprefix',
                    help='in case you want to name the output. by default it will be your vcf file name ')

args = parser.parse_args()

out = VCF2SFS(vcffile=args.vcffile, popfile=args.popfile,
              sfs_pop=args.sfs_pop.split(","), chunk_length=args.chunksize, out=args.outprefix)
print(pandas.DataFrame(out).transpose().to_csv(index=False), end="")
