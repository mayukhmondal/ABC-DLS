#!/usr/bin/python

"""
This file to convert vcf file to sfs (csv) format
"""
import argparse

# noinspection PyUnresolvedReferences
from Class import VCF2SFS
from _version import __version__
##input argument is done
parser = argparse.ArgumentParser(description='To convert a vcf file for the distant matrix which can be used in ABC')
parser.add_argument('-v', '--version', action='version',
                    version='ABC-TFK {version}'.format(version=__version__))
parser.add_argument('vcffile',
                    help='the path of the vcf file. can be zipped')
parser.add_argument('--popfile',
                    help="The file where population format is written. first column is individual, second column is "
                         "population. the file should be sorted according the simulation paradigm",
                    required=True)
parser.add_argument('--sfs_pop', help="the name of pop. important for the order. example: pop1,pop2,pop3 ",
                    required=True)
parser.add_argument('--chunksize',
                    help='If too big for the memory use chunk size. relatively slow but no problem with ram',
                    type=float, default=1e6)
parser.add_argument('--outprefix',
                    help='in case you want to name the output. by default it will be your vcf file name ')

args = parser.parse_args()

out = VCF2SFS(vcffile=args.vcffile, popfile=args.popfile,
                      sfs_pop=args.sfs_pop.split(","), chunk_length=args.chunksize, out=args.outprefix)
