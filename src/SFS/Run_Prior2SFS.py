#!/usr/bin/python

"""
This file to create Site Frequency Spectrum (SFS) from Priors or parameters
"""
import argparse

# noinspection PyUnresolvedReferences
#import Demography
from Simulation.msprime import Demography
# noinspection PyUnresolvedReferences
import Class
from Classes import Misc
from _version import __version__

parser = argparse.ArgumentParser(description='To run msprime and create sfs from priors')
parser.add_argument('-v', '--version', action='version',
                    version='ABC-DLS {version}'.format(version=__version__))
parser.add_argument('demography', help='The msprime demography function that has to be load from Demogrpahy.py')
parser.add_argument('--inds',
                    help='The number of individuals  per populations. All the output populations should be mentioned in'
                         ' the inds. again separated by inds1,inds2. remember 1 inds = 2 haplotypes. thus from 5 '
                         'individuals  you would get total 11 (0 included) different allele counts ',
                    required=True)
parser.add_argument('--params_file', required=True,
                    help='All the priors for the parameters on which the simulation will run. Should be "," comma '
                         'separated csv format. Different rows signify different run. columns different parameters',
                    type=lambda x: Misc.args_valid_file(parser, x))
parser.add_argument('--total_length',
                    help='total length of the genome. default is 3gb roughly the length of human genome',
                    type=float, default=3e9)
parser.add_argument('--ldblock', help='Length of simulated blocks. Default is 1mb', default=1e6, type=float)
parser.add_argument('--mutation_rate', help='mutation rate. default is 1.45e-8 per gen', type=float, default=1.45e-8)
parser.add_argument('--threads', help='the number of threads. default is 1', type=int, default=1)
parser.add_argument('--sfs2c',
                help="Instead of SFS output SFS with two population combination. Better in case you have too many "
                     "samples",
                action="store_true")
args = parser.parse_args()

demography = eval('Demography.' + args.demography)
if args.sfs2c:
    params_sfs = Class.MsPrime2SFS2c.wrapper(sim_func=demography, params_file=args.params_file, samples=args.inds,
                             total_length=args.total_length, ldblock=args.ldblock, mut_rate=args.mutation_rate,
                             threads=args.threads)
else:
    params_sfs = Class.MsPrime2SFS(sim_func=demography, params_file=args.params_file, samples=args.inds,
                             total_length=args.total_length, ldblock=args.ldblock, mut_rate=args.mutation_rate,
                             threads=args.threads)


print(params_sfs.to_csv(index=False))
