#!/usr/bin/python

"""
This file to create priors with uniform distribution if upper and lower limit are given
"""
import argparse

# noinspection PyUnresolvedReferences
import Demography
# noinspection PyUnresolvedReferences
from Class import MsPrime2SFS

from _version import __version__

##input argument is done
parser = argparse.ArgumentParser(description='To run msprime and create sfs from priors')
parser.add_argument('-v','--version', action='version',
                    version='ABC-DLS {version}'.format(version=__version__))
parser.add_argument('demography', help='The msprime demography function that has to be load from Demogrpahy.py')
parser.add_argument('--inds',
                    help='The number of inds per populations. All the output populations should be mentioned in the '
                         'inds. again separated by inds1,inds2. remember 1 inds = 2 haplotypes. thus from 5 inds you '
                         'would get total 11 (0 included) different allele counts ',
                    required=True)
parser.add_argument('--params_file', required=True,
                    help='All the priors for the parameteres on which the simulation will run. Should be "," comma '
                         'separated csv format. Different rows signify different run. columns different parameters')
parser.add_argument('--total_length',
                    help='total length of the genome. default is 3gb roughly the length of human genome',
                    type=float, default=3e9)
parser.add_argument('--ldblock', help='Length of simulated blocks. Default is 1mb', default=1e6)
parser.add_argument('--mutation_rate', help='mutation rate. default is 1.45e-8 per gen', type=float, default=1.45e-8)
parser.add_argument('--threads', help='the number of threads. default is 1', type=int, default=1)
args = parser.parse_args()

demography = eval('Demography.' + args.demography)
params_sfs = MsPrime2SFS.wrapper(sim_func=demography, params_file=args.params_file, samples=args.inds,
                                 total_length=args.total_length, ldblock=args.ldblock, mut_rate=args.mutation_rate,
                                 threads=args.threads)
print(params_sfs.to_csv(index=False))
