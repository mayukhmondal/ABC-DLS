#!/usr/bin/python
"""
This file to create Coalescence Rate Trajectory (CRT) from Priors or parameters
"""
import argparse

# noinspection PyUnresolvedReferences
import Demography
#from Simulation.msprime import DemographyCRT as Demography
# noinspection PyUnresolvedReferences
from Class import MsPrime2CRT
from Classes import Misc
from _version import __version__

parser = argparse.ArgumentParser(description='To run msprime and create CRT from priors')
parser.add_argument('-v', '--version', action='version',
                    version='ABC-DLS {version}'.format(version=__version__))
parser.add_argument('demography', help='The msprime demography function that has to be load from Demogrpahy.py')
parser.add_argument('--inds',
                    help='The number of individuals  per populations. All the output populations should be mentioned in'
                         ' the inds. again separated by inds1,inds2. remember 1 inds = 2 haplotypes. thus from 5 '
                         'individuals  you would get total 11 (0 included) different allele counts ',
                    required=True)
parser.add_argument('--params', required=True,
                    help='All file path of the priors for the parameters on which the simulation will run. Should be '
                         ' "," comma separated csv format. Different rows signify different run. columns different '
                         'parameters', type=lambda x: Misc.args_valid_file(parser, x))
parser.add_argument('--gen', required=True,
                    help='The generations of time step at which point the CRT will be calculated. Every line signifies '
                         'different generations steps. Should be in increasing order. Does not have to be integer'
                         'and should not have header', type=lambda x: Misc.args_valid_file(parser, x))
parser.add_argument('--threads', help='the number of threads. default is 1', type=int, default=1)
args = parser.parse_args()

demography = eval('Demography.' + args.demography)
params_sfs = MsPrime2CRT(sim_func=demography, params_file=args.params, samples=args.inds,
                                 gen_file=args.gen, threads=args.threads)
print(params_sfs.to_csv(index=False))
