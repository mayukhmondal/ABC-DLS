#!/usr/bin/python

"""
This file to create priors with uniform distribution if upper and lower limit are given
"""
import argparse

# noinspection PyUnresolvedReferences
from Class import Range2UniformPrior

##input argument is done
parser = argparse.ArgumentParser(description='To convert a vcf file for the distant matrix which can be used in ABC')
parser.add_argument('--upper',
                    help='the upper limits for all the paramteres on which the simulation will run, should be comman '
                         'separated ',
                    required=True)
parser.add_argument('--lower',
                    help='the lower limits for all the paramteres on which the simulation will run, should be comman '
                         'separated ',
                    required=True)
parser.add_argument('--par_names',
                    help='The names of the parameters. not mandatory. in case not given it will assume '
                         'param1,param2,..')
parser.add_argument('--repeats', help='the number of repeats. default is 1e6', type=float, default=2e4)
args = parser.parse_args()

out = Range2UniformPrior(upper=args.upper, lower=args.lower, variable_names=args.par_names,
                                 repeats=args.repeats)
print(out.to_csv(index=False))
