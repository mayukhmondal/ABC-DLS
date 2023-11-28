#!/usr/bin/python

"""
This file to run ABC-DLS for parameter estimation of a given model for specifically Sequential Monte Carlo Approach
"""
import argparse
import os

from Classes import ABC
from Classes import Misc
from _version import __version__

# input argument is done
parser = argparse.ArgumentParser(description='This file will run Nested Sampling Approach for Parameter Estimation.')
parser.add_argument('-v', '--version', action='version',
                    version='ABC-DLS {version}'.format(version=__version__))
subparsers = parser.add_subparsers(help='sub-commands help')

sp = subparsers.add_parser('All', help='The whole run of the NN for parameter estimation from first to last')
sp.set_defaults(cmd='All')
sp.add_argument('info',
                help='the path for the info file. whose first column will be file path (.csv or .csv.gz) and tab '
                     'delimited column with and the number of parameter in that file. ex. <file1.csv.gz> <n>. Only '
                     'the first one will be taken as valid', type=lambda x: Misc.args_valid_file(parser, x))
sp.add_argument('--folder',
                help='in case you want to run the codes not in current working directory give the path', default='')
sp.add_argument('--test_size',
                help='test size for r abc. everything else will be used for training purpose. default is 10 thousands',
                default=10000, type=int)
sp.add_argument('--chunksize',
                help='If two big for the memory use chunk size. relatively slow but no problem with ram', type=float,
                default=1e2)
sp.add_argument('--ssfile', help="The summary statistics file from real data with header. Can have multiple line"
                                 " assuming every single line is different run of same summary statistics for different"
                                 "individuals. The predictions from NN will be averaged in case of multiple"
                                 " lines.", required=True, type=lambda x: Misc.args_valid_file(parser, x))
sp.add_argument('--nn',
                help='The NeuralNetwork.py file full path. If this is given it will assume it has better function cater '
                     'to your own problem. The def ANNModelCheck should be inside. One example extras/Dynamic.py',
                type=lambda x: Misc.args_valid_file(parser, x))
sp.add_argument('--resume',
                help='The path of already ran ModelParamPrediction.h5 and resume training from there. --nn not needed'
                     'if you use this option but you need the --resume_fit in case you do not want to use the default'
                     'fitting approach',
                type=lambda x: Misc.args_valid_file(parser, x))
sp.add_argument('--resume_fit',
                help='In case you want a special fit and do not want to use the default model.fit only with --resume. '
                     'As resume cannot use the older model fit parameter. path of the file One example is '
                     'extras/Resume_fit.py',
                type=lambda x: Misc.args_valid_file(parser, x))
sp.add_argument('--method',
                help='Method used for R abc classification. can be  "rejection", "loclinear", and "neuralnet". default'
                     ' is "rejection" ',
                default='rejection', choices=["rejection", "loclinear", "neuralnet"])
sp.add_argument('--tolerance', help='tolerance limit for r abc. default is .01 ', default=.01, type=float)
sp.add_argument('--scale',
                help="To scale the data. n: not to scale anything (default), x: to scale x (ss), y: to scale y "
                     "(parameters), b: to scale both (ss+parameters). default is b",
                default='b', choices=["n", "x", "y", "b"])
sp.add_argument('--csvout',
                help="If you want reuse the simulations with new updated range",
                action="store_true")
sp.add_argument('--decrease', help='minimum amount of decrease of the range needed to register as true. default is .95.'
                                   ' lower means  stronger filter ', default=.95, type=float)
sp.add_argument('--frac', help='If you multiply all the observed ss with some fraction. Important in case simulated '
                               'data and observed data are not from same length.default is 1 ', default=1.0,
                type=float)
sp.add_argument('--increase', help='If you want to increase the new range. Important in case the newrange has '
                                   'missed the true parameters. The value is in fraction to distance of new range'
                                   ' min and max (similar to decrease). a good value is 5 times lower than '
                                   '1-decrease. Only increase in case of no decrease is detected (>decrease). It '
                                   'is better to use hardrange to make it understand what should '
                                   'be the hard cut off if not newrange can be outside of possible values. '
                                   'default is  0', default=0.0, type=float)
sp.add_argument('--hardrange',
                help="csv format of hardrange file path. Should have 3 columns. params_names, lower and upper limit. "
                     "every row is define a parameters. no header. same as Newrange.csv. important to define what is "
                     "possible for range", type=lambda x: Misc.args_valid_file(parser, x))
sp = subparsers.add_parser('Pre_train', help='To prepare the data for training ANN.')
sp.set_defaults(cmd='Pre_train')
sp.add_argument('info',
                help='the path for the info file. whose first column will be file path (.csv or .csv.gz) and tab '
                     'delimited column with and the number of parameter in that file. ex. <file1.csv.gz> <n>. Only the '
                     'first one will be taken as valid', type=lambda x: Misc.args_valid_file(parser, x))
sp.add_argument('--folder',
                help='in case you want to run the codes not in current working directory give the path', default='')
sp.add_argument('--chunksize',
                help='If two big for the memory use chunk size. relatively slow but no problem with ram. default 100 ',
                type=int, default=100)
sp.add_argument('--scale',
                help="To scale the data. n: not to scale anything (default), x: to scale x (ss), y: to scale y "
                     "(parameters), b: to scale both (ss+parameters)",
                default='b', choices=["n", "x", "y", "b"])
sp.add_argument('--resume',
                help='The path of already ran ModelParamPrediction.h5 and resume training from there. --nn not needed'
                     'if you use this option but you need the --resume_fit in case you do not want to use the default'
                     'fitting approach',
                type=lambda x: Misc.args_valid_file(parser, x))

args = parser.parse_args()
scaling_x, scaling_y = False, False
# checking inputs
if args.cmd == 'All':
    if not os.path.isfile(args.ssfile):
        print("The sfs file could not be found please check")
    if args.chunksize:
        args.chunksize = int(args.chunksize)
    if args.scale == 'n':
        scaling_x = False
        scaling_y = False
    elif args.scale == 'x':
        scaling_x = True
        scaling_y = False
    elif args.scale == 'y':
        scaling_x = False
        scaling_y = True
    elif args.scale == 'b':
        scaling_x = True
        scaling_y = True
    # running
    newrange = ABC.ABC_DLS_SMC(info=args.info, ssfile=args.ssfile, nn=args.nn, resume=args.resume,
                               resume_fit=args.resume_fit, method=args.method, tol=args.tolerance,
                               test_size=args.test_size, chunksize=args.chunksize, csvout=args.csvout,
                               scaling_x=scaling_x, scaling_y=scaling_y, decrease=args.decrease,
                               folder=args.folder, frac=args.frac, increase=args.increase,
                               hardrange_file=args.hardrange)
    print(newrange)
elif args.cmd == 'Pre_train':
    if args.chunksize:
        args.chunksize = int(args.chunksize)
    if args.scale == 'n':
        scaling_x = False
        scaling_y = False
    elif args.scale == 'x':
        scaling_x = True
        scaling_y = False
    elif args.scale == 'y':
        scaling_x = False
        scaling_y = True
    elif args.scale == 'b':
        scaling_x = True
        scaling_y = True
    ABC.ABC_DLS_SMC_PreTrain(info=args.info, chunksize=args.chunksize, scaling_x=scaling_x,
                                scaling_y=scaling_y, folder=args.folder, resume=args.resume)
