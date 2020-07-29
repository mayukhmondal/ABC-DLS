#!/usr/bin/python

"""
This file to run ABC-TFK for parameter estimation of a given model
"""
import argparse
import os

from Classes import ABC

# input argument is done
parser = argparse.ArgumentParser(description='The info file should look like this\n'
                                             'file1.csv.gz    10\n'
                                             'file2.csv.gz    14\n', formatter_class=argparse.RawTextHelpFormatter)
subparsers = parser.add_subparsers(help='sub-commands help')

sp = subparsers.add_parser('All', help='The whole run of the NN for parameter estimation from first to last')
sp.set_defaults(cmd='All')
sp.add_argument('info',
                help='the path for the info file. whose first column will be file path (.csv or .csv.gz) and tab '
                     'delimited column with and the number of parameter in that file. ex. <file1.csv.gz> <n>. Only '
                     'the first one will be taken as valid')
sp.add_argument('--folder',
                help='in case you want to run the codes not in current working directory give the path', default='')
sp.add_argument('--test_size',
                help='test size for r abc. everything else will be used for training purpose. default is 10 thousands',
                default=10000, type=int)
sp.add_argument('--chunksize',
                help='If two big for the memory use chunk size. relatively slow but no problem with ram', type=float,
                default=1e2)
sp.add_argument('--ssfile', help="The summary statistics file from real data", required=True)
sp.add_argument('--demography',
                help='The demography.py file full path. If this is given it will assume it has better function cater '
                     'to your own demography. The def ANNModelCheck should be inside')
sp.add_argument('--method',
                help='Method used for R abc classification. can be  "rejection", "loclinear", and "neuralnet". default'
                     ' is "rejection" ',
                default='rejection', choices=["rejection", "loclinear", "neuralnet"])
sp.add_argument('--tolerance', help='tolerance limit for r abc. default is .01 ', default=.01, type=float)
sp.add_argument('--scale',
                help="To scale the data. n: not to scale anything (default), x: to scale x (ss), y: to scale y "
                     "(parameters), b: to scale both (ss+parameters). deafult is b",
                default='b', choices=["n", "x", "y", "b"])
sp.add_argument('--csvout',
                help="If you want reuse the simulations with new updated range",
                action="store_true")
sp.add_argument('--imp', help='minimum amount of improvement needed to register as true. default is .95. lower means '
                              'stronger filter ', default=.95, type=float)
sp.add_argument('--frac', help='If you multiply all the observed ss with some fraction. Important in case simulated '
                               'data and observed data are not from same lengrth.default is 1 ', default=1.0,
                type=float)
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
    newrange = ABC.ABC_TFK_NS.wrapper(info=args.info, ssfile=args.ssfile, demography=args.demography,
                                      method=args.method, tol=args.tolerance, test_size=args.test_size,
                                      chunksize=args.chunksize,csvout=args.csvout,
                                      scaling_x=scaling_x, scaling_y=scaling_y,imp=args.imp,
                                      folder=args.folder,frac=args.frac)
    print(newrange)
