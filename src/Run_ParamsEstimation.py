#!/usr/bin/python

"""
This file to run ABC-TFK for parameter estimation of a given model
"""
import argparse
import os
import sys

from Classes import ABC

# input argument is done
parser = argparse.ArgumentParser(description='This file will run Parameter estimation.')
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
                help='If two big for the memory use chunk size. relatively slow but no problem with ram', type=float)
sp.add_argument('--ssfile', help="The summary statistics file from real data", required=True)
sp.add_argument('--demography',
                help='The demography.py file full path. If this is given it will assume it has better function cater '
                     'to your own demography. The def ANNModelCheck should be inside')
sp.add_argument('--together',
                help="If the you want to send both train and test together in tfk model fit. Useful for early stoppping"
                     " validation test set. need a specific format for demography.py. Look at Extra/Dynamic.py. Should "
                     "not be used for big test data as it loads in the memory",
                action="store_true")
sp.add_argument('--method',
                help='Method used for R abc classification. can be  "rejection", "loclinear", and "neuralnet". default'
                     ' is "neuralnet" ',
                default='neuralnet', choices=["rejection", "loclinear", "neuralnet"])
sp.add_argument('--tolerance', help='tolerance limit for r abc. default is .01 ', default=.01, type=float)
sp.add_argument('--scale',
                help="To scale the data. n: not to scale anything (default), x: to scale x (ss), y: to scale y "
                     "(parameters), b: to scale both (ss+parameters)",
                default='x', choices=["n", "x", "y", "b"])
sp.add_argument('--cvrepeats', help='The number of time cross validation will be caluclated. default is 100 ',
                default=100, type=int)
sp.add_argument('--csvout',
                help="If the predicted values are needed to out put as csv format for further use in R_ABC",
                action="store_true")
sp.add_argument('--frac', help='If you multiply all the observed ss with some fraction. Important in case simulated '
                               'data and observed data are not from same lengrth.default is 1 ', default=1.0,
                type=float)

sp = subparsers.add_parser('Pre_train', help='To prepare the data for training ANN.')
sp.set_defaults(cmd='Pre_train')
sp.add_argument('info',
                help='the path for the info file. whose first column will be file path (.csv or .csv.gz) and tab '
                     'delimited column with and the number of parameter in that file. ex. <file1.csv.gz> <n>. Only the '
                     'first one will be taken as valid')
sp.add_argument('--folder',
                help='in case you want to run the codes not in current working directory give the path', default='')
sp.add_argument('--chunksize',
                help='If two big for the memory use chunk size. relatively slow but no problem with ram. default 100 ',
                type=int, default=100)
sp.add_argument('--scale',
                help="To scale the data. n: not to scale anything (default), x: to scale x (ss), y: to scale y "
                     "(parameters), b: to scale both (ss+parameters)",
                default='x', choices=["n", "x", "y", "b"])

sp = subparsers.add_parser('Train', help='The training part of the ANN. Should be done after Pre_train part')
sp.set_defaults(cmd='Train')
sp.add_argument('--folder',
                help='in case you want to run the codes not in current working directory give the path', default='')
sp.add_argument('--demography',
                help='The demography.py file full path. If this is given it will assume it has better function cater to'
                     ' your own demography. The def it can have ')
sp.add_argument('--test_size',
                help='test size for r abc. everything else will be used for training purpose. default is 10 thousands',
                default=10000, type=int)
sp.add_argument('--together',
                help="If the you want to send both train and test together in tfk model fit. Useful for early stoppping"
                     " validation test set. need a specific format for demography.py. Look at Extra/Dynamic.py. Should "
                     "not be used for big test data as it loads in the memory",
                action="store_true")

sp = subparsers.add_parser('CV',
                           help='After the training only to get the result of cross validation test. Good for '
                                'unavailable real data')
sp.set_defaults(cmd='CV')
sp.add_argument('--folder',
                help='in case you want to run the codes not in current working directory give the path', default='')
sp.add_argument('--method',
                help='Method used for R abc classification. can be  "rejection", "loclinear", and "neuralnet". default '
                     'is "neuralnet" ',
                default='neuralnet', choices=["rejection", "loclinear", "neuralnet"])
sp.add_argument('--test_size',
                help='test size for r abc. everything else will be used for training purpose. default is 10 thousands',
                default=10000, type=int)
sp.add_argument('--tolerance', help='tolerance limit for r abc. default is .01 ', default=.01, type=float)
sp.add_argument('--cvrepeats', help='The number of time cross validation will be caluclated. default is 100 ',
                default=100, type=int)

sp = subparsers.add_parser('After_train', help='This is to run the ABC analysis after the training part is done')
sp.set_defaults(cmd='After_train')
sp.add_argument('--folder',
                help='in case you want to run the codes not in current working directory give the path', default='')
sp.add_argument('--ssfile', help="The sfs file created from moments", required=True)
sp.add_argument('--test_size',
                help='test size for r abc. everything else will be used for training purpose. default is 10 thousands',
                default=10000, type=int)
sp.add_argument('--method',
                help='Method used for R abc classification. can be  "rejection", "loclinear", and "neuralnet". default '
                     'is "loclinear" ',
                default='loclinear', choices=["rejection", "loclinear", "neuralnet"])
sp.add_argument('--tolerance', help='tolerance limit for r abc. default is .01 ', default=.01, type=float)
sp.add_argument('--cvrepeats', help='The number of time cross validation will be caluclated. default is 100 ',
                default=100, type=int)
sp.add_argument('--csvout',
                help="If the predicted values are needed to out put as csv format for further use in R_ABC",
                action="store_true")
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

    if args.cvrepeats > args.test_size:
        print('CV repeats has to be smaller than the sample size (test_size)')
        print('CV:', args.cvrepeats)
        print('test_size:', args.test_size)
        sys.exit(1)
    # running
    ABC.ABC_TFK_Params(info=args.info, ssfile=args.ssfile, demography=args.demography,together=args.together,
                       method=args.method, tol=args.tolerance, test_size=args.test_size,
                       chunksize=args.chunksize,
                       csvout=args.csvout, scaling_x=scaling_x, scaling_y=scaling_y, cvrepeats=args.cvrepeats,
                       folder=args.folder,frac=args.frac)
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
    ABC.ABC_TFK_Params_PreTrain(info=args.info, chunksize=args.chunksize, test_size=10, scaling_x=scaling_x,
                                scaling_y=scaling_y, folder=args.folder)
elif args.cmd == 'Train':
    ABC.ABC_TFK_Params_Train(demography=args.demography, test_rows=args.test_size, folder=args.folder,
                             together=args.together)
elif args.cmd == 'CV':
    if args.cvrepeats <= args.test_size:
        ABC.ABC_TFK_Params_CV(test_size=args.test_size, tol=args.tolerance, method=args.method,
                              cvrepeats=args.cvrepeats, folder=args.folder)
    else:
        print('CV repeats has to be smaller than the sample size (test_size)')
        print('CV:', args.cvrepeats)
        print('test_size:', args.test_size)
        sys.exit(1)
elif args.cmd == 'After_train':
    if not os.path.isfile(args.ssfile):
        print("The sfs file could not be found please check")
    if args.cvrepeats > args.test_size:
        print('CV repeats has to be smaller than the sample size (test_size)')
        print('CV:', args.cvrepeats)
        print('test_size:', args.test_size)
        sys.exit(1)
    ABC.ABC_TFK_Params_After_Train(ssfile=args.ssfile, test_size=args.test_size, tol=args.tolerance, method=args.method,
                                   csvout=args.csvout, cvrepeats=args.cvrepeats, folder=args.folder,frac=args.frac)
