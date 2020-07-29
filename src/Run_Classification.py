#!/usr/bin/python

"""
This file to run ABC-TFK for classification between models
"""
import argparse

from Classes import ABC

# to stop future warning every time to print out

# input argument is done
parser = argparse.ArgumentParser(description='This is run classification between models using any summary statistics.\n'
                                             'The info file should look like this\n'
                                             'file1.csv.gz    10\n'
                                             'file2.csv.gz    14\n', formatter_class=argparse.RawTextHelpFormatter)
subparsers = parser.add_subparsers(help='sub-commands help')

sp = subparsers.add_parser('All', help='The whole run of the NN for parameter estimation from first to last')
sp.set_defaults(cmd='All')
sp.add_argument('info',
                help='the path for the info file. whose first column will be file path (.csv or .csv.gz) and tab '
                     'delimited column with and the number of parameter in that file. ex. <file1.csv.gz> <n>')
sp.add_argument('--folder',
                help='in case you want to run the codes not in current working directory give the path', default='')
sp.add_argument('--test_size',
                help='test size for r abc. everything else will be used for training purpose. default is 10 thousands',
                default=10000, type=int)
sp.add_argument('--chunksize',
                help='If two big for the memory use chunk size. relatively slow but no problem with ram',
                type=float)
sp.add_argument('--scale', help="To scale the data ", action="store_true")
sp.add_argument('--ssfile', help="The summary statistics file from real data", required=True)
sp.add_argument('--demography',
                help='The demography.py file full path. If this is given it will assume it has better function cater to'
                     ' your own demography. The def ANNModelCheck should be inside')
sp.add_argument('--together',
                help="If the you want to send both train and test together in tfk model fit. Useful for early stoppping"
                     " validation test set. need a specific format for demography.py. Look at "
                     "Extra/ModelClassTogether.py. Should not be used for big test data as it loads in the memory",
                action="store_true")
sp.add_argument('--method',
                help='Method used for R abc classification. can be  "rejection", "mnlogistic", "neuralnet". default is'
                     ' rejection" ',
                default='rejection', choices=["rejection", "mnlogistic", "neuralnet"])
sp.add_argument('--tolerance', help='tolerance limit for r abc. default is .005 ', default=.005, type=float)
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
                     'delimited column with and the number of parameter in that file. ex. <file1.csv.gz> <n>')
sp.add_argument('--folder',
                help='in case you want to run the codes not in current working directory give the path', default='')
sp.add_argument('--chunksize',
                help='If two big for the memory use chunk size. relatively slow but no problem with ram. In this case,'
                     ' chunksize is mandatory. default value 10000',
                type=float, default=10000)
sp.add_argument('--scale', help="To scale the data ", action="store_true")

sp = subparsers.add_parser('Train', help='The training part of the ANN. Should be done after Pre_train part')
sp.set_defaults(cmd='Train')
sp.add_argument('--folder',
                help='in case you want to run the codes not in current working directory give the path', default='')
sp.add_argument('--demography',
                help='The demography.py file full path. If this is given it will assume it has better function cater to'
                     ' your own demography. The def ANNModelCheck should be inside')
sp.add_argument('--test_size',
                help='test size for r abc. everything else will be used for training purpose. default is 10 thousands',
                default=10000, type=int)
sp.add_argument('--together',
                help="If the you want to send both train and test together in tfk model fit. Useful for early stoppping"
                     " validation test set. need a specific format for demography.py. Look at "
                     "Extra/ModelClassTogether.py. Should not be used for big test data as it loads in the memory",
                action="store_true")
sp = subparsers.add_parser('CV',
                           help='After the training only to get the result of cross validation test. Good for '
                                'unavailable real data')
sp.set_defaults(cmd='CV')
sp.add_argument('--folder',
                help='in case you want to run the codes not in current working directory give the path', default='')
sp.add_argument('--test_size',
                help='test size for r abc. everything else will be used for training purpose. default is 10 thousands',
                default=10000, type=int)
sp.add_argument('--method',
                help='Method used for R abc classification. can be  "rejection", "mnlogistic", "neuralnet". default is'
                     ' rejection" ',
                default='rejection', choices=["rejection", "mnlogistic", "neuralnet"])
sp.add_argument('--tolerance', help='tolerance limit for r abc. default is .005 ', default=.005, type=float)
sp.add_argument('--cvrepeats', help='The number of time cross validation will be caluclated. default is 100 ',
                default=100, type=int)

sp = subparsers.add_parser('After_train', help='This is to run the ABC analysis after the training part is done')
sp.set_defaults(cmd='After_train')
sp.add_argument('--folder',
                help='in case you want to run the codes not in current working directory give the path', default='')
sp.add_argument('--ssfile', help="The summary statistics file from real data", required=True)
sp.add_argument('--test_size',
                help='test size for r abc. everything else will be used for training purpose. default is 10 thousands',
                default=10000, type=int)
sp.add_argument('--method',
                help='Method used for R abc classification. can be  "rejection", "mnlogistic", "neuralnet". default is'
                     ' rejection" ',
                default='rejection', choices=["rejection", "mnlogistic", "neuralnet"])
sp.add_argument('--tolerance', help='tolerance limit for r abc. default is .005 ', default=.005, type=float)
sp.add_argument('--cvrepeats', help='The number of time cross validation will be caluclated. default is 100 ',
                default=100, type=int)
sp.add_argument('--csvout',
                help="If the predicted values are needed to out put as csv format for further use in R_ABC",
                action="store_true")
sp.add_argument('--frac', help='If you multiply all the observed ss with some fraction. Important in case simulated '
                               'data and observed data are not from same lengrth.default is 1 ', default=1.0,
                type=float)
args = parser.parse_args()

# running
if args.cmd == 'All':
    if args.chunksize:
        args.chunksize = int(args.chunksize)
    ABC.ABC_TFK_Classification(info=args.info, ssfile=args.ssfile, chunksize=args.chunksize,
                               demography=args.demography, method=args.method, together=args.together,
                               tolerance=args.tolerance, test_size=args.test_size, scale=args.scale, csvout=args.csvout,
                               cvrepeats=args.cvrepeats, folder=args.folder,frac=args.frac)
elif args.cmd == 'Pre_train':
    if args.chunksize:
        args.chunksize = int(args.chunksize)
    ABC.ABC_TFK_Classification_PreTrain(info=args.info, test_size=10, chunksize=args.chunksize,
                                        scale=args.scale, folder=args.folder)
elif args.cmd == 'Train':
    ABC.ABC_TFK_Classification_Train(demography=args.demography, test_rows=args.test_size, folder=args.folder,
                                     together=args.together)

elif args.cmd == 'CV':
    ABC.ABC_TFK_Classification_CV(test_size=args.test_size, tol=args.tolerance, method=args.method,
                                  cvrepeats=args.cvrepeats, folder=args.folder)

elif args.cmd == 'After_train':
    ABC.ABC_TFK_Classification_After_Train(ssfile=args.ssfile, test_size=args.test_size, tol=args.tolerance,
                                           method=args.method, csvout=args.csvout, cvrepeats=args.cvrepeats,
                                           folder=args.folder,frac=args.frac)
