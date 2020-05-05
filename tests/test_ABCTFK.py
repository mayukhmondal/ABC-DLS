#!/usr/bin/python

"""
This file to test the Classification part of the ABC-TFK file
"""
import sys

sys.path.append("..")

from pathlib import Path
from src.Classes import ABC
from src.Classes import Misc
from collections import Counter
from keras.utils import HDF5Matrix
import pandas
import numpy
import joblib
import shutil
import os
if os.path.isdir('cls'):
    shutil.rmtree('cls')
if os.path.isdir('par'):
    shutil.rmtree('par')

def test_Classification_Pre_train(info: str = 'Model.info', test_size: int = 1, chunksize: int = 5, scale: bool = True,
                                  outfolder: str = 'cls'):
    # main check
    assert ABC.ABC_TFK_Classification_PreTrain(info=info, test_size=test_size, chunksize=chunksize,
                                               scale=scale,
                                               folder=outfolder), 'The code even did not run properly for pretrain ' \
                                                                  'classification '

    # file existen check
    files = ['cls/x.h5', 'cls/y.h5', 'cls/y_cat_dict.txt', 'cls/scale_x.sav']
    not_exist = [file for file in files if not Path(file).exists()]
    assert not not_exist, f'{not_exist} file was not created by ABC_TFK_Classification_PreTrain'

    # model name check
    assert eval(open('cls/y_cat_dict.txt', 'r').read()), 'Cant read the y_cat_dict.txt file. Look likes the' \
                                                         'format is not simple text dict '

    y_cat_names = list(eval(open('cls/y_cat_dict.txt', 'r').read()).values())
    assert Counter(y_cat_names) == Counter(
        ['OOA', 'OOA_B', 'OOA_M']), 'Model.info model names do not match with y_cat_dict.txt'

    # checking scale file
    assert joblib.load('cls/scale_x.sav'), "cant read the file properly. scale_x.sav"
    # h5 file checks
    assert HDF5Matrix('cls/x.h5',
                      'mydata'), 'Cant even read x.h5. The file do not looks like h5py. At least keras cant read it'
    assert HDF5Matrix('cls/y.h5',
                      'mydata'), 'Cant even read y.h5. The file do not looks like h5py. At least keras cant read it'

    xshape = HDF5Matrix('cls/x.h5', 'mydata').shape
    yshape = HDF5Matrix('cls/y.h5', 'mydata').shape
    # row check
    assert xshape[0] == yshape[0], 'Row numbers between x.h5 and/or y.h5 file do not match'
    assert xshape[0] == 15, 'Row numbers of h5 files do not match with expected number of rows'
    # ycol check
    assert yshape[1] == 3, 'Number of columns in y.h5 do not match with the number of names present in ' \
                           'Model.info'
    # ncol check
    assert xshape[1] == 1331, 'Column number of x.h5 do not match with expected'
    # checking y.h5 file
    # y.h5 either 0 and 1
    y_unique_val = list(numpy.unique(pandas.DataFrame(HDF5Matrix('cls/y.h5', 'mydata')[:]).values.flatten()))
    assert Counter(y_unique_val) == Counter([1, 0.0]), 'y values can either be 0 or 1'

    # the number of count  of models
    y = HDF5Matrix('cls/y.h5', 'mydata')
    y_cat_dict = eval(open('cls/y_cat_dict.txt', 'r').read())
    modelcount = pandas.DataFrame(numpy.argmax(y[:], axis=1, out=None))[0].replace(y_cat_dict).value_counts().to_dict()
    assert modelcount == {'OOA': 5, 'OOA_M': 5, 'OOA_B': 5}, 'y.h5 have a different count of names than it should'

    # checking x.h5 file'
    x = HDF5Matrix('cls/x.h5', 'mydata')
    xdf = pandas.DataFrame(x[:])
    # col types in x.h5
    assert all(xdf[i].dtype.kind == 'f' for i in xdf), 'Not all the columns are float for x.h5'
    # min and max scaler worked or noy
    assert xdf.iloc[:, 1:-1].min().sum() == 0, 'Not all the columns have 0 minimum value'
    assert xdf.iloc[:, 1:-1].max().sum() == 1329, 'Not all the columns have 1 maximum value'

    # reading csv files
    files = ['OOA.csv', 'OOA_B.csv', 'OOA_M.csv']
    params = [7, 12, 12]
    df = pandas.DataFrame()
    for i, file in enumerate(files):
        if df.empty:
            df = pandas.read_csv(file).iloc[:, params[i]:]
        else:
            temp = pandas.read_csv(file).iloc[:, params[i]:]
            df = pandas.concat([df, temp], ignore_index=True)

    ss = df.astype('float32')
    scale_x = joblib.load('cls/scale_x.sav')
    xss = pandas.DataFrame(scale_x.inverse_transform(xdf.values)).astype('float32')
    xss.columns = ss.columns
    # if x.h5 == concat csv files
    pandas.testing.assert_frame_equal(ss.sort_values(by=list(ss.columns)).reset_index(drop=True),
                                      xss.sort_values(by=list(xss.columns)).reset_index(
                                          drop=True)), 'The x.h5 do not match with the test data'
    # if x.h5 truly randomized
    assert (ss - xss).abs().values.sum() > 0, 'Looks like x.h5 did not reshuffled'

    # y.h5 and x.h5 correspondence
    # will do it later


def test_Classification_Train(demography: str = '../src/extras/ModelClass.py', test_size: int = 0, folder: str = 'cls'):
    # main check
    ABC.ABC_TFK_Classification_Train(demography=demography, test_rows=test_size, folder=folder)

    # file checks
    files = ['cls/ModelClassification.h5']
    not_exist = [file for file in files if not Path(file).exists()]
    assert not not_exist, f'{not_exist} file was not created by ABC_TFK_Classification_Train'


def test_Classification_CV(test_size: int = 15, tol: float = 0.5, method: str = 'mnlogistic', folder: str = 'cls'):
    # main check
    ABC.ABC_TFK_Classification_CV(test_size=test_size, tol=tol, method=method, cvrepeats=2, folder=folder)
    # file checks
    files = ['cls/CV.pdf']
    not_exist = [file for file in files if not Path(file).exists()]
    assert not not_exist, f'{not_exist} file was not created by ABC_TFK_Classification_CV'


def test_Classification_After_train(test_size: int = 15, tol: float = 0.5, method: str = 'mnlogistic',
                                    ssfile: str = '../examples/YRI_CEU_CHB.observed.csv', cvrepeats: int = 2,
                                    folder: str = 'cls',csvout=True):
    # main check
    ABC.ABC_TFK_Classification_After_Train(test_size=test_size, tol=tol, method=method, cvrepeats=cvrepeats,
                                           ssfile=ssfile, folder=folder,csvout=csvout)

    # file checks
    files = ['cls/NN.pdf','cls/model_index.csv.gz','cls/ss_predicted.csv.gz','cls/ss_target.csv.gz']
    not_exist = [file for file in files if not Path(file).exists()]
    assert not not_exist, f'{not_exist} file was not created by ABC_TFK_Classification_CV'


def test_Params_Pre_train(info: str = 'Model.info', test_size: int = 1, chunksize: int = 5, scale: bool = True,
                          folder: str = 'par'):
    # main check
    assert ABC.ABC_TFK_Params_PreTrain(info=info, test_size=test_size, chunksize=chunksize,
                                       scaling_x=scale,
                                       scaling_y=scale,folder=folder), 'The code did not even run properly for pretrain ' \
                                                         'parameter estimation '
    # file existen check
    files = ['par/x.h5', 'par/y.h5', 'par/scale_x.sav','par/scale_y.sav','par/params_header.csv']
    not_exist = [file for file in files if not Path(file).exists()]
    assert not not_exist, f'{not_exist} file was not created by ABC_TFK_Params_PreTrain'

def test_Params_Train(demography: str = '../src/extras/ModelParams.py', test_size: int = 0, folder: str = 'par'):

    #main check
    ABC.ABC_TFK_Params_Train(demography=demography, test_rows=test_size,folder=folder)

    # file checks
    files = ['par/ModelParamPrediction.h5']
    not_exist = [file for file in files if not Path(file).exists()]
    assert not not_exist, f'{not_exist} file was not created by ABC_TFK_Params_Train'

def test_Params_CV(test_size: int = 5, tol: float = 0.5, method: str = 'rejection', folder: str = 'par',cvrepeats:int=1):
    # main check
    ABC.ABC_TFK_Params_CV(test_size=test_size, tol=tol, method=method,cvrepeats=cvrepeats,folder=folder)

    # file checks
    files = ['par/nnparamcv.pdf','par/nnparamcv_together.pdf']
    not_exist = [file for file in files if not Path(file).exists()]
    assert not not_exist, f'{not_exist} file was not created by ABC_TFK_Params_CV'

def test_Params_After_train(test_size: int = 5, tol: float = 0.5, method: str = 'rejection',
                                    ssfile: str = '../examples/YRI_CEU_CHB.observed.csv', cvrepeats: int = 2,
                                    folder: str = 'par',csvout=True):
    # main check
    ABC.ABC_TFK_Params_After_Train(test_size=test_size, tol=tol, method=method, cvrepeats=cvrepeats,
                                           ssfile=ssfile,folder=folder,csvout=csvout)

    # file checks
    files = ['par/paramposterior.pdf','par/params.csv.gz','par/ss_predicted.csv.gz','par/ss_target.csv.gz']
    not_exist = [file for file in files if not Path(file).exists()]
    assert not not_exist, f'{not_exist} file was not created by ABC_TFK_Params_After_Train'
#
# def test_Params_Train_together(demography: str = '../src/extras/Dynamic.py', test_size: int = 1000, folder: str = 'par'):
#
#     #main check
#     if os.path.isdir('par'):
#         shutil.rmtree('par')
#     from tensorflow.keras import backend
#     backend.clear_session()
#     ABC.ABC_TFK_Params_PreTrain(info='Model2.info', chunksize=100,
#                                 scaling_x=True,
#                                 scaling_y=True, folder=folder)
#     y_train = ABC.ABC_TFK_Classification_Train.reading_train(file=folder + '/y.h5', test_rows=test_size)
#     assert 1000==y_train.shape[0]
#
#     ABC.ABC_TFK_Params_Train(demography=demography, test_rows=test_size,folder=folder,together=True)
#
#     # file checks
#     files = ['par/ModelParamPrediction.h5']
#     not_exist = [file for file in files if not Path(file).exists()]
#     assert not not_exist, f'{not_exist} file was not created by ABC_TFK_Params_Train_together'

# if os.path.isdir('cls'):
#     shutil.rmtree('cls')
# if os.path.isdir('par'):
#     shutil.rmtree('par')