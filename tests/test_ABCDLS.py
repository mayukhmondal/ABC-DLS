#!/usr/bin/python

"""
This file to test the Classification part of the ABC-DLS file
"""
import sys

sys.path.append("..")

from pathlib import Path
from src.Classes import ABC
from src.Classes import Misc
from SFS import Class

from collections import Counter
from tensorflow.keras.utils import HDF5Matrix
import pandas
import numpy
import joblib
import shutil
import os


def test_Classification_Pre_train(info: str = 'Model.info', test_size: int = 1, chunksize: int = 5, scale: bool = True,
                                  outfolder: str = 'cls'):
    # main check
    assert ABC.ABC_DLS_Classification_PreTrain(info=info, test_size=test_size, chunksize=chunksize,
                                               scale=scale,
                                               folder=outfolder), 'The code even did not run properly for pretrain ' \
                                                                  'classification '

    # file existen check
    files = ['cls/x.h5', 'cls/y.h5', 'cls/y_cat_dict.txt', 'cls/scale_x.sav']
    not_exist = [file for file in files if not Path(file).exists()]
    assert not not_exist, f'{not_exist} file was not created by ABC_DLS_Classification_PreTrain'

    # model name check
    assert eval(open('cls/y_cat_dict.txt', 'r').read()), 'Cant read the y_cat_dict.txt file. Look likes the' \
                                                         'format is not simple text dict '

    y_cat_names = list(eval(open('cls/y_cat_dict.txt', 'r').read()).values())
    assert Counter(y_cat_names) == Counter(
        ['BNDX', 'MNDX', 'SNDX']), 'Model.info model names do not match with y_cat_dict.txt'

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
    assert modelcount == {'BNDX': 5, 'MNDX': 5, 'SNDX': 5}, 'y.h5 have a different count of names than it should'

    # checking x.h5 file'
    x = HDF5Matrix('cls/x.h5', 'mydata')
    xdf = pandas.DataFrame(x[:])
    # col types in x.h5
    assert all(xdf[i].dtype.kind == 'f' for i in xdf), 'Not all the columns are float for x.h5'
    # min and max scaler worked or noy
    assert xdf.iloc[:, 1:-1].min().sum() == 0, 'Not all the columns have 0 minimum value'
    assert xdf.iloc[:, 1:-1].max().sum() == 1329, 'Not all the columns have 1 maximum value'

    # reading csv files
    files = ['BNDX.csv', 'MNDX.csv', 'SNDX.csv']
    params = [24, 24, 19]
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


def test_Classification_Train(nn: str = '../src/extras/ModelClass.py', test_size: int = 0, folder: str = 'cls'):
    # main check
    ABC.ABC_DLS_Classification_Train(nn=nn, test_rows=test_size, folder=folder)

    # file checks
    files = ['cls/ModelClassification.h5']
    not_exist = [file for file in files if not Path(file).exists()]
    assert not not_exist, f'{not_exist} file was not created by ABC_DLS_Classification_Train'


def test_Classification_CV(test_size: int = 15, tol: float = 0.5, method: str = 'mnlogistic', folder: str = 'cls'):
    # main check
    ABC.ABC_DLS_Classification_CV(test_size=test_size, tol=tol, method=method, cvrepeats=2, folder=folder)
    # file checks
    files = ['cls/CV.pdf']
    not_exist = [file for file in files if not Path(file).exists()]
    assert not not_exist, f'{not_exist} file was not created by ABC_DLS_Classification_CV'


def test_Classification_After_train(test_size: int = 15, tol: float = 0.5, method: str = 'mnlogistic',
                                    ssfile: str = '../examples/YRI_FRN_HAN.observed.csv', cvrepeats: int = 2,
                                    folder: str = 'cls', csvout=True):
    # main check
    ABC.ABC_DLS_Classification_After_Train(test_size=test_size, tol=tol, method=method, cvrepeats=cvrepeats,
                                           ssfile=ssfile, folder=folder, csvout=csvout)

    # file checks
    files = ['cls/NN.pdf', 'cls/model_index.csv.gz', 'cls/ss_predicted.csv.gz', 'cls/ss_target.csv.gz']
    not_exist = [file for file in files if not Path(file).exists()]
    assert not not_exist, f'{not_exist} file was not created by ABC_DLS_Classification_CV'


def test_Classification_Train_together(info: str = 'Model.info', nn: str = '../src/extras/ModelClassTogether.py',
                                       test_size: int = 3,
                                       folder: str = 'cls'):
    # main check
    ABC.ABC_DLS_Classification_PreTrain(info=info, chunksize=1, scale=True, folder=folder)
    y_train, _ = ABC.ABC_DLS_Classification_Train.train_test_split_hdf5(file=folder + '/y.h5', test_rows=test_size)
    assert 12 == y_train.shape[0]

    ABC.ABC_DLS_Classification_Train(nn=nn, test_rows=test_size, folder=folder, together=True)

    # file checks
    files = ['cls/ModelClassification.h5']
    not_exist = [file for file in files if not Path(file).exists()]
    assert not not_exist, f'{not_exist} file was not created by ABC_DLS_Classification_Train_together'
    if os.path.isdir('cls'):
        shutil.rmtree('cls')


def test_Params_Pre_train(info: str = 'Model.info', test_size: int = 1, chunksize: int = 5, scale: bool = True,
                          folder: str = 'par'):
    # main check
    assert ABC.ABC_DLS_Params_PreTrain(info=info, test_size=test_size, chunksize=chunksize,
                                       scaling_x=scale,
                                       scaling_y=scale,
                                       folder=folder), 'The code did not even run properly for pretrain ' \
                                                       'parameter estimation '
    # file existen check
    files = ['par/x.h5', 'par/y.h5', 'par/scale_x.sav', 'par/scale_y.sav', 'par/params_header.csv']
    not_exist = [file for file in files if not Path(file).exists()]
    assert not not_exist, f'{not_exist} file was not created by ABC_DLS_Params_PreTrain'


def test_Params_Train(nn: str = '../src/extras/ModelParams.py', test_size: int = 0, folder: str = 'par'):
    # main check
    ABC.ABC_DLS_Params_Train(nn=nn, test_rows=test_size, folder=folder)

    # file checks
    files = ['par/ModelParamPrediction.h5']
    not_exist = [file for file in files if not Path(file).exists()]
    assert not not_exist, f'{not_exist} file was not created by ABC_DLS_Params_Train'


def test_Params_CV(test_size: int = 5, tol: float = 0.5, method: str = 'rejection', folder: str = 'par',
                   cvrepeats: int = 1):
    # main check
    ABC.ABC_DLS_Params_CV(test_size=test_size, tol=tol, method=method, cvrepeats=cvrepeats, folder=folder)

    # file checks
    files = ['par/nnparamcv.pdf', 'par/nnparamcv_together.pdf']
    not_exist = [file for file in files if not Path(file).exists()]
    assert not not_exist, f'{not_exist} file was not created by ABC_DLS_Params_CV'


def test_Params_After_train(test_size: int = 5, tol: float = 0.5, method: str = 'rejection',
                            ssfile: str = '../examples/YRI_FRN_HAN.observed.csv', cvrepeats: int = 2,
                            folder: str = 'par', csvout=True):
    # main check
    ABC.ABC_DLS_Params_After_Train(test_size=test_size, tol=tol, method=method, cvrepeats=cvrepeats,
                                   ssfile=ssfile, folder=folder, csvout=csvout)

    # file checks
    files = ['par/paramposterior.pdf', 'par/params.csv.gz', 'par/ss_predicted.csv.gz', 'par/ss_target.csv.gz']
    not_exist = [file for file in files if not Path(file).exists()]
    assert not not_exist, f'{not_exist} file was not created by ABC_DLS_Params_After_Train'


if os.path.isdir('par2'):
    shutil.rmtree('par2')


def test_Params_Train_together(info: str = 'Model.info', nn: str = '../src/extras/Dynamic.py', test_size: int = 1,
                               folder: str = 'par'):
    # main check
    ABC.ABC_DLS_Params_PreTrain(info=info, chunksize=1,
                                scaling_x=True,
                                scaling_y=True, folder=folder)
    y_train, _ = ABC.ABC_DLS_Classification_Train.train_test_split_hdf5(file=folder + '/y.h5', test_rows=test_size)
    assert 4 == y_train.shape[0]

    ABC.ABC_DLS_Params_Train(nn=nn, test_rows=test_size, folder=folder, together=True)

    # file checks
    files = ['par/ModelParamPrediction.h5']
    not_exist = [file for file in files if not Path(file).exists()]
    assert not not_exist, f'{not_exist} file was not created by ABC_DLS_Params_Train_together'
    if os.path.isdir('par'):
        shutil.rmtree('par')


def test_ABC_DLS_SMC(info: str = 'Model2.info', nn: str = '../src/extras/ModelParamsTogether.py',
                    ssfile: str = '../examples/YRI_FRN_HAN.observed.csv',
                    chunksize: int = 100, test_size: int = 100, tol: float = 0.5, method: str = 'rejection',
                    csvout=True, folder: str = 'ns', increase=0.005):
    ABC.ABC_DLS_SMC(info=info, ssfile=ssfile, chunksize=chunksize, test_size=test_size, tol=tol, method=method,
                   csvout=csvout, folder=folder, nn=nn, increase=increase,scaling_x=True,
                scaling_y = True)
    files = ['ns/ModelParamPrediction.h5', 'ns/Narrowed.csv', 'ns/params_header.csv', 'ns/x.h5', 'ns/y.h5',
             'ns/Newrange.csv']
    not_exist = [file for file in files if not Path(file).exists()]
    assert not not_exist, f'{not_exist} file was not created by ABC_DLS_SMC'
    if os.path.isdir('ns'):
        shutil.rmtree('ns')


def test_vcf2ss(vcffile='../examples/Examples.vcf.gz', popfile='../examples/Input.tsv', sfs_pop=('YRI', 'FRN', 'HAN'),
                chunk_length=int(100), out='test_out'):
    out = Class.VCF2SFS.wrapper(vcffile=vcffile, popfile=popfile, sfs_pop=sfs_pop, chunk_length=chunk_length, out=out)
    print(out.sum())
    assert 22547 == out.sum(), 'The total number of segregating sites do not match with vcf'
    files = ['test_out.csv']
    not_exist = [file for file in files if not Path(file).exists()]
    assert not not_exist, f'{not_exist} file was not created by VCF2SFS'


def test_range2prior(upper="10,1,100", lower="1,0,2.5", repeats=10):
    priors = Class.Range2UniformPrior(upper=upper, lower=lower, repeats=repeats)
    assert 10 == priors.shape[0], 'The priors row numbers do not match with repeats. check Class.Range2UniformPrior'
    assert 3 == priors.shape[1], 'The priors column numbers do not match with upper parameters length. ' \
                                 'Class.Range2UniformPrior('
    sort_p = priors.sort_values(by=['param_1', 'param_2', 'param_3'])
    assert (sort_p.iloc[0] == [1.0, 0.0,
                               2.5]).all(), "The minimum do not match with lower limit. Class.Range2UniformPrior"
    assert (sort_p.iloc[-1] == [10.0, 1, 100.0]).all(), "The maximum do not match with upper limit. " \
                                                        "Class.Range2UniformPrior"


def test_MsPrime2SFS(demography='OOA', params_file='Priors.csv', samples='5,5,5', total_length=1e7):
    # noinspection PyUnresolvedReferences
    from src.SFS import Demography
    demography = eval('Demography.' + demography)
    priors = Class.Range2UniformPrior(upper="25e3, 2e5, 2e5, 2e5,1e4, 1e4, 1e4,80, 320, 700,50,50,50,50",
                                      lower="5000, 10000, 10000, 10000, 500,500,500, 15, 5, 5,0,0,0,0", repeats=10)
    priors.to_csv('Priors.csv', index=False)
    prisfs = Class.MsPrime2SFS.wrapper(sim_func=demography, params_file=params_file, samples=samples,
                                       total_length=total_length)
    assert 10 == prisfs.shape[0], "The sfs output do not have same rows as in the prior"
    assert 1345 == prisfs.shape[1], "The sfs do not have expected number of columns (1331+priors)"
    Misc.removefiles(['Priors.csv', 'test_out.csv'])
