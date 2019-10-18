#!/usr/bin/python

"""
This file to test the Classification part of the ABC-TFK file
"""
import sys
sys.path.append("..")

from pathlib import Path
from src.Classes import ABC
from collections import Counter
from keras.utils import HDF5Matrix
import pandas
import numpy


def test_Classification_Pre_train():
    info='Model.info'
    test_size=1
    chunksize=5
    scale=True

    #
    ABC.ABC_TFK_Classification_PreTrain(info=info, test_size=test_size, chunksize=chunksize,scale=scale)

    # file existen check
    files=['x.h5','y.h5','y_cat_dict.txt','scale_x.sav']
    not_exist=[file for file in files if not Path(file).exists()]
    assert not not_exist, f'{not_exist} file was not created by ABC_TFK_Classification_PreTrain'

    # model name check
    files, paramnumbers, names = ABC.ABC_TFK_Classification_PreTrain.read_info('Model.info')
    y_cat_names = list(eval(open('y_cat_dict.txt', 'r').read()).values())
    assert Counter(y_cat_names) == Counter(['OOA', 'OOA_B', 'OOA_M']), 'Model.info model names do not match with y_cat_dict.txt'

    # h5 file checks

    xshape= HDF5Matrix('x.h5', 'mydata').shape
    yshape = HDF5Matrix('y.h5', 'mydata').shape
    # row check
    assert xshape[0]==yshape[0], 'Row numbers between x.h5 and/or y.h5 file do not match'
    assert xshape[0]==15, 'Row numbers of h5 files do not match with expected number of rows'
    # ycol check
    assert yshape[1]==3, 'Number of columns in y.h5 do not match with the number of names present in ' \
                                  'Model.info'
    #ncol check
    assert xshape[1]==1331, 'Column number of x.h5 do not match with expected'
    # checking y.h5 file
    # y.h5 either 0 and 1
    y_unique_val=list(numpy.unique(pandas.DataFrame(HDF5Matrix('y.h5', 'mydata')[:]).values.flatten()))
    assert Counter(y_unique_val)==Counter([1,0.0]), 'y values can either be 0 or 1'

    # the number of count  of models
    y = HDF5Matrix('y.h5', 'mydata')
    y_cat_dict = eval(open('y_cat_dict.txt', 'r').read())
    modelcount=pandas.DataFrame(numpy.argmax(y[:], axis=1, out=None))[0].replace(y_cat_dict).value_counts().to_dict()
    assert modelcount=={'OOA':5,'OOA_M':5,'OOA_B':5}, 'y.h5 have a different count of names than it should'

    #checking x.h5 file'
    x = HDF5Matrix('x.h5', 'mydata')
    xdf=pandas.DataFrame(x[:])
    assert all (xdf[i].dtype.kind == 'f' for i in xdf), 'Not all the columns are float for x.h5'
