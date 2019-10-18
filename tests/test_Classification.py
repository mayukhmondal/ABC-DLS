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
def test_Classification_Pre_train():
    info='Model.info'
    test_size=1
    chunksize=5
    scale=False
    ABC.ABC_TFK_Classification_PreTrain(info=info, test_size=test_size, chunksize=chunksize,scale=scale)

    # file existen check
    files=['x.h5','y.h5','y_cat_dict.txt']
    not_exist=[file for file in files if not Path(file).exists()]
    assert not not_exist, f'{not_exist} file was not created by ABC_TFK_Classification_PreTrain'

    # model name check
    files, paramnumbers, names = ABC.ABC_TFK_Classification_PreTrain.read_info('Model.info')
    y_cat_names = list(eval(open('y_cat_dict.txt', 'r').read()).values())
    assert Counter(y_cat_names) == Counter(names), 'Model.info model names do not match with y_cat_dict.txt'

    # h5 file checks

    xshape=HDF5Matrix('x.h5', 'mydata').shape
    yshape = HDF5Matrix('y.h5', 'mydata').shape
    # row check
    assert xshape[0]==yshape[0], 'Row numbers between x.h5 and/or y.h5 file do not match'
    assert xshape[0]==15, 'Row numbers of h5 files do not match with expected number of rows'
    # ycol check
    assert yshape[1]==len(names), 'Number of columns in y.h5 do not match with the number of names present in ' \
                                  'Model.info'
    #ncol check
    expectedcol=pandas.read_csv(files[0]).shape[1]-paramnumbers[0]
    assert xshape[1]==expectedcol, 'Column number of x.h5 do not match with expected'

    # checking y.h5 file