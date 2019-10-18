#!/usr/bin/python

"""
This file to test the Classification part of the ABC-TFK file
"""
from ..src.Classes import ABC


def test_Classification_Pre_train():
    info='Model.info'
    test_size=1
    chunksize=5
    scale=False
    ABC.ABC_TFK_Classification_PreTrain(info=info, test_size=test_size, chunksize=chunksize,scale=scale)