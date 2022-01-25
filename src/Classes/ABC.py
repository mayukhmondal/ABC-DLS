#!/usr/bin/python
"""
This file will hold all the classes for ABC
"""
import os
import subprocess
import sys
# to stop future warning every time to print out
import warnings
# type hint for readability
from typing import List, Dict, Tuple, Optional, Callable, Union

import h5py
import joblib
import numpy
import pandas
# rpy2 stuff
from rpy2 import robjects
from rpy2.robjects import pandas2ri
# Tensor flow stuff
from sklearn import preprocessing

# my stuff
from . import Misc

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    # tensorflow stuff
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, GaussianNoise, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.callbacks import ReduceLROnPlateau
    from tensorflow.keras.utils import HDF5Matrix
# activating R
abc = Misc.importr_tryhard('abc')
pandas2ri.activate()


class ABC_DLS_Classification:
    """
    Main classification class. It will distinguish between different models. with given underlying models it will
    compare with real data and will predict how much it sure about which model can bet predict the real data.

    :param info: the path of info file whose file column is the path of the file and second column defining the
        number of  parameters
    :param ssfile: the summary statistic on real data set. should be csv format
    :param nn: custom function made for keras model. the path of that .py file. Should have a def
        ANNModelCheck
    :param method: to tell which method is used in abc. default is mnlogistic. but can be rejection, neural net etc.
        as documented in the r.abc
    :param tolerance: the level of tolerance for abc. default is .005
    :param test_size:  the number of test rows. everything else will be used for train. 10k is default
    :param chunksize:  the number of rows accessed at a time.
    :param scale: to tell if the data should be scaled or not. default is false. will be scaled by MinMaxscaler.The
        scaling will only happen on the ss.
    :param together: in case you want to send both train and test together (for validation data set). important if
        you do not want to lose data for earlystop validation split. look at extras/Dynamic.py to see how the tfknn
        should look like. Should not be used for big validation data set. Takes too much memory.
    :param csvout:  in case of everything satisfied. this will output the test data set in csv format. can be used
        later by r
    :param cvrepeats: the number of repeats will be used for CV calculations
    :param folder: to define the output folder. default is '' meaning current folder
    :param frac: To multiply all the observed ss with some fraction. Important in case simulated data and observed
        data are not from same length. default is 1
    :return: will not return anything but will plot and print the power
    """

    def __new__(cls, info: str, ssfile: str, nn: Optional[str] = None, method: str = "mnlogistic",
                tolerance: float = .001, test_size: int = int(1e4), chunksize: Optional[int] = int(1e4),
                scale: bool = False, csvout: bool = False, cvrepeats: int = 100, together: bool = False,
                folder: str = '', frac: float = 1.0) -> None:
        """
        This will automatically call the wrapper function and to do the necessary work.

        :param info: the path of info file whose file column is the path of the file and second column defining the
            number of  parameters
        :param ssfile: the summary statistic on real data set. should be csv format
        :param nn: custom function made for keras model. the path of that .py file. Should have a def
            ANNModelCheck
        :param method: to tell which method is used in abc. default is mnlogistic. but can be rejection, neural net etc.
            as documented in the r.abc
        :param tolerance: the level of tolerance for abc. default is .005
        :param test_size:  the number of test rows. everything else will be used for train. 10k is default
        :param chunksize:  the number of rows accessed at a time.
        :param scale: to tell if the data should be scaled or not. default is false. will be scaled by MinMaxscaler.The
            scaling will only happen on the ss.
        :param together: in case you want to send both train and test together (for validation data set). important if
            you do not want to lose data for earlystop validation split. look at extras/Dynamic.py to see how the tfknn
            should look like. Should not be used for big validation data set. Takes too much memory.
        :param csvout:  in case of everything satisfied. this will output the test data set in csv format. can be used
            later by r
        :param cvrepeats: the number of repeats will be used for CV calculations
        :param folder: to define the output folder. default is '' meaning current folder
        :param frac: To multiply all the observed ss with some fraction. Important in case simulated data and observed
            data are not from same length. default is 1
        :return: will not return anything but will plot and print the power
        """
        return cls.wrapper(info=info, ssfile=ssfile, nn=nn, method=method, tolerance=tolerance,
                           test_size=test_size, chunksize=chunksize, scale=scale, together=together, csvout=csvout,
                           cvrepeats=cvrepeats, folder=folder, frac=frac)

    @classmethod
    def wrapper(cls, info: str, ssfile: str, nn: Optional[str] = None, method: str = "mnlogistic",
                tolerance: float = .005, test_size: int = int(1e4), chunksize: Optional[int] = None,
                scale: bool = False,
                together: bool = False, csvout: bool = False, cvrepeats: int = 100, folder: str = '',
                frac: float = 1.0) -> None:
        """
        the total wrapper of the classification method. with given underlying models it will compare with real data and
        will predict how much it sure about which model can bet predict the real data.
        wrapper_pre_train(Misc.removefiles -> cls.read_info -> Misc.getting_line_count ->
        cls.subsetting_file_concating-> cls.shufling_joined_models -> if chunksize :  cls.preparingdata_hdf5;
        else: cls.data_prep4ANN) -> wrapper_train Misc.loading_def_4m_file -> def ANNModelCheck )
        wrapper_after_train(ModelSeparation.evaluate -> cls.read_ss_2_series -> cls.plot_power_of_ss (cls.r_summary) ->
        cls.model_selection (cls.r_summary)-> cls.gfit_all (cls.r_summary) -> cls.csvout)

        :param info: the path of info file whose file column is the path of the file and second column defining the
            number of  parameters
        :param ssfile: the summary statistic on real data set. should be csv format
        :param nn: custom function made for keras model. the path of that .py file. should have a def
            ANNModelCheck
        :param method: to tell which method is used in abc. default is mnlogistic. but can be rejection, neural net etc.
            as documented in the r.abc
        :param tolerance: the level of tolerance for abc. default is .005
        :param test_size:  the number of test rows. everything else will be used for train. 10k is default
        :param chunksize:  the number of rows accessed at a time.
        :param scale: to tell if the data should be scaled or not. default is false. will be scaled by MinMaxscaler.
            The scaling will only happen on the ss.
        :param together: in case you want to send both train and test together (for validation data set). important if
            you do not want to lose data for earlystop validation split. look at extras/Dynamic.py to see how the tfknn
            should look like. Should not be used for big validation data set. Takes too much memory.
        :param csvout:  in case of everything satisfied. this will output the test dataset in csv format. can be used
            later by r
        :param cvrepeats: the number of repeats will be used for CV calculations
        :param folder: to define the output folder. default is '' meaning current folder
        :param frac: To multiply all the observed ss with some fraction. Important in case simulated data and observed
            data are not from same length. default is 1
        :return: will not return anything but will plot and print the power
        """
        folder = Misc.creatingfolders(folder)
        x_train, x_test, y_train, y_test, scale_x, y_cat_dict = cls.wrapper_pre_train(info=info, test_size=test_size,
                                                                                      chunksize=chunksize, scale=scale,
                                                                                      folder=folder)
        if together:
            ModelSeparation = cls.wrapper_train(x_train=(x_train, x_test), y_train=(y_train, y_test),
                                                nn=nn, folder=folder)
        else:
            ModelSeparation = cls.wrapper_train(x_train=x_train, y_train=y_train, nn=nn, folder=folder)

        cls.wrapper_after_train(ModelSeparation=ModelSeparation, x_test=x_test, y_test=y_test, scale_x=scale_x,
                                y_cat_dict=y_cat_dict, ssfile=ssfile, method=method, tolerance=tolerance,
                                csvout=csvout, cvrepeats=cvrepeats, folder=folder, frac=frac)

    @classmethod
    def wrapper_pre_train(cls, info: str, test_size: int = int(1e4), chunksize: Optional[int] = int(1e4),
                          scale: bool = False, folder: Optional[str] = None) -> \
            Tuple[
                Union[numpy.ndarray, HDF5Matrix], Union[numpy.ndarray, HDF5Matrix], Union[numpy.ndarray, HDF5Matrix],
                Union[
                    numpy.ndarray, HDF5Matrix], Optional[preprocessing.MinMaxScaler], Dict[int, str]]:
        """
        This the wrapper for pre_training part of the classification. it will produce data in hdf5 format which then
        easily can be used in training part of the classification. it will also delete all the files that can be output
        by the classification. so that it will work on a clean sheet.
        Misc.removefiles -> cls.read_info -> Misc.getting_line_count ->  cls.subsetting_file_concating->
        cls.shufling_joined_models -> if chunksize :  cls.preparingdata_hdf5; else: cls.data_prep4ANN

        :param info: the path of info file whose file column is the path of the file and second column defining the
            number of  parameters
        :param test_size: the number of test rows. everything else will be used for train. 10k is default
        :param chunksize:  the number of rows accessed at a time.
        :param scale: to tell if the data should be scaled or not. default is false. will be scaled by MinMaxscaler.
            The scaling will only happen on the ss.
        :param folder: to define the output folder. default is none meaning current folder
        :return: will return data needed for training. will return x_(train,test), y_(train,test) scale_x (MinMaxScaler
            or None) and y_cat_dict ({0:'model1',1:'model2'..})
        """
        outfolder = Misc.creatingfolders(folder)
        previousfiles = ('scale_x.sav', 'scale_y.sav', 'x_test.h5', 'y_test.h5', 'y.h5', 'x.h5',
                         'ModelClassification.h5',
                         'Comparison.csv', 'shuf.csv', 'models.csv', 'ss.csv', 'y_cat_dict.txt', 'model_index.csv.gz',
                         'params.csv.gz', 'ss_predicted.csv.gz', 'ss_target.csv.gz', 'NN.pdf', 'CV.pdf',
                         'Checkpoint.h5', 'nnparamcv.pdf', 'nnparamcv_together.pdf', 'paramposterior.pdf',
                         'paramposterior_together.pdf')
        previousfilesfullpath = tuple(outfolder + file for file in previousfiles)
        Misc.removefiles(previousfilesfullpath)
        files, paramnumbers, names = cls.read_info(info=info)
        minlines = min([Misc.getting_line_count(file) for file in files]) - 1
        # header creation
        pandas.DataFrame(
            ['models'] + list(pandas.read_csv(files[0], nrows=1).columns[paramnumbers[0]:])).transpose().to_csv(
            outfolder +
            'Comparison.csv', index=False, header=False)
        # adding line after sub-setting
        [cls.subsetting_file_concating(filename=files[i], params_number=paramnumbers[i], nrows=minlines,
                                       modelname=names[i], outfolder=outfolder) for i in range(len(files))]
        shuffile = cls.shufling_joined_models(inputcsv='Comparison.csv', output='shuf.csv',
                                              outfolder=outfolder)

        if chunksize:
            x_train, x_test, y_train, y_test, scale_x, y_cat_dict = cls.preparingdata_hdf5(filename=shuffile,
                                                                                           chunksize=chunksize,
                                                                                           test_size=test_size,
                                                                                           scale=scale,
                                                                                           outfolder=outfolder)
            f = open(outfolder + "y_cat_dict.txt", "w")
            f.write(str(y_cat_dict))
            f.close()

        else:
            results = pandas.read_csv(shuffile, index_col=0)
            x_train, x_test, y_train, y_test, scale_x, y_cat_dict = cls.data_prep4ANN(results, test_size=test_size,
                                                                                      scale=scale, outfolder=outfolder)
        Misc.removefiles([outfolder + 'Comparison.csv', outfolder + 'shuf.csv'])

        return x_train, x_test, y_train, y_test, scale_x, y_cat_dict

    @classmethod
    def read_info(cls, info: str) -> Tuple[list, list, list]:
        """
        reading the info file. whose file column is the path of the file and second column defining the number of
        parameters

        :param info:  the path of info file whose file column is the path of the file and second column defining the
            number of  parameters
        :return: will return the the files path, number of parameters and names of the model all in three lists
        """
        infodata = Misc.reading_bylines_small(info)
        infodata = [info for info in infodata if info is not '']
        files = [line.split()[0] for line in infodata]
        paramnumbers = [int(line.split()[1]) for line in infodata]
        names = [Misc.filenamewithoutextension_checking_zipped(file) for file in files]
        tobeprint, absentfiles = Misc.file_existence_checker(names, files)
        if len(absentfiles) > 0:
            print(tobeprint)
            sys.exit(1)
        return files, paramnumbers, names

    @classmethod
    def read_ss_2_series(cls, file: str) -> pandas.Series:
        """
        To read summary statistics of real data. the file format should be in csv with one row. all the  ss should
        separated by comma (as with csv format). can be zipped. It will assume no header if only one line is present in
        the file. If two lines it will assume first line is header. If three lines it will assume it is moments or dadi
        related fs format.

        :param file: the path of the sfs file
        :return: will return the sfs in series format
        """
        ss = pandas.read_csv(file)
        return ss

    @classmethod
    def check_results(cls, results: List[pandas.DataFrame], observed: pandas.Series) -> None:
        """
        To check every thing is ok with the  results and observed values. for example if two files of results have
        different rows it will take the lower minima. as you need all the model with same rows of repetition

        :param results: the results list with different model result in dataframe format
        :param observed: the observed results
        :return: will not return anything. but in case of problem will stop
        """
        result_columns = [result.shape[1] for result in results]
        if observed.ndim > 1:
            if len(set(result_columns + [observed.shape[1]])) > 1:
                print("the observed columns and/or result columns do no match. check")
                print("result_columns:", result_columns)
                print("observed_columns", observed.shape[0])
                sys.exit(1)
        else:
            if len(set(result_columns + [observed.shape[0]])) > 1:
                print("the observed columns and/or result columns do no match. check")
                print("result_columns:", result_columns)
                print("observed_columns", observed.shape[0])
                sys.exit(1)

    @classmethod
    def subsetting_file_concating(cls, filename: str, params_number: int, nrows: int, modelname: str,
                                  outfolder: str = '') -> None:
        """
        to read the msprime sfs out files. which has both params and sfs together. and then prepare for merging.
        this will remove the params columns (As not needed for model comparison) and make same number of rows per model
        so that it is not biased. also add the very first column as the name of the model

        :param filename: the csv file path. whose first columns are parameters and all the rest are sfs or ss. comma
            separated
        :param params_number: the number of parameters present in the file. rest are sfs or ss
        :param nrows: the number of rows to include in the file. remember the nrows start with 0. thus if you want to
            include 100 line put 99. the first line is header thus ignored
        :param modelname: name of the model. string
        :param outfolder: to define the output folder. default is current folder
        :return: will not return anything but will create or update Comparison.csv file .where all the ss with the name
            of the models are written
        """
        if filename[-2:] == 'gz':
            command = Misc.joinginglistbyspecificstring(
                ["zcat", filename, '|', 'cut -f' + str(params_number + 1) + '-', '-d ","',
                 '''|awk '$0="''' + modelname + ""","$0'""", "|tail -n+2", "|head -n", nrows, '|grep -v ",,"', ">>",
                 outfolder + "Comparison.csv"])
        else:
            command = Misc.joinginglistbyspecificstring(
                ["cat", filename, '|', 'cut -f' + str(params_number + 1) + '-', '-d ","',
                 '''|awk '$0="''' + modelname + ""","$0'""", "|tail -n+2", "|head -n", nrows, '|grep -v ",,"', ">>",
                 outfolder + "Comparison.csv"])
        p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        if stderr:
            print(stderr)
            sys.exit(1)
        # checking for nan in csv files
        if filename[-2:] == 'gz':
            nancheckcommand = Misc.joinginglistbyspecificstring(['zcat ', filename, '| grep ",,"|wc -l '])
        else:
            nancheckcommand = Misc.joinginglistbyspecificstring(['cat ', filename, '| grep ",,"|wc -l '])
        p = subprocess.Popen(nancheckcommand, stdout=subprocess.PIPE, shell=True, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        if stderr:
            print(stderr)
            sys.exit(1)
        if int(stdout) > 0:
            print(filename, "has nan elements in ", int(stdout),
                  "rows. They are automatically removed. ", "Small number of removed is ok.",
                  "But if they are huge, they can create problem in the downstream as the number of models are not "
                  "equal")

        return None

    @classmethod
    def shufling_joined_models(cls, inputcsv: str = "Comparison.csv", output: str = 'shuffle.csv',
                               header: bool = True, outfolder: str = '') -> str:
        """
        it will shuffle the line of joined csv model (file Comparison.csv) and read it in pandas format for further
        evaluation

        :param inputcsv: the joined csv file (Comparison.csv) with ss and model names default:"Comparison.csv"
        :param output: the shuffled csv file path. default: 'shuffle.csv'
        :param header: if the header should be kept or not. default is true
        :param outfolder: to define the output folder. default is current folder
        :return: will return output which is the shuffled rows of input
        """
        import os
        import shutil
        terashuf = os.path.dirname(os.path.abspath(__file__)) + '/shuffle.py'
        parentfolder = os.getcwd()
        if outfolder != '':
            os.chdir(outfolder)
        Misc.creatingfolders('temp')
        if header:

            command = Misc.joinginglistbyspecificstring(['cat <(head -n 1', inputcsv, ') <(tail -n+2', inputcsv,
                                                         ' | python ' + terashuf + ' ) > ',
                                                         os.getcwd() + '/' + output]).strip()

        else:

            command = Misc.joinginglistbyspecificstring(['cat', inputcsv, '|', 'python ', terashuf, ">", output])

        p = subprocess.Popen([command], executable='/bin/bash', stdout=subprocess.PIPE, shell=True,
                             stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        if stderr:
            print(stderr)
            sys.exit(1)
        shutil.rmtree('temp')
        if outfolder != '':
            os.chdir(parentfolder)
        return outfolder + output

    @classmethod
    def data_prep4ANN(cls, raw: pandas.DataFrame, test_size: int = int(1e4), scale: bool = False,
                      outfolder: str = '') -> Tuple[
        numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, Optional[preprocessing.MinMaxScaler], Dict[
            int, str]]:
        """
        Preparing data for NN classification method. The MinMaxScaler is used to normalize in case normalization needed

        :param raw: raw summary statistics dataframe.
        :param test_size: the number of test rows. everything else will be used for train. 10k is default
        :param scale: if the raw data should be scaled or not. default is false. will be scaled by MinMaxscaler
        :param outfolder: to define the output folder. default is current folder
        :return: will return value which will be important to training ann. will return x_(train,test), y_(train,test)
            scale_x (MinMaxScaler or None) and y_cat_dict ({0:'model1',1:'model2'..})
        """
        from sklearn.model_selection import train_test_split

        raw = raw.sample(frac=1)
        y_cat_dict = dict(zip(pandas.Categorical(raw.index).codes, raw.index))
        y = keras.utils.to_categorical(pandas.Categorical(raw.index).codes, len(y_cat_dict))
        if scale:
            scale_x = preprocessing.MinMaxScaler()
            x = scale_x.fit_transform(raw.values)
        else:
            x = raw.values
            scale_x = None
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
        Misc.numpy2hdf5(x_test, outfolder + 'x_test.h5')
        Misc.numpy2hdf5(y_test, outfolder + 'y_test.h5')
        if scale_x:
            joblib.dump(scale_x, outfolder + "scale_x.sav")
        return x_train, x_test, y_train, y_test, scale_x, y_cat_dict

    @classmethod
    def preparingdata_hdf5(cls, filename: str, chunksize: int, test_size: int = int(1e4), scale: bool = False,
                           outfolder: str = '') -> Tuple[
        HDF5Matrix, HDF5Matrix, HDF5Matrix, HDF5Matrix, Optional[preprocessing.MinMaxScaler], Dict[int, str]]:
        """
        In case of chunk size is mentioned it will be assumed that the data is too big to save in ram and it will be
        saved in hdf5 format and will be split it in necessary steps

        :param filename: the file path of csv where the first column is the models name and rest are ss. every row
            different simulation. header included and shuffled data. output of shufling_joined_models
        :param chunksize: the number of rows accessed at a time.
        :param test_size: the number of test rows. everything else will be used for train. 10k is default
        :param scale: to tell if needed to be scaled or not. by default is false. if true will also save scale_x.sav to
            be used later if needed
        :param outfolder: to define the output folder. default is current folder
        :return: will return train and test data fro both x and y
        """

        xfile = outfolder + "x.h5"
        yfile = outfolder + "y.h5"
        Misc.removefiles([xfile, yfile])
        ss_command = Misc.joinginglistbyspecificstring(["cut -f 2- ", '-d ","', filename, ">", outfolder + "ss.csv"])
        os.system(ss_command)
        if scale:
            scale_x = cls.MinMax4bigfile(csvpath=outfolder + 'ss.csv', h5path=xfile, chunksize=chunksize)
        else:
            scale_x = cls.MinMax4bigfile(csvpath=outfolder + 'ss.csv', h5path=xfile,
                                         chunksize=chunksize, scaling=False)
        x_train, x_test = cls.train_test_split_hdf5(xfile, test_rows=int(test_size))

        if scale_x:
            joblib.dump(scale_x, outfolder + "scale_x.sav")
        models_command = Misc.joinginglistbyspecificstring(
            ["cut -f 1 ", '-d ","', filename, ">", outfolder + "models.csv"])
        os.system(models_command)
        model_index = pandas.read_csv(outfolder + 'models.csv')
        y_cat_dict = dict(zip(pandas.Categorical(model_index.iloc[:, 0]).codes, model_index.iloc[:, 0]))
        y = keras.utils.to_categorical(pandas.Categorical(model_index.iloc[:, 0]).codes, len(y_cat_dict), 'uint8')
        Misc.numpy2hdf5(y, yfile)
        y_train, y_test = cls.train_test_split_hdf5(yfile, test_rows=int(test_size))
        Misc.removefiles([outfolder + 'ss.csv', outfolder + 'models.csv'])
        return x_train, x_test, y_train, y_test, scale_x, y_cat_dict

    @classmethod
    def MinMax4bigfile(cls, csvpath: str, h5path: str = 'temp.h5', scaling: bool = True,
                       expectedrows: Optional[int] = None, chunksize: int = 10, verbose: bool = False,
                       special_func: Optional[Callable] = None, **kwargs) -> Optional[preprocessing.MinMaxScaler]:
        """
        This to to convert a very big csv file which cannot be put inside ram to save it in the h5 format and use the
        minmax scaler to scale the data

        :param csvpath: the path of ss csv file. can be zipped
        :param h5path: output of h5path. default is temp.h5
        :param scaling: to tell if the data should be scaled or not. default is false. will be scaled by MinMaxscaler
        :param expectedrows: the number of rows is expected for the data output. if not given it will calculate from
            the csv file
        :param chunksize: the number of rows can be accessed at a time. if memory error put lower number. default is 10
        :param verbose: in case to check the progression put verbose true. default false
        :param special_func: in case some kind of special function has to be done before the min max scaler
        :param kwargs: all the related parameters for special_func
        :return: will return the min max scaler which can be used later. or none. it will save h5path the whole data
        """
        # the scaling part
        newss = numpy.array([])
        chunk = pandas.DataFrame()
        if scaling:
            scale = preprocessing.MinMaxScaler()
            row = 0
            if verbose:
                print('scaling')
            for chunk in pandas.read_csv(csvpath, chunksize=chunksize):
                if verbose:
                    print(row)
                row = row + chunksize
                if special_func:
                    if verbose:
                        print(special_func(chunk, **kwargs).values)
                    newss = special_func(chunk, **kwargs).values
                    if len(newss.shape) == 1:
                        newss = newss.reshape(-1, 1)
                    scale.partial_fit(newss)
                else:
                    scale.partial_fit(chunk.values)
        else:
            scale = None
            chunk = pandas.read_csv(csvpath, nrows=chunksize)
            if special_func:
                if verbose:
                    print(special_func(chunk, **kwargs).values)
                newss = special_func(chunk, **kwargs).values

        # initializing the hdf5 part
        if expectedrows is None:
            expectedrows = Misc.getting_line_count(csvpath) - 1
        if special_func:
            expectedcolumns = newss.shape[1]
        else:
            expectedcolumns = chunk.shape[1]

        arraysize = (expectedrows, expectedcolumns)

        f = h5py.File(h5path, 'w')
        transh5 = f.create_dataset('mydata', arraysize, chunks=True)

        # transforming
        if verbose:
            print('transforming')
        row = 0
        for chunk in pandas.read_csv(csvpath, chunksize=chunksize):
            if verbose:
                print(row)
            if scaling:
                if special_func:
                    newss = special_func(chunk, **kwargs).values
                    if len(newss.shape) == 1:
                        newss = newss.reshape(-1, 1)
                    transformed = scale.transform(newss)

                else:
                    transformed = scale.transform(chunk.values)
            else:
                if special_func:
                    transformed = special_func(chunk, **kwargs).values
                else:
                    transformed = chunk.values
            transh5[row:row + chunksize] = transformed
            row = row + chunksize
        f.close()

        return scale

    @classmethod
    def train_test_split_hdf5(cls, file: str, dataset: str = 'mydata', test_rows: Union[int, float] = int(1e4)) -> \
            Tuple[HDF5Matrix, HDF5Matrix]:
        """
        Special way to train test split for hdf5. will take the first n-test_rows for training and rest for test

        :param file: the path of .h5 file
        :param dataset: the name of the dataset of h5py file. default 'mydata'
        :param test_rows: the number of rows for test every thing will be left for training. default is 10k
        :return: will return the test, train split for hdf5
        """
        if isinstance(test_rows, float):
            test_rows = int(test_rows)
        if os.path.isfile(file):
            with h5py.File(file, 'r') as f:
                rows = f[dataset].shape[0]
                train = f[dataset][:rows - test_rows]
                test = f[dataset][rows - test_rows: rows]
                return train, test
        else:
            print('Could not find file ', file)
            sys.exit(1)

    @classmethod
    def wrapper_train(cls, x_train: Union[
        numpy.ndarray, HDF5Matrix, Tuple[Union[numpy.ndarray, HDF5Matrix], Union[numpy.ndarray, HDF5Matrix]]],
                      y_train: Union[numpy.ndarray, HDF5Matrix, Tuple[
                          Union[numpy.ndarray, HDF5Matrix], Union[numpy.ndarray, HDF5Matrix]]],
                      nn: Optional[str] = None, folder: str = '') -> keras.models.Model:
        """
        This the wrapper for training part of the classification method. it need training data set for x and y. can be
        either numpy array or hdf5 matrix format (HD5matrix) of keras
        Misc.loading_def_4m_file -> def ANNModelCheck

        :param x_train: train part of x aka summary statistics
        :param y_train: training part of y aka models names. should be used keras.utils.to_categorical to better result
        :param nn: custom function made for keras model. the path of that .py file. should have a def
            ANNModelCheck
        :param folder: to define the output folder. default is '' meaning current folder
        :return: will return the keras model. it will also save the model in ModelClassification.h5
        """
        folder = Misc.creatingfolders(folder)

        Misc.removefiles((folder + "ModelClassification.h5", folder + "Checkpoint.h5"))
        if nn:
            ANNModelCheck = Misc.loading_def_4m_file(filepath=nn, defname='ANNModelCheck')
            if ANNModelCheck is None:
                print('Could not find the ANNModelCheck in', nn,
                      '. Please check. Now using the default ANNModelCheck')
                ANNModelCheck = cls.ANNModelCheck
        else:
            ANNModelCheck = cls.ANNModelCheck
        # needed as Checkpoint.h5 should be inside the folder and i do not want to make ANNModelCheck complicated with
        # another variable 'folder'
        parentfolder = os.getcwd()
        if folder != '':
            os.chdir(folder)
        ModelSeparation = ANNModelCheck(x=x_train, y=y_train)
        cls.check_save_tfk_model(model=ModelSeparation, output="ModelClassification.h5",
                                 check_point='Checkpoint.h5')
        # same as above to change back to previous stage
        if folder != '':
            os.chdir(parentfolder)
        return ModelSeparation

    @classmethod
    def Gaussian_noise(cls, input_layer, sd: float = .01):
        """
        Gaussian noise to the input data. Same as Keras.GaussianNoise but it will not only work with training part but
        will work on test data set and observed data. Thus every time it will run will give slightly different results.
        Good to produce a distribution from a single observation. This is not used anymore but i still kept it in case
        needed use model.add(Lambda(cls.Gaussian_noise, input_shape=(x.shape[1],)))

        :param input_layer: tensorflow input layer
        :param sd: the standard deviation present will be present in the noise random normal distribution
        :return: will add the noise to the input_layer
        """
        import tensorflow as tf
        noise = tf.random.normal(shape=tf.shape(input_layer), mean=0.0, stddev=sd, dtype=tf.float32)
        return input_layer + noise

    @classmethod
    def ANNModelCheck(cls, x: Union[numpy.ndarray, HDF5Matrix],
                      y: Union[numpy.ndarray, HDF5Matrix]) -> keras.models.Model:
        """
        The Tensor flow for model check

        :param x: the x or summary statistics. can be numpy array or hdf5.
        :param y: the y or model names or classification. can be numpy array of hdf5
        :return: will return the trained model
        """
        model = Sequential()
        # model.add(Lambda(cls.Gaussian_noise, input_shape=(x.shape[1],)))
        model.add(GaussianNoise(0.01, input_shape=(x.shape[1],)))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(.01))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(.01))
        model.add(Dense(y.shape[1], activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['categorical_accuracy'])

        # adding an early stop so that it does not overfit
        ES = EarlyStopping(monitor='val_loss', patience=100)
        # checkpoint
        CP = ModelCheckpoint('Checkpoint.h5', verbose=1, save_best_only=True)
        # Reduce learning rate
        RL = ReduceLROnPlateau(factor=0.5)

        model.fit(x, y, epochs=int(2e6), verbose=2, shuffle=True, callbacks=[ES, CP, RL], validation_split=.2)

        return model

    @classmethod
    def check_save_tfk_model(cls, model: keras.models.Model, output: str = 'Model.h5',
                             check_point: str = 'Checkpoint.h5') -> None:
        """
        This will save the keras model as a h5 file. It will also check if check_point (default=Checkpoint.h5)  was
        created before. In case check point was created before it will rename that as output. In case no checkpoint it
        will save the model. Needed in case nn (model train) do not create checkpoint.

        :param model: trained model by keras tf
        :param output: output file name for h5. default Model.h5
        :param check_point: check point file name. default Checkpoint.h5
        :return: wil not return anything but will save the model
        """
        if os.path.isfile(check_point):
            os.rename(check_point, output)
        else:
            model.save(output)

    @classmethod
    def wrapper_after_train(cls, ModelSeparation: keras.models.Model, x_test: Union[numpy.ndarray, HDF5Matrix],
                            y_test: Union[numpy.ndarray, HDF5Matrix], scale_x: Optional[preprocessing.MinMaxScaler],
                            y_cat_dict: Dict[int, str], ssfile: str, method: str = "mnlogistic",
                            tolerance: float = .005, csvout: bool = False, cvrepeats: int = 100,
                            folder: str = '', pred_repeat: int = 1, frac: float = 1.0) -> None:
        """
        This the wrapper for after training part of the classification. after training is done it will test on the test
        data set to see the power and then use a real data set to show how likely it support one model over another.
        it will use abc to give the power or standard deviation of the model that is predicted to know how much we are
        sure about the results. mainly it will do three parts of abc. one cv error , model selection and goodness of fit
        ModelSeparation.evaluate -> cls.read_ss_2_series -> cls.plot_power_of_ss (cls.r_summary) -> cls.model_selection
        (cls.r_summary)-> cls.gfit_all (cls.r_summary) -> cls.csvout

        :param ModelSeparation: The fitted keras model
        :param x_test: the test part of x aka summary statistics
        :param y_test: the test part of y aka models name. should be used keras.utils.to_categorical to better result
        :param scale_x: the MinMax scaler of x axis. can be None
        :param y_cat_dict: name of all the models. will be printed the pdf
        :param ssfile: the summary statistic on real data set. should be csv format
        :param method: to tell which method is to be used in abc. default is mnlogistic. but can be rejection, neural net
            etc. as documented in the r.abc
        :param tolerance: the level of tolerance. default is .005
        :param csvout: in case of everything satisfied. this will output the test data set in csv format. can be used
            later by
        :param cvrepeats: the number of repeats will be used for CV calculations
        :param folder: to define the output/input folder. default is '' meaning current folder
        :param pred_repeat: in case you want to run multiple run of prediction on observed data. if you use it your
            training must need some randomization like the custom Gaussiannoise i have implemented here. The idea is by
            using random noise you are producing multiple run of the same observed data
        :param frac: To multiply all the observed ss with some fraction. Important in case simulated data and observed
            data are not from same length. default is 1
        :return: will not return anything but will produce the graphs and print out how much it is sure about any model
        """
        sfs = cls.read_ss_2_series(file=ssfile)
        sfs = sfs * numpy.array(frac)
        cls.check_results(results=[x_test[0:2]], observed=sfs)
        print("Evaluate with test:")
        ModelSeparation.evaluate(x_test, y_test, verbose=2)
        # abc and plot by r
        ssnn, predictednn = cls.prepare4ABC(ModelSeparation=ModelSeparation, sfs=sfs, x_test=x_test, y_test=y_test,
                                            scale_x=scale_x, y_cat_dict=y_cat_dict, pred_repeat=pred_repeat)
        print("Number of Samples per model for test:")
        model_counts = cls.count_samples(indexes=ssnn.index, y_cat_dict=y_cat_dict)
        print(model_counts.to_string())
        print('Predicted by NN')
        print(predictednn.rename(columns=y_cat_dict).to_string())
        if model_counts.iloc[0].min() < cvrepeats:
            print('Cv repeats cannot be more than the number of samples present for a particular model. Please use '
                  'lesser number of cv repeats or more number of samples per model.')
            print(model_counts.T)
            sys.exit(1)
        robjects.r['pdf'](folder + "NN.pdf")
        cls.plot_power_of_ss(ss=ssnn, index=ssnn.index, tol=tolerance, method=method, repeats=cvrepeats)
        for index, row in predictednn.iterrows():
            ln = index + 1
            print("# SS_Line", ln)
            cls.model_selection(target=row, index=ssnn.index, ss=ssnn, method=method, tol=tolerance)
            cls.gfit_all(observed=row, ss=ssnn, y_cat_dict=y_cat_dict, extra='_nn_: SS_Line ' + str(ln), tol=tolerance,
                         repeats=cvrepeats)
        robjects.r['dev.off']()
        if csvout:
            cls.outputing_csv(modelindex=ssnn.index,
                              ss_predictions=pandas.DataFrame(ModelSeparation.predict(x_test[:])).rename(
                                  columns=y_cat_dict),
                              predict4mreal=predictednn.rename(columns=y_cat_dict), folder=folder)

    @classmethod
    def prepare4ABC(cls, ModelSeparation: keras.models.Model, sfs: pandas.Series,
                    x_test: Union[numpy.ndarray, HDF5Matrix], y_test: Union[numpy.ndarray, HDF5Matrix],
                    scale_x: Optional[preprocessing.MinMaxScaler], y_cat_dict: Dict[int, str],
                    pred_repeat: int = 1) -> [pandas.DataFrame, pandas.DataFrame]:
        """
        prepare data for ABC from NN predictions.
        :param ModelSeparation: The fitted keras model
        :param sfs: the sfs in series format
        :param x_test: the test part of x aka summary statistics
        :param y_test: the test part of y aka models name. should be used keras.utils.to_categorical to better result
        :param scale_x: the MinMax scaler of x axis. can be None
        :param y_cat_dict: name of all the models in a dict format
        :param pred_repeat: in case you want to run multiple run of prediction on observed data. if you use it your
            training must need some randomization like the custom Gaussian noise i have implemented here. The idea is by
            using random noise you are producing multiple run of the same observed data
        :return: will return nn predicted summary statistics ssnn and nn predicted on real data predictednn
        """
        if pred_repeat > 1:
            ssnn = cls.predict_repeats_mean(ModelSeparation, x_test, repeats=pred_repeat)
            if scale_x:
                predictednn = cls.predict_repeats_mean(ModelSeparation, scale_x.transform(sfs.values.reshape(1, -1)))
            else:
                predictednn = cls.predict_repeats_mean(ModelSeparation, sfs.values.reshape(1, -1))
        else:
            ssnn = pandas.DataFrame(ModelSeparation.predict(x_test[:]))
            if scale_x:
                if sfs.shape[0] > 1:
                    predictednn = pandas.DataFrame(ModelSeparation.predict(scale_x.transform(sfs.values)))
                else:
                    predictednn = pandas.DataFrame(
                        ModelSeparation.predict(scale_x.transform(sfs.values.reshape(1, -1))))
            else:
                if sfs.shape[0] > 1:
                    predictednn = pandas.DataFrame(ModelSeparation.predict(sfs.values))
                else:
                    predictednn = pandas.DataFrame(ModelSeparation.predict(sfs.values.reshape(1, -1)))
        indexnn = pandas.DataFrame(numpy.argmax(y_test, axis=1, out=None))[0].replace(y_cat_dict)
        ssnn.index = indexnn
        # prepare for R as it do not like very small numbers
        ssnn = ssnn.round(5)
        predictednn = predictednn.round(5)
        return ssnn, predictednn

    @classmethod
    def count_samples(cls, indexes: Union[pandas.Series, pandas.Index], y_cat_dict: Dict[int, str]) -> pandas.DataFrame:
        """
        counting the number of samples present in y_test data

        :param indexes: the index of the models
        :param y_cat_dict: name of all the models in a dict format
        :return: will return the counts per model in pandas dataframe format
        """
        counts = pandas.Series(0, index=y_cat_dict.values()).sort_index()
        counts.update(indexes.value_counts())
        return counts.to_frame().T

    @classmethod
    def predict_repeats_mean(cls, Model: keras.models.Model, x: Union[numpy.ndarray, HDF5Matrix],
                             repeats: int = 100) -> pandas.DataFrame:
        """
        Instead of predicting once on NNModel. It will predict multiple times [important to use
        Lambda(cls.Gaussian_noise, input_shape=(x.shape[1],)) on the starting layer] from same data and return mean on
        those repeats

        :param Model: the keras trained model
        :param x: x or summary statistics. can be both x_test or observed
        :param repeats: the number of repeats to be used on such prediction. default is 100
        :return: will return a pandas dataframe of predicted values
        """
        ssnn = [Model.predict(x[:]) for _ in range(repeats)]
        ssnn = numpy.mean(numpy.array(ssnn), axis=0)
        return pandas.DataFrame(ssnn)

    @classmethod
    def print_after_match_linestart(cls, file: str, match: str) -> None:
        """
        Print everything  after a match (which is starts a line) from a file

        :param file: the text file path
        :param match: the matching pattern which the starts the line
        :return: will not return anything but print the whole stuff (everything after the match)
        """
        with open(file, "r") as ifile:
            for line in ifile:
                if line.startswith(match):
                    print(line.strip())
                    break
            for line in ifile:
                print(line.strip())
        return None

    @classmethod
    def r_summary(cls, rmodel: robjects, target: str = 'Data:') -> None:
        """
        This is specifically to print the summary done on abc. Apparently there are some bug in r_python code. cant be
        solved. so it will print anything after 'Data:' or any string if done a summary on the abc models

        :param rmodel: the abc model
        :param target: the target after which every thing will be print. the target will be included in the print
        :return: will not return anything but print the summary everything after a line starts with 'Data:' or target
        """
        import tempfile
        temp_name = next(tempfile._get_candidate_names())
        robjects.r.options(width=10000)
        robjects.r['sink'](temp_name)
        robjects.r['summary'](rmodel)
        robjects.r['sink']()
        cls.print_after_match_linestart(temp_name, target)
        os.remove(temp_name)

    @classmethod
    def plot_power_of_ss(cls, ss: pandas.DataFrame, index: Union[pandas.Series, pandas.Index],
                         tol: float = .005, repeats: int = 100,
                         method: str = "mnlogistic") -> None:
        """
        now to test the power of summary statistics using r_abc

        :param ss: the summary statics in dataframe format
        :param index: the index of the models
        :param tol: the level of tolerance. default is .005
        :param repeats: the number of repeat to use is on cv (cross validation) error. default is 100
        :param method: to tell which method is used in abc. default is mnlogistic. but can be rejection, neural net etc.
            as documented in the r.abc
        :return: will plot it the folder to see the confusion matrix. also print out the summary of the model to see the
            confusion matrix in text format
        """
        if method != 'rejection':
            print(
                "Adding small noises if method rejection was not used. As 0 standard deviation in columns can create "
                "error in ABC_CV")
            noise = numpy.random.normal(0, 1e-4, ss.shape)
            ss = (ss + noise).clip(0)

        cvmodsel = abc.cv4postpr(index=index, sumstat=ss, nval=repeats, tol=tol, method=method)
        # text wrapping problem in r which cant be solved by options(width=10000) in rpy2. this is abc problem
        import tempfile
        temp_name = next(tempfile._get_candidate_names())
        robjects.r.options(width=10000)
        robjects.r['sink'](temp_name)
        x = robjects.r['summary'](cvmodsel)
        robjects.r['sink']()
        line = open(temp_name).readline()
        print(line, end='')
        os.remove(temp_name)
        print(x)
        # instead we could have used robjects.r['summary'](cvmodsel) if it was not bugged
        robjects.r['plot'](cvmodsel)

    @classmethod
    def model_selection(cls, target: pandas.DataFrame, ss: pandas.DataFrame,
                        index: Union[pandas.Series, pandas.Index], tol: float = .005,
                        method: str = "mnlogistic") -> None:
        """
        As the name suggest. Given the number of model it will select correct model using postpr in abc

        :param target: the observed summary statistic i n a dataframe format with only one line.
        :param ss: the summary statics in dataframe format
        :param index:  the index of the models in pandas.Series format
        :param tol: the level of tolerance. default is .005
        :param method: to tell which method is to be used in abc. default is mnlogistic. but can be rejection, neural net
         etc. as documented in the r.abc
        :return: will not return anything but will print summary of model selection
        """
        if method != 'rejection':
            noise = numpy.random.normal(0, 1e-4, ss.shape)
            ss = (ss + noise).clip(0)
        modsel = abc.postpr(target=target, index=index, sumstat=ss, tol=tol, method=method)
        cls.r_summary(modsel)
        return None

    @classmethod
    def gfit_all(cls, observed: pandas.DataFrame, ss: pandas.DataFrame, y_cat_dict: Dict[int, str], extra: str = '',
                 tol: float = .005, repeats: int = 100) -> None:
        """
        wrapper of goodness of fit. different goodness of fit for different models

        :param observed: the observed summary statistic in a in a dataframe format with single line
        :param ss:  the simulated summary statics in dataframe format.
        :param y_cat_dict: name of all the models. will be printed the pdf
        :param extra: internal. to add in the graph to say about the method
        :param tol: the level of tolerance. default is .005
        :param repeats: the number of nb.replicates to use to calculate the null
        :return: will not return anything. rather call goodness_fit to plot stuff and print the summary of gfit
        """
        # abc complains if there is no std in the data columns (ss)
        noise = numpy.random.normal(0, 1e-4, ss.shape)
        ss = (ss + noise).clip(0)
        for key in y_cat_dict:
            modelindex = ss.reset_index(drop=True).index[pandas.Series(ss.index) == y_cat_dict[key]].values
            ss_sub = ss.iloc[modelindex]
            cls.goodness_fit(target=observed, ss=ss_sub, name=y_cat_dict[key], extra=extra, tol=tol, repeats=repeats)

    @classmethod
    def goodness_fit(cls, target: pandas.DataFrame, ss: pandas.DataFrame, name: str, tol: float = .005,
                     extra: str = '', repeats: int = 100):
        """
        To test for goodness of fit of every model

        :param target:  the observed summary statistic in a dataframe format with single line
        :param ss:  the simulated summary statics in dataframe format.
        :param name: name of the nn
        :param tol: the level of tolerance. default is .005
        :param extra: internal. to add in the graph to say about the method
        :param repeats: the number of nb.replicates to use to calculate the null
        :return: will not return anything. but will plot goodness of fit also print the summary
        """
        fit = abc.gfit(target, ss, repeats, tol=tol)
        print(name)
        print(robjects.r['summary'](fit))
        out = name + ' ' + extra
        robjects.r['plot'](fit, main="Histogram under H0:" + out)

    @classmethod
    def outputing_csv(cls, modelindex: Union[pandas.Series, pandas.Index],
                      ss_predictions: pandas.DataFrame,
                      predict4mreal: pandas.DataFrame, folder: str = ''):
        """
        in case of everything satisfied. this will output the test data set in csv format which then later can be used
        by r directly. if you use it, it will delete all the middle files from the current directory if exists:
        x_test.h5, y_test.h5, x.h5, y.h5,scale_x.sav, scale_y.sav, params_header.csv

        :param modelindex: the model indexes in pandas series format
        :param ss_predictions: the predicted ss by nn on simulation[meaning nn(ss)]
        :param predict4mreal: the predicted ss by nn on real data [meaning nn(ss_real))]
        :param folder: to define the output/input folder. default is '' meaning current folder
        :return: will not return anything but will create files model_index.csv.gz,ss_predicted.csv.gz,ss_target.csv.gz
            and will remove x_test.h5, y_test.h5, x.h5, y.h5, scale_x.sav, scale_y.sav, params_header.csv
        """
        modelindex = pandas.Series(modelindex).rename('model_name')
        modelindex.to_csv(folder + 'model_index.csv.gz', index=False, header=True)
        ss_predictions.to_csv(folder + 'ss_predicted.csv.gz', index=False)
        predict4mreal.to_csv(folder + 'ss_target.csv.gz', index=False)
        filesremove = ['x_test.h5', 'y_test.h5', 'x.h5', 'y.h5', 'scale_x.sav', 'scale_y.sav', 'params_header.csv',
                       'y_cat_dict.txt']
        filesremove = [folder + file for file in filesremove]
        Misc.removefiles(filesremove)


class ABC_DLS_Classification_PreTrain(ABC_DLS_Classification):
    """
    Subset of class ABC_DLS_Classification. Specifically to do the pre train stuff. it will produce data in hdf5 format
    which then easily can be used in training part of the classification. it will also delete all the files that can be
    output by the classification. so that it will work on a clean sheet.

    :param info: the path of info file whose file column is the path of the file and second column defining the
        number of  parameters
    :param test_size: the number of test rows. everything else will be used for train. 10k is default
    :param chunksize:  the number of rows accessed at a time.
    :param scale: to tell if the data should be scaled or not. default is false. will be scaled by MinMaxscaler. The
        scaling will only happen on the ss.
    :param folder: to define the output folder. default is none meaning current folder
    :return: will not return anything but will create x.hdf5 ,y.hdf5 and scale_x so that it can be used for training
        later
    """

    def __new__(cls, info: str, test_size: int = int(1e4), chunksize: int = int(1e4), scale: bool = False,
                folder: str = ''):
        """
        Will call the wrapper_pre_train function from ABC_DLS_Classification

        :param info: the path of info file whose file column is the path of the file and second column defining the
            number of  parameters
        :param test_size: the number of test rows. everything else will be used for train. 10k is default
        :param chunksize:  the number of rows accessed at a time.
        :param scale: to tell if the data should be scaled or not. default is false. will be scaled by MinMaxscaler.
            The scaling will only happen on the ss.
        :param folder: to define the output folder. default is '' meaning current folder
        :return: will return data needed for training. will return x_(train,test), y_(train,test) scale_x (MinMaxScaler
            or None) and y_cat_dict ({0:'model1',1:'model2'..})
        """
        return cls.wrapper_pre_train(info=info, test_size=test_size, chunksize=chunksize, scale=scale, folder=folder)


class ABC_DLS_Classification_Train(ABC_DLS_Classification):
    """
    Subset of class ABC_DLS_Classification. Specifically to do the train stuff. it need training data set for x.h5 and
    y.h5 in the cwd in hdf5 matrix format (HD5matrix) of keras

    :param nn: custom function made for keras model. the path of that .py file. should have a def
        ANNModelCheck
    :param test_rows: the number of test rows. everything else will be used for train. 10k is default
    :param together: in case you want to send both train and test together (for validation data set). important if
            you do not want to lose data for earlystop validation split. look at extras/Dynamic.py to see how the tfknn
            should look like. Should not be used for big validation data set. Takes too much memory.
    :return: will not return anything but will train and save the file ModelClassification.h5
    """

    def __new__(cls, nn=None, test_rows=int(1e4), folder: str = '', together: bool = False):
        """
        This will call the wrapper function

        :param nn: custom function made for keras model. the path of that .py file. should have a def
            ANNModelCheck
        :param test_rows: the number of test rows. everything else will be used for train. 10k is default
        :param folder: to define the output folder. default is '' meaning current folder
        :param together: in case you want to send both train and test together (for validation data set). important if
            you do not want to lose data for earlystop validation split. look at extras/Dynamic.py to see how the tfknn
            should look like. Should not be used for big validation data set. Takes too much memory.
        :return: will not return anything but will train and save the file ModelClassification.h5
        """
        return cls.wrapper(nn=nn, test_rows=test_rows, folder=folder, together=together)

    @classmethod
    def wrapper(cls, nn: Optional[str] = None, test_rows: int = int(1e4), folder: str = '',
                together: bool = False) -> None:
        """
        wrapper for the class ABC_DLS_Classification_Train. it will train the data set in a given folder where x.h5 and
        y.h5 present.

        :param nn: custom function made for keras model. the path of that .py file. should have a def
            ANNModelCheck
        :param test_rows: the number of test rows. everything else will be used for train. 10k is default
        :param folder: to define the output folder. default is '' meaning current folder
        :param together: in case you want to send both train and test together (for validation data set). important if
            you do not want to lose data for earlystop validation split. look at extras/Dynamic.py to see how the tfknn
            should look like. Should not be used for big validation data set. Takes too much memory.
        :return: will not return anything but will train and save the file ModelClassification.h5
        """
        folder = Misc.creatingfolders(folder)
        y_train, y_test = cls.train_test_split_hdf5(file=folder + 'y.h5', test_rows=test_rows)
        x_train, x_test = cls.train_test_split_hdf5(file=folder + 'x.h5', test_rows=test_rows)
        if together:
            cls.wrapper_train(x_train=(x_train, x_test), y_train=(y_train, y_test),
                              nn=nn, folder=folder)
        else:

            cls.wrapper_train(x_train=x_train, y_train=y_train, nn=nn, folder=folder)


class ABC_DLS_Classification_CV(ABC_DLS_Classification):
    """
    Subset of class ABC_DLS_Classification. Specifically to calculate cross validation test. good if you do not have
    real data

    :param test_size: the number of test rows. everything else will be used for train. 10k is default
    :param tol: the level of tolerance for abc. default is .005
    :param method: to tell which method is used in abc. default is mnlogistic. but can be rejection, neural net etc.
        as documented in the r.abc
    :return: will not return anything but will plot the cross validation stuff of different models
    """

    def __new__(cls, test_size: int = int(1e4), tol: float = 0.05, method: str = 'rejection', cvrepeats: int = 100,
                folder: str = '') -> None:
        """
        This will call the wrapper function

        :param test_size: the number of test rows. everything else will be used for train. 10k is default
        :param tol: the level of tolerance for abc. default is .005
        :param method: to tell which method is used in abc. default is mnlogistic. but can be rejection, neural net etc.
            as documented in the r.abc
        :param cvrepeats: the number of repeats will be used for CV calculations
        :param folder: to define the output folder. default is '' meaning current folder
        :return: will not return anything but will plot the cross validation stuff of different models
        """
        return cls.wrapper(test_size=test_size, tol=tol, method=method, cvrepeats=cvrepeats, folder=folder)

    @classmethod
    def wrapper(cls, test_size: int = int(1e4), tol: float = 0.05, method: str = 'rejection',
                cvrepeats: int = 100, folder: str = '') -> None:
        """
        this will produce do the cross validation stuff using abc on the nn predicted stuff. good in case real data is
        not available yet

        :param test_size: the number of test rows. everything else will be used for train. 10k is default
        :param tol: the level of tolerance for abc. default is .005
        :param method: to tell which method is used in abc. default is mnlogistic. but can be rejection, neural net etc.
            as documented in the r.abc
        :param cvrepeats: the number of repeats will be used for CV calculations
        :param folder: to define the output folder. default is '' meaning current folder
        :return: will not return anything but will plot the cross validation stuff of different models
        """
        folder = Misc.creatingfolders(folder)
        ModelSeparation, x_test, y_test, scale_x, scale_y, y_cat_dict = cls.read_data(test_rows=test_size,
                                                                                      folder=folder)

        print("Evaluate with test:")
        ModelSeparation.evaluate(x_test, y_test, verbose=2)

        ssnn = cls.predict_repeats_mean(ModelSeparation, x_test, repeats=100)
        if y_cat_dict:
            indexnn = pandas.DataFrame(numpy.argmax(y_test, axis=1, out=None))[0].replace(y_cat_dict)
        else:
            indexnn = pandas.DataFrame(numpy.argmax(y_test, axis=1, out=None))[0]
        ssnn.index = indexnn
        if indexnn.value_counts().min() < cvrepeats:
            print('Cv repeats cannot be more than the number of samples present for a particular model. Please use '
                  'lesser number.')
            print(indexnn.value_counts())
            sys.exit(1)
        robjects.r['pdf'](folder + "CV.pdf")
        cls.plot_power_of_ss(ss=ssnn, index=pandas.Series(ssnn.index.values), tol=tol, method=method, repeats=cvrepeats)
        robjects.r['dev.off']()

    @classmethod
    def read_data(cls, test_rows: int = int(1e4), folder: str = '') -> Tuple[
        keras.models.Model, HDF5Matrix, HDF5Matrix, Optional[preprocessing.MinMaxScaler], Optional[
            preprocessing.MinMaxScaler], Dict[int, str]]:
        """
        wrapper to read all the data before doing the abc stuff

        :param test_rows: the number of rows kept for test data set. it will return those lines from the end
        :param folder: to define the output folder. default is '' meaning current folder
        :return: The fitted keras model, test data set of x and y, scale of x and y if exists and name of all the models
            ({0:'model1',1:'model2'..}
        """
        ModelSeparation = cls.loadingkerasmodel(ModelParamPredictionFile=folder + 'ModelClassification.h5')
        y_test = cls.reading_y_test(test_rows=test_rows, folder=folder)
        x_test = cls.reading_x_test(test_rows=test_rows, folder=folder)
        scale_x, scale_y = cls.read_scalex_scaley(folder=folder)
        y_cat_dict = cls.read_y_cat_dict(folder=folder)
        return ModelSeparation, x_test, y_test, scale_x, scale_y, y_cat_dict

    @classmethod
    def loadingkerasmodel(cls, ModelParamPredictionFile: str = 'ModelClassification.h5') -> keras.models.Model:
        """
        to load the saved keras model

        :param ModelParamPredictionFile: the .h5 file where it is saved
        :return: will return the model
        """
        from tensorflow.keras.models import load_model
        import tensorflow
        if os.path.isfile(ModelParamPredictionFile):
            try:
                model = load_model(ModelParamPredictionFile)
            # except AttributeError:
            #     model = load_model(ModelParamPredictionFile, custom_objects={'Gaussian_noise': cls.Gaussian_noise})
            except NameError:
                model = load_model(ModelParamPredictionFile, custom_objects={'tensorflow': tensorflow})

        else:
            print('The ANN model file could not be found please check. ', ModelParamPredictionFile)
            sys.exit(1)
        return model

    @classmethod
    def reading_y_test(cls, test_rows: int = int(1e4), folder: str = '') -> HDF5Matrix:
        """
        reading the file for y.h5/y_test.h5 and then return the y_test using hdf5matrix

        :param test_rows:  the number of rows kept for test data set. it will return those lines from the end
        :param folder: to define the output folder. default is '' meaning current folder
        :return: return y_test hdf5 format
        """
        if os.path.isfile(folder + 'y_test.h5'):
            y_test, _ = cls.train_test_split_hdf5(file=folder + 'y_test.h5', dataset='mydata', test_rows=0)
        elif os.path.isfile(folder + 'y.h5'):
            _, y_test = cls.train_test_split_hdf5(file=folder + 'y.h5', dataset='mydata', test_rows=test_rows)
        else:
            print('Could not file y.h5 or y_test.h5')
            sys.exit(1)
        return y_test

    @classmethod
    def reading_x_test(cls, test_rows: int = int(1e4), folder: str = '') -> HDF5Matrix:
        """
        reading the file for x.h5/x_test.h5 and then return the x_test using hdf5matrix

        :param test_rows:  the number of rows kept for test data set. it will return those lines from the end
        :param folder: to define the output folder. default is '' meaning current folder
        :return: return x_test hdf5 format
        """
        if os.path.isfile(folder + 'x_test.h5'):
            x_test, _ = cls.train_test_split_hdf5(file=folder + 'x_test.h5', dataset='mydata', test_rows=0)
        elif os.path.isfile(folder + 'x.h5'):
            _, x_test = cls.train_test_split_hdf5(file=folder + 'x.h5', dataset='mydata', test_rows=test_rows)
        else:
            print('Could not file x.h5 or x_test.h5')
            sys.exit(1)
        return x_test

    @classmethod
    def read_scalex_scaley(cls, folder: str = '') -> Tuple[
        Optional[preprocessing.MinMaxScaler], Optional[preprocessing.MinMaxScaler]]:
        """
        read if scale_x and scale_y is present in the folder and return it (MinMaxscaler)

        :param folder: to define the output folder. default is '' meaning current folder
        :return: return x_scale min max scaler if present
        """
        if os.path.isfile(folder + 'scale_x.sav'):
            scale_x = joblib.load(folder + 'scale_x.sav')
        else:
            print('scale_x.sav not found. Assuming no scaling is required')
            scale_x = None
        scale_y = None
        return scale_x, scale_y

    @classmethod
    def read_y_cat_dict(cls, folder: str = '') -> Dict[int, str]:
        """
        read the y_cat_dict.txt file in the cwd and return the dict

        :param folder: to define the output folder. default is '' meaning current folder
        :return: y_cat_dict in dict format ({0:'model1',1:'model2'..})
        """
        if os.path.isfile(folder + 'y_cat_dict.txt'):
            y_cat_dict = eval(open(folder + 'y_cat_dict.txt', 'r').read())

        else:
            print("Could not found the y_cat_dict.txt file. Thus the population names will be remain unknown")
            y_cat_dict = None
        return y_cat_dict


class ABC_DLS_Classification_After_Train(ABC_DLS_Classification_CV):
    """
    Subset of class ABC_DLS_Classification. To do the ABC part.  after training is done it will test on the test
    data set to see the power and then use a real data set to show how likely it support one model over another.
    it will use abc to give the power or standard deviation of the model that is predicted to know how much we are
    sure about the results. mainly it will do three parts of abc. one cv error , model selection and goodness of fit

    :param ssfile: the summary statistic on real data set. should be csv format
    :param test_size: the number of test rows. everything else will be used for train. 10k is default
    :param tol: the level of tolerance for abc. default is .01
    :param method: to tell which method is used in abc. default is rejection. but can be rejection, neural net etc. as
        documented in the r.abc
    :param csvout: in case of everything satisfied. this will output the test dataset in csv format. can be used  later
        by r
    :param folder: to define the output/input folder. default is '' meaning current folder
    :return: will not return anything but will produce the graphs and print out how much it is sure about any model
    """

    def __new__(cls, ssfile: str, test_size: int = int(1e4), tol: float = 0.05, method: str = 'rejection',
                csvout: bool = False, cvrepeats: int = 100, folder: str = '', frac: float = 1.0):
        """
        This will call the wrapper function

        :param ssfile:  the summary statistic on real data set. should be csv format
        :param test_size:  the number of test rows. everything else will be used for train. 10k is default
        :param tol: the level of tolerance for abc. default is .01
        :param method: to tell which method is used in abc. default is rejection. but can be rejection, neural net etc.
            as documented in the r.abc
        :param csvout: in case of everything satisfied. this will output the test dataset in csv format. can be used
            later by r
        :param cvrepeats: the number of repeats will be used for CV calculations
        :param folder: to define the output/input folder. default is '' meaning current folder
        :param frac: To multiply all the observed ss with some fraction. Important in case simulated data and observed
            data are not from same length. default is 1
        :return: will not return anything but will produce the graphs and print out how much it is sure about any model
        """
        return cls.wrapper(ssfile=ssfile, test_size=test_size, tol=tol, method=method, csvout=csvout,
                           cvrepeats=cvrepeats, folder=folder, frac=frac)

    @classmethod
    def wrapper(cls, ssfile: str, test_size: int = int(1e4), tol: float = 0.01, method: str = 'rejection',
                csvout: bool = False, cvrepeats: int = 100, folder: str = '', frac: float = 1.0) -> None:
        """
        This the wrapper for after training part of the classification. after training is done it will test on the test
        data set to see the power and then use a real data set to show how likely it support one model over another.
        it will use abc to give the power or standard deviation of the model that is predicted to know how much we are
        sure about the results. mainly it will do three parts of abc. one cv error , model selection and goodness of fit

        :param ssfile:  the summary statistic on real data set. should be csv format
        :param test_size:  the number of test rows. everything else will be used for train. 10k is default
        :param tol: the level of tolerance for abc. default is .01
        :param method: to tell which method is used in abc. default is rejection. but can be rejection, neural net etc.
            as documented in the r.abc
        :param csvout: in case of everything satisfied. this will output the test dataset in csv format. can be used
            later by r
        :param cvrepeats: the number of repeats will be used for CV calculations
        :param folder: to define the output/input folder. default is '' meaning current folder
        :param frac: To multiply all the observed ss with some fraction. Important in case simulated data and observed
            data are not from same length. default is 1
        :return: will not return anything but will produce the graphs and print out how much it is sure about any model
        """
        folder = Misc.creatingfolders(folder)
        ModelSeparation, x_test, y_test, scale_x, scale_y, y_cat_dict = cls.read_data(test_rows=test_size,
                                                                                      folder=folder)
        cls.wrapper_after_train(ModelSeparation=ModelSeparation, x_test=x_test, y_test=y_test, scale_x=scale_x,
                                y_cat_dict=y_cat_dict, ssfile=ssfile, method=method,
                                tolerance=tol, csvout=csvout, cvrepeats=cvrepeats, folder=folder, frac=frac)


# DLS parameter estimation stuff
class ABC_DLS_Params(ABC_DLS_Classification):
    """
    This is the main class to do the parameter estimation of ABC_DLS method. with given model underlying parameters
    it will compare with real data and will predict which parameter best predict the real data.

    :param info: the path of info file whose file column is the path of the file and second column defining the
        number of  parameters. only the first line will be used
    :param ssfile: the summary statistic on real data set. should be csv format
    :param chunksize: the number of rows accessed at a time.
    :param test_size:  the number of test rows. everything else will be used for train. 10k is default
    :param tol: the level of tolerance for abc. default is .005
    :param method: to tell which method is used in abc. default is mnlogistic. but can be rejection, neural net etc.
        as documented in the r.abc
    :param nn:  custom function made for keras model. the path of that .py file. should have a def
        ANNModelCheck
    :param together: in case you want to send both train and test together (for validation data set). important if
            you do not want to lose data for earlystop validation split. look at extras/Dynamic.py to see how the tfknn
            should look like. Should not be used for big validation data set. Takes too much memory.
    :param csvout:  in case of everything satisfied. this will output the test data set in csv format. can be used
        later by r
    :param scaling_x: to tell if the x (ss) should be scaled or not. default is false. will be scaled by MinMaxscaler.
    :param scaling_y: to tell if the y (parameters) should be scaled or not. default is false. will be scaled by
            MinMaxscaler.
    :param cvrepeats: the number of repeats will be used for CV calculations
    :param folder: to define the output folder. default is '' meaning current folder
    :return:  will not return anything but will plot and print the parameters

    """

    def __new__(cls, info: str, ssfile: str, chunksize: Optional[int] = None, test_size: int = int(1e4),
                tol: float = .005, method: str = 'rejection',
                nn: Optional[str] = None, together: bool = False, csvout: bool = False, scaling_x: bool = False,
                scaling_y: bool = False, cvrepeats: int = 100, folder: str = '', frac: float = 1.0) -> None:
        """
        This will call the wrapper function

        :param info: the path of info file whose file column is the path of the file and second column defining the
            number of  parameters. only the first line will be used
        :param ssfile: the summary statistic on real data set. should be csv format
        :param chunksize: the number of rows accessed at a time.
        :param test_size:  the number of test rows. everything else will be used for train. 10k is default
        :param tol: the level of tolerance for abc. default is .005
        :param method: to tell which method is used in abc. default is mnlogistic. but can be rejection, neural net etc.
            as documented in the r.abc
        :param nn:  custom function made for keras model. the path of that .py file. should have a def
            ANNModelCheck
        :param together: in case you want to send both train and test together (for validation data set). important if
            you do not want to lose data for earlystop validation split. look at extras/Dynamic.py to see how the tfknn
            should look like. Should not be used for big validation data set. Takes too much memory.
        :param csvout:  in case of everything satisfied. this will output the test dataset in csv format. can be used
            later by r
        :param scaling_x: to tell if the x (ss) should be scaled or not. default is false. will be scaled by
            MinMaxscaler.
        :param scaling_y: to tell if the y (parameters) should be scaled or not. default is false. will be scaled by
            MinMaxscaler.
        :param cvrepeats: the number of repeats will be used for CV calculations
        :param folder: to define the output folder. default is '' meaning current folder
        :param frac: To multiply all the observed ss with some fraction. Important in case simulated data and observed
            data are not from same length. default is 1
        :return:  will not return anything but will plot and print the parameters
        """
        return cls.wrapper(info=info, ssfile=ssfile, chunksize=chunksize, test_size=test_size, tol=tol, method=method,
                           nn=nn, together=together, csvout=csvout, scaling_x=scaling_x,
                           scaling_y=scaling_y, cvrepeats=cvrepeats, folder=folder, frac=frac)

    @classmethod
    def wrapper(cls, info: str, ssfile: str, chunksize: Optional[int] = None, test_size: int = int(1e4),
                tol: float = .005, method: str = 'rejection',
                nn: Optional[str] = None, together: bool = False, csvout: bool = False, scaling_x: bool = False,
                scaling_y: bool = False, cvrepeats: int = 100, folder: str = '', frac: float = 1.0) -> None:
        """
        the total wrapper of the parameter estimation method. with given model underlying parameters it will compare with
        real data and will predict which parameter best predict the real data.
        wrapper_pretrain(Misc.removefiles-> cls.read_info ->cls.separation_param_ss -> if chunksize :preparingdata_hdf5
        ;else preparingdata->Misc.removefiles) ->wrapper_train(Misc.loading_def_4m_file -> def ANNModelCheck)->
        wrapper_after_train(ModelParamPrediction.evaluate-> cls.read_ss_2_series-> cls.preparing_for_abc->
        cls.plot_param_cv_error->cls.abc_params-> Misc.removefiles->cls.csvout)

        :param info: the path of info file whose file column is the path of the file and second column defining the
            number of  parameters. only the first line will be used
        :param ssfile: the summary statistic on real data set. should be csv format
        :param chunksize: the number of rows accessed at a time.
        :param test_size:  the number of test rows. everything else will be used for train. 10k is default
        :param tol: the level of tolerance for abc. default is .005
        :param method: to tell which method is used in abc. default is mnlogistic. but can be rejection, neural net etc.
            as documented in the r.abc
        :param nn:  custom function made for keras model. the path of that .py file. should have a def
            ANNModelCheck
        :param together: If you want to send both train and test together in tfk model (train). Useful for validation
            test set in early stopping . need a specific format for nn.py. Look at Extra/Dynamic.py
        :param csvout:  in case of everything satisfied. this will output the test dataset in csv format. can be used
            later by r
        :param scaling_x: to tell if the x (ss) should be scaled or not. default is false. will be scaled by
            MinMaxscaler.
        :param scaling_y: to tell if the y (parameters) should be scaled or not. default is false. will be scaled by
            MinMaxscaler.
        :param cvrepeats: the number of repeats will be used for CV calculations
        :param folder: to define the output folder. default is '' meaning current folder
        :param frac: To multiply all the observed ss with some fraction. Important in case simulated data and observed
            data are not from same length. default is 1
        :return:  will not return anything but will plot and print the parameters
        """
        folder = Misc.creatingfolders(folder)
        x_train, x_test, scale_x, y_train, y_test, scale_y, paramfile = cls.wrapper_pre_train(info=info,
                                                                                              chunksize=chunksize,
                                                                                              test_size=test_size,
                                                                                              scaling_x=scaling_x,
                                                                                              scaling_y=scaling_y,
                                                                                              folder=folder)
        if together:
            ModelParamPrediction = cls.wrapper_train(x_train=(x_train, x_test), y_train=(y_train, y_test),
                                                     nn=nn, folder=folder)
        else:
            ModelParamPrediction = cls.wrapper_train(x_train=x_train, y_train=y_train, nn=nn,
                                                     folder=folder)
        cls.wrapper_aftertrain(ModelParamPrediction=ModelParamPrediction, x_test=x_test, y_test=y_test,
                               ssfile=ssfile, scale_x=scale_x, scale_y=scale_y,
                               paramfile='params_header.csv', method=method, tol=tol, csvout=csvout,
                               cvrepeats=cvrepeats,
                               folder=folder, frac=frac)

    @classmethod
    def wrapper_pre_train(cls, info: str, chunksize: Optional[int] = None, test_size: int = int(1e4),
                          scaling_x: bool = False, scaling_y: bool = False, folder: str = '') -> Tuple[
        Union[numpy.ndarray, HDF5Matrix], Union[numpy.ndarray, HDF5Matrix], Optional[preprocessing.MinMaxScaler], Union[
            numpy.ndarray, HDF5Matrix], Union[numpy.ndarray, HDF5Matrix], Optional[preprocessing.MinMaxScaler], str]:
        """
        This is a a wrapper on the pretrain for parameter estimation. this will build stuff just before the training in
        ANN.it will produce data in hdf5 or numpy array format which then easily can be used in training part, it will
        also delete all the files that can be output from ABC-DLS thus not clashing with them
        Misc.removefiles-> cls.read_info ->cls.separation_param_ss -> if chunksize :preparingdata_hdf5 ;else
        preparingdata->Misc.removefiles

        :param info: the path of info file whose file column is the path of the file and second column defining the
            number of  parameters. only first line will be used
        :param chunksize: the number of rows accessed at a time. in case of big data
        :param test_size: the number of test rows. everything else will be used for train. 10k is default
        :param scaling_x: to tell if the x (ss) should be scaled or not. default is false. will be scaled by
            MinMaxscaler.
        :param scaling_y: to tell if the y (parameters) should be scaled or not. default is false. will be scaled by
            MinMaxscaler.
        :param folder: to define the output folder. default is '' meaning current folder
        :return: will return (x_train, x_test, scale_x), (y_train, y_test, scale_y) and  header file path
            (params_header.csv)
        """
        folder = Misc.creatingfolders(folder)
        previousfiles = (
            'scale_x.sav', 'scale_y.sav', 'x_test.h5', 'y_test.h5', 'y.h5', 'x.h5', 'ModelParamPrediction.h5',
            'params.csv', 'ss.csv', 'params_header.csv', 'Checkpoint.h5')
        previousfilesfullpath = tuple(folder + file for file in previousfiles)
        Misc.removefiles(previousfilesfullpath)
        files, paramnumbers, names = cls.read_info(info=info)
        if len(files) > 1:
            print("there are more than one file. Only will work with the first file:", files[0])
        paramfile, simss = cls.separation_param_ss(filename=files[0], params_number=paramnumbers[0], folder=folder)
        if chunksize:
            x_train, x_test, scale_x, y_train, y_test, scale_y = cls.preparingdata_hdf5(paramfile=paramfile,
                                                                                        simss=simss,
                                                                                        chunksize=chunksize,
                                                                                        test_size=test_size,
                                                                                        scaling_x=scaling_x,
                                                                                        scaling_y=scaling_y,
                                                                                        folder=folder)
        else:
            x_train, x_test, scale_x, y_train, y_test, scale_y = cls.preparingdata(paramfile=paramfile,
                                                                                   simssfile=simss,
                                                                                   test_size=test_size,
                                                                                   scaling_x=scaling_x,
                                                                                   scaling_y=scaling_y, folder=folder)
        header = folder + 'params_header.csv'
        pandas.DataFrame(index=pandas.read_csv(paramfile, nrows=10).columns).transpose().to_csv(header,
                                                                                                index=False)
        Misc.removefiles([simss, paramfile])

        return x_train, x_test, scale_x, y_train, y_test, scale_y, header

    @classmethod
    def separation_param_ss(cls, filename: str, params_number: int, folder: str = '') -> Tuple[str, str]:
        """
        It will separate the parameters and ss in two different csv files. which then can be read by
        pandas.read_csv

        :param filename: the path of info or csv file. can be both csv or gz
        :param params_number: the number of parameters
        :param folder: to define the output folder. default is '' meaning current folder
        :return: will produce params.csv and ss.csv file
        """
        import os
        paramfile = folder + 'params.csv'
        ssfile = folder + 'ss.csv'

        if filename[-3:] == '.gz':
            paramcommand = Misc.joinginglistbyspecificstring(
                ["zcat", filename, '|', 'grep -v ",," |', 'cut -f-' + str(params_number), '-d ","', ' > ', paramfile])
            sscommand = Misc.joinginglistbyspecificstring(
                ["zcat", filename, '|', 'grep -v ",," |', 'cut -f' + str(params_number + 1) + '-', '-d ","', ' > ',
                 ssfile])
        else:
            paramcommand = Misc.joinginglistbyspecificstring(
                ["cat", filename, '|', 'grep -v ",," |', 'cut -f-' + str(params_number), '-d ","', ' > ', paramfile])
            sscommand = Misc.joinginglistbyspecificstring(
                ["cat", filename, '|', 'grep -v ",," |', 'cut -f' + str(params_number + 1) + '-', '-d ","', ' > ',
                 ssfile])
        os.system(paramcommand)
        os.system(sscommand)

        return paramfile, ssfile

    @classmethod
    def save_scale(cls, scale_x: preprocessing.MinMaxScaler, scale_y: Optional[preprocessing.MinMaxScaler],
                   folder: str = '') -> None:
        """
        As the name suggest it will save the scaling if scaling is done (MinMaxscaler) in a file. sklearn scaling will
        be saved.

        :param scale_x: scaled for x
        :param scale_y: scaled for y
        :param folder: to define the output folder. default is '' meaning current folder
        :return: will save in  scale_x.sav, scale_y.sav file. will return nothing
        """
        scaler_filename = folder + "scale_x.sav"
        if scale_x:
            joblib.dump(scale_x, scaler_filename)
        scaler_filename = folder + "scale_y.sav"
        if scale_y:
            joblib.dump(scale_y, scaler_filename)

    @classmethod
    def preparingdata_hdf5(cls, paramfile: str, simss: str, chunksize: int = 100, test_size: int = int(1e4),
                           scaling_x: bool = False, scaling_y: bool = False, folder: str = '') -> Tuple[
        HDF5Matrix, HDF5Matrix, Optional[preprocessing.MinMaxScaler], HDF5Matrix, HDF5Matrix, Optional[
            preprocessing.MinMaxScaler]]:
        """
        In case of chunk size is mentioned it will be assumed that the data is too big to save in ram and it will be
        saved in hdf5 format and it will split it in necessary steps

        :param paramfile: the parameter csv file path for y
        :param simss: the ss file path for x
        :param chunksize: the number of rows accessed at a time. default 100
        :param test_size: the number of test rows. everything else will be used for train. 10k is default
        :param scaling_x: to tell if the x (ss) should be scaled or not. default is false. will be scaled by
            MinMaxscaler.
        :param scaling_y: to tell if the y (parameters) should be scaled or not. default is false. will be scaled by
            MinMaxscaler.
        :param folder: to define the output folder. default is '' meaning current folder
        :return: will return train and test data for both x and y in hdf5 matrix format with scale_x and scale_y if
            required
        """

        xfile = folder + "x.h5"
        yfile = folder + "y.h5"
        Misc.removefiles([xfile, yfile])
        if scaling_x:
            scale_x = cls.MinMax4bigfile(csvpath=simss, h5path=xfile, chunksize=chunksize)
        else:
            scale_x = cls.MinMax4bigfile(csvpath=simss, h5path=xfile, chunksize=chunksize, scaling=False)
        x_train, x_test = cls.train_test_split_hdf5(xfile, test_rows=int(test_size))

        if scaling_y:
            scale_y = cls.MinMax4bigfile(csvpath=paramfile, h5path=yfile, chunksize=chunksize)
        else:
            scale_y = cls.MinMax4bigfile(csvpath=paramfile, h5path=yfile, chunksize=chunksize, scaling=False)
        y_train, y_test = cls.train_test_split_hdf5(yfile, test_rows=int(test_size))
        cls.save_scale(scale_x, scale_y, folder=folder)
        return x_train, x_test, scale_x, y_train, y_test, scale_y

    @classmethod
    def preparingdata(cls, paramfile: str, simssfile: str, test_size: int = int(1e4), scaling_x: bool = False,
                      scaling_y: bool = False, folder: str = '') -> Tuple[
        numpy.ndarray, numpy.ndarray, Optional[preprocessing.MinMaxScaler], numpy.ndarray, numpy.ndarray, Optional[
            preprocessing.MinMaxScaler]]:
        """
        In case the data is smaller and can be fit inside ram it will load the data in the ram and split it for train
        and test data for params (y) and sfs/ss (x). it will use min max scaler to scale it if required

        :param paramfile: the parameter csv file or y
        :param simssfile: the ss file path for x
        :param test_size: the number of test rows. everything else will be used for train. 10k is default
        :param scaling_x: to tell if the x (ss) should be scaled or not. default is false. will be scaled by
            MinMaxscaler.
        :param scaling_y: to tell if the y (parameters) should be scaled or not. default is false. will be scaled by
            MinMaxscaler.
        :param folder: to define the output folder. default is '' meaning current folder
        :return: will return train and test data fro both x and y in numpy format with scale_x and scale_y if required
        """
        from sklearn.model_selection import train_test_split
        params = pandas.read_csv(paramfile)
        if scaling_y:
            scale_y = preprocessing.MinMaxScaler()
            y = scale_y.fit_transform(params.values)
        else:
            y = params.values
            scale_y = None

        simss = pandas.read_csv(simssfile)
        if scaling_x:

            scale_x = preprocessing.MinMaxScaler()
            x = scale_x.fit_transform(simss.values)
        else:
            x = simss.values
            scale_x = None

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

        cls.save_scale(scale_x, scale_y, folder=folder)
        x_test_file = folder + "x_test.h5"
        y_test_file = folder + "y_test.h5"
        Misc.removefiles([x_test_file, y_test_file])
        Misc.numpy2hdf5(x_test, x_test_file)
        Misc.numpy2hdf5(y_test, y_test_file)
        return x_train, x_test, scale_x, y_train, y_test, scale_y

    @classmethod
    def wrapper_train(cls, x_train: Union[numpy.ndarray, HDF5Matrix, tuple],
                      y_train: Union[numpy.ndarray, HDF5Matrix, tuple],
                      nn: Optional[str] = None, folder: str = '') -> keras.models.Model:
        """
        This is to the wrapper for the training for parameter estimation. the slowest part of the code.it need training
        data set for x and y. can be either numpy array or hdf5 matrix format (HD5matrix) of keras
        Misc.loading_def_4m_file -> def ANNModelCheck

        :param x_train: train part of x aka summary statistics
        :param y_train: train part of all the parameters
        :param nn: custom function made for keras model. the path of that .py file. should have a def has
            ANNModelParams as def in Any.py
        :param folder: to define the output folder. default is '' meaning current folder
        :return: will return the keras model. it will also save the model in ModelParamPrediction.h5
        """
        folder = Misc.creatingfolders(folder)
        Misc.removefiles((folder + "ModelParamPrediction.h5", folder + "Checkpoint.h5"))
        if nn:
            ANNModelParams = Misc.loading_def_4m_file(filepath=nn, defname='ANNModelParams')
            if ANNModelParams is None:
                print('Could not find the ANNModelParams in', nn,
                      '. Please check. Now using the default ANNModelParams')
                ANNModelParams = cls.ANNModelParams
        else:
            ANNModelParams = cls.ANNModelParams
        # needed as Checkpoint.h5 should be inside the folder and i do not want to make ANNModelCheck complicated with
        # another variable 'folder'
        parentfolder = os.getcwd()
        if folder != '':
            os.chdir(folder)
        ModelParamPrediction = ANNModelParams(x=x_train, y=y_train)
        cls.check_save_tfk_model(model=ModelParamPrediction, output="ModelParamPrediction.h5",
                                 check_point='Checkpoint.h5')
        # same as above to change back to previous stage
        if folder != '':
            os.chdir(parentfolder)
        return ModelParamPrediction

    @classmethod
    def ANNModelParams(cls, x: Union[numpy.ndarray, HDF5Matrix],
                       y: Union[numpy.ndarray, HDF5Matrix]) -> keras.models.Model:
        """
        A basic model for ANN to calculate parameters

        :param x:  the x or summary statistics. can be numpy array or hdf5.
        :param y: the parameters which produced those ss
        :return: will return the trained model
        """
        model = Sequential()
        model.add(GaussianNoise(0.01, input_shape=(x.shape[1],)))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(.01))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(.01))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(.01))
        model.add(Dense(y.shape[1]))
        model.compile(loss='logcosh', optimizer='Nadam', metrics=['accuracy'])
        # adding an early stop so that it does not overfit
        ES = EarlyStopping(monitor='val_loss', patience=100)
        # checkpoint
        CP = ModelCheckpoint('Checkpoint.h5', verbose=1, save_best_only=True)

        model.fit(x, y, epochs=int(2e6), verbose=2, shuffle=True, callbacks=[ES, CP], validation_split=.2)

        return model

    @classmethod
    def wrapper_aftertrain(cls, ModelParamPrediction: keras.models.Model, x_test: Union[numpy.ndarray, HDF5Matrix],
                           y_test: Union[numpy.ndarray, HDF5Matrix], ssfile: str,
                           scale_x: Optional[preprocessing.MinMaxScaler], scale_y: Optional[preprocessing.MinMaxScaler],
                           paramfile: str = 'params_header.csv', method: str = 'rejection', tol: float = .005,
                           csvout: bool = False, cvrepeats: int = 100, folder: str = '', frac: float = 1.0) -> None:
        """
        The wrapper to test how the training using ANN works. after training is done it will test on the test  data set
        to see the power and then use a real data set to show what most likely parameters can create the real data.
        it will use abc to give the power or standard deviation of the parameters that is predicted by nn to know how
        much we are sure about the results. mainly it will do two parts of abc. one cv error and parameter estimation
        ModelParamPrediction.evaluate-> cls.read_ss_2_series-> cls.preparing_for_abc->cls.plot_param_cv_error->
        cls.abc_params-> Misc.removefiles->cls.csvout

        :param ModelParamPrediction: The fitted keras model
        :param x_test: the test part of x aka summary statistics
        :param y_test: the test part of y aka parameter dataset
        :param ssfile: the real ss file path
        :param scale_x: the scale of ss if exist
        :param scale_y: the scale of parameters if exist
        :param paramfile: the path of parameter header file. default is 'params_header.csv'
        :param method: the method to be used in r_abc. default is rejection
        :param tol: the tolerance level to be used in r_abc. default is .005
        :param csvout: in case of everything satisfied. this will output the test dataset in csv format. can be used
            later by r
        :param cvrepeats: the number of repeats will be used for CV calculations
        :param folder: to define the output folder. default is '' meaning current folder
        :param frac: To multiply all the observed ss with some fraction. Important in case simulated data and observed
            data are not from same length. default is 1
        :return: will not return anything but print out the result and save files for plot
        """
        print("Evaluate with test:")
        ModelParamPrediction.evaluate(x_test[:], y_test[:], verbose=2)
        params_names = pandas.read_csv(folder + paramfile).columns
        ss = cls.read_ss_2_series(file=ssfile)
        ss = ss * numpy.array(frac)
        test_predictions, predict4mreal, params_unscaled = cls.preparing_for_abc(ModelParamPrediction, x_test, y_test,
                                                                                 scale_x, scale_y, params_names, ss)

        print("ANN predict")
        print(predict4mreal.transpose())

        print('correlation between params. Prior')
        print(params_unscaled.corr().to_string())

        print('correlation between predicted params. Posterior')
        print(test_predictions.corr().to_string())

        cls.plot_param_cv_error(param=params_unscaled, ss=test_predictions, name=folder + 'nnparamcv.pdf', tol=tol,
                                method=method, repeats=cvrepeats)
        cls.abc_params(target=predict4mreal, param=params_unscaled, ss=test_predictions, method=method, tol=tol,
                       name=folder + 'paramposterior.pdf')
        if csvout:
            cls.outputing_csv(params_unscaled=params_unscaled, test_predictions=test_predictions,
                              predict4mreal=predict4mreal, folder=folder)

    @classmethod
    def R_std_columns(cls, df: pandas.DataFrame) -> pandas.Series:
        """
        As pandas 2 r we lose precision. The std should be calculated in r for r_ABC

        :param df: the pandas dataframe
        :return: a pandas series with the columns and their standard deviation
        """
        std_r = robjects.r['sapply'](df, 'sd')
        return pandas.Series(std_r, index=df.columns)

    @classmethod
    def remove_constant(cls, test_predictions: pandas.DataFrame, predict4mreal: pandas.DataFrame,
                        params_unscaled: pandas.DataFrame) -> Tuple[
        pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
        """
        R_abc complains if the parameters do not have std (all are same or constant). this step will remove those

        :param test_predictions: the predicted values dataframe from ANN model
        :param predict4mreal: the predicted value from the real data
        :param params_unscaled: the real parameters to produce the ss
        :return: will return the all the inputs but will remove columns with 0 std
        """

        std = cls.R_std_columns(params_unscaled)
        columns2bdrop = std[std == 0].index

        if len(columns2bdrop) == params_unscaled.shape[1]:
            print('All the columns in the params have no standard deviation. Please check in params for y_test')
            sys.exit(1)
        elif len(columns2bdrop) > 0:
            print('These columns do not have any std in the test data set. Is being removed before ABC')
            print(columns2bdrop)
            test_predictions = test_predictions.drop(columns2bdrop, axis=1)
            predict4mreal = predict4mreal.drop(columns2bdrop, axis=1)
            params_unscaled.drop(columns2bdrop, axis=1)

        std = cls.R_std_columns(test_predictions)
        columns2bdrop = std[std == 0].index
        if len(columns2bdrop) == test_predictions.shape[1]:
            print(
                'All the columns in the prediction have no standard deviation. Please check in predictions for ANN '
                'model. The ANN training was not good')
            sys.exit(1)
        elif len(columns2bdrop) > 0:
            print(
                'These columns do not have any std in the predicted data set. Is being removed before ABC. This is not '
                'a good sign. Please check the training model')
            print(columns2bdrop)
            test_predictions = test_predictions.drop(columns2bdrop, axis=1)
            predict4mreal = predict4mreal.drop(columns2bdrop, axis=1)
            params_unscaled.drop(columns2bdrop, axis=1)

        return test_predictions, predict4mreal, params_unscaled

    @classmethod
    def preparing_for_abc(cls, ModelParamPrediction: keras.models.Model, x_test: Union[numpy.ndarray, HDF5Matrix],
                          y_test: Union[numpy.ndarray, HDF5Matrix], scale_x: Optional[preprocessing.MinMaxScaler],
                          scale_y: Optional[preprocessing.MinMaxScaler], params_names: numpy.ndarray,
                          ss: pandas.Series) -> Tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
        """
        as the name suggest after ann ran it will prepare the data for abc analysis

        :param ModelParamPrediction: the ann model that was run by keras tf
        :param x_test: the test part of x aka summary statistics
        :param y_test: the y_test or parameters series which never ran on the ann itself
        :param scale_x: the MinMax scaler of x axis. can be None
        :param scale_y: the MinMax scaler of y axis. can be None
        :param params_names: all the parameter or y header in a numpy.array
        :param ss: the real ss in a pandas series
        :return: will return test_prediction [ANN(x_test)_ unscaled y], predict4mreal [ANN(real_ss_scaled)_unscaled y]
            params_unscaled [y_test_unscaled y]
        """
        if scale_y:
            test_predictions = scale_y.inverse_transform(ModelParamPrediction.predict(x_test))
            test_predictions = pandas.DataFrame(test_predictions, columns=params_names[:y_test.shape[1]])
            params_unscaled = pandas.DataFrame(scale_y.inverse_transform(y_test[:]),
                                               columns=params_names[-y_test.shape[1]:])
        else:
            test_predictions = ModelParamPrediction.predict(x_test[:])
            test_predictions = pandas.DataFrame(test_predictions, columns=params_names[:y_test.shape[1]])
            params_unscaled = pandas.DataFrame(y_test[:], columns=params_names[-y_test.shape[1]:])
        if scale_x:
            ssscaled = scale_x.transform(ss.values.reshape(1, -1))
            if scale_y:
                predict4mreal = pandas.DataFrame(scale_y.inverse_transform(ModelParamPrediction.predict(ssscaled)))
            else:
                predict4mreal = pandas.DataFrame(ModelParamPrediction.predict(ssscaled))
        else:
            ssscaled = ss.values.reshape(1, -1)
            if scale_y:
                predict4mreal = pandas.DataFrame(scale_y.inverse_transform(ModelParamPrediction.predict(ssscaled)))
            else:
                predict4mreal = pandas.DataFrame(ModelParamPrediction.predict(ssscaled))

        predict4mreal.columns = params_names[:y_test.shape[1]]

        test_predictions, predict4mreal, params_unscaled = cls.remove_constant(test_predictions=test_predictions,
                                                                               predict4mreal=predict4mreal,
                                                                               params_unscaled=params_unscaled)
        return test_predictions, predict4mreal, params_unscaled

    @classmethod
    def plot_param_cv_error(cls, param: pandas.DataFrame, ss: pandas.DataFrame, tol: float = .001, repeats: int = 1000,
                            method: str = "loclinear", name: str = 'cvparam.pdf') -> None:
        """
        to plot the cv error per parameters. in case of neuralnet it will calculate cv by putting all columns together
        and loclinear it will do independently per column. for rejection both will be done

        :param param: the parameter data frame format (y_test)
        :param ss: the summary statics in dataframe format
        :param tol: the tolerance level. default is .001
        :param repeats: the number of repeats for cv calculation. default is 100
        :param method: the method to calculate abc cv error. can be "rejection", "loclinear", and "neuralnet". default
            loclinear
        :param name: the ouput save file name
        :return: will not return anything but save the plot and also print out summary of cv
        """
        trace = False
        import tempfile
        temp_name = next(tempfile._get_candidate_names())
        if method == 'rejection' or method == 'loclinear':
            robjects.r['pdf'](name)
            print('Cv error independently')
            for colnum in range(param.shape[1]):
                cvresreg = abc.cv4abc(param=pandas.DataFrame(param.iloc[:, colnum]),
                                      sumstat=pandas.DataFrame(ss.iloc[:, colnum]), nval=repeats, tols=tol,
                                      method=method, trace=trace)
                print(param.iloc[:, colnum].name)
                print(robjects.r['summary'](cvresreg))
                robjects.r['plot'](cvresreg, ask=False)
            robjects.r['dev.off']()
        if method == 'rejection' or method == 'neuralnet':
            print('Cv error together')
            robjects.r['pdf'](name[:-4] + '_together.pdf')
            if method == 'neuralnet':
                robjects.r['sink'](temp_name)
                cvresreg = abc.cv4abc(param=param, sumstat=ss, nval=repeats, tols=tol, method=method, trace=trace)
                robjects.r['sink']()
                os.remove(temp_name)
            else:
                cvresreg = abc.cv4abc(param=param, sumstat=ss, nval=repeats, tols=tol, method=method, trace=trace)
            # text wrapping problem in r which cant be solved by options(width=10000) in rpy2
            robjects.r['sink'](temp_name)
            together = robjects.r['summary'](cvresreg)
            line = open(temp_name).readline()
            print(line)
            os.remove(temp_name)
            print(pandas.DataFrame(list(together), index=together.colnames,
                                   columns=together.rownames).transpose().to_string())
            robjects.r['plot'](cvresreg, ask=False)
            # instead we could have used robjects.r['summary'](cvresreg) if it was not bugged
            robjects.r['dev.off']()

    @classmethod
    def abc_params(cls, target: pandas.DataFrame, param: pandas.DataFrame, ss: pandas.DataFrame, tol: float = .01,
                   method: str = "loclinear", name: str = 'paramposterior.pdf') -> None:
        """
        the final abc calculation on real data.

        :param target: the real ss in a pandas  data frame format
        :param param: the parameter data frame format (y_test)
        :param ss: the summary statics in dataframe format
        :param tol: the tolerance level. default is .001
        :param method: the method to calculate abc cv error. can be "rejection", "loclinear", and "neuralnet". default
            loclinear
        :param name: the ouput save file name
        :return: will not return anything but save the plot and print out the summary
        """
        import tempfile
        temp_name = next(tempfile._get_candidate_names())
        if method == 'rejection' or method == 'loclinear':
            print('Separately')
            robjects.r['pdf'](name)
            for colnum in range(param.shape[1]):
                res = abc.abc(target=pandas.DataFrame(target.iloc[:, colnum]),
                              param=pandas.DataFrame(param.iloc[:, colnum]),
                              sumstat=pandas.DataFrame(ss.iloc[:, colnum]),
                              method=method, tol=tol)
                cls.r_summary(res)
                if method == 'rejection':
                    robjects.r['hist'](res, ask=False)
                else:
                    robjects.r['plot'](res, param=param.iloc[:, colnum], ask=False)
            robjects.r['dev.off']()
        if method == 'rejection' or method == 'neuralnet':
            print('together')
            if method == 'neuralnet':
                robjects.r['sink'](temp_name)
                res = abc.abc(target=target, param=param, sumstat=ss, method=method, tol=tol)
                robjects.r['sink']()
                os.remove(temp_name)
            else:
                res = abc.abc(target=target, param=param, sumstat=ss, method=method, tol=tol)
            cls.r_summary(res)
            if method == 'rejection':
                print('rejection plot not implemented yet')
            else:
                robjects.r['plot'](res, param=param, onefile=True, file=name[:-4] + '_together')

    @classmethod
    def outputing_csv(cls, params_unscaled: pandas.DataFrame, test_predictions: pandas.DataFrame,
                      predict4mreal: pandas.DataFrame, folder: str = '') -> None:
        """
        in case you need csv file output of predicted params from nn, which then can be directly used by R. if you use
        it, it will delete all the middle files from the current directory if exists: x_test.h5, y_test.h5, x.h5, y.h5,
        scale_x.sav, scale_y.sav, params_header.csv

        :param test_predictions: the predicted values dataframe from ANN model
        :param predict4mreal: the predicted value from the real data
        :param params_unscaled: the real parameters to produce the ss
        :param folder: to define the output folder. default is '' meaning current folder
        :return: will not return anything but save 3 files params.csv.gz, ss_predicted.csv.gz, ss_target.csv.gz which
            can be used for further in R for in depth abc
        """
        params_unscaled.to_csv(folder + 'params.csv.gz', index=False)
        test_predictions.to_csv(folder + 'ss_predicted.csv.gz', index=False)
        predict4mreal.to_csv(folder + 'ss_target.csv.gz', index=False)
        notrequired = ('x_test.h5', 'y_test.h5', 'x.h5', 'y.h5', 'scale_x.sav', 'scale_y.sav', 'params_header.csv')
        notrequired = tuple(folder + file for file in notrequired)
        Misc.removefiles(notrequired)


class ABC_DLS_Params_PreTrain(ABC_DLS_Params):
    """
    Subset of Parameter estimation. just to prepare the data for tfk.this will build stuff just before the training in
    ANN.it will produce data in hdf5 or numpy array format which then easily can be used in training part, it will
    also delete all the files that can be output from ABC-DLS thus not clashing with them

    :param info: the path of info file whose file column is the path of the file and second column defining the
            number of  parameters. only first line will be used
    :param chunksize: the number of rows accessed at a time. in case of big data
    :param test_size: the number of test rows. everything else will be used for train. 10k is default
    :param scaling_x: to tell if the x (ss) should be scaled or not. default is false. will be scaled by
        MinMaxscaler.
    :param scaling_y: to tell if the y (parameters) should be scaled or not. default is false. will be scaled by
        MinMaxscaler.
    :param folder: to define the output folder. default is '' meaning current folder
    :return: will not return anything but will create x.hdf5 ,y.hdf5, scale_x, scale_x  and params_header.csv
    """

    def __new__(cls, info: str, test_size: int = int(1e4), chunksize: Optional[int] = int(1e4),
                scaling_x: bool = False, scaling_y: bool = False, folder: str = ''):
        """
        This will  call the wrapper_pre_train function from ABC_DLS_Params

        :param info: the path of info file whose file column is the path of the file and second column defining the
            number of  parameters. only first line will be used
        :param chunksize: the number of rows accessed at a time. in case of big data
        :param test_size: the number of test rows. everything else will be used for train. 10k is default
        :param scaling_x: to tell if the x (ss) should be scaled or not. default is false. will be scaled by
            MinMaxscaler.
        :param scaling_y: to tell if the y (parameters) should be scaled or not. default is false. will be scaled by
            MinMaxscaler.
        :param folder: to define the output folder. default is '' meaning current folder
        :return: will not return anything but will create x.hdf5 ,y.hdf5, scale_x, scale_x  and params_header.csv
        """
        return cls.wrapper_pre_train(info=info, test_size=test_size, chunksize=chunksize, scaling_x=scaling_x,
                                     scaling_y=scaling_y, folder=folder)


class ABC_DLS_Params_Train(ABC_DLS_Params):
    """
    Subset for the training of parameter estimation. the slowest part of the code.it need training data set for x and y.
    can be hdf5 matrix format (HD5matrix) of keras

    :param test_rows: the number of rows kept for test data set. it will return those lines from the end
    :param nn: custom function made for keras model. the path of that .py file. should have a def has
        ANNModelParams as def in Any.py
    :param folder: to define the output folder. default is '' meaning current folder
    :return: will not return anything but save the keras model
    """

    def __new__(cls, test_rows: int = int(1e4), nn: Optional[str] = None, folder: str = '',
                together: bool = False) -> None:
        """
        This will call the wrapper function

        :param test_rows: the number of rows kept for test data set. it will return those lines from the end
        :param nn: custom function made for keras model. the path of that .py file. should have a def has
            ANNModelParams as def in Any.py
        :param folder: to define the output folder. default is '' meaning current folder
        :param together: in case you want to send both train and test together (for validation data set). important if
            you do not want to lose data for earlystop validation split. look at extras/Dynamic.py to see how the tfknn
            should look like. Should not be used for big validation data set. Takes too much memory.
        :return: will not return anything but save the keras model
        """
        return cls.wrapper(test_rows=test_rows, nn=nn, folder=folder, together=together)

    @classmethod
    def wrapper(cls, test_rows: int = int(1e4), nn: Optional[str] = None, folder: str = '',
                together: bool = False) -> None:
        """
        This is the wrapper. Will write later

        :param test_rows: the number of rows kept for test data set. it will return those lines from the end
        :param nn: custom function made for keras model. the path of that .py file. should have a def has
            ANNModelParams as def in Any.py
        :param folder: to define the output folder. default is '' meaning current folder
        :param together: in case you want to send both train and test together (for validation data set). important if
            you do not want to lose data for earlystop validation split. look at extras/Dynamic.py to see how the tfknn
            should look like. Should not be used for big validation data set. Takes too much memory.
        :return: will not return anything but save the keras model
        """
        folder = Misc.creatingfolders(folder)
        y_train, y_test = ABC_DLS_Classification_Train.train_test_split_hdf5(file=folder + 'y.h5', test_rows=test_rows)
        x_train, x_test = ABC_DLS_Classification_Train.train_test_split_hdf5(file=folder + 'x.h5', test_rows=test_rows)
        if together:

            cls.wrapper_train(x_train=(x_train, x_test), y_train=(y_train, y_test),
                              nn=nn, folder=folder)

        else:
            cls.wrapper_train(x_train=x_train, y_train=y_train, nn=nn,
                              folder=folder)


class ABC_DLS_Params_CV(ABC_DLS_Params):
    """
    Subset of parameter estimation Specifically to calculate cross validation test. good if you do not have
    real data

    :param test_size: the number of test rows. everything else will be used for train. 10k is default
    :param tol: the level of tolerance for abc. default is .005
    :param method: to tell which method is used in abc. default is mnlogistic. but can be rejection, neural net etc.
        as documented in the r.abc
    :param folder: to define the output folder. default is '' meaning current folder
    :return: will not return anything but will plot the cross validation stuff for parameter estimation
    """

    def __new__(cls, test_size: int = int(1e3), tol: float = 0.01, method: str = 'neuralnet',
                cvrepeats: int = 100, folder: str = '') -> None:
        """
        This will call the wrapper function

        :param test_size: the number of test rows. everything else will be used for train. 10k is default
        :param tol: the level of tolerance for abc. default is .005
        :param method: to tell which method is used in abc. default is mnlogistic. but can be rejection, neural net etc.
            as documented in the r.abc
        :param cvrepeats: the number of repeats will be used for CV calculations
        :param folder: to define the output folder. default is '' meaning current folder
        :return: will not return anything but will plot the cross validation stuff for parameter estimation
        """
        return cls.wrapper(test_size=test_size, tol=tol, method=method, cvrepeats=cvrepeats, folder=folder)

    @classmethod
    def wrapper(cls, test_size: int = int(1e3), tol: float = 0.01, method: str = 'neuralnet',
                cvrepeats: int = 100, folder: str = '') -> None:
        """
       Subset of Parameter estimation Specifically to calculate cross validation test. good if you do not have
       real data

       :param test_size: the number of test rows. everything else will be used for train. 10k is default
       :param tol: the level of tolerance for abc. default is .005
       :param method: to tell which method is used in abc. default is mnlogistic. but can be rejection, neural net etc.
       :param cvrepeats: the number of repeats will be used for CV calculations
       :param folder: to define the output folder. default is '' meaning current folder
       :return: will not return anything but will plot the cross validation stuff for parameter estimation
       """
        folder = Misc.creatingfolders(folder)
        ModelParamPrediction, x_test, y_test, scale_x, scale_y = cls.read_data(test_rows=test_size, folder=folder)

        print("Evaluate with test:")
        ModelParamPrediction.evaluate(x_test[:], y_test[:], verbose=2)
        params_names = pandas.read_csv(folder + 'params_header.csv').columns
        test_predictions, params_unscaled = cls.preparing_for_abc(ModelParamPrediction=ModelParamPrediction,
                                                                  x_test=x_test, y_test=y_test, scale_x=scale_x,
                                                                  scale_y=scale_y, params_names=params_names)
        print('correlation between params. Prior')
        print(params_unscaled.corr().to_string())

        print('correlation between predicted params. Posterior')
        print(test_predictions.corr().to_string())

        cls.plot_param_cv_error(param=params_unscaled, ss=test_predictions, name=folder + 'nnparamcv.pdf', tol=tol,
                                method=method, repeats=cvrepeats)

    @classmethod
    def read_data(cls, test_rows: int = int(1e4), folder: str = '') -> Tuple[
        keras.models.Model, Union[numpy.ndarray, HDF5Matrix], Union[numpy.ndarray, HDF5Matrix], Optional[
            preprocessing.MinMaxScaler], Optional[preprocessing.MinMaxScaler]]:
        """
        to read all the data before doing the abc stuff

        :param test_rows: the number of rows kept for test data set. it will return those lines from the end
        :param folder: to define the output folder. default is '' meaning current folder
        :return: The fitted keras model, test data set of x and y, scale of x and y if exists
        """
        ModelParamPrediction = ABC_DLS_Classification_CV.loadingkerasmodel(folder + 'ModelParamPrediction.h5')
        y_test = ABC_DLS_Classification_CV.reading_y_test(test_rows=test_rows, folder=folder)
        x_test = ABC_DLS_Classification_CV.reading_x_test(test_rows=test_rows, folder=folder)
        scale_x, scale_y = cls.read_scalex_scaley(folder=folder)
        return ModelParamPrediction, x_test, y_test, scale_x, scale_y

    @classmethod
    def read_scalex_scaley(cls, folder: str = '') -> Tuple[
        Optional[preprocessing.MinMaxScaler], Optional[preprocessing.MinMaxScaler]]:
        """
        this to read if scale_x and scale_y is present in the folder and return it (MinMaxscaler)

        :param folder: to define the output folder. default is '' meaning current folder
        :return: return x_scale and y_scale min max scaler if present
        """
        if os.path.isfile(folder + 'scale_x.sav'):
            scale_x = joblib.load(folder + 'scale_x.sav')
        else:
            print('scale_x.sav not found. Assuming no scaling is required for x ')
            scale_x = None
        if os.path.isfile(folder + 'scale_y.sav'):
            scale_y = joblib.load(folder + 'scale_y.sav')
        else:
            print('scale_y.sav not found. Assuming no scaling is required for y')
            scale_y = None
        return scale_x, scale_y

    @classmethod
    def preparing_for_abc(cls, ModelParamPrediction: keras.models.Model, x_test: Union[numpy.ndarray, HDF5Matrix],
                          y_test: Union[numpy.ndarray, HDF5Matrix], scale_x: Optional[preprocessing.MinMaxScaler],
                          scale_y: Optional[preprocessing.MinMaxScaler], params_names: numpy.ndarray) -> Tuple[
        pandas.DataFrame, pandas.DataFrame]:
        """
        as the name suggest after ann ran it will prepare the data for abc analysis

        :param ModelParamPrediction: the ann model that was run by keras tf
        :param x_test: the test part of x aka summary statistics
        :param y_test: the y_test series which never ran on the ann itself
        :param y_test: the y_test or parameters series which never ran on the ann itself
        :param scale_x: the MinMax scaler of x axis. can be None
        :param scale_y: the MinMax scaler of y axis. can be None
        :param params_names: all the parameter or y header in a numpy.array
        :return: will return test_prediction [ANN(x_test)_ unscaled y], predict4mreal [ANN(real_ss_scaled)_unscaled y]
            params_unscaled [y_test_unscaled y]
        """
        if scale_y:
            test_predictions = scale_y.inverse_transform(ModelParamPrediction.predict(x_test[:]))
            test_predictions = pandas.DataFrame(test_predictions, columns=params_names[:y_test.shape[1]])
            params_unscaled = pandas.DataFrame(scale_y.inverse_transform(y_test[:]),
                                               columns=params_names[-y_test.shape[1]:])
        else:
            test_predictions = ModelParamPrediction.predict(x_test[:])
            test_predictions = pandas.DataFrame(test_predictions, columns=params_names[:y_test.shape[1]])
            params_unscaled = pandas.DataFrame(y_test[:], columns=params_names[-y_test.shape[1]:])

        return test_predictions, params_unscaled


class ABC_DLS_Params_After_Train(ABC_DLS_Params):
    """
   The subset class to test to parameter estimation. after training is done it will test on the test data set
   to see the power and then use a real data set to show what most likely parameters can create the real data.
   it will use abc to give the power or standard deviation of the parameters that is predicted by nn to know how
   much we are sure about the results. mainly it will do two parts of abc. one cv error and parameter estimation

   :param ssfile:  the real ss file path
   :param test_size:  the number of test rows. everything else will be used for train. 10k is default
   :param tol: the level of tolerance for abc. default is .005
   :param method: to tell which method is used in abc. default is mnlogistic. but can be rejection, neural net etc.
        as documented in the r.abc
   :param csvout: in case of everything satisfied. this will output the test dataset in csv format. can be used
        later by r
   :param cvrepeats: the number of repeats will be used for CV calculations
   :param folder: to define the output folder. default is '' meaning current folder
   :return: will not return anything but will plot and print the parameters
   """

    def __new__(cls, ssfile: str, test_size: int = int(1e4), tol: float = .01, method: str = 'neuralnet',
                csvout: bool = False, cvrepeats: int = 100, folder: str = '', frac: float = 1.0) -> None:
        """
        This will call the wrapper function

       :param ssfile:  the real ss file path
       :param test_size:  the number of test rows. everything else will be used for train. 10k is default
       :param tol: the level of tolerance for abc. default is .005
       :param method: to tell which method is used in abc. default is mnlogistic. but can be rejection, neural net etc.
            as documented in the r.abc
       :param csvout: in case of everything satisfied. this will output the test dataset in csv format. can be used
            later by r
       :param cvrepeats: the number of repeats will be used for CV calculations
       :param folder: to define the output folder. default is '' meaning current folder
       :param frac: To multiply all the observed ss with some fraction. Important in case simulated data and observed
            data are not from same length. default is 1
       :return: will not return anything but will plot and print the parameters
        """
        cls.wrapper(ssfile=ssfile, test_size=test_size, tol=tol, method=method, csvout=csvout, cvrepeats=cvrepeats,
                    folder=folder, frac=frac)

    @classmethod
    def wrapper(cls, ssfile: str, test_size: int = int(1e4), tol: float = 0.01, method: str = 'neuralnet',
                csvout: bool = False, cvrepeats: int = 100, folder: str = '', frac: float = 1.0) -> None:
        """
        The wrapper to test how the training using ANN works. after training is done it will test on the test  data set
        to see the power and then use a real data set to show what most likely parameters can create the real data.
        it will use abc to give the power or standard deviation of the parameters that is predicted by nn to know how
        much we are sure about the results. mainly it will do two parts of abc. one cv error and parameter estimation

        :param ssfile:  the real ss file path
        :param test_size:  the number of test rows. everything else will be used for train. 10k is default
        :param tol: the level of tolerance for abc. default is .005
        :param method: to tell which method is used in abc. default is mnlogistic. but can be rejection, neural net etc.
            as documented in the r.abc
        :param csvout: in case of everything satisfied. this will output the test data set in csv format. can be used
            later by r
        :param cvrepeats: the number of repeats will be used for CV calculations
        :param folder: to define the output folder. default is '' meaning current folder
        :param frac: To multiply all the observed ss with some fraction. Important in case simulated data and observed
            data are not from same length. default is 1
        :return: will not return anything but will plot and print the parameters
        """
        folder = Misc.creatingfolders(folder)
        ModelParamPrediction, x_test, y_test, scale_x, scale_y = ABC_DLS_Params_CV.read_data(test_rows=test_size,
                                                                                             folder=folder)
        cls.wrapper_aftertrain(ModelParamPrediction=ModelParamPrediction, ssfile=ssfile, x_test=x_test, y_test=y_test,
                               scale_x=scale_x, scale_y=scale_y, paramfile='params_header.csv', method=method, tol=tol,
                               csvout=csvout, cvrepeats=cvrepeats, folder=folder, frac=frac)


# SMC stuff
class ABC_DLS_SMC(ABC_DLS_Params):
    """
    This is the main class  for ABC_DLS_NS SMC. with given model underlying parameters it will compare
    with real data and will predict minima and maxima with in the parameter range can be for real data

    :param info: the path of info file whose file column is the path of the file and second column defining the
        number of  parameters
    :param ssfile: the summary statistic on real data set. should be csv format
    :param chunksize: the number of rows accessed at a time.
    :param test_size: the number of test rows. everything else will be used for train. 10k is default
    :param tol: the level of tolerance for abc. default is .005
    :param method: to tell which method is used in abc. default is mnlogistic. but can be rejection, neural net etc.
        as documented in the r.abc
    :param nn: custom function made for keras model. the path of that .py file. should have a def
        ANNModelCheck
    :param scaling_x: to tell if the x (ss) should be scaled or not. default is false. will be scaled by
        MinMaxscaler.
    :param scaling_y: to tell if the y (parameters) should be scaled or not. default is false. will be scaled by
        MinMaxscaler.
    :param csvout: in case of everything satisfied. this will output the test dataset in csv format. can be used
        later by r
    :param folder: to define the output folder. default is '' meaning current folder
    :param decrease: minimum amount of decreaseing of range needed to register as true. default is .95.
    :param frac: To multiply all the observed ss with some fraction. Important in case simulated data and observed
            data are not from same length. default is 1
    :param increase: the amount of fraction for increasing of range from the distance between  params[min]-params[max]
    :param hardrange_file: csv format of hardrange file path. Should have 3 columns. params_names, lower and upper
        limit. every row is define a parameters. no header. same as Newrange.csv. important when used
        increase as not go awry for simulation parameters
    :return: will return the new range in pandas dataframe format as well as create Narrowed.csv which will keep the
        simulations which are within that new range
    """
    info: str
    ssfile: str
    chunksize: Optional[int] = None
    test_size: int = int(1e4)
    tol: float = .005
    method: str = 'rejection'
    nn: Optional[str] = None
    scaling_x: bool = False
    scaling_y: bool = False
    csvout: bool = False
    folder: str = ''
    decrease: float = 0.95
    frac: float = 1.0

    def __new__(cls, info: str, ssfile: str, chunksize: Optional[int] = None, test_size: int = int(1e4),
                tol: float = .005, method: str = 'rejection', nn: Optional[str] = None, scaling_x: bool = False,
                scaling_y: bool = False, csvout: bool = False, folder: str = '', decrease: float = 0.95,
                frac: float = 1.0, increase: float = 0.0,
                hardrange_file: Optional[str] = None) -> pandas.DataFrame:
        """
        This will call the wrapper function

        :param info: the path of info file whose file column is the path of the file and second column defining the
            number of  parameters
        :param ssfile: the summary statistic on real data set. should be csv format
        :param chunksize: the number of rows accessed at a time.
        :param test_size: the number of test rows. everything else will be used for train. 10k is default
        :param tol: the level of tolerance for abc. default is .005
        :param method: to tell which method is used in abc. default is mnlogistic. but can be rejection, neural net etc.
            as documented in the r.abc
        :param nn: custom function made for keras model. the path of that .py file. should have a def
            ANNModelCheck
        :param scaling_x: to tell if the x (ss) should be scaled or not. default is false. will be scaled by
            MinMaxscaler.
        :param scaling_y: to tell if the y (parameters) should be scaled or not. default is false. will be scaled by
            MinMaxscaler.
        :param csvout: in case of everything satisfied. this will output the test dataset in csv format. can be used
            later by r
        :param folder: to define the output folder. default is '' meaning current folder
        :param decrease: minimum amount of decreasing of range needed to register as true. default is .95.
        :param frac: To multiply all the observed ss with some fraction. Important in case simulated data and observed
            data are not from same length. default is 1
        :param increase: the amount of fraction for increase of range from the distance between  params[min]-params[max]
        :param hardrange_file: csv format of hardrange file path. Should have 3 columns. params_names, lower and upper
            limit. every row is define a parameters. no header. same as Newrange.csv. important when used
            increase as not go awry for simulation parameters
        :return: will return the new range in pandas dataframe format as well as create Narrowed.csv which will keep the
            simulations which are within that new range
        """
        return cls.wrapper(info=info, ssfile=ssfile, chunksize=chunksize, test_size=test_size, tol=tol, method=method,
                           nn=nn, scaling_x=scaling_x, scaling_y=scaling_y, csvout=csvout,
                           folder=folder, decrease=decrease, frac=frac, increase=increase,
                           hardrange_file=hardrange_file)

    @classmethod
    def wrapper(cls, info: str, ssfile: str, chunksize: Optional[int] = None, test_size: int = int(1e4),
                tol: float = .005, method: str = 'rejection', nn: Optional[str] = None, scaling_x: bool = False,
                scaling_y: bool = False, csvout: bool = False, folder: str = '', decrease: float = 0.95,
                frac: float = 1.0, increase: float = 0.0,
                hardrange_file: Optional[str] = None) -> pandas.DataFrame:
        """
        The main wrapper for ABC_DLS_NS neseted sampling. with given model underlying parameters it will compare with
        real data and will predict minima and maxima with in the parameter range can be for real data

        :param info: the path of info file whose file column is the path of the file and second column defining the
            number of  parameters
        :param ssfile: the summary statistic on real data set. should be csv format
        :param chunksize: the number of rows accessed at a time.
        :param test_size: the number of test rows. everything else will be used for train. 10k is default
        :param tol: the level of tolerance for abc. default is .005
        :param method: to tell which method is used in abc. default is mnlogistic. but can be rejection, neural net etc.
            as documented in the r.abc
        :param nn: custom function made for keras model. the path of that .py file. should have a def
            ANNModelCheck
        :param scaling_x: to tell if the x (ss) should be scaled or not. default is false. will be scaled by
            MinMaxscaler.
        :param scaling_y: to tell if the y (parameters) should be scaled or not. default is false. will be scaled by
            MinMaxscaler.
        :param csvout: in case of everything satisfied. this will output the test dataset in csv format. can be used
            later by r
        :param folder: to define the output folder. default is '' meaning current folder
        :param decrease: minimum amount of decreasing of range needed to register as true. default is .95.
        :param increase: the amount of fraction for increase of range from the distance between  params[min]-params[max]
        :param hardrange_file: csv format of hardrange file path. Should have 3 columns. params_names, lower and upper
            limit. every row is define a parameters. no header. same as Newrange.csv. important when used
            increase as not go awry for simulation parameters
        :param frac: To multiply all the observed ss with some fraction. Important in case simulated data and observed
            data are not from same length. default is 1
        :return: will return the new range in pandas dataframe format as well as create Narrowed.csv which will keep the
            simulations which are within that new range
        """
        folder = Misc.creatingfolders(folder)
        x_train, x_test, scale_x, y_train, y_test, scale_y, paramfile = cls.wrapper_pre_train(info=info,
                                                                                              chunksize=chunksize,
                                                                                              test_size=test_size,
                                                                                              scaling_x=scaling_x,
                                                                                              scaling_y=scaling_y,
                                                                                              folder=folder)

        ModelParamPrediction = cls.wrapper_train(x_train=(x_train, x_test), y_train=(y_train, y_test),
                                                 nn=nn, folder=folder)

        return cls.wrapper_aftertrain(ModelParamPrediction=ModelParamPrediction, x_test=x_test, y_test=y_test,
                                      ssfile=ssfile, scale_x=scale_x, scale_y=scale_y, info=info, csvout=csvout,
                                      paramfile='params_header.csv', method=method, tol=tol, folder=folder,
                                      decrease=decrease,
                                      frac=frac, increase=increase, hardrange_file=hardrange_file)

    @classmethod
    def ANNModelParams(cls, x: Tuple[Union[numpy.ndarray, HDF5Matrix], Union[numpy.ndarray, HDF5Matrix]],
                       y: Tuple[
                           Union[numpy.ndarray, HDF5Matrix], Union[numpy.ndarray, HDF5Matrix]]) -> keras.models.Model:
        """
        A basic model for ANN to calculate parameters. to make it more efficient train and test together. but make it
        more memory inefficient

        :param x: the x_train and x_test of summary statistics. can be numpy array or hdf5 of tuple
        :param y: the x_train and x_test of the parameters which produced those ss. can be numpy array or hdf5 of tuple
        :return: will return the trained model
        """

        x_train, x_test = x
        y_train, y_test = y

        model = Sequential()
        model.add(GaussianNoise(0.05, input_shape=(x_train.shape[1],)))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(y_train.shape[1]))

        model.compile(loss='logcosh', optimizer='Nadam', metrics=['accuracy'])
        # adding an early stop so that it does not overfit
        ES = EarlyStopping(monitor='val_loss', patience=100)
        # checkpoint
        CP = ModelCheckpoint('Checkpoint.h5', verbose=0, save_best_only=True)
        # Reduce learning rate
        RL = ReduceLROnPlateau(factor=0.2)

        model.fit(x_train, y_train, epochs=int(2e6), verbose=0, shuffle=True, callbacks=[ES, CP, RL],
                  validation_data=(numpy.array(x_test), numpy.array(y_test)))

        return model

    @classmethod
    def wrapper_aftertrain(cls, info: str, ModelParamPrediction: keras.models.Model,
                           x_test: Union[numpy.ndarray, HDF5Matrix],
                           y_test: Union[numpy.ndarray, HDF5Matrix], ssfile: str,
                           scale_x: Optional[preprocessing.MinMaxScaler], scale_y: Optional[preprocessing.MinMaxScaler],
                           paramfile: str = 'params_header.csv', method: str = 'rejection', tol: float = .005,
                           folder: str = '', csvout: bool = False, decrease: float = 0.95,
                           frac: float = 1.0, increase: float = 0.0,
                           hardrange_file: Optional[str] = None) -> pandas.DataFrame:
        """
        The wrapper to test how the training using ANN works. after training is done it will test on the test  data set
        to see the power and then use a real data set to show what most likely parameters can create the real data.
        it will use abc to give the lower minima and upper maxima that can create observed SFS. This is the main
        difference from parameter estimation part of ABC-DLS. Other part before are ver similar. It will also produce
        a narrowed csv so that the models ,which are with in the new limit, can be reused.

        :param info: the path of info file whose file column is the path of the file and second column defining the
            number of  parameters
        :param ModelParamPrediction: The fitted keras model
        :param x_test: the test part of x aka summary statistics
        :param y_test: the test part of y aka parameter dataset
        :param ssfile: the real ss file path
        :param scale_x: the scale of ss if exist
        :param scale_y: the scale of parameters if exist
        :param paramfile: the path of parameter header file. default is 'params_header.csv'
        :param method: the method to be used in r_abc. default is rejection
        :param tol: the tolerance level to be used in r_abc. default is .005
        :param folder: to define the output folder. default is '' meaning current folder
        :param csvout: n case of everything satisfied. this will output the test dataset in csv format. can be used
            later by r
        :param decrease: minimum amount of decreasing of range needed to register as true. default is .95.
        :param increase: the amount of fraction for increase of range from the distance between  params[min]-params[max]
        :param hardrange_file: csv format of hardrange file path. Should have 3 columns. params_names, lower and upper
            limit. every row is define a parameters. no header. same as Newrange.csv. important when used
            increase as not go awry for simulation parameters
        :param frac: To multiply all the observed ss with some fraction. Important in case simulated data and observed
            data are not from same length. default is 1
        :return: will return the new range in pandas dataframe format as well as create Narrowed.csv which will keep the
            simulations which are within that new range
        """

        Misc.removefiles([folder + 'Narrows.csv', folder + 'Narrowed.csv'], printing=False)
        params_names = pandas.read_csv(folder + paramfile).columns
        ss = cls.read_ss_2_series(file=ssfile)
        ss = ss * numpy.array(frac)
        test_predictions, predict4mreal, params_unscaled = cls.preparing_for_abc(
            ModelParamPrediction=ModelParamPrediction, x_test=x_test, y_test=y_test, scale_x=scale_x, scale_y=scale_y,
            params_names=params_names, ss=ss)
        parmin, parmax = cls.abc_params_min_max(target=predict4mreal, param=params_unscaled, ss=test_predictions,
                                                tol=tol, method=method)
        newrange = pandas.concat([parmin, parmax], axis=1)
        newrange.index = params_names
        newrange.columns = ['min', 'max']
        params = cls.extracting_params(variable_names=params_names, scale_y=scale_y, yfile=folder + 'y.h5')
        oldrange = pandas.concat([params.min(), params.max()], axis=1)
        oldrange.columns = ['min', 'max']
        newrange = cls.updating_newrange(newrange=newrange, oldrange=oldrange, decrease=decrease)
        if increase > 0:
            if hardrange_file:
                hardrange = pandas.read_csv(hardrange_file, index_col=0, header=None, names=['', 'min', 'max'],
                                            usecols=[0, 1, 2])
            else:
                hardrange = pandas.DataFrame()
            newrange = cls.noise_injection_update(newrange=newrange, increase=increase, hardrange=hardrange,
                                                  oldrange=oldrange, decrease=decrease)
            if hardrange_file:
                lmrd = cls.lmrd_calculation(newrange=newrange, hardrange=hardrange)
                print('log of mean range decrease:', lmrd)
        newrange.to_csv(folder + 'Newrange.csv', header=False)
        if csvout:
            _ = cls.narrowing_input(info=info, params=params, newrange=newrange, folder=folder)
        return newrange

    @classmethod
    def abc_params_min_max(cls, target: pandas.DataFrame, param: pandas.DataFrame, ss: pandas.DataFrame,
                           tol: float = .01, method: str = "rejection") -> [pandas.Series, pandas.Series]:
        """
        Calculating the min max range for parameters using ABC

        :param target: the real ss in a pandas  data frame format
        :param param: the parameter data frame format (y_test)
        :param ss: the summary statics in dataframe format
        :param tol: the tolerance level. default is .001
        :param method: the method to calculate abc cv error. can be "rejection", "loclinear", and "neuralnet". default
            rejection
        :return: will return min and max ranges in pandas Series format
        """
        parmin = []
        parmax = []
        if method == 'loclinear' or method == 'rejection':
            for colnum in range(param.shape[1]):
                res = abc.abc(target=pandas.DataFrame(target.iloc[:, colnum]),
                              param=pandas.DataFrame(param.iloc[:, colnum]),
                              sumstat=pandas.DataFrame(ss.iloc[:, colnum]),
                              method=method, tol=tol)
                mincol, maxcol = cls.r_summary_min_max_single(res)
                parmin.append(mincol)
                parmax.append(maxcol)
            parmin = pandas.Series(numpy.hstack(parmin))
            parmax = pandas.Series(numpy.hstack(parmax))
        elif method == 'neuralnet':
            print('Together')
            res = abc.abc(target=target, param=param, sumstat=ss, method=method, tol=tol)
            parmin, parmax = cls.r_summary_min_max_all(res)
        return parmin, parmax

    @classmethod
    def r_summary_min_max_single(cls, rmodel) -> Tuple[float, float]:
        """
        reading summary statistics coming from r and converting it for pandas. this is for per parameters

        :param rmodel: the rmodel of abc
        :return: will return min and maxima of the parameter in tuple format
        """
        import tempfile
        temp_name = next(tempfile._get_candidate_names())
        robjects.r.options(width=10000)
        robjects.r['sink'](temp_name)
        singlemin = robjects.r['summary'](rmodel)[0]
        robjects.r['sink']()
        robjects.r.options(width=10000)
        robjects.r['sink'](temp_name)
        singlemax = robjects.r['summary'](rmodel)[6]
        robjects.r['sink']()
        os.remove(temp_name)
        return singlemin, singlemax

    @classmethod
    def r_summary_min_max_all(cls, rmodel) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
        """
        reading summary statistics coming from r and converting it for pandas. this is for all parameters together

        :param rmodel: the rmodel of abc
        :return: will return min and max of the parameter in pandas.DataFrame format
        """
        import tempfile
        temp_name = next(tempfile._get_candidate_names())
        robjects.r.options(width=10000)
        robjects.r['sink'](temp_name)
        min_df = pandas.DataFrame(numpy.array(robjects.r['summary'](rmodel))).iloc[0, :]
        robjects.r['sink']()
        robjects.r.options(width=10000)
        robjects.r['sink'](temp_name)
        max_df = pandas.DataFrame(numpy.array(robjects.r['summary'](rmodel))).iloc[-1, :]
        robjects.r['sink']()
        os.remove(temp_name)
        return min_df, max_df

    @classmethod
    def extracting_params(cls, variable_names: List, scale_y: Optional[preprocessing.MinMaxScaler] = None,
                          yfile: str = 'y.h5') -> pandas.DataFrame:
        """
        re-extracting (or re-transform) the parameters from y.h5 so that we can reuse it later for narrow

        :param variable_names: the name of the variables in list format
        :param scale_y: the minmax scale on parameters
        :param yfile: the y.h5 file path
        :return: will return the parameters in pandas DataFrame format with rescaled back
        """
        y, _ = cls.train_test_split_hdf5(yfile, test_rows=0)
        if scale_y:
            params = pandas.DataFrame(scale_y.inverse_transform(y[:]), columns=variable_names[-y.shape[1]:])
        else:
            params = pandas.DataFrame(y[:], columns=variable_names[-y.shape[1]:])
        return params

    @classmethod
    def noise_injection_update(cls, newrange: pandas.DataFrame, oldrange: pandas.DataFrame, increase: float = 0.005,
                               hardrange: pandas.DataFrame = pandas.DataFrame(),
                               decrease: float = .95) -> pandas.DataFrame:
        """
        in case you want to use some noise injection to the newrange. important as sometime when ABC-DLS is working
        recursively by chance it misses the true values. By using this noise injection you broaden up the upper and
        lowerlimit in a fraction manner, thus even if it misses the true result in a cycle it can come back to the real
        results in further cycle

        :param newrange: the new range in pandas dataframe format. columns should be max and min and indexes should be
            the parameters
        :param increase: the amount of fraction for increase of range from the distance between  params[min]-params[max]
        :param hardrange: the hard range in pandas dataframe format. columns should be max and min and indexes should be
            the parameters
        :param oldrange: the old range in pandas dataframe format. columns should be max and min and indexes should be
        the parameters
        :return: will return a newrange pandas dataframe which are with relaxed using the noise injection and then
            tested to be within hardrange
        """
        dist = (newrange['max'] - newrange['min']) * increase * 0.5
        newrange.loc[newrange['decrease'] > decrease, 'min'] = (newrange['min'] - dist).loc[
            newrange['decrease'] > decrease]
        newrange.loc[newrange['decrease'] > decrease, 'max'] = (newrange['max'] + dist).loc[
            newrange['decrease'] > decrease]
        if not hardrange.empty:
            newrange['min'] = pandas.concat([hardrange['min'], newrange['min']], axis=1).max(axis=1)
            newrange['max'] = pandas.concat([hardrange['max'], newrange['max']], axis=1).min(axis=1)
        newrange['decrease'] = (newrange['max'] - newrange['min']) / (oldrange['max'] - oldrange['min'])

        return newrange

    @classmethod
    def updating_newrange(cls, newrange: pandas.DataFrame, oldrange: pandas.DataFrame,
                          decrease: float = .95) -> pandas.DataFrame:
        """
        This will check if the new range decreasing is more than 95%. if true it will update the new range or else keep
        the old range assuming there is no direct decreasing. this step is necessary so that you do not get smaller
        range just because you ran it several time

        :param newrange: the new range in pandas dataframe format. columns should be max and min and indexes should be
            the parameters
        :param oldrange: the old range in pandas dataframe format. columns should be max and min and indexes should be
            the parameters
        :param decrease: the amount of decreasing required to update the new decreasing. default is 95%
        :return: will return the updated newrange dataframe. where if decrease is less than .95 then new range rows if not
            old range rows
        """
        newrange['decrease'] = (newrange['max'] - newrange['min']) / (oldrange['max'] - oldrange['min'])
        newrange.loc[(newrange['decrease'] > decrease) & (newrange['decrease'] < 1), ['min', 'max']] = oldrange[
            (newrange['decrease'] > decrease) & (newrange['decrease'] < 1)]
        newrange['decrease'] = (newrange['max'] - newrange['min']) / (oldrange['max'] - oldrange['min'])
        return newrange

    @classmethod
    def narrowing_input(cls, info: str, params: pandas.DataFrame, newrange: pandas.DataFrame, folder: str = '') -> str:
        """
        To reuse some simulations for next round. ABC will predict narrower posterior for parameters. No point rerun
        everything. rather use those simulations which are within the limit of new range.

        :param info: the path of info file whose file column is the path of the file and second column defining the
            number of  parameters
        :param params: the parameters coming from y.h5 rescaled. remember this step assumed that you do not randomized
            the line between csv and .h5 files. rather it will in this step
        :param newrange: the new range in pandas dataframe format. columns should be max and min and indexes should be
            the parameters
        :param folder: to define the output folder. default is '' meaning current folder
        :return: will return the path of 'Narrowed.csv'
        """
        linenumbers = cls.narrowing_params(params=params, parmin=newrange['min'], parmax=newrange['max'])
        inputfiles, _, _ = cls.read_info(info=info)
        temp = cls.extracting_by_linenumber(file=inputfiles[0], linenumbers=linenumbers,
                                            outputfile=folder + 'Narrows.csv')
        if Misc.getting_line_count(temp) > 0:
            _ = cls.shufling_joined_models(inputcsv=temp, output=folder + 'Narrowed.csv', header=False)
            Misc.removefiles([folder + 'Narrows.csv'], printing=False)
        else:
            os.rename(temp, folder + 'Narrowed.csv')
        return folder + 'Narrowed.csv'

    @classmethod
    def narrowing_params(cls, params: pandas.DataFrame, parmin: pandas.Series,
                         parmax: pandas.Series) -> pandas.core.indexes.range.RangeIndex:
        """
        narrowing the pandas params with new range

        :param params: pandas.DataFrame for the parameters
        :param parmin: pandas.Series about the minimum range of every parameters
        :param parmax: pandas.Series about the maximum range of every parameters
        :return: will return the lienumber+2 where we have ranges with in the new limit. use ful to directly extract
            from the csv file
        """

        for index in range(params.shape[1]):
            params = params[params.iloc[:, index].between(parmin[index], parmax[index])]
        linenumbers = params.index.values + 2
        return linenumbers

    @classmethod
    def extracting_by_linenumber(cls, file: str, linenumbers: Union[list, numpy.array],
                                 outputfile: str = 'out.txt') -> str:
        """
        given line numbers it will extract the lines from the text file

        :param file: the path of the text file
        :param linenumbers: the line numbers which you want to extract
        :param outputfile: the path of output file where you want to write
        :return: will return the path of output file
        """
        if file[-3:] == '.gz':
            os.system("zcat " + file + " > temp.csv ")
            file = 'temp.csv'
        linenumbers = linenumbers - 1  # as python start with 0
        output = open(outputfile, 'w')
        with open(file) as f:
            for lno, ln in enumerate(f):
                if lno in linenumbers:
                    print(ln, file=output, end='')
        output.close()
        Misc.removefiles(['temp.csv'], printing=False)
        return outputfile

    @classmethod
    def lmrd_calculation(cls, newrange: pandas.DataFrame, hardrange: pandas.DataFrame) -> float:
        """
        this will output log of mean range decrease for every cycle

        :param newrange:  the new and updated range
        :param hardrange: the old and hard range from where it all started
        :return: will return the lmrd
        """
        newrange_dist = (newrange.iloc[:, 1] - newrange.iloc[:, 0]).abs()
        hardrange_dist = (hardrange.iloc[:, 1] - hardrange.iloc[:, 0]).abs()
        lmrd = numpy.log((newrange_dist / hardrange_dist).mean())
        return lmrd
