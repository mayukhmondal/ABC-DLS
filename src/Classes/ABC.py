#!/usr/bin/python
"""
This file will hold all the classes for ABC
"""
import os
import subprocess
import sys

import h5py
import numpy
import pandas
###rpy2 stuff
from rpy2 import robjects
from rpy2.robjects import pandas2ri

# my stuff
from . import Misc

abc = Misc.importr_tryhard('abc')
pandas2ri.activate()

####Tensor flow stuff
from sklearn import preprocessing
import joblib
###to stop future warning every time to print out
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from tensorflow.python import keras
    from tensorflow.python.keras.models import Sequential
    from tensorflow.python.keras.layers import Dense
    from keras.utils import HDF5Matrix

##type hint for readability
from typing import List, Dict, Tuple, Optional, Callable, Union


class ABC_TFK_Classification():
    """
    Main classification class. It will distinguish between different models. with given underlying models it will
    compare with real data and will predict how much it sure about which model can bet predict the real data.
    :param info: the path of info file whose file column is the path of the file and second column defining the
        number of  parameters
    :param ssfile: the summary statisfic on real data set. should be csv format
    :param demography: custom function made for keras model. the path of that .py file. shoul have a def
    ANNModelCheck
    :param method: to tell which method is used in abc. default is mnlogitic. but can be rejection, neural net etc.
    as documented in the r.abc
    :param tolerance: the level of tolerance for abc. default is .005
    :param test_size:  the number of test rows. everything else will be used for train. 10k is default
    :param chunksize:  the number of rows accessed at a time.
    :param scale: to tell if the data should be scaled or not. default is false. will be scaled by MinMaxscaler.
    The scaling will only happen on the ss.
    :param csvout:  in case of everything satisfied. this will output the test dataset in csv format. can be used
    later by r
    :return: will not return anything but will plot and print the power
    """

    def __new__(cls, info, ssfile, demography=None, method="mnlogistic", tolerance=.001, test_size=int(1e4),
                chunksize=int(1e4), scale=False, csvout=False):
        return cls.wrapper(info=info, ssfile=ssfile, demography=demography, method=method, tolerance=tolerance,
                           test_size=test_size, chunksize=chunksize, scale=scale, csvout=csvout)

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
        the file. If two lines it will assumme first line is header. If three lines it will assume it is momnets or dadi
        related fs fomat.
        :param file: the path of the sfs file
        :return: will return the sfs in series format
        """
        count = Misc.getting_line_count(file)
        if count == 1:
            ###without header format
            ss = pandas.read_csv(file, header=None).transpose()
        elif count == 2:
            ##with header format
            ss = pandas.read_csv(file).transpose()
        elif count == 3:
            ##dadi or moments format
            ss = pandas.read_csv(file, skiprows=1, nrows=2, header=None, sep=' ').transpose()
        else:
            print('Cant understand the format of the input sfs file. Please check')
            sys.exit(1)
        return ss[0]

    @classmethod
    def check_results(cls, results: List[pandas.DataFrame], observed: pandas.Series) -> None:
        """
        To check every thing is ok with the  results and observed values. for example if two files of results have
        different rows it will take the lower minima. as you need all the model with same rows of repetition
        :param results: the results list with differernt model result in dataframe format
        :param observed: the observed results
        :return: will not retirnm anything. but in case of problem will stop
        """
        result_columns = [result.shape[1] for result in results]
        if len(set(result_columns + [observed.shape[0]])) > 1:
            print("the observed columns and/or result columns do no match. check")
            print("result_columns:", result_columns)
            print("observed_columns", observed.shape[0])
            sys.exit(1)

    @classmethod
    def subsetting_file_concating(cls, filename: str, params_number: int, nrows: int, modelname: str) -> None:
        """
        to read the msprime sfs out files. which has both params and sfs together. and then prepare for merging.
        this will remove the params columns (As not needed for model comparison) and make same number of rows per model
        so that it is not biased. also add the very first column as the name of the model
        :param filename: the csv file path. whose first columns are paramteres and all the rest are sfs or ss. comma
        separated
        :param params_number: the number of paratmetes presnt in the file. rest are sfs or ss
        :param nrows: the number of rows to include in the file. remember the nrows start with 0. thus if you want to
        include 100 line put 99. the first line is header thus ignored
        :param modelname:name of the model. string
        :return:will not return anything but will create or update Comparison.csv file .where all the ss with the name
        of the models are written
        """

        command = Misc.joinginglistbyspecificstring(
            ["zcat", filename, '|', 'cut -f' + str(params_number + 1) + '-', '-d ","',
             '''|awk '$0="''' + modelname + ""","$0'""", "|tail -n+2", "|head -n", nrows, " >> Comparison.csv"])
        p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        if stderr:
            print(stderr)
            sys.exit(1)
        return None

    @classmethod
    def shufling_joined_models(cls, input: str = "Comparison.csv", output: str = 'shuffle.csv',
                               header: bool = True) -> str:
        """
        it will shuffle the line of joined csv model (file Comparison.csv) and read it in pandas format for further
        evaluation
        :param input: the joined csv file (Comparison.csv) with ss and model names default:"Comparison.csv"
        :param output: the shuffled csv file path. default: 'shuffle.csv'
        :param header: if the header should be kept or not. default is true
        :return: will return output which is the shuffled rows of input
        """
        import os
        terashuf = os.path.dirname(os.path.abspath(__file__)) + '/shuffle.py'
        Misc.creatingfolders('temp')
        if header:
            if os.path.exists(terashuf):
                command = Misc.joinginglistbyspecificstring(['cat <(head -n 1', input, ') <(tail -n+2', input,
                                                             ' | python ' + terashuf + ' ) > ',
                                                             os.getcwd() + '/' + output]).strip()
            else:
                print(
                    "terashuf is not found. will use shuf. in case out of memory. please install it and put it in the calsses folder")
                command = Misc.joinginglistbyspecificstring(
                    ['cat <(head -n 1', input, ') <(tail -n+2', input, ' | shuf ) > ',
                     os.getcwd() + '/' + output]).strip()
        else:
            if os.path.exists(terashuf):
                command = Misc.joinginglistbyspecificstring(
                    ['python ', terashuf, input, ">", output])
            else:
                print(
                    "terashuf is not found. will use shuf. in case out of memory. please install it and put it in the calsses folder")

                command = Misc.joinginglistbyspecificstring(["shuf ", input, ">", output])
        p = subprocess.Popen([command], executable='/bin/bash', stdout=subprocess.PIPE, shell=True,
                             stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        # print (command)
        if stderr:
            print(stderr)
            sys.exit(1)
        os.rmdir('temp')
        return output

    @classmethod
    def data_prep4ANN(cls, raw: pandas.DataFrame, test_size: int = int(1e4), scale: bool = False) -> Tuple[
        numpy.array, numpy.array, numpy.array, numpy.array, Optional[preprocessing.MinMaxScaler], Dict[int, str]]:
        """
        Preparing data for NN classification method. The MinMaxScaler is used to normalize in case normalization needed
        :param raw: raw summary statistics dataframe.
        :param test_size: the number of test rows. everything else will be used for train. 10k is default
        :param scale: if the raw data should be scaled or not. default is false. will be saled by MinMaxscaler
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
        Misc.numpy2hdf5(x_test, 'x_test.h5')
        Misc.numpy2hdf5(y_test, 'y_test.h5')
        if scale_x:
            joblib.dump(scale_x, "scale_x.sav")
        return x_train, x_test, y_train, y_test, scale_x, y_cat_dict

    @classmethod
    def MinMax4bigfile(cls, csvpath: str, h5path: str = 'temp.h5', scaling: bool = True,
                       expectedrows: Optional[int] = None, chunksize: int = 10, verbose: bool = False,
                       special_func: Optional[Callable] = None, **kwargs) -> Optional[preprocessing.MinMaxScaler]:
        """
        This to to convert a very big csv file which cannot be put inside ram to save it in the h5 format and use the
        minmax scaler to scale the data
        :param csvpath: the path of ss csv file. can be zipped
        :param h5path:output of h5path. default is temp.h5
        :param scaling: to tell if the data should be scaled or not. default is false. will be scaled by MinMaxscaler
        :param expectedrows: the number of rows is expected for the data output. if not given it will calculate from
        the csv file
        :param chunksize: the number of rows can be accessed at a time. if memory error put lower number. default is 10
        :param verbose: in case to check the progression put verbose true. default false
        :param special_func: in case some kind of special function has to be done before the min max scaler
        :param kwargs:all the related parameters for special_func
        :return:will return the min max scaler which can be used later. or none. it will save h5path the whole data
        """
        ###the scaling part
        if scaling:
            scale = preprocessing.MinMaxScaler()
            row = 0
            if verbose:
                print('scaling')
            for chunk in pandas.read_csv(csvpath, chunksize=chunksize, memory_map=True, low_memory=False):
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
            chunk = pandas.read_csv(csvpath, nrows=chunksize, memory_map=True, low_memory=False)
            if special_func:
                if verbose:
                    print(special_func(chunk, **kwargs).values)
                newss = special_func(chunk, **kwargs).values

        ###intializing the hdf5 part
        if expectedrows is None:
            expectedrows = Misc.getting_line_count(csvpath) - 1
        if special_func:
            expectedcolumns = newss.shape[1]
        else:
            expectedcolumns = chunk.shape[1]

        arraysize = (expectedrows, expectedcolumns)

        f = h5py.File(h5path, 'w')
        transh5 = f.create_dataset('mydata', arraysize, chunks=True)

        ###transorming
        if verbose:
            print('transforming')
        row = 0
        for chunk in pandas.read_csv(csvpath, chunksize=chunksize, memory_map=True, low_memory=False):
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
    def train_test_split_hdf5(cls, path: str, dataset: str = 'mydata', test_rows: int = int(1e4)) -> Tuple[
        HDF5Matrix, HDF5Matrix]:
        """
        Special way to train test split for hdf5. will take the first n-test_rows for training and rest for test
        :param path:the path of .h5 file
        :param dataset: the name of the dataset of h5py file. default 'mydata'
        :param test_rows:the number of rows for test every thing will be left for training. default is 10k
        :return:will return the test, train split for hdf5
        """
        rows = HDF5Matrix(path, dataset).shape[0]
        train = HDF5Matrix(path, dataset, start=0, end=rows - test_rows)
        test = HDF5Matrix(path, dataset, start=rows - test_rows, end=rows)
        return train, test

    @classmethod
    def preparingdata_hdf5(cls, filename: str, chunksize: int, test_size: int = int(1e4), scale: bool = False) -> Tuple[
        HDF5Matrix, HDF5Matrix, HDF5Matrix, HDF5Matrix, Optional[preprocessing.MinMaxScaler], Dict[int, str]]:
        """
        In case of chunk size is mentioned it will be assumed that the data is too big to save in ram and it will be
        saved in hdf5 format and will be split it in necessary steps
        :param filename: the file path of csv where the first column is the models name and rest are ss. every row
        different
        simulation. header included and shuffled data. output of shufling_joined_models
        :param chunksize: the number of rows accessed at a time.
        :param test_size: the number of test rows. everything else will be used for train. 10k is default
        :param scale: to tell if needed to be scaled or not. by default is false. if true will also save scale_x.sav to
        be used later if needed
        :return: will return train and test data fro both x and y
        """

        xfile = "x.h5"
        yfile = "y.h5"
        Misc.removefiles([xfile, yfile])
        ss_command = Misc.joinginglistbyspecificstring(["cut -f 2- ", '-d ","', filename, "> ss.csv"])
        os.system(ss_command)
        if scale:
            scale_x = cls.MinMax4bigfile(csvpath='ss.csv', h5path=xfile, chunksize=chunksize)
        else:
            scale_x = cls.MinMax4bigfile(csvpath='ss.csv', h5path=xfile,
                                         chunksize=chunksize, scaling=False)
        x_train, x_test = cls.train_test_split_hdf5(xfile, test_rows=int(test_size))

        if scale_x:
            joblib.dump(scale_x, "scale_x.sav")
        models_command = Misc.joinginglistbyspecificstring(["cut -f 1 ", '-d ","', filename, "> models.csv"])
        os.system(models_command)
        model_index = pandas.read_csv('models.csv')
        y_cat_dict = dict(zip(pandas.Categorical(model_index.iloc[:, 0]).codes, model_index.iloc[:, 0]))
        y = keras.utils.to_categorical(pandas.Categorical(model_index.iloc[:, 0]).codes, len(y_cat_dict))
        Misc.numpy2hdf5(y, yfile)
        y_train, y_test = cls.train_test_split_hdf5(yfile, test_rows=int(test_size))
        Misc.removefiles(['ss.csv', 'models.csv'])
        return x_train, x_test, y_train, y_test, scale_x, y_cat_dict

    @classmethod
    def wrapper_pre_train(cls, info: str, test_size: int = int(1e4), chunksize: Optional[int] = int(1e4),
                          scale: bool = False) -> \
            Tuple[
                Union[numpy.array, HDF5Matrix], Union[numpy.array, HDF5Matrix], Union[numpy.array, HDF5Matrix], Union[
                    numpy.array, HDF5Matrix], Optional[preprocessing.MinMaxScaler], Dict[int, str]]:
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
        :return: will return data needed for training. will return x_(train,test), y_(train,test) scale_x (MinMaxScaler
        or None) and y_cat_dict ({0:'model1',1:'model2'..})
        """
        Misc.removefiles(
            ['scale_x.sav', 'scale_y.sav', 'x_test.h5', 'y_test.h5', 'y.h5', 'x.h5', 'ModelClassification.h5',
             'Comparison.csv', 'shuf.csv', 'models.csv', 'ss.csv', 'y_cat_dict.txt'])
        files, paramnumbers, names = cls.read_info(info=info)
        minlines = min([Misc.getting_line_count(file) for file in files]) - 1
        pandas.DataFrame(
            ['models'] + list(pandas.read_csv(files[0], nrows=10).columns[paramnumbers[0]:])).transpose().to_csv(
            'Comparison.csv', index=False, header=False)
        [cls.subsetting_file_concating(filename=files[i], params_number=paramnumbers[i], nrows=minlines,
                                       modelname=names[i]) for i in range(len(files))]
        shuffile = cls.shufling_joined_models(input='Comparison.csv', output='shuf.csv')

        if chunksize:
            x_train, x_test, y_train, y_test, scale_x, y_cat_dict = cls.preparingdata_hdf5(filename=shuffile,
                                                                                           chunksize=chunksize,
                                                                                           test_size=test_size,
                                                                                           scale=scale)
            f = open("y_cat_dict.txt", "w")
            f.write(str(y_cat_dict))
            f.close()

        else:
            results = pandas.read_csv(shuffile, index_col=0)
            x_train, x_test, y_train, y_test, scale_x, y_cat_dict = cls.data_prep4ANN(results, test_size=test_size,
                                                                                      scale=scale)
        Misc.removefiles(['Comparison.csv', shuffile])

        return x_train, x_test, y_train, y_test, scale_x, y_cat_dict

    @classmethod
    def ANNModelCheck(cls, x: Union[numpy.array, HDF5Matrix], y: Union[numpy.array, HDF5Matrix]) -> keras.models.Model:
        """
        The Tensor flow for model check
        :param x: the x or summary statistics. can be numpy array or hdf5.
        :param y: the y or model names or classification. can be numpy array of hdf5
        :return: will return the trained model
        """
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(x.shape[1],)))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(y.shape[1], activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
        model.fit(x, y, epochs=500, verbose=2, shuffle="batch")

        return model

    @classmethod
    def wrapper_train(cls, x_train: Union[numpy.array, HDF5Matrix], y_train: Union[numpy.array, HDF5Matrix],
                      demography: Optional[str] = None) -> keras.models.Model:
        """
        This the wrapper for training part of the classification method. it need trainging data set for x and y. can be
        either numpy array or hdf5 matrix format (HD5matrix) of keras
        Misc.loading_def_4m_file -> def ANNModelCheck
        :param x_train: train part of x aka summary statistics
        :param y_train:traing part of y aka models names. should be used keras.utils.to_categorical to better result
        :param demography: custom function made for keras model. the path of that .py file. shoul have a def
        ANNModelCheck
        :return:will return the keras model. it will also save the model in ModelClassification.h5
        """
        if demography:
            ANNModelCheck = Misc.loading_def_4m_file(filepath=demography, defname='ANNModelCheck')
            if ANNModelCheck:
                ModelSeparation = ANNModelCheck(x=x_train, y=y_train)
            else:
                ModelSeparation = cls.ANNModelCheck(x=x_train, y=y_train)
        else:
            ModelSeparation = cls.ANNModelCheck(x=x_train, y=y_train)
        ModelSeparation.save("ModelClassification.h5")
        return ModelSeparation

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
        robjects.r['sink']('temp.txt')
        robjects.r['summary'](rmodel)
        robjects.r['sink']()
        cls.print_after_match_linestart('temp.txt', target)
        os.remove('temp.txt')

    @classmethod
    def plot_power_of_ss(cls, ss: pandas.DataFrame, index: pandas.Series, tol: float = .005, repeats: int = 100,
                         method: str = "mnlogistic") -> None:
        """
        now to test the power of summary statistics using r_abc
        :param ss: the summary statics in dataframe format
        :param index: the index of the models
        :param tol: the level of tolerance. default is .005
        :param repeats: the number of repeat to use is on cv (cross validation) error. default is 100
        :param method: to tell which method is used in abc. default is mnlogitic. but can be rejection, neural net etc.
        as documented in the r.abc
        :return: will plot it the folder to see the confusion matrix. also print out the summary of the model to see the
         confusion matrix in text format
        """

        cvmodsel = abc.cv4postpr(index=index, sumstat=ss, nval=repeats, tol=tol, method=method)
        # cls.r_summary(cvmodsel,target='Confusion')
        x, y = robjects.r['summary'](cvmodsel)
        robjects.r['plot'](cvmodsel)

    @classmethod
    def model_selection(cls, target: pandas.DataFrame, ss: pandas.DataFrame, index: pandas.Series, tol: float = .005,
                        method: str = "mnlogistic") -> None:
        """
        As the name suggest. Given the number of model it will select correct model using postpr in abc
        :param target: the observed summary statistic i n a dataframe format with only one line.
        :param ss: the summary statics in dataframe format
        :param index:  the index of the models in pandas.Series format
        :param tol: the level of tolerance. default is .005
        :param method: to tell which method is to be used in abc. default is mnlogitic. but can be rejection, neural net
         etc. as docuemented in the r.abc
        :return: will not return anything but will print summary of model selection
        """

        modsel = abc.postpr(target=target, index=index,
                            sumstat=ss, tol=tol, method=method)
        cls.r_summary(modsel)
        return None

    @classmethod
    def goodness_fit(cls, target: pandas.DataFrame, ss: pandas.DataFrame, name: str, tol: float = .005,
                     extra: str = ''):
        """
        To test for goodness of fit of every model
        :param target:  the observed summary statistic in a dataframe format with single line
        :param ss:  the simulated summary statics in dataframe format.
        :param name: name of the demography
        :param extra: internal. to add in the graph to say about the method
        :return: will not return anything. but will plot goodness of fit also print the summary
        """
        fit = abc.gfit(target, ss, 100, tol=tol)
        print(name)
        print(robjects.r['summary'](fit))
        out = name + ' ' + extra
        robjects.r['plot'](fit, main="Histogram under H0:" + out)

    @classmethod
    def gfit_all(cls, observed: pandas.DataFrame, ss: pandas.DataFrame, y_cat_dict: Dict[int, str], extra: str = '',
                 tol: float = .005) -> None:
        """
        wrapper of goodness of fit. different goodness of fit for different models
        :param observed: the observed summary statistic in a in a dataframe format with single line
        :param ss:  the simulated summary statics in dataframe format.
        :param y_cat_dict: name of all the models. will be printed the pdf
        :param extra: internal. to add in the graph to say about the method
        :return:will not return anything. rather call googness_fit to plot stuff and print the summary of gfit
        """
        best_index = int(observed.idxmax(axis=1).values)
        for key in y_cat_dict:
            modelindex = ss.reset_index(drop=True).index[pandas.Series(ss.index) == y_cat_dict[key]].values
            ss_sub = ss.iloc[modelindex]
            if best_index == key:
                dropping_columns = [keyother for keyother in y_cat_dict.keys() if keyother not in [key]]
                if len(dropping_columns) > 1:
                    dropping_columns = dropping_columns[1:]
            else:
                dropping_columns = [keyother for keyother in y_cat_dict.keys() if keyother not in [key, best_index]]
            cls.goodness_fit(target=observed.drop(dropping_columns, axis=1), ss=ss_sub.drop(dropping_columns, axis=1),
                             name=y_cat_dict[key], extra=extra,
                             tol=tol)

    @classmethod
    def csvout(cls, modelindex: pandas.Series, ss_predictions: pandas.DataFrame, predict4mreal: pandas.DataFrame):
        """
        in case of everything satisfied. this will output the test dataset in csv format which then later can be used by
        r directly. if you use it, it will delete all the middle files from the current directory if exists: x_test.h5,
        y_test.h5, x.h5, y.h5,scale_x.sav, scale_y.sav, params_header.csv
        :param modelindex: the model indexes in panas series format
        :param ss_predictions: the predicted ss by nn on simulation[meaning nn(ss)]
        :param predict4mreal: the predicted ss by nn on real data [meaning nn(ss_real))]
        :return: will not return anything but will create files model_index.csv.gz,ss_predicted.csv.gz,ss_target.csv.gz
        and will remove x_test.h5, y_test.h5, x.h5, y.h5, scale_x.sav, scale_y.sav, params_header.csv
        """
        modelindex = modelindex.rename('model_name')
        modelindex.to_csv('model_index.csv.gz', index=False, header=True)
        ss_predictions.to_csv('ss_predicted.csv.gz', index=False)
        predict4mreal.to_csv('ss_target.csv.gz', index=False)
        Misc.removefiles(['x_test.h5', 'y_test.h5', 'x.h5', 'y.h5', 'scale_x.sav', 'scale_y.sav', 'params_header.csv',
                          'y_cat_dict.txt'])

    @classmethod
    def wrapper_after_train(cls, ModelSeparation: keras.models.Model, x_test: Union[numpy.array, HDF5Matrix],
                            y_test: Union[numpy.array, HDF5Matrix], scale_x: Optional[preprocessing.MinMaxScaler],
                            y_cat_dict: Dict[int, str], ssfile: str, method: str = "mnlogistic",
                            tolerance: float = .005, csvout: bool = False) -> None:
        """
        This the wrapper for after training part of the classification. after training is done it will test on the test
        data set to see the power and then use a real data setto show how likely it support one model over another.
        it will use abc to give the power or standard deviation of the model that is predicted to know how much we are
        sure about the results. mainly it will do three parts of abc. one cv error , model selection and goodness of fit
        ModelSeparation.evaluate -> cls.read_ss_2_series -> cls.plot_power_of_ss (cls.r_summary) -> cls.model_selection
        (cls.r_summary)-> cls.gfit_all (cls.r_summary) -> cls.csvout
        :param ModelSeparation: The fitted keras model
        :param x_test: the test part of x aka summary statistics
        :param y_test: the test part of y aka models name. should be used keras.utils.to_categorical to better result
        :param scale_x: the MinMax scaler of x axis. can be None
        :param y_cat_dict: name of all the models. will be printed the pdf
        :param ssfile: the summary statisfic on real data set. should be csv format
        :param method: to tell which method is to be used in abc. default is mnlogitic. but can be rejection, neural net
         etc. as docuemented in the r.abc
        :param tolerance: the level of tolerance. default is .005
        :param csvout: in case of everything satisfied. this will output the test dataset in csv format. can be used
        later by r
        :return: will not return anything but will produce the graphs and print out how much it isure about any model
        """

        print("Evaluate with test:")
        ModelSeparation.evaluate(x_test, y_test, verbose=2)

        ssnn = pandas.DataFrame(ModelSeparation.predict(x_test[:]))
        indexnn = pandas.DataFrame(numpy.argmax(y_test, axis=1, out=None))[0].replace(y_cat_dict)
        ssnn.index = indexnn
        sfs = cls.read_ss_2_series(file=ssfile)
        cls.check_results(results=[x_test[0:2]], observed=sfs)
        if scale_x:
            predictednn = pandas.DataFrame(ModelSeparation.predict(scale_x.transform(sfs.values.reshape(1, -1))))
        else:
            predictednn = pandas.DataFrame(ModelSeparation.predict(sfs.values.reshape(1, -1)))
        print('Predicted by NN')
        print(sorted(y_cat_dict.items()))
        print(predictednn)

        robjects.r['pdf']("NN.pdf")
        cls.plot_power_of_ss(ss=ssnn.iloc[:, 1:], index=ssnn.index, tol=tolerance, method=method)
        cls.model_selection(target=predictednn.iloc[:, 1:], index=ssnn.index, ss=ssnn.iloc[:, 1:], method=method,
                            tol=tolerance)

        cls.gfit_all(observed=predictednn, ss=ssnn, y_cat_dict=y_cat_dict, extra='_nn_', tol=tolerance)
        robjects.r['dev.off']()
        if csvout:
            cls.csvout(modelindex=indexnn,
                       ss_predictions=pandas.DataFrame(ModelSeparation.predict(x_test[:])).rename(columns=y_cat_dict),
                       predict4mreal=predictednn.rename(columns=y_cat_dict))

    @classmethod
    def wrapper(cls, info: str, ssfile: str, demography: Optional[str] = None, method: str = "mnlogistic",
                tolerance: float = .005, test_size: int = int(1e4),
                chunksize: Optional[int] = None, scale: bool = False, csvout: bool = False):
        """
        the total wrapper of the classification method. with given underlying models it will compare with real data and
        will predict how much it sure about which model can bet predict the real data.
        wrapper_pre_train(Misc.removefiles -> cls.read_info -> Misc.getting_line_count ->  cls.subsetting_file_concating->
        cls.shufling_joined_models -> if chunksize :  cls.preparingdata_hdf5; else: cls.data_prep4ANN) -> wrapper_train
        Misc.loading_def_4m_file -> def ANNModelCheck ) wrapper_after_train(ModelSeparation.evaluate ->
        cls.read_ss_2_series -> cls.plot_power_of_ss (cls.r_summary) -> cls.model_selection (cls.r_summary)->
        cls.gfit_all (cls.r_summary) -> cls.csvout)
        :param info: the path of info file whose file column is the path of the file and second column defining the
        number of  parameters
        :param ssfile: the summary statisfic on real data set. should be csv format
        :param demography: custom function made for keras model. the path of that .py file. shoul have a def
        ANNModelCheck
        :param method: to tell which method is used in abc. default is mnlogitic. but can be rejection, neural net etc.
        as documented in the r.abc
        :param tolerance: the level of tolerance for abc. default is .005
        :param test_size:  the number of test rows. everything else will be used for train. 10k is default
        :param chunksize:  the number of rows accessed at a time.
        :param scale: to tell if the data should be scaled or not. default is false. will be scaled by MinMaxscaler.
        The scaling will only happen on the ss.
        :param csvout:  in case of everything satisfied. this will output the test dataset in csv format. can be used
        later by r
        :return: will not return anything but will plot and print the power
        """

        x_train, x_test, y_train, y_test, scale_x, y_cat_dict = cls.wrapper_pre_train(info=info, test_size=test_size,
                                                                                      chunksize=chunksize, scale=scale)
        ModelSeparation = cls.wrapper_train(x_train=x_train, y_train=y_train, demography=demography)
        cls.wrapper_after_train(ModelSeparation=ModelSeparation, x_test=x_test, y_test=y_test, scale_x=scale_x,
                                y_cat_dict=y_cat_dict, ssfile=ssfile, method=method, tolerance=tolerance,
                                csvout=csvout)


class ABC_TFK_Classification_PreTrain(ABC_TFK_Classification):
    """
    Subset of class ABC_TFK_Classification. Specifically to do the pre train stuff. it will produce data in hdf5 format which
     then easily can be used in training part of the classification. it will also delete all the files that can be
     output by the classification. so that it will work on a clean sheet.
     :param info: the path of info file whose file column is the path of the file and second column defining the
        number of  parameters
    :param test_size: the number of test rows. everything else will be used for train. 10k is default
    :param chunksize:  the number of rows accessed at a time.
    :param scale: to tell if the data should be scaled or not. default is false. will be scaled by MinMaxscaler.
    The scaling will only happen on the ss.
    :return: will not return anything but will create x.hdf5 ,y.hdf5 and scale_x so that it can be used for training
    later
    """

    def __new__(cls, info, test_size=int(1e4), chunksize=int(1e4), scale=False):
        return cls.wrapper_pre_train(info=info, test_size=test_size, chunksize=chunksize, scale=scale)


class ABC_TFK_Classification_Train(ABC_TFK_Classification):
    """
    Subset of class ABC_TFK_Classification. Specifically to do the train stuff. it need training data set for x.h5 and
    y.h5 in the cwd in hdf5 matrix format (HD5matrix) of keras
    :param demography: custom function made for keras model. the path of that .py file. shoul have a def
    ANNModelCheck
     :param test_rows: the number of test rows. everything else will be used for train. 10k is default
    :return: will not return anything but will train and save the file ModelClassification.h5
    """

    def __new__(cls, demography=None, test_rows=int(1e4)):
        return cls.wrapper(demography=demography, test_rows=test_rows)

    @classmethod
    def reading_y_train(cls, test_rows: int = int(1e4)) -> HDF5Matrix:
        """
        reading the file for y.h5 and then return the y_train using hdf5matrix
        :param test_rows: the number of rows kept for test data set. it will remove those lines from the end
        :return: return y_train hdf5 format
        """
        if os.path.isfile('y.h5'):
            rows = HDF5Matrix('y.h5', 'mydata').shape[0]
            y_train = HDF5Matrix('y.h5', 'mydata', start=0, end=rows - test_rows)
        else:
            print('Could not find file y.h5')
            sys.exit(1)
        return y_train

    @classmethod
    def reading_x_train(cls, test_rows: int = int(1e4)) -> HDF5Matrix:
        """
        reading the file for x.h5 and then return the x_train using hdf5matrix
        :param test_rows:  the number of rows kept for test data set. it will remove those lines from the end
        :return:  return x_train hdf5 format
        """
        if os.path.isfile('x.h5'):

            rows = HDF5Matrix('x.h5', 'mydata').shape[0]

            x_train = HDF5Matrix('x.h5', 'mydata', start=0, end=rows - test_rows)
        else:
            print('Could not file x.h5 ')
            sys.exit(1)
        return x_train

    @classmethod
    def wrapper(cls, demography: Optional[str] = None, test_rows: int = int(1e4)) -> None:
        """
        wrapper for the class ABC_TFK_Classification_Train. it will train the data set in a given folder where x.h5 and
        y.h5 present.
        :param demography:custom function made for keras model. the path of that .py file. shoul have a def
        ANNModelCheck
        :param test_rows: the number of test rows. everything else will be used for train. 10k is default
        :return: will not return anything but will train and save the file ModelClassification.h5
        """
        y_train = cls.reading_y_train(test_rows=test_rows)
        x_train = cls.reading_x_train(test_rows=test_rows)
        ModelSeparation = cls.wrapper_train(x_train=x_train, y_train=y_train, demography=demography)


class ABC_TFK_Classification_CV(ABC_TFK_Classification):
    """
    Subset of class ABC_TFK_Classification. Specifically to calculate crooss validation test. good if you dont have
    real data
    :param test_size: the number of test rows. everything else will be used for train. 10k is default
    :param tol: the level of tolerance for abc. default is .005
    :param method: to tell which method is used in abc. default is mnlogitic. but can be rejection, neural net etc.
    as documented in the r.abc
    :return: will not return anything but will plot the cross validation stuff of different models
    """

    def __new__(cls, test_size=int(1e4), tol=0.05, method='rejection'):
        return cls.wrapper(test_size=test_size, tol=tol, method=method)

    @classmethod
    def loadingkerasmodel(cls, ModelParamPredictionFile: str = 'ModelClassification.h5') -> keras.models.Model:
        """
        to load the saved keras model
        :param ModelParamPredictionFile: the .h5 file where it is saved
        :return: will return the model
        """
        from tensorflow.keras.models import load_model
        if os.path.isfile(ModelParamPredictionFile):
            model = load_model(ModelParamPredictionFile)
        else:
            print('The ANN model file could not be found please check. ', ModelParamPredictionFile)
            sys.exit(1)
        return model

    @classmethod
    def reading_y_test(cls, test_rows: int = int(1e4)) -> HDF5Matrix:
        """
        reading the file for y.h5/y_test.h5 and then return the y_test using hdf5matrix
        :param test_rows:  the number of rows kept for test data set. it will return those lines from the end
        :return: return y_test hdf5 format
        """
        if os.path.isfile('y_test.h5'):
            y_test = HDF5Matrix('y_test.h5', 'mydata')
        elif os.path.isfile('y.h5'):
            rows = HDF5Matrix('y.h5', 'mydata').shape[0]
            y_test = HDF5Matrix('y.h5', 'mydata', start=rows - test_rows, end=rows)
        else:
            print('Could not file y.h5 or y_test.h5')
            sys.exit(1)
        return y_test

    @classmethod
    def reading_x_test(cls, test_rows: int = int(1e4)) -> HDF5Matrix:
        """
        reading the file for x.h5/x_test.h5 and then return the x_test using hdf5matrix
        :param test_rows:  the number of rows kept for test data set. it will return those lines from the end
        :return: return x_test hdf5 format
        """

        if os.path.isfile('x_test.h5'):
            x_test = HDF5Matrix('x_test.h5', 'mydata')
        elif os.path.isfile('x.h5'):

            rows = HDF5Matrix('x.h5', 'mydata').shape[0]
            x_test = HDF5Matrix('x.h5', 'mydata', start=rows - test_rows, end=rows)
        else:
            print('Could not file x.h5 or x_test.h5')
            sys.exit(1)
        return x_test

    @classmethod
    def read_scalex_scaley(cls) -> Tuple[Optional[preprocessing.MinMaxScaler], Optional[preprocessing.MinMaxScaler]]:
        """
        read if scale_x and scale_y is present in the folder and return it (MinMaxscaler)
        :return: return x_scale min max scaler if present
        """
        if os.path.isfile('scale_x.sav'):
            scale_x = joblib.load('scale_x.sav')
        else:
            print('scale_x.sav not found. Assuming no scaling is required')
            scale_x = None
        scale_y = None
        return scale_x, scale_y

    @classmethod
    def read_y_cat_dict(cls) -> Dict[int, str]:
        """
        read the y_cat_dict.txt file in the cwd and return the dict
        :return: y_cat_dict in dict format ({0:'model1',1:'model2'..})
        """
        if os.path.isfile('y_cat_dict.txt'):
            y_cat_dict = eval(open('y_cat_dict.txt', 'r').read())

        else:
            print("Could not found the y_cat_dict.txt file. Thus the population names will be remain unknown")
            y_cat_dict = None
        return y_cat_dict

    @classmethod
    def read_data(cls, test_rows: int = int(1e4)) -> Tuple[
        keras.models.Model, HDF5Matrix, HDF5Matrix, Optional[preprocessing.MinMaxScaler], Optional[
            preprocessing.MinMaxScaler], Dict[int, str]]:
        """
        wrapper to read all the data before doing the abc stuff
        :param test_rows: the number of rows kept for test data set. it will return those lines from the end
        :return: The fitted keras model, test data set of x and y, scale of x and y if exists and name of all the models
        ({0:'model1',1:'model2'..}
        """
        ModelSeparation = cls.loadingkerasmodel()
        y_test = cls.reading_y_test(test_rows=test_rows)
        x_test = cls.reading_x_test(test_rows=test_rows)
        scale_x, scale_y = cls.read_scalex_scaley()
        y_cat_dict = cls.read_y_cat_dict()
        return ModelSeparation, x_test, y_test, scale_x, scale_y, y_cat_dict

    @classmethod
    def wrapper(cls, test_size: int = int(1e4), tol: float = 0.05, method: str = 'rejection') -> None:
        """
        this will produce do the cross validation stuff using abc on the nn predicted stuff. good in case real data is
        not avaliable yet
        :param test_size: the number of test rows. everything else will be used for train. 10k is default
        :param tol: the level of tolerance for abc. default is .005
        :param method: to tell which method is used in abc. default is mnlogitic. but can be rejection, neural net etc.
        as documented in the r.abc
        :return: will not return anything but will plot the cross validation stuff of different models
        """
        ModelSeparation, x_test, y_test, scale_x, scale_y, y_cat_dict = cls.read_data(test_rows=test_size)
        print("Evaluate with test:")
        ModelSeparation.evaluate(x_test, y_test, verbose=2)
        ssnn = pandas.DataFrame(ModelSeparation.predict(x_test[:]))[0]
        if y_cat_dict:
            indexnn = pandas.DataFrame(numpy.argmax(y_test, axis=1, out=None))[0].replace(y_cat_dict)
        else:
            indexnn = pandas.DataFrame(numpy.argmax(y_test, axis=1, out=None))[0]
        ssnn.index = indexnn
        print('Predicted by NN')
        print(sorted(y_cat_dict.items()))

        robjects.r['pdf']("CV.pdf")
        cls.plot_power_of_ss(ss=ssnn, index=ssnn.index, tol=tol, method=method)
        robjects.r['dev.off']()


class ABC_TFK_Classification_After_Train(ABC_TFK_Classification_CV):
    """
    Subset of class ABC_TFK_Classification. To do the ABC part.  after training is done it will test on the test
    data set to see the power and then use a real data setto show how likely it support one model over another.
    it will use abc to give the power or standard deviation of the model that is predicted to know how much we are
    sure about the results. mainly it will do three parts of abc. one cv error , model selection and goodness of fit
    :param ssfile:  the summary statisfic on real data set. should be csv format
    :param test_size:  the number of test rows. everything else will be used for train. 10k is default
    :param tol: the level of tolerance for abc. default is .01
    :param method:to tell which method is used in abc. default is rejection. but can be rejection, neural net etc.
    as documented in the r.abc
    :param csvout:in case of everything satisfied. this will output the test dataset in csv format. can be used
    later by r
    :return:will not return anything but will produce the graphs and print out how much it isure about any model
    """

    def __new__(cls, ssfile: str, test_size: int = int(1e4), tol: float = 0.05, method: str = 'rejection',
                csvout: bool = False):
        return cls.wrapper(ssfile=ssfile, test_size=test_size, tol=tol, method=method, csvout=csvout)

    @classmethod
    def wrapper(cls, ssfile: str, test_size: int = int(1e4), tol: float = 0.01, method: str = 'rejection',
                csvout: bool = False) -> None:
        """
        This the wrapper for after training part of the classification. after training is done it will test on the test
        data set to see the power and then use a real data setto show how likely it support one model over another.
        it will use abc to give the power or standard deviation of the model that is predicted to know how much we are
        sure about the results. mainly it will do three parts of abc. one cv error , model selection and goodness of fit
        :param ssfile:  the summary statisfic on real data set. should be csv format
        :param test_size:  the number of test rows. everything else will be used for train. 10k is default
        :param tol: the level of tolerance for abc. default is .01
        :param method:to tell which method is used in abc. default is rejection. but can be rejection, neural net etc.
        as documented in the r.abc
        :param csvout:in case of everything satisfied. this will output the test dataset in csv format. can be used
        later by r
        :return:will not return anything but will produce the graphs and print out how much it isure about any model
        """
        ModelSeparation, x_test, y_test, scale_x, scale_y, y_cat_dict = cls.read_data(test_rows=test_size)
        cls.wrapper_after_train(ModelSeparation=ModelSeparation, x_test=x_test, y_test=y_test, scale_x=scale_x,
                                y_cat_dict=y_cat_dict, ssfile=ssfile, method=method,
                                tolerance=tol, csvout=csvout)


#########TFK paramter estimation stuff
class ABC_TFK_Params(ABC_TFK_Classification):
    """
    This is the main class to do the parameter estimation of ABC_TFK method. with given model underlying parameters
    it will compare with real data and will predict which parameter best predict the real data.
    :param info: the path of info file whose file column is the path of the file and second column defining the
    number of  parameters. only the first line will be used
    :param ssfile: the summary statisfic on real data set. should be csv format
    :param chunksize: the number of rows accessed at a time.
    :param test_size:  the number of test rows. everything else will be used for train. 10k is default
    :param tol: the level of tolerance for abc. default is .005
    :param method: to tell which method is used in abc. default is mnlogitic. but can be rejection, neural net etc.
    as documented in the r.abc
    :param demography:  custom function made for keras model. the path of that .py file. shoul have a def
    ANNModelCheck
    :param csvout:  in case of everything satisfied. this will output the test dataset in csv format. can be used
    later by r
    :param scale: to tell if the data should be scaled or not. default is false. will be scaled by MinMaxscaler.
    The scaling will only happen on the ss.
    :return:  will not return anything but will plot and print the parameters

    """

    def __new__(cls, info: str, ssfile: str, chunksize: Optional[int] = None, test_size: int = int(1e4),
                tol: float = .005, method: str = 'rejection',
                demography: Optional[str] = None, csvout: bool = False, scale: bool = False) -> None:
        return cls.wrapper(info=info, ssfile=ssfile, chunksize=chunksize, test_size=test_size, tol=tol, method=method,
                           demography=demography, csvout=csvout, scale=scale)

    @classmethod
    def separation_param_ss(cls, filename: str, params_number: int) -> Tuple[str, str]:
        """
        It will separate the parameters and ss in two different csv files. which then can be read by
        pandas.read_csv
        :param filename: the path of info or csv file. can be both csv or gz
        :param params_number: the number of parameteres
        :return: will produce params.csv and ss.csv file
        """
        import os
        paramfile = 'params.csv'

        ssfile = 'ss.csv'
        if filename[-3:] == '.gz':
            paramcommand = Misc.joinginglistbyspecificstring(
                ["zcat", filename, '|', 'cut -f-' + str(params_number), '-d ","', ' > ', paramfile])
            sscommand = Misc.joinginglistbyspecificstring(
                ["zcat", filename, '|', 'cut -f' + str(params_number + 1) + '-', '-d ","', ' > ', ssfile])
        else:
            paramcommand = Misc.joinginglistbyspecificstring(
                ["cat", filename, '|', 'cut -f-' + str(params_number), '-d ","', ' > ', paramfile])
            sscommand = Misc.joinginglistbyspecificstring(
                ["cat", filename, '|', 'cut -f' + str(params_number + 1) + '-', '-d ","', ' > ', ssfile])
        os.system(paramcommand)
        os.system(sscommand)

        return paramfile, ssfile

    @classmethod
    def save_scale(cls, scale_x: preprocessing.MinMaxScaler, scale_y: Optional[preprocessing.MinMaxScaler]) -> None:
        """
        As the name suggest it will save the scaling  (MinMaxscaler) in a file. sklearn scaling will be saved
        :param scale_x: scaled for x
        :param scale_y: scaled for y
        :return: will save in  scale_x.sav, scale_y.sav file. will return nothing
        """
        scaler_filename = "scale_x.sav"
        joblib.dump(scale_x, scaler_filename)
        scaler_filename = "scale_y.sav"
        joblib.dump(scale_y, scaler_filename)

    @classmethod
    def preparingdata_hdf5(cls, paramfile: str, simss: str, chunksize: int = 100, test_size: int = int(1e4),
                           scale: bool = False) -> Tuple[
        HDF5Matrix, HDF5Matrix, Optional[preprocessing.MinMaxScaler], HDF5Matrix, HDF5Matrix, Optional[
            preprocessing.MinMaxScaler]]:
        """
        In case of chunk size is mentioned it will be assumed that the data is too big to save in ram and it will be
        saved in hdf5 format and it will split it in necessary steps
        :param paramfile: the parameter csv file path for y
        :param simss: the ss file path for x
        :param chunksize: the number of rows accesed at a time. default 100
        :param test_size: the number of test rows. everything else will be used for train. 10k is default
        :param scale: to tell if the data should be scaled or not. default is false. will be scaled by MinMaxscaler.
        :return: will return train and test data for both x and y in hdf5 matrix format with scale_x and scale_y if
        required
        """
        xfile = "x.h5"
        yfile = "y.h5"
        Misc.removefiles([xfile, yfile])
        if scale:
            scale_x = cls.MinMax4bigfile(csvpath=simss, h5path=xfile, chunksize=chunksize)
        else:
            scale_x = cls.MinMax4bigfile(csvpath=simss, h5path=xfile, chunksize=chunksize, scaling=False)
        x_train, x_test = cls.train_test_split_hdf5(xfile, test_rows=int(test_size))

        if scale:
            scale_y = cls.MinMax4bigfile(csvpath=paramfile, h5path=yfile, chunksize=chunksize)
        else:
            scale_y = cls.MinMax4bigfile(csvpath=paramfile, h5path=yfile, chunksize=chunksize, scaline=False)
        y_train, y_test = cls.train_test_split_hdf5(yfile, test_rows=int(test_size))
        if scale_x and scale_y:
            cls.save_scale(scale_x, scale_y)
        return x_train, x_test, scale_x, y_train, y_test, scale_y

    @classmethod
    def preparingdata(cls, paramfile: str, simssfile: str, test_size: int = int(1e4), scale: bool = False) -> Tuple[
        numpy.array, numpy.array, Optional[preprocessing.MinMaxScaler], numpy.array, numpy.array, Optional[
            preprocessing.MinMaxScaler]]:
        """
        In case the data is smaller and can be fit inside ram it will load the data in the ram and split it for train
        and test data for params (y) and sfs/ss (x). it will use min max scaler to scale it if required
        :param paramfile: the parameter csv file or y
        :param simssfile: the ss file path for x
        :param testsize: the number of test rows. everything else will be used for train. 10k is default
        :return: will return train and test data fro both x and y in numpy format with scale_x and scale_y if required
        """
        from sklearn.model_selection import train_test_split
        params = pandas.read_csv(paramfile)
        if scale:
            scale_y = preprocessing.MinMaxScaler()
            y = scale_y.fit_transform(params.values)
        else:
            y = params.values
            scale_y = None

        simss = pandas.read_csv(simssfile)
        if scale:

            scale_x = preprocessing.MinMaxScaler()
            x = scale_x.fit_transform(simss.values)
        else:
            x = simss.values
            scale_x = None

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
        if scale_x and scale_y:
            cls.save_scale(scale_x, scale_y)
        x_test_file = "x_test.h5"
        y_test_file = "y_test.h5"
        Misc.removefiles([x_test_file, y_test_file])
        Misc.numpy2hdf5(x_test, 'x_test.h5')
        Misc.numpy2hdf5(y_test, 'y_test.h5')
        return x_train, x_test, scale_x, y_train, y_test, scale_y

    @classmethod
    def wrapper_pre_train(cls, info: str, chunksize: Optional[int] = None, test_size: int = int(1e4),
                          scale: bool = False) -> Tuple[
        Union[numpy.array, HDF5Matrix], Union[numpy.array, HDF5Matrix], Optional[preprocessing.MinMaxScaler], Union[
            numpy.array, HDF5Matrix], Union[numpy.array, HDF5Matrix], Optional[preprocessing.MinMaxScaler], str]:
        """
        This is a a wrapper on the pretrain for parameter estimation. this will build stuff just before the training in
        ANN.it will produce data in hdf5 or numpy array format which then easily can be used in training part, it will
        also delete all the files that can be output from ABC-TFK thus not clashing with them

        Misc.removefiles-> cls.read_info ->cls.separation_param_ss -> if chunksize :preparingdata_hdf5 ;else
        preparingdata->Misc.removefiles
        :param info: the path of info file whose file column is the path of the file and second column defining the
        number of  parameters. only first line will be used
        :param chunksize: the number of rows accesed at a time. in case of big data
        :param testsize: the number of test rows. everything else will be used for train. 10k is default
        :param scale: to tell if the data should be scaled or not. default is false. will be scaled by MinMaxscaler.
        The scaling will be on both params and ss
        :return: will return (x_train, x_test, scale_x), (y_train, y_test, scale_y) and  header file path
        (params_header.csv)
        """
        Misc.removefiles(
            ['scale_x.sav', 'scale_y.sav', 'x_test.h5', 'y_test.h5', 'y.h5', 'x.h5', 'ModelParamPrediction.h5',
             'params.csv', 'ss.csv', 'params_header.csv'])
        files, paramnumbers, names = cls.read_info(info=info)
        if len(files) > 1:
            print("there are more than one file. Only will work with the first file:", files[0])
        paramfile, simss = cls.separation_param_ss(filename=files[0], params_number=paramnumbers[0])
        if chunksize:
            x_train, x_test, scale_x, y_train, y_test, scale_y = cls.preparingdata_hdf5(paramfile=paramfile,
                                                                                        simss=simss,
                                                                                        chunksize=chunksize,
                                                                                        test_size=test_size,
                                                                                        scale=scale)
        else:
            x_train, x_test, scale_x, y_train, y_test, scale_y = cls.preparingdata(paramfile=paramfile,
                                                                                   simssfile=simss,
                                                                                   test_size=test_size, scale=scale)
        header = 'params_header.csv'
        pandas.DataFrame(index=pandas.read_csv(paramfile, nrows=10).columns).transpose().to_csv(header,
                                                                                                index=False)
        Misc.removefiles([simss, paramfile])

        return x_train, x_test, scale_x, y_train, y_test, scale_y, header

    @classmethod
    def ANNModelParams(cls, x: Union[numpy.array, HDF5Matrix], y: Union[numpy.array, HDF5Matrix]) -> keras.models.Model:
        """
        A basic model for ANN to calculate parameters
        :param x:  the x or summary statistics. can be numpy array or hdf5.
        :param y: the parameters which produced those ss
        :return: will return the trained model
        """
        ####
        # callbacks = myCallback()
        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(x.shape[1],)))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(y.shape[1]))
        model.compile(loss='logcosh', optimizer='Nadam', metrics=['accuracy'])
        # model.fit(x, y, epochs=500, shuffle="batch", callbacks=[callbacks], verbose=2)
        model.fit(x, y, epochs=500, shuffle="batch", verbose=2)

        return model

    @classmethod
    def wrapper_train(cls, x_train: Union[numpy.array, HDF5Matrix], y_train: Union[numpy.array, HDF5Matrix],
                      demography: Optional[str] = None) -> keras.models.Model:
        """
        This is to the wrapper for the training for parameter estimation. the slowest part of the code.it need trainging
        data set for x and y. can be either numpy array or hdf5 matrix format (HD5matrix) of keras
        Misc.loading_def_4m_file -> def ANNModelCheck
        :param x_train: train part of x aka summary statistics
        :param y_train: train part of all the parameters
        :param demography: custom function made for keras model. the path of that .py file. should have a def has
        ANNModelParams as def in Any.py
        :return: will return the keras model. it will also save the model in ModelParamPrediction.h5
        """
        if demography:
            ANNModelParams = Misc.loading_def_4m_file(filepath=demography, defname='ANNModelParams')
            if ANNModelParams:
                ModelParamPrediction = ANNModelParams(x=x_train, y=y_train)
            else:
                ModelParamPrediction = cls.ANNModelParams(x=x_train, y=y_train)
        else:
            ModelParamPrediction = cls.ANNModelParams(x=x_train, y=y_train)

        ModelParamPrediction.save("ModelParamPrediction.h5")
        return ModelParamPrediction

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
        Rabc complains if the parameters do not have std (all are same or constant). this step will remove those
        :param test_predictions: the predicted values dataframe from ANN model
        :param predict4mreal: the predicted value from the real data
        :param params_unscaled: the real paramters to produce the ss
        :return: will return the all the inputs but will remove columns with 0 std
        """

        std = cls.R_std_columns(params_unscaled)
        columns2bdrop = std[std == 0].index

        if len(columns2bdrop) == params_unscaled.shape[1]:
            print('All the columns in the params have no standrd deviation. Please check in params for y_test')
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
                'All the columns in the prediction have no standard deviation. Please check in predictions for ANN model. The ANN training was not good')
            sys.exit(1)
        elif len(columns2bdrop) > 0:
            print(
                'These columns do not have any std in the predicted data set. Is being removed before ABC. This is not a good sign. Please check the trainging model')
            print(columns2bdrop)
            test_predictions = test_predictions.drop(columns2bdrop, axis=1)
            predict4mreal = predict4mreal.drop(columns2bdrop, axis=1)
            params_unscaled.drop(columns2bdrop, axis=1)

        return test_predictions, predict4mreal, params_unscaled

    @classmethod
    def preparing_for_abc(cls, ModelParamPrediction: keras.models.Model, x_test: Union[numpy.array, HDF5Matrix],
                          y_test: Union[numpy.array, HDF5Matrix], scale_x: Optional[preprocessing.MinMaxScaler],
                          scale_y: Optional[preprocessing.MinMaxScaler], params_names: numpy.array,
                          ss: pandas.Series) -> Tuple[
        pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
        """
        as the name suggest after ann ran it will prepare the data for abc analysis
        :param ModelParamPrediction: the ann model that was run by keras tf
        :param x_test: the test part of x aka summary statistics
        :param y_test:the y_test or parameters series which never ran on the ann itself
        :param scale_x: the MinMax scaler of x axis. can be None
        :param scale_y:the MinMax scaler of y axis. can be None
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

            predict4mreal = pandas.DataFrame(scale_y.inverse_transform(ModelParamPrediction.predict(ssscaled)))
        else:
            ssscaled = ss.values.reshape(1, -1)

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
        if method == 'rejection' or method == 'loclinear':
            robjects.r['pdf'](name)
            print('Cv error independently')
            for colnum in range(param.shape[1]):
                cvresreg = abc.cv4abc(param=pandas.DataFrame(param.iloc[:, colnum]),
                                      sumstat=pandas.DataFrame(ss.iloc[:, colnum]), nval=repeats, tols=tol,
                                      method=method, trace=trace)
                print(param.iloc[:, colnum].name)
                print(robjects.r['summary'](cvresreg)[0][0])
                robjects.r['plot'](cvresreg, ask=False)
            robjects.r['dev.off']()
        if method == 'rejection' or method == 'neuralnet':
            print('Cv error together')
            robjects.r['pdf'](name[:-4] + '_together.pdf')
            if method == 'neuralnet':
                robjects.r['sink']('temp.txt')
                cvresreg = abc.cv4abc(param=param, sumstat=ss, nval=repeats, tols=tol, method=method, trace=trace)
                robjects.r['sink']()
                os.remove('temp.txt')
            else:
                cvresreg = abc.cv4abc(param=param, sumstat=ss, nval=repeats, tols=tol, method=method, trace=trace)
            print(robjects.r['summary'](cvresreg))
            robjects.r['plot'](cvresreg, ask=False)
            robjects.r['dev.off']()

    @classmethod
    def abc_params(cls, target: pandas.DataFrame, param: pandas.DataFrame, ss: pandas.DataFrame, tol: float = .01,
                   method: str = "loclinear", name: str = 'paramposterior.pdf') -> None:
        """
        the final abc calculation on real data.
        :param target: the real ss in a pandas  data frameformat
        :param param: the parameter data frame format (y_test)
        :param ss: the summary statics in dataframe format
        :param tol: the tolerance level. default is .001
        :param method: the method to calculate abc cv error. can be "rejection", "loclinear", and "neuralnet". default
        loclinear
        :param name: the ouput save file name
        :return: will not return anything but save the plot and print out the summary
        """

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
                robjects.r['sink']('temp.txt')
                res = abc.abc(target=target, param=param, sumstat=ss, method=method, tol=tol)
                robjects.r['sink']()
                os.remove('temp.txt')
            else:
                res = abc.abc(target=target, param=param, sumstat=ss, method=method, tol=tol)
            cls.r_summary(res)
            if method == 'rejection':
                print('rejection plot not implemented yet')
            else:
                robjects.r['plot'](res, param=param, onefile=True, file=name[:-4] + '_together')

    @classmethod
    def csvout(cls, params_unscaled: pandas.DataFrame, test_predictions: pandas.DataFrame,
               predict4mreal: pandas.DataFrame) -> None:
        """
        in case you need csv file output of predicted params from nn, which then can be directly used by R. if you use
        it, it will delete all the middle files from the current directory if exists: x_test.h5, y_test.h5, x.h5, y.h5,
        scale_x.sav, scale_y.sav, params_header.csv
        :param test_predictions: the predicted values dataframe from ANN model
        :param predict4mreal: the predicted value from the real data
        :param params_unscaled: the real parameters to produce the ss
        :return: will not return anything but save 3 files params.csv.gz, ss_predicted.csv.gz, ss_target.csv.gz which
        can be used for further in R for in depth abc
        """
        params_unscaled.to_csv('params.csv.gz', index=False)
        test_predictions.to_csv('ss_predicted.csv.gz', index=False)
        predict4mreal.to_csv('ss_target.csv.gz', index=False)
        Misc.removefiles(['x_test.h5', 'y_test.h5', 'x.h5', 'y.h5', 'scale_x.sav', 'scale_y.sav', 'params_header.csv'])

    @classmethod
    def wrapper_aftertrain(cls, ModelParamPrediction: keras.models.Model, x_test: Union[numpy.array, HDF5Matrix],
                           y_test: Union[numpy.array, HDF5Matrix], ssfile: str,
                           scale_x: Optional[preprocessing.MinMaxScaler], scale_y: Optional[preprocessing.MinMaxScaler],
                           paramfile: str = 'params_header.csv', method: str = 'rejection', tol: float = .005,
                           csvout: bool = False) -> None:
        """
        The wrapper to test how the traingin usin ANN works. after training is done it will test on the test  data set
        to see the power and then use a real data set to show what most likely parameters can create the real data.
        it will use abc to give the power or standard deviation of the parameters that is predicted by nn to know how
        much we are sure about the results. mainly it will do two parts of abc. one cv error and parameter estimation
        ModelParamPrediction.evaluate-> cls.read_ss_2_series-> cls.preparing_for_abc->cls.plot_param_cv_error->
        cls.abc_params-> Misc.removefiles->cls.csvout
        :param ModelParamPrediction:The fitted keras model
        :param x_test: the test part of x aka summary statistics
        :param y_test: the test part of y aka parameter dataset
        :param ssfile: the real ss file path
        :param scale_x: the scale of ss if exist
        :param scale_y: the scale of parameters if exist
        :param paramfile: the path of paramter header file. default is 'params_header.csv'
        :param method: the method to be used in r_abc. default is rejection
        :param tol: the tolerance level to be used in r_abc. default is .005
        :param csvout: in case of everything satisfied. this will output the test dataset in csv format. can be used
        later by r
        :return: will not return anything but print out the result and save files for plot
        """
        print("Evaluate with test:")
        ModelParamPrediction.evaluate(x_test[:], y_test[:], verbose=2)
        params_names = pandas.read_csv(paramfile).columns
        ss = cls.read_ss_2_series(file=ssfile)

        test_predictions, predict4mreal, params_unscaled = cls.preparing_for_abc(ModelParamPrediction, x_test, y_test,
                                                                                 scale_x, scale_y, params_names, ss)

        print("ANN predict")
        print(predict4mreal.transpose())

        print('correlation between params. Prior')
        print(params_unscaled.corr().to_string())

        print('correlation between predicted params. Posterior')
        print(test_predictions.corr().to_string())

        cls.plot_param_cv_error(param=params_unscaled, ss=test_predictions, name='nnparamcv.pdf', tol=tol,
                                method=method)
        cls.abc_params(target=predict4mreal, param=params_unscaled, ss=test_predictions, method=method, tol=tol)
        Misc.removefiles([paramfile])
        if csvout:
            cls.csvout(params_unscaled=params_unscaled, test_predictions=test_predictions, predict4mreal=predict4mreal)

    @classmethod
    def wrapper(cls, info: str, ssfile: str, chunksize: Optional[int] = None, test_size: int = int(1e4),
                tol: float = .005, method: str = 'rejection',
                demography: Optional[str] = None, csvout: bool = False, scale: bool = False) -> None:
        """
        the total wrapper of the pameter estimation method. with given model underlying parameters it will compare with
        real data and will predict which parameter best predict the real data.
        wrapper_pretrain(Misc.removefiles-> cls.read_info ->cls.separation_param_ss -> if chunksize :preparingdata_hdf5 ;else
        preparingdata->Misc.removefiles) ->wrapper_train(Misc.loading_def_4m_file -> def ANNModelCheck)->
        wrapper_after_train(ModelParamPrediction.evaluate-> cls.read_ss_2_series-> cls.preparing_for_abc->
        cls.plot_param_cv_error->cls.abc_params-> Misc.removefiles->cls.csvout)
        :param info: the path of info file whose file column is the path of the file and second column defining the
        number of  parameters. only the first line will be used
        :param ssfile: the summary statisfic on real data set. should be csv format
        :param chunksize: the number of rows accessed at a time.
        :param test_size:  the number of test rows. everything else will be used for train. 10k is default
        :param tol: the level of tolerance for abc. default is .005
        :param method: to tell which method is used in abc. default is mnlogitic. but can be rejection, neural net etc.
        as documented in the r.abc
        :param demography:  custom function made for keras model. the path of that .py file. shoul have a def
        ANNModelCheck
        :param csvout:  in case of everything satisfied. this will output the test dataset in csv format. can be used
        later by r
        :param scale: to tell if the data should be scaled or not. default is false. will be scaled by MinMaxscaler.
        The scaling will only happen on the ss.
        :return:  will not return anything but will plot and print the parameters
        """

        x_train, x_test, scale_x, y_train, y_test, scale_y, paramfile = cls.wrapper_pre_train(info=info,
                                                                                              chunksize=chunksize,
                                                                                              test_size=test_size,
                                                                                              scale=scale)
        ModelParamPrediction = cls.wrapper_train(x_train=x_train, y_train=y_train, demography=demography)
        cls.wrapper_aftertrain(ModelParamPrediction=ModelParamPrediction, x_test=x_test, y_test=y_test,
                               ssfile=ssfile, scale_x=scale_x, scale_y=scale_y,
                               paramfile=paramfile, method=method, tol=tol, csvout=csvout)


class ABC_TFK_Params_PreTrain(ABC_TFK_Params):
    """
    Subset of Parameter estimation. just to prepare the data for tfk.this will build stuff just before the training in
    ANN.it will produce data in hdf5 or numpy array format which then easily can be used in training part, it will
    also delete all the files that can be output from ABC-TFK thus not clashing with them
    """

    def __new__(cls, info: str, test_size: int = int(1e4), chunksize: Optional[int] = int(1e4),
                scale: bool = False) -> None:
        return cls.wrapper_pre_train(info=info, test_size=test_size, chunksize=chunksize, scale=scale)


class ABC_TFK_Params_Train(ABC_TFK_Params):
    """
    Subset for the training of parameter estimation. the slowest part of the code.it need training data set for x and y.
     can be hdf5 matrix format (HD5matrix) of keras
    :param x_train: train part of x aka summary statistics
    :param y_train: train part of all the parameters
    :param demography: custom function made for keras model. the path of that .py file. should have a def has
    ANNModelParams as def in Any.py
    :return: will return the keras model. it will also save the model in ModelParamPrediction.h5
    """

    def __new__(cls, test_rows: int = int(1e4), demography: Optional[str] = None) -> None:
        return cls.wrapper(test_rows=test_rows, demography=demography)

    @classmethod
    def wrapper(cls, test_rows: int = int(1e4), demography: Optional[str] = None) -> None:
        # print (x_dimension)
        y_train = ABC_TFK_Classification_Train.reading_y_train(test_rows=test_rows)
        x_train = ABC_TFK_Classification_Train.reading_x_train(test_rows=test_rows)
        ModelParamPrediction = cls.wrapper_train(x_train=x_train, y_train=y_train, demography=demography)


class ABC_TFK_Params_CV(ABC_TFK_Params):
    """
    Subset of Paramter estimation Specifically to calculate cross validation test. good if you dont have
    real data
    :param test_size: the number of test rows. everything else will be used for train. 10k is default
    :param tol: the level of tolerance for abc. default is .005
    :param method: to tell which method is used in abc. default is mnlogitic. but can be rejection, neural net etc.
    as documented in the r.abc
    :return: will not return anything but will plot the cross validation stuff for parameter estimation
    """

    def __new__(cls, test_size: int = int(1e3), tol: float = 0.01, method: str = 'neuralnet') -> None:
        return cls.wrapper(test_size=test_size, tol=tol, method=method)

    @classmethod
    def read_scalex_scaley(cls) -> Tuple[Optional[preprocessing.MinMaxScaler], Optional[preprocessing.MinMaxScaler]]:
        """
        this to read if scale_x and scale_y is present in the folder and return it (MinMaxscaler)
        :return: return x_scale and y_scale min max scaler if present
        """
        if os.path.isfile('scale_x.sav'):
            scale_x = joblib.load('scale_x.sav')
        else:
            print('scale_x.sav not found. Assuming no scaling is required')
            scale_x = None
        if os.path.isfile('scale_y.sav'):
            scale_y = joblib.load('scale_y.sav')
        else:
            print('scale_y.sav not found.Assuming no scaling is required')
            scale_y = None
        return scale_x, scale_y

    @classmethod
    def read_data(cls, test_rows: int = int(1e4)) -> Tuple[
        keras.models.Model, Union[numpy.array, HDF5Matrix], Union[numpy.array, HDF5Matrix], Optional[
            preprocessing.MinMaxScaler], Optional[preprocessing.MinMaxScaler]]:
        """
        to read all the data before doing the abc stuff
        :param test_rows: the number of rows kept for test data set. it will return those lines from the end
        :return:  The fitted keras model, test data set of x and y, scale of x and y if exists
        """
        ModelParamPrediction = ABC_TFK_Classification_CV.loadingkerasmodel('ModelParamPrediction.h5')
        y_test = ABC_TFK_Classification_CV.reading_y_test(test_rows=test_rows)
        x_test = ABC_TFK_Classification_CV.reading_x_test(test_rows=test_rows)
        scale_x, scale_y = cls.read_scalex_scaley()
        return ModelParamPrediction, x_test, y_test, scale_x, scale_y

    @classmethod
    def preparing_for_abc(cls, ModelParamPrediction: keras.models.Model, x_test: Union[numpy.array, HDF5Matrix],
                          y_test: Union[numpy.array, HDF5Matrix], scale_x: Optional[preprocessing.MinMaxScaler],
                          scale_y: Optional[preprocessing.MinMaxScaler], params_names: numpy.array) -> Tuple[
        pandas.DataFrame, pandas.DataFrame]:
        """
        as the name suggest after ann ran it will prepare the data for abc analysis
        :param ModelParamPrediction: the ann model that was run by keras tf
        :param x_test:the test part of x aka summary statistics
        :param y_test:the y_test series which never ran on the ann itself
        :param y_test:the y_test or parameters series which never ran on the ann itself
        :param scale_x: the MinMax scaler of x axis. can be None
        :param scale_y:the MinMax scaler of y axis. can be None
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

        return test_predictions, params_unscaled

    @classmethod
    def wrapper(cls, test_size: int = int(1e3), tol: float = 0.01, method: str = 'neuralnet') -> None:
        """
       Subset of Paramter estimation Specifically to calculate cross validation test. good if you dont have
       real data
       :param test_size: the number of test rows. everything else will be used for train. 10k is default
       :param tol: the level of tolerance for abc. default is .005
       :param method: to tell which method is used in abc. default is mnlogitic. but can be rejection, neural net etc.
       as documented in the r.abc
       :return: will not return anything but will plot the cross validation stuff for parameter estimation
       """
        ModelParamPrediction, x_test, y_test, scale_x, scale_y = cls.read_data(test_rows=test_size)

        print("Evaluate with test:")
        ModelParamPrediction.evaluate(x_test[:], y_test[:], verbose=2)
        test_predictions, params_unscaled = cls.preparing_for_abc(ModelParamPrediction=ModelParamPrediction,
                                                                  x_test=x_test,
                                                                  y_test=y_test, scale_x=scale_x, scale_y=scale_y,
                                                                  params_names=pandas.read_csv(
                                                                      'params_header.csv').columns)

        cls.plot_param_cv_error(param=params_unscaled, ss=test_predictions, name='nnparamcv.pdf', tol=tol,
                                method=method)


class ABC_TFK_Params_After_Train(ABC_TFK_Params):
    """
   The subset class to test to paramter estimation. after training is done it will test on the test data set
   to see the power and then use a real data set to show what most likely parameters can create the real data.
   it will use abc to give the power or standard deviation of the parameters that is predicted by nn to know how
   much we are sure about the results. mainly it will do two parts of abc. one cv error and parameter estimation
   :param ssfile:  the real ss file path
   :param test_size:  the number of test rows. everything else will be used for train. 10k is default
   :param tol: the level of tolerance for abc. default is .005
   :param method: to tell which method is used in abc. default is mnlogitic. but can be rejection, neural net etc.
   as documented in the r.abc
   :param csvout: in case of everything satisfied. this will output the test dataset in csv format. can be used
   later by r
   :return: will not return anything but will plot and print the parameters
   """

    def __new__(cls, ssfile: str, test_size: int = int(1e4), tol: float = .01, method: str = 'neuralnet',
                csvout: bool = False) -> None:
        cls.wrapper(ssfile=ssfile, test_size=test_size, tol=tol, method=method, csvout=csvout)

    @classmethod
    def wrapper(cls, ssfile: str, test_size: int = int(1e4), tol: float = 0.01, method: str = 'neuralnet',
                csvout: bool = False) -> None:
        """
        The wrapper to test how the traingin usin ANN works. after training is done it will test on the test  data set
        to see the power and then use a real data set to show what most likely parameters can create the real data.
        it will use abc to give the power or standard deviation of the parameters that is predicted by nn to know how
        much we are sure about the results. mainly it will do two parts of abc. one cv error and parameter estimation
        :param ssfile:  the real ss file path
        :param test_size:  the number of test rows. everything else will be used for train. 10k is default
        :param tol: the level of tolerance for abc. default is .005
        :param method: to tell which method is used in abc. default is mnlogitic. but can be rejection, neural net etc.
        as documented in the r.abc
        :param csvout: in case of everything satisfied. this will output the test dataset in csv format. can be used
        later by r
        :return: will not return anything but will plot and print the parameters
        """
        ModelParamPrediction, x_test, y_test, scale_x, scale_y = ABC_TFK_Params_CV.read_data(test_rows=test_size)
        cls.wrapper_aftertrain(ModelParamPrediction=ModelParamPrediction, ssfile=ssfile, x_test=x_test, y_test=y_test,
                               scale_x=scale_x, scale_y=scale_y, paramfile='params_header.csv', method=method, tol=tol,
                               csvout=csvout)