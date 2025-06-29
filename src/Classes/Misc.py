#!/usr/bin/python
"""
This file is for miscellaneous def
"""
import os
# type hint for readability
from typing import Tuple, Optional, Callable, Union


# R package check


def importr_tryhard(packname: str):
    """
    It will check if the package is installed under rpy2. if not it will automatically install it from the fist online
    server

    :param packname: the name of the package which we want to load
    :return: it will return the package loaded.
    """
    import rpy2
    from rpy2.robjects.packages import importr
    try:
        rpack = importr(packname)
    except rpy2.robjects.packages.PackageNotInstalledError:
        utils = importr('utils')
        utils.chooseCRANmirror(ind=1)
        utils.install_packages(packname)
        rpack = importr(packname)
    return rpack


# reading files


def reading_bylines_small(filepath: str) -> list:
    """
    Will return a list where every element is a line. Very basic and can only be done on small files

    :param filepath: The file to be read
    :return: The list to be returned
    """
    with open(filepath, 'r') as fileread:
        lines = fileread.readlines()
    lines = [line.strip("\n") for line in lines]
    return lines


def getting_line_count(file: str) -> int:
    """
    To get the the number of lines present in the file

    :param file: full path of the file
    :return: will return the line count in total
    """
    import gzip
    if file[-3:] == '.gz':
        with gzip.open(file, 'rb') as f:
            count = sum(1 for _ in f)
    else:
        with open(file, 'r') as f:
            count = sum(1 for _ in f)
    return count


# file name manipulations


def gettingfilename(filepath: str) -> str:
    """
    It wil give back the name of the file from the filepath

    :return: Name of the file
    """
    if "/" in filepath:
        words = filepath.split("/")
        filename = words[len(words) - 1]
        return filename
    else:
        return filepath


def gettingextension(filepath: str) -> str:
    """
    As the name suggest from the filepath it will give the file extension. Remember if the filename do not have dot (.)
    it will give back the whole filename

    :param filepath: The whole file path of the file whose extension we wanted to get
    :return: the extension itself
    """
    import re
    filename = gettingfilename(filepath)
    if re.search(".", filename):
        splitfilename = filename.split(".")
        return splitfilename[len(splitfilename) - 1]
    else:
        return ""


def filenamewithoutextension(filepath: str) -> str:
    """
    As the name suggest it will remove the file extension from the full filename. Additionally if its in a path it will
    remove the path as well. If it has multiple dot will only remove the last one (i.e. bla.vcf.gz will give bla.vcf)

    :param filepath: The file path
    :return: will give only file name without extension.
    """
    filename = gettingfilename(filepath)
    extension = gettingextension(filepath)
    if len(extension) > 0:
        return filename[:-(len(extension) + 1)]
    else:
        return filename


def filenamewithoutextension_checking_zipped(filepath: str) -> str:
    """
    As the name suggest it is like filename without extension but it will check if the file is zipped (end with gz) and
    then run it twice to remove double extension. name.vcf.gz/name.vcf will give name

    :param filepath: the whole path of the file
    :return: will return the name with out the extension
    """
    filename = filenamewithoutextension(filepath)
    if filepath[-2:] == 'gz':
        filename = filenamewithoutextension(filename)
    return filename


# files checker


def file_existence_checker(names: list, files: list) -> Tuple[str, list]:
    """
    As most of the clasess needs different type of check of existence of the file. It will check if the file exist and
    then return appropriate error message to print.
    One way to use the output: \n
        tobeprint, absentfiles = Misc.file_existence_checker(names, files)\n
        if len(absentfiles) > 0: \n
            print (tobeprint) \n
            sys.exit(1)

    :param names: the name or the purpose of the file
    :param files: The file which we are checking
    :return: If exist the files it will return nothing. If any file does not exist it will return the indexes which
        files do not exist with a printable command saying which files (with the name or purpose) does not exist
    """
    import os
    printer = []
    absence = []
    for index, file in enumerate(files):
        if not os.path.isfile(file):
            printerlinetemp = ["The", names[index], "file could not be found:", file]
            printer.append(joinginglistbyspecificstring(printerlinetemp))
            absence.append(file)
    return joinginglistbyspecificstring(printer, "\n"), absence

def args_valid_file(parser, arg):
    """
    This will check with in the argparse and report if the file exist or not. should be used in the argparse command
    parser.add_argument('file',type=lambda x: Misc.args_valid_file(parser, x))
    Args:
        parser: argparse.ArgumentParser() which is the main object of argparse command
        arg: the path of the command itself

    Returns: if the file exist it will return the path if not it will raise an error through argparse itself

    """
    if arg:
        if not os.path.exists(arg):
            parser.error("The file %s does not exist!" % arg)
    return arg  # return an open file handle


# python string commands


def joinginglistbyspecificstring(listinput: list, string: str = " ") -> str:
    """
    It join a list with the given string

    :param listinput: the input list
    :param string: the string with which it has to join. by default is space
    :return: the joined string
    """
    listinput = [x for x in listinput if x is not None]  # to remove none
    listinput = list(map(str, listinput))
    return string.join(listinput)


# del files


def removefilescommands(files: Union[list, Tuple]) -> list:
    """
    Giving a list of files it will give back the commands necessary for deleting such files.

    :param files: the list of files
    :return: it will return the list of commands
    """

    return ['rm -f ' + file for file in files]


def removefiles(files: Union[list, Tuple], printing: bool = True) -> None:
    """
    Its the wrapper of removefiles commands. if you want ot remove the files right now. also print it is removing.
    good to know which files are getting removed always. It will print remove even if the file do not exist in the first
    place

    :param files: the list of files
    :param printing: to print if removing the file. bool. default is true. If false noting will be printed just deleted
    :return: will not return anything just remove and print
    """
    commands = removefilescommands(files)
    if printing:
        for file in files:
            print("Removing if exist:", file, flush=True)
    [os.system(command) for command in commands]
    return None


# hdf5 commands


def numpy2hdf5(x, outfile: str, dataset: str = 'mydata') -> None:
    """
    This will save the numpy to hdf5 format.

    :param x: the array (can be anything which can be converted to numpy array)
    :param outfile: the name of the output file
    :param dataset: name of the data set. default is mydata
    :return: will not return anything but will save the h5f file
    """
    import h5py
    import numpy
    x = numpy.array(x)
    f = h5py.File(outfile, 'w')
    f.create_dataset(dataset, data=x)
    f.close()


# importing functions


def loading_def_4m_file(filepath: str, defname: str) -> Optional[Callable]:
    """
    loading defintion from a file. some of the mar deprecate. will check later.

    :param filepath: the full path of the .py file
    :param defname: the name of the definition
    :return: will return the func if exist in the file. if not will return None
    """
    import importlib
    if not os.path.isfile(filepath):
        print('The file count not be found to load. Please check:', filepath)
    modname = filenamewithoutextension(filepath)
    spec = importlib.util.spec_from_file_location(modname, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if defname in dir(mod):

        return getattr(mod, defname)
    else:
        return None


# shell script commands


def creatingfolders(specificfolder: str) -> str:
    """
    As the name suggest it will create a folder if the folder do not exist. it will also check if the end is '/' and add
     it if not there. it will return the folder it created

    :param specificfolder: The folder needs to be created
    :return: will not return anything. Either it will create if the folder do not exist or not return anything
    """
    import os
    if specificfolder != '':
        if specificfolder[-1] != '/':
            specificfolder = specificfolder + '/'

        specificfolder = os.path.expanduser(specificfolder)
        if not os.path.exists(specificfolder):
            os.makedirs(specificfolder)
    return specificfolder


def adding_pop(alldata, popfilepath):
    """
    This will return a column with pop information on a dataframe
    :param alldata: the data, where you need at least one column with inds information and it should be the first column
    :param popfilepath: the path of pop infor. The file should have first column with inds and the second is for pop. No header
    :return: will return a column (Series) of exact size of input (alldata) so that it can be concatenate with the data itself
    """
    import numpy, pandas
    popfile = pandas.read_csv(popfilepath, header=None, names=['inds', 'pop'], delim_whitespace=True)
    alldata = alldata[:]
    alldata = alldata.assign(pop=numpy.nan)
    for index, row in alldata.iterrows():
        try:
            alldata.loc[index, "pop"] = list(popfile[popfile["inds"] == row[0]]["pop"].values)[0]
        except IndexError:
            pass
    return alldata['pop']

def reading_csv_no_header(file, **kwrds):
    """
    As I always forget how to read pandas csv files without reading the header. Here is the automatic version
    Args:
        file: the csv file to read.
        **kwrds: all the other commands except header=None can be passed here. header=None is to tell them that there is
        no header line in the file
    Returns: will return the pandas dataframe

    """
    import pandas
    data = pandas.read_csv(file, header=None, **kwrds)
    return data