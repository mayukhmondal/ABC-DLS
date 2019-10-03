#!/usr/bin/python
"""
This file is for miscellaneous def
"""
import os
##type hint for readability
from typing import List, Dict, Tuple, Optional, Callable, Union

####R packaage check
def importr_tryhard(packname:str):
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
    except rpy2.rinterface.RRuntimeError:
        utils = importr('utils')
        utils.chooseCRANmirror(ind=1)
        utils.install_packages(packname)
        rpack = importr(packname)
    return rpack

######reading filesssss
def reading_bylines_small(filepath:str)-> list:
    """
    Will return a list where every element is a line. Very basic and can only be done on small files
    :param filepath: The file to be read
    :return: The list to be returned
    """
    with open(filepath, 'r') as fileread:
        lines = fileread.readlines()
    lines = [line.strip("\n") for line in lines]
    return lines

def getting_line_count(file:str)-> int:
    """
    To get the the number of lines present in the file
    :param file: full path of the file
    :return: will return the line count in total
    """
    import gzip
    if file[-3:]=='.gz':
        with gzip.open(file, 'rb') as f:
            count=sum(1 for _ in f)
    else:
        with open(file,'r') as f:
            count=sum(1 for _ in f)
    return count

##file name manipulations
def gettingfilename(filepath:str)-> str:
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


def gettingextension(filepath:str)->str:
    """
    As the name suggest from the filepath it will give the file extension. Remember if the filename do not have dot (.)
    it will give back the whole filename
    :param filepath: The whole file path of the file whose extension we wanted to get
    :return:the extension itself
    """
    import re
    filename = gettingfilename(filepath)
    if re.search(".", filename):
        splitfilename = filename.split(".")
        return splitfilename[len(splitfilename) - 1]
    else:
        return ""
def filenamewithoutextension(filepath:str)-> str:
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
def filenamewithoutextension_checking_zipped(filepath:str)->str:
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

##files checker
def file_existence_checker(names:list, files:list) -> Tuple[str,list]:
    """
    As most of the clasess needs different type of check of existence of the file. It will check if the file exist and
    then return appropriate error message to print.
    One way to use the output
    tobeprint, absentfiles = Misc.file_existence_checker(names, files)
    if len(absentfiles) > 0:
        print (tobeprint)
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

#######python string commands
def joinginglistbyspecificstring(listinput:list, string:str=" ")->str:
    """
    It join a list with the given string
    :param listinput: the input list
    :param string: the string with which it has to join. by default is space
    :return: the joined string
    """
    listinput = [x for x in listinput if x is not None]  ##to remove none
    listinput = list(map(str, listinput))
    return string.join(listinput)

##del files
def removefilescommands(files:list)->list:
    """
    Giving a list of files it will give back the commands necessary for deleting such files.
    :param files: the list of files
    :return: it will return the list of commands
    """

    return ['rm -f ' + file for file in files]

def removefiles(files:list)-> None:
    """
    Its the wrapper of removefiles commands. if you want ot remove the files right now. also print it is removing.
    good to know which files are getting removed always. It will print remove even if the file do not exist in the first
    place
    :param files: the list of files
    :return: will not return anything just remove and print
    """
    commands=removefilescommands(files)
    for file in files:
        print ("This file is removed:",file)
    [os.system(command) for command in commands]
    return None

####hdf5 commands
def numpy2hdf5(x,outfile:str,dataset:str='mydata')-> None:
    """
    This will save the numpy to hdf5 format.
    :param x: the array (can be anything which can be converted to numpy array)
    :param outfile: the name of the output file
    :param dataset: name of the data set. default is mydata
    :return: will not return anything but will save the h5f file
    """
    import h5py
    import numpy
    x=numpy.array(x)
    f = h5py.File(outfile, 'w')
    h5f = f.create_dataset(dataset, data=x)
    f.close()

####importing functions
def loading_def_4m_file(filepath:str,defname:str)-> Optional[Callable]:
    """
    loading defintion from a file. some of the mar deprecate. will check later.
    :param filepath: the full path of the .py file
    :param defname: the name of the definition
    :return: will return the func if exist in the file. if not will return None
    """
    import importlib,imp
    if not os.path.isfile(filepath):
        print ('The file count not be found to load. Please check:',filepath)
    modname = filenamewithoutextension(filepath)
    mod = imp.load_source(modname, filepath)
    if defname in dir(mod):
        spec = importlib.util.spec_from_file_location(modname, filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, defname)
    else:
        return None
##shell script commands
def creatingfolders(specificfolder:str)-> None:
    """
    As the name suggest it will create a folder if the folder do not exist. As simple as that
    :param specificfolder: The folder needs to be created
    :return: will not return anything. Either it will create if the folder do not exist or not return anything
    """
    import os
    specificfolder = os.path.expanduser(specificfolder)
    if not os.path.exists(specificfolder):
        os.makedirs(specificfolder)