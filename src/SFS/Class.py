#!/usr/bin/python
"""
This file will hold all the classes for a specific case which is SFS. This should taken as example how to use ABC-DLS
rather than only way to use it
"""
import itertools
import os
import sys
from multiprocessing import Pool as ThreadPool
# type hint for readability
from typing import Optional, Union, List, Tuple, Callable

import allel
import numpy
import pandas

from Classes import ABC
from Classes import Misc


class VCF2SFS():
    """
    VCF 2 SFS (csv format) conversion using Sickit
    """

    def __new__(cls, vcffile: str, popfile: str, sfs_pop: Union[List[str], Tuple[str]], chunk_length: float = 1e6,
                out: Optional[str] = None) -> pandas.Series:
        """
        This will automatically call the wrapper function and to do the necessary work.
        :param vcffile: the path of the vcffile. can be zipped and indexed
        :param popfile: The file where population format is written. first column is individual, second column is
                population. the file should be sorted according the simulation paradigm
        :param sfs_pop: the name of pops. important for the order. example: [pop1,pop2,pop3]
        :param chunk_length: If two big for the memory use chunk size. relatively slow but no problem
                with ram. Defaults to 1e6. out
        :param out: in case you want to name the output. by default it will be your vcf file name (None)
        """

        return cls.wrapper(vcffile=vcffile, popfile=popfile, sfs_pop=sfs_pop, chunk_length=chunk_length, out=out)

    @classmethod
    def wrapper(cls, vcffile: str, popfile: str, sfs_pop: Union[List[str], Tuple[str]], chunk_length: float = 1e6,
                out: Optional[str] = None) -> pandas.Series:
        """This the wrapper function for VCF2SFS class. It will take necessary information vcf, popfile and sfs_pop and
        will convert to sfs (csv)

        Args:
            vcffile (str): the path of the vcffile. can be zipped and indexed
            popfile (str): The file where population format is written. first column is individual, second column is
                population. the file should be sorted according the simulation paradigm
            sfs_pop (list[str]): the name of pops. important for the order. example: [pop1,pop2,pop3]
            chunk_length (float, optional): If two big for the memory use chunk size. relatively slow but no problem
                with ram. Defaults to 1e6. out
            (str, optional): in case you want to name the output. by default it will be your vcf file name. Defaults to
                None.
            out (str) : in case you want to name the output. by default it will be your vcf file name (None)

        Returns:
            pandas.Series: SFS in pandas.Series format
        """
        popdict = cls.popdict4sickitallel(vcffile=vcffile, popfile=popfile)
        subpopdict = cls.subsetting_pop(popdict=popdict, sfs_pop=sfs_pop)
        fs_shape = [len(popdict[pop]) * 2 + 1 for pop in sfs_pop]
        sfs = cls.vcf2sfs(vcf=vcffile, fs_shape=fs_shape, sfs_pop=sfs_pop, subpopdict=subpopdict,
                          chunk_length=chunk_length)
        indx_names = cls.sfs2indexnames(sfs)
        sfs = pandas.Series(sfs.flatten(), index=indx_names)
        if out:
            cls.ss2csv(sfs, out + '.csv')
        return sfs

    @classmethod
    def popdict4sickitallel(cls, vcffile: str, popfile: str) -> dict:
        """
        To create a popdict worthy of sickit allel class

        :param vcffile: the path of the vcffile. can be zipped and indexed
        :param popfile: The file where population format is written. first column is individual, second column is
        population. the file should be sorted according the simulation paradigm
        :return: will return pop dict for sicikt. {pop1:[0,1,2..],pop2:[33,53..]}
        """
        pops = pandas.DataFrame(allel.read_vcf_headers(vcffile)[-1])
        pops['pop'] = Misc.adding_pop(pops, popfile)
        pops = pops.dropna()
        subpopdict = {}
        for pop in list(pops['pop'].unique()):
            subpopdict[pop] = list(pops[pops['pop'] == pop].index)
        return subpopdict

    @classmethod
    def subsetting_pop(cls, popdict: dict, sfs_pop: Union[List[str], Tuple[str]], ) -> dict:
        """
        sub-setting sickitalllele popdict to only relevant populations. important in case in the popfile there are other
        populations which were not used

        Args:
            popdict: pop dict for sicikt. {pop1:[0,1,2..],pop2:[33,53..]}
            sfs_pop: the name of pops. important for the order. example: [pop1,pop2,pop3]

        Returns: subdict of popdict. only keep the pops which are necessary {pop1:[0,1,2..],pop2:[33,53..]}
        """
        subpop = {}
        for pop in sfs_pop:
            if pop not in popdict:
                print(
                    'The population could not be found in popfile. Might be non ind left in the vcf file Please check:',
                    pop)
                sys.exit(1)
            else:
                subpop[pop] = popdict[pop]
        return subpop

    @classmethod
    def vcf2sfs(cls, vcf: str, fs_shape: Union[list, tuple], sfs_pop: Union[List[str], Tuple[str]], subpopdict: dict,
                chunk_length: float = 1e6) -> numpy.array:
        """
        practically the main func. this will directly convert a vcf file to sfs with required options

        Args:
            vcf: the path of the vcf file
            fs_shape: the sfs shape. if 5,5,5 inds, then fs_shape (11,11,11)
            sfs_pop: the name of pops. important for the order. example: [pop1,pop2,pop3]
            subpopdict: sickitallele accepteable popdict. it is necesary to keep only the pop that is necessary. thus
                sub
            chunk_length: to load the number of lines into memory. more takes more memory but will be faster

        Returns: will return the numpy version of sfs
        """
        callsets = allel.iter_vcf_chunks(vcf, fields=['variants/CHROM', 'variants/POS', 'variants/REF', 'variants/ALT',
                                                      'variants/AA', 'calldata/GT'], chunk_length=int(chunk_length))[-1]

        sfs = numpy.zeros(fs_shape, numpy.int_)
        for callset in callsets:
            alleles = pandas.DataFrame({'REF': callset[0]['variants/REF'], 'ALT': callset[0]['variants/ALT'][:, 0],
                                        'AA': callset[0]['variants/AA']})
            alleles = alleles.apply(lambda x: x.str.capitalize())
            derivedcount = pandas.DataFrame()
            ac = allel.GenotypeArray(
                callset[0]['calldata/GT']).count_alleles_subpops(subpopdict)
            for pop in sfs_pop:
                count = pandas.concat(
                    [alleles, pandas.DataFrame(ac[pop])], axis=1)
                derivedcount[pop] = cls.count2derived(count)
            # duplicate drop
            pos = pandas.DataFrame(
                {'CHR': callset[0]['variants/CHROM'], 'POS': callset[0]['variants/POS']})
            derivedcount = derivedcount.loc[pos.drop_duplicates(
                keep=False).index]
            derivedcount = derivedcount.dropna().astype(int)
            # removing monotones
            derivedcount = cls.remove_monotones(derivedcount=derivedcount, fs_shape=fs_shape)
            sfs += cls.freq_spectrum(derivedcount, fs_shape)
        return sfs

    @classmethod
    def count2derived(cls, count: pandas.DataFrame) -> pandas.Series:
        """
        convert count to derived count using sickit allele
        Args:
            count: pandas dataframe format of count alleles. should have [REF,ALT,AA, 0,1] where 0 is ref and 1 is
                alternative

        Returns: return pandas series of derived allele count

        """
        # no match
        count.loc[(count.REF != count.AA) & (
                count.ALT != count.AA), 1] = numpy.nan
        # flipped
        count.loc[count.ALT == count.AA,
        1] = count.loc[count.ALT == count.AA, 0]
        return count.loc[:, 1]

    @classmethod
    def remove_monotones(cls, derivedcount: pandas.DataFrame, fs_shape: Union[list, tuple],
                         mac: int = 1) -> pandas.DataFrame:
        """
        from the genome type array remove all the monotones (or any snp which is lee than mac minor allele count)
        Args:
            derivedcount: pandas data frame format of derived allele count. where every column is populations and rows
                are snps
            fs_shape: the expected shape of the final sfs. required to know the upper limit of allele count
            mac: the minor allele count which is necessary to pass this filter. default is 1 meaning all the monotones
                in the data will be removed

        Returns: will return genome type array (dataframe format) same as derived count but without those snps (rows)
            which is less than mac

        """
        derivedcount = derivedcount[derivedcount.sum(axis=1) >= mac]
        total_count = numpy.sum(numpy.array(fs_shape) - 1)
        derivedcount = derivedcount[derivedcount.sum(
            axis=1) <= total_count - mac]
        return derivedcount

    @classmethod
    def freq_spectrum(cls, freq: pandas.DataFrame, fs_shape: Union[list, tuple]) -> numpy.array:
        """
        freq file to freq spectrum conversion. We have freq count per snp convert it to spectrum file
        :param freq: per snp count per population
        :param fs_shape: expected frequency spectrum shape. which is generally 1+(2 x Sample size). thus if (5,10,15) then
        (11,21,31)
        :return: will return sfs (which is numpy.array
        """
        data = numpy.zeros(fs_shape, numpy.int_)
        indices, count = numpy.unique(freq, axis=0, return_counts=True)
        for index, x in numpy.ndenumerate(count):
            data[tuple(indices[index])] = x
        return data

    @classmethod
    def sfs2indexnames(cls, sfs: numpy.array) -> list:
        """
        String format of indexes of sfs. for example 0,0,1 sfs will be represented as 0_0_1. necessary for header if
        wished
        :param sfs: the multi-dimensional numpy array
        :return: will return one dimensional string formatted indexes
        """
        str_index = [Misc.joinginglistbyspecificstring(x, '_') for x, y in numpy.ndenumerate(sfs)]
        return str_index

    @classmethod
    def ss2csv(cls, sfs: pandas.Series, name: str) -> None:
        """
        pandas series format of ss saving in csv format. will save the ss in a single line. with columns denoting the
            ss with header
        Args:
            sfs: pandas.Series format of sfs. with index name
            name: the output name of the file

        Returns: will not return any thing but will save a csv file with the given name

        """
        pandas.DataFrame(sfs).transpose().to_csv(name, index=False)


class Range2UniformPrior():
    """
    This class will crate uniform distribution of parameters if the range for that parameters are given
    """

    def __new__(cls, upper: str, lower: str, variable_names: Optional[str] = None,
                repeats: Union[float, int] = 2e4) -> pandas.DataFrame:
        """
        This will call the wrapper function for Range2UniformPrior so the class will behave like a function

        :param upper: upper limit for the parameters. string format
        :param lower: lower limit for the parameters. string format
        :param variable_names: the names of the variables. string format
        :param repeats: number of repeats that you want create. can use float but will convert it to int
        :return: wii return the pandas dataframe format of parameters. whose columns are parameters and rows are
        different repeats (or instance)
        """
        return cls.wrapper(upper=upper, lower=lower, variable_names=variable_names, repeats=repeats)

    @classmethod
    def wrapper(cls, upper: str, lower: str, variable_names: Optional[str] = None,
                repeats: Union[float, int] = 2e4) -> pandas.DataFrame:
        """
        Main def for the class. given upper and lower limit it will create a uniform distribution of parameters which
        then can used as prior for our analysis (ABC-DLS)

        :param upper: upper limit for the parameters. string format
        :param lower: lower limit for the parameters. string format
        :param variable_names: the names of the variables. string format
        :param repeats: number of repeats that you want create. can use float but will convert it to int
        :return: wii return the pandas dataframe format of parameters. whose columns are parameters and rows are
        different repeats (or instance)
        """
        upper = cls.string_param_2_array(upper)
        lower = cls.string_param_2_array(lower)
        if variable_names:
            variable_names = cls.string_param_2_array(variable_names, type='str')
        else:
            variable_names = cls.givingname2parameters(params_length=len(upper))
        cls.checks(upper=upper, lower=lower, variable_names=variable_names)
        simulations = cls.simulating_params_4m_uni_dist((upper), (lower), variable_names, repeats=repeats)
        return simulations

    @classmethod
    def string_param_2_array(cls, paramsstr: str, type: str = 'float') -> list:
        """
        Just to break the param string send by the shell script to numpy array

        :param paramsstr: the string format of the parameters. comma separated '1122,112,.443..'
        :param type: to tell what will be the format for all the params. default is float
        :return: will send the numpy array [1122.0,112.0,0.443]
        """
        params = paramsstr.split(',')
        params = [param.strip() for param in params]
        params = list(map(eval(type), params))
        return params

    @classmethod
    def givingname2parameters(cls, params_length: int) -> list:
        """
        if the names of the variables were not given it will give them a name

        :param params_length: the number of parameters. length of upper or lower
        :return: will return a list with [param_1,param_2..]
        """
        return ['param_' + str(x) for x in range(1, params_length + 1)]

    @classmethod
    def checks(cls, upper: list, lower: list, variable_names: list) -> None:
        """
        This is basic checking system if Submit_SFSOutput4ABC is right before printing commands. Basically checks if the
         input length of upper_limit, lower_limit and variable_names are equal

        :param upper: upper limit for the parameters. list of numpy array [1,.2, ..]
        :param lower: lower limit for the parameters. list of numpy array [1,.2, ..]
        :param variable_names: the names of the variables. list of strings [params1,params2..]
        :return: will not return anything. but in case of non matching length it will print out and exit
        """
        if len(upper) != len(lower):
            print("length of upper and lower dimension do not match. please check")
            print(upper)
            print("length of upper", len(upper))
            print(lower)
            print("length of lower", len(lower))
            sys.exit(1)
        if len(upper) != len(variable_names):
            print("length of upper or lower dimension do not match with variable names please check. please check")
            print(variable_names)
            print("length of variable names", len(variable_names))
            print(upper)
            print("length of upper", len(upper))
            print(lower)
            print("length of lower", len(lower))
            sys.exit(1)

    @classmethod
    def simulating_params_4m_uni_dist(cls, upper: List[float], lower: List[float], variable_names: List[str],
                                      repeats: Union[float, int] = 1e4) -> pandas.DataFrame:
        """
        It will create the parameters from uniform distribution. where 0<U<1, and params= U(upper-lower)+lower in
        another word parameter will be with in upper limit and lower limit with the uniform distribution

        :param upper: upper limit for the parameters. list of numpy array [1,.2, ..]
        :param lower: lower limit for the parameters. list of numpy array [1,.2, ..]
        :param repeats: the number of repeats you want to simulate
        :return: will return pandas dataframe whose columns are parameters and rows are different runs of parameters.
        """
        upper = numpy.array(upper)
        lower = numpy.array(lower)
        repeats = int(repeats)
        uni = numpy.random.uniform(size=(repeats - 2, upper.shape[0]))
        dist = upper - lower
        uni = pandas.DataFrame(uni).mul(dist).add(lower)
        uni = pandas.concat([uni, pandas.DataFrame([upper, lower])]).sample(frac=1)
        uni.columns = variable_names
        return uni


class MsPrime2SFS:
    """
    Given a msprime demographic python file and priors it can produce sfs out of it.
    """

    def __new__(cls, sim_func: Callable, params_file: str, samples: str, total_length: float = 1e7,
                ldblock: float = 1e6, mut_rate: float = 1.45e-8, rec_rate: float = 1e-8,
                threads: int = 1) -> pandas.DataFrame:
        """
        This will call the wrapper function for MsPrime2SFS so the class will behave like a function

        :param sim_func: the msprime demography func which will simulate a given demography using msprime.simulate and
        return it
        :param params_file: the csv file where parameters are written. All the priors for the parameters on which the
        simulation will run. Should be "," comma separated csv format. Different rows signify different run.
        columns different parameters
        :param samples: The number of inds per populations to run simulation. All the output populations should be
        mentioned in the inds. again separated by inds1,inds2. remember 1 inds = 2 haplotypes. thus from 5 inds you
        would get total 11 (0 included) different allele counts
        :param total_length: total length of the genome. default is 3gb roughly the length of human genome
        :param ldblock: Length of simulated blocks. Default is 1mb
        :param mut_rate: the mutation rate. default is the one every body uses
        :param rec_rate: the recombination rate for msprime. does it matter for sfs?
        :param threads: the number of threads to run parallel
        :return: will return a pandas dataframe with parameters and sfs together
        """
        return cls.wrapper(sim_func=sim_func, params_file=params_file, samples=samples, total_length=total_length,
                           ldblock=ldblock, mut_rate=mut_rate, rec_rate=rec_rate, threads=threads)

    @classmethod
    def wrapper(cls, sim_func: Callable, params_file: str, samples: str, total_length: float = 1e7,
                ldblock: float = 1e6, mut_rate: float = 1.45e-8, rec_rate: float = 1e-8,
                threads: int = 1) -> pandas.DataFrame:
        """
        the wrapper for the class. this just wrapping around perline so that it can run it parallel.

        :param sim_func: the msprime demography func which will simulate a given demography using msprime.simulate and
        return it
        :param params_file: the csv file where parameters are written. All the priors for the parameters on which the
        simulation will run. Should be "," comma separated csv format. Different rows signify different run.
        columns different parameters
        :param samples: The number of inds per populations to run simulation. All the output populations should be
        mentioned in the inds. again separated by inds1,inds2. remember 1 inds = 2 haplotypes. thus from 5 inds you
        would get total 11 (0 included) different allele counts
        :param total_length: total length of the genome. default is 3gb roughly the length of human genome
        :param ldblock: Length of simulated blocks. Default is 1mb
        :param mut_rate: the mutation rate. default is the one every body uses
        :param rec_rate: the recombination rate for msprime. does it matter for sfs?
        :param threads: the number of threads to run parallel
        :return: will return a pandas dataframe with parameters and sfs together
        """
        samples = list(numpy.array(Range2UniformPrior.string_param_2_array(samples, 'int')) * 2)
        remainder_length = int(total_length % ldblock)
        replicates = int(total_length / ldblock)
        paramsdf = pandas.read_csv(params_file, index_col=False).dropna()
        pool = ThreadPool(threads)
        input = zip(itertools.repeat(sim_func), paramsdf.values, itertools.repeat(samples),
                    itertools.repeat(ldblock), itertools.repeat(mut_rate), itertools.repeat(int(replicates)),
                    itertools.repeat(rec_rate), itertools.repeat(remainder_length))
        results = pool.starmap(cls.perline, input)
        results = pandas.DataFrame([result.flatten() for result in results])
        results.columns = cls.haps2indexnames(haps=samples)
        params_sfs = pandas.concat([paramsdf, results], axis=1)
        return params_sfs

    @classmethod
    def perline(cls, sim_func: Callable, params, samples: Union[numpy.array, list], length: Union[float, int] = 1e6,
                mut_rate: float = 1.45e-8, replicates: Union[float, int] = 100, rec_rate: float = 1e-8,
                remainder_length: Union[float, int] = 0):
        """
        simulations2sfs per line or parameters. kind of the real wrapper. but we created another wrapper to take care of
        the multithreading

        :param sim_func: the msprime demography func which will simulate a given demography using msprime.simulate and
        return it
        :param params: All the parameters required for the model. except the samples. in numpy array or list
        :param samples: the number of samples in tuple format
        :param length:  the length of the chromosome for simulation. default is 1e6 which is fast enough and big enough
        :param mut_rate: the mutation rate. default is the one every body uses
        :param replicates:  the number of replicates. default 100 is fast
        :param rec_rate: the recombination rate for msprime. does it matter for sfs?
        :param remainder_length: as generally we are simulating 1Mb region. if the region is slightly larger than Mbs
            (like 1e6+100), it is defined by this remainder length (100bp in this case)
        :return: will return the multi-dimensional sfs in numpy format
        """
        replicates = int(replicates)
        sfs = cls.sims2sfs(sim_func=sim_func, params=params, replicates=replicates, length=length,
                           mut_rate=mut_rate, rec_rate=rec_rate, samples=samples)
        if remainder_length > 0:
            sfs += cls.sims2sfs(sim_func=sim_func, params=params, samples=samples,
                                length=int(remainder_length),
                                mut_rate=mut_rate, replicates=1, rec_rate=rec_rate)
        return sfs

    @classmethod
    def sims2sfs(cls, sim_func: Callable, params: Union[numpy.array, list], samples: Union[List[int], Tuple[int]],
                 length: float = 1e6, mut_rate: float = 1.45e-8, replicates: int = 100,
                 rec_rate: float = 1e-8) -> numpy.array:
        """
        msprime simulated function are converted to sfs after running for the parameters.

        :param sim_func: the msprime demography func which will simulate a given demography using msprime.simulate and
        return it
        :param params: All the parameters required for the model. except the samples. in numpy array or list
        :param samples: the number of samples in tuple format
        :param length: the length of the chromosome for simulation. default is 1e6 which is fast enough and big enough
        :param mut_rate: the mutation rate. default is the one every body uses
        :param replicates: the number of replicates. default 100 is fast
        :param rec_rate: the recombination rate for msprime. does it matter for sfs?
        :return: will return the multi-dimensional sfs in numpy format
        """
        samples_exist = [i for i in samples if i != 0]
        fs_shape = numpy.asarray(samples_exist) + 1
        sfs = numpy.zeros(fs_shape)
        sample_shape = numpy.split(numpy.arange(sum(samples_exist)), numpy.cumsum(list(samples_exist))[:-1])
        sims = sim_func(params, samples, length=length, mutation_rate=mut_rate, replicates=replicates,
                        recombination_rate=rec_rate)
        for sim in sims:
            sfs += sim.allele_frequency_spectrum(sample_shape, polarised=True, span_normalise=False)
        return sfs

    @classmethod
    def haps2indexnames(cls, haps):
        """
        to get the columns names of sfs from haplotype counts per populations
        :param haps: the number of haplotypes in tuple format
        :return: will return one dimensional string formatted indexes. for example derived count 0,1,2 for pop1,pop2,
        pop3 will be return as 0_1_2, all in list format
        """
        samples_exist = [i for i in haps if i != 0]
        fs_shape = numpy.asarray(samples_exist) + 1
        sfs = numpy.zeros(fs_shape)
        index = VCF2SFS.sfs2indexnames(sfs)
        return index


class MsPrime2SFS2c(MsPrime2SFS):
    """
    Given a msprime demographic python file and priors it can produce cross population sfs out of it.
    """

    def __new__(cls, sim_func: Callable, params_file: str, samples: str, total_length: float = 1e7,
                ldblock: float = 1e6, mut_rate: float = 1.45e-8, rec_rate: float = 1e-8,
                threads: int = 1) -> pandas.DataFrame:
        """
        This will call the wrapper function for MsPrime2SFS so the class will behave like a function

        :param sim_func: the msprime demography func which will simulate a given demography using msprime.simulate and
        return it
        :param params_file: the csv file where parameters are written. All the priors for the parameters on which the
        simulation will run. Should be "," comma separated csv format. Different rows signify different run.
        columns different parameters
        :param samples: The number of inds per populations to run simulation. All the output populations should be
        mentioned in the inds. again separated by inds1,inds2. remember 1 inds = 2 haplotypes. thus from 5 inds you
        would get total 11 (0 included) different allele counts
        :param total_length: total length of the genome. default is 3gb roughly the length of human genome
        :param ldblock: Length of simulated blocks. Default is 1mb
        :param mut_rate: the mutation rate. default is the one every body uses
        :param rec_rate: the recombination rate for msprime. does it matter for sfs?
        :param threads: the number of threads to run parallel
        :return: will return a pandas dataframe with parameters and cross population sfs together
        """
        return cls.wrapper(sim_func=sim_func, params_file=params_file, samples=samples, total_length=total_length,
                           ldblock=ldblock, mut_rate=mut_rate, rec_rate=rec_rate, threads=threads)

    @classmethod
    def perline(cls, sim_func: Callable, params, samples: Union[numpy.array, list], length: Union[float, int] = 1e6,
                mut_rate: float = 1.45e-8, replicates: Union[float, int] = 100, rec_rate: float = 1e-8,
                remainder_length: Union[float, int] = 0):
        """
        simulations to sfs2c per line or parameters. kind of the real wrapper. but we created another wrapper to take
        care of the multithreading

        :param sim_func: the msprime demography func which will simulate a given demography using msprime.simulate and
        return it
        :param params: All the parameters required for the model. except the samples. in numpy array or list
        :param samples: the number of samples in tuple format
        :param length:  the length of the chromosome for simulation. default is 1e6 which is fast enough and big enough
        :param mut_rate: the mutation rate. default is the one every body uses
        :param replicates:  the number of replicates. default 100 is fast
        :param rec_rate: the recombination rate for msprime. does it matter for sfs?
        :param remainder_length: as generally we are simulating 1Mb region. if the region is slightly larger than Mbs
            (like 1e6+100), it is defined by this remainder length (100bp in this case)
        :return:  it will return one dimentional numpy array with all the combinations of cross population sfs. The
        columns would be SFS(pop1,pop2) pop combination, then SFS(pop1,pop2).. sfs(popn-1,popn). according to
        itertools.combination(list,2)

        """
        replicates = int(replicates)
        sfs = cls.sims2sfs(sim_func=sim_func, params=params, replicates=replicates, length=length,
                           mut_rate=mut_rate, rec_rate=rec_rate, samples=samples)
        if remainder_length > 0:
            sfs += cls.sims2sfs(sim_func=sim_func, params=params, samples=samples,
                                length=int(remainder_length),
                                mut_rate=mut_rate, replicates=1, rec_rate=rec_rate)
        sfs = cls.sfs2sfs2c(sfs)
        return sfs

    @classmethod
    def sfs2sfs2c(cls, sfs: numpy.array) -> numpy.array:
        """
        multidimentional sfs will be converted to cross population sfs with all the combinations of populations.
        :param sfs: the multi-dimensional sfs in numpy format
        :return: will return one dimentional numpy array with all the combinations of cross population sfs.
        """
        haps = pandas.Series(sfs.shape) - 1
        combs = list(itertools.combinations(haps.index, 2))
        if len(haps) > 2:
            sfs2c = [cls.sfs_multid_2sfs_subd(sfs=sfs, keep=comb) for comb in combs]
            sfs2c = pandas.concat([pandas.Series(sfs.flatten()) for sfs in sfs2c])
            sfs2c = sfs2c.reset_index(drop=True)
            return sfs2c.values
        else:
            return sfs.flatten()

    @classmethod
    def sfs_multid_2sfs_subd(cls, sfs: numpy.array, keep: Union[List[int], Tuple[int]]) -> numpy.array:
        """
        main def for this class. given which population to keep to get a sfs with 2 dimension sfs for targeted
        population. can be used for any dimensional SFS
        :param sfs: the multi-dimensional sfs in numpy format
        :param keep: a list format with two populations that you want to keep. generally, the populations are the
        dimensions. meaning the two dimensions that you want to keep
        :return: a sub dimentional sfs
        """
        haps = pandas.Series(sfs.shape) - 1
        margin_axis = haps.drop(list(keep)).index
        for axis in sorted(margin_axis)[::-1]:
            sfs = sfs.sum(axis=axis)
        return sfs

    @classmethod
    def haps2indexnames(cls, haps: list) -> list:
        """
        Given the number of haplotypes this will give indexes for cross population SFS for all the combinations.
        for exampel 0:0_1:0,0:0_1:1,0:0_1:2,...popn-1:haplotypes_popn:haplotypes. differnt populations will be divided
        by _. The index of population will be in the left side of : and the right side is the allele count
        :param haps: the number of hapltoypes present per population in a list format
        :return: will retrun a list of indexes with 0:0_1:0,0:0_1:1,0:0_1:2,...popn-1:haplotypes_popn:haplotypes
        """
        samples_exist = [i for i in haps if i != 0]
        fs_shape = numpy.asarray(samples_exist) + 1
        combs = list(itertools.combinations(range(len(samples_exist)), 2))
        index = []
        for comb in combs:
            prods = itertools.product(range(fs_shape[comb[0]]), range(fs_shape[comb[1]]))
            for prod in prods:
                temp = f'{comb[0]}:{prod[0]}_{comb[1]}:{prod[1]}'
                index.append(temp)
        return index


class SFS_to_SFS2c:
    """
    To convert a multi-dimensional SFS csv file to cross population SFS csv file
    """

    @classmethod
    def main(cls, file: str, chunksize: int = 1000, params: int = 0) -> str:
        """
        The main wrapper of the class. This will read the multi dimensional SFS file by chunk size. Then it will
        separate params (if exist) and SFS columns. Then it will convert multidimentional SFS (per row) to cross
        populations SFS per row. After that it will concatenate with parameters and then pandas.df to to_csv format.
        This will be yielded, so that it does not saved totally on the ram.
        :param file: The path of the multi-dimensional SFS csv file with proper header coming from either
        Run_Prior2SFS.py or Run_VCF2SFS.py. Can be zipped
        :param chunksize: If too big for the memory use chunk size. relatively slow but no problem with ram
        :param params: The number of columns with parameters, the first few columns can be parameteres if it is
        coming from Run_Prior2SFS.py and rest are SFS. default is 0, meaning no parameters (for example coming from
        Run_VCF2SFS.py
        :return: will return parameters and cross population SFS in csv format
        """
        header = True
        for chunk in pandas.read_csv(file, chunksize=chunksize):
            params_df = chunk.iloc[:, :params]
            sfs = cls.sfs_marg2(sfs=chunk.iloc[:, params:])
            sfs.index = params_df.index
            chunk = pandas.concat([params_df, sfs], axis=1)
            yield chunk.to_csv(index=False, header=header)
            header = False

    @classmethod
    def sfs_marg2(cls, sfs: pandas.DataFrame) -> pandas.DataFrame:
        """
        This will convert multidimensional SFS in pandas.DF to cross population SFS format
        :param sfs: pandas.Dataframe foramt of SFS, whose every line is an independent run, and columns are the indexes
        of SFS in 0_1_2_3 format
        :return: will return all the combination of cross population SFS in a pandas.DataFrame format. every row is
        different run and every columns one cross population count
        """
        haps = cls.top_haplotypes(sfs.columns)
        combs = list(itertools.combinations(haps.index, 2))
        sfs = [cls.sfs2sfs_marg(sfs=sfs, margin_axis=haps.drop(list(comb)).index, haps=haps) for comb in combs]
        sfs = pandas.concat(sfs, axis=1)
        return sfs

    @classmethod
    def top_haplotypes(cls, sfs_columns: pandas.Series) -> pandas.Series:
        """
        from the SFS column names it will extract the number of haplotypes per populations present in the sfs
        :param sfs_columns: the columns name of SFS, should be like  0_1_2_3 format. i.e. first population derived
        allele count is 0, second population derived allele count is 1 so forth
        :return: will return the highest number of haplotpyes present in SFS per populsiton in pandas Series format
        """
        haps = pandas.DataFrame([col.split("_") for col in sfs_columns]).astype(int).max()
        return haps

    @classmethod
    def sfs2sfs_marg(cls, sfs, margin_axis, haps):
        """
        This will convert multidimentional SFS cross population SFS for a given two populations, which will be
        calculated from which axis/pop to drop (margin_axis), and total number of population (in haps)
        :param sfs: the SFS is pandas.Datamframe format
        :param margin_axis: the axis or population to be dropped. in a series foramt
        :param haps: the total number of haplotypes per population in a Series format
        :return: will return the two dimentional SFS for the two population
        """
        fs_shape = haps + 1
        sfs = sfs.values.reshape([sfs.shape[0]] + list(fs_shape))
        for axis in sorted(margin_axis)[::-1]:
            sfs = sfs.sum(axis=axis + 1)
        modified_haps = haps.drop(margin_axis)
        columns = cls.sfs_marg2index(sfs=sfs[0], modified_haps=modified_haps)
        sfs = sfs.reshape(sfs.shape[0], numpy.prod(modified_haps.values + 1))
        sfs = pandas.DataFrame(sfs, columns=columns)
        return sfs

    @classmethod
    def sfs_marg2index(cls, sfs: numpy.ndarray, modified_haps: pandas.Series) -> list:
        """
        getting the index of cross population SFS from SFS
        :param sfs: multidimentional SFS in numpy array format
        :param modified_haps: targetted hapltoypes count and their corresponding population in the index. in a
        pandas.Series format
        :return: will return all the indexes in a list format
        """
        out_index = []
        for index, values in numpy.ndenumerate(sfs):
            index_df = pandas.DataFrame([modified_haps.index, index])
            temp = [":".join(row) for row in index_df.astype(str).T.values]
            temp = "_".join(temp)
            out_index.append(temp)
        return out_index


class ABC_DLS_SMC_Snakemake(ABC.ABC_DLS_SMC):
    """
    Just to have extrac function required for Snakemake file for SFS creation
    """

    @classmethod
    def multiple_newrange2updatednewrange(cls, newrange_files: list, csvfile: str, params_length: int,
                                          decrease: float = 0.95, increase: float = 0.0,
                                          hardrange_file: Optional[str] = None,
                                          outfile: str = 'Newrange.csv') -> pandas.DataFrame:
        """
        main wrapper for multiple newrange to updated newrange pandas.df in snakemake. the update will output median of
        multiple values. This will make the parameter estimation slightly more robust. important mainly for multiple
        cycles. as one cycle one wrong prediction can make output of all consecutive runs wrong. This will make 10 run
        per cycle and from that will take the median which will make it more robust
        :param newrange_files: the files path in a list format
        :param csvfile: the main simulated SFS.csv path
        :param params_length: the number of columns for parameters present in csvfile
        :param decrease: minimum amount of decrease of the range needed to register as true. default is .95. lower means
        stronger filter
        :param increase: If you want to increase the new range. Important in case the newrange has missed the true
        parameters. The value is in fraction to distance of new range min and max (similar to decrease). a good value is
        5 times lower than 1-decrease. Only increase in case of no decrease is detected (>decrease). It is better to
        use hardrange to make it understand what should be the hard cut off if not newrange can be outside of possible
        values. default is  0
        :param hardrange_file: csv format of hardrange file path. Should have 3 columns. params_names, lower and upper
        limit. every row is define a parameters. no header. same as Newrange.csv. important to define what is possible
        for range
        :param outfile: the medina newrange should be saved in specific path
        :return: will return the median newrange df
        """
        newrange = cls.median_newrange(newrange_files)
        oldrange = cls.extract_oldrange(file=csvfile, params_length=params_length)
        newrange = cls.update_newrange_using_oldrange_hardrange(newrange=newrange, oldrange=oldrange, increase=increase,
                                                                decrease=decrease, hardrange_file=hardrange_file)
        newrange.to_csv(outfile, header=False)
        return newrange

    @classmethod
    def median_newrange(cls, files: list) -> pandas.DataFrame:
        """
        This will read all the newranges file from a list format and then caclulate the median of every parameters
        :param files:  the files path in a list format
        :return: wil return a pandas dataframe of newrange with median values of every parameters
        """
        alldata = [Misc.reading_csv_no_header(file).iloc[:, :3] for file in files]
        min_series = pandas.DataFrame([data.iloc[:, 1] for data in alldata]).median()
        max_series = pandas.DataFrame([data.iloc[:, 2] for data in alldata]).median()
        newrange = pandas.concat([min_series, max_series], axis=1)
        newrange.columns = ['min', 'max']
        newrange.index = list(alldata[0].iloc[:, 0])
        return newrange

    @classmethod
    def extract_oldrange(cls, file: str, params_length: int) -> pandas.DataFrame:
        """
        reading the simulated sfs file to get the parameters and from that creatign the oldrange file, which is
        basically the min and max values of every parameters
        :param file: the main simulated SFS.csv path
        :param params_length: the number of columns for parameters present in csvfile
        :return: will return the oldrange parameter values
        """
        params = pandas.read_csv(file, usecols=range(params_length))
        oldrange = pandas.concat([params.min(), params.max()], axis=1)
        oldrange.columns = ['min', 'max']
        return oldrange

    @classmethod
    def update_newrange_using_oldrange_hardrange(cls, newrange: pandas.DataFrame, oldrange: pandas.DataFrame,
                                                 decrease: float = 0.95, increase: float = 0.0,
                                                 hardrange_file: Optional[str] = None) -> pandas.DataFrame:
        """
        This will update the newrange Will check if it has decrease the newrange more than the decrease if not it will
        try to increase the range taken care of hardrange so that newrange is never outside of hardrange.
        :param newrange: the median newrange df
        :param oldrange: the oldrange parameter values
        :param decrease: minimum amount of decrease of the range needed to register as true. default is .95. lower means
        stronger filter
        :param increase: If you want to increase the new range. Important in case the newrange has missed the true
        parameters. The value is in fraction to distance of new range min and max (similar to decrease). a good value is
        5 times lower than 1-decrease. Only increase in case of no decrease is detected (>decrease). It is better to
        use hardrange to make it understand what should be the hard cut off if not newrange can be outside of possible
        values. default is  0
        :param hardrange_file: csv format of hardrange file path. Should have 3 columns. params_names, lower and upper
        limit. every row is define a parameters. no header. same as Newrange.csv. important to define what is possible
        for range
        :return: will return the newrange with updated values as well as the amount of improvement for this cycle
        """
        newrange = cls.updating_newrange(newrange=newrange, oldrange=oldrange, decrease=decrease)
        if increase > 0:
            if hardrange_file:
                hardrange = pandas.read_csv(hardrange_file, index_col=0, header=None, names=['', 'min', 'max'],
                                            usecols=[0, 1, 2])
            else:
                hardrange = pandas.DataFrame()
            newrange = cls.noise_injection_update(newrange=newrange, increase=increase, hardrange=hardrange,
                                                  oldrange=oldrange, decrease=decrease)
        return newrange

    @classmethod
    def narrowing_input(cls, paramsnumbers: int, inputfile: str, rangefile: str, folder: str = '') -> int:
        """
        Narrowing the All.csv file with the range that is calculated by SFS so that it can be used for next cycle

        :param paramsnumbers: the number of parameters in the files. so that the ss part is separated
        :param inputfile: generally all.csv file. whose first few columns are the parameters and last few columns are
            ss
        :param rangefile: the new range in pandas dataframe format. columns should be max and min and indexes should be
                the parameters
        :param folder: to define the output folder. default is '' meaning current folder
        :return: will return the number of lines present in Narrowed.csv file which is created by the function
        """
        newrange = pandas.read_csv(rangefile, header=None, index_col=0).iloc[:, :2]
        newrange.columns = ['min', 'max']
        if Misc.getting_line_count(inputfile) > 0:
            params = pandas.read_csv(inputfile, usecols=range(paramsnumbers), header=None)
            linenumbers = (cls.narrowing_params(params=params, parmin=newrange['min'],
                                                parmax=newrange['max'])) - 1
            temp = cls.extracting_by_linenumber(file=inputfile, linenumbers=linenumbers,
                                                outputfile=folder + 'Narrows.csv')
            if Misc.getting_line_count(temp) > 0:
                _ = cls.shufling_joined_models(inputcsv=temp, output=folder + 'Narrowed.csv', header=False)
                Misc.removefiles([folder + 'Narrows.csv'], printing=False)
            else:
                os.rename(temp, folder + 'Narrowed.csv')
        else:
            with open("Narrowed.csv", "w") as my_empty_csv:
                pass
        narrow_count = Misc.getting_line_count(folder + 'Narrowed.csv')
        return narrow_count

    @classmethod
    def remove_repeated_params(cls, inputfile: str, paramsnumbers: int, outputfile: str) -> str:
        """
        remove the repeated parameters lines. We donot need different runs for same simulated parameters

        :param inputfile: generally all.csv file. whose first few columns are the parameters and last few columns are
            ss
        :param paramsnumbers: the number of parameters in the files. so that the ss part is separated
        :param outputfile: the path of the output file
        :return: will return the path of outputfile
        """
        params = pandas.read_csv(inputfile, usecols=range(paramsnumbers), header=None)
        linenumbers = params.drop_duplicates().index.values
        temp = cls.extracting_by_linenumber(file=inputfile, linenumbers=linenumbers,
                                            outputfile=outputfile)
        return temp

    @classmethod
    def lmrd4mcsv(cls, hardrange_file: str, newrange_file: str) -> float:
        """
        This will read hardrange_file and newrange_file and calcualte log of mean range decrease (lmrd). So you know how
        much improvement you got in this cycle
        :param hardrange_file: csv format of hardrange file path. Should have 3 columns. params_names, lower and upper
            limit. every row is define a parameters. no header. important when used increase as not to go awry for
            simulation parameters
        :param newrange_file: the updated range of parameters. same as hardrange_file
        :return: will return the lmrd in float
        """
        hardrange = pandas.read_csv(hardrange_file, index_col=0, header=None, names=['lower', 'upper'])
        newrange = pandas.read_csv(newrange_file, index_col=0, header=None, names=['lower', 'upper', 'imp'])

        lmrd = cls.lmrd_calculation(newrange=newrange, hardrange=hardrange)
        return lmrd
