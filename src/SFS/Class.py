#!/usr/bin/python
"""
This file will hold all the classes for a specific case which is SFS. This should taken as example how to use ABC-TFK
rather than only way to use it
"""
import itertools
import math
import msprime
import numpy
import sys
from multiprocessing import Pool as ThreadPool
# type hint for readability
from typing import Optional, Union, List, Tuple

import allel
import pandas

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
        else:
            cls.ss2csv(sfs, Misc.filenamewithoutextension_checking_zipped(
                vcffile) + '.csv')
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
        return cls.wrapper(upper=upper, lower=lower, variable_names=variable_names, repeats=repeats)

    @classmethod
    def wrapper(cls, upper: str, lower: str, variable_names: Optional[str] = None,
                repeats: Union[float, int] = 2e4) -> pandas.DataFrame:
        """
        Main def for the class. given upper and lower limit it will create a uniform distribution of parameters which
        then can used as prior for our analysis (ABC-TFK)

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

    @classmethod
    def wrapper(cls, sim_func, params_file, samples, total_length=1e7, ldblock=1e6, mut_rate=1.45e-8, rec_rate=1e-8,
                threads=1):
        samples=list(numpy.array(Range2UniformPrior.string_param_2_array(samples,'int'))*2)
        remainder_length = int(total_length % ldblock)
        replicates = int(total_length / ldblock)
        paramsdf = pandas.read_csv(params_file, index_col=False).dropna()
        pool = ThreadPool(threads)
        input = zip(itertools.repeat(sim_func), paramsdf.values, itertools.repeat(samples),
                    itertools.repeat(ldblock), itertools.repeat(mut_rate), itertools.repeat(int(replicates)),
                    itertools.repeat(rec_rate), itertools.repeat(remainder_length))
        results = pool.starmap(cls.perline, input)
        # results = itertools.starmap(cls.perline, input)
        params_sfs = pandas.concat([paramsdf, pandas.DataFrame([result.flatten() for result in results])], axis=1)
        return params_sfs

    @classmethod
    def perline(cls, sim_func, params, samples, length=1e6, mut_rate=1.45e-8, replicates=1e2,
                rec_rate=1e-8, remainder_length=0):
        replicates = int(replicates)
        sfs = cls.run_simulation(sim_func=sim_func, params=params, replicates=replicates, length=length,
                                 mut_rate=mut_rate, rec_rate=rec_rate, samples=samples)
        if remainder_length > 0:
            sfs = sfs + cls.run_simulation(sim_func=sim_func.msprime_func, params=params, samples=samples,
                                           length=int(remainder_length),
                                           mut_rate=mut_rate, replicates=1, rec_rate=rec_rate)
        return sfs

    @classmethod
    def run_simulation(cls, sim_func, params, samples, length=1e6, mut_rate=1.45e-8, replicates=1e2, rec_rate=1e-8):
        sims = sim_func(params, samples, length=length, mutation_rate=mut_rate, replicates=replicates,
                        recombination_rate=rec_rate)
        sfs = cls.genotypes_to_fs(sims, samples)
        return sfs

    @classmethod
    def genotypes_to_fs(cls, sims, pop_samples):
        """
        As msprime instead of genotypes all together sent it via iterator. Thus it is a good idea do it line by line thus not using too much ram
        :param genotypes: the genotype iterator from msprime
        :param pop_samples: the number of samples for every populations. POP1,POP2..=> 20,20.. in a tupple format
        :return: will return the sfs dadi style
        """
        fs_shape = numpy.asarray(pop_samples) + 1
        all_data = numpy.zeros(fs_shape)
        sample_shape = numpy.split(numpy.arange(sum(pop_samples)), numpy.cumsum(list(pop_samples))[:-1])
        for sim in sims:
            all_data = all_data + sim.allele_frequency_spectrum(sample_shape, polarised=True, span_normalise=False)
        return all_data