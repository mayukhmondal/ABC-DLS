#!/usr/bin/python
"""
This file will hold all the classes for a specific case which is SFS. This should taken as example how to use ABC-TFK
rather than only way to use it
"""
import sys
# type hint for readability
from typing import Optional, Union, List, Tuple

import allel
import numpy
import pandas

from Classes import Misc


class VCF2SFS():
    """
    VCF 2 SFS (csv format) conversion using Sickit
    """

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
    def subsetting_pop(cls, popdict: dict, sfs_pop: Union[List[str], Tuple[str]],) -> dict:
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
