import itertools
from multiprocessing import Pool as ThreadPool
# type hint for readability
from typing import Callable, List, Dict, Union

import numpy
import pandas
# My stuff
from Common import Misc

from SFS.Class import Range2UniformPrior


class MsPrime2CRT:
    """
    Given a msprime demographic python file returning DemographyDebugger and priors, it can produce Coalescence Rate
    Trajectory (CRT) out of it.
    """

    def __new__(cls, sim_func: Callable, params_file: str, samples: str, gen_file: str,
                threads: int = 1) -> pandas.DataFrame:
        """
        This will call the wrapper function for MsPrime2CRT so the class will behave like a function

        :param sim_func: the msprime demography func which will simulate a given demography using
        msprime.DemographyDebugger and return it
        :param params_file: the csv file where parameters are written. All the priors for the parameters on which the
        simulation will run. Should be "," comma separated csv format. Different rows signify different run.
        columns different parameters
        :param samples: The number of inds per populations to run simulation. All the output populations should be
        mentioned in the inds. again separated by inds1,inds2. remember 1 inds = 2 haplotypes. thus from 5 inds you
        would get total 11 (0 included) different allele counts
        :param gen_file: The generations of time step at which point the CRT will be calculated. Every line signifies
        different generations steps. Should be in increasing order. Does not have to be integer and should not have
        header
        :param threads: the number of threads to run parallel
        :return: will return a pandas dataframe with parameters and crt together
        """
        return cls.wrapper(sim_func=sim_func, params_file=params_file, samples=samples, gen_file=gen_file,
                           threads=threads)

    @classmethod
    def wrapper(cls, sim_func: Callable, params_file: str, samples: str, gen_file: str,
                threads: int = 1) -> pandas.DataFrame:
        """
        the wrapper for the class. This will autamtically run every line coming from Priors.csv and will produce the CRT
        which can then be used as input in the NN
        :param sim_func: the msprime demography func which will simulate a given demography using
        msprime.DemographyDebugger and return it
        :param params_file: the csv file where parameters are written. All the priors for the parameters on which the
        simulation will run. Should be "," comma separated csv format. Different rows signify different run.
        columns different parameters
        :param samples: The number of inds per populations to run simulation. All the output populations should be
        mentioned in the inds. again separated by inds1,inds2. remember 1 inds = 2 haplotypes. thus from 5 inds you
        would get total 11 (0 included) different allele counts
        :param gen_file: The generations of time step at which point the CRT will be calculated. Every line signifies
        different generations steps. Should be in increasing order. Does not have to be integer and should not have
        header
        :param threads: the number of threads to run parallel
        :return: will return a pandas dataframe with parameters and crt together
        """
        pop_indexes = cls.inds2pop_index(samples=samples)
        het_dict, het_name = cls.het_combinations_dict(pops=pop_indexes)
        hom_dict, hom_name = cls.hom_combinations_dict(pops=pop_indexes)
        gen = pandas.read_csv(gen_file, header=None).iloc[:, 0].values
        crt_header = cls.crt_header_creation(lineage_combs=hom_name + het_name, gen=gen)
        paramsdf = pandas.read_csv(params_file, index_col=False).dropna()
        pool = ThreadPool(threads)
        input = zip(itertools.repeat(sim_func), paramsdf.values, itertools.repeat(gen),
                    itertools.repeat(hom_dict + het_dict))
        results = pool.starmap(cls.sims2crt, input)
        results = pandas.DataFrame([result.values.flatten() for result in results], columns=crt_header)
        params_sfs = pandas.concat([paramsdf, results], axis=1)
        return params_sfs

    @classmethod
    def inds2pop_index(cls, samples: str) -> List:
        """
        This will change the inds string (the number of individuals  per populations) and only keep those populations
        which has at least 1 samples. Thus having CRT ready for only those populations with count at least one samples

        :param samples: the individuals list in a string format. ind1,ind2,ind3
        :return: will return a list (Int64Index) of populations which have samples more than 0
        """
        samples = Range2UniformPrior.string_param_2_array(samples)
        samples = pandas.Series(samples).astype(int)
        pop_indexes = samples[samples > 0].index
        return pop_indexes

    @classmethod
    def het_combinations_dict(cls, pops: List) -> (List[Dict], List[str]):
        """
        this will create all the combination of dict (lineage) that is necessary to get the CRT for het or two
        populations.

        :param pops: the population number or pop indexes needed for the CRT
        :return: will return both the combination dict needed for lineage as well as the string format of those lineage
        in a list format which will be used later for header
        """
        combs = list(itertools.combinations(pops, 2))
        combs_dict = [{i: 1, j: 1} for i, j in combs]
        combs_name = [str(i) + "_" + str(j) for i, j in combs]
        return combs_dict, combs_name

    @classmethod
    def hom_combinations_dict(cls, pops: List) -> (List[Dict], List[str]):
        """
        this will create all the combination of dict (lineage) that is necessary to get the CRT for hom or single
        populations.

        :param pops: the population number or pop indexes needed for the CRT
        :return: will return both the combination dict needed for lineage as well as the string format of those lineage
        in a list format which will be used later for header
        """
        combs_dict = [{pop: 2} for pop in pops]
        combs_name = [str(pop) + "_" + str(pop) for pop in pops]
        return combs_dict, combs_name

    @classmethod
    def crt_header_creation(cls, lineage_combs: List, gen: List) -> List[str]:
        """
        This will create the necessary header for the CRT output. The format is Lineage0_Lineage1:Generation (i.e.
        0_1:100)

        :param lineage_combs: all the lineage combinations in a list format.
        :param gen: the generations for which CRT will be calculated. in a list of array format
        :return: will return the header for the CRT output
        """
        header = [str(lineage) + ":" + str(generation) for generation, lineage in
                  itertools.product(gen, lineage_combs)]
        return header

    @classmethod
    def sims2crt(cls, sim_func: Callable, params: Union[numpy.array, List], gen: List[float],
                 dict_combs: List[Dict]) -> pandas.DataFrame:
        """
        for the msprime function this will return CRT for combinations of populations and for the given time steps
        in generations

        :param sim_func: the msprime demography func which will simulate a given demography using
        msprime.DemographyDebugger
        :param params: All the parameters required for the model.  in numpy array or list
        :param gen: the generations for which CRT will be calculated. in a list of array format
        :param dict_combs: all the lineage combinations dict format needed as in msprime.coalescence_rate_trajectory
        in a list format.
        :return: This will return pandas.Dataframe format of the combinations of populations in columns and in rows as
        time steps of crt calculated by msprime
        """
        run = sim_func(params=params)
        crt = cls.crt_combinations(msprime_debug_run=run, dict_combs=dict_combs, gen=gen)
        return crt

    @classmethod
    def crt_combinations(cls, msprime_debug_run, dict_combs: List[Dict], gen: List) -> pandas.DataFrame:
        """
        This will automatically try all the combinations of lineages for crt and create a pandas dataframe format for
        crts

        :param msprime_debug_run: msprime.DemographyDebugger run
        :param dict_combs: all the lineage combinations dict format needed as in msprime.coalescence_rate_trajectory
        in a list format.
        :param gen: the generations for which CRT will be calculated. in a list of array format
        :return: This will return pandas.Dataframe format of the combinations of populations in columns and in rows as
        time steps of crt calculated by msprime
        """
        crt_all = []
        for j, dict_comb in enumerate(dict_combs):
            crt, _ = msprime_debug_run.coalescence_rate_trajectory(gen, dict_comb)
            crt_all.append(crt)
        combs_names = [Misc.joinginglistbyspecificstring(list(comb.keys()), '_') for comb in dict_combs]
        crt_all = pandas.DataFrame(crt_all, columns=gen, index=combs_names).transpose()
        return crt_all
