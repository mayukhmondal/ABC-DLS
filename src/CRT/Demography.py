#!/usr/bin/python
import math

import msprime
import numpy


def OOA(params):
    """
    This is the Model from Gravel et al. 2013 PNAS for Out of Africa

    :param params: all the parameters necessary for this model in a list or array format
        N_A: The ancestral effective population size
        N_AF: Modern effective population size of Africa population
        N_EU: Modern effective population size of European population
        N_AS: Modern effective population size of East Asian population
        N_EU0: Effective population size of European population before exponential growth.
        N_AS0: Effective population size of East Asian population before exponential growth.
        N_B: Effective population size of Out of Africa (OOA) populations
        T_EU_AS: Time interval for separation of European and East Asian from now. in kilo year ago (kya)
        T_B: Time interval for separation between Africa and OOA populations from T_EU_AS. in kya
        T_AF: Time interval for decrease of effective population size of African population to ancestral effective
            population size from T_B. in kya
        m_AF_B: Bi-directional migration rate between African and OOA populations (x10^-5)
        m_AF_EU: Bi-directional migration rate between African and European populations (x10^-5)
        m_AF_AS: Bi-directional migration rate between African and East Asian populations (x10^-5)
        m_EU_AS: Bi-directional migration rate between European and East Asian populations (x10^-5)
    :param inds: the number of haplotypes per populations for example (10,10,10)
    :param length: the length of chromosome that has to be simulated. default is 1mb region
    :param mutation_rate: the amount of mutation rate. default is 1.45x10^-8 per generation per base
    :param recombination_rate: the amount of recombination rate. default is 10^-8 per generation per base
    :param replicates: the number of replicated of length chromosome. default is 300
    :return: will return the msprime DemographyDebugger. which then can be used to extract SFS
    """
    (N_A, N_AF, N_EU, N_AS, N_EU0, N_AS0, N_B, T_EU_AS, T_B, T_AF, m_AF_B, m_AF_EU, m_AF_AS, m_EU_AS) = params

    T_EU_AS, T_B, T_AF = numpy.array([T_EU_AS, T_B, T_AF]) * (1e3 / 29.0)
    m_AF_B, m_AF_EU, m_AF_AS, m_EU_AS = numpy.array([m_AF_B, m_AF_EU, m_AF_AS, m_EU_AS]) * 1e-5
    r_EU = (math.log(N_EU / N_EU0) / T_EU_AS)
    r_AS = (math.log(N_AS / N_AS0) / T_EU_AS)
    population_configurations = [
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N_AF),
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N_EU, growth_rate=r_EU),
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N_AS, growth_rate=r_AS)
    ]
    migration_matrix = [
        [0, m_AF_EU, m_AF_AS],
        [m_AF_EU, 0, m_EU_AS],
        [m_AF_AS, m_EU_AS, 0],
    ]
    demographic_events = [
        # CEU and CHB merge into B with rate changes at T_EU_AS
        msprime.MassMigration(
            time=T_EU_AS, source=2, destination=1, proportion=1.0),
        msprime.MigrationRateChange(time=T_EU_AS, rate=0),
        msprime.MigrationRateChange(
            time=T_EU_AS, rate=m_AF_B, matrix_index=(0, 1)),
        msprime.MigrationRateChange(
            time=T_EU_AS, rate=m_AF_B, matrix_index=(1, 0)),
        msprime.PopulationParametersChange(
            time=T_EU_AS, initial_size=N_B, growth_rate=0, population_id=1),
        msprime.PopulationParametersChange(
            time=T_EU_AS, growth_rate=0, population_id=2),
        # Population B merges into YRI at T_B
        msprime.MassMigration(
            time=T_B + T_EU_AS, source=1, destination=0, proportion=1.0),
        msprime.MigrationRateChange(time=T_B + T_EU_AS, rate=0),
        # Size changes to N_A at T_AF
        msprime.PopulationParametersChange(
            time=T_AF + T_B + T_EU_AS, initial_size=N_A, population_id=0)]
    geno = msprime.DemographyDebugger(population_configurations=population_configurations,
                                      demographic_events=demographic_events, migration_matrix=migration_matrix)
    return geno
