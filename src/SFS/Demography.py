#!/usr/bin/python
import collections
import math

import msprime
import numpy


def OOA(params, inds, length=1e6, mutation_rate=1.45e-8, recombination_rate=1e-8, replicates=300):
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
        T_EU_AS: Time interval for separation of European and East Asian from now. in kilo year (ky)
        T_B: Time interval for separation between Africa and OOA populations from T_EU_AS. in ky
        T_AF: Time interval for decrease of effective population size of African population to ancestral effective
            population size from T_B. in ky
        m_AF_B: Bi-directional migration rate between African and OOA populations (x10^-5)
        m_AF_EU: Bi-directional migration rate between African and European populations (x10^-5)
        m_AF_AS: Bi-directional migration rate between African and East Asian populations (x10^-5)
        m_EU_AS: Bi-directional migration rate between European and East Asian populations (x10^-5)
    :param inds: the number of haplotypes per populations for example (10,10,10)
    :param length: the length of chromosome that has to be simulated. default is 1mb region
    :param mutation_rate: the amount of mutation rate. default is 1.45x10^-8 per generation per base
    :param recombination_rate: the amount of recombination rate. default is 10^-8 per generation per base
    :param replicates: the number of replicated of length chromosome. default is 300
    :return: will return the msprime simulations. which then can be used to extract SFS
    """
    (N_A, N_AF, N_EU, N_AS, N_EU0, N_AS0, N_B, T_EU_AS, T_B, T_AF, m_AF_B, m_AF_EU, m_AF_AS, m_EU_AS) = params
    (n1, n2, n3) = inds

    T_EU_AS, T_B, T_AF = numpy.array([T_EU_AS, T_B, T_AF]) * (1e3 / 29.0)
    m_AF_B, m_AF_EU, m_AF_AS, m_EU_AS = numpy.array([m_AF_B, m_AF_EU, m_AF_AS, m_EU_AS]) * 1e-5
    r_EU = (math.log(N_EU / N_EU0) / T_EU_AS)
    r_AS = (math.log(N_AS / N_AS0) / T_EU_AS)
    population_configurations = [
        msprime.PopulationConfiguration(
            sample_size=n1, initial_size=N_AF),
        msprime.PopulationConfiguration(
            sample_size=n2, initial_size=N_EU, growth_rate=r_EU),
        msprime.PopulationConfiguration(
            sample_size=n3, initial_size=N_AS, growth_rate=r_AS)
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
    geno = msprime.simulate(
        population_configurations=population_configurations,
        migration_matrix=migration_matrix,
        demographic_events=demographic_events, length=length, mutation_rate=mutation_rate,
        num_replicates=replicates,
        recombination_rate=recombination_rate)
    return geno


# Back to Africa models
def SNDX(params, inds, length=1e6, mutation_rate=1.45e-8, recombination_rate=1e-8, replicates=300):
    """
    This is the simple out of Africa model with Neanderthal to OOA population, Denisova or Unknown to East Asia and
        African archaic to African populations.

    :param params: all the parameters necessary for this model in a list or array format
        N_A: The ancestral effective population size
        N_AF: Modern effective population size of Africa population
        N_EU: Modern effective population size of European population
        N_AS: Modern effective population size of East Asian population
        N_EU0: Effective population size of European population before exponential growth.
        N_AS0: Effective population size of East Asian population before exponential growth.
        N_B: Effective population size of Out of Africa (OOA) populations
        T_DM: Time interval for introgression in East Asian from Denisova or Unknown from now. in kilo year (ky)
        T_EU_AS: Time interval for separation of European and East Asian from T_DM. in ky
        T_NM: Time interval for introgression in OOA from Neanderthal from T_EU_AS. in ky
        T_XM: Time interval for introgression in African population from African archaic from T_EU_AS. in ky
        T_B: Time interval for separation between Africa and OOA populations from maximum between T_NM and T_XM.
            in ky
        T_AF: Time interval for decrease of effective population size of African population to ancestral effective
            population size from T_B. in ky
        T_N_D: Time interval for separation between Neanderthal and Denisova or Unknwon from now. in ky
        T_H_A: Time interval for separation between Neanderthal and modern humans from T_N_D. in ky
        T_H_X: Time interval for separation between African archaic and modern humans from now. in ky
        NMix: the fraction of introgression happened to OOA populations.
        DMix: the fraction of introgression happened to East Asians
        XMix: the fraction of introgression happened to African populations
    :param inds: the number of haplotypes per populations for example (10,10,10)
    :param length: the length of chromosome that has to be simulated. default is 1mb region
    :param mutation_rate: the amount of mutation rate. default is 1.45x10^-8 per generation per base
    :param recombination_rate: the amount of recombination rate. default is 10^-8 per generation per base
    :param replicates: the number of replicated of length chromosome. default is 300
    :return: will return the msprime simulations. which then can be used to extract SFS
    """
    (N_A, N_AF, N_EU, N_AS, N_EU0, N_AS0, N_B, T_DM, T_EU_AS, T_NM, T_XM, T_B,
     T_AF, T_N_D, T_H_A, T_H_X, NMix, DMix, XMix) = params
    (n1, n2, n3) = inds
    T_DM, T_EU_AS, T_NM, T_XM, T_B, T_AF, T_N_D, T_H_A, T_H_X = numpy.array(
        [T_DM, T_EU_AS, T_NM, T_XM, T_B, T_AF, T_N_D, T_H_A, T_H_X]) * (1e3 / 29.0)
    AFR, EUR, ASN, NEA, DEN, XAF = 0, 1, 2, 3, 4, 5
    events = {}
    r_EU = (math.log(N_EU / N_EU0) / (T_DM + T_EU_AS))
    r_AS = (math.log(N_AS / N_AS0) / (T_DM + T_EU_AS))
    population_configurations = [
        msprime.PopulationConfiguration(
            sample_size=n1, initial_size=N_AF),
        msprime.PopulationConfiguration(
            sample_size=n2, initial_size=N_EU, growth_rate=r_EU),
        msprime.PopulationConfiguration(
            sample_size=n3, initial_size=N_AS, growth_rate=r_AS),
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N_A),
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N_A),
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N_A)
    ]

    # Denisova or unknown admixture
    events['deni_intro_asn'] = T_DM
    deni_intro_asn = [msprime.MassMigration(
        time=events['deni_intro_asn'], source=ASN, destination=DEN, proportion=DMix)]
    # CEU and CHB merge into B with rate changes at T_EU_AS
    events['split_eu_as'] = events['deni_intro_asn'] + T_EU_AS
    split_eu_as = [
        msprime.MassMigration(
            time=events['split_eu_as'], source=ASN, destination=EUR, proportion=1.0),
        msprime.PopulationParametersChange(
            time=events['split_eu_as'], initial_size=N_B, growth_rate=0, population_id=EUR)]
    # introgression Nean
    events['nean_intro_eur'] = T_NM + events['split_eu_as']
    nean_intro_eur = [msprime.MassMigration(
        time=events['nean_intro_eur'], source=EUR, destination=NEA, proportion=NMix)]

    # introgression XAFR
    events['xafr_intro_afr'] = T_XM + events['split_eu_as']
    xafr_intro_afr = [msprime.MassMigration(
        time=events['xafr_intro_afr'], source=AFR, destination=XAF, proportion=XMix)]
    # Population B merges into YRI at T_B
    events['split_afr_ooa'] = max(events['nean_intro_eur'], events['xafr_intro_afr']) + T_B
    split_afr_ooa = [
        msprime.MassMigration(
            time=events['split_afr_ooa'], source=EUR, destination=AFR, proportion=1.0)]
    events['ancestral_size'] = events['split_afr_ooa'] + T_AF
    # Size changes to N_A at T_AF
    ancestral_size = [msprime.PopulationParametersChange(
        time=events['ancestral_size'], initial_size=N_A, population_id=AFR)]

    # Denisova or Unknwon merging with Neanderthal
    events['neanderthal_denisova'] = T_N_D
    neanderthal_denisova = [msprime.MassMigration(
        time=events['neanderthal_denisova'], source=DEN, destination=NEA, proportion=1.0),
        msprime.PopulationParametersChange(
            time=events['neanderthal_denisova'], initial_size=N_A, population_id=NEA)]
    # Neanderthal merging with humans
    events['human_neanderthal'] = T_N_D + T_H_A
    human_neanderthal = [msprime.MassMigration(
        time=events['human_neanderthal'], source=NEA, destination=AFR, proportion=1.0)]
    # XAFR merging with humans
    events['human_xafr'] = T_H_X
    human_xafr = [msprime.MassMigration(
        time=events['human_xafr'], source=XAF, destination=AFR, proportion=1.0)]

    demographic_events = []
    for event in collections.OrderedDict(sorted(events.items(), key=lambda x: x[1])).keys():
        demographic_events = demographic_events + eval(event)
    geno = msprime.simulate(
        population_configurations=population_configurations,
        demographic_events=demographic_events, length=length, mutation_rate=mutation_rate,
        num_replicates=replicates,
        recombination_rate=recombination_rate)
    return geno


def BNDX(params, inds, length=1e6, mutation_rate=1.45e-8, recombination_rate=1e-8, replicates=300):
    """
    This is the back to Africa model with Neanderthal to OOA population, Denisova or Unknown to East Asia and
        African archaic to African populations.

    :param params: all the parameters necessary for this model in a list or array format
        N_A: The ancestral effective population size
        N_AF: Modern effective population size of Africa population
        N_EU: Modern effective population size of European population
        N_AS: Modern effective population size of East Asian population
        N_EU0: Effective population size of European population before exponential growth.
        N_AS0: Effective population size of East Asian population before exponential growth.
        N_BC: Effective population size of Back to Africa migrated population.
        N_B: Effective population size of Out of Africa (OOA) populations
        N_AF0: Effective population size of African populations before Back to Africa migration
        T_DM: Time interval for introgression in East Asian from Denisova or Unknown from now. in kilo year (ky)
        T_EU_AS: Time interval for separation of European and East Asian from T_DM. in ky
        T_NM: Time interval for introgression in OOA from Neanderthal from T_EU_AS. in ky
        T_XM: Time interval for introgression in African population from African archaic from T_EU_AS. in kya
        T_Mix: Time interval for mixing with Back to Africa population from T_EU_AS. in ky
        T_Sep: Time interval for separation of Back to Africa population from OOA from T_Mix. in ky
        T_B: Time interval for separation between Africa and OOA populations from maximum between T_NM, T_XM and
            T_Sep. in ky
        T_AF: Time interval for decrease of effective population size of African population to ancestral effective
            population size from T_B. in ky
        T_N_D: Time interval for separation between Neanderthal and Denisova or Unknown from now. in ky
        T_H_A: Time interval for separation between Neanderthal and modern humans from T_N_D. in ky
        T_H_X: Time interval for separation between African archaic and modern humans from now. in ky
        Mix: the fraction of African genome is replaced but Back to Africa population
        NMix: the fraction of introgression happened to OOA populations.
        DMix: the fraction of introgression happened to East Asians
        XMix: the fraction of introgression happened to African populations
    :param inds: the number of haplotypes per populations for example (10,10,10)
    :param length: the length of chromosome that has to be simulated. default is 1mb region
    :param mutation_rate: the amount of mutation rate. default is 1.45x10^-8 per generation per base
    :param recombination_rate: the amount of recombination rate. default is 10^-8 per generation per base
    :param replicates: the number of replicated of length chromosome. default is 300
    :return: will return the msprime simulations. which then can be used to extract SFS
    """
    (N_A, N_AF, N_EU, N_AS, N_EU0, N_AS0, N_BC, N_B, N_AF0, T_DM, T_EU_AS, T_NM,
     T_XM, T_Mix, T_Sep, T_B, T_AF, T_N_D, T_H_A, T_H_X, Mix, NMix, DMix,
     XMix) = params
    (n1, n2, n3) = inds
    T_DM, T_EU_AS, T_NM, T_XM, T_Mix, T_Sep, T_B, T_AF, T_N_D, T_H_A, T_H_X = numpy.array(
        [T_DM, T_EU_AS, T_NM, T_XM, T_Mix, T_Sep, T_B, T_AF, T_N_D, T_H_A, T_H_X]) * (1e3 / 29.0)

    events = {}
    AFR, EUR, ASN, GST, NEA, DEN, XAF = 0, 1, 2, 3, 4, 5, 6

    r_EU = (math.log(N_EU / N_EU0) / (T_DM + T_EU_AS))
    r_AS = (math.log(N_AS / N_AS0) / (T_DM + T_EU_AS))
    population_configurations = [
        msprime.PopulationConfiguration(
            sample_size=n1, initial_size=N_AF),
        msprime.PopulationConfiguration(
            sample_size=n2, initial_size=N_EU, growth_rate=r_EU),
        msprime.PopulationConfiguration(
            sample_size=n3, initial_size=N_AS, growth_rate=r_AS),
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N_BC),
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N_A),
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N_A),
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N_A)
    ]

    # Denisova or unknown admixture
    events['deni_intro_asn'] = T_DM
    deni_intro_asn = [msprime.MassMigration(
        time=events['deni_intro_asn'], source=ASN, destination=DEN, proportion=DMix)]
    # CEU and CHB merge into B with rate changes at T_EU_AS
    events['split_eu_as'] = events['deni_intro_asn'] + T_EU_AS
    split_eu_as = [msprime.MassMigration(
        time=events['split_eu_as'], source=ASN, destination=EUR, proportion=1.0),
        msprime.PopulationParametersChange(
            time=events['split_eu_as'], initial_size=N_B, growth_rate=0, population_id=EUR),
        msprime.MigrationRateChange(time=events['split_eu_as'], rate=0)]
    # introgression
    events['nean_intro_eur'] = T_NM + events['split_eu_as']
    nean_intro_eur = [msprime.MassMigration(
        time=events['nean_intro_eur'], source=EUR, destination=NEA, proportion=NMix)]
    # introgression XAFR
    events['xafr_intro_afr'] = T_XM + events['split_eu_as']
    xafr_intro_afr = [msprime.MassMigration(
        time=events['xafr_intro_afr'], source=AFR, destination=XAF, proportion=XMix)]

    # back migration
    events['back_migration'] = events['split_eu_as'] + T_Mix
    back_migration = [msprime.MassMigration(time=events['back_migration'], source=AFR,
                                            destination=GST, proportion=Mix), msprime.PopulationParametersChange(
        time=events['back_migration'], initial_size=N_AF0, population_id=AFR)]
    # spearation between back and OOA
    events['split_ooa_back'] = events['back_migration'] + T_Sep
    split_ooa_back = [msprime.MassMigration(time=events['split_ooa_back'], source=GST,
                                            destination=EUR, proportion=1.0)]
    # Population B merges into YRI at T_B
    events['split_afr_ooa'] = max(events['split_ooa_back'], events['xafr_intro_afr'],
                                  events['nean_intro_eur']) + T_B
    split_afr_ooa = [msprime.MassMigration(
        time=events['split_afr_ooa'], source=EUR, destination=AFR, proportion=1.0),
        msprime.MigrationRateChange(time=events['split_afr_ooa'], rate=0)]
    # Size changes to N_A at T_AF
    events['ancestral_size'] = events['split_afr_ooa'] + T_AF
    ancestral_size = [msprime.PopulationParametersChange(
        time=events['ancestral_size'], initial_size=N_A, population_id=AFR)]
    # Denisova or Unknwon merging with Neanderthal
    events['neanderthal_denisova'] = T_N_D
    neanderthal_denisova = [msprime.MassMigration(
        time=events['neanderthal_denisova'], source=DEN, destination=NEA, proportion=1.0),
        msprime.PopulationParametersChange(
            time=events['neanderthal_denisova'], initial_size=N_A, population_id=NEA)]
    # Neanderthal merging with humans
    events['human_neanderthal'] = T_N_D + T_H_A
    human_neanderthal = [msprime.MassMigration(
        time=events['human_neanderthal'], source=NEA, destination=AFR, proportion=1.0)]
    # XAFR merging with humans
    events['human_xafr'] = T_H_X
    human_xafr = [msprime.MassMigration(
        time=events['human_xafr'], source=XAF, destination=AFR, proportion=1.0)]

    demographic_events = []
    for event in collections.OrderedDict(sorted(events.items(), key=lambda x: x[1])).keys():
        demographic_events = demographic_events + eval(event)
    geno = msprime.simulate(
        population_configurations=population_configurations,
        demographic_events=demographic_events, length=length, mutation_rate=mutation_rate,
        num_replicates=replicates,
        recombination_rate=recombination_rate)

    return geno


def MNDX(params, inds, length=1e6, mutation_rate=1.45e-8, recombination_rate=1e-8, replicates=300):
    """
        This is the mix OOA model with Neanderthal to OOA population, Denisova or Unknown to East Asia and
            African archaic to African populations.

        :param params: all the parameters necessary for this model in a list or array format
            N_A: The ancestral effective population size
            N_AF: Modern effective population size of Africa population
            N_EU: Modern effective population size of European population
            N_AS: Modern effective population size of East Asian population
            N_EU0: Effective population size of European population before exponential growth.
            N_AS0: Effective population size of East Asian population before exponential growth.
            N_MX: Effective population size of Mix to OOA population (assume second OOA).
            N_B: Effective population size of Out of Africa (OOA) populations after admixture with mix population
            N_B0: Effective population size of Out of Africa before admixture (assume fist OOA).
            T_DM: Time interval for introgression in East Asian from Denisova or Unknown from now. in kilo year (ky)
            T_EU_AS: Time interval for separation of European and East Asian from T_DM. in ky
            T_NM: Time interval for introgression in OOA from Neanderthal from T_EU_AS. in ky
            T_XM: Time interval for introgression in African population from African archaic from T_EU_AS. in ky
            T_Mix: Time interval for mixing between OOA (OOA_1) and Mix (OOA_2) from T_EU_AS. in ky
            T_Sep: Time interval for separation of Mix population from Africa from T_Mix. in ky
            T_B: Time interval for separation between Africa and OOA populations from maximum between T_NM,T_XM
                and T_Sep. in ky
            T_AF: Time interval for decrease of effective population size of African population to ancestral effective
                population size from T_B. in ky
            T_N_D: Time interval for separation between Neanderthal and Denisova or Unknwon from now. in ky
            T_H_A: Time interval for separation between Neanderthal and modern humans from T_N_D. in ky
            T_H_X: Time interval for separation between African archaic and modern humans from now. in ky
            Mix: the fraction of OOA (OOA_1) genome is replaced by Mix population (OOA_2)
            NMix: the fraction of introgression happened to OOA populations.
            DMix: the fraction of introgression happened to East Asians
            XMix: the fraction of introgression happened to African populations
        :param inds: the number of haplotypes per populations for example (10,10,10)
        :param length: the length of chromosome that has to be simulated. default is 1mb region
        :param mutation_rate: the amount of mutation rate. default is 1.45x10^-8 per generation per base
        :param recombination_rate: the amount of recombination rate. default is 10^-8 per generation per base
        :param replicates: the number of replicated of length chromosome. default is 300
        :return: will return the msprime simulations. which then can be used to extract SFS
        """
    (N_A, N_AF, N_EU, N_AS, N_EU0, N_AS0, N_MX, N_B1, N_B2, T_DM, T_EU_AS,
     T_NM, T_XM, T_Mix, T_Sep, T_B, T_AF, T_N_D, T_H_A, T_H_X, Mix, NMix,
     DMix, XMix) = params
    (n1, n2, n3) = inds
    T_DM, T_EU_AS, T_NM, T_XM, T_Mix, T_Sep, T_B, T_AF, T_N_D, T_H_A, T_H_X = numpy.array(
        [T_DM, T_EU_AS, T_NM, T_XM, T_Mix, T_Sep, T_B, T_AF, T_N_D, T_H_A, T_H_X]) * (1e3 / 29.0)

    events = {}
    AFR, EUR, ASN, GST, NEA, DEN, XAF = 0, 1, 2, 3, 4, 5, 6

    r_EU = (math.log(N_EU / N_EU0) / (T_DM + T_EU_AS))
    r_AS = (math.log(N_AS / N_AS0) / (T_DM + T_EU_AS))
    population_configurations = [
        msprime.PopulationConfiguration(
            sample_size=n1, initial_size=N_AF),
        msprime.PopulationConfiguration(
            sample_size=n2, initial_size=N_EU, growth_rate=r_EU),
        msprime.PopulationConfiguration(
            sample_size=n3, initial_size=N_AS, growth_rate=r_AS),
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N_MX),
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N_A),
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N_A),
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N_A)
    ]

    # Denisova or unknown admixture
    events['deni_intro_asn'] = T_DM
    deni_intro_asn = [msprime.MassMigration(
        time=events['deni_intro_asn'], source=ASN, destination=DEN, proportion=DMix)]
    # CEU and CHB merge into B with rate changes at T_EU_AS
    events['split_eu_as'] = events['deni_intro_asn'] + T_EU_AS
    split_eu_as = [msprime.MassMigration(
        time=events['split_eu_as'], source=ASN, destination=EUR, proportion=1.0),
        msprime.PopulationParametersChange(
            time=events['split_eu_as'], initial_size=N_B1, growth_rate=0, population_id=EUR),
        msprime.MigrationRateChange(time=events['split_eu_as'], rate=0)]
    # NEAN introgression
    events['nean_intro_eur'] = T_NM + events['split_eu_as']
    nean_intro_eur = [msprime.MassMigration(
        time=events['nean_intro_eur'], source=EUR, destination=NEA, proportion=NMix)]

    # introgression XAFR
    events['xafr_intro_afr'] = T_XM + events['split_eu_as']
    xafr_intro_afr = [msprime.MassMigration(
        time=events['xafr_intro_afr'], source=AFR, destination=XAF, proportion=XMix)]
    # mix migration
    events['mix_migration'] = T_Mix + events['split_eu_as']
    mix_migration = [msprime.MassMigration(time=events['mix_migration'], source=EUR,
                                           destination=GST, proportion=Mix), msprime.PopulationParametersChange(
        time=events['mix_migration'], initial_size=N_B2, population_id=EUR)]

    # spearation between back and OOA
    events['split_afr_mix'] = events['mix_migration'] + T_Sep
    split_afr_mix = [msprime.MassMigration(time=events['split_afr_mix'], source=GST,
                                           destination=AFR, proportion=1.0)]
    # Population B merges into YRI at T_B
    events['split_afr_ooa'] = max(events['split_afr_mix'], events['xafr_intro_afr'], events['nean_intro_eur']) + T_B
    split_afr_ooa = [msprime.MassMigration(
        time=events['split_afr_ooa'], source=EUR, destination=AFR, proportion=1.0),
        msprime.MigrationRateChange(time=events['split_afr_ooa'], rate=0)]
    # Size changes to N_A at T_AF
    events['ancestral_size'] = events['split_afr_ooa'] + T_AF
    ancestral_size = [msprime.PopulationParametersChange(
        time=events['ancestral_size'], initial_size=N_A, population_id=AFR)]
    # Denisova or Unknwon merging with Neanderthal
    events['neanderthal_denisova'] = T_N_D
    neanderthal_denisova = [msprime.MassMigration(
        time=events['neanderthal_denisova'], source=DEN, destination=NEA, proportion=1.0),
        msprime.PopulationParametersChange(
            time=events['neanderthal_denisova'], initial_size=N_A, population_id=NEA)]
    # Neanderthal merging with humans
    events['human_neanderthal'] = T_N_D + T_H_A
    human_neanderthal = [msprime.MassMigration(
        time=events['human_neanderthal'], source=NEA, destination=AFR, proportion=1.0)]
    # XAFR merging with humans
    events['human_xafr'] = T_H_X
    human_xafr = [msprime.MassMigration(
        time=events['human_xafr'], source=XAF, destination=AFR, proportion=1.0)]

    demographic_events = []
    for event in collections.OrderedDict(sorted(events.items(), key=lambda x: x[1])).keys():
        demographic_events = demographic_events + eval(event)
    geno = msprime.simulate(
        population_configurations=population_configurations,
        demographic_events=demographic_events, length=length, mutation_rate=mutation_rate,
        num_replicates=replicates,
        recombination_rate=recombination_rate)
    return geno


# Papua New Guinean (PNG) models
def PNG_A_ND(params, inds, length=1e6, recombination_rate=1e-8, mutation_rate=1.25e-8, replicates=300):
    """
    This is the Papuan population is a sister group of East Asians.
    :param params: all the parameters necessary for this model in a list or array format
        N_A: The ancestral effective population size
        N_AF: Modern effective population size of Africa population
        N_EU: Modern effective population size of European population
        N_AS: Modern effective population size of East Asian population
        N_PA: Modern effective population size of PNG
        N_NE: Effective population size of Neanderthals
        N_DE: Effective population size of Denisovans
        N_EU0: Effective population size of European population before exponential growth.
        N_AS0: Effective population size of East Asian population before exponential growth.
        N_PA0: Effective population size of PNG population before exponential growth.
        N_B: Effective population size of Out of Africa (OOA) populations
        N_ND: Effective population size of ancestor of Neanderthals and Denisovans
        T_DPM: Time interval for introgression from Denisova to PNG population. in kilo year (ky)
        T_AS_PA: Time interval for separation of PNG and East Asian from T_DPM. in ky
        T_EU_AS: Time interval for separation of European and Asian (PNG+East Asian) from T_DPM. in ky
        T_NOM: Time interval for introgression from Neanderthal to Out of African (OOA) population from T_EU_AS. in ky
        T_B: Time interval for separation between Africa and OOA populations from T_NOM. in ky
        T_AF: Time interval for decrease of effective population size of ancestor of modern human population to
            ancestral effective population size from T_B. in ky
        T_NI_NS: Time interval for separation between introgressed Neanderthal and sequenced Neanderthal from T_NOM. in
            ky
        T_DI_DS: Time interval for separation between introgressed Denisovan and sequenced Denisovan from T_DPM. in ky
        T_N_D: Time interval for separation between Neanderthal and Denisova from maximum between T_NI_NS and T_DI_DS.
            in ky
        T_H_A: Time interval for separation between ancestor of Neanderthal/Denisova and modern humans from maximum
            between T_N_D and out of Africa event. in ky
        NMixO: the fraction of introgression happened to OOA populations from Neanderthals.
        DMixP: the fraction of introgression happened to PNG from Denisovan

    :param inds: the number of haplotypes per populations for example (10,10,10)
    :param length: the length of chromosome that has to be simulated. default is 1mb region
    :param mutation_rate: the amount of mutation rate. default is 1.45x10^-8 per generation per base
    :param recombination_rate: the amount of recombination rate. default is 10^-8 per generation per base
    :param replicates: the number of replicated of length chromosome. default is 300
    :return: will return the msprime simulations. which then can be used to extract SFS
    """
    (N_A, N_AF, N_EU, N_AS, N_PA, N_NE, N_DE, N_EU0, N_AS0, N_PA0, N_B, N_ND,
     T_DPM, T_AS_PA, T_EU_AS, T_NOM, T_B, T_AF, T_NI_NS, T_DI_DS, T_N_D, T_H_A,
     NMixO, DMixP) = params
    (n1, n2, n3, n4, n5, n6) = inds
    T_DPM, T_AS_PA, T_EU_AS, T_NOM, T_B, T_AF, T_NI_NS, T_DI_DS, T_N_D, T_H_A = numpy.array(
        [T_DPM, T_AS_PA, T_EU_AS, T_NOM, T_B, T_AF, T_NI_NS, T_DI_DS, T_N_D, T_H_A]) * (1e3 / 29.0)
    NMixO, DMixP = numpy.array([NMixO, DMixP]) / 100
    AFR, EUR, ASN, PAP, NEAI, NEAS, DENI, DENS = 0, 1, 2, 3, 4, 5, 6, 7

    events = {}
    r_EU = (math.log(N_EU / N_EU0) / (T_DPM + T_AS_PA + T_EU_AS))
    r_AS = (math.log(N_AS / N_AS0) / (T_DPM + T_AS_PA + T_EU_AS))
    r_PA = (math.log(N_PA / N_PA0) / (T_DPM + T_AS_PA))
    population_configurations = [
        msprime.PopulationConfiguration(
            sample_size=n1, initial_size=N_AF),
        msprime.PopulationConfiguration(
            sample_size=n2, initial_size=N_EU, growth_rate=r_EU),
        msprime.PopulationConfiguration(
            sample_size=n3, initial_size=N_AS, growth_rate=r_AS),
        msprime.PopulationConfiguration(
            sample_size=n4, initial_size=N_PA, growth_rate=r_PA),
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N_NE),
        msprime.PopulationConfiguration(
            sample_size=n5, initial_size=N_NE),
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N_DE),
        msprime.PopulationConfiguration(
            sample_size=n6, initial_size=N_DE)
    ]
    # Denisova admixture
    events['deni_intro_pap'] = T_DPM
    deni_intro_pap = [msprime.MassMigration(
        time=events['deni_intro_pap'], source=PAP, destination=DENI, proportion=DMixP)]
    # PNG and East Asian merge
    events['split_as_pa'] = events['deni_intro_pap'] + T_AS_PA
    split_as_pa = [
        msprime.MassMigration(
            time=events['split_as_pa'], source=PAP, destination=ASN, proportion=1.0),
        msprime.PopulationParametersChange(
            time=events['split_as_pa'], growth_rate=0, population_id=PAP)
    ]
    # European and Asian+PNG  merge into OOA
    events['split_eu_as'] = events['split_as_pa'] + T_EU_AS
    split_eu_as = [
        msprime.MassMigration(
            time=events['split_eu_as'], source=ASN, destination=EUR, proportion=1.0),
        msprime.PopulationParametersChange(
            time=events['split_eu_as'], growth_rate=0, population_id=ASN),
        msprime.PopulationParametersChange(
            time=events['split_eu_as'], growth_rate=0, population_id=EUR, initial_size=N_B)
    ]
    # introgression Nean
    events['nean_intro_ooa'] = T_NOM + events['split_eu_as']
    nean_intro_ooa = [msprime.MassMigration(
        time=events['nean_intro_ooa'], source=EUR, destination=NEAI, proportion=NMixO)]
    # OOA merges into African
    events['split_afr_ooa'] = events['nean_intro_ooa'] + T_B
    split_afr_ooa = [msprime.MassMigration(
        time=events['split_afr_ooa'], source=EUR, destination=AFR, proportion=1.0)]
    # Size changes to N_A
    events['ancestral_size'] = events['split_afr_ooa'] + T_AF
    ancestral_size = [msprime.PopulationParametersChange(
        time=events['ancestral_size'], initial_size=N_A, population_id=AFR)]
    # neanderthal together
    events['common_neanderthal'] = events['nean_intro_ooa'] + T_NI_NS
    common_neanderthal = [msprime.MassMigration(
        time=events['common_neanderthal'], source=NEAS, destination=NEAI, proportion=1.0)]
    # denisova togehter
    events['common_denisova'] = events['deni_intro_pap'] + T_DI_DS
    common_denisova = [msprime.MassMigration(
        time=events['common_denisova'], source=DENS, destination=DENI, proportion=1.0)]
    # Denisova merging with Neanderthal
    events['neanderthal_denisova'] = max(events['common_neanderthal'], events['common_denisova']) + T_N_D
    neanderthal_denisova = [msprime.MassMigration(
        time=events['neanderthal_denisova'], source=DENI, destination=NEAI, proportion=1.0),
        msprime.PopulationParametersChange(
            time=events['neanderthal_denisova'], initial_size=N_ND, population_id=NEAI)]
    # Archaics merging with humans
    events['human_archaic'] = max(events['neanderthal_denisova'], events['split_afr_ooa']) + T_H_A
    human_neanderthal = [msprime.MassMigration(
        time=events['human_archaic'], source=NEAI, destination=AFR, proportion=1.0)]

    demographic_events = []
    for event in collections.OrderedDict(sorted(events.items(), key=lambda x: x[1])).keys():
        demographic_events = demographic_events + eval(event)
    geno = msprime.simulate(
        population_configurations=population_configurations,
        demographic_events=demographic_events, length=length, mutation_rate=mutation_rate,
        num_replicates=replicates,
        recombination_rate=recombination_rate)
    return geno


def PNG_O_ND(params, inds, length=1e6, recombination_rate=1e-8, mutation_rate=1.25e-8, replicates=300):
    """
    This is the Papuan population is an out group of Europeans and East Asians.
    :param params: all the parameters necessary for this model in a list or array format
        N_A: The ancestral effective population size
        N_AF: Modern effective population size of Africa population
        N_EU: Modern effective population size of European population
        N_AS: Modern effective population size of East Asian population
        N_PA: Modern effective population size of PNG
        N_NE: Effective population size of Neanderthals
        N_DE: Effective population size of Denisovans
        N_EU0: Effective population size of European population before exponential growth.
        N_AS0: Effective population size of East Asian population before exponential growth.
        N_PA0: Effective population size of PNG population before exponential growth.
        N_B: Effective population size of Out of Africa (OOA) populations
        N_ND: Effective population size of ancestor of Neanderthals and Denisovans
        T_DPM: Time interval for introgression from Denisova to PNG population. in kilo year (ky)
        T_EU_AS: Time interval for separation of European and East Asian. in ky
        T_EA_PA: Time interval for separation of PNG and ancestor of European/East Asian from maximum between T_DPM and
            T_EU_AS. in ky
        T_NOM: Time interval for introgression from Neanderthal to Out of African (OOA) population from T_EA_PA. in ky
        T_B: Time interval for separation between Africa and OOA populations from T_NOM. in ky
        T_AF: Time interval for decrease of effective population size of ancestor of modern human population to
            ancestral effective population size from T_B. in ky
        T_NI_NS: Time interval for separation between introgressed Neanderthal and sequenced Neanderthal from T_NOM. in
            ky
        T_DI_DS: Time interval for separation between introgressed Denisovan and sequenced Denisovan from T_DPM. in ky
        T_N_D: Time interval for separation between Neanderthal and Denisova from maximum between T_NI_NS and T_DI_DS.
            in ky
        T_H_A: Time interval for separation between ancestor of Neanderthal/Denisova and modern humans from maximum
            between T_N_D and out of Africa event. in ky
        NMixO: the fraction of introgression happened to OOA populations from Neanderthals.
        DMixP: the fraction of introgression happened to PNG from Denisovan

    :param inds: the number of haplotypes per populations for example (10,10,10)
    :param length: the length of chromosome that has to be simulated. default is 1mb region
    :param mutation_rate: the amount of mutation rate. default is 1.45x10^-8 per generation per base
    :param recombination_rate: the amount of recombination rate. default is 10^-8 per generation per base
    :param replicates: the number of replicated of length chromosome. default is 300
    :return: will return the msprime simulations. which then can be used to extract SFS
    """
    (N_A, N_AF, N_EU, N_AS, N_PA, N_NE, N_DE, N_EU0, N_AS0, N_PA0, N_B, N_ND,
     T_DPM, T_EU_AS, T_EA_PA, T_NOM, T_B, T_AF, T_NI_NS, T_DI_DS, T_N_D, T_H_A,
     NMixO, DMixP) = params
    (n1, n2, n3, n4, n5, n6) = inds
    T_DPM, T_EU_AS, T_EA_PA, T_NOM, T_B, T_AF, T_NI_NS, T_DI_DS, T_N_D, T_H_A = numpy.array(
        [T_DPM, T_EU_AS, T_EA_PA, T_NOM, T_B, T_AF, T_NI_NS, T_DI_DS, T_N_D, T_H_A]) * (1e3 / 29.0)
    NMixO, DMixP = numpy.array([NMixO, DMixP]) / 100
    AFR, EUR, ASN, PAP, NEAI, NEAS, DENI, DENS = 0, 1, 2, 3, 4, 5, 6, 7

    events = {}
    r_EU = (math.log(N_EU / N_EU0) / (T_EU_AS))
    r_AS = (math.log(N_AS / N_AS0) / (T_EU_AS))
    r_PA = (math.log(N_PA / N_PA0) / (T_EU_AS))
    population_configurations = [
        msprime.PopulationConfiguration(
            sample_size=n1, initial_size=N_AF),
        msprime.PopulationConfiguration(
            sample_size=n2, initial_size=N_EU, growth_rate=r_EU),
        msprime.PopulationConfiguration(
            sample_size=n3, initial_size=N_AS, growth_rate=r_AS),
        msprime.PopulationConfiguration(
            sample_size=n4, initial_size=N_PA, growth_rate=r_PA),
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N_NE),
        msprime.PopulationConfiguration(
            sample_size=n5, initial_size=N_NE),
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N_DE),
        msprime.PopulationConfiguration(
            sample_size=n6, initial_size=N_DE)
    ]
    # Denisova admixture
    events['deni_intro_pap'] = T_DPM
    deni_intro_pap = [msprime.MassMigration(
        time=events['deni_intro_pap'], source=PAP, destination=DENI, proportion=DMixP)]
    # European and East Asian merge
    events['split_eu_as'] = T_EU_AS
    split_eu_as = [
        msprime.MassMigration(
            time=events['split_eu_as'], source=ASN, destination=EUR, proportion=1.0),
        msprime.PopulationParametersChange(
            time=events['split_eu_as'], growth_rate=0, population_id=PAP),
        msprime.PopulationParametersChange(
            time=events['split_eu_as'], growth_rate=0, population_id=ASN),
        msprime.PopulationParametersChange(
            time=events['split_eu_as'], growth_rate=0, population_id=EUR, initial_size=N_B)
    ]
    # Eurasian and PNG merge
    events['split_ea_pa'] = max(events['split_eu_as'], events['deni_intro_pap']) + T_EA_PA
    split_ea_pa = [
        msprime.MassMigration(
            time=events['split_ea_pa'], source=PAP, destination=EUR, proportion=1.0)
    ]
    # introgression Nean
    events['nean_intro_ooa'] = T_NOM + events['split_ea_pa']
    nean_intro_ooa = [msprime.MassMigration(
        time=events['nean_intro_ooa'], source=EUR, destination=NEAI, proportion=NMixO)]
    # OOA merge with African
    events['split_afr_ooa'] = events['nean_intro_ooa'] + T_B
    split_afr_ooa = [msprime.MassMigration(
        time=events['split_afr_ooa'], source=EUR, destination=AFR, proportion=1.0)]
    # Size changes to N_A
    events['ancestral_size'] = events['split_afr_ooa'] + T_AF
    ancestral_size = [msprime.PopulationParametersChange(
        time=events['ancestral_size'], initial_size=N_A, population_id=AFR)]
    # neanderthal together
    events['common_neanderthal'] = events['nean_intro_ooa'] + T_NI_NS
    common_neanderthal = [msprime.MassMigration(
        time=events['common_neanderthal'], source=NEAS, destination=NEAI, proportion=1.0)]
    # denisova togehter
    events['common_denisova'] = events['deni_intro_pap'] + T_DI_DS
    common_denisova = [msprime.MassMigration(
        time=events['common_denisova'], source=DENS, destination=DENI, proportion=1.0)]
    # Denisova merging with Neanderthal
    events['neanderthal_denisova'] = max(events['common_neanderthal'], events['common_denisova']) + T_N_D
    neanderthal_denisova = [msprime.MassMigration(
        time=events['neanderthal_denisova'], source=DENI, destination=NEAI, proportion=1.0),
        msprime.PopulationParametersChange(
            time=events['neanderthal_denisova'], initial_size=N_ND, population_id=NEAI)]
    # archaics merging with humans
    events['human_archaic'] = max(events['neanderthal_denisova'], events['split_afr_ooa']) + T_H_A
    human_archaic = [msprime.MassMigration(
        time=events['human_archaic'], source=NEAI, destination=AFR, proportion=1.0)]

    demographic_events = []
    for event in collections.OrderedDict(sorted(events.items(), key=lambda x: x[1])).keys():
        demographic_events = demographic_events + eval(event)
    geno = msprime.simulate(
        population_configurations=population_configurations,
        demographic_events=demographic_events, length=length, mutation_rate=mutation_rate,
        num_replicates=replicates,
        recombination_rate=recombination_rate)
    return geno


def PNG_M_ND(params, inds, length=1e6, recombination_rate=1e-8, mutation_rate=1.25e-8, replicates=300):
    """
    A mixture between model O and model A, i.e. PNG is mixture between a sister group of Asian and an out group of
    European and East Asian
    :param params: all the parameters necessary for this model in a list or array format
        N_A: The ancestral effective population size
        N_AF: Modern effective population size of Africa population
        N_EU: Modern effective population size of European population
        N_AS: Modern effective population size of East Asian population
        N_PA: Modern effective population size of PNG
        N_NE: Effective population size of Neanderthals
        N_DE: Effective population size of Denisovans
        N_EU0: Effective population size of European population before exponential growth.
        N_AS0: Effective population size of East Asian population before exponential growth.
        N_PA0: Effective population size of PNG population before exponential growth.
        N_B: Effective population size of Out of Africa (OOA) populations
        N_ND: Effective population size of ancestor of Neanderthals and Denisovans
        T_DPM: Time interval for introgression from Denisova to PNG population. in kilo year (ky)
        T_Mix: Time interval of admixture between sister group of Asian and an out group of Europe and East Asian,
            which produced the PNG population from T_DPM. in kilo year (ky)
        T_EU_AS: Time interval for separation of European and Asian (PNG+East Asian) from T_Mix. in ky
        T_EA_PA: Time interval for separation of outgroup PNG and Eurasian from T_EA_PA. in ky
        T_NOM: Time interval for introgression from Neanderthal to Out of African (OOA) population from T_EU_AS. in ky
        T_B: Time interval for separation between Africa and OOA populations from T_NOM. in ky
        T_AF: Time interval for decrease of effective population size of ancestor of modern human population to
            ancestral effective population size from T_B. in ky
        T_NI_NS: Time interval for separation between introgressed Neanderthal and sequenced Neanderthal from T_NOM. in
            ky
        T_DI_DS: Time interval for separation between introgressed Denisovan and sequenced Denisovan from T_DPM. in ky
        T_N_D: Time interval for separation between Neanderthal and Denisova from maximum between T_NI_NS and T_DI_DS.
            in ky
        T_H_A: Time interval for separation between ancestor of Neanderthal/Denisova and modern humans from maximum
            between T_N_D and out of Africa event. in ky
        NMixO: the fraction of introgression happened to OOA populations from Neanderthals.
        DMixP: the fraction of introgression happened to PNG from Denisovan
        mix: the fraction of sister group of Asia contributed to PNG population
    :param inds: the number of haplotypes per populations for example (10,10,10)
    :param length: the length of chromosome that has to be simulated. default is 1mb region
    :param mutation_rate: the amount of mutation rate. default is 1.45x10^-8 per generation per base
    :param recombination_rate: the amount of recombination rate. default is 10^-8 per generation per base
    :param replicates: the number of replicated of length chromosome. default is 300
    :return: will return the msprime simulations. which then can be used to extract SFS
    """
    (N_A, N_AF, N_EU, N_AS, N_PA, N_NE, N_DE, N_EU0, N_AS0, N_PA0, N_B, N_ND,
     T_DPM, T_Mix, T_EU_AS, T_EA_PA, T_NOM, T_B, T_AF, T_NI_NS, T_DI_DS, T_N_D, T_H_A,
     NMixO, DMixP, mix) = params
    (n1, n2, n3, n4, n5, n6) = inds
    T_DPM, T_Mix, T_EU_AS, T_EA_PA, T_NOM, T_B, T_AF, T_NI_NS, T_DI_DS, T_N_D, T_H_A = numpy.array(
        [T_DPM, T_Mix, T_EU_AS, T_EA_PA, T_NOM, T_B, T_AF, T_NI_NS, T_DI_DS, T_N_D, T_H_A]) * (1e3 / 29.0)
    NMixO, DMixP, mix = numpy.array([NMixO, DMixP, mix]) / 100
    AFR, EUR, ASN, PAP, NEAI, NEAS, DENI, DENS = 0, 1, 2, 3, 4, 5, 6, 7

    events = {}
    r_EU = (math.log(N_EU / N_EU0) / (T_DPM + T_Mix + T_EU_AS))
    r_AS = (math.log(N_AS / N_AS0) / (T_DPM + T_Mix + T_EU_AS))
    r_PA = (math.log(N_PA / N_PA0) / (T_DPM + T_Mix))
    population_configurations = [
        msprime.PopulationConfiguration(
            sample_size=n1, initial_size=N_AF),
        msprime.PopulationConfiguration(
            sample_size=n2, initial_size=N_EU, growth_rate=r_EU),
        msprime.PopulationConfiguration(
            sample_size=n3, initial_size=N_AS, growth_rate=r_AS),
        msprime.PopulationConfiguration(
            sample_size=n4, initial_size=N_PA, growth_rate=r_PA),
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N_NE),
        msprime.PopulationConfiguration(
            sample_size=n5, initial_size=N_NE),
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N_DE),
        msprime.PopulationConfiguration(
            sample_size=n6, initial_size=N_DE)
    ]
    # Denisova admixture
    events['deni_intro_pap'] = T_DPM
    deni_intro_pap = [msprime.MassMigration(
        time=events['deni_intro_pap'], source=PAP, destination=DENI, proportion=DMixP)]
    # admixture between sister group of Asia and out group of Eurasia
    events['asn_migration'] = events['deni_intro_pap'] + T_Mix
    asn_migration = [
        msprime.MassMigration(time=events['asn_migration'], source=PAP, destination=ASN, proportion=mix),
        msprime.PopulationParametersChange(
            time=events['asn_migration'], growth_rate=0, population_id=PAP)
    ]
    # European and East Asian merge
    events['split_eu_as'] = events['asn_migration'] + T_EU_AS
    split_eu_as = [
        msprime.MassMigration(
            time=events['split_eu_as'], source=ASN, destination=EUR, proportion=1.0),
        msprime.PopulationParametersChange(
            time=events['split_eu_as'], growth_rate=0, population_id=ASN),
        msprime.PopulationParametersChange(
            time=events['split_eu_as'], initial_size=N_B, growth_rate=0, population_id=EUR),
    ]
    # merge between Eurasia and PNG
    events['split_ea_pa'] = events['split_eu_as'] + T_EA_PA
    split_ea_pa = [
        msprime.MassMigration(
            time=events['split_ea_pa'], source=PAP, destination=EUR, proportion=1.0)
    ]
    # introgression Nean
    events['nean_intro_ooa'] = T_NOM + events['split_ea_pa']
    nean_intro_ooa = [msprime.MassMigration(
        time=events['nean_intro_ooa'], source=EUR, destination=NEAI, proportion=NMixO)]
    # OOA merge with African
    events['split_afr_ooa'] = events['nean_intro_ooa'] + T_B
    split_afr_ooa = [msprime.MassMigration(
        time=events['split_afr_ooa'], source=EUR, destination=AFR, proportion=1.0)]
    # Size changes to N_A
    events['ancestral_size'] = events['split_afr_ooa'] + T_AF
    ancestral_size = [msprime.PopulationParametersChange(
        time=events['ancestral_size'], initial_size=N_A, population_id=AFR)]
    # neanderthal together
    events['common_neanderthal'] = events['nean_intro_ooa'] + T_NI_NS
    common_neanderthal = [msprime.MassMigration(
        time=events['common_neanderthal'], source=NEAS, destination=NEAI, proportion=1.0)]
    # denisova togehter
    events['common_denisova'] = events['deni_intro_pap'] + T_DI_DS
    common_denisova = [msprime.MassMigration(
        time=events['common_denisova'], source=DENS, destination=DENI, proportion=1.0)]
    # Denisova merging with Neanderthal
    events['neanderthal_denisova'] = max(events['common_neanderthal'], events['common_denisova']) + T_N_D
    neanderthal_denisova = [msprime.MassMigration(
        time=events['neanderthal_denisova'], source=DENI, destination=NEAI, proportion=1.0),
        msprime.PopulationParametersChange(
            time=events['neanderthal_denisova'], initial_size=N_ND, population_id=NEAI)]
    # archaics merging with humans
    events['human_archaic'] = max(events['neanderthal_denisova'], events['split_afr_ooa']) + T_H_A
    human_archaic = [msprime.MassMigration(
        time=events['human_archaic'], source=NEAI, destination=AFR, proportion=1.0)]

    demographic_events = []
    for event in collections.OrderedDict(sorted(events.items(), key=lambda x: x[1])).keys():
        demographic_events = demographic_events + eval(event)
    geno = msprime.simulate(
        population_configurations=population_configurations,
        demographic_events=demographic_events, length=length, mutation_rate=mutation_rate,
        num_replicates=replicates,
        recombination_rate=recombination_rate)
    return geno


def PNG_AX_ND(params, inds, length=1e6, recombination_rate=1e-8, mutation_rate=1.25e-8, replicates=300):
    """
    This is the PNG population (sister group of Asia) has a contribution from first OOA population.
    :param params: all the parameters necessary for this model in a list or array format
        N_A: The ancestral effective population size
        N_AF: Modern effective population size of Africa population
        N_EU: Modern effective population size of European population
        N_AS: Modern effective population size of East Asian population
        N_PA: Modern effective population size of PNG
        N_NE: Effective population size of Neanderthals
        N_DE: Effective population size of Denisovans
        N_EU0: Effective population size of European population before exponential growth.
        N_AS0: Effective population size of East Asian population before exponential growth.
        N_PA0: Effective population size of PNG population before exponential growth.
        N_B: Effective population size of Out of Africa (OOA) populations
        N_ND: Effective population size of ancestor of Neanderthals and Denisovans
        T_DPM: Time interval for introgression from Denisova to PNG population. in kilo year (ky)
        T_Mix: Time interval of admixture between PNG (sister group of Asia) and an first OOA population from T_DPM. in
            kilo year (ky)
        T_EU_AS: Time interval for separation of European and Asian (PNG+East Asian) from T_Mix. in ky
        T_NOM: Time interval for introgression from Neanderthal to Out of African (OOA) population from T_EU_AS. in ky
        T_B: Time interval for separation between Africa and OOA populations from T_NOM. in ky
        T_X_H: Time interval for separation between Africa and first OOA populations from T_B. in ky
        T_AF: Time interval for decrease of effective population size of ancestor of modern human population to
            ancestral effective population size from T_X_H. in ky
        T_NI_NS: Time interval for separation between introgressed Neanderthal and sequenced Neanderthal from T_NOM. in
            ky
        T_DI_DS: Time interval for separation between introgressed Denisovan and sequenced Denisovan from T_DPM. in ky
        T_N_D: Time interval for separation between Neanderthal and Denisova from maximum between T_NI_NS and T_DI_DS.
            in ky
        T_H_A: Time interval for separation between ancestor of Neanderthal/Denisova and modern humans from maximum
            between T_N_D and out of Africa event. in ky
        NMixO: the fraction of introgression happened to OOA populations from Neanderthals.
        DMixP: the fraction of introgression happened to PNG from Denisovan
        mix: the fraction of sister group of Asia contributed to PNG population

    :param inds: the number of haplotypes per populations for example (10,10,10)
    :param length: the length of chromosome that has to be simulated. default is 1mb region
    :param mutation_rate: the amount of mutation rate. default is 1.45x10^-8 per generation per base
    :param recombination_rate: the amount of recombination rate. default is 10^-8 per generation per base
    :param replicates: the number of replicated of length chromosome. default is 300
    :return: will return the msprime simulations. which then can be used to extract SFS
    """
    (N_A, N_AF, N_EU, N_AS, N_PA, N_NE, N_DE, N_EU0, N_AS0, N_PA0, N_B, N_ND,
     T_DPM, T_Mix, T_EU_AS, T_NOM, T_B, T_X_H, T_AF, T_NI_NS, T_DI_DS, T_N_D, T_H_A,
     NMixO, DMixP, mix) = params
    (n1, n2, n3, n4, n5, n6) = inds
    T_DPM, T_Mix, T_EU_AS, T_NOM, T_B, T_X_H, T_AF, T_NI_NS, T_DI_DS, T_N_D, T_H_A = numpy.array(
        [T_DPM, T_Mix, T_EU_AS, T_NOM, T_B, T_X_H, T_AF, T_NI_NS, T_DI_DS, T_N_D, T_H_A]) * (1e3 / 29.0)
    NMixO, DMixP, mix = numpy.array([NMixO, DMixP, mix]) / 100
    AFR, EUR, ASN, PAP, NEAI, NEAS, DENI, DENS = 0, 1, 2, 3, 4, 5, 6, 7

    events = {}
    r_EU = (math.log(N_EU / N_EU0) / (T_DPM + T_Mix + T_EU_AS))
    r_AS = (math.log(N_AS / N_AS0) / (T_DPM + T_Mix + T_EU_AS))
    r_PA = (math.log(N_PA / N_PA0) / (T_DPM + T_Mix))
    population_configurations = [
        msprime.PopulationConfiguration(
            sample_size=n1, initial_size=N_AF),
        msprime.PopulationConfiguration(
            sample_size=n2, initial_size=N_EU, growth_rate=r_EU),
        msprime.PopulationConfiguration(
            sample_size=n3, initial_size=N_AS, growth_rate=r_AS),
        msprime.PopulationConfiguration(
            sample_size=n4, initial_size=N_PA, growth_rate=r_PA),
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N_NE),
        msprime.PopulationConfiguration(
            sample_size=n5, initial_size=N_NE),
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N_DE),
        msprime.PopulationConfiguration(
            sample_size=n6, initial_size=N_DE)
    ]
    # Denisova admixture
    events['deni_intro_pap'] = T_DPM
    deni_intro_pap = [msprime.MassMigration(
        time=events['deni_intro_pap'], source=PAP, destination=DENI, proportion=DMixP)]
    # PNG admixed with first OOA
    events['asn_migration'] = events['deni_intro_pap'] + T_Mix
    asn_migration = [
        msprime.MassMigration(time=events['asn_migration'], source=PAP, destination=ASN, proportion=mix),
        msprime.PopulationParametersChange(
            time=events['asn_migration'], growth_rate=0, population_id=PAP)
    ]
    # European and East Asian merge
    events['split_eu_as'] = events['asn_migration'] + T_EU_AS
    split_eu_as = [
        msprime.MassMigration(
            time=events['split_eu_as'], source=ASN, destination=EUR, proportion=1.0),
        msprime.PopulationParametersChange(
            time=events['split_eu_as'], growth_rate=0, population_id=ASN),
        msprime.PopulationParametersChange(
            time=events['split_eu_as'], initial_size=N_B, growth_rate=0, population_id=EUR),
    ]

    # introgression Nean
    events['nean_intro_ooa'] = T_NOM + events['split_eu_as']
    nean_intro_ooa = [msprime.MassMigration(
        time=events['nean_intro_ooa'], source=EUR, destination=NEAI, proportion=NMixO)]

    # OOA merge with African
    events['split_afr_ooa'] = events['nean_intro_ooa'] + T_B
    split_afr_ooa = [msprime.MassMigration(
        time=events['split_afr_ooa'], source=EUR, destination=AFR, proportion=1.0)]
    # fist OOA merge with African
    events['split_ea_pa'] = events['split_afr_ooa'] + T_X_H
    split_ea_pa = [
        msprime.MassMigration(
            time=events['split_ea_pa'], source=PAP, destination=AFR, proportion=1.0)
    ]
    # Size changes to N_A
    events['ancestral_size'] = events['split_ea_pa'] + T_AF
    ancestral_size = [msprime.PopulationParametersChange(
        time=events['ancestral_size'], initial_size=N_A, population_id=AFR)]
    # neanderthal together
    events['common_neanderthal'] = events['nean_intro_ooa'] + T_NI_NS
    common_neanderthal = [msprime.MassMigration(
        time=events['common_neanderthal'], source=NEAS, destination=NEAI, proportion=1.0)]
    # denisova togehter
    events['common_denisova'] = events['deni_intro_pap'] + T_DI_DS
    common_denisova = [msprime.MassMigration(
        time=events['common_denisova'], source=DENS, destination=DENI, proportion=1.0)]
    # Denisova merging with Neanderthal
    events['neanderthal_denisova'] = max(events['common_neanderthal'], events['common_denisova']) + T_N_D
    neanderthal_denisova = [msprime.MassMigration(
        time=events['neanderthal_denisova'], source=DENI, destination=NEAI, proportion=1.0),
        msprime.PopulationParametersChange(
            time=events['neanderthal_denisova'], initial_size=N_ND, population_id=NEAI)]
    # archaics merging with humans
    events['human_archaic'] = max(events['neanderthal_denisova'], events['split_ea_pa']) + T_H_A
    human_archaic = [msprime.MassMigration(
        time=events['human_archaic'], source=NEAI, destination=AFR, proportion=1.0)]

    demographic_events = []
    for event in collections.OrderedDict(sorted(events.items(), key=lambda x: x[1])).keys():
        demographic_events = demographic_events + eval(event)
    geno = msprime.simulate(
        population_configurations=population_configurations,
        demographic_events=demographic_events, length=length, mutation_rate=mutation_rate,
        num_replicates=replicates,
        recombination_rate=recombination_rate)
    return geno


def PNG_OX_ND(params, inds, length=1e6, recombination_rate=1e-8, mutation_rate=1.25e-8, replicates=300):
    """
    This is the PNG population (out group of Eurasia) has a contribution from first OOA population.
    :param params: all the parameters necessary for this model in a list or array format
        N_A: The ancestral effective population size
        N_AF: Modern effective population size of Africa population
        N_EU: Modern effective population size of European population
        N_AS: Modern effective population size of East Asian population
        N_PA: Modern effective population size of PNG
        N_NE: Effective population size of Neanderthals
        N_DE: Effective population size of Denisovans
        N_EU0: Effective population size of European population before exponential growth.
        N_AS0: Effective population size of East Asian population before exponential growth.
        N_PA0: Effective population size of PNG population before exponential growth.
        N_B: Effective population size of Out of Africa (OOA) populations
        N_ND: Effective population size of ancestor of Neanderthals and Denisovans
        T_DPM: Time interval for introgression from Denisova to PNG population. in kilo year (ky)
        T_Mix: Time interval of admixture between PNG (out group of Eurasia) and an first OOA population from T_DPM. in
            kilo year (ky)
        T_EU_AS: Time interval for separation of European and Asian. in ky
        T_EA_PA: Time interval for separation of PNG and Eurasian from maximum of T_EU_AS and T_Mix. in ky
        T_NOM: Time interval for introgression from Neanderthal to Out of African (OOA) population from T_EA_PA. in ky
        T_B: Time interval for separation between Africa and OOA populations from T_NOM. in ky
        T_X_H: Time interval for separation between Africa and first OOA populations from T_B. in ky
        T_AF: Time interval for decrease of effective population size of ancestor of modern human population to
            ancestral effective population size from T_X_H. in ky
        T_NI_NS: Time interval for separation between introgressed Neanderthal and sequenced Neanderthal from T_NOM. in
            ky
        T_DI_DS: Time interval for separation between introgressed Denisovan and sequenced Denisovan from T_DPM. in ky
        T_N_D: Time interval for separation between Neanderthal and Denisova from maximum between T_NI_NS and T_DI_DS.
            in ky
        T_H_A: Time interval for separation between ancestor of Neanderthal/Denisova and modern humans from maximum
            between T_N_D and out of Africa event. in ky
        NMixO: the fraction of introgression happened to OOA populations from Neanderthals.
        DMixP: the fraction of introgression happened to PNG from Denisovan
        mix: the fraction of first OOA contributed to PNG population

    :param inds: the number of haplotypes per populations for example (10,10,10)
    :param length: the length of chromosome that has to be simulated. default is 1mb region
    :param mutation_rate: the amount of mutation rate. default is 1.45x10^-8 per generation per base
    :param recombination_rate: the amount of recombination rate. default is 10^-8 per generation per base
    :param replicates: the number of replicated of length chromosome. default is 300
    :return: will return the msprime simulations. which then can be used to extract SFS
    """
    (N_A, N_AF, N_EU, N_AS, N_PA, N_NE, N_DE, N_EU0, N_AS0, N_PA0, N_B, N_ND,
     T_DPM, T_Mix, T_EU_AS, T_EA_PA, T_NOM, T_B, T_X_H, T_AF, T_NI_NS, T_DI_DS, T_N_D, T_H_A,
     NMixO, DMixP, mix) = params
    (n1, n2, n3, n4, n5, n6) = inds
    T_DPM, T_Mix, T_EU_AS, T_EA_PA, T_NOM, T_B, T_X_H, T_AF, T_NI_NS, T_DI_DS, T_N_D, T_H_A = numpy.array(
        [T_DPM, T_Mix, T_EU_AS, T_EA_PA, T_NOM, T_B, T_X_H, T_AF, T_NI_NS, T_DI_DS, T_N_D, T_H_A]) * (1e3 / 29.0)
    NMixO, DMixP, mix = numpy.array([NMixO, DMixP, mix]) / 100
    AFR, EUR, ASN, PAP, XOA, NEAI, NEAS, DENI, DENS = 0, 1, 2, 3, 4, 5, 6, 7, 8

    events = {}
    r_EU = (math.log(N_EU / N_EU0) / (T_EU_AS))
    r_AS = (math.log(N_AS / N_AS0) / (T_EU_AS))
    r_PA = (math.log(N_PA / N_PA0) / (T_EU_AS))
    population_configurations = [
        msprime.PopulationConfiguration(
            sample_size=n1, initial_size=N_AF),
        msprime.PopulationConfiguration(
            sample_size=n2, initial_size=N_EU, growth_rate=r_EU),
        msprime.PopulationConfiguration(
            sample_size=n3, initial_size=N_AS, growth_rate=r_AS),
        msprime.PopulationConfiguration(
            sample_size=n4, initial_size=N_PA, growth_rate=r_PA),
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N_PA0),
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N_NE),
        msprime.PopulationConfiguration(
            sample_size=n5, initial_size=N_NE),
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N_DE),
        msprime.PopulationConfiguration(
            sample_size=n6, initial_size=N_DE)
    ]
    # Denisova admixture
    events['deni_intro_pap'] = T_DPM
    deni_intro_pap = [msprime.MassMigration(
        time=events['deni_intro_pap'], source=PAP, destination=DENI, proportion=DMixP)]
    # PNG admixed with first OOA
    events['xooa_mix'] = events['deni_intro_pap'] + T_Mix
    xooa_mix = [
        msprime.MassMigration(time=events['xooa_mix'], source=PAP, destination=XOA, proportion=mix)]
    # European and East Asian merge
    events['split_eu_as'] = T_EU_AS
    split_eu_as = [
        msprime.MassMigration(
            time=events['split_eu_as'], source=ASN, destination=EUR, proportion=1.0),
        msprime.PopulationParametersChange(
            time=events['split_eu_as'], growth_rate=0, population_id=PAP),
        msprime.PopulationParametersChange(
            time=events['split_eu_as'], growth_rate=0, population_id=ASN),
        msprime.PopulationParametersChange(
            time=events['split_eu_as'], growth_rate=0, population_id=EUR, initial_size=N_B)
    ]
    # Eurasia and PNG merged
    events['split_ea_pa'] = max(events['split_eu_as'], events['xooa_mix']) + T_EA_PA
    split_ea_pa = [
        msprime.MassMigration(
            time=events['split_ea_pa'], source=PAP, destination=EUR, proportion=1.0)
    ]
    # introgression Nean
    events['nean_intro_ooa'] = T_NOM + events['split_ea_pa']
    nean_intro_ooa = [msprime.MassMigration(
        time=events['nean_intro_ooa'], source=EUR, destination=NEAI, proportion=NMixO)]
    # OOA merge with African
    events['split_afr_ooa'] = events['nean_intro_ooa'] + T_B
    split_afr_ooa = [msprime.MassMigration(
        time=events['split_afr_ooa'], source=EUR, destination=AFR, proportion=1.0)]

    # fist OOA merge with African
    events['split_xOOA'] = events['split_afr_ooa'] + T_X_H
    split_xOOA = [
        msprime.MassMigration(
            time=events['split_xOOA'], source=XOA, destination=AFR, proportion=1.0)
    ]
    # Size changes to N_A
    events['ancestral_size'] = events['split_xOOA'] + T_AF
    ancestral_size = [msprime.PopulationParametersChange(
        time=events['ancestral_size'], initial_size=N_A, population_id=AFR)]
    # neanderthal together
    events['common_neanderthal'] = events['nean_intro_ooa'] + T_NI_NS
    common_neanderthal = [msprime.MassMigration(
        time=events['common_neanderthal'], source=NEAS, destination=NEAI, proportion=1.0)]
    # denisova together
    events['common_denisova'] = events['deni_intro_pap'] + T_DI_DS
    common_denisova = [msprime.MassMigration(
        time=events['common_denisova'], source=DENS, destination=DENI, proportion=1.0)]
    # Denisova merging with Neanderthal
    events['neanderthal_denisova'] = max(events['common_neanderthal'], events['common_denisova']) + T_N_D
    neanderthal_denisova = [msprime.MassMigration(
        time=events['neanderthal_denisova'], source=DENI, destination=NEAI, proportion=1.0),
        msprime.PopulationParametersChange(
            time=events['neanderthal_denisova'], initial_size=N_ND, population_id=NEAI)]
    # archaics merging with humans
    events['human_archaic'] = max(events['neanderthal_denisova'], events['split_afr_ooa']) + T_H_A
    human_archaic = [msprime.MassMigration(
        time=events['human_archaic'], source=NEAI, destination=AFR, proportion=1.0)]

    demographic_events = []
    for event in collections.OrderedDict(sorted(events.items(), key=lambda x: x[1])).keys():
        demographic_events = demographic_events + eval(event)
    geno = msprime.simulate(
        population_configurations=population_configurations,
        demographic_events=demographic_events, length=length, mutation_rate=mutation_rate,
        num_replicates=replicates,
        recombination_rate=recombination_rate)
    return geno
