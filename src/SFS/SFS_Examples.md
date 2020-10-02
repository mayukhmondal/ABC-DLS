# ABC-TFK with SFS Examples

In principle ABC-TFK is meant for any summary statistics (ss) that can be used to predict the parameters. But 
unfortunately it is impossible to give examples for every possible ss. On the other hand, it is harder to understand and
implement if examples are not given. Thus here we used one of the ss to give an idea how to implement such a pipeline 
for population genomics background. Joint Site Frequency Spectrum (SFS) were shown to be good ss for this kind of 
approach. Although they are good but does not mean they are sufficient, thus ABC-TFK should not be shackled by only 
using SFS but this is a good starting point.      

## Installation

To run and create SFS, we need some other packages. I divided the packages from the default packages of ABC-TFK as SFS 
are not necessary for ABC-TFK and more packages mean more dependency hell. The packages needed on top of the previous packages are:

- scikit-allel
- msprime
- snakemake

You can install as previously (of course in the same environment as ABC-TFK):
```shell script
conda install --file src/SFS/requirements.txt
```
or last tested version:
```shell script
conda env update -f src/SFS/requirements.yml
```
## VCF to SFS
First we need a real or observed ss which can be produced from a vcf file. Before using the vcf file we need some filter
and information so that it can be used for SFS.
- Every body has their own strategy of filtering. But the strategy I generally use is following (For explanations for the commands used please see the respective software sites [vcftools](https://vcftools.github.io/index.html) and [bcftools](http://samtools.github.io/bcftools/bcftools.html). ):
```shell script
vcftools --gzvcf  <in.vcf.gz> --max-missing 1.0 --remove-indels --min-alleles 2 --max-alleles 2 --mac 1 --keep <in.pop> --stdout --recode|bcftools view -T examples/masked_regions.bed.gz| bcftools  annotate -x ^FORMAT/GT -O z -o <out.vcf.gz> 
bcftools index <out.vcf.gz>
```
- We need to add the ancestral information of the alleles in the INFO tag of the vcf file. You can use bcftools 
annotate to add those information if you have the ancestral allele file. 

After we have our filtered and annotated vcf file (you can look for an example in examples/Examples.vcf.gz), we can 
convert it to SFS file using:
```shell script
python src/SFS/Run_VCF2SFS.py --popfile Input.pop --sfs_pop YRI,FRN,HAN examples/Examples.vcf.gz
``` 
 Which will create a Examples.csv. This can be used as observed SFS for further analysis. 

## Creating Uniform Priors from range
Before running the simulations, we need to create priors for the parameters on which simulations can run. The priors can
be created by different methods and from different distributions. But one of the most used scenario is to produce the 
priors from a uniform distribution with a given range for minimum and maximum values. Here we took the example from a 
well known model (Gutenkunst et al. 2009, Gravel et al. 2013) 
To produce uniform distribution 
you can use:
```shell script
python /home/mayukh/PycharmProjects/ABC-TFK/src/SFS/Run_Range2UniParameters.py --upper 25e3,2e5,2e5,2e5,1e4,1e4,1e4,80,320,700,50,50,50,50 --lower 5e3,1e4,1e4,1e4,500,500,500,15,5,5,0,0,0,0 --par_names N_A,N_AF,N_EU,N_AS,N_EU0,N_AS0,N_B,T_EU_AS,T_B,T_AF,m_AF_B,m_AF_EU,m_AF_AS,m_EU_AS  --repeats 10 > Params.csv
``` 
This will create csv files whose every row denote different run for simulations and the columns denote different 
parameters with the given range:

| Parameters | Upper Limit | lower Limit | 
| ---------- | ----------- | ----------- |
|N_A         |25,000       |5,000        |
|N_AF        |200,000      |10,000       |
|N_EU        |200,000      |10,000       |
|N_AS        |200,000      |10,000       |
|N_EU0       |10,000       |500          |
|N_AS0       |10,000       |500          |
|N_B         |10,000       |500          |
|T_EU_AS     |80 ky        |15 ky        |
|T_B         |320 ky       |5 ky         |
|T_AF        |700 ky       |5 ky         |
|m_AF_B     |50 x 10<sup>^-5 </sup>|0    |
|m_AF_EU    |50 x 10<sup>^-5 </sup>|0    |
|m_AF_AS    |50 x 10<sup>^-5 </sup>|0    |  
|m_AF_EU    |50 x 10<sup>^-5 </sup>|0    | 

where ky is kilo years
## Prior to SFS 
Again SFS can be created by lots of other methods but here we have used only msprime (which is fast enough as we need 
a lot of simulations). For the simulations we used a previously well known model from [msprime](https://msprime.readthedocs.io/en/stable/tutorial.html#demography) itself with slight changes:
```python
import math

import msprime
import numpy
def OOA(params, inds, length=1e6, mutation_rate=1.45e-8, recombination_rate=1e-8, replicates=300):
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
``` 
The notable difference is with scaling of events and migration rates. msprime uses generations thus our input has to be 
scaled with 1000/29 as our events are in ky and the migrations are multiplieded by 10<sup>^-5</sup> to make it correct scale. 
We used more human readable version of priors instead of required one so that we can easily understand the output of 
the results. Of course we could have used directly generations and correct scaled amount of migration matrix but then 
it will be difficult to understand. Of course if we are running the parameter estimation only for once it does not 
matter but it become more easier when we use recursively to understand if our results not going awry. You can add your 
own code in the SFS/Demography.py which has to follow some simple rule:
``` python
import msprime
def demo(params, inds, length=1e6, mutation_rate=1.45e-8, recombination_rate=1e-8, replicates=300):
    your own code
    retrun geno
``` 
    