# ABC-DLS with SFS Examples
In principle, ABC-DLS is meant for any summary statistics (ss) that can be used to predict the parameters. But unfortunately, it is impossible to give examples for every possible ss. On the other hand, it is harder to understand and implement if examples are not provided (especially for SMC part). Here we used one of the ss to explain how to implement such a pipeline for population genomics background. Joint Site Frequency Spectrum (SFS) was shown to be good ss for this kind of approach. Although they are good but do not mean they are sufficient, ABC-DLS should not be shackled by only using SFS but this is a good starting point. 
## Installation
To run and create SFS, we need some new packages. I divided the packages from ABC-DLS default packages as SFS are not necessary for ABC-DLS and more packages mean more dependency hell. The packages needed on top of the previous packages are:
- scikit-allel
- msprime
- snakemake   

You can install as previously (of course in the same environment as ABC-DLS):
```shell script
conda install -c conda-forge -c bioconda --file src/SFS/requirements.txt
```
or last tested version:
```shell script
conda env update -f src/SFS/requirements.yml
```
## VCF to SFS
First, we need a real or observed ss, which can be produced from a vcf file. Before using the vcf file, we need some filters and information so that it can be used to create SFS.
- filters: Everybody has their strategy of filtering. But the strategy I generally use is the following (for explanations for the commands used, please see the respective software sites [vcftools](https://vcftools.github.io/index.html) and [bcftools](http://samtools.github.io/bcftools/bcftools.html).):
```shell script
vcftools --gzvcf  <in.vcf.gz> --max-missing 1.0 --remove-indels --min-alleles 2 --max-alleles 2 --mac 1 --keep <in.pop> --stdout --recode|bcftools view -T examples/masked_regions.bed.gz| bcftools  annotate -x ^FORMAT/GT -O z -o <out.vcf.gz> 
bcftools index <out.vcf.gz>
```
- Ancestral allele: We need to add the alleles' ancestral information in the INFO tag of the vcf file. You can use bcftools annotate to add that information if you have the ancestral allele file. If you do not have ancestral allele information, you can still use ABC-DLS. You have to fold the SFS. I have not tested it myself but in principle it should be a problem.  

After we have our filtered and annotated vcf file (you can look for an example in [examples/Examples.vcf.gz](../../examples/Examples.vcf.gz), we can convert it to an SFS file using:
```shell script
python src/SFS/Run_VCF2SFS.py --popfile Input.tsv --sfs_pop YRI,FRN,HAN examples/Examples.vcf.gz
``` 
 This command will create an Examples.csv. This file can be used as observed SFS for further analysis. 
- --popfile Input.tsv  
This file should have information on populations per individual. The first column should be the individual name present in the vcf file and the second column should be the population name in a tab-separated format. 
- --sfs_pop YRI,FRN,HAN  
This option used to give in which sequence the sfs should be produced as SFS (Pop1, Pop2, Pop3) != SFS(Pop2, Pop1, Pop3). Where Pop are populations.  
- [examples/Examples.vcf.gz](../../examples/Examples.vcf.gz) is the vcf file.
## Creating Uniform Priors from range
Before running the simulations, we need to create priors for the parameters on which simulations can run. The priors can be created by different methods and from different distributions. But one of the most used scenarios is to produce the priors from a uniform distribution with a given range for minimum and maximum values. Here we took the example from a well-known model ([Gutenkunst et al. 2009](https://doi.org/10.1371/journal.pgen.1000695), 
[Gravel et al. 2013](https://doi.org/10.1073/pnas.1019276108)).    
To produce uniform distributions of parameters, you can use:
```shell script
python src/SFS/Run_Range2UniParameters.py --upper 25e3,2e5,2e5,2e5,1e4,1e4,1e4,80,320,700,50,50,50,50 --lower 5e3,1e4,1e4,1e4,500,500,500,15,5,5,0,0,0,0 --par_names N_A,N_AF,N_EU,N_AS,N_EU0,N_AS0,N_B,T_EU_AS,T_B,T_AF,m_AF_B,m_AF_EU,m_AF_AS,m_EU_AS  --repeats 10 > Params.csv
``` 
This command will create csv file whose every row denote different run for simulations and the columns represent various parameters with the given range:

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
|m_AF_B     |50 x 10<sup>-5 </sup>|0    |
|m_AF_EU    |50 x 10<sup>-5 </sup>|0    |
|m_AF_AS    |50 x 10<sup>-5 </sup>|0    |  
|m_AF_EU    |50 x 10<sup>-5 </sup>|0    | 
where ky is kilo years
## Prior to SFS 
### Simulation (msprime)
Again SFS can be created by lots of other methods but here, we have used only msprime (which is fast enough as we need a lot of simulations). For the simulations, we used a previously well-known model from [msprime](https://msprime.readthedocs.io/en/stable/tutorial.html#demography) itself with slight changes:
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
The notable differences here are with the scaling of events and migration rates. msprime uses generations. Thus our input events were scaled with 1000/29 as they were in ky and the migrations were multiplied by 10<sup>-5</sup> to make the correct scaling. We used a more human-readable version for priors instead of the required one to understand the output easily. Of course, we could have used direct generations and the correct amount of migration matrix directly, but it won't be easy to understand. Of course, if we are running the parameter estimation only once, it does not matter. However, it becomes easier when we use recursively to understand if our results are not going awry (especially useful when we are using [recursive.sh](recursive.sh) with snakemake pipeline. See later for more information. You can add your code in the [src/SFS/Demography.py](Demography.py), which has to follow some simple rule:
```python
import msprime
def demo(params, inds, length=1e6, mutation_rate=1.45e-8, recombination_rate=1e-8, replicates=300):
    your own code
    geno=msprime.simulate(...)
    return geno
```
### Creating SFS csv file
After making the correct format for demography, it is time to create the csv file from the simulations. 
```shell
python src/SFS/Run_Prior2SFS.py  OOA --params_file Params.csv --inds 5,5,5 --threads 5 --total_length 1e7 |gzip > OOA.csv.gz
```
- OOA   
This is the name of the demography, which is present in [src/SFS/Demography.py](Demography.py). You can use whatever name as the definition and can be accessed from here.  
- --params_file Params.csv   
This file is the priors that are produced in the previous step. 
- --inds 5,5,5   
The number of individuals per population.
- --threads 5   
The number of threads to be used. As this step is the bottleneck for the whole approach, this is important to use. 
- --total_length 1e7     
The total length of the genome has to be simulated. Here we are simulating 10 mbp region. The code will run 1mb region at a time by default. Thus 10mbp means repeating the 1mbp regions ten times. Because of this approach, I would suggest not to use seeds or use them with caution. If not, it will produce the same SFS again and again. Thus running it ten times will not be any improvement. 

This command will create an Out of Africa simulated SFS with parameter file, which then can be directly used as an input for the parameter estimation or SMC method. Undoubtedly, this part is the main bottleneck for the whole approach. Thus this part should be mostly improved if we use an even more, parallelize approach. For example, only using threads will not be enough. 
In my case, I used cluster to massively parallelize by submitting multiple productions of csv together using snakemake (see later).  

## Going forward with ABC-DLS (SFS to Posteriors)
With this prior stuff done, finally, we reached a situation where we can use the ABC-DLS method to get out the posterior range. 
```shell 
echo -e "OOA.csv.gz\t14" > Model.info
python src/Run_SMC.py All --test_size 5 --tolerance .5 --ssfile examples/Examples.csv --scale b Model.info --decrease 0.95 --increase .01 --hardrange src/SFS/Startrange.csv
``` 
Understandably, we do not expect to reach any level of correctness at all with ten simulations. To have a good level of power, we need to use much more simulations:
```shell
python src/SFS/Run_Range2UniParameters.py --upper 25e3,2e5,2e5,2e5,1e4,1e4,1e4,80,320,700,50,50,50,50 --lower 5e3,1e4,1e4,1e4,500,500,500,15,5,5,0,0,0,0 --par_names N_A,N_AF,N_EU,N_AS,N_EU0,N_AS0,N_B,T_EU_AS,T_B,T_AF,m_AF_B,m_AF_EU,m_AF_AS,m_EU_AS  --repeats 2e4 > Params.csv
python src/SFS/Run_Prior2SFS.py  OOA --params_file Params.csv --inds 5,5,5 --threads 5 --total_length 1e6 |gzip > OOA.csv.gz 
echo -e "OOA.csv.gz\t14" > Model.info
python src/Run_SMC.py All --ssfile Examples.csv --scale b Model.info --frac 0.16442630347307816 --decrease 0.95 --increase .01 --hardrange src/SFS/Startrange.csv
```
The frac (fraction) was calculated with the available amount of data for chr22 (for the vcf file), which is 6,081,752. Thus to make it equal with the simulations, we have to multiply 1e6/6081752 or 0.16442630347307816. You will see a decrease (dec) for several parameters. But this ran only once. To use it recursively, we can use the output (Newrange.csv) and use it as input to rerun it till there is no decrease possible. Please check [examples/Examples.md](../../examples/Examples.md) for all the information for all the other options. 
## Snakemake 
We added a snakemake pipeline to run those commands automatically. Snakemake pipeline will be beneficial for clusters, where we can run multiple jobs together. Unfortunately, here we can not talk about how to install and implement snakemake pipeline in the cluster. For further information, you have to see [snakemake](https://snakemake.readthedocs.io/en/stable/) tutorial. The snakemake pipeline takes config.yml as an input, which takes several necessary configuration parameters together inside a yml file. Here we add an example in [config.yml](config.yml) file:
```yaml
sc_priors: Run_Range2UniParameters.py
sc_sfs: Run_Prior2SFS.py
sc_abc: ../Run_SMC.py
sfsfile: ../../examples/YRI_FRN_HAN.observed.csv
priors_range: Oldrange.csv
demography: OOA
inds: 5,5,5
threads: 1
jobs: 10
repeats: 2000
total_length: 1000000
test_size: 1000
decrease: 0.95
increase: .01
hardrange_file: Startrange.csv
tolerance: .1
frac: 0.0015455858426908665
``` 
 - sc_priors, sc_sfs and sc_abc are the scripts for running the necessary commands for Run_Range2UniParameters.py, Run_Prior2SFS.py and Run_SMC.py. If you want to run it somewhere else, you can put the full path instead of the relative path. 
- sfsfile is the full path where you have your real or observed SFS file in csv format. You can use python src/SFS/Run_VCF2SFS.py to produce such a file. Please see above how to do it. Here we used one real observed data of Yoruba, French and Han Chinese downloaded from [High Coverage HGDP data](https://doi.org/10.1126/science.aay5012).
-  priors_range is the file path for a csv file that has three columns. The first columns are the parameters name, the second column is the lower limit and the third column is the upper limit for every parameter. No header is expected. Example:
```csv
N_A,5000,25000
N_AF,10000,200000
N_EU,10000,200000
N_AS,10000,200000
N_EU0,500,10000
N_AS0,500,10000
N_B,500,10000
T_EU_AS,15,80
T_B,5,320
T_AF,5,700
m_AF_B,0,50
m_AF_EU,0,50
m_AF_AS,0,50
m_EU_AS,0,50
```
- demography is the name of the demography def, which is saved in the [Demography.py](Demography.py) file.
- inds is the number of individuals per population. 
- threads is the number of threads that would be used per simulation run where we create SFS using msprime using Run_Prior2SFS.py. 
- jobs are the number of different jobs you want to run separately. This option will break the Priors.csv into several smaller but independent files that can then run separately in parallel. 
- repeats are the number of repeats that are needed to run ABC-DLS SMC. Remember, this does not correspond to the number of the simulation run for this particular loop as we reuse some of the older simulations using  Narrowed.csv.
- total_length is the total length of the simulation that we want to run. Remember the total length is divided by equal 1mbp of LD region or chromosome to run it separately. 
- test_size is the number of test_size that would be kept for ABC-DLS. Everything else will be used for training. 
- decrease is the amount of decrease necessary to regard it as a true improvement. For more information please see 
[examples/Examples.md](../../examples/Examples.md) under SMC. 
- increase is the amount that should be added to the posterior ranges if it does not have any improvement or decrease. 
- hardrange_file is the starting range file. Important in case you use increase. 
- tolerance amount of tolerance that is necessary for ABC analysis.
 - frac is the amount of fraction to multiply with observed SFS to be equal to the simulated SFS. You can also write there !!float 1/647 
 
To run the snakemake, you can use just run in this folder:
```shell script
snakemake --jobs 10 
```  
This snkamake command will run all the necessary commands. It will run parallel ten jobs and will produce a Newrange.csv and Narrowed.csv. But this only one recursion. We need to do multiple recursion to make our posterior range much smaller. To do it, we need to put this code inside a while loop. We should also change the Newrange.csv to Oldrange.csv to run inside a loop until it reaches convergence.     
## Recursion 
The last and final part of this approach is to put it (snakemake pipeline) inside a recursive loop. We can do this in several ways. Here we present a simple shell script approach to do it ([recursive.sh](recursive.sh)).
```shell script
#!/bin/bash
imp=0
touch Narrowed.csv  All.csv
cp Startrange.csv Oldrange.csv
while [ "$(echo "$imp < 0.95"| bc -l)"  -eq 1 ]
do
	snakemake -q --jobs 6
	imp=$(cut -f4  -d ","  Newrange.csv |sort -n | head -n 1)
	mv Newrange.csv Oldrange.csv
done
mv Oldrange.csv Finalrange.csv
``` 
To run this code, we have to use:
```commandline
sh recursive.sh
```
This script will automatically run the recursion of the snakemake pipeline inside a while loop. We used [Startrange.csv](Startrange.csv) as a starting point of the whole recursion and Oldrange.csv as the starting point of every loop. The snakemake pipeline will submit six jobs in parallel, and when it reaches convergence (which is a minimum of imp > 0.95), it will stop
the while loop and save it as Finalrange.csv. This script is a very basic way to do it. Of course, you are free to update it and make it more complex to benefit your analysis.  
    