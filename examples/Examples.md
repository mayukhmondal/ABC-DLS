# ABC-DLS Examples
This software is a python pipeline where you can use simulated summary statistics to predict which underlying model can
better explain the observed or real-world results (classification) and can guess which parameters can produce such results (parameter estimation) with the help of Approximate Bayesian Computation (ABC), Deep Learning (using Tensorflow Keras Backend) and Sequential Monte Carlo (SMC). The simulations are preferably coming from pop genome simulators i.e. ms, msprime, FastSimcoal etc. but not necessarily bounded only by that. The whole method is written in python and thus easy to read. It can be accessed entirely through the command line, therefore knowing python is not essential, though it will be helpful to know python and a little bit of R as some of the packages in the pipeline is based on those languages. Remember, both classification and parameter estimation will create files in the current working directory or a directory mentioned in the command. Thus you cannot run multiple runs together in the same folder as it would conflict with other runs. Either run one code at a time or run them in different folders using the --folder option.
## Classification and Model Selection 
This part explains how to choose the best model to explain the observed data (or, in this case, real sequenced data). 
This programme only accepts csv files (can be zipped) as an input. The simulations (i.e., ms, msprime, FastSimcoal etc.) 
should be done elsewhere to produce summary statistics (ss) csv files (please see [SFS](../src/SFS/SFS_Examples.md) 
or [cSFS](../src/SFS/cSFS_Examples.md) for more details). One example of ss can be Site Frequency Spectrum (SFS). 
However, any ss, which can be represented in a single row with a similar number of columns independent of parameters or 
demography, can be used. The csv files should
be written like this: 
- Every row denotes one simulation under the model.
- First few columns should be the parameters that created the summary statistics.
- Everything else should be as the ss, which would be used by TensorFlow (TF) to differentiate between models or predict back the parameters. 
- The files should have a header. 

You can look inside the examples/*.csv.gz files to get an idea. 
```sh
N_A,N_AF,N_EU,N_AS,...,0_0_0,0_0_1,0_0_2,0_0_3,0_0_4,...
14542.382466372237,119646.25929867383,75043.3425995741,103496.95227233913,...,0.0,676743.0,128199.0,47163.0,19450.0,...
14780.566576552284,20743.142386406427,90821.23078107052,117292.89382816535,...,0.0,609543.0,132621.0,54121.0,23711.0,...
15068.855032319485,71129.50749663456,71222.94672045513,119242.37644455553,...,0.0,611102.0,132722.0,57052.0,27447.0,...
14533.876139703001,25492.550958201846,78599.68927419234,74284.75369536638,...,0.0,781011.0,169080.0,78278.0,43865.0,...
14620.827267084273,92068.73287607494,99109.69362439432,140694.91698723484,...,0.0,632596.0,116719.0,44490.0,19010.0,...
```
In principle, you do not need parameter columns for the classification part, but we kept the file format similar to the later part where parameters are required. 
``` sh
python  src/Run_Classification.py --help 
``` 
There are 5 different part of the methods:

| Methods | Helps | 
| ------ | ------ | 
| Pre_train | To prepare the data for training in Neural Network (NN) | 
| Train | The training part of the NN. Should be done after Pre_train part | 
| CV | After the training only to get the result of the cross-validation test. Good for unavailable real data | 
| After_train | This is to run the ABC analysis after the training part is done | 
| All| The whole run of the ABC-DLS for classification from first to last. It will run all of the above steps together.  | 
### Pre_train  
This is the pre training part. Where the data is prepared for the NN.  
``` commandline
python src/Run_Classification.py Pre_train examples/Model.info --scale 
``` 
- input: [examples/Model.info](Model.info)  
It can be any text file that has all the demographic models that we want to compare together. Every line is denoted for one demographic simulation csv file. We should also denote the number of columns that are present in that file for the parameters. It will remove those columns from the files as they are not ss (parameters in this case) for the comparison.  
It should look like:
``` text
<Model1.csv.gz> <param_n>
<Model2.csv.gz> <param_n>
```
- --scale  
This option would scale the SFS data per column. So that all the data inside a column should be within 0-1. This strategy improves the prediction substantially.

This command will create in total three files. x.h5 (this is for ss), y.h5 (models names in integer format) and y_cat_dict.txt (models name and their corresponding integer number). These first two files will be needed to run the NN training for TF. The last part will be used later. 
### Train 
The next part is to run the training part by NN. This command will train the NN to differentiate between models.
```commandline
python src/Run_Classification.py Train --nn src/extras/ModelClass.py --test_size 1000 
``` 
This command will train the model.  
-  --nn [src/extras/ModelClass.py](../src/extras/ModelClass.py)  
ABC-DLS has a default training neural model. But it is impossible to predict which model should be better for the input data. Thus, we can define custom made model cater to our own need. One such example is [src/extras/ModelClass.py](../src/extras/ModelClass.py). Here, we put very few epochs (only 20) to get faster results. More is merrier, of course. The *.py should have a definition name ANNModelCheck, which should return the trained model (after model.fit) and has two inputs, x and y. Example:
```python
from tensorflow.python import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

def ANNModelCheck(x, y):
    """
    your own set of code for tensorflow model. can be both Sequential or Model class (functional API). check 
    src/extras/*.py to have an idea. also check https://keras.io/ to understand how to make Keras models
    """
    model = Sequential()
    model.add(...)
    ...
    model.add(Dense(y.shape[1], activation='softmax')) # this line is important for classification
    # Model Class can also be used. Example: 
    # x_0 = Input(shape=(x.shape[1],))
    # x_1 = Dense(128, activation='relu')(x_0)
    # x_1= Dense(y.shape[1], activation='softmax') (x_1)
    # model = Model(inputs=x_0, outputs=x_1)
    model.compile(...)
    # we found for classification in model.complile loss=keras.losses.categorical_crossentropy, optimizer='adam'; these 
    # two gives the best results
    model.fit(x, y,...)
    return model
```
- --test_size 1000  
This option will keep the last 1000 lines left for the test data set and be used for ABC analysis. It has already shuffled data in the previous step. Thus, you would expect nearly an equal number of simulations for every demography in the last 1000 lines.  

This command will save the neural model as ModelClassification.h5, which we can use later.
### CV 
The next will be calculating the CV error to see if our NN can differentiate between models. This step is essential if we do not have the observed data. If you have the observed data in hand, use the next part (After_train). 
``` commandline
python src/Run_Classification.py CV --test_size 1000 --tolerance 0.01 
``` 
- --test_size 1000  
This option will define the number of simulation rows (in total) used for the test data set. This test data set has never been seen by the neural network, which is useful for seeing if your model is overfitting. Only this test data set will be used for ABC (r package) analysis.  
- --tolerance .01  
The ABC analysis needs this parameter to tell how much tolerance your model can have. 
 
This command will print out the confusion matrix as well as save the CV.pdf where we can understand the power of NN to differentiate between models. 
### After_train 
This part is similar to the CV part, but it has the observed file together in the step. Thus can be used to see which demographic model can better explain the observed data.  
``` commandline
python src/Run_Classification.py After_train --test_size 1000 --tolerance 0.01 --ssfile examples/YRI_FRN_HAN.observed.csv --frac  4.636757528 --csvout 
``` 
- --test_size 1000 and --tolerance .01  
Same as above for CV 
- --ssfile [examples/YRI_FRN_HAN.observed.csv](YRI_FRN_HAN.observed.csv)  
To define the observed csv file. Here we put YRI_FRN_HAN (Yoruba, French and Han Chinese) from the [High Coverage HGDP data](https://doi.org/10.1126/science.aay5012) SFS file as SS. Having header is mandatory. It can have multiple lines. Individual line signifies same summary statistics coming from different individuals.  If have many lines, every line will run independently for abc analysis (unlike for parameter estimation or SMC, where an average of the nn predicted values will be used).  
- --frac  4.636757528
This option will define a fraction that has to be multiplied with observed data if the length of the simulated region does no match with observed or real data. For example, I have simulated 3gbp regions per individual in this particular case, but the real data comes after filtering around 647mbp region. To make it equal, I have to multiply the observed data with (3gbp/647mbp) or 4.636757528
- --csvout  
If you are happy with all the results, you can use csvout. This option will remove .h5 files to free up space and produce csv files, which can be used to improve the results further using R_abc if necessary. As all the commands of abc is not supported here in ABC-DLS directly.

This command will print out (including the CV part) which underlying model better explains the observed data. It will also print out the goodness of fit to see if our observed model predicted by NN comes naturally under all the distribution of such model. If you use csvout, it will additionally output model_index.csv.gz (all the model indexes), ss_predicted.csv.gz (prediction from simulated ss by NN) and ss_target.csv.gz (prediction of the observed or real data). 
### All
In case, rather than doing it separately, we can do all these together in one command.  
``` commandline
python src/Run_Classification.py All --test_size 1000 --tolerance 0.01 --ssfile examples/YRI_FRN_HAN.observed.csv --nn src/extras/ModelClass.py examples/Model.info --folder check --scale --frac  4.636757528
```
- --folder   
This option will run the whole stuff inside a folder so that it does not create a lot of files in the current directory.
 
It will produce the same files as previously but all of them together. If we do not use --chunksize, it will create x_test.h5 and y_test.h5 (of course, if we use csvout it will be deleted) instead of x.h5 y.h5 as it will keep the training part inside the RAM itself. If you reach memory error, please use chunksize, which will be relatively slower but do not have any upper limit for the file size.   
### Optional 
We can easily use this result in R to further our analysis: 
``` R
library(abc) 
ss_predict=read.csv('ss_predicted.csv.gz') 
target=read.csv('ss_target.csv.gz') 
index=as.vector(factor(as.matrix(read.table('model_index.csv.gz',header=TRUE)))) 
cv4postpr(as.vector(index),as.matrix(ss_predict[,-1]),nval=1000,method='rejection',tols=c(.01,.1)) 
``` 
To see with different amount of tolerance level and nval how the abc analysis changes. 
## Parameter Estimation
Here, we try to predict the parameters which can explain the observed results for a given model. This part will follow similarly to the previous steps. But instead of multiple files for different demography, it will only use one of the files (the very first) for parameter estimation (for a given model). It has a similar structure like Classification: All, Pre_train, Train, CV, After_train. 

### Pre_train
As the classification part, it will prepare the data for training. 
```commandline
python src/Run_ParamsEstimation.py Pre_train examples/Model.info --scale b
```
- input: [examples/Model.info](Model.info)   
Exactly like the classification part. But if the Model.info file has multiple lines, it will only use the first line for the parameter estimation. 
--scale b
If the data (both x, which is the ss and y, which is the parameters) is not already scaled, it will be scaled. It will be scaled using MinMaxscaler, which means all the numbers will be within 0 to 1 per column. 

It will produce x.h5 and y.h5 like previously. The main significant difference is for y.h5. Whereas in the classification part, it only needed model names; here, it will use parameters. It will also save params_header.csv to know the name of the parameters. 
### Train
Same as the classification part, this will train the model. Unlike the classification, it takes many epochs and a complicated NN model to get high scores for accuracy (using the SMC part before further increases the prediction power with more precision ). 
```commandline
python src/Run_ParamsEstimation.py Train --nn src/extras/ModelParams.py --test_size 1000
```
- --demography [src/extras/ModelParams.py](../src/extras/ModelParams.py)  
Although there is a default method present in ABC-DLS (meaning python src/Run_ParamsEstimation.py Train --test_size 1000 will also work), we can give a model from outside. Here we kept 100 epochs to make it faster. *.py must have a def name ANNModelParams. We can decide the number of epochs and other stuff inside that definition. The structure of the file is very similar. Only ANNModelParams instead of ANNModelCheck and the NN output is linear (which is default for keras) instead of softmax. Everything else should be done as your model prefers. Example:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *


def ANNModelParams(x, y):
    """
    your own set of code for tensorflow model. can be both Sequential or Model class (functional API). check 
    src/extras/*.py to have an idea. also check https://keras.io/ to understand how to make Keras models
    """
    model = Sequential()
    model.add(...)
    ...
    model.add(Dense(y.shape[1])) # this line is important for parameter estimation
    # Model Class can also be used. Example: 
    # x_0 = Input(shape=(x.shape[1],))
    # x_1 = Dense(128, activation='relu')(x_0)
    # x_1= Dense(y.shape[1]) (x_1)
    # model = Model(inputs=x_0, outputs=x_1)
    model.compile(...)
    # we found for parameter estimation in model.complile, loss='logcosh' and optimizer='Nadam', these two gives the 
    # best results
    model.fit(x, y,...)
    return model
``` 
- --test_size 1000  
As above, it kept 1000 samples for later use. All the other samples are used for the training part. 

This command will save the model as ModelParamPrediction.h5, which later can be used for prediction and stuff. 
### CV
To calculate the cross-validation error of the parameters. 
```commandline
python src/Run_ParamsEstimation.py CV --test_size 1000 --tolerance .01 --method loclinear
```
- --test_size 1000 --tolerance .01  
Same as clasiification.
- --method loclinear  
This option is used to tell which method to be used for CV. We found that generally, loclinear is good for CV. On top of that, if we use 
ABC-DLS cv method will either produce the CV independently per column for NN prediction (when using loclinear or rejection) or calcualte CV by all the columns together by NN prediction (when using neuralnet or rejection). 

It will either produce nnparamcv.pdf (for using loclinear), nnparamcv_together.pdf (for using neuralnet), or both (for rejection) and print the cv error table in the terminal. It will also produce a correlation matrix for prior and posterior. This matrix is important to see if some parameters are becoming more correlated than the prior, which might be the drawback of the ss, or the NN used by TF. 
### After_train
After everything is done, we can use the After_train to use the ABC analysis. 
```sh
python src/Run_ParamsEstimation.py After_train --test_size 1000 --tolerance .01 --method loclinear --csvout --ssfile examples/YRI_FRN_HAN.observed.csv --frac  4.636757528
```
This command will calculate both the CV part as well as will compare it with the observed data. The command will produce paramposterior.pdf to see the prior vs posterior. It will also create the same csv file as before, but instead of model_index.csv.gz will generate params.csv.gz. Inside those files, there will be necessary information for the parameters. 

### All
To put all these parts together, we can use:
```sh
python src/Run_ParamsEstimation.py All --nn src/extras/ModelParams.py --test_size 1000 --tolerance .01 --method loclinear --csvout --ssfile examples/YRI_FRN_HAN.observed.csv --scale b --frac  4.636757528 examples/Model.info
```
It will produce a similar result but running all the commands together.
### Optional 
We can use further our analysis in R:
```R
library(abc)
params=read.csv('params.csv.gz')
ss=read.csv('ss_predicted.csv.gz')
target=read.csv('ss_target.csv.gz')
res=abc(target = target,param=params,sumstat = ss,tol=.01,method='neuralnet',transf = 'log')
summary(res)
plot(res,param=params)
```
This command will transform the parameter values in log scale. Thus, we can calculate the distance much more precisely. 
## Parameter Estimation by SMC
Now Parameter Estimation by ABC-DLS is good, but what if we want to do it recursively. First and foremost, we need to understand why doing parameter estimation recursively is better in this case. For example, think it like this: before the training, we did not know the amount of admixture from the first population to the second population (suppose the actual amount is 30%). As our priors are 10%-90%, it can be 10%, 20%, .. and 90%. Anything is possible. So we run every possible admixture amount and NN learns how the ss should look under 10%,20%..90% admixture amount. We use the real/observed data, suppose it predicted the amount is 20-50% (posterior). Now we can only concentrate on simulations from 20-50% admixture as there is no need to make the NN learn how the ss behaves in those extreme conditions (<20% and >50%) when they are unlikely. NN now can specialize in much smaller deviated ss, making it much powerful for prediction. Of course, we can simulate an infinite number of lines to make the NN learn from that. Instead, we are making the NN learn recursively for the amount we think is accurate and by doing that, we are making it much more specialized. The simplified idea is to get the minimum and maximum value for every parameter as a posterior and use that posterior as a prior to create a new set of simulations and repeat it (aka Sequential Monte Carlo or SMC, which is also sometimes called Particle Filter). This recursion should be done till convergence is reached. In this case, when dec (decrease) of every parameter is more than 95% (default value), we can assume we have acquired enough convergence and the NN now cannot make any more improvement. 

<img src="https://latex.codecogs.com/gif.latex?dec=\frac{Posterior_{max}-Posterior_{min}}{Prior_{max}-Prior_{min}}" title="dec=\frac{Posterior_{max}-Posterior_{min}}{Prior_{max}-Prior_{min}}" />  

To run the SMC for Parameter Estimation for a single time:   
```shell
python src/Run_SMC.py All --folder SMC --nn src/extras/ModelParamsTogether.py --test_size 1000 --tolerance .05 --csvout --ssfile examples/YRI_FRN_HAN.observed.csv --scale b examples/Model.info --decrease 0.95 --increase .01 --hardrange examples/hardrange.csv
``` 
The code is similar to the parameter estimation part. Some added changes make it more efficient for recursion (removing most of the extra tests and graphs and only producing the range).  
 - --demography [src/extras/ModelParamsTogether.py](../src/extras/ModelParamsTogether.py)  
The format is slightly different than the Parameter Estimation. The idea is train and test are send together for the training part to make it more efficient. As it is a recursive method, wasting of simulations does not make sense. In case you want, you can use the default NN, which works most of the time. In case you want to use your own NN model, you have to follow this format:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
import numpy
# from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def ANNModelParams(x, y):
    # main difference from Parameter Estimation. Sending both (_train,_test) together as tupple 
    x_train, x_test = x
    y_train, y_test = y
    #This part is same as before
    model = Sequential()
    model.add(...(input_shape=(x_train.shape[1],)))
    ...
    model.add(Dense(y_train.shape[1]))
    #example
    # model.add(GaussianNoise(0.05, input_shape=(x_train.shape[1],)))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dense(y_train.shape[1]))
    model.compile(...)
    #example
    # model.compile(loss='logcosh', optimizer='Nadam', metrics=['accuracy'])
    model.fit(x_train, y_train, ...,validation_data=(numpy.array(x_test), numpy.array(y_test)))
    # example 
    # adding an early stop so that it does not overfit
    # ES = EarlyStopping(monitor='val_loss', patience=100)
    # checkpoint
    # CP = ModelCheckpoint('Checkpoint.h5', verbose=0, save_best_only=True)
    # Reduce learning rate
    # RL = ReduceLROnPlateau(factor=0.2)
    # model.fit(x_train, y_train, epochs=5, verbose=0, shuffle="batch", callbacks=[ES, CP, RL],
    #           validation_data=(numpy.array(x_test), numpy.array(y_test)))

    return model
```   
- --csvout  
This option will keep the simulations that are within the new (posterior) range of parameters. Thus can be reused for another round(s) of iteration.  
- --decrease 0.95  
This option is the amount of decrease necessary to regard it as a genuine improvement. Because we are choosing the top 5% for the parameters in ABC, we always expect that the posterior range would be smaller than prior (even though the NN do not have power to predict that). To remove such estimation, we use this 95% filter, which means if the posterior is not less than 95% of the prior, we will regard the prior range as the posterior range. This filtering is essential as we are recursing so that the ranges do not decrease incorrectly.
- --increase .01  
There is always a chance when some decrease of range happened in one cycle, it missed the actual target value (suppose your actual introgression amount is 3%, but it was predicted in a cycle to be 1-2% wrongly). We use this parameter to get back the true introgression in the subsequent cycle.  This option will increase the distance between the lower and upper limit by 1% (if 0.01 was used). So in a sense, you can treat increase and decrease two opposing forces. The decrease will shorten the distance between the upper and lower range, whereas the increase will broaden it up. After multiple cycles, they (increase and decrease) together generally reach a convergence. But remember to put the increase much lower than the decrease (typically five times lesser than 1-decrease). If not, it can be stuck in an infinite loop.  
    <img src="https://latex.codecogs.com/svg.latex?Posterior_{min}=Prior_{min}-(Prior_{max}-Prior_{min})\times%20\frac{increase}{2} " />   
  <img src="https://latex.codecogs.com/svg.latex?Posterior_{max}=Prior_{max}+(Prior_{max}-Prior_{min})\times%20\frac{increase}{2} " />
- --hardrange [examples/hardrange.csv](hardrange.csv)    
If the increase was not used, this file is not required as in every cycle the distance between lower and  upper limit can only go lower. But in case of increase is used, the range can grow bigger and sometimes come to a  point where the range does not make sense anymore (for example, admixture amount more than 100%), or the range is outside of what is your prior belief. Thus it is a good idea to give the starting range as a hardrange file so that your simulations will always be within that limit of starting range. Please follow the format in the examples/hardrange.csv file.

It will printout the Posterior range (which has information from ABC minimum and maximum range) and if hardrange is given it will also print out log of mean range decrease (lmrd). lmrd is a easy way to understand how much decrease or improvement you have gotten in the posterior over the hardrange or starting range. It will also save a new file called Newrange.csv, which would have information about the posterior range. 

### Recursion
If we cannot do the recursion, there is not much difference between Parameter Estimation and Parameter Estimation with SMC. SMC part is a subset and efficient version of the normal parameter estimation part for a single iteration. ABC-DLS is, in principle, meant for any ss (not only SFS). All the possible ways of producing ss can't be written here. Secondly, the production of ss files takes time and it is difficult to run the code for the production of ss on a single computer. It would help if you had a cluster and a pipeline (for example, snakemake), submitting multiple ss files in parallel. Nonetheless, here we will give an idea of how to do it but its reader discretion how to implement such a pipeline.

#### Iterations
```text
do while any parameter imp < 0.95
    Produce the prior parameters with in some range
    Produce SS from those parameters (heavily parallelize here)
    Merge parameters and their corresponding ss together so it can be used in ABC-DLS
    python src/Run_SMC.py ..
    remove unimportant files
```
You can look at [SFS_Examples](../src/SFS/SFS_Examples.md) or 
[cSFS_Examples](../src/SFS/cSFS_Examples.md) to have an idea how to do it. 

## Good Practices
- Never believe in one run of ABC-DLS. From my experience, generally the results do not change much under different conditions but as NN is a black box approach, it is always better to be sure than sorry. You can run several separate runs (from pre-train) with several different neural models (you can find some in [src/extras/](src/extras/) folder) and see if it reaches the same outcome. After training, also use differently observed ss (after train) files give identical results. One example might be to use a different mask strategy than what is used here, like mappability mask (using [snapable programme](http://lh3lh3.users.sourceforge.net/snpable.shtml)) ss files and compare the results coming from them. The same things can also be achieved by producing the same observed ss from different individuals or using bootstrap results from the same individuals (in case there are no other sequences available).     
- Take care of overfitting by checking accuracy in training data set vs. test data set. If training accuracy is very high compared to the test data set, try to run a smaller number of epochs or use more data to train. If you are using less data than needed, your train data set accuracy will diverge from test data set accuracy very early on (<50 epochs). This outcome suggests more data might improve your training.  On the other hand, more data is always better and we can simulate more and more data effortlessly. In principle, memory is not a problem for ABC-DLS as the code is implemented in hdf5 format. Thus you have unlimited memory (in principle). But take care, as more data also means it will take more time to converge. As a rule of thumb, we found that 2k (1k for training and 1k for ABC) simulations for the classification, 60k (50k for training and 10k for ABC) simulations for the parameter estimation and 20k (10 k for training and 10k for ABC) simulations for SMC approach are generally enough. 
- Use migrations (under island model) cautiously till a better method (or ss) to calculate migrations is found. Although this approach gives the freedom to use migrations, we should use it moderately. We discovered that migrations make the result less accurate (at least in the current form SFS + Neural network) and significant amount of migrations between populations has demonstrated to change the underlying tree of demographic history and thus the interpretation of the whole model itself. Migrations can affect the result indirectly, which is not easy to understand. Try to check if your model gives the same result with or without migrations. If not, revisit your model without migration, instead of believing your model with migrations. 
- Remember garbage in garbage out principle. If you use nonsense data as an input, you will get a nonsense result as an output. Although by using ABC method, it is easier to catch such a situation (as ABC gives posterior distribution rather than a single number) but it is not full-proof. On top of it, NN is a black-box approach. Thus, it is sometimes nearly impossible to catch such mistakes (NN will learn anything if you force it to learn). The suggested direction will be to start from an already known and accepted result. See if you get similar results and then try to make more complex models on top of it. 

