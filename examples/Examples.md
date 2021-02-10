# ABC-DLS Examples
This is a python pipeline where you can simply use summary statistics created from pop genome simulation files (ms, 
msprime, FastSimcoal etc.) to predict which underlying model can better explain the observed results, as well as which 
parameters can produce such results with the help of Deep Learning (using Tensorflow Keras Backended), Sequential Monte
Carlo (SMC) and Aproximate Bayesian Computation (ABC). The whole method is written in python and thus easy to read as 
well as can be accessed completely through command line thus knowing python is not needed. Although it will be helpful 
to know python as well as R as some of the packages here used is based on those languages. Remember both Classification 
and Parameter estimation will create file in the current working directory, thus you cannot run multiple runs together 
(as it would conflict with other runs). Thus either run one code at a time or run them in different folders using 
--folder option. 

## Classification and Model Selection 
This part explains how to choose the best model which can explain the observed data (in this case real sequenced data). 
This programme only accepts csv files (can be zipped) as an input. Thus simulations (i.e. ms, msprime, FastSimcoal etc) 
should be done elsewhere to produce summary statistics (ss) csv files (please see [SFS](../src/SFS/SFS_Examples.md)  for
more details). One example of ss can be Site Frequency Spectrum (SFS) but in principle anything, which can be 
represented in a single row with similar number of columns regardless of parameters or demography. The csv files should
be written like this: 
- Every row denotes one simulation under the model.
- First few columns should be the parameters that created the summary statistics.
- Everything else should be as the ss which would be used by TensorFlow (TF) to differentiate between models or predict 
back the parameters. 
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
In principle you do not need parameter columns for the classification part, but we kept it to make it similar to later 
part where parameters are required. 
``` sh
python  src/Run_Classification.py --help 
``` 
Will show 5 different methods: 


| Methods | Helps | 
| ------ | ------ | 
| All| The whole run of the NN for parameter estimation from first to last | 
| Pre_train | To prepare the data for training ANN | 
| Train | The training part of the ANN. Should be done after Pre_train part | 
| CV | After the training only to get the result of cross validation test. Good for unavailable real data | 
| After_train | This is to run the ABC analysis after the training part is done | 
### Pre_train  
This is the pre training part. Where the data is prepared for the TF.  
``` commandline
python src/Run_Classification.py Pre_train examples/Model.info --scale 
``` 
- input: [examples/Model.info](Model.info)  
Can be any text file which has all the demographic models that we want to compare together. Every line is denoted for 
one demographic simulation csv file. We should also denote the number of columns which are for the parameters present 
in that file. This will remove those columns from the files as they are not ss (parameters in this case)for the 
comparison.  
Should look like:
``` text
<Model1.csv.gz> <param_n>
<Model2.csv.gz> <param_n>
```
- --scale  
To scale the SFS data per column.So that all the data inside a column should be with in 0-1. This improves the 
prediction substantially.

This will create in total 3 files. x.h5 (this is for ss), y.h5 (models names in integer format) and y_cat_dict.txt 
(models name and their corresponding integer number). These first two part will be needed to run the neural network 
training for TF. The last part will be used later. 
### Train 
The next part is to run the training part by neural network. This will train the network to differentiate between models
.  
```commandline
python src/Run_Classification.py Train --demography src/extras/ModelClass.py --test_size 1000 
``` 
This will train the model.  
-  --nn src/extras/ModelClass.py 
ABC-DLS has a default training neural model. But it is impossible to predict which model should be better for the input 
data. Thus, we can define custom made model cater to our own need. One such example is 
[src/extras/ModelClass.py](../src/extras/ModelClass.py). Here we put very less epochs (only 20) to get faster result. 
More is merrier of course. The *.py should have a definition name ANNModelCheck which should return the trained model 
(after model.fit) and has two input x and y. Example:
```python
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import *

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
This will keep the last 1000 line save for test data set and will be used for ABC analysis. In the previous step it is 
already shuffled the data thus last 1000 lines you would expect near equal number of demographies.  

This will save the neural model as ModelClassification.h5 which we can use later for other use.  
### CV 
The next will be calculating the CV error to see if our Neural network is even capable of differentiating between models
. This step is important if we do not have the observed data. If you have the observed data in hand use the next part 
(After_train) 
``` commandline
python src/Run_Classification.py CV --test_size 1000 --tolerance 0.01 
``` 
- --test_size 1000  
This is the number of simulation rows (in total) will be used for test data set. This test data set is never been seen 
by the neural network, thus useful to see if your model is over fitting. Only this test data set will be used for ABC 
(r package) analysis.  
- --tolerance .01  
This parameter is needed by the ABC analysis to tell how much tolerance your model can have. 
 
This will print out the confusion matrix as well as save the CV.pdf where we can understand the power of neural network 
to differentiate between models. 
### After_train 
This part is similar to CV part, but it has the observed file together in the step thus can be used to see which 
demographic model can better explain the result.  
``` commandline
python src/Run_Classification.py After_train --test_size 1000 --tolerance 0.01 --ssfile examples/YRI_FRN_HAN.observed.csv --frac  4.636757528 --csvout 
``` 
- --test_size 1000 and --tolerance .01  
Same as above for CV 
- ssfile [examples/YRI_FRN_HAN.observed.csv](YRI_FRN_HAN.observed.csv)  
To define the observed csv file. Here we put YRI_FRN_HAN (Yoruba, French and Han Chinese) from the 
[High Coverage HGDP data](https://doi.org/10.1126/science.aay5012) sfs file as ss.  
- --frac  
  To define a fraction which has to be multiplied with observed data, in case the length of simulated region do no match
  with observed or real data. For example in this case I have simulated 3gbp regions for individuals, but the real
  data comes after filtering around 647mbp region. To make it equal I have to multiply the observed data with 
  (3gbp/647mbp) or 4.636757528
- --csvout  
If you are happy with all the result you can use csvout. This will remove .h5 files to free up space but also will 
produce csv files which then can directly be used R to further improve the results using abc if necessary. As all the 
commands of abc is not supported here in ABC-DLS directly.

This will print out (including the CV part) which models is better explained by the NN. It will also print out the 
goodness of fit to see if our observed model predicted by NN comes naturally under all the distribution of such model. 
If you csvout it will additionally output model_index.csv.gz (all the model indexes), ss_predicted.csv.gz (prediction 
from simulated ss by NN) and ss_target.csv.gz (prediction of the observed or real data). 
### All
In case rather than doing it separately, we can do all these stuffs together in one command.  
``` commandline
python src/Run_Classification.py All --test_size 1000 --tolerance 0.01 --ssfile examples/YRI_FRN_HAN.observed.csv --nn src/extras/ModelClass.py examples/Model.info --folder check --scale --frac  4.636757528
```
- --folder   
to run the whole stuff inside a folder so that it does not create a lot of files in the current directory.
 
It will produce the same files as previously but all of them together. If we do not use --chunksize it will produce 
x_test.h5 and y_test.h5 (of course if we use csvout it will be deleted) instead of x.h5 y.h5 as it will keep the 
training part on the ram itself. If you reach ram memory error, please use chunksize which will be relatively slower 
but do not have any upper limit for the file size.   
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
Here we try to predict the parameters which can explain the observed results for a given model. This will follow 
similarly as the previous steps. But instead of multiple files for different demography it will only use one of the 
files for parameter estimation (for a given model). It has similar structure like Classification: All, Pre_train, Train,
CV, After_train. 
### Pre_train
As classification part it will prepare the data for training. 
```commandline
python src/Run_ParamsEstimation.py Pre_train examples/Model.info --scale b
```
- input: [examples/Model.info](Model.info)   
Exactly like classification. But in case the Model.info file has multiple lines; it will only use the first line for the
 parameter estimation. 
--scale b
If the data (both x which is ss and y which is parameter) is not already scaled it will be scaled. It will be scaled 
using MinMaxscaler, which means all the number will be with in 0 to 1. 

It will produce x.h5 and y.h5 like previously. The main important different is for y.h5 whereas in classification part 
it only needed model names, here it will use parameters. It will also save params_header.csv to know the name of the 
parameters. 
### Train
Same as classification this will train the model. Unlike the classification, it takes a lot of epochs and very 
convoluted model to get high score for the accuracy (to further increase use the SMC part before). 
```commandline
python src/Run_ParamsEstimation.py Train --nn src/extras/ModelParams.py --test_size 1000
```
- --demography src/extras/ModelParams.py
Although there is a default method present in ABC-DLS (meaning python src/Run_ParamsEstimation.py Train --test_size 1000
will also work), we can give a model from outside. Here we kept 100 to make it faster. *.py must have a def name 
ANNModelParams. We can decide the number of epochs and other stuff inside that definition. The structure of the file is 
very similar. Only ANNModelParams instead of ANNModelCheck and also the output of neural network is linear (which is 
default for keras) instead of softmax. Everything else should be done as your model prefers. Example:
```python
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import *


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
As above kept 1000 samples for later use. All the other samples are used for training part. 

This will save the model as ModelParamPrediction.h5, which later can be used for prediction and stuff. 
### CV
To calculate cross validation error of the parameters. 
```commandline
python src/Run_ParamsEstimation.py CV --test_size 1000 --tolerance .01 --method loclinear
```

- --test_size 1000 --tolerance .01  
Same as clasiification.
- --method loclinear  
To tell which method to be used for CV. We found that generally loclinear is good for CV. On top of that if we use 
ABC-DLS cv method either it will produce the CV independently per column for NN prediction (when using loclinear or 
rejection) or it will produce CV by all the columns together by NN prediction (when using neuralnet or rejection). 

It will either produce nnparamcv.pdf (for using loclinear), nnparamcv_together.pdf (for using neuralnet) or both (for 
rejection) and print the cv error table in the terminal. It will also produce a correlation matrix for prior and 
posterior. This is important to see if there are some parameters becoming more correlated than the prior which might be 
the drawback of the ss, or the neural network used by TF. 
### After_train
After everything is done, we can use the After_train to use the ABC analysis. 
```sh
python src/Run_ParamsEstimation.py After_train --test_size 1000 --tolerance .01 --method loclinear --csvout --ssfile examples/YRI_FRN_HAN.observed.csv --frac  4.636757528
```
This will calculate both the CV part as well as will compare with the observed data. This will produce 
paramposterior.pdf to see the prior vs posterior. It will also produce the same csv file as before but instead of 
model_index.csv.gz will produce params.csv.gz. Inside those files will be necessary information for the parameters. 
### All
To put all these parts together we can use: 
```sh
python src/Run_ParamsEstimation.py All --nn src/extras/ModelParams.py --test_size 1000 --tolerance .01 --method loclinear --csvout --ssfile examples/YRI_FRN_HAN.observed.csv --scale b --frac  4.636757528 examples/Model.info
```
It will produce similar result but running all the commands together.
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
This will transform the parameter values in log scale. Thus, we can calculate the distance much more precisely. 
## Parameter Estimation by SMC
Now Parameter Estimation by ABC-DLS is good but what if we want to do it recursively. First and foremost we need to 
understand why doing it recursively is better in this case. Think it like this, before the training we did not know 
what is the amount of admixture from first population to second populations (suppose the true amount is 30%). As our 
priors it can be 10%, can be 20%.. 90%. anything is possible. So we run every possible admixture amount and neural 
network learns how the ss should look under 10%,20%..90% admixture. After we use the real/observed data suppose it 
predicted the amount is 20-50% (posterior). Now we can only concentrate on simulations from 20-50% admixture as there is
no need of making the neural network learn how the ss behaves in those extreme conditions (<20% and >50%) when they are 
unlikely. Neural network now can specialize in much smaller deviated ss which in turn make it much powerful for 
prediction. Of course, we can simulate infinite number of lines to make the neural network learn from that. Instead we 
are making the neural network learn recursively for the amount which we think is true and by doing that we are making it
 much more specialized. The simplified idea is to get the minimum and maximum value for every parameter as a posterior 
 and use that posterior as a prior to create new simulation and repeat it again (aka Sequential Monte Carlo or SMC which
 is also sometime called Particle Filter). This recursion should be done till convergence is reached. In this case when 
 imp (improvement) of every parameter is more than 95% (default value), we can assume we have reached enough convergence
 and neural network now cannot make any more improvement.
<img src="https://latex.codecogs.com/gif.latex?imp=\frac{Posterior_{max}-Posterior_{min}}{Prior_{max}-Prior_{min}}" title="imp=\frac{Posterior_{max}-Posterior_{min}}{Prior_{max}-Prior_{min}}" />  
To run the SMC for Parameter Estimation for a single time:   
```shell
python src/Run_SMC.py All --folder SMC --nn src/extras/ModelParamsTogether.py --test_size 1000 --tolerance .05 --csvout --ssfile examples/YRI_FRN_HAN.observed.csv --scale b examples/Model.info
``` 
The code is similar to Parameter Estimation part with some added changes which makes it more efficient for recursion 
(which basically means it will remove most of the extra test and graphs and only produce the range).  
 - --demography src/extras/ModelParamsTogether.py  
 The format is slightly different than the Parameter Estimation. The idea is train and test both are send together for 
 training part to make it more efficient. As it is recursive method wasting of simulations do not make sense. In case 
 you want you can use the default nueral network which works most of the time. In case you want to use your own neural 
 network model you have to follow this format:
```python
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import *
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
will keep the simulation which are with in the new (posterior) range of parameters. Thus can be reused another round(s) 
of iteration.  

It will also save a new file called Newrange.csv which would have information about the posterior range (which has 
information from ABC minimum and maximum range). 
### Recursion
If we cannot do the recursion, there is no difference between Parameter Estimation and Parameter Estimation with SMC. 
You can think of SMC part is a subset and efficient version of normal Parameter Estimation part for a single iteration. 
First of all ABC-DLS is in principle meant for any ss (not only SFS). Thus not all possible ss can be written here. 
Secondly, the production of ss files takes times and it is impossible to run the code for production of ss in a single 
computer. You need a cluster as well as a pipeline (for example snakemake), which can submit multiple ss files in 
parallel. Nonetheless here we will give an idea how to do it but its reader discretion how to implement such a pipeline:
#### Iterations
```text
do while any parameter imp < 0.95
    Produce the prior parameters with in some range
    Produce SS from those parameters (heavily parallelize here)
    Merge parameters and their corresponding ss together so it can be used in ABC-DLS
    python src/Run_SMC.py ..
    remove unimportant files
```
You can look at [src/SFS/SFS_Examples.md](../src/SFS/SFS_Examples.md) to have an idea how to do it. 
## Good Practices
- Never believe in one run of ABC-DLS. It is always better to run several separate simulations (pre-train) with several 
different neural models (training, you can find some in [src/extras/](src/extras/) folder) and see it reaches the same 
outcome. After training also use differently observed ss (after train) files give same results. One example might be to 
use different mask strategy than what is used here like mappability mask (using 
[snapable programme](http://lh3lh3.users.sourceforge.net/snpable.shtml)) ss files and compare the results coming from 
them. Same things can also be achieved by producing same ss from different individuals or use boot strap results from 
same individuals.     
- Take care of over fitting by checking accuracy in training data set vs test data set. If training accuracy is very 
high compared to test data set, try to run a smaller number of epochs and or use more data to train. If you are using 
less data than what is needed, your train data set accuracy will diverge from test data set accuracy very early on 
(<50 epochs). This suggests your training might be improved by using more data to train. On the other hand more data is 
always better. Especially because we can simulate easily more and more data synthetically. In principle, memory is not a
problem for ABC-DLS as the code is implemented in hdf5 format. Thus you have unlimited memory. But take care, as more 
data also means it will take more time to converge. As a rule of thumb, we found that 2k (1k for training and 1k for ABC
  ) simulations for classification, 60k (50k for training and 10k for ABC) simulations for parameter estimation and 20k 
  (10 k for training and 10k for ABC) simulations for SMC approach per 
  model is generally enough when using SFS as a ss. 
- It is easy to build different models, but it is your responsibility to make them as much different as possible. For 
example, a 0% admixture proportion essentially means a model without admixture (aka in this case normal Out of Africa 
from the paper). Thus, admixture model should be more than 0% of admixture. What minimum percentage of admixture is good
to train the data depends on the model itself (e.g. older the population split; it is easier to differentiate thus 
needed lesser admixture to detect). You can look at different models for SMC approach and if they converge then most 
probably it is better idea to put some hard range thus the two models cannot be equal (fore example the admixture model
put the admixture amount from 1-99%). 
- Use migrations (under island model) cautiously, till a better method (or ss) to calculate migrations was found. 
Although this approach gives the freedom to use migrations, it should be used moderately. We found that migrations make 
the result less accurate (at least in the current form SFS + Neural network), as well as big amount of migrations 
between populations has demonstrated to change the underlying tree of demographic history and thus the interpretation of
the whole model itself. Migrations can affect the result indirectly which is not easy to understand. Try to check if 
your model gives the same result with or without migrations. If not revisit your model without migration, rather 
believing your model with migrations. 
- Remember garbage in garbage out principle. If you use nonsense data as an input, you will get nonsense result as an 
output. Although using ABC method, it is easier now to catch such situation (as ABC gives posterior distribution rather 
a single number) but by no means it is full proof. On top of it, neural network is black box thus some time it is near 
impossible to catch such mistakes (neural network will learn anything if you force it to learn). Suggested direction 
will be to start from an already known and accepted results. See if you get similar results and then try to make more 
complex models on top of it. 

 




