# ABC-TFK 

ABC-TFK is a python pipeline where you can simply use summary statistics created from pop genome simulation files (ms, msprime, FastSimcoal etc.) to predict which underlying model can better explain the observed results, as well as which parameters can produce such results. The whole method is written in python and thus easy to read as well as can be accessed completely through command line thus knowing python is not needed. Although it will be helpful to know python as well as R as some of the packages here used is based on those languages. 

## Classification and Model Selection 

The first part is to choose the best model which can explain the observed data (in this case real sequenced data). This programme only accepts csv files (can be zipped) as an input. Thus simulations (i.e. ms, msprime, FastSimcoal etc) should be done elsewhere to produce summary statistics (ss) csv files. One example of ss can be Site Frequency Spectrum [SFS] but can anything in principle which can be represented in a single row with similar number of columns regardless of parameters or demography. The csv files should be written like this: 

- Every row denotes one simulation under this model  

- First few columns should be the parameters that created the summary statistics.  

- Everything else should be as the ss which would be used by TensorFlow (TF) to differentiate between models or predict back the parameters. 

- The files should have a header.  

You can look inside the examples/*.csv.gz files to get an idea. 
```sh
N_AF,N_EU,N_AS,N_B,T_AF,T_B,T_EU_AS,0_0_0,0_0_1,0_0_2,...
41.91238472966386,8.309530807784103,2.765119195470757,0.033056478661619815,0.3241105917566006,1.9075173699598655,0.08600273078185215,0.0,0.01710902810227033,0.0024309585132660265,...
1.4036446805365959,28.806143458313183,40.793276938448685,0.07667751542033917,2.7942812729697404,0.1611504840934601,0.0009608538419194447,0.0,0.011552644156829899,0.002952178031510643,...
14.4699687789293,13.076507818094042,15.468141529239466,0.7566621341914272,2.827195148492236,0.2683689967394146,0.019711951401945545,0.0,0.06450083622477325,0.01793789070092734,...
33.49003204256888,8.991749677628123,38.47138632478331,0.05440611526914164,1.7826816037574387,1.7932789841164765,0.09935861063949145,0.0,0.018857425585701144,0.0017586872079860553,...
31.045117227506175,10.91187123855422,21.944990557726523,0.5565143283059646,2.8951686614340337,0.3440145000742446,0.10954860391870827,0.0,0.054948080811165076,0.01490294823057456,...
```
In principle you do not need parameter columns for the classification part, but we kept it to make it similar to later part where parameters are required. 
``` sh
python  src/Run_Classification.py --help 
``` 
Will output 5 different methods: 

| Methods | Helps | 
| ------ | ------ | 
| All| The whole run of the NN for parameter estimation from first to last | 
| Pre_train | To prepare the data for training ANN | 
| Train | The training part of the ANN. Should be done after Pre_train part | 
| CV | After the training only to get the result of cross validation test. Good for unavailable real data | 
| After_train | This is to run the ABC analysis after the training part is done | 
### Pre_train  
This is the pre training part. Where the data is prepared for the TF.  
``` sh
python src/Run_Classification.py Pre_train examples/Model.info  
``` 
- input: examples/Model.info  
Can be any text file which has all the demographic models that we want to compare together. Every line is denoted for one demographic simulation csv file. We should also denote the number of columns which are for the parameters present in that file. This will remove those columns from the files as they are not ss for the comparison.  
Should look like: 

> <Model1.csv.gz> <param_n> 

> <Model2.csv.gz> <param_n> 

>... 

This will create in total 3 files. x.h5 (this is for ss), y.h5 (models names in integer format) and y_cat_dict.txt (models name and their corresponding integer number). These first two part will be needed to run the neural network training for TF. The last part will be used later. 
### Train 
The next part is to run the training part by neural network. This will train the network to differentiate between models.  
``` sh
python src/Run_Classification.py Train --demography src/extras/ModelClass.py --test_size 1000 
``` 
This will train the model.  
-  --demography src/extras/ModelClass.py 
ABC-TFK has a default training neural model. But it is impossible to predict which model should be better for the input data. Thus, we can define custom made model cater to our own need. One such example is src/extras/ModelClass.py. Here we put very less epochs (only 20) to get faster result. More is merrier of course. 
- --test_size 1000  
This will keep the last 1000 line save for test data set and will be used for ABC analysis. In the previous step it is already shuffled the data thus last 1000 lines you would expect near equal number of demographies.  

This will save the neural model as ModelClassification.h5 which we can use later for other use.  
### CV 
The next will be calculating the CV error to see if our Neural network is even capable of differentiating between models. This step is important if we do not have the observed data. If you have the observed data in hand use the next part (After_train) 
``` sh
python src/Run_Classification.py CV --test_size 1000 --tolerance 0.01 
``` 
- --test_size 1000  
This is the number of simulation rows (in total) will be used for test data set. This test data set is never been seen by the neural network, thus useful to see if your model is overfitting. Only this test data set will be used for ABC (r package) analysis.  
- --tolerance .01  
This parameter is needed by the ABC analysis to tell how much tolerance your model can have.  

This will print out the confusion matrix as well as save the CV.pdf where we can understand the power of neural network to differentiate between models. 
### After_train 
This part is similar to CV part, but it has the observed file together in the step thus can be used to see which demographic model can better explain the result.  
``` sh
python src/Run_Classification.py After_train --test_size 1000 --tolerance 0.01 --ssfile examples/YRI_CEU_CHB.observed.csv --csvout 
``` 
- --test_size 1000 and --tolerance .01 
Same as above for CV 
- ssfile examples/YRI_CEU_CHB.observed.csv 
To define the observed csv file. Here we put YRI_CEU_CHB from the high coverage 1000 genome data sfs file as ss. We have also normalized the sfs thus sum(sfs)==1 as our simulated ss file is same (we found this gives higher accuracy but can be used without normalization in that case use --scale). 
- --csvout 
If you are happy with all the result you can use csvout. This will remove .h5 files to free up space but also will produce csv files which then can directly be used R to further improve the results using abc if necessary. As all the commands of abc is not supported here in ABC-TFK.

This will print out (including the CV part) which models is better explained by the NN. It will also print out the goodness of fit to see if our observed model comes naturally under all the distribution of such model. If you csvout it will additionally output model_index.csv.gz (all the model indexes), ss_predicted.csv.gz (prediction from simulated ss by NN) and ss_target.csv.gz (prediction of the observed or real data). 
### All
In case rather than doing it separately, we can do all these stuffs together in one command.  
``` sh
python src/Run_Classification.py All --test_size 1000 --tolerance 0.01 --ssfile examples/YRI_CEU_CHB.observed.csv --csvout  --demography src/extras/ModelClass.py examples/Model.info 
``` 
It will produce the same files as previously but all of them together. If we do not use --chunksize it will produce x_test.h5 and y_test.h5 (of course if we use csvout it will be deleted) instead of x.h5 y.h5 as it will keep the training part on the ram itself. If you reach ram memory error, please use chunksize which will be relatively slower but do not have any upper limit for the file size.   
### Optional 
We can easily use this result in R further to improve our analysis: 
``` sh
library(abc) 
ss_predict=read.csv('ss_predicted.csv.gz') 
target=read.csv('ss_target.csv.gz') 
index=as.vector(factor(as.matrix(read.table('model_index.csv.gz',header=TRUE)))) 
cv4postpr(as.vector(index),as.matrix(ss_predict[,-1]),nval=1000,method='rejection',tols=c(.01,.1)) 
``` 

To see with different amount of tolerance level and nval how the abc analysis changes. 
## Parameter Estimation
After we have chosen our best model, we try to predict the parameters which can explain the observed results. This will follow similarly as the previous files. But instead of multiple files for different demography it will only use one of the files for parameter estimation (given a model). It has similar structure like Classification: All, Pre_train, Train, CV, After_train. 
### Pre_train
As classification part it will prepare the data for training. 
```sh
python src/Run_ParamsEstimation.py Pre_train examples/Model.info --scale
```
- input: examples/Model.info   
Exactly like classification. But in case the Model.info file has multiple lines; it will only use the first line for the parameter estimation. 
--scale 
If the data is not already scaled it will be scaled. It will be scaled using MinMaxscaler, which means all the number will be with in 0 to 1. Unlike classification, scaling is very important here. Also, we found that theoretical scaling is much better than MinMaxscaler. Thus, if you know how to do the theoretical scaling, it will be preferable to do it before and then use that as an input rather than using the unscaled data. For example, the effective population size parameters should be scaled by NA (Ancestral effective population size). Also, the sfs should be scaled such a way that sum (sfs)==1. 

It will produce x.h5 and y.h5 like previously. The main important different is for y.h5 whereas in classification part it only needed model names, here it will use parameters. It will also save params_header.csv to know the name of the parameters. 
### Train
Same as classification this will train the model. Unlike the classification, it takes a lot of epochs and very convoluted model to get high score for the accuracy. Thus, if you see you do not reach higher accuracy do not be disheartened. 
```sh
python src/Run_ParamsEstimation.py Train --demography src/extras/ModelParams.py --test_size 1000
```
- --demography src/extras/ModelParams.py
Although there is a default method present in ABC-TFK (meaning python src/Run_ParamsEstimation.py Train --test_size 1000 will also work), we can give a model from outside. It must have a def name ANNModelParams. We can decide the number of epochs and other stuff inside that definition. Here we kept 100 to make it faster.  
- --test_size 1000
As above kept 1000 samples for later use. All the other samples are used for training part. 

This will save the model as ModelParamPrediction.h5, which later can be used for prediction and stuff. 
### CV
To calculate cross validation error of the parameters. 
```sh
python src/Run_ParamsEstimation.py CV --test_size 1000 --tolerance .01 --method loclinear
```

- --test_size 1000 --tolerance .01 
Same as clasiification.
--method loclinear
To tell which method to be used for CV. We found that generally loclinear is good for CV and neuralnet for observed data. On top of that if we use ABC-TFK cv method either it will produce the CV independently per column for NN prediction (when using loclinear or rejection) or it will produce CV by all the columns together by NN prediction (when using neuralnet or rejection). 

It will either produce nnparamcv.pdf (for using loclinear), nnparamcv_together.pdf (for using neuralnet) or both (for rejection) and print the cv error table in the terminal. In case of neuralnet it will also produce a correlation matrix for prior and posterior. This is important to see if there are some parameters becoming more correlated than the prior which might be the drawback of the ss, or the neural network used by TF. For example, we found that migrations and split time become correlated in case of SFS. 
### After_train
After everything is done, we can use the After_train to use the ABC analysis. 
```sh
python src/Run_ParamsEstimation.py After_train --test_size 1000 --tolerance .01 --method loclinear --csvout --ssfile examples/YRI_CEU_CHB.observed.csv
```
This will calculate both the CV part as well as will compare with the observed data. This will produce paramposterior.pdf to see the prior vs posterior. It will also produce the same csv file as before but instead of model_index.csv.gz will produce params.csv.gz. Inside those files will be necessary information for the parameters. 
### All
To put all these parts together we can use: 
```sh
python src/Run_ParamsEstimation.py All --demography src/extras/ModelParams.py --test_size 1000 --tolerance .01 --method loclinear --csvout --ssfile examples/YRI_CEU_CHB.observed.csv --scale examples/Model.info
```
It will produce similar result. 
### Optional 
We can use further the results in R:
```sh
library(abc)
params=read.csv('params.csv.gz')
ss=read.csv('ss_predicted.csv.gz')
target=read.csv('ss_target.csv.gz')
res=abc(target = target,param=params,sumstat = ss,tol=.01,method='neuralnet',transf = 'log')
summary(res)
plot(res,param=params)
```
This will transform the parameter values in log scale. Thus, we can calculate the distance much more precisely. 





