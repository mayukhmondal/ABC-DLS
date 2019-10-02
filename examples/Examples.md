# ABC-TFK
ABC-TFK is a python pipeline where you can simply use summary statistics created from pop genome simulation files (ms, msprime, FastSimcoal etc.) to predict which underlying model can better explain the observed results, as well as which parameters can produce such results. The whole method is written in python and thus easy to read as well as can be accessed completely through command line thus knowing python is not needed. Although it will be helpful to know python as well as R as some of the packages here used is based on those languages.
## Classification and Model Selection
The first part is to chose the best model which can explain the observed data (in this case real sequenced data). This programme only accepts csv files (can be zipped) as an input. Thus simulations (i.e. ms, msprime, FastSimcoal etc) should be done elsewhere to produce summary statistics (ss) csv files. One example of ss can be Site Frequency Spectrum [SFS] but can anything in principle which can be represented in a single row with similar number of columns regardless of paramteres or demography. The csv files should be written like this:
- Every row denote one simulation under this model 
- First few columns should be the parameters that created the summary statistics. 
- Everything else should be as the ss which would be used by TensorFlow (TF) to differentitate between models or predict back the parameters.
- The files should have a header. 
You can look inside the examples/*.csv.gz files to get an idea.
```sh
N_AF,N_EU,N_AS,N_B,T_AF,T_B,T_EU_AS,0_0_0,0_0_1,0_0_2,...
51345.18223692785,12088.175310709128,57003.53191147518,2836.3397488400688,19919.590697189542,6745.785581877058,2044.914414636497,0.0,0.05763950461706744,0.011561403720162306,...
44461.927156180536,19591.694811174035,50700.19845184081,2642.3892773045395,2718.7327121234107,4662.7293230893265,2379.931022189205,0.0,0.08125373412395043,0.015558448274754432,...
51431.42563665978,36277.816701087,64579.202181147644,1245.2241624937133,2949.691221614999,2733.2034297360074,1822.4184924786944,0.0,0.06643583266923958,0.010742217176571226,...
33688.093216904344,25472.873973886602,30435.727894210475,1464.36792729051,11392.273637177412,6731.070839338572,2388.1057949481874,0.0,0.05962954644324739,0.011929341763292071,...
54183.559035734295,73182.44963451767,65784.87589486384,5368.394693791342,23710.073995546874,5807.500359988167,1869.7751657298722,0.0,0.06323807875620768,0.013285449995686458,...
33768.001079286056,63595.15247825567,21653.347671160893,4839.534165227084,21733.51348118516,2137.418318876772,2374.811025752982,0.0,0.07321064169101178,0.020707344946100742,...
```
In principle you donot need parameter columns for the classification part but we kept it to make it similar to later part where parameters are required.
```
python  src/Run_Classification.py --help
```
Will out put 5 different methods:
| Methods | Helps |
| ------ | ------ |
| All| The whole run of the NN for paramter estimation from first to last |
| Pre_train | To prepare the data for training ANN |
| Train | The trainging part of the ANN. Should be done after Pre_train part |
| CV | After the training only to get the result of cross validation test. Good for unavailable real data |
| After_train | This is to run the ABC analysis after the traingin part is done |

### Pre_train 
This is the pre training part. Where the data is prepared for the TF. 
```
python src/Run_Classification.py Pre_train examples/Model.info 
```
- input: examples/Model.info 
Can be any text file which has all the demographic models that we want to compare together. Every line is denoted for one demographic simulation csv file. We should also denote the number of columns which are for the paramters present in that file. This will remove those columns from the files as they are not ss for the comparison. 
Should looke like:
> <Model1.csv.gz> <param_n>
> <Model2.csv.gz> <param_n>
>...

This will create in total 3 files. x.h5 (this is for ss), y.h5 (models names in integer format) and y_cat_dict.txt (models name and their corresponsding integer number). These first two part will be needed to run the neural network training for TF. The last part will be used later.

### Train
The next part is to run the training part by neural network. This will train the network to differentitate between models. 
```
python src/Run_Classification.py Train --demography src/extras/ModelClass.py --test_size 1000
```
This will train the model. 
-  --demography src/extras/ModelClass.py
ABC-TFK has a default training neural model. But it is impossible to predict which model should be better for the input data. Thus we can define custom made model cater to our own need. One such example is src/extras/ModelClass.py. Here we put very less epochs (only 20) to get faster result. More is merrier of course. 
 - --test_size 1000 
This will keep the last 1000 line save for test data set and will be used for ABC analysis. In the previous step it is already shuffled the data thus last 1000 lines you would expect near equal number of demographies. 

This will save the neural model as ModelClassification.h5 which we can use later for other use. 

### CV







