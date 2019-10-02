# ABC-TFK (Approximate Bayesian Computation using TensorFlow Keras backended)
This is a python pipeline where you can simply use summary statistics created from pop genome simulation files (ms, msprime, FastSimcoal etc.) to predict which underlying model can better explain the observed results, as well as which parameters can produce such results. The whole method is written in python and thus easy to read as well as can be accessed completely through command line thus knowing python is not needed. Although it will be helpful to know python as well as R as some of the packages here used is based on those languages.  

## Getting Started
The codes are written in python3 (>=python3.6.9). This programme comes with several dependencies:

- numpy
- sklearn
- pandas
- h5py
- rpy2
- tensorflow
- keras 

The easiest way to install this dependencies is using conda. 
```
conda install --file requirements.txt
```
Wait a little bit as it can take long time to install all the dependencies. 
After installing all the dependencies you can just run either
```
python src/Run_Classification.py --help
```
or 
```
python src/Run_ParamsEstimation.py --help
```

For Model selection and for parameters estimation respectively. 
For more information please see examples/Guide.pdf

The first time you run it will also try to install abc from r package manager. 

## Contact 
The code is maintained by Mayukh Mondal. In case you need further assistance please contact <mondal.mayukh@gmail.com>
