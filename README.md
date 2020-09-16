# ABC-TFK (Approximate Bayesian Computation using TensorFlow Keras backended)
This is a python pipeline where you can simply use summary statistics created from pop genome simulation files (ms, msprime, FastSimcoal etc.) to predict which underlying model can better explain the observed results, as well as which parameters can produce such results. The whole method is written in python and thus easy to read as well as can be accessed completely through command line thus knowing python is not needed. Although it will be helpful to know python as well as R as some of the packages here used is based on those languages.  

## Getting Started
To download the package either click the download button in <https://github.com/mayukhmondal/ABC-TFK> or use:
```bash
git clone https://github.com/mayukhmondal/ABC-TFK
cd ABC-TFK
```
The codes are written in python3 (>=python3.6.9). This programme comes with several dependencies:

- numpy
- scikit-learn
- joblib
- pandas
- h5py
- rpy2
- r-essentials
- tzlocal
- tensorflow=1
- keras

The easiest way to install all of these dependencies is using conda. To install conda please visit <https://www.anaconda.com/distribution/>. After installing conda (remember to install python 3), use:
```shell script
conda install --file requirements.txt
```
or for last tested version 
```shell script
conda env update -f requirements.yml
```
or if you want to make a different environment:
```shell script
conda create --name ABC-TFK --file requirements.txt  python=3
conda activate ABC-TFK
```
same but for last tested version
```shell script
conda env create -f requirements.yml --name ABC-TFK
conda activate ABC-TFK
```
Wait a little bit as it can take a long time to install all the dependencies.  
After installing all the dependencies you can just run either
```
python src/Run_Classification.py --help
```
or 
```
python src/Run_ParamsEstimation.py --help
```

For Model selection and for parameters estimation respectively. The first time you run it will also try to install abc from r package manager (automatically).   
Please see [examples/Examples.md](examples/Examples.md) for a detailed guide how to use the codes.
Right now this code are written and checked in linux system. I can not guarantee it will work on other system but you are welcome to try. 
### Installation Issues 
In case you try to install it to already existed conda environment which has already R (r-base), it can conflict with the rpy2 when it tries to automatically download abc package from R saying abc package do not exist. In that case create a new environement. 
## Contact 
The code is maintained by Dr. Mayukh Mondal. In case you need further assistance please contact <mondal.mayukh@gmail.com>
