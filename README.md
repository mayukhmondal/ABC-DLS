# ABC-DLS (Approximate Bayesian Computation with Deep Learning and Sequential monte carlo)
This software is a python pipeline where you can use simulated summary statistics to predict which underlying model can better explain the observed or real-world results (classification) and can guess which parameters can produce such results (parameter estimation) with the help of Approximate Bayesian Computation (ABC), Deep Learning (using Tensorflow Keras Backend) and Sequential Monte Carlo (SMC).
The whole method is written in python and easy to read and can be accessed entirely through the command line, therefore knowing python is not needed. Although it will be helpful to know python and R as some of the packages here used are based on those languages.  

## Getting Started
To download the package, either click the download button in <https://github.com/mayukhmondal/ABC-DLS> or use:
```bash
git clone https://github.com/mayukhmondal/ABC-DLS
cd ABC-DLS
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
- tensorflow
- keras

The easiest way to install all of these dependencies is using conda. To install conda please visit [anaconda](https://www.anaconda.com/distribution/). After installing conda (remember to install python 3), use:
```shell script
conda install --file requirements.txt
```
or for last tested version 
```shell script
conda env update -f requirements.yml
```
or if you want to make a different environment:
```shell script
conda create --name ABC-DLS --file requirements.txt  python=3
conda activate ABC-DLS
```
same but for last tested version. Use it in case you getting conflict between package versions. 
```shell script
conda env create -f requirements.yml --name ABC-DLS
conda activate ABC-DLS
```
Please wait a little bit as it can take a long time to install all the dependencies.  After installing all the dependencies, you can just run either
```
python src/Run_Classification.py --help
```
or 
```
python src/Run_ParamsEstimation.py --help
```
or 
```
python src/Run_SMC.py --help
```
For Model selection, parameters estimation and parameter estimation using SMC, respectively. The first time you run, it will also try to install abc from r package manager (automatically). Please see [examples/Examples.md](examples/Examples.md) for a detailed guide how to use the codes. Right now, this code is written and checked in the linux system. I can not guarantee it will work on other systems, but you are welcome to try. 
### Installation Issues 
In case you try to install it to an already existed conda environment that already has R (r-base), it can conflict with the rpy2 when it tries to automatically download an abc package from R saying abc package does not exist. In that case, create a new environment. 
## Citation

### Revisiting the out of Africa event with a deep learning approach  
Francesco Montinaro, Vasili Pankratov, Burak Yelmen, Luca Pagani, Mayukh Mondal  
The American Journal of Human Genetics; doi: https://doi.org/10.1016/j.ajhg.2021.09.006  

If you use cSFS please cite:
### Resolving out of Africa event for Papua New Guinean population using neural network
Mayukh Mondal, Mathilde Andre, Ajai K. Pathak, Nicolas Brucato, Francois-Xavier Ricaut, Mait Metspalu, Anders Eriksson   
bioRxiv 2024.09.19.613861; doi: https://doi.org/10.1101/2024.09.19.613861
## Contact 
The code is maintained by Dr. Mayukh Mondal. In case you need further assistance please contact 
<mondal.mayukh@gmail.com>
