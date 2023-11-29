# FFA

This repository contains the implementation used in the TMLR submission. The implementation aims at generating exact and approximate feature attribution in gradient boosted tress (BTs) based on formal explanation enumeration by applying apparatus of formal explainable AI (XAI). Formal feature attribution (FFA) is argued to be advantageous over the existing methods, both formal and non-formal. 


## Instruction <a name="instrt"></a>
Before using the implementation, we need to extract the datasets stored in ```datasets.tar.xz```. To extract the datasets, please ensure ```tar``` is installed and run:
```
$ tar -xvf datasets.tar.xz
```

## Table of Content
* **[Required Packages](#require)** 
* **[Usage](#usage)**
	* [Example Tutorial](#tut)
	* [Prepare a Dataset](#prepare)
	* [Generate a Boosted Tree](#bt)
	* [Enumerate Abductive Explanations (AXp's) as Dual Explanations](#enum)
* **[Reproducing Experimental Results](#expr)**

## Required Packages <a name="require"></a>
The implementation is written as a set of Python scripts. The python version used in the experiments is 3.8.5. Some packages are required. To install requirements:
```
$ pip install -r requirements.txt
```

## Usage <a name="usage"></a>

### Example Tutorial <a name="tut"></a> 
For example usage, take a look at the following tutorial (generated from ipython notebooks):
- [src/example.ipynb](src/example.ipynb)

Alternatively, we can follow the steps below:

### Preparing a dataset <a name="prepare"></a>  <a name="prepare"></a>
First, change to the source directory
```
$ cd src/
```

`FFA` can address datasets in the CSV format. Before enumerating abductive explanations (AXp's) and generating feature attribution, we need to prepare the datasets the train a BT model.

1. Assume a target dataset is stored in ```somepath/dataset.csv```
2. Create an extra file named ```somepath/dataset.csv.catcol``` containing the indices of the categorical columns ofthe target dataset. For example, if columns ```0```, ```3```, and ```6``` are categorical features, the file should be as follow:
	```
	0
	3
	6
	```
3. With the two files above, we can run:
```
$ python explain.py -p --pfiles dataset.csv,somename somepath/
```
to create a new dataset file `somepath/somename_data.csv` with the categorical features properly addressed. For example:
```
$ python explain.py -p --pfiles compas_train_data.csv,compas_train_data ../datasets/tabular/train/compas/
```

### Training a gradient boosted tree model  <a name="bt"></a>
A gradient boosted tree model is required before generating a decision set. Run the following command to train a BT model:
```
$ python ./explain.py -o ./btmodels/compas/ -c --testsplit 0 -t -n 25 -d 3 ../datasets/tabular/train/compas/compas_train_data.csv
```
Here, a boosted tree consisting of 25 trees per class is trained, where the maximum depth of each tree is 3. ``` ../datasets/tabular/train/compas/compas_train_data.csv
 ``` is the dataset to be trained. The value of ```--testsplit``` ranges from 0.0 to 1.0. In this command line, the given dataset is split into 100% to train and 0% to test. ```./btmodels/compas/``` is the output path to store the trained model. In this example, the generated model is saved in ```./btmodels/compas/compas_train_data/compas_train_data_nbestim_25_maxdepth_3_testsplit_0.0.mod.pkl```


### Enumerating Abductive Explanations (AXp's) as Dual Explanations  <a name="enum"></a>
To enumerate abductive or contrastive explanations for BTs, run:
```
$ python -u ./explain.py -e mx --am1 -E -T 1 -z -vvv -c --xtype <string> -R lin --sort abs --explain_ formal --xnum all -M --cut <int> --explains <dataset.csv> <model.pkl> 

```
Here, parameter ```--cut``` is optional, where the value of ```--cut``` indicate the instance index to enumeration explanations. By default, all instances in the dataset are considered. ```<dataset.csv>``` and ```<model.pkl>``` specify the dataset and BT model.

For example:

```
$ python -u ./explain.py -e mx --am1 -E -T 1 -z -vvv -c --xtype con -R lin --sort abs --explain_ formal --xnum all -M --cut 5 --explains ../datasets/tabular/test/compas/compas_test_data.csv ./btmodels/compas/compas_train_data/compas_train_data_nbestim_25_maxdepth_3_testsplit_0.0.mod.pkl 
```

The command above will enumerate AXp's as dual explanations for *compas* dataset.


## Reproducing  Experimental Results <a name="expr"></a>
Due to randomization used in the sampling process in LIME and KernelSHAP, it seems unlikely that the experimental results reported in the submission can be completely reproduced.
Similar experimental results can be obtained by the following script:

```
$ cd ./src/; ./experiment/repro_exp.sh
```

Since the total number of datasets and instances considered is large, running the experiments will take a while.
