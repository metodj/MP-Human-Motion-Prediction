## Human Motion Prediction

Machine Perception, Spring 2019 ETH Zürich.

Project authors are Metod Jazbec and Rok Šikonja.

This README file contains information on replication of results presented in the paper. It should guide the reader through
necessary pre-processing and later training and evaluation steps. In addition, it describes the modeling framework, which
is designed to build a family of human motion prediction models. 

### Data

Let's assume that provided human motion dataset is stored under the directory ```./data/```, which contains training,
validation and test poses in child directories ```training/```, ```validation/``` and ```test/```. If running on Leonhard
Cluster, then this directory should be replaced by ```/cluster/project/infk/hilliges/lectures/mp19/project4 ```.


### Data Preprocessing

Originally, human body is represented as a set of 15 three-dimensional rotation matrices, flattened into a one-dimensional
vector. Rotations can be expressed also in the form of angle-axis representation. In this regard, data should be pre-processed
beforehand by running the following command, which will create a new directory ```./data_angles/``` containing human body
represented as angle-axis representations.

```
python precompute_angles.py 
--read_dir ./data/
--write_dir ./data_angles/
```

Standardization requires statistics to be known. Statistics file for angle-axis representation is obtained by running the
following script, which creates a new file```./data_angles/training/stats.npz```.

```
python angles_stats.py
--read_dir ./data_angles/training/
```




## Model Parameters






Code was adopted from Manuel Kaufmann, Emre Aksan.
