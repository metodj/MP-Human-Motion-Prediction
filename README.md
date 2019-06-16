## Human Motion Prediction

Machine Perception, Spring 2019 ETH Zürich.

Project authors are Metod Jazbec and Rok Šikonja.

This README file contains information on replication of results presented in the paper. It should guide the reader through
necessary pre-processing and later training and evaluation steps. In addition, it describes the modeling framework, which
is designed to build a family of human motion prediction models. 

#### Data

Let's assume that provided human motion dataset is stored under the directory ```./data/```, which contains training,
validation and test poses in child directories ```training/```, ```validation/``` and ```test/```. If running on Leonhard
Cluster, then this directory should be replaced by ```/cluster/project/infk/hilliges/lectures/mp19/project4 ```.


#### Data Preprocessing

Originally, human body is represented as a set of 15 three-dimensional rotation matrices, flattened into a one-dimensional
vector. Rotations can be expressed also in the form of angle-axis representation (in the papers also referred to as exponential-map representation). In this regard, data should be pre-processed
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

#### Model Parameters

Default parameters.

```
python train.py
--use_cpu False
--log False  # create log file

--data_dir ./data/  # or ./data_angles/
--save_dir ./experiments/  # constant
--experiment_name None

--seq_length_in 120  # constant
--seq_length_out 24  # constant

--learning_rate 0.001
--batch_size 16
--num_epochs 5
--optimizer Adam  # loss minimizer
--print_every 100
--test_every 200
--loss geo  # loss function: geodesic or mean square error

--to_angles False  # should be used together with ./data_angles/ for angle-axis representation 
--stand False  # enable standardization of features

--model_type seq2seq  # seq2seq or zero_velocity
--cell_type lstm # RNN cell type
--cell_size 256 # RNN hidden size in s2s
--cell_size_disc 256 # RNN hidden size in GAN discriminators
--input_hidden_size None  # input dense layer size
--activation_input None  # input dense layer activation function
--activation_fn None  # output dense layer activation fuction
--residuals False  # enable residual connection
--samp_loss False  # enable sampling loss
--num_rnn_layers 1  # number rnn layers in seq2seq
--fidelity False  # enable fidelity discriminator, should be used together with --continuity
--continuity False  # enable continuity discriminator, should be used together with --fidelity
--lambda 0.6  # weight for discriminator loss w.r.t. predictor loss

--update_ckpt False  # only store model if eval loss was improved during current epoch
--weight_sharing w/o  # weight sharing between input dense layers, options: w/o (without), s2s (encoder, decoder), all (encoder, decoder, discriminators)
--weight_sharing_rnn False  # weight sharing between encoder's and decoder's rnn cells
--bi False  # enable bidirectional encoder
--epsilon 0.00000001  # epsilon parameter for Adam optimizer
--dropout None  # dropout rate for RNN cells
--dropout_lin None  # dropout rate for dense layers
--l2 0.0  # l2 regularization parameter
--exp_decay None  # learning rate decay
```

### Results

#### Our Model

On Leonhard: bsub -n 6 -W 4:00 -R "rusage[mem=2048, ngpus_excl_p=1]" python train.py --data_dir /cluster/project/infk/hilliges/lectures/mp19/project4 --save_dir ./experiments --experiment_name seq2seq_gru_multi --model_type seq2seq --batch_size 100 --log --cell_type gru --num_rnn_layers 4 --input_hidden_size 512 --cell_size 512 --residuals --loss geo --num_epochs 150 --learning_rate 0.0005 --dropout 0.3 --exp_decay 0.96 --stand --weight_sharing_rnn --weight_sharing s2s --update_ckpt

4-layer GRU with shared weights, input and RNN, between encoder and decoder.  

```
python train.py 
--log
--data_dir ./data/
--save_dir ./experiments 
--experiment_name seq2seq_gru_multi 
--model_type seq2seq

--stand 
--batch_size 100 

--cell_type gru 
--num_rnn_layers 4 
--input_hidden_size 512 
--cell_size 512 
--residuals 
--loss geo 
--num_epochs 150 
--learning_rate 0.0005 
--dropout 0.3 
--exp_decay 0.96  
--weight_sharing_rnn 
--weight_sharing s2s 
--update_ckpt
```




#### On Human Motion Prediction using Recurrent Neural Networks

Replication of model from paper: seq2seq with residual connections and sampling loss.

```
python train.py
--data_dir ./data_angles/
--save_dir ./experiments/ 
--experiment_name martinez

--loss mse
--learning_rate 0.005
--batch_size 16
--num_epochs 50

--to_angles
--stand

--model_type seq2seq 
--cell_type gru
--cell_size  1024
--input_hidden_size None
--residuals
--samp_loss
--weight_sharing_rnn
```

#### Adversarial Geometry-Aware Human Motion Prediction

Replication of model from paper: seq2seq with geodesic loss, residual connections, sampling loss and fidelity and 
continuity discriminators.
```
python train.py
--data_dir ./data_angles/
--save_dir ./experiments/ 
--experiment_name aged

--loss geo
--learning_rate 0.005
--batch_size 16
--num_epochs 50

--to_angles
--stand

--model_type seq2seq 
--cell_type gru
--cell_size  1024
--input_hidden_size 1024
--residuals
--samp_loss
--weight_sharing w/o

--fidelity
--continuity
--lambda_ 0.6
```

####

bsub -n 6 -W 4:00 -R "rusage[mem=2048, ngpus_excl_p=1]" python train.py --data_dir /cluster/project/infk/hilliges/lectures/mp19/project4 --save_dir ./experiments --experiment_name seq2seq_karolis --model_type seq2seq --batch_size 100 --log --cell_type gru --num_rnn_layers 4 --input_hidden_size 512 --cell_size 512 --residuals --loss geo --num_epochs 150 --learning_rate 0.0005 --dropout 0.3 --exp_decay 0.96 --stand --weight_sharing_rnn --weight_sharing s2s --update_ckpt

bsub -n 6 -W 4:00 -R "rusage[mem=2048, ngpus_excl_p=1]" python train.py --data_dir /cluster/project/infk/hilliges/lectures/mp19/project4 --save_dir ./experiments --experiment_name seq2seq_karolis --model_type seq2seq --batch_size 100 --log --cell_type gru --num_rnn_layers 4 --input_hidden_size 512 --cell_size 512 --residuals --loss geo --num_epochs 150 --learning_rate 0.0005 --dropout 0.3 --exp_decay 0.96 --stand --weight_sharing_rnn --weight_sharing s2s --update_ckpt



bsub -n 6 -W 4:00 -R "rusage[mem=2048, ngpus_excl_p=1]" python train.py --data_dir /cluster/project/infk/hilliges/lectures/mp19/project4 --save_dir ./experiments --experiment_name seqseq --model_type seq2seq --log --cell_type lstm --input_hidden_size 1024 --cell_size 1024 --residuals --num_epochs 150 --learning_rate 0.0005 --weight_sharing s2s --weight_sharing_rnn --update_ckpt --stand --batch_size 100 --dropout 0.3 --exp_decay 0.96


Code was adopted from Manuel Kaufmann, Emre Aksan of AIT Lab, ETH Zurich.
