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

Default parameters.

```
python train.py
--data_dir ./data/  # or ./data_angles/
--save_dir ./experiments/  # constant
--experiment_name None

--seq_length_in 120  # constant
--seq_length_out 24  # constant


--learning_rate 0.001
--batch_size 16
--num_epochs 5
--print_every 100
--test_every 200
--use_cpu False

--to_angles False  # should be used together with ./data_angles/ for angle-axis representation 
--stand False  # enable standardization of features

--model_type seq2seq


```

parser.add_argument("--cell_type", type=str, default="lstm", help="RNN cell type: lstm, gru")
parser.add_argument("--cell_size", type=int, default=256, help="RNN cell size.")
parser.add_argument("--cell_size_disc", type=int, default=256, help="RNN cell size.")
parser.add_argument("--input_hidden_size", type=int, default=None, help="Input dense layer before the recurrent cell.")
parser.add_argument("--activation_fn", type=str, default=None, help="Activation Function on the output.")
parser.add_argument("--activation_input", type=str, default=None, help="input layer activation")

#seq2seq
parser.add_argument("--residuals", action="store_true", help="Use of residuals in the decoder part of seq2seq model.")
parser.add_argument("--optimizer", type=str, default="Adam", help="optimizer: Adam or SGD")
parser.add_argument("--loss", type=str, default="geo", help="mean squared error (mse) or geodesic (geo) loss")
parser.add_argument("--samp_loss", action="store_true", help="sampling loss: rnn output from previous is feed to input")
parser.add_argument("--num_rnn_layers", type=int, default=1, help="depth of rnn layer")

parser.add_argument("--log", action="store_true", help="create log file")
parser.add_argument("--fidelity", action="store_true", help="fidelity discriminator")
parser.add_argument("--continuity", action="store_true", help="continuity discriminator")
parser.add_argument("--lambda_", type=float, default=0.6, help="regularization parameter for discriminators")
parser.add_argument("--update_ckpt", action="store_true", help="Only store model if eval loss was improved during current epoch.")
parser.add_argument("--weight_sharing", type=str, default="w/o", help="other options: seq2seq only (s2s), all (all)")
parser.add_argument("--weight_sharing_rnn", action="store_true", help="Rnn weight sharing.")
parser.add_argument("--epsilon", type=float, default="0.00000001", help="epsilon param for Adam optimizer")
parser.add_argument("--dropout", type=float, default=None, help="Dropout rate for rnn cells.")
parser.add_argument("--dropout_lin", type=float, default=None, help="Dropout rate for linear layers.")
parser.add_argument("--exp_decay", type=float, default=None, help="Decay rate.")
parser.add_argument("--bi", action="store_true", help="Use bidirectional encoder.")
parser.add_argument("--l2", type=float, default=0.0, help="l2 regularization parameter")



Code was adopted from Manuel Kaufmann, Emre Aksan.
