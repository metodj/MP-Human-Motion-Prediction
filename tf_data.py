"""Copyright (c) 2019 AIT Lab, ETH Zurich, Manuel Kaufmann, Emre Aksan

Students and holders of copies of this code, accompanying datasets,
and documentation, are not allowed to copy, distribute or modify
any of the mentioned materials beyond the scope and duration of the
Machine Perception course projects.

That is, no partial/full copy nor modification of this code and
accompanying data should be made publicly or privately available to
current/future students or other parties.
"""
import tensorflow as tf
import numpy as np
import os
import functools

from utils import rotmat_to_euler
from constants import Constants as C


class Dataset(object):
    """
    A wrapper class around tf.data.Dataset API.
    """

    def __init__(self, data_path, meta_data_path, batch_size, shuffle, **kwargs):
        self.tf_data = None
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Load statistics stored in the meta-data file.
        self.meta_data = self.load_meta_data(meta_data_path)

        # A scalar mean and standard deviation computed over the entire training set
        self.mean_all = self.meta_data['mean_all']
        self.var_all = self.meta_data['var_all']

        # A scalar mean and standard deviation per degree of freedom computed over the entire training set
        self.mean_channel = self.meta_data['mean_channel']
        self.var_channel = self.meta_data['var_channel']

        # Do some preprocessing.
        self.tf_data_transformations()
        self.tf_data_to_model()

        # Intialize iterator that loops over the data.
        self.iterator = self.tf_data.make_initializable_iterator()
        self.tf_samples = self.iterator.get_next()

    def load_meta_data(self, meta_data_path):
        """
        Loads *.npz meta-data file given the path.
        Args:
            meta_data_path: Path to the meta-data file.
        Returns:
            Meta-data dictionary or False if it is not found.
        """
        if not meta_data_path or not os.path.exists(meta_data_path):
            print("Meta-data not found.")
            return False
        else:
            return np.load(meta_data_path, allow_pickle=True)['stats'].tolist()

    def tf_data_transformations(self):
        """Loads the raw data and applies some pre-processing."""
        raise NotImplementedError('Subclass must override this method.')

    def tf_data_to_model(self):
        """Converts the data into the format that a model expects. Creates input, target, sequence_length, etc."""
        raise NotImplementedError('Subclass must override this method.')

    def get_iterator(self):
        return self.iterator

    def get_tf_samples(self):
        return self.tf_samples


class TFRecordMotionDataset(Dataset):
    """
    Dataset class for motion samples stored as TFRecord files.
    """
    def __init__(self, data_path, meta_data_path, batch_size, shuffle, **kwargs):
        # Size of windows to be extracted.
        self.extract_windows_of = kwargs.get("extract_windows_of", 0)

        # Whether to extract windows randomly or from the beginning of the sequence.
        self.extract_random_windows = kwargs.get("extract_random_windows", True)

        # If the sequence is shorter than this, it will be ignored.
        self.length_threshold = kwargs.get("length_threshold", self.extract_windows_of)

        # Number of parallel threads accessing the data.
        self.num_parallel_calls = kwargs.get("num_parallel_calls", 16)

        self.to_angles = kwargs.get("to_angles", False)

        super(TFRecordMotionDataset, self).__init__(data_path, meta_data_path, batch_size, shuffle, **kwargs)

    def tf_data_transformations(self):
        """
        Loads the raw data and applies some preprocessing.
        """
        tf_data_opt = tf.data.Options()
        tf_data_opt.experimental_autotune = True

        # Gather all tfrecord filenames and load them in parallel.
        self.tf_data = tf.data.TFRecordDataset.list_files(self.data_path, seed=C.SEED, shuffle=self.shuffle)
        self.tf_data = self.tf_data.with_options(tf_data_opt)
        self.tf_data = self.tf_data.apply(
            tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset,
                                                     cycle_length=self.num_parallel_calls, block_length=1,
                                                     sloppy=self.shuffle))

        # Function that maps the tfrecords to a dictionary
        self.tf_data = self.tf_data.map(
            functools.partial(self._parse_single_tfexample_fn), num_parallel_calls=self.num_parallel_calls)

        # Makes everything faster.
        self.tf_data = self.tf_data.prefetch(self.batch_size*10)

        # Maybe shuffle.
        if self.shuffle:
            self.tf_data = self.tf_data.shuffle(self.batch_size*10)

        # Preprocessing to euler angles
        if self.to_angles:
            self.tf_data = self.tf_data.map(functools.partial(self._my_own_preprocessing),
                                            num_parallel_calls=self.num_parallel_calls)

        # Maybe extract windows
        if self.extract_windows_of > 0:
            # Make sure input pose is at least as big as the requested window.
            self.tf_data = self.tf_data.filter(functools.partial(self._pp_filter))
            if self.extract_random_windows:
                # Extract a random window from somewhere in the sequence. Useful for training.
                self.tf_data = self.tf_data.map(functools.partial(self._pp_get_windows_randomly),
                                                num_parallel_calls=self.num_parallel_calls)
            else:
                # Extract one window from the beginning of the sequence. Useful for validation and test.
                self.tf_data = self.tf_data.map(functools.partial(self._pp_get_windows_from_beginning),
                                                num_parallel_calls=self.num_parallel_calls)

        # Set the feature size explicitly, otherwise it will be unknown in the model class.
        self.tf_data = self.tf_data.map(functools.partial(self._pp_set_feature_size),
                                        num_parallel_calls=self.num_parallel_calls)

        # If you want to do some pre-processing on the extracted windows, here is the place to do it.

    def tf_data_to_model(self):
        """Converts the data into the format that a model expects. Creates input, target, sequence_length, etc."""
        # Convert to model input format using our custom function.
        self.tf_data = self.tf_data.map(functools.partial(self._to_model_inputs),
                                        num_parallel_calls=self.num_parallel_calls)

        # Pad the sequences if necessary.
        self.tf_data = self.tf_data.padded_batch(self.batch_size, padded_shapes=self.tf_data.output_shapes)

        # Speedup.
        self.tf_data = self.tf_data.prefetch(2)
        # UNCOMMENT when running on Leonhard
        self.tf_data = self.tf_data.apply(tf.data.experimental.prefetch_to_device('/device:GPU:0'))

    def _pp_filter(self, sample):
        """Filter out samples that are smaller then the required window size."""
        return tf.shape(sample["poses"])[0] >= self.length_threshold

    def _pp_get_windows_randomly(self, sample):
        """Extract a random window from somewhere in the sequence."""
        start = tf.random_uniform((1, 1), minval=0, maxval=tf.shape(sample["poses"])[0]-self.extract_windows_of+1,
                                  dtype=tf.int32)[0][0]
        end = tf.minimum(start+self.extract_windows_of, tf.shape(sample["poses"])[0])
        sample["poses"] = sample["poses"][start:end, :]
        sample["shape"] = tf.shape(sample["poses"])
        return sample

    def _pp_get_windows_from_beginning(self, sample):
        """Extract window from the beginning of the sequence."""
        sample["poses"] = sample["poses"][0:self.extract_windows_of, :]
        sample["shape"] = tf.shape(sample["poses"])
        return sample

    def _pp_set_feature_size(self, sample):
        """Set the shape of the poses explicitly."""
        # This is required as otherwise the last dimension of the batch is unknown, which is a problem for the model.
        seq_len = sample["poses"].get_shape().as_list()[0]
        sample["poses"].set_shape([seq_len, self.mean_channel.shape[0]])
        return sample

    @staticmethod
    def _parse_single_tfexample_fn(proto):
        """
        Transforms a sample read from a tfrecord file into a dictionary.
        Args:
            proto: The input sample read from a tfrecord file.

        Returns:
            A dictionary with keys 'poses' and 'file_id' containing the respective data.
        """
        feature_to_type = {
            "file_id": tf.FixedLenFeature([], dtype=tf.string),
            "shape": tf.FixedLenFeature([2], dtype=tf.int64),
            "poses": tf.VarLenFeature(dtype=tf.float32),
        }

        # Reshape the flattened poses to their original shape.
        parsed_features = tf.parse_single_example(proto, feature_to_type)
        parsed_features["poses"] = tf.reshape(tf.sparse.to_dense(parsed_features["poses"]), parsed_features["shape"])

        return parsed_features

    @staticmethod
    def _to_model_inputs(tf_sample_dict):
        """
        Transforms a TFRecord sample into a more general sample representation where we use global keys to represent
        the required fields by the models.
        Args:
            tf_sample_dict: The sample as loaded from the tfrecord files, as a dictionary.
        Returns: A dictionary that is more compatible with what the models expect as input.
        """
        model_sample = dict()
        model_sample[C.BATCH_SEQ_LEN] = tf_sample_dict["shape"][0]
        model_sample[C.BATCH_INPUT] = tf_sample_dict["poses"]
        model_sample[C.BATCH_TARGET] = tf_sample_dict["poses"]
        model_sample[C.BATCH_ID] = tf_sample_dict["file_id"]
        return model_sample

    @staticmethod
    def _my_own_preprocessing(tf_sample_dict):
        """
        Placeholder for custom pre-processing.
        Args:
            tf_sample_dict: The dictionary returned by `_parse_single_tfexample_fn`.

        Returns:
            The same dictionary, but pre-processed.
        """
        def _my_np_func(p):
            """
            Args: r # (num_poses, 135)
            """
            p = np.reshape(p, newshape=(-1, 15, 3, 3))
            p = np.reshape(p, newshape=(-1, 3, 3))

            a = np.zeros(shape=(p.shape[0], 3))

            for i in range(p.shape[0]):
                r = p[i, :, :]
                theta = rotmat_to_euler(r)
                a[i, :] = theta

            a = np.reshape(a, newshape=(-1, 15*3*3))
            return a

        # A useful function provided by TensorFlow is `tf.py_func`. It wraps python functions so that they can
        # be used inside TensorFlow. This means, you can program something in numpy and then use it as a node
        # in the computational graph.
        processed = tf.py_func(_my_np_func, [tf_sample_dict["poses"]], tf.float32)

        # Set the shape on the output of `py_func` again explicitly, otherwise some functions might complain later on.
        processed.set_shape([None, 135])

        # Update the sample dict and return it.
        model_sample = tf_sample_dict
        model_sample["poses"] = processed
        model_sample["shape"] = tf.shape(processed)
        return model_sample
