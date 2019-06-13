import tensorflow as tf
import functools
import numpy as np
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--read_dir', required=True, default='C:/Users/roksi/data/', help='Where the tfrecords are stored.')
ARGS = parser.parse_args()


def read_tfrecords(tfrecords_path):
    """Read tfrecord file.

        Args:
            tfrecords_path: file path

        Returns:
            list of 2D numpy arrays (nr_frames, 45)
    """

    def _parse_tf_example(proto):
        feature_to_type = {
            "file_id": tf.FixedLenFeature([], dtype=tf.string),
            "shape": tf.FixedLenFeature([2], dtype=tf.int64),
            "poses": tf.VarLenFeature(dtype=tf.float32),
        }

        parsed_features = tf.parse_single_example(proto, feature_to_type)
        parsed_features["poses"] = tf.reshape(tf.sparse.to_dense(parsed_features["poses"]), parsed_features["shape"])
        return parsed_features

    tf_data = tf.data.TFRecordDataset.list_files(tfrecords_path)
    tf_data = tf_data.interleave(tf.data.TFRecordDataset, cycle_length=4, block_length=1)
    tf_data = tf_data.map(functools.partial(_parse_tf_example), 4)

    iterator = tf_data.make_one_shot_iterator()
    samples = []
    for s in iterator:
        tmp = s["poses"].numpy()
        samples.append(tmp)

    print(tfrecords_path + " read.")
    return samples


flatten = lambda l: [item for sublist in l for item in sublist]


if __name__ == '__main__':

    tf.enable_eager_execution()

    data_path = ARGS.read_dir

    if not os.path.exists(data_path):
        raise ValueError("Specified path does not exist!")

    poses = []
    for file in os.listdir(data_path):
        filename = os.fsdecode(file)
        if filename[0] != "s":  # to exclude stats.npz files
            start = time.time()
            data_path_ = data_path + filename
            samples = read_tfrecords(data_path_)
            poses.append(samples)

    poses = flatten(poses)
    nr_samples = len(poses)

    poses = np.vstack(poses)
    means = poses.mean(axis=0)
    vars = poses.var(axis=0)
    mean_ = poses.mean()
    var_ = poses.var()
    min_ = poses.min()
    max_ = poses.max()

    # sanity check
    lens = [i.shape[0] for i in poses]
    min_lens = min(lens)
    max_lens = max(lens)

    stats = dict(mean_channel=means, mean_all=mean_, var_channel=vars, var_all=var_, \
             min_all=min_, max_all=max_, min_seq_len=min_lens, max_seq_len=max_lens, num_samples=nr_samples)

    np.savez(data_path + 'stats', stats=stats)

