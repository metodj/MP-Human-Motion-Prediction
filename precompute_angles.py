import tensorflow as tf
import functools
import numpy as np
import os
import time
import argparse
from pp_utils import rot_mats_to_angle_axis, rot_mats_to_angle_axis_cv2

parser = argparse.ArgumentParser()
parser.add_argument('--read_dir', required=True, default='C:/Users/roksi/data/', help='Where the tfrecords are stored.')
parser.add_argument('--write_dir', required=True, default='C:/Users/roksi/data_angles/', help='Where to save tfrecords.')

ARGS = parser.parse_args()


def read_tfrecords(tfrecords_path, nr=5, angles=True):
    """Read tfrecord file.

        Args:
            tfrecords_path: file path
            nr: parameter for debugging
            angles: parameter for debugging, i.e. when reading newly written tf_records and comparing them with the
            previous ones

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
    counter = 0
    for s in iterator:
        tmp = s["poses"].numpy()
        if angles:
            # Own and map
            # tmp = rot_mats_to_angle_axis(tmp)
            # CV2 and for loop (2x faster)
            tmp = rot_mats_to_angle_axis_cv2(tmp)
            # print(s["poses"].numpy().shape, tmp.shape)
        samples.append(tmp)
        # if counter >= nr:
        #     break
        # counter += 1

    print(tfrecords_path + " read.")
    return samples


def write_tfrecords(samples, output_filename):
    """Writes samples to tf_records."""

    writer = tf.python_io.TFRecordWriter(output_filename)

    def float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    for i in range(len(samples)):
        feature_dict = {
            'file_id': bytes_feature(bytes(output_filename + "_" + str(i), encoding='utf-8')),
            'shape': int64_feature(samples[i].shape),
            'poses': float_feature(np.reshape(samples[i], (-1,))),
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

        writer.write(example.SerializeToString())

    writer.close()
    print(output_filename + " written.")


if __name__ == '__main__':

    tf.enable_eager_execution()

    data_path = ARGS.read_dir
    data_angles_path = ARGS.write_dir

    print(data_path)
    print(data_angles_path)

    # IMPORTANT: code here assumes writing directory already exists!

    for file in os.listdir(data_path):
        filename = os.fsdecode(file)
        for file2 in os.listdir(data_path + filename + '/'):
            filename2 = os.fsdecode(file2)
            if filename2[0] != "s":  # to exclude stats.npz files
                start = time.time()
                data_path_ = data_path + filename + '/' + filename2
                data_angles_path_ = data_angles_path + filename + '/' + filename2
                samples = read_tfrecords(data_path_)
                write_tfrecords(samples, data_angles_path_)
                print("Elapsed time: ", time.time() - start)

    # data_path = "./data/training/poses-00001-of-00016"
    # data_angle_path = "./data_angles/training/poses-00001-of-00016"
    #
    # samples = read_tfrecords(data_path, 5)
    # # for i in samples:
    # #     print(i.shape)
    # # print(len(samples))
    #
    # write_tfrecords(samples, data_angle_path)
    #
    # # samples2 = read_tfrecords(data_angle_path, 5 ,angles=False)
    # # print(np.all(samples[3] == samples2[3]))



