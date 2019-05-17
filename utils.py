"""Copyright (c) 2019 AIT Lab, ETH Zurich, Manuel Kaufmann, Emre Aksan

Students and holders of copies of this code, accompanying datasets,
and documentation, are not allowed to copy, distribute or modify
any of the mentioned materials beyond the scope and duration of the
Machine Perception course projects.

That is, no partial/full copy nor modification of this code and
accompanying data should be made publicly or privately available to
current/future students or other parties.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import zipfile
from constants import Constants as C


def get_activation_fn(activation=C.RELU):
    """
    Return tensorflow activation function given string name.

    Args:
        activation: The requested activation function.

    Returns: The tf op corresponding to the requested activation function.
    """
    # Check if the activation is already callable.
    if callable(activation):
        return activation

    if activation is None:
        return None
    elif activation == C.RELU:
        return tf.nn.relu
    elif activation == C.TANH:
        return tf.nn.tanh
    elif activation == C.SIGMOID:
        return tf.nn.sigmoid
    else:
        raise Exception("Activation function is not implemented.")


def export_code(file_list, output_file):
    """
    Adds the given file paths to a zip file.
    Args:
        file_list: List of paths to files
        output_file: Name and path of the zip archive to be created
    """
    zipf = zipfile.ZipFile(output_file, mode="w", compression=zipfile.ZIP_DEFLATED)
    for f in file_list:
        zipf.write(f)
    zipf.close()


def export_results(eval_result, output_file):
    """
    Write predictions into a csv file that can be uploaded to the submission system.
    Args:
        eval_result: A dictionary {sample_id => (prediction, seed)}. This is exactly what is returned
          by `evaluate_test.evaluate_model`.
        output_file: Where to store the file.
    """

    def to_csv(fname, poses, ids, split=None):
        n_samples, seq_length, dof = poses.shape
        data_r = np.reshape(poses, [n_samples, seq_length * dof])
        cols = ['dof{}'.format(i) for i in range(seq_length * dof)]

        # add split id very last
        if split is not None:
            data_r = np.concatenate([data_r, split[..., np.newaxis]], axis=-1)
            cols.append("split")

        data_frame = pd.DataFrame(data_r,
                                  index=ids,
                                  columns=cols)
        data_frame.index.name = 'Id'

        if not fname.endswith('.gz'):
            fname += '.gz'

        data_frame.to_csv(fname, float_format='%.8f', compression='gzip')

    sample_file_ids = []
    sample_poses = []
    for k in eval_result:
        sample_file_ids.append(k)
        sample_poses.append(eval_result[k][0])

    to_csv(output_file, np.stack(sample_poses), sample_file_ids)


def geodesic_distance(x1, x2, angle_axis):
    # if angle_axis:
    #     x1 = eulers_to_rotmats(x1)
    #     x2 = eulers_to_rotmats(x2)
    y1 = tf.reshape(x1, shape=[-1, 3, 3])
    y2 = tf.reshape(x2, shape=[-1, 3, 3])
    y2 = tf.transpose(y2, perm=[0, 2, 1])

    z = tf.matmul(y1, y2)
    zt = tf.transpose(z, perm=[0, 2, 1])

    u = (z - zt) / 2

    v = tf.square(tf.reshape(u, shape=(-1, 9)))
    w = tf.sqrt(tf.reduce_sum(v, axis=1))

    a_norm = tf.divide(w, np.sqrt(2))
    a_norm = tf.clip_by_value(a_norm, -1.0, 1.0)  # Account for numerical errors

    return tf.reduce_mean(tf.abs(tf.asin(a_norm)))


def is_rotmat(r):
    rt = np.transpose(r)
    n = np.linalg.norm(np.eye(3, dtype=r.dtype) - np.dot(rt, r))
    return n < 1e-6


def rodrigues_rok(input):
    if input.shape == (3,) or input.shape == (3, 1):
        def k_mat(axis):
            return np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])

        theta = np.linalg.norm(input)

        if theta < 1e-30:
            return np.eye(3)
        else:
            axis_ = input / theta
            K = k_mat(axis_)

            return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    elif input.shape == (3, 3):
        assert is_rotmat(input)

        angle_axis = np.zeros(shape=(3,))

        if np.all(input == np.eye(3)):
            return angle_axis
        else:
            k = (input - input.T) / 2
            angle_axis[0] = k[2, 1]
            angle_axis[1] = k[0, 2]
            angle_axis[2] = k[1, 0]

            norm = np.linalg.norm(angle_axis)
            angle_axis = angle_axis / norm

            theta_1 = np.arccos((np.trace(input) - 1) / 2)
            # theta_2 = np.arcsin(np.clip(norm, 0, 1))

            angle_axis = angle_axis * theta_1

            return angle_axis


def rot_mats_to_angle_axis(rot_mat_tensor):
    s = rot_mat_tensor.shape  # (16, 24, 135) / (384, 135)

    rot_mat_tensor = np.reshape(rot_mat_tensor, newshape=(-1, 3, 3))  # (5760, 3, 3)
    angle_axis_tensor = np.stack(list(map(rodrigues_rok, rot_mat_tensor)), axis=0)  # (5760, 3)

    if len(s) == 2:
        angle_axis_tensor = np.reshape(angle_axis_tensor, newshape=(-1, 45))  # (384, 45)
    elif len(s) == 3:
        angle_axis_tensor = np.reshape(angle_axis_tensor, newshape=(s[0], s[1], 45))  # (16, 24, 45)
    return angle_axis_tensor


def angle_axis_to_rot_mats(angle_axis_tensor):
    s = angle_axis_tensor.shape  # (16, 24, 45) / (384, 45)

    angle_axis_tensor = np.reshape(angle_axis_tensor, newshape=(-1, 3))  # (5760, 3)
    rot_mat_tensor = np.stack(list(map(rodrigues_rok, angle_axis_tensor)), axis=0)  # (5760, 3, 3)

    if len(s) == 2:
        rot_mat_tensor = np.reshape(rot_mat_tensor, newshape=(-1, 135))  # (384, 135)
    elif len(s) == 3:
        rot_mat_tensor = np.reshape(rot_mat_tensor, newshape=(s[0], s[1], 135))  # (16, 24, 135)
    return rot_mat_tensor


