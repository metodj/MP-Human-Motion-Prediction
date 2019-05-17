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
import cv2
from motion_metrics import get_closest_rotmat
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
    if angle_axis:
        x1 = eulers_to_rotmats(x1)
        x2 = eulers_to_rotmats(x2)
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


def rodrigues(input, rotmat_to_angle=True):
    if rotmat_to_angle:
        assert is_rotmat(input)
        angle_axis = np.zeros(shape=(3,))

        if np.all(input == np.eye(3)):
            return angle_axis

        rot = 0.5*(input - input.T)
        angle_axis[0] = rot[2, 1]
        angle_axis[1] = rot[0, 2]
        angle_axis[2] = rot[1, 0]

        norm = np.linalg.norm(angle_axis)
        norm = np.clip(norm, -1, 1)

        # TODO: which of the versions below is correct?
        # angle_axis = angle_axis / norm
        angle_axis = (angle_axis*np.arcsin(norm)) / norm
        return angle_axis
    else:
        rot_ = np.zeros(shape=(3, 3))
        theta = np.linalg.norm(input)
        if theta < 1e-5:
            return np.eye(3)
        angle_vec = input / theta
        rot_[0, 1] = -angle_vec[2]
        rot_[0, 2] = angle_vec[1]
        rot_[1, 0] = angle_vec[2]
        rot_[1, 2] = -angle_vec[0]
        rot_[2, 0] = -angle_vec[1]
        rot_[2, 1] = angle_vec[0]

        rot = np.cos(theta)*np.eye(3) + (1-np.cos(theta))*np.outer(angle_vec, angle_vec) \
                                                    + np.sin(theta)*rot_
        return rot



def rotmats_to_eulers(p):
    p = np.reshape(p, newshape=(-1, 3, 3))
    # a = np.apply_over_axes(rodrigues, p, [1, 2])
    # a = np.apply_along_axis(rodrigues, 1, p)

    a = np.zeros(shape=(p.shape[0], 3), dtype=np.float32)

    for i in range(p.shape[0]):
        # theta = rotmat_to_euler(r)
        # theta, _ = cv2.Rodrigues(r)
        theta = rodrigues(p[i, :, :])

        # TODO: divide by pi or not?
        # a[i, :] = np.reshape(theta, newshape=(3,))/np.pi
        a[i, :] = np.reshape(theta, newshape=(3,))

    a = np.reshape(a, newshape=(-1, 15 * 3))
    return a


def eulers_to_rotmats(a):
    s = a.shape  # (16, 24, 45)

    a = np.reshape(a, newshape=(-1, 3))  # (384, 3)
    p = np.zeros(shape=(a.shape[0], 3, 3), dtype=np.float32)  # (384, 3, 3)

    for i in range(a.shape[0]):
        # TODO: multiply by pi or not?
        # theta = a[i, :] * np.pi  # (3, )

        # r = euler_to_rotmat(theta)
        # r, _ = cv2.Rodrigues(theta)
        r = rodrigues(a[i, :], rotmat_to_angle=False)
        p[i, :, :] = r

    # p = get_closest_rotmat(p)
    p = np.reshape(p, newshape=(s[0], s[1], 135))
    return p


