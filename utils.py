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


def geodesic_distance(x1, x2):
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


def rotmat_to_euler(r):
    assert (is_rotmat(r))

    sy = np.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(r[2, 1], r[2, 2])
        y = np.arctan2(-r[2, 0], sy)
        z = np.arctan2(r[1, 0], r[0, 0])
    else:
        x = np.arctan2(-r[1, 2], r[1, 1])
        y = np.arctan2(-r[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def euler_to_rotmat(theta):
    r_x = np.array([[1, 0, 0],
                    [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0, np.sin(theta[0]), np.cos(theta[0])]
                    ])

    r_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                    [0, 1, 0],
                    [-np.sin(theta[1]), 0, np.cos(theta[1])]
                    ])

    r_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    r = np.dot(r_z, np.dot(r_y, r_x))
    return r


def rotmats_to_eulers(p):
    p = np.reshape(p, newshape=(-1, 3, 3))
    a = np.zeros(shape=(p.shape[0], 3), dtype=np.float32)

    for i in range(p.shape[0]):
        r = p[i, :, :]
        # theta = rotmat_to_euler(r)
        theta, _ = cv2.Rodrigues(r)
        a[i, :] = np.reshape(theta, newshape=(3,))/np.pi

    a = np.reshape(a, newshape=(-1, 15 * 3))
    return a


def eulers_to_rotmats(a):
    s = a.shape  # (16, 24, 45)

    a = np.reshape(a, newshape=(-1, 3))  # (384, 3)
    p = np.zeros(shape=(a.shape[0], 3, 3), dtype=np.float32)  # (384, 3, 3)

    for i in range(s[0]):
        theta = a[i, :] * np.pi  # (3, )
        # r = euler_to_rotmat(theta)
        r, _ = cv2.Rodrigues(theta)
        p[i, :, :] = r

    p = get_closest_rotmat(p)
    p = np.reshape(p, newshape=(s[0], s[1], 135))
    return p


