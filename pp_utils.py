import numpy as np
import cv2


def is_rotmat(r):
    rt = np.transpose(r)
    n = np.linalg.norm(np.eye(3, dtype=r.dtype) - np.dot(rt, r))
    return n < 1e-6


def rotation(theta):
    cx, cy, cz = np.cos(theta)
    sx, sy, sz = np.sin(theta)
    r = np.array(
        [
            [cy*cz, sx*sy*cz - cx*sz, cx*sy*cz + sx*sz],
            [cy*sz, sx*sy*sz + cx*cz, cx*sy*sz - sx*cz],
            [-sy, sx*cy, cx*cy]
        ]
    )

    r = r / np.linalg.det(r)
    return r


def random_rotation_matrix():
    theta = np.random.uniform(0, 2 * np.pi, size=(3,))
    r = rotation(theta)
    return r


def rodrigues_metod(input, rotmat_to_angle=True):
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


def rodrigues(input):
    if len(input) == 9:
        input = np.reshape(input, newshape=(3, 3))

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
            K = (input - input.T) / 2
            angle_axis[0] = K[2, 1]
            angle_axis[1] = K[0, 2]
            angle_axis[2] = K[1, 0]

            norm = np.linalg.norm(angle_axis)
            angle_axis = angle_axis / norm

            theta_1 = np.arccos((np.trace(input) - 1) / 2)
            # theta_2 = np.arcsin(np.clip(norm, 0, 1))

            angle_axis = angle_axis * theta_1

            return angle_axis


def rot_mats_to_angle_axis(rot_mat_tensor):
    s = rot_mat_tensor.shape  # (16, 24, 135) / (384, 135)

    rot_mat_tensor = np.reshape(rot_mat_tensor, newshape=(-1, 3, 3))  # (5760, 3, 3)
    angle_axis_tensor = np.stack(list(map(rodrigues, rot_mat_tensor)), axis=0)  # (5760, 3)

    if len(s) == 2:
        angle_axis_tensor = np.reshape(angle_axis_tensor, newshape=(-1, 45))  # (384, 45)
    elif len(s) == 3:
        angle_axis_tensor = np.reshape(angle_axis_tensor, newshape=(s[0], s[1], 45))  # (16, 24, 45)
    return angle_axis_tensor.astype(np.float32)


def angle_axis_to_rot_mats(angle_axis_tensor):
    s = angle_axis_tensor.shape  # (16, 24, 45) / (384, 45)

    angle_axis_tensor = np.reshape(angle_axis_tensor, newshape=(-1, 3))  # (5760, 3)
    rot_mat_tensor = np.stack(list(map(rodrigues, angle_axis_tensor)), axis=0)  # (5760, 3, 3)

    if len(s) == 2:
        rot_mat_tensor = np.reshape(rot_mat_tensor, newshape=(-1, 135))  # (384, 135)
    elif len(s) == 3:
        rot_mat_tensor = np.reshape(rot_mat_tensor, newshape=(s[0], s[1], 135))  # (16, 24, 135)
    return rot_mat_tensor.astype(np.float32)


def angle_axis_to_rot_mats_cv2(angle_axis_tensor):
    s = angle_axis_tensor.shape  # (16, 24, 45) / (384, 45)

    angle_axis_tensor = np.reshape(angle_axis_tensor, newshape=(-1, 3))  # (5760, 3)
    rot_mat_tensor = np.zeros(shape=(angle_axis_tensor.shape[0], 3, 3), dtype=np.float32)  # (5760, 3, 3)

    for idx in range(angle_axis_tensor.shape[0]):
        angle_axis = angle_axis_tensor[idx, :]
        rot_mat, _ = cv2.Rodrigues(angle_axis)

        assert is_rotmat(rot_mat)
        rot_mat_tensor[idx, :, :] = rot_mat

    if len(s) == 2:
        rot_mat_tensor = np.reshape(rot_mat_tensor, newshape=(-1, 135))  # (384, 135)
    elif len(s) == 3:
        rot_mat_tensor = np.reshape(rot_mat_tensor, newshape=(s[0], s[1], 135))  # (16, 24, 135)
    return rot_mat_tensor.astype(np.float32)


def rot_mats_to_angle_axis_cv2(rot_mat_tensor):
    s = rot_mat_tensor.shape  # (16, 24, 135) / (384, 135)

    rot_mat_tensor = np.reshape(rot_mat_tensor, newshape=(-1, 3, 3))  # (5760, 3, 3)
    angle_axis_tensor = np.zeros(shape=(rot_mat_tensor.shape[0], 3), dtype=np.float32)

    for idx in range(rot_mat_tensor.shape[0]):
        rot_mat = rot_mat_tensor[idx, :, :]

        assert is_rotmat(rot_mat)
        angle_axis, _ = cv2.Rodrigues(rot_mat)
        angle_axis = np.reshape(angle_axis, newshape=(3, ))

        angle_axis_tensor[idx, :] = angle_axis

    if len(s) == 2:
        angle_axis_tensor = np.reshape(angle_axis_tensor, newshape=(-1, 45))  # (384, 45)
    elif len(s) == 3:
        angle_axis_tensor = np.reshape(angle_axis_tensor, newshape=(s[0], s[1], 45))  # (16, 24, 45)
    return angle_axis_tensor.astype(np.float32)


if __name__ == "__main__":
    batch_size = 16
    seq_len = 24

    X1 = np.zeros(shape=(batch_size, seq_len, 135))
    X1_shape = X1.shape
    X1 = np.reshape(X1, newshape=(-1, 3, 3))

    for i in range(X1.shape[0]):
        R = random_rotation_matrix()
        X1[i, :, :] = R

    X1 = np.reshape(X1, newshape=(X1_shape[0], X1_shape[1], X1_shape[2]))

    # TRANSFORM
    Y1 = rot_mats_to_angle_axis(X1)
    Y2 = angle_axis_to_rot_mats(Y1)

    Y1_cv2 = rot_mats_to_angle_axis(X1)
    Y2_cv2 = angle_axis_to_rot_mats_cv2(Y1_cv2)

    print("X1", X1.shape, "\tY1", Y1.shape, "Y2", Y2.shape, "\tY1_cv2", Y1_cv2.shape, "Y2_cv2", Y2_cv2.shape)

    X1 = np.reshape(X1, newshape=(-1, 3, 3))
    Y1 = np.reshape(Y1, newshape=(-1, 3))
    Y2 = np.reshape(Y2, newshape=(-1, 3, 3))
    Y1_cv2 = np.reshape(Y1_cv2, newshape=(-1, 3))
    Y2_cv2 = np.reshape(Y2_cv2, newshape=(-1, 3, 3))

    print("X1", X1.shape, "\tY1", Y1.shape, "Y2", Y2.shape, "\tY1_cv2", Y1_cv2.shape, "Y2_cv2", Y2_cv2.shape)

    eps = 1e-6

    for i in range(X1.shape[0]):
        print("\n", i)
        print("||X1 - Y2||", np.linalg.norm(X1[i, :, :] - Y2[i, :, :]) < eps)
        print("||X1 - Y2_cv2||", np.linalg.norm(X1[i, :, :] - Y2_cv2[i, :, :]) < eps)
        print("||Y1_cv2 - Y1||", np.linalg.norm(Y1_cv2[i, :] - Y1[i, :]) < eps)
