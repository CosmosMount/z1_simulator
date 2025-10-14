"""Utility functions for GRR"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

grd_yup2grd_zup = np.array([[0, 0, -1, 0],
                            [-1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]])

T_base_to_link06 = np.array([[ 1.00000000e+00, 3.72528985e-09, -1.13686824e-13, -1.27999783e-02],
                             [-3.72528985e-09, 1.00000000e+00, 2.46634302e-15, 4.76927942e-11],
                             [ 1.13686824e-13, -2.46634260e-15, 1.00000000e+00, 1.60500005e-01],
                             [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

M_bias = np.array([[0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0.2],
                   [0, 0, 0, 0]])

link06_init_pose = np.array([-0.012799978256225586, 4.769279415839378e-11, 0.16050000488758087, 0, 0, 0, 1])

def quat_multiply(q1, q2):
    """
    Multiplies two quaternions in [x, y, z, w] format.
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    ])


def quat_conjugate(q):
    """
    Computes the conjugate of a quaternion in [x, y, z, w] format.
    """
    q = np.array(q)
    return np.array([-q[0], -q[1], -q[2], q[3]])


def interpolate_quat(quat1, quat2, u):
    """Interpolate between two rotation vectors given a ratio"""
    quat1 = R.from_quat(quat1).as_quat()
    quat2 = R.from_quat(quat2).as_quat()

    # Spherical linear interpolation (SLERP)
    rotations = R.from_quat([quat1, quat2])
    slerp = Slerp([0, 1], rotations)
    interpolated_quat = slerp([u])[0].as_quat()

    return interpolated_quat


def rotvec_to_quat(rotvec):
    """Convert rotation vector to a quaternion"""
    return R.from_rotvec(rotvec).as_quat()


def quat_to_rotvec(quat):
    """Convert a quaternion to a rotation vector"""
    return R.from_quat(quat).as_rotvec()

def quat_to_euler(quat):
    """Convert a quaternion to Euler angles"""
    return R.from_quat(quat).as_euler('xyz')

def mat_update(prev_mat, mat):
    if np.linalg.det(mat) == 0:
        return prev_mat
    else:
        return mat

def fast_mat_inv(mat):
    ret = np.eye(4)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret

