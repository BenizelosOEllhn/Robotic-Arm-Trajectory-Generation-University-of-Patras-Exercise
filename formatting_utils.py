import numpy as np
from scipy.spatial.transform import Rotation as R_scipy


def format_vector(vec, precision=3):
    return np.array2string(vec, precision=precision, suppress_small=True)


def format_rpy_deg(R):
    rpy = R_scipy.from_matrix(R).as_euler('xyz', degrees=True)
    return np.array2string(rpy, precision=2, suppress_small=True)


def format_joints_deg(q):
    return np.array2string(np.degrees(q), precision=1, suppress_small=True)
