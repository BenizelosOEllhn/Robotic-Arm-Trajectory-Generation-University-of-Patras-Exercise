import numpy as np


def damped_pinv(J, lam=0.05):
    """Return damped least-squares pseudoinverse for potentially singular matrices."""
    JJt = J @ J.T
    return J.T @ np.linalg.inv(JJt + (lam ** 2) * np.eye(JJt.shape[0]))


def hat(vec):
    v = vec.reshape((3,))
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])


def exp_rotation(p):
    phi = p.reshape((3, 1))
    theta = np.linalg.norm(phi)
    if theta < 1e-12:
        return np.eye(3, 3)
    a = phi / theta
    return (
        np.eye(3) * np.cos(theta)
        + (1.0 - np.cos(theta)) * a @ a.T
        + np.sin(theta) * hat(a)
    )


def log_rotation(R):
    theta = np.arccos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0)))
    if theta < 1e-12:
        return np.zeros((3, 1))
    mat = R - R.T
    r = np.array([mat[2, 1], mat[0, 2], mat[1, 0]]).reshape((3, 1))
    return theta / (2.0 * np.sin(theta)) * r


def exp_pose(tau):
    theta = np.linalg.norm(tau[:3, :])
    R = np.eye(3)
    p = np.zeros((3, 1))
    if not np.isclose(theta, 0.0):
        r = tau[:3, :] / theta
        rho = tau[3:, :] / theta
        rh = hat(r)
        R = exp_rotation(tau[:3, :])
        p = (
            (np.eye(3) * theta)
            + (1.0 - np.cos(theta)) * rh
            + (theta - np.sin(theta)) * (rh @ rh)
        ) @ rho
    else:
        p = tau[3:, :]
    return np.block([[R, p], [np.zeros((1, 3)), 1.0]])


def log_pose(T):
    R = T[:3, :3]
    p = T[:3, 3:]
    rt = log_rotation(R)
    theta = np.linalg.norm(rt)
    if np.allclose(theta, 0.0):
        return np.block([[np.zeros((3, 1))], [p]])
    rh = hat(rt / theta)
    Ginv = (
        (1.0 / theta) * np.eye(3)
        - 0.5 * rh
        + (1.0 / theta - 0.5 / np.tan(theta / 2.0)) * (rh @ rh)
    )
    return np.block([[rt], [Ginv @ p * theta]])


def Adj(R, p):
    Tadj = np.zeros((6, 6))
    Tadj[:3, :3] = R
    Tadj[3:, 3:] = R
    Tadj[3:, :3] = hat(p) @ R
    return Tadj


def homogeneous(R, p=np.zeros((3, 1))):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3:] = p.reshape((3, 1))
    return T
