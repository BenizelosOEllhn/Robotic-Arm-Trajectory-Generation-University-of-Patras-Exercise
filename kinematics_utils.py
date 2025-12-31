import numpy as np

from math_utils import exp_pose, Adj, homogeneous, log_pose

# Dimensions (Meters)
d1 = 0.1695
d3 = 0.1155
d5 = 0.12783
d7 = 0.06598
z_off = 0.075   # tool_link offset
x_off = 0.008

z_total = d1 + d3 + d5 + d7 + z_off

# Home Matrix M
M = np.array([
    [1, 0, 0, x_off],
    [0, 1, 0, 0.0],
    [0, 0, 1, z_total],
    [0, 0, 0, 1]
])

h_j2 = d1
h_j4 = d1 + d3
h_j6 = d1 + d3 + d5

S_list = np.array([
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # J1
    [0.0, 1.0, 0.0, -h_j2, 0.0, 0.0],  # J2
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # J3
    [0.0, -1.0, 0.0, h_j4, 0.0, 0.0],  # J4
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # J5
    [0.0, -1.0, 0.0, h_j6, 0.0, 0.0],  # J6
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # J7
]).T

JOINT_LIMITS_DEG = [
    (-160, 160), (-70, 115), (-170, 170),
    (-113, 75), (-170, 170), (-115, 115), (-180, 180)
]


def FK_PoE(thetas):
    T = np.eye(4)
    for i in range(7):
        twist = S_list[:, i].reshape((6, 1)) * thetas[i]
        T = T @ exp_pose(twist)
    return T @ M


def FK_all_links(thetas):
    T = np.eye(4)
    T_list = [T.copy()]
    for i in range(7):
        twist = S_list[:, i].reshape((6, 1)) * thetas[i]
        T = T @ exp_pose(twist)
        T_list.append(T.copy())
    return T_list


def JacobianSpace(thetas):
    Js = np.zeros((6, 7))
    T = np.eye(4)
    Js[:, 0] = S_list[:, 0]
    for i in range(1, 7):
        twist_prev = S_list[:, i - 1].reshape((6, 1)) * thetas[i - 1]
        T = T @ exp_pose(twist_prev)
        R = T[:3, :3]
        p = T[:3, 3:]
        AdT = Adj(R, p)
        Js[:, i] = AdT @ S_list[:, i]
    return Js


def normalize_angles(thetas):
    return (thetas + np.pi) % (2 * np.pi) - np.pi


def check_limits(thetas):
    deg = np.degrees(thetas)
    for val, (low, high) in zip(deg, JOINT_LIMITS_DEG):
        if not (low <= val <= high):
            return False
    return True


def NewtonRaphsonIK(T_des, theta_guess):
    theta = np.copy(theta_guess)
    for _ in range(120):
        T_curr = FK_PoE(theta)
        T_rel = T_des @ np.linalg.inv(T_curr)
        V_err = log_pose(T_rel).reshape((6,))
        if np.linalg.norm(V_err[:3]) < 1e-3 and np.linalg.norm(V_err[3:]) < 1e-3:
            return normalize_angles(theta), True
        Js = JacobianSpace(theta)
        theta += np.linalg.pinv(Js) @ V_err
    return theta, False


def solve_with_restarts(T_target, attempts=80):
    smart_guess = np.radians([0, 30, 0, -90, 0, 30, 0])
    q, conv = NewtonRaphsonIK(T_target, smart_guess)
    if conv and check_limits(q):
        return q, True, 0
    for i in range(attempts):
        rand_guess = np.random.uniform(-1.5, 1.5, 7)
        q, conv = NewtonRaphsonIK(T_target, rand_guess)
        if conv and check_limits(q):
            return q, True, i + 1
    return np.zeros(7), False, -1
