import numpy as np
import matplotlib.pyplot as plt

from math_utils import log_pose, exp_pose, homogeneous


def cubic_time_scaling(t, T):
    if T <= 0:
        return 1.0
    tau = np.clip(t / T, 0.0, 1.0)
    return 3.0 * tau ** 2 - 2.0 * tau ** 3


def cubic_time_scaling_first_derivative(t, T):
    t = np.asarray(t, dtype=float)
    if T <= 0:
        return np.zeros_like(t)
    tau = np.clip(t / T, 0.0, 1.0)
    return (6.0 * tau - 6.0 * tau ** 2) / T


def cubic_time_scaling_second_derivative(t, T):
    t = np.asarray(t, dtype=float)
    if T <= 0:
        return np.zeros_like(t)
    tau = np.clip(t / T, 0.0, 1.0)
    return (6.0 - 12.0 * tau) / (T ** 2)


def plot_time_scaling_profiles(T=5.0, num_samples=300, show=True, save_path=None):
    T_plot = max(T, 1e-6)
    t_vals = np.linspace(0.0, T_plot, num_samples)
    s_vals = np.array([cubic_time_scaling(t, T_plot) for t in t_vals])
    s_dot_vals = cubic_time_scaling_first_derivative(t_vals, T_plot)
    s_ddot_vals = cubic_time_scaling_second_derivative(t_vals, T_plot)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6, 8))
    axes[0].plot(t_vals, s_vals, color='tab:blue')
    axes[0].set_ylabel('s(t)')
    axes[1].plot(t_vals, s_dot_vals, color='tab:orange')
    axes[1].set_ylabel('ŝ(t)')
    axes[2].plot(t_vals, s_ddot_vals, color='tab:green')
    axes[2].set_ylabel('s¨(t)')
    axes[2].set_xlabel('t [s]')

    for ax in axes:
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

    fig.suptitle('Cubic Time-Scaling Profiles', fontsize=14)
    plt.tight_layout(pad=1.2)

    if save_path:
        fig.savefig(save_path, dpi=200)

    if show:
        plt.show(block=False)
    else:
        plt.close(fig)


def SE3_screw_interpolation(T_start, T_end, s):
    T_rel = np.linalg.inv(T_start) @ T_end
    V = log_pose(T_rel)
    return T_start @ exp_pose(V * s)


def make_translation(dx, dy, dz):
    return homogeneous(np.eye(3), np.array([dx, dy, dz]))


def generate_piecewise_SE3(waypoints, durations, dt):
    assert len(waypoints) >= 2
    assert len(durations) == len(waypoints) - 1
    T_list = []
    t_list = []

    t_global = 0.0
    for i in range(len(durations)):
        Ta = waypoints[i]
        Tb = waypoints[i + 1]
        Ti = durations[i]
        steps = max(2, int(np.ceil(Ti / dt)))

        for k in range(steps):
            t_local = (k / (steps - 1)) * Ti
            s = cubic_time_scaling(t_local, Ti)
            Td = SE3_screw_interpolation(Ta, Tb, s)
            T_list.append(Td)
            t_list.append(t_global + t_local)

        t_global += Ti

    return T_list, t_list


def finite_difference_Vd_left(T_list, dt):
    Vd_list = []
    for k in range(len(T_list) - 1):
        T_rel = T_list[k + 1] @ np.linalg.inv(T_list[k])
        V = log_pose(T_rel).reshape((6,))
        Vd_list.append(V / dt)
    Vd_list.append(Vd_list[-1].copy())
    return Vd_list
