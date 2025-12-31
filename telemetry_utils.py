import os
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # Ensures 3D projections are registered
from matplotlib.patches import FancyBboxPatch


class TelemetryLogger:
    """Collects per-cycle telemetry and renders diagnostic figures."""

    def __init__(self, state_labels: List[str], output_dir: str = "figures"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.state_labels = list(state_labels)
        self.state_to_idx = {name: idx for idx, name in enumerate(self.state_labels)}
        self.reset()

    def reset(self):
        self.times: List[float] = []
        self.state_indices: List[int] = []
        self.state_names: List[str] = []
        self.pos_error: List[float] = []
        self.orientation_error: List[float] = []
        self.cube_distance: List[float] = []
        self.controller_effort: List[float] = []
        self.jacobian_condition: List[float] = []
        self.task_space_errors: List[np.ndarray] = []
        self.joint_velocities: List[np.ndarray] = []
        self.actual_positions: List[np.ndarray] = []
        self.desired_positions: List[np.ndarray] = []

    def path_for(self, filename: str) -> str:
        return os.path.join(self.output_dir, filename)

    def record(self,
               sim_time: float,
               state_name: str,
               Td: np.ndarray,
               T_curr: np.ndarray,
               V_err: np.ndarray,
               qdot: np.ndarray,
               Js: np.ndarray,
               cube_pos: np.ndarray,
               p_grasp: np.ndarray):
        idx = self.state_to_idx.get(state_name)
        if idx is None:
            idx = len(self.state_labels)
            self.state_to_idx[state_name] = idx
            self.state_labels.append(state_name)
        self.times.append(sim_time)
        self.state_indices.append(idx)
        self.state_names.append(state_name)

        actual_pos = T_curr[:3, 3]
        desired_pos = Td[:3, 3]
        self.actual_positions.append(actual_pos.copy())
        self.desired_positions.append(desired_pos.copy())

        pos_err = np.linalg.norm(desired_pos - actual_pos)
        self.pos_error.append(pos_err)
        orientation_err = np.linalg.norm(V_err[:3])
        self.orientation_error.append(orientation_err)
        self.task_space_errors.append(V_err.copy())

        controller_norm = float(np.linalg.norm(qdot))
        self.controller_effort.append(controller_norm)
        self.joint_velocities.append(qdot.copy())

        cube_dist = np.linalg.norm(cube_pos - p_grasp)
        self.cube_distance.append(cube_dist)

        try:
            cond = float(np.linalg.cond(Js))
        except np.linalg.LinAlgError:
            cond = float('inf')
        self.jacobian_condition.append(cond)

    def has_data(self) -> bool:
        return len(self.times) > 1

    def plot_all(self) -> List[str]:
        if not self.has_data():
            print("[TELEMETRY] Not enough samples to plot.")
            return []
        paths = []
        paths.append(self._plot_scalar(self.pos_error, "Position error [m]", "pos_error.png",
                                       "End-effector position tracking error"))
        paths.append(self._plot_scalar(self.orientation_error, "Orientation error [rad]",
                                       "orientation_error.png",
                                       "End-effector orientation error magnitude"))
        paths.append(self._plot_scalar(self.cube_distance, "Cube–gripper distance [m]",
                                       "cube_distance.png",
                                       "Cube to gripper distance"))
        paths.append(self._plot_scalar(self.controller_effort, "||qdot|| [rad/s]",
                                       "controller_effort.png",
                                       "Controller effort norm"))
        paths.append(self._plot_scalar(self.jacobian_condition, "cond(J)",
                                       "jacobian_condition.png",
                                       "Jacobian condition number"))
        paths.append(self._plot_state_timeline())
        paths.append(self._plot_task_space_error())
        paths.append(self._plot_joint_velocities())
        paths.append(self._plot_trajectory_3d())
        return [p for p in paths if p]

    # ----- Internal plotting helpers -----
    def _plot_scalar(self, series: List[float], ylabel: str, filename: str, title: str) -> str:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self.times, series, color='tab:blue', linewidth=1.5)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
        fig.tight_layout()
        path = self.path_for(filename)
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    def _plot_state_timeline(self) -> str:
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.step(self.times, self.state_indices, where='post', color='tab:purple')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('State')
        ax.set_yticks(range(len(self.state_labels)))
        ax.set_yticklabels(self.state_labels)
        ax.set_title('State-machine timeline')
        ax.grid(True, axis='x', linestyle='--', linewidth=0.6, alpha=0.5)
        fig.tight_layout()
        path = self.path_for('state_timeline.png')
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    def _plot_task_space_error(self) -> str:
        data = np.array(self.task_space_errors)
        fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharex=True)
        labels = ['w_x', 'w_y', 'w_z', 'v_x', 'v_y', 'v_z']
        for i, ax in enumerate(axes.flat):
            ax.plot(self.times, data[:, i], linewidth=1.2)
            ax.set_ylabel(f'{labels[i]} [unit]')
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        axes[-1][0].set_xlabel('Time [s]')
        axes[-1][1].set_xlabel('Time [s]')
        fig.suptitle('Task-space error twist components vs time')
        fig.tight_layout(rect=[0, 0.02, 1, 0.97])
        path = self.path_for('task_space_error.png')
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    def _plot_joint_velocities(self) -> str:
        data = np.array(self.joint_velocities)
        fig, ax = plt.subplots(figsize=(10, 4))
        for j in range(data.shape[1]):
            ax.plot(self.times, data[:, j], label=f'joint{j + 1}')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Joint velocity [rad/s]')
        ax.set_title('Joint velocities vs time')
        ax.legend(loc='upper right', ncol=2, fontsize=8)
        ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
        fig.tight_layout()
        path = self.path_for('joint_velocities.png')
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    def _plot_trajectory_3d(self) -> str:
        actual = np.array(self.actual_positions)
        desired = np.array(self.desired_positions)
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(desired[:, 0], desired[:, 1], desired[:, 2], label='Desired', color='tab:blue')
        ax.plot(actual[:, 0], actual[:, 1], actual[:, 2], label='Actual', color='tab:orange')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title('Desired vs actual end-effector trajectory')
        ax.legend(loc='upper right')
        path = self.path_for('ee_traj_3d.png')
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path


def render_state_machine_diagram(state_info: List[Tuple[str, str]],
                                  transitions: List[Tuple[str, str, str]],
                                  output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis('off')

    n = len(state_info)
    x_positions = np.linspace(0.05, 0.95, n)
    y_box = 0.55
    width = 0.12
    height = 0.18

    name_to_center = {}

    for (name, description), x in zip(state_info, x_positions):
        box = FancyBboxPatch((x - width / 2, y_box - height / 2), width, height,
                             boxstyle='round,pad=0.02', linewidth=1.5,
                             edgecolor='tab:blue', facecolor='white')
        ax.add_patch(box)
        ax.text(x, y_box, name, ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(x, y_box - 0.16, description, ha='center', va='top', fontsize=8, wrap=True)
        name_to_center[name] = x

    for src, dst, label in transitions:
        if src not in name_to_center or dst not in name_to_center:
            continue
        x0 = name_to_center[src]
        x1 = name_to_center[dst]
        ax.annotate('', xy=(x1 - width / 2 + 0.01, y_box),
                    xytext=(x0 + width / 2 - 0.01, y_box),
                    arrowprops=dict(arrowstyle='->', linewidth=1.2, color='tab:gray'))
        mid = (x0 + x1) / 2
        ax.text(mid, y_box + 0.08, label, ha='center', va='bottom', fontsize=8, color='tab:gray')

    ax.set_title('State machine overview')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
