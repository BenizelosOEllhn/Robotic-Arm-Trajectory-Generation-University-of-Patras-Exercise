import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from visualization_msgs.msg import MarkerArray
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
import threading
import sys

from math_utils import damped_pinv, log_pose, homogeneous
from trajectory_utils import (
    plot_time_scaling_profiles,
    generate_piecewise_SE3,
    finite_difference_Vd_left,
    make_translation,
    SE3_screw_interpolation
)
from kinematics_utils import (
    FK_PoE,
    FK_all_links,
    JacobianSpace,
    solve_with_restarts
)
from marker_utils import (
    create_marker,
    create_sphere_marker,
    create_trajectory_marker
)
from workspace_utils import (
    setup_workspace,
    is_pose_in_collision,
    is_robot_in_collision
)
from reporting_utils import (
    print_pose_catalog,
    log_ik_summary,
    run_fk_validations
)
from telemetry_utils import TelemetryLogger, render_state_machine_diagram

# ==========================================
# PART 3: ROS 2 VISUALIZATION NODE
# ==========================================

class HW2Node(Node):
    def __init__(self):
        super().__init__('hw2_minimal_grasp_place')
        self.pub_joints = self.create_publisher(JointState, '/joint_states', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/visualization_marker_array', 10)

        print("\n=== CALCULATING INVERSE KINEMATICS ===")
        np.set_printoptions(precision=4, suppress=True)
        self._ik_header_shown = False

        # --------- Home (all zeros) ----------
        self.q_home = np.zeros(7)
        self.T_home = FK_PoE(self.q_home)

        # --------- Grasp pose (Z=0.030) ----------
        self.T_grasp = np.array([
            [1,  0,  0, 0.175],
            [0, -1,  0, 0.025],
            [0,  0, -1, 0.030],
            [0,  0,  0, 1.000]
        ])

        # Hover waypoint directly above grasp to keep approach collision-free
        self.hover_offset = 0.25
        self.hover_side_offset = -0.03
        self.hover_roll_deg = -5.0
        self.hover_tilt_deg = 10.0
        self.T_hover = self.T_grasp.copy()
        hover_pos_delta = np.array([0.0, self.hover_side_offset, self.hover_offset])
        self.T_hover[:3, 3] = self.T_grasp[:3, 3] + hover_pos_delta
        hover_rot = R_scipy.from_euler(
            'yx',
            [np.radians(self.hover_tilt_deg), np.radians(self.hover_roll_deg)]
        ).as_matrix() @ self.T_grasp[:3, :3]
        self.T_hover[:3, :3] = hover_rot

        # --------- Place pose (Rotated) ----------
        self.T_place = np.array([
            [ 0.707,  0.707,  0.000,  0.100], 
            [ 0.000,  0.000,  1.000,  0.080], 
            [ 0.707, -0.707,  0.000,  0.100], 
            [ 0.000,  0.000,  0.000,  1.000]
        ])

        self.obstacles, self.collision_margin, self.safe_height = setup_workspace()

        # --------- HW3 Task 5: SE(3) trajectory waypoints ----------
        # Robot starts at home, then moves to grasp (approach = 0 offset), lifts upward, and routes safely to place
        h_app = 0.0      # no vertical offset before grasp
        h_lift_world = 0.12  # lift upward in world frame to avoid dipping below table

        self.T_pregrasp = self.T_grasp.copy()
        self.T_postgrasp = self.T_grasp.copy()
        self.T_postgrasp[2, 3] += h_lift_world
        self.depart_offset = np.array([-0.045, -0.055, 0.08])
        self.T_depart = self.T_postgrasp.copy()
        self.T_depart[:3, 3] = self.T_postgrasp[:3, 3] + self.depart_offset

        # Waypoint list: home → hover → pregrasp → grasp → postgrasp → depart → place
        self.T_waypoints = [
            self.T_home,
            self.T_hover,
            self.T_pregrasp,
            self.T_grasp,
            self.T_postgrasp,
            self.T_depart,
            self.T_place
        ]

        # Segment durations (seconds) for each waypoint-to-waypoint motion (slower)
        self.seg_durations = [4.0, 2.0, 0.8, 1.2, 2.0, 5.0]
        self.dt_traj = 0.05                   # should match your timer dt

        print_pose_catalog([
            ("home", self.T_home),
            ("hover", self.T_hover),
            ("pregrasp", self.T_pregrasp),
            ("grasp", self.T_grasp),
            ("postgrasp", self.T_postgrasp),
            ("depart", self.T_depart),
            ("place", self.T_place)
        ])

        print("\n=== IK SOLUTIONS (deg) ===")
        self._ik_header_shown = log_ik_summary('home', True, 0, self.q_home, self._ik_header_shown)
        self.solve_pose_with_logging('hover', self.T_hover)
        self.solve_pose_with_logging('pregrasp', self.T_pregrasp)
        self.solve_pose_with_logging('grasp', self.T_grasp)
        self.solve_pose_with_logging('postgrasp', self.T_postgrasp)
        self.solve_pose_with_logging('depart', self.T_depart)
        self.solve_pose_with_logging('place', self.T_place)

        plot_time_scaling_profiles(
            T=max(self.seg_durations),
            num_samples=400,
            show=False,
            save_path='time_scaling_profiles.png'
        )
        self.get_logger().info("Saved cubic time-scaling figure to time_scaling_profiles.png")

        self.T_traj, self.t_traj = generate_piecewise_SE3(
            self.T_waypoints, self.seg_durations, self.dt_traj
        )

        # (Optional) precompute desired twists for later controller
        self.Vd_traj = finite_difference_Vd_left(self.T_traj, self.dt_traj)

        # Indices of important poses along the trajectory
        self.idx_home = 0
        self.idx_grasp = self.find_closest_pose_index(self.T_traj, self.T_grasp)
        self.idx_depart = self.find_closest_pose_index(self.T_traj, self.T_depart)
        self.idx_place = len(self.T_traj) - 1

        # Precompute end-effector positions for trajectory visualization
        self.traj_points = [T[:3, 3] for T in self.T_traj]

        print(f"[TRAJ] Generated {len(self.T_traj)} desired SE(3) poses.")

        self.run_fk_validations()

        # --------- Cube Logic ----------
        # Cube initial state (on table)
        self.cube_table_pos = np.array([0.175, 0.025, 0.025])
        self.cube_table_rot = np.eye(3)

        # Target final position (next to hole)
        self.cube_final_pos = self.T_place[:3, 3]
        self.cube_final_rot = self.T_place[:3, :3]

        # Grasp/release thresholds
        self.grasp_dist_thresh = 0.20  # allow grasp command up to 10 cm away
        self.place_dist_thresh = 0.01  # 1 cm for release

        # Grasp point in tool frame (10 cm along tool Z)
        self.T_grasp_point = homogeneous(np.eye(3), np.array([0, 0, -0.10]))

        # Grasp offset (computed when grasping)
        self.T_off = homogeneous(np.eye(3), np.zeros(3))

        # ---------- HW3 Task 6: Kinematic simulator state ----------
        self.dt = 0.05  # must match timer period

        # Robot "simulated" state: start at HOME configuration
        self.theta = self.q_home.copy()

        # Command: joint velocities
        self.theta_dot_cmd = np.zeros(7)

        # Cube grasp state
        self.is_grasped = False

        # ---------- HW3 Task 7: Task-space velocity control parameters ----------
        # Gains (diagonal): [w_x w_y w_z v_x v_y v_z]
        self.Kp = np.diag([2.5, 2.5, 2.5, 4.0, 4.0, 4.0])

        # Damping for pseudoinverse
        self.lam = 0.05

        # Joint velocity saturation (rad/s)
        self.max_qdot = 1.5

        # Trajectory index
        self.traj_idx = 0

        # Manual grasp/release commands from user
        self.grasp_cmd = False
        self.release_cmd = False
        self.pending_start = False

        # State machine for task execution
        self.state = "idle"
        self.return_home_start_time = 0.0
        self.return_home_duration = 4.0  # seconds to return to home
        self.return_path = []
        self.return_vd = []
        self.return_idx = 0

        self.sim_time = 0.0
        self.plot_request = False

        self.state_labels = [
            "idle",
            "approach",
            "wait_grasp",
            "post_grasp_motion",
            "wait_release",
            "return_home"
        ]
        self.telemetry = TelemetryLogger(self.state_labels, output_dir="figures")

        state_cards = [
            ("idle", "HOME pose; wait for 'start'"),
            ("approach", "Follow HOME->hover->pregrasp->grasp path"),
            ("wait_grasp", "Hold grasp pose until 'grasp'"),
            ("post_grasp_motion", "Execute postgrasp->depart->place"),
            ("wait_release", "Hold place pose until 'release'"),
            ("return_home", "Interpolate back to HOME")
        ]
        transitions = [
            ("idle", "approach", "'start'"),
            ("approach", "wait_grasp", "Reached grasp"),
            ("wait_grasp", "post_grasp_motion", "'grasp'"),
            ("post_grasp_motion", "wait_release", "Trajectory done"),
            ("wait_release", "return_home", "'release'"),
            ("return_home", "idle", "Back at HOME")
        ]
        diagram_path = self.telemetry.path_for('state_machine_graph.png')
        render_state_machine_diagram(state_cards, transitions, diagram_path)
        self.get_logger().info(f"Saved state-machine diagram to {diagram_path}")

        # Start user input thread
        self.input_thread = threading.Thread(target=self.user_input_thread, daemon=True)
        self.input_thread.start()

        # Timer
        self.timer = self.create_timer(self.dt, self.timer_callback)

        print("\n=== STARTING VISUALIZATION LOOP ===")
        print("Commands: type 'start' to begin approach, 'grasp' to pick up cube, 'release' to place cube")
    def user_input_thread(self):
        # Background thread to read user commands from terminal.
        while True:
            try:
                user_input = input(">>> ").strip().lower()
                if user_input == "grasp":
                    self.grasp_cmd = True
                    print("[CMD] Grasp command received")
                elif user_input == "release":
                    self.release_cmd = True
                    print("[CMD] Release command received")
                elif user_input == "start":
                    self.pending_start = True
                    print("[CMD] Start command received")
                elif user_input == "plots":
                    self.plot_request = True
                    print("[CMD] Plot generation requested")
            except:
                pass

        # Timer

    # ---------- Helper functions ----------
    def solve_pose_with_logging(self, label, T_target):
        q, ok, attempts = solve_with_restarts(T_target)
        setattr(self, f"q_{label}", q)
        setattr(self, f"ok_{label}", ok)
        setattr(self, f"att_{label}", attempts)
        self._ik_header_shown = log_ik_summary(
            label,
            ok,
            attempts,
            q,
            self._ik_header_shown
        )
        if not ok:
            self.get_logger().error(f"[IK] {label}: solver failed; trajectory may be invalid.")
        return q, ok, attempts

    def run_fk_validations(self):
        entries = [
            ('home', self.q_home, self.T_home, 0, True),
            ('hover', getattr(self, 'q_hover', None), self.T_hover, getattr(self, 'att_hover', 0), getattr(self, 'ok_hover', False)),
            ('pregrasp', getattr(self, 'q_pregrasp', None), self.T_pregrasp, getattr(self, 'att_pregrasp', 0), getattr(self, 'ok_pregrasp', False)),
            ('grasp', getattr(self, 'q_grasp', None), self.T_grasp, getattr(self, 'att_grasp', 0), getattr(self, 'ok_grasp', False)),
            ('postgrasp', getattr(self, 'q_postgrasp', None), self.T_postgrasp, getattr(self, 'att_postgrasp', 0), getattr(self, 'ok_postgrasp', False)),
            ('depart', getattr(self, 'q_depart', None), self.T_depart, getattr(self, 'att_depart', 0), getattr(self, 'ok_depart', False)),
            ('place', getattr(self, 'q_place', None), self.T_place, getattr(self, 'att_place', 0), getattr(self, 'ok_place', False))
        ]
        run_fk_validations(self.get_logger(), entries)

    def can_grasp(self, p_grasp):
        # Check if gripper is close enough to grasp the cube (only when not grasped).
        if self.is_grasped:
            return False
        return np.linalg.norm(p_grasp - self.cube_table_pos) <= self.grasp_dist_thresh

    def grasp(self, T_ee, cube_rot, cube_pos):
        # Grasp the cube (compute offset). Returns True if grasped.
        if self.is_grasped:
            return False
        T_cube_world = homogeneous(cube_rot, cube_pos)
        self.T_off = np.linalg.inv(T_ee) @ T_cube_world
        self.is_grasped = True
        return True

    def can_release(self):
        # Check if release is allowed (only when grasped).
        return self.is_grasped

    def release(self):
        # Release the cube. Returns True if released.
        if not self.is_grasped:
            return False
        self.is_grasped = False
        return True

    def find_closest_pose_index(self, traj, target):
        # Return index of trajectory pose closest to target pose.
        best_idx = 0
        best_err = float('inf')
        for i, T in enumerate(traj):
            pos_err = np.linalg.norm(T[:3, 3] - target[:3, 3])
            rot_err = np.linalg.norm(T[:3, :3] - target[:3, :3])
            err = pos_err + 0.1 * rot_err
            if err < best_err:
                best_err = err
                best_idx = i
        return best_idx

    def plan_return_home(self, T_start):
        # Create SE(3) path back to home pose after release.
        steps = max(3, int(self.return_home_duration / self.dt))
        self.return_path = []
        for k in range(steps):
            s = k / (steps - 1)
            self.return_path.append(SE3_screw_interpolation(T_start, self.T_home, s))
        self.return_vd = finite_difference_Vd_left(self.return_path, self.dt)
        self.return_idx = 0

    # ---------- Main callback ----------
    def timer_callback(self):
        # ---------------------------------------------------------
        # STATE MACHINE: idle → approach → wait_grasp → post_motion → wait_release → return_home
        # ---------------------------------------------------------
        if self.pending_start:
            if self.state == "idle":
                self.pending_start = False
                self.state = "approach"
                self.traj_idx = 0
                print(">>> Starting approach trajectory from HOME.")
            else:
                print("[WARN] Start command ignored; robot is busy. Wait until robot is HOME/idle.")
                self.pending_start = False

        if self.state == "idle":
            Td = self.T_home
            Vd = np.zeros(6)

        elif self.state == "approach":
            Td = self.T_traj[self.traj_idx]
            Vd = self.Vd_traj[self.traj_idx]
            if self.traj_idx < self.idx_grasp:
                self.traj_idx += 1
            else:
                self.state = "wait_grasp"
                self.traj_idx = self.idx_grasp
                print(">>> Reached grasp pose. Waiting for 'grasp' command...")

        elif self.state == "wait_grasp":
            Td = self.T_traj[self.idx_grasp]
            Vd = np.zeros(6)

        elif self.state == "post_grasp_motion":
            if self.traj_idx < len(self.T_traj):
                Td = self.T_traj[self.traj_idx]
                Vd = self.Vd_traj[self.traj_idx]
                self.traj_idx += 1
                if self.traj_idx >= len(self.T_traj):
                    self.state = "wait_release"
                    print(">>> At placement pose. Waiting for 'release' command...")
            else:
                Td = self.T_traj[-1]
                Vd = np.zeros(6)

        elif self.state == "wait_release":
            Td = self.T_traj[-1]
            Vd = np.zeros(6)

        elif self.state == "return_home":
            if self.return_idx < len(self.return_path):
                Td = self.return_path[self.return_idx]
                Vd = self.return_vd[self.return_idx]
                self.return_idx += 1
            else:
                Td = self.T_home
                Vd = np.zeros(6)
                self.state = "idle"
                self.traj_idx = 0
                print(">>> Back at HOME. Waiting for 'start' command...")
        else:
            Td = self.T_home
            Vd = np.zeros(6)

        nearing_place = (
            self.state == "post_grasp_motion" and
            self.traj_idx >= max(self.idx_depart, 0)
        )
        allow_contact = self.state == "wait_release" or nearing_place
        collided, link_id = is_robot_in_collision(
            self.theta,
            self.obstacles,
            self.collision_margin
        )
        if collided and not allow_contact:
            Td = Td.copy()
            Td[2, 3] = max(Td[2, 3], self.safe_height)
            self.get_logger().warn(
                f"Collision avoided: link {link_id} intersects container. Lifting EE."
            )

        # ---------------------------------------------------------
        # (2) Task-space controller: qdot = J† (Vd + Kp * Verr)
        # ---------------------------------------------------------
        T_curr = FK_PoE(self.theta)  # current end-effector pose
        T_err = Td @ np.linalg.inv(T_curr)  # left-invariant error
        V_err = log_pose(T_err).reshape((6,))  # twist error

        V_cmd = Vd + (self.Kp @ V_err)  # feedforward + feedback

        Js = JacobianSpace(self.theta)  # 6x7 Space Jacobian
        Js_pinv = damped_pinv(Js, self.lam)  # 7x6 damped pseudoinverse

        qdot = Js_pinv @ V_cmd  # joint velocities

        # Saturation
        qdot = np.clip(qdot, -self.max_qdot, self.max_qdot)

        self.theta_dot_cmd = qdot

        # ---------------------------------------------------------
        # (3) Integrate joint velocities (kinematic simulator)
        # ---------------------------------------------------------
        self.theta = self.theta + self.theta_dot_cmd * self.dt

        # ---------------------------------------------------------
        # (4) Publish joint state (for visualization)
        # ---------------------------------------------------------
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [
            'joint1_to_base', 'joint2_to_joint1', 'joint3_to_joint2',
            'joint4_to_joint3', 'joint5_to_joint4', 'joint6_to_joint5',
            'joint7_to_joint6'
        ]
        msg.position = [float(x) for x in self.theta]
        self.pub_joints.publish(msg)

        # ---------------------------------------------------------
        # (5) Compute end-effector and grasp point
        # ---------------------------------------------------------
        T_ee = FK_PoE(self.theta)
        T_grasp_world = T_ee @ self.T_grasp_point
        p_grasp = T_grasp_world[:3, 3]
        V_err_actual = log_pose(Td @ np.linalg.inv(T_ee)).reshape((6,))

        # ---------------------------------------------------------
        # (6) Update cube state (grasp/release logic) - MANUAL CONTROL
        # ---------------------------------------------------------
        cube_pos = self.cube_table_pos.copy()
        cube_rot = self.cube_table_rot.copy()

        if not self.is_grasped:
            # Cube is on table

            if self.state == "wait_grasp" and self.grasp_cmd:
                self.grasp_cmd = False
                if self.can_grasp(p_grasp):
                    if self.grasp(T_ee, cube_rot, cube_pos):
                        print(">>> GRASP executed (cube attached)")
                        self.state = "post_grasp_motion"
                        self.traj_idx = max(self.idx_grasp, self.traj_idx)
                else:
                    print("[WARN] Gripper not within grasp distance. Command ignored.")
            elif self.grasp_cmd:
                # Command issued in wrong phase
                print("[WARN] Cannot grasp right now. Wait until robot reaches grasp pose.")
                self.grasp_cmd = False

        else:
            # Cube rigidly follows end-effector
            T_cube = T_ee @ self.T_off
            cube_rot = T_cube[:3, :3]
            cube_pos = T_cube[:3, 3]

            if self.state == "wait_release" and self.release_cmd:
                self.release_cmd = False
                if self.release():
                    print(">>> RELEASE executed (cube detached)")
                    self.cube_table_pos = cube_pos.copy()
                    self.cube_table_rot = cube_rot.copy()
                    self.plan_return_home(T_ee)
                    self.state = "return_home"
                    print(">>> Returning to HOME configuration...")
            elif self.release_cmd:
                print("[WARN] Cannot release until robot reaches placement pose.")
                self.release_cmd = False

        self.telemetry.record(
            sim_time=self.sim_time,
            state_name=self.state,
            Td=Td,
            T_curr=T_ee,
            V_err=V_err_actual,
            qdot=self.theta_dot_cmd,
            Js=Js,
            cube_pos=cube_pos.copy(),
            p_grasp=p_grasp.copy()
        )
        self.sim_time += self.dt

        if self.plot_request:
            self.plot_request = False
            figure_paths = self.telemetry.plot_all()
            if figure_paths:
                joined = ", ".join(figure_paths)
                self.get_logger().info(f"Saved telemetry figures to {joined}")

        # ---------------------------------------------------------
        # (7) Publish visualization markers
        # ---------------------------------------------------------
        markers = MarkerArray()

        # Blue cube (object being manipulated)
        markers.markers.append(
            create_marker(
                id=1,
                scale=[0.05, 0.05, 0.05],
                color=[0.0, 0.0, 1.0, 1.0],
                pos=cube_pos,
                rot=cube_rot
            )
        )

        # Red box (container walls)
        red_rot = np.array([
            [ 0,  1,  0],
            [-1,  0,  0],
            [ 0,  0,  1]
        ])
        markers.markers.append(
            create_marker(
                id=2,
                scale=[0.05, 0.20, 0.20],
                color=[0.8, 0.2, 0.2, 0.6],
                pos=[0.10, 0.125, 0.10],
                rot=red_rot
            )
        )

        # The hole (darker inner square)
        markers.markers.append(
            create_marker(
                id=3,
                scale=[0.05, 0.05, 0.05],
                color=[0.0, 0.0, 0.0, 0.8],
                pos=[0.10, 0.125, 0.10],
                rot=self.cube_final_rot
            )
        )

        # Invisible masking cube
        markers.markers.append(
            create_marker(
                id=4,
                scale=[0.05, 0.05, 0.05],
                color=[0.0, 0.0, 0.0, 0.0],
                pos=[0.10, 0.125, 0.10],
                rot=self.cube_final_rot
            )
        )

        # ---- Waypoint markers (green spheres) ----
        for i, T_wp in enumerate(self.T_waypoints):
            wp_pos = T_wp[:3, 3]
            markers.markers.append(
                create_sphere_marker(
                    id=100 + i,
                    pos=wp_pos,
                    radius=0.015,
                    color=[0.0, 1.0, 0.0, 1.0]
                )
            )

        # ---- Dotted trajectory line (white) ----
        markers.markers.append(
            create_trajectory_marker(
                id=200,
                points=self.traj_points,
                color=[1.0, 1.0, 1.0, 1.0]
            )
        )

        self.pub_markers.publish(markers)


# ==========================================
# MAIN
# ==========================================

def main():
    rclpy.init()
    node = HW2Node()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()