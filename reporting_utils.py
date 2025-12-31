import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

from formatting_utils import format_vector, format_rpy_deg, format_joints_deg
from kinematics_utils import FK_PoE


def print_pose_catalog(pose_entries):
    print("\n--- Target Pose Catalog (pos [m], RPY [deg]) ---")
    for label, T in pose_entries:
        pos_str = format_vector(T[:3, 3])
        rpy_str = format_rpy_deg(T[:3, :3])
        print(f"{label:<10} | pos {pos_str} | rpy {rpy_str}")


def log_ik_summary(label, success, attempts, q, header_shown):
    status = "OK" if success else "FAIL"
    joints_str = format_joints_deg(q)
    if not header_shown:
        print("Pose       | Status | Attempts | Joint Angles (deg)")
        print("-" * 78)
        header_shown = True
    print(f"{label:<10} | {status:^6} | {attempts:^8} | {joints_str}")
    return header_shown


def report_fk_pose(logger, label, q, T_target, attempts):
    T_fk = FK_PoE(q)
    pos_fk = T_fk[:3, 3]
    rpy_fk = R_scipy.from_matrix(T_fk[:3, :3]).as_euler('xyz', degrees=True)
    pos_target = T_target[:3, 3]
    rot_rel = T_target[:3, :3].T @ T_fk[:3, :3]
    rot_vec = R_scipy.from_matrix(rot_rel).as_rotvec()
    pos_err_mm = np.linalg.norm(pos_fk - pos_target) * 1000.0
    rot_err_deg = np.degrees(np.linalg.norm(rot_vec))
    pos_fk_str = format_vector(pos_fk, precision=4)
    pos_tar_str = format_vector(pos_target, precision=4)
    rpy_str = format_rpy_deg(T_fk[:3, :3])
    joints_str = format_joints_deg(q)
    logger.info(
        f"[FK] {label:<10} pos {pos_fk_str} (target {pos_tar_str}) | "
        f"err {pos_err_mm:.2f} mm, rot_err {rot_err_deg:.3f} deg, attempts {attempts}"
    )
    logger.info(
        f"[FK] {label:<10} RPY(deg) {rpy_str} | joints(deg) {joints_str}"
    )


def run_fk_validations(logger, entries):
    for label, q, T, attempts, ok in entries:
        if ok and q is not None:
            report_fk_pose(logger, label, q, T, attempts)
        else:
            logger.error(f"[FK] {label}: skipping (IK unavailable).")
