import numpy as np

from kinematics_utils import FK_all_links


def setup_workspace():
    """Return default workspace obstacles, margin, and safe height."""
    container_center = np.array([0.10, 0.125, 0.10])
    container_half_extents = np.array([0.025, 0.10, 0.10])
    obstacles = [{
        'name': 'container',
        'min': container_center - container_half_extents,
        'max': container_center + container_half_extents,
        'center': container_center,
        'half': container_half_extents
    }]
    collision_margin = 0.015
    safe_height = 0.25
    return obstacles, collision_margin, safe_height


def point_in_aabb(p, center, half, margin):
    return np.all(np.abs(p - center) <= (half + margin))


def is_pose_in_collision(T, obstacles, margin):
    pos = np.array(T[:3, 3], dtype=float)
    for obs in obstacles:
        min_pt = obs['min'] - margin
        max_pt = obs['max'] + margin
        if np.all(pos >= min_pt) and np.all(pos <= max_pt):
            return True, obs['name']
    return False, ''


def is_robot_in_collision(thetas, obstacles, margin):
    T_list = FK_all_links(thetas)
    for obs in obstacles:
        center = obs['center']
        half = obs['half']
        for i in range(len(T_list) - 1):
            p0 = T_list[i][:3, 3]
            p1 = T_list[i + 1][:3, 3]
            for alpha in np.linspace(0.0, 1.0, 8):
                p = (1.0 - alpha) * p0 + alpha * p1
                if point_in_aabb(p, center, half, margin):
                    return True, i
    return False, -1
