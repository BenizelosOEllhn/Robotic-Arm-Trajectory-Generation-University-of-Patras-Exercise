from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import ColorRGBA
from scipy.spatial.transform import Rotation as R_scipy


def _quat_from_matrix(mat):
    return R_scipy.from_matrix(mat).as_quat()


def create_marker(id, scale, color, pos, rot, frame="base"):
    marker = Marker()
    marker.header.frame_id = frame
    marker.id = id
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.scale = Vector3(x=scale[0], y=scale[1], z=scale[2])
    marker.color = ColorRGBA(
        r=float(color[0]),
        g=float(color[1]),
        b=float(color[2]),
        a=float(color[3])
    )
    marker.pose.position = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
    quat = _quat_from_matrix(rot)
    marker.pose.orientation.x = float(quat[0])
    marker.pose.orientation.y = float(quat[1])
    marker.pose.orientation.z = float(quat[2])
    marker.pose.orientation.w = float(quat[3])
    return marker


def create_sphere_marker(id, pos, radius=0.01, color=(0.0, 1.0, 0.0, 1.0)):
    marker = Marker()
    marker.header.frame_id = "base"
    marker.id = id
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.scale = Vector3(x=radius, y=radius, z=radius)
    marker.color = ColorRGBA(
        r=float(color[0]),
        g=float(color[1]),
        b=float(color[2]),
        a=float(color[3])
    )
    marker.pose.position = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
    marker.pose.orientation.w = 1.0
    return marker


def create_trajectory_marker(id, points, color=(1.0, 1.0, 1.0, 1.0)):
    marker = Marker()
    marker.header.frame_id = "base"
    marker.id = id
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.003
    marker.color = ColorRGBA(
        r=float(color[0]),
        g=float(color[1]),
        b=float(color[2]),
        a=float(color[3])
    )
    marker.points = [
        Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
        for p in points
    ]
    return marker
