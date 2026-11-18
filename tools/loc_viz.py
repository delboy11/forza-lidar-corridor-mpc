"""RViz display of the LiDAR localization error (evaluation only: the driver never sees the GT used here).

Subscribes /lidar_loc/pose (the driver's own estimate, published by avoid_node) and /car_state/pose_GT (simulator
ground truth), pairs them by header stamp and publishes /lidar_loc/viz (MarkerArray):
  - the estimated pose as an arrow (green),
  - the error vector from GT to estimate, magnified x100 (orange line),
  - a text above the car: current position / yaw error and running mean / max.
usage: python3 tools/loc_viz.py
"""
import collections
import math

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

MAG = 100.0


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class LocViz(Node):
    def __init__(self):
        super().__init__('loc_viz')
        self.gt = collections.OrderedDict()
        self.err = collections.deque(maxlen=2500)          # last ~50 s
        self.create_subscription(PoseStamped, '/car_state/pose_GT', self.gt_cb, 50)
        self.create_subscription(PoseStamped, '/lidar_loc/pose', self.est_cb, 50)
        self.pub = self.create_publisher(MarkerArray, '/lidar_loc/viz', 10)

    def gt_cb(self, m):
        k = (m.header.stamp.sec, m.header.stamp.nanosec)
        self.gt[k] = (m.pose.position.x, m.pose.position.y, yaw_of(m.pose.orientation))
        while len(self.gt) > 200:
            self.gt.popitem(last=False)

    def est_cb(self, m):
        k = (m.header.stamp.sec, m.header.stamp.nanosec)
        g = self.gt.get(k)
        if g is None:
            return
        ex, ey, eyaw = m.pose.position.x, m.pose.position.y, yaw_of(m.pose.orientation)
        dpos = math.hypot(ex - g[0], ey - g[1])
        dyaw = abs(math.atan2(math.sin(eyaw - g[2]), math.cos(eyaw - g[2])))
        self.err.append((dpos, dyaw))
        e = np.array(self.err)
        arr = MarkerArray()
        base = dict(frame_id='map')
        a = Marker(); a.header.frame_id = 'map'; a.ns = 'est'; a.id = 0; a.type = Marker.ARROW
        a.pose = m.pose; a.pose.position.z = 0.15
        a.scale.x, a.scale.y, a.scale.z = 0.45, 0.07, 0.07
        a.color.r, a.color.g, a.color.b, a.color.a = 0.1, 0.9, 0.2, 0.9
        l = Marker(); l.header.frame_id = 'map'; l.ns = 'err_x100'; l.id = 1; l.type = Marker.LINE_STRIP
        l.scale.x = 0.04; l.color.r, l.color.g, l.color.b, l.color.a = 1.0, 0.55, 0.0, 1.0
        l.pose.orientation.w = 1.0
        l.points = [Point(x=g[0], y=g[1], z=0.2), Point(x=g[0] + MAG * (ex - g[0]), y=g[1] + MAG * (ey - g[1]), z=0.2)]
        t = Marker(); t.header.frame_id = 'map'; t.ns = 'text'; t.id = 2; t.type = Marker.TEXT_VIEW_FACING
        t.pose.position.x, t.pose.position.y, t.pose.position.z = g[0], g[1] + 0.9, 0.5
        t.pose.orientation.w = 1.0
        t.scale.z = 0.38
        t.color.r, t.color.g, t.color.b, t.color.a = 1.0, 0.25, 0.0, 1.0
        t.text = (f'LiDAR loc err {dpos * 1000:5.1f} mm  yaw {math.degrees(dyaw):4.2f} deg\n'
                  f'mean {e[:, 0].mean() * 1000:4.1f} mm  max {e[:, 0].max() * 1000:4.1f} mm  (last {len(e)})\n'
                  f'orange line = error x{int(MAG)}')
        arr.markers = [a, l, t]
        self.pub.publish(arr)


def main():
    rclpy.init()
    rclpy.spin(LocViz())


if __name__ == '__main__':
    main()
