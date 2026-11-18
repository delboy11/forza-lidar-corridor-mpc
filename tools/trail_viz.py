"""RViz display of the path the car actually drove (simulator pose, display only), coloured by speed.

Subscribes /car_state/pose_GT and /car_state/odom_GT, keeps the last `keep_s` seconds (about one and a half laps) and
publishes /viz/trail (MarkerArray): a LINE_STRIP coloured blue (slow) -> green -> red (fast), plus a speed legend.
usage: python3 tools/trail_viz.py [keep_s]
"""
import collections
import sys

import rclpy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

V_MIN, V_MAX = 2.0, 10.0


def speed_color(v):
    u = min(max((v - V_MIN) / (V_MAX - V_MIN), 0.0), 1.0)
    if u < 0.5:                                   # blue -> green
        return ColorRGBA(r=0.0, g=2 * u, b=1.0 - 2 * u, a=1.0)
    return ColorRGBA(r=2 * (u - 0.5), g=1.0 - 2 * (u - 0.5), b=0.0, a=1.0)   # green -> red


class Trail(Node):
    def __init__(self, keep_s):
        super().__init__('trail_viz')
        self.v = 0.0
        self.pts = collections.deque(maxlen=int(keep_s * 25))
        self.last_t = None
        self.create_subscription(Odometry, '/car_state/odom_GT', lambda m: setattr(self, 'v', m.twist.twist.linear.x), 50)
        self.create_subscription(PoseStamped, '/car_state/pose_GT', self.pose_cb, 50)
        self.pub = self.create_publisher(MarkerArray, '/viz/trail', 10)
        self.create_timer(0.1, self.publish)

    def pose_cb(self, m):
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        if self.last_t is not None and t - self.last_t < 0.04:      # 25 Hz is plenty
            return
        if self.pts and (abs(m.pose.position.x - self.pts[-1][0]) > 1.0 or abs(m.pose.position.y - self.pts[-1][1]) > 1.0):
            self.pts.clear()                                         # car was re-placed: start a new trail
        self.last_t = t
        self.pts.append((m.pose.position.x, m.pose.position.y, self.v))

    def publish(self):
        if len(self.pts) < 2:
            return
        line = Marker()
        line.header.frame_id = 'map'; line.ns = 'trail'; line.id = 0; line.type = Marker.LINE_STRIP
        line.pose.orientation.w = 1.0
        line.scale.x = 0.07
        line.points = [Point(x=x, y=y, z=0.05) for x, y, _ in self.pts]
        line.colors = [speed_color(v) for _, _, v in self.pts]
        txt = Marker()
        txt.header.frame_id = 'map'; txt.ns = 'legend'; txt.id = 1; txt.type = Marker.TEXT_VIEW_FACING
        txt.pose.position.x, txt.pose.position.y, txt.pose.position.z = -4.0, 1.2, 0.5
        txt.pose.orientation.w = 1.0
        txt.scale.z = 0.4
        txt.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        txt.text = f'driven path: blue {V_MIN:.0f} m/s -> green -> red {V_MAX:.0f} m/s'
        self.pub.publish(MarkerArray(markers=[line, txt]))


def main():
    rclpy.init()
    rclpy.spin(Trail(float(sys.argv[1]) if len(sys.argv) > 1 else 14.0))


if __name__ == '__main__':
    main()
