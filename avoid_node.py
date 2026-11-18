"""ROS 2 wrapper for an offline-harness driver: /scan + /car_state/pose_GT + /car_state/odom_GT -> /drive.

The driver class is the same one evaluated in harness.py (act(scan, x, y, yaw, v, yaw_rate, t)),
so what was measured offline is what runs here.

ROS specifics the harness does not have:
- scan and pose arrive as separate messages. The bridge stamps scan, odom and pose of one sim
  tick with the same time, so each call uses the newest scan whose stamp has a matching pose and
  odom (a pose 10 ms off already turns wall hits into phantom obstacles), and t = that stamp.
- one call per new sim observation at `rate` Hz (50 Hz: what the drivers were tuned at).
- /initialpose (operator or scorer reset) -> driver.reset(); a driver with set_initial_pose(x, y, yaw) (e.g.
  drivers.lidar_loc:LidarLocDriver, which estimates its state from the scan) also gets the operator's pose, and the
  scans received before the reset are dropped (they may predate the teleport).
- a driver with gt_free == True (LidarLocDriver with forbid_gt=True) gets NaN instead of pose_GT / odom_GT and is called
  on every new scan without waiting for GT messages: nothing of the ground truth reaches it.
- an exception in act() publishes a stop instead of leaving the bridge on the last command.

usage: python3 avoid_node.py <module:Class> ['<json kwargs>'] [rate_hz]   (run from ~/avoid)
"""
import collections
import gc
import importlib
import json
import math
import sys
import time
import traceback

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


def key(stamp):
    return (stamp.sec, stamp.nanosec)


class AvoidNode(Node):
    def __init__(self, spec, kwargs, rate):
        super().__init__('avoid_driver')
        mod, cls = spec.split(':')
        self.driver = getattr(importlib.import_module(mod), cls)(**kwargs)
        self.driver.reset()
        self.has_init_pose = callable(getattr(self.driver, 'set_initial_pose', None))
        self.gt_free = bool(getattr(self.driver, 'gt_free', False))
        self.scans = collections.deque(maxlen=10)          # (key, ranges)
        self.poses = collections.OrderedDict()             # key -> (x, y, yaw)
        self.odoms = collections.OrderedDict()             # key -> (v, r)
        self.last_key = None
        self.failed = False
        self.ms, self.lat_ms, self.unmatched = [], [], 0
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 20)
        self.create_subscription(PoseStamped, '/car_state/pose_GT', self.pose_cb, 20)
        self.create_subscription(Odometry, '/car_state/odom_GT', self.odom_cb, 20)
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.reset_cb, 10)
        self.pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        # drivers that estimate their own state (LidarLocDriver) publish it for display / evaluation in RViz
        self.est_pub = (self.create_publisher(PoseStamped, '/lidar_loc/pose', 10)
                        if getattr(self.driver, 'loc', None) is not None else None)
        self.create_timer(1.0 / rate, self.step)
        self.create_timer(5.0, self.report)
        self.last_call = None
        self.max_gap = 0.0
        # A full garbage collection of everything built so far (ROS, numba, the driver) takes 70-85 ms: a stall that
        # long leaves the bridge on a stale command at a corner exit. Freeze the start-up heap so the collector never
        # walks it again (driving is bit-identical; work/berlin_tt/VERIFY.md F1).
        gc.collect()
        gc.freeze()
        self.get_logger().info(f'avoid driver ready: {spec} {kwargs} at {rate} Hz')

    @staticmethod
    def _push(d, k, v):
        d[k] = v
        while len(d) > 30:
            d.popitem(last=False)

    def scan_cb(self, m):
        self.scans.append((key(m.header.stamp), np.asarray(m.ranges, dtype=np.float64)))

    def pose_cb(self, m):
        q = m.pose.orientation
        self._push(self.poses, key(m.header.stamp), (m.pose.position.x, m.pose.position.y,
                   math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))))

    def odom_cb(self, m):
        self._push(self.odoms, key(m.header.stamp), (m.twist.twist.linear.x, m.twist.twist.angular.z))

    def reset_cb(self, m):
        self.driver.reset()
        self.last_key = None
        if self.has_init_pose:
            p, q = m.pose.pose.position, m.pose.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            self.driver.set_initial_pose(p.x, p.y, yaw)
            self.scans.clear()                   # taken before the teleport
            self.get_logger().info(f'initialpose -> set_initial_pose({p.x:.3f}, {p.y:.3f}, {yaw:.3f})')
        if self.failed:                          # a re-placed car gets another chance after an act() exception
            self.failed = False
            self.get_logger().warn('initialpose -> clearing the failed state')
        self.get_logger().info('initialpose -> driver.reset()')

    def publish(self, steer, speed):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.steering_angle = float(steer)
        msg.drive.speed = float(speed)
        self.pub.publish(msg)

    def step(self):
        if self.failed:
            self.publish(0.0, 0.0)
            return
        if self.gt_free:
            match = self.scans[-1] if self.scans else None
        else:
            match = next(((k, r) for k, r in reversed(self.scans) if k in self.poses and k in self.odoms), None)
        if match is None:
            if self.scans:
                self.unmatched += 1
            return
        k, ranges = match
        if k == self.last_key:                   # no new sim tick since the last call
            return
        self.last_key = k
        t = k[0] + k[1] * 1e-9
        t0 = time.perf_counter()
        if self.last_call is not None:
            self.max_gap = max(self.max_gap, t0 - self.last_call)
        self.last_call = t0
        try:
            if self.gt_free:
                steer, speed = self.driver.act(ranges, *(math.nan,) * 5, t)
            else:
                steer, speed = self.driver.act(ranges, *self.poses[k], *self.odoms[k], t)
        except Exception:
            self.failed = True
            self.get_logger().error('driver.act failed, stopping the car:\n' + traceback.format_exc())
            self.publish(0.0, 0.0)
            return
        self.publish(steer, speed)
        if self.est_pub is not None and getattr(self.driver.loc, 's', None) is not None:
            s = self.driver.loc.s
            m = PoseStamped()
            m.header.stamp.sec, m.header.stamp.nanosec = k
            m.header.frame_id = 'map'
            m.pose.position.x, m.pose.position.y = float(s[0]), float(s[1])
            m.pose.orientation.z, m.pose.orientation.w = math.sin(float(s[2]) / 2), math.cos(float(s[2]) / 2)
            self.est_pub.publish(m)
        self.ms.append((time.perf_counter() - t0) * 1e3)
        self.lat_ms.append((time.time() - t) * 1e3)   # bridge stamps with wall-clock time

    def report(self):
        if self.ms:
            v = self.odoms[next(reversed(self.odoms))][0] if self.odoms else 0.0
            if self.gt_free:
                v = float(getattr(self.driver, 'v_est', math.nan))
            self.get_logger().info(
                f'v={v:.2f} act {np.mean(self.ms):.1f} ms (p99 {np.percentile(self.ms, 99):.1f}), '
                f'obs->cmd {np.mean(self.lat_ms):.1f} ms (p99 {np.percentile(self.lat_ms, 99):.1f}, '
                f'max {np.max(self.lat_ms):.0f}), max call gap {self.max_gap * 1e3:.0f} ms, '
                f'{len(self.ms)} calls, {self.unmatched} unmatched')
            self.ms, self.lat_ms, self.unmatched, self.max_gap = [], [], 0, 0.0


def main():
    rclpy.init()
    spec = sys.argv[1]
    kwargs = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    rate = float(sys.argv[3]) if len(sys.argv) > 3 else 50.0
    rclpy.spin(AvoidNode(spec, kwargs, rate))
    rclpy.shutdown()


if __name__ == '__main__':
    main()
