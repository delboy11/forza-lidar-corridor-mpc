#!/bin/bash
# Start the simulator, (optionally) the race manager, and this repository's driver node (avoid_node.py).
# The perception / state machine / spline / pure-pursuit chain of the stack is not used: one driver node turns /scan
# (+ GT pose in GT mode) into /drive.
#
# usage: ./run_avoid.sh <module:Class> ['<json kwargs>'] [num_obstacles] [seed] [map]
#   GT-free obstacle MPC (LiDAR localization + CorridorMPC berlin_obs):
#     ./run_avoid.sh drivers.lidar_loc:LidarLocDriver \
#       '{"init": "start", "forbid_gt": true, "inner": "drivers.cmpc_berlin_obs:CorridorMPC", "profile": "profiles/berlin_obs.json"}' 4 7 berlin
#   same MPC on the simulator's GT pose:
#     ./run_avoid.sh drivers.cmpc_berlin_obs:CorridorMPC "$(cat profiles/berlin_obs.json)" 4 7 berlin
# stop: Ctrl-C here, or `kill -TERM <pid of this script>` from another terminal (cleans up only what this script started;
#   a script started in the background ignores SIGINT, so use -TERM there).
# On a shared PC: export ROS_DOMAIN_ID=<n> first.
#
# Assumed layout (see README): race stack in ~/forza_ws/race_stack (ROS 2 Humble, built), maps in
# ~/forza_ws/race_stack/stack_master/maps/<map>/ with <map>.yaml / .png and global_waypoints.json.
# The race manager (random obstacles + lap scorer) is optional: without it the car drives without obstacles and nothing
# is scored.
SPEC=${1:?driver module:Class}
KW=${2:-'{}'}
NOBS=${3:-4}
SEED=${4:--1}
MAP=${5:-berlin}
HERE=$(cd "$(dirname "$0")" && pwd)
STACK=~/forza_ws/race_stack          # the drivers read their maps from this path too (drivers/*.py: MAPS / MAPDIR)
MAPS=$STACK/stack_master/maps
[ -f "$MAPS/$MAP/global_waypoints.json" ] || { echo "no map $MAPS/$MAP (global_waypoints.json)"; exit 1; }

# The driver reads the map and raceline itself: pass map_name when the driver accepts it.
MODFILE=$HERE/$(echo "${SPEC%%:*}" | tr . /).py
if ! grep -q '"map_name"' <<< "$KW" && grep -q "map_name" "$MODFILE" 2>/dev/null; then
  KW=$(python3 -c 'import json, sys; d = json.loads(sys.argv[1]); d["map_name"] = sys.argv[2]; print(json.dumps(d))' "$KW" "$MAP")
  echo "driver kwargs + map_name=$MAP -> $KW"
fi
read SX SY QZ QW < <(python3 -c "
import json, math; w = json.load(open('$MAPS/$MAP/global_waypoints.json'))['global_traj_wpnts_iqp']['wpnts'][0]
print(w['x_m'], w['y_m'], math.sin(w['psi_rad'] / 2), math.cos(w['psi_rad'] / 2))")
cd "$STACK"
source /opt/ros/humble/setup.bash
source install/setup.bash
LOG=~/race_logs; mkdir -p $LOG

# refuse to start next to another stack / driver node of this user (two controllers publishing /drive)
GUARD="ros2 launch stack_master|avoid_node.py|controller_manager_cpp|sqp_avoidance_node"
if pgrep -u "$(id -u)" -f "$GUARD" > /dev/null; then
  echo "another stack / driver node is still running, stop it first:"; pgrep -u "$(id -u)" -af "$GUARD"; exit 1
fi

# every background job gets its own process group (set -m): background jobs of a non-interactive script start with
# SIGINT ignored, so launch's children would be orphaned unless the whole group is signalled
set -m
cleanup() {
  trap - EXIT INT TERM
  local groups=$(jobs -p)
  for g in $groups; do kill -INT -- -$g 2>/dev/null; done
  for i in $(seq 15); do
    alive=0; for g in $groups; do kill -0 -- -$g 2>/dev/null && alive=1; done
    [ $alive -eq 0 ] && break; sleep 1
  done
  for g in $groups; do kill -TERM -- -$g 2>/dev/null; done; sleep 2
  for g in $groups; do kill -KILL -- -$g 2>/dev/null; done
  exit 0
}
trap cleanup EXIT INT TERM

wait_for() {  # wait_for <pattern> <file> <timeout_s> <pid>
  local t=0
  until grep -qE "$1" "$2" 2>/dev/null; do
    sleep 1; t=$((t + 1))
    if [ $t -ge $3 ] || ! kill -0 $4 2>/dev/null; then echo "start failed: '$1' not in $2 after $t s"; tail -20 "$2"; exit 1; fi
  done
}

ros2 launch stack_master base_system_launch.xml map_name:=$MAP sim:=true pose_source:=gt > $LOG/av_base.log 2>&1 &
wait_for "Managed nodes are active" $LOG/av_base.log 120 $!
echo "[1/3] simulator ready ($MAP)"

if ros2 pkg prefix race_manager > /dev/null 2>&1; then
  ros2 launch race_manager race_manager_launch.xml map_name:=$MAP seed:=$SEED num_obstacles:=$NOBS > $LOG/av_race.log 2>&1 &
  wait_for "lap 0 \[" $LOG/av_race.log 60 $!
  echo "[2/3] race manager ready ($NOBS obstacles per lap, seed $SEED)"
else
  echo "[2/3] no race_manager package: driving without obstacles and without lap scoring"
  : > $LOG/av_race.log
fi

# operator start pose (the GT-free driver takes it as its one initial pose)
ros2 topic pub --once -w 2 /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "{header: {frame_id: map}, pose: {pose: {position: {x: $SX, y: $SY, z: 0.0}, orientation: {z: $QZ, w: $QW}}, covariance: [0.25,0,0,0,0,0, 0,0.25,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0.068]}}" > /dev/null
(cd "$HERE" && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 exec python3 avoid_node.py "$SPEC" "$KW" 50 > $LOG/av_driver.log 2>&1) &
wait_for "avoid driver ready|Traceback" $LOG/av_driver.log 120 $!
grep -q Traceback $LOG/av_driver.log && { cat $LOG/av_driver.log; exit 1; }
echo "[3/3] driving with $SPEC (Ctrl-C to stop); driver log: $LOG/av_driver.log"
tail -n 0 -f $LOG/av_race.log | grep --line-buffered -E "  lap [0-9]+  best|collision #|obstacles —" &
wait
