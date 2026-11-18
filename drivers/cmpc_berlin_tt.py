"""Design C "corridor_mpc", version 2 for the berlin no-obstacle time trial (work/berlin_tt/COMBINE.md).

cmpc_berlin_tt = cmpc_berlin_line (65088a80) = corridor_mpc_v2 (808cc1ba) + kwarg raceline_file + kwarg lat_alat:
  lat_alat = [t0, t1, a1] (None = v2): delay-adaptive lateral acceleration.  The online command-latency estimate
  t_lat (v2's multiple-model lat_est, 10 ms steps) scales a_lat and a_lat_hard linearly from their values at
  t_lat <= t0 down to a1 (a_lat_hard by the same ratio) at t_lat >= t1.  a_lat 14 is clean up to ~60 ms command delay
  on berlin but hit the bottom-hairpin exit at 65-75 ms; a_lat 12 is clean there (COMBINE.md).  At ROS-like timing
  (estimate 0.02-0.04 s) nothing changes: bit-identical to lat_alat=None.
The time trial is a kwargs profile:
profiles/berlin_tt.json (fast) and profiles/berlin_tt_safe.json (more margin), both with the reference line
maps_ext/berlin/tt/mtr_def_0.2.json (md5 f18868f4).  Neither profile relaxes the sight / stop envelopes
(v_vis_safe stays 2.5): the driver stays fully obstacle-aware.  Two stall-compensation code options were tried and dropped (no gain
under 100 ms stalls; COMBINE.md, work/berlin_tt/comb/cmpc_berlin_tt_stallexp.py).

cmpc_berlin_line header:
Design C "corridor_mpc", version 2 + berlin reference-line option (work/berlin/raceline_REPORT.md).

cmpc_berlin_line = corridor_mpc_v2 (808cc1ba) + kwarg raceline_file: a reference line shipped with the driver
(e.g. maps_ext/berlin/*.json) replaces the map raceline inside the driver only (DP deviation cost, MPC reference,
Frenet frame, wall tables, horizon end speed).  Its s = 0 is re-anchored on the map raceline's finish line
(s = 0 of global_traj_wpnts_iqp), so the lap-scoped obstacle memory is still cleared where the race manager
redraws the layout.  raceline_file=None (default) is exactly v2.

v2 header:
Design C "corridor_mpc", version 2 (robustness round, work/rob2/INTEGRATION.md).

v2 = the original (drivers/corridor_mpc_v1.py) + the five rob2 workstreams, merged by hand in this order: review
(latent-bug fixes: Frenet re-acquisition, monotonic clock, long gaps, non-finite inputs, pose jumps, solver failures),
compute (numba kernels, same numerics), timing (online latency estimate, measured call period, hold compensation),
model (online tyre-stiffness scale, adaptive measurement filters) and maps (raceline repair, geometric curvature
limit in the DP, speed-dependent wall clearance).  Each round is described below.

Pipeline per act():
  1. state estimation / delay compensation: the steering servo (3.2 rad/s, command latency + 2-step
     buffer) and the speed P-controller are simulated from the commands this driver issued; the CG
     sideslip (not measured) is estimated with the single-track model driven by the measured yaw rate
     and corrected by the course angle of successive poses; the state is predicted with the nonlinear
     single-track model to the time the new steering command bites.
  2. perception: scan hits -> map frame; hits farther than `obs_edt` from any occupied map cell
     (distance transform of hyhall12_orig) are obstacle cells.  Lap-scoped memory (cleared when s wraps,
     with a short blind window for the stale scan) and free-space carving of cells the beams see through.
     Cells are clustered; each cluster becomes a Frenet box, completed for the hidden back side.
  3. DP over a Frenet grid picks one consistent side for every visible obstacle (all of them, not only
     the closest), with a side-flip penalty (hysteresis) that becomes almost hard close to the obstacle.
     The chosen sides define an exact corridor lb(s) <= d <= ub(s) for the body line (obstacle and wall
     bounds already inflated by the collision-box half width + margin).
  4. spatial LTV-MPC (OSQP) on the simulator's own linear-tyre single-track model in Frenet coordinates:
     states d, mu (yaw rel. raceline), beta, r, delta; input steering rate (|.| <= 3.2 rad/s); exact
     discretisation per station along the planned speed profile, load transfer from the planned
     acceleration.  Corridor constraints on three points of the collision box (soft, heavy penalty),
     soft lateral-acceleration limit |v r| <= a_lat_hard.
  5. speed profile on the MPC path: lateral-acceleration limit, planned braking only within the
     steerable (yaw-stable) deceleration, plus an emergency envelope: stop (with the larger, still stable
     deceleration) before a closed corridor, and reach the first path point the scan cannot see no faster
     than v_vis_safe.  Emergency brake (fast jerk) when the MPC needs corridor slack near the car.

Changes of the second development round (see work/corridor_mpc/):
  * acc_model: the MPC's load-transfer model uses the acceleration the jerk-limited speed command will really
    produce (_cmd_sim), not the bang-bang station-to-station acceleration of the speed profile.  That mismatch
    made the predicted yaw rate 30-45 % wrong in corner entries; fixing it gave -0.3 s per lap at lower p99 |v r|.
  * occl_fix: lateral edges of a cluster are told apart by beam order; an edge hidden behind something closer is
    extended to the largest footprint until it is seen free (a single visible cell used to be extended the wrong
    way, the hidden side of the second obstacle of a chicane was missed).
  * min_gap_w: an obstacle side whose corridor to the wall is narrower than 6 cm is not offered to the DP (such
    gaps open and close with every 2 cm of the estimate and caused late side flips); fallback without the rule.
  * em_jerk: emergency braking is not jerk limited.
  * resolve_da: when the new speed profile changes the near-term acceleration by > 2 m/s^2 (e.g. a swerve that
    starts braking now), the MPC is solved a second time with it (load-transfer transients, p99 |v r| outliers).
  * v_vis_safe: the first unseen path point must be reached at <= 2.5 m/s (not at standstill).
Optional, off by default (tested, no net gain): yaw_lim, gg, swerve_w, a_acc_launch, escape, vis_obs_stop.

Compute-time round (work/rob_compute/): act() spent most of its time in many small numpy calls (Python overhead,
slow with the cold caches of a busy machine), in HPIPM (generic blasfeo build, ~130-180 us per IPM iteration, two
solves when resolve_da fires) and, with noisy scans, in Python loops over the remembered cells / boxes and linear
searches over lap-long lists; nothing per call scaled with the map, but the slowest calls (MPC phase with a re-solve)
reached 30+ ms on a loaded machine.  With jit=True (default) the per-call numerics run in numba kernels (compiled /
loaded in __init__, one signature each) that repeat the numpy code operation by operation (same results up to the
last bit of a few transcendental functions; jit=False is the unchanged reference code):
  * MPC data straight into HPIPM's buffers (_mpc_hp_k, structure-aware expm), speed profile (_speed_k), the plan's
    inputs (_plan_prep_k), DP grid / costs / back-tracking (_dp_k), actuator model loops (_obs_main_k, _obs_beta_k, _predict_k).
  * perception: scan hits (_hits_k); the remembered cells keep their map attributes and counts in arrays aligned
    with the sorted ids (_merge_k / _carve_k: no map-sized gathers, no sort-based set operations per call, no
    map-sized count grid to allocate in reset()); the cells' view of the scan is computed once per call (numpy,
    float32 as before) for carving and _boxes; selection, clustering and per-cluster statistics in _boxes_k.
  * lookups in box_conf (grows over the lap), prev_sides and seen_boxes are bucketed by s instead of linear scans.
  The QP is still solved by HPIPM with the same settings: the commands stay within ~3e-9 of the reference code (a
  few transcendental functions differ in the last bits).  Closed loop (16 seeds x 3 laps): nominal, lat50 and hz20
  episodes are identical to the reference code; long map episodes drift apart late (lap times +-0.1 s, chaotic
  small_hall more), failures per condition equal within +-2.  A custom IPM for this QP (~0.6 ms less per solve)
  was tried and left out: its ~1e-4 different solutions change the outcomes from the first lap on
  (work/rob2/compute_REPORT.md).
Result (interleaved open-loop replays of recorded episodes, same machine load): act() p99 hz20 16.6 -> 8.4 ms,
combo 24.7 -> 11.3, evans_esp 8.7 -> 4.5; mean 2-3x lower.

Timing robustness (work/rob2/timing_REPORT.md): the command latency and the call rate are not known in advance.
  * lat_est: the command latency is estimated online (multiple-model): the simulator's speed P-controller is driven
    by this driver's own speed commands delayed by each candidate latency (0..150 ms, 10 ms steps) and compared with
    the measured speed; the candidate with the smallest forgotten squared innovation is used for every delay model
    (servo/sideslip estimate, prediction to the bite time, speed at the controller).  t_lat is only the prior
    (with t_lat fixed at 0.03 lat65 had 0.44 failures per lap, 1.4 with the rest of these changes also off).
  * t_period: the call period is measured (median of recent gaps) and used where t_call was assumed: steering target
    = planned angle one period after the bite, jerk limit per call in _cmd_sim.
  * hold_comp: a speed command is held for one period; it is chosen so that the P-controller's mean acceleration over
    the hold equals the desired one (the plain inverse a / kp delivered 83 % of a braking request at 50 Hz, 65 % at
    20 Hz; without it hz20 had 0.33 failures per lap).
  * the MPC is also solved when its plan is older than 1.5 periods (a call after a stall), not only every second call.
  Not splitting DP / MPC at low call rates (tested: better lat_acc at 20-30 Hz, ~1.75x compute per call) is not used.

Model mismatch / noisy state inputs (work/rob2/model_REPORT.md):
  * c_adapt: online tyre-stiffness scale c (mu * C_S of the linear tyre: friction and stiffness act alike).  A grid of
    parallel single-track models (stiffness x extra command delay relative to the lat_est latency, _mm_prop) is driven
    by the commands this driver sent; the output error against the measured yaw rate (3 s memory) picks c.  c scales
    the model everywhere (sideslip observer, prediction, MPC, steerable-deceleration tables, DP curvature).  With the
    nominal model a 20 % weaker tyre (cs80) ran into outer walls.  (The model workstream's own use of the extra delay
    for the command timing, lat_adapt, is replaced by lat_est above.)
  * pose_filt: x, y, yaw, v, r are filtered against the model prediction with steady-state Kalman gains from the
    measured noise level (innovation correlation, _gain): exact inputs keep gain 1 (no lag), 2 cm / 0.01 rad pose
    noise gets ~0.1.  The course-angle sideslip correction is weighted by its noise (it amplified 2 cm position noise
    to ~0.1 rad sideslip noise: steering oscillation, p99 |v r| 11.2, and wall points became phantom obstacles).
    Pose jumps larger than pose_gate (localisation reset) are taken as they are, not smoothed in.

Unseen / narrow maps (work/rob2/maps_REPORT.md):
  * line repair: a raceline tighter than the car can steer at all (small_hall: kappa 5.5 against tan(delta_max)/WB =
    1.35) is replaced at start-up by the nearest line with |kappa| <= 1.0, a bounded curvature rate and wall clearance
    (sparse least squares).  The car used to stop and creep at every hairpin (19 s laps without obstacles, 8 s now).
    Only lines tighter than line_ktrig * tan(delta_max)/WB are replaced: hyhall12 and the other maps keep their own
    raceline (repairing a marginal one, glc_ot_ez with max kappa 1.07, cost more than it gained).
  * dp_kgeo: the DP charges slope changes that make the geometric path curvature kr / D + d'' / D^2 (D = 1 - kr d)
    exceed the steering limit; it used to bound d'' only, so the inside of a tight turn looked as easy as the outside.
  * wall_clear_v: the wall clearance grows with speed up to 4 m/s (tracking error in tight S-bends along the walls;
    the front corner touched walls the plan cleared by a few cm).
  * min_gap_keep: hysteresis of min_gap_w for the side already chosen (a narrow gap that shrank by 3 cm at 8 m/s
    forced a flip 4 m before the obstacle).
"""
import json
import math
import os
import sys

import numpy as np
import osqp
import scipy.sparse as sp
import yaml
from PIL import Image
from scipy import ndimage
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

MAPDIR = os.path.expanduser('~/forza_ws/race_stack/stack_master/maps/hyhall12')

# vehicle: the simulator's single-track model (sim_params.yaml of the race stack)
MU_, CSF_, CSR_, LF_, LR_, HCG_, G_ = 1.0489, 4.718, 5.4562, 0.15875, 0.17145, 0.074, 9.81
M_, I_ = 3.47, 0.04712
WB = LF_ + LR_
STEER_MAX = 0.4189
STEER_RATE = 3.2
KP_ACC, KP_BRK, A_MAX, V_SWITCH = 10.0 * 9.51 / 20.0, 10.0 * 9.51 / 5.0, 9.51, 7.319
LIDAR_X = 0.27
BOX_HW = 0.1016        # referee box half width
BOX_FRONT = 0.435      # referee box front, ahead of the CG
ANG_MIN = -2.35
ANG_INC = 4.7 / 1079
K_US0 = 0.002787


def k_us(a, c=1.0):
    """understeer gradient under longitudinal acceleration a (load transfer); c scales the tyre stiffness"""
    a = np.minimum(np.maximum(a, -A_MAX), A_MAX)
    return (LR_ / (MU_ * CSF_ * (G_ * LR_ - a * HCG_)) - LF_ / (MU_ * CSR_ * (G_ * LF_ + a * HCG_))) / c


def bicycle_s(v, a, c=1.0):
    """scalar version of bicycle()"""
    a = -A_MAX if a < -A_MAX else (A_MAX if a > A_MAX else a)
    cf = c * MU_ * CSF_ * (G_ * LR_ - a * HCG_)
    cr = c * MU_ * CSR_ * (G_ * LF_ + a * HCG_)
    vl = v * WB
    return (-(cr + cf) / vl, (cr * LR_ - cf * LF_) / (v * vl) - 1.0, cf / vl,
            M_ * (cr * LR_ - cf * LF_) / (I_ * WB), -M_ * (LF_ * LF_ * cf + LR_ * LR_ * cr) / (v * I_ * WB),
            M_ * LF_ * cf / (I_ * WB))


def bicycle(v, a, c=1.0):
    """linear single-track coefficients: beta' = a11 b + a12 r + b1 delta ; r' = a21 b + a22 r + b2 delta.
    c scales the cornering stiffness of both axles (mu * C_S of the linear tyre: friction and stiffness act alike)"""
    a = np.minimum(np.maximum(a, -A_MAX), A_MAX)
    cf = c * MU_ * CSF_ * (G_ * LR_ - a * HCG_)
    cr = c * MU_ * CSR_ * (G_ * LF_ + a * HCG_)
    a11 = -(cr + cf) / (v * WB)
    a12 = (cr * LR_ - cf * LF_) / (v * v * WB) - 1.0
    b1 = cf / (v * WB)
    a21 = M_ * (cr * LR_ - cf * LF_) / (I_ * WB)
    a22 = -M_ * (LF_ * LF_ * cf + LR_ * LR_ * cr) / (v * I_ * WB)
    b2 = M_ * LF_ * cf / (I_ * WB)
    return a11, a12, b1, a21, a22, b2


def p_accel(v_cmd, v):
    """the simulator's speed P-controller + acceleration constraints"""
    dv = v_cmd - v
    if v > 0.0:
        a = (KP_ACC if dv > 0 else KP_BRK) * dv
    else:
        a = (2.0 * 9.51 / 20.0 if dv > 0 else 2.0 * 9.51 / 5.0) * dv
    pos = A_MAX * V_SWITCH / v if v > V_SWITCH else A_MAX
    return max(-A_MAX, min(pos, a))


def expm_batch(M):
    """matrix exponential of a batch (n, k, k) by scaling and squaring + Taylor(10)"""
    nrm = np.max(np.sum(np.abs(M), axis=2))
    s = max(0, int(math.ceil(math.log2(max(nrm, 1e-12)))) + 1)
    A = M / (2.0 ** s)
    k = M.shape[1]
    E = np.broadcast_to(np.eye(k), M.shape).copy()
    T = E.copy()
    for j in range(1, 11):
        T = T @ A / j
        E = E + T
    for _ in range(s):
        E = E @ E
    return E


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


try:
    from numba import njit
except Exception:          # pragma: no cover - numba ships with the simulator
    def njit(*a, **k):
        return (lambda f: f) if not (len(a) == 1 and callable(a[0])) else a[0]


@njit(cache=True)
def _dp_core(cost0, node_cost, blocked, kv, cst, cv_cost, sl_cost, dlo, dhi, w_geo):
    """2nd-order DP: state (lateral cell j, slope k); returns back pointers, final cost, last reached station.
    dlo/dhi (M+1, J): slope change (in quanta) at the vertex (station, cell) that keeps the geometric curvature
    within the steering limit; a change outside costs w_geo per squared quantum of excess (soft: never blocks)"""
    M1, J = node_cost.shape
    K = kv.shape[0]
    M = M1 - 1
    back = np.zeros((M, J, K), np.int8)
    cost = cost0.copy()
    new = np.empty((J, K))
    reach = M
    cmax = (cv_cost.shape[0] - 1) // 2
    for i in range(1, M + 1):
        anyf = False
        c = cst[i]
        for j in range(J):
            if blocked[i, j]:
                for k in range(K):
                    new[j, k] = np.inf
                continue
            nc = node_cost[i, j]
            for k in range(K):
                jp = j - kv[k]
                best = np.inf
                bd = 0
                if jp >= 0 and jp < J:
                    glo = dlo[i - 1, jp]
                    ghi = dhi[i - 1, jp]
                    for dl in range(-c, c + 1):
                        kp = k - dl
                        if kp >= 0 and kp < K:
                            val = cost[jp, kp] + cv_cost[dl + cmax]
                            ex = glo - dl if glo - dl > dl - ghi else dl - ghi
                            if ex > 0.0:
                                val += w_geo * ex * ex
                            if val < best:
                                best = val
                                bd = dl + cmax
                if best < np.inf:
                    new[j, k] = best + nc + sl_cost[k]
                    back[i - 1, j, k] = bd
                    anyf = True
                else:
                    new[j, k] = np.inf
        if not anyf:
            reach = i - 1
            break
        tmp = cost
        cost = new
        new = tmp
    return back, cost, reach


@njit(cache=True)
def _passes(vlim, dpl, tab_plan, v0, a_acc, ka, a_lat, gg, gg_min):
    """backward pass with the speed dependent (steerable) deceleration; with gg > 0 the deceleration shrinks with
    the lateral usage (v^2 kappa / a_lat) of the station (brake before the corner, not in it)"""
    N1 = vlim.shape[0]
    v = vlim.copy()
    nt = tab_plan.shape[0]
    for i in range(N1 - 2, -1, -1):
        vn = v[i + 1]
        ab = tab_plan[min(int(vn * 20.0), nt - 1)]
        g1 = math.sqrt(vn * vn + 2.0 * ab * dpl[i])
        ab = tab_plan[min(int(g1 * 20.0), nt - 1)]
        if gg > 0.0:
            kk = ka[i] if ka[i] > ka[i + 1] else ka[i + 1]
            u = vn * vn * kk / a_lat
            f = 1.0 - gg * u * u
            if f < gg_min:
                f = gg_min
            ab = ab * math.sqrt(f)
        g1 = math.sqrt(vn * vn + 2.0 * ab * dpl[i])
        if g1 < v[i]:
            v[i] = g1
    return v


@njit(cache=True)
def _fwd(v, dpl, v0, a_acc, tab_emerg):
    N1 = v.shape[0]
    vf = v.copy()
    vf[0] = min(v[0], max(v0, 0.0))
    for i in range(N1 - 1):
        g = math.sqrt(vf[i] * vf[i] + 2.0 * a_acc * dpl[i])
        vf[i + 1] = min(v[i + 1], g)
    vpred = vf.copy()
    vb = v0
    nt = tab_emerg.shape[0]
    for i in range(N1):
        vpred[i] = max(vf[i], vb)
        if i < N1 - 1:
            ab = tab_emerg[min(int(max(vb, 0.0) * 20.0), nt - 1)]
            vb = math.sqrt(max(vb * vb - 2.0 * ab * dpl[i], 0.0))
    return vf, vpred


@njit(cache=True)
def _stable_accel_k(vpred, dpl, te, a_acc):
    v = vpred.copy()
    nt = te.shape[0]
    n = dpl.shape[0]
    acc = np.empty(n)
    for i in range(n):
        x = v[i] if v[i] > 0.0 else 0.0
        ab = te[min(int(x * 20.0), nt - 1)]
        vm = v[i] * v[i] - 2.0 * ab * dpl[i]
        vmin = math.sqrt(vm) if vm > 0.0 else 0.0
        if v[i + 1] < vmin:
            v[i + 1] = vmin
        a = (v[i + 1] * v[i + 1] - v[i] * v[i]) / (2.0 * dpl[i])
        if a < -ab:
            a = -ab
        if a > a_acc:
            a = a_acc
        acc[i] = a
    return v, acc


@njit(cache=True)
def _cmd_sim(vprof, cum, dpl, v0, a0, a_acc, jerk, tab_emerg, t_call):
    """speed and acceleration the car will actually have along the stations when act() keeps tracking vprof:
    look-ahead target speed, stable braking limit, jerk limit per call (t_call), P-controller ~ commanded accel"""
    N = dpl.shape[0]
    vs = np.empty(N + 1)
    acc = np.empty(N)
    nt = tab_emerg.shape[0]
    v = v0 if v0 > 0.0 else 0.0
    a = a0
    vs[0] = v
    for k in range(N):
        s_ahead = 0.1 * v
        if s_ahead < 0.3:
            s_ahead = 0.3
        sq = cum[k] + s_ahead
        # interpolate vprof at sq
        j = k
        while j < N and cum[j + 1] < sq:
            j += 1
        if j >= N:
            vt = vprof[N]
        else:
            f = (sq - cum[j]) / max(cum[j + 1] - cum[j], 1e-9)
            vt = vprof[j] * (1.0 - f) + vprof[j + 1] * f
        ad = (vt * vt - v * v) / (2.0 * s_ahead)
        ab = tab_emerg[min(int(v * 20.0), nt - 1)]
        if ad < -ab:
            ad = -ab
        if ad > a_acc:
            ad = a_acc
        vm = v if v > 0.5 else 0.5
        dt = dpl[k] / vm
        # the jerk limit acts per control call
        dj = jerk * (dt if dt > t_call else t_call)
        if ad > a + dj:
            ad = a + dj
        if ad < a - dj:
            ad = a - dj
        if v > 7.319 and ad > 9.51 * 7.319 / v:
            ad = 9.51 * 7.319 / v
        a = ad
        acc[k] = a
        v2 = v * v + 2.0 * a * dpl[k]
        v = math.sqrt(v2) if v2 > 0.0 else 0.0
        vs[k + 1] = v
    return vs, acc


@njit(cache=True)
def _lat_sim_k(Ti, Vc, Lc, vL, t0, n):
    """speed after n simulator steps (10 ms) from t0 for every latency hypothesis Lc: the P-controller (p_accel) is
    driven by the latest command issued (times Ti, speeds Vc, ascending) at least Lc before each step; ok = False when
    the command acting for the longest hypothesis is older than the history"""
    v = vL.copy()
    ok = True
    for k in range(n):
        tk = t0 + 0.01 * k + 1e-6
        for j in range(Lc.shape[0]):
            i = Ti.shape[0] - 1
            while i >= 0 and Ti[i] > tk - Lc[j]:
                i -= 1
            if i < 0:
                ok = False
                i = 0
            vj = v[j]
            dv = Vc[i] - vj
            if vj > 0.0:
                kp = KP_ACC if dv > 0 else KP_BRK
            else:
                kp = 0.951 if dv > 0 else 3.804
            a = kp * dv
            pos = A_MAX * V_SWITCH / vj if vj > V_SWITCH else A_MAX
            if a > pos:
                a = pos
            if a < -A_MAX:
                a = -A_MAX
            v[j] = vj + 0.01 * a
    return v, ok


@njit(cache=True)
def _cluster_k(xs, ys, link):
    """union-find single-linkage clustering of points (O(n^2), n is a few hundred)"""
    n = xs.shape[0]
    par = np.arange(n)
    l2 = link * link
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx * dx + dy * dy < l2:
                a = i
                while par[a] != a:
                    a = par[a]
                b = j
                while par[b] != b:
                    b = par[b]
                if a != b:
                    par[b] = a
    lab = np.empty(n, np.int64)
    for i in range(n):
        a = i
        while par[a] != a:
            a = par[a]
        lab[i] = a
    return lab


@njit(cache=True)
def _expm_k(M):
    """batched matrix exponential (scaling and squaring + Taylor 10) for small square matrices"""
    nb, k, _ = M.shape
    out = np.empty_like(M)
    A = np.empty((k, k))
    T = np.empty((k, k))
    T2 = np.empty((k, k))
    E = np.empty((k, k))
    for b in range(nb):
        nrm = 0.0
        for i in range(k):
            ssum = 0.0
            for j in range(k):
                ssum += abs(M[b, i, j])
            if ssum > nrm:
                nrm = ssum
        s = 0
        if nrm > 0.5:
            s = int(math.ceil(math.log(nrm / 0.5) / math.log(2.0)))
        sc = 1.0 / (2.0 ** s)
        for i in range(k):
            for j in range(k):
                A[i, j] = M[b, i, j] * sc
                T[i, j] = A[i, j]
                E[i, j] = A[i, j] + (1.0 if i == j else 0.0)
        for p in range(2, 11):
            for i in range(k):
                for j in range(k):
                    acc = 0.0
                    for q in range(k):
                        acc += T[i, q] * A[q, j]
                    T2[i, j] = acc / p
            for i in range(k):
                for j in range(k):
                    T[i, j] = T2[i, j]
                    E[i, j] += T2[i, j]
        for _ in range(s):
            for i in range(k):
                for j in range(k):
                    acc = 0.0
                    for q in range(k):
                        acc += E[i, q] * E[q, j]
                    T2[i, j] = acc
            for i in range(k):
                for j in range(k):
                    E[i, j] = T2[i, j]
        for i in range(k):
            for j in range(k):
                out[b, i, j] = E[i, j]
    return out


@njit(cache=True)
def _expm_fr_k(M):
    """_expm_k for the augmented Frenet single-track Jacobian of _mpc_hp_k (bitwise the same result): rows 5, 6
    (input, affine term) are zero, so they stay identity rows, and of rows 0..4 only these entries are nonzero:
    (0,1) (0,2) (0,6) (1,0) (1,1) (1,2) (1,3) (1,6) (2,2) (2,3) (2,4) (3,2) (3,3) (3,4) (4,5).  The products skip
    only exact-zero terms and keep the summation order, ~4x fewer flops."""
    nb = M.shape[0]
    out = np.zeros((nb, 7, 7))
    A = np.zeros((7, 7))
    T = np.zeros((7, 7))
    T2 = np.zeros((7, 7))
    E = np.zeros((7, 7))
    for b in range(nb):
        nrm = 0.0
        for i in range(5):
            ssum = 0.0
            for j in range(7):
                ssum += abs(M[b, i, j])
            if ssum > nrm:
                nrm = ssum
        s = 0
        if nrm > 0.5:
            s = int(math.ceil(math.log(nrm / 0.5) / math.log(2.0)))
        sc = 1.0 / (2.0 ** s)
        for i in range(7):
            for j in range(7):
                A[i, j] = M[b, i, j] * sc
                T[i, j] = A[i, j]
                E[i, j] = A[i, j] + (1.0 if i == j else 0.0)
        a01, a02, a06 = A[0, 1], A[0, 2], A[0, 6]
        a10, a11, a12, a13, a16 = A[1, 0], A[1, 1], A[1, 2], A[1, 3], A[1, 6]
        a22, a23, a24, a32, a33, a34, a45 = A[2, 2], A[2, 3], A[2, 4], A[3, 2], A[3, 3], A[3, 4], A[4, 5]
        for p in range(2, 11):
            for i in range(5):
                t0, t1, t2, t3, t4 = T[i, 0], T[i, 1], T[i, 2], T[i, 3], T[i, 4]
                T2[i, 0] = (t1 * a10) / p
                T2[i, 1] = (t0 * a01 + t1 * a11) / p
                T2[i, 2] = (t0 * a02 + t1 * a12 + t2 * a22 + t3 * a32) / p
                T2[i, 3] = (t1 * a13 + t2 * a23 + t3 * a33) / p
                T2[i, 4] = (t2 * a24 + t3 * a34) / p
                T2[i, 5] = (t4 * a45) / p
                T2[i, 6] = (t0 * a06 + t1 * a16) / p
            for i in range(5):
                for j in range(7):
                    T[i, j] = T2[i, j]
                    E[i, j] += T2[i, j]
        for _ in range(s):
            for i in range(5):
                for j in range(7):
                    acc = 0.0
                    for q in range(5):
                        acc += E[i, q] * E[q, j]
                    if j == 5:
                        acc += E[i, 5]
                    elif j == 6:
                        acc += E[i, 6]
                    T2[i, j] = acc
            for i in range(5):
                for j in range(7):
                    E[i, j] = T2[i, j]
        for i in range(7):
            for j in range(7):
                out[b, i, j] = E[i, j]
    return out


@njit(cache=True)
def _corr1(s, wlo, whi, nf, fds, bx, bsd, w_h):
    """corridor [lo, hi] of the body line at station s (scalar _corridor: wall bounds + chosen obstacle sides)"""
    i = int(np.rint(s / fds)) % nf
    lo = wlo[i]
    hi = whi[i]
    for b in range(bx.shape[0]):
        if bsd[b] == 0 or s < bx[b, 0] or s > bx[b, 1]:
            continue
        if bsd[b] > 0:
            v = bx[b, 3] + w_h
            if v > lo:
                lo = v
        else:
            v = bx[b, 2] - w_h
            if v < hi:
                hi = v
    return lo, hi


@njit(cache=True)
def _mpc_hp_k(st, X0, Xbar, kr, vpred, acc, dpl, dpath, lpts, wlo, whi, nf, fds, bx, bsd, w_h, v_min_plan, c_pref,
              q_d, q_mu, q_a, a_lat_hard, tab_plan, bA, bB, bb, bQ, bq, bC, blg, bug, blbx, bubx, cmu, lgc, ugc, dt,
              c_ty):
    """CorridorMPC._mpc + _solve_hpipm data for HPIPM in one pass (same formulas, written straight into the solver's
    persistent column-major buffers): exact discretisation (expm) of the linearised Frenet single-track model per
    station, corridor rows of the collision-box points, |v r| row, tracking cost.  cmu/lgc/ugc/dt are returned for
    the slack evaluation and the steering rate."""
    N = dpl.shape[0]
    nl = lpts.shape[0]
    vp = np.empty(N + 1)
    for k in range(N + 1):
        vp[k] = vpred[k] if vpred[k] > v_min_plan else v_min_plan
    # ---- dynamics: continuous Jacobian (augmented with input and affine term) * dt -> expm
    M = np.zeros((N, 7, 7))
    for k in range(N):
        vi = vp[k]
        a = acc[k]
        a = -A_MAX if a < -A_MAX else (A_MAX if a > A_MAX else a)
        cf = c_ty * MU_ * CSF_ * (G_ * LR_ - a * HCG_)
        cr = c_ty * MU_ * CSR_ * (G_ * LF_ + a * HCG_)
        a11 = -(cr + cf) / (vi * WB)
        a12 = (cr * LR_ - cf * LF_) / (vi * vi * WB) - 1.0
        b1 = cf / (vi * WB)
        a21 = M_ * (cr * LR_ - cf * LF_) / (I_ * WB)
        a22 = -M_ * (LF_ * LF_ * cf + LR_ * LR_ * cr) / (vi * I_ * WB)
        b2 = M_ * LF_ * cf / (I_ * WB)
        dbar = Xbar[k, 0]
        chib = Xbar[k, 1] + Xbar[k, 2]
        cb = math.cos(chib)
        sb = math.sin(chib)
        Dk = 1.0 - kr[k] * dbar
        if Dk < 0.2:
            Dk = 0.2
        krr = kr[k]
        cd = -krr * krr * vi * cb / Dk ** 2
        cm = krr * vi * sb / Dk
        dt[k] = dpl[k] / vi
        h = dt[k]
        M[k, 0, 1] = vi * cb * h
        M[k, 0, 2] = vi * cb * h
        M[k, 0, 6] = vi * (sb - cb * chib) * h
        M[k, 1, 0] = cd * h
        M[k, 1, 1] = cm * h
        M[k, 1, 2] = cm * h
        M[k, 1, 3] = h
        M[k, 1, 6] = (-krr * vi * cb / Dk - cd * dbar - cm * chib) * h
        M[k, 2, 2] = a11 * h
        M[k, 2, 3] = a12 * h
        M[k, 2, 4] = b1 * h
        M[k, 3, 2] = a21 * h
        M[k, 3, 3] = a22 * h
        M[k, 3, 4] = b2 * h
        M[k, 4, 5] = h
    E = _expm_fr_k(M)
    for k in range(N):
        for i in range(5):
            for j in range(5):
                bA[k, i, j] = E[k, j, i]
            bB[k, 0, i] = E[k, i, 5]
            bb[k, i] = E[k, i, 6]
    # ---- reference: DP path pushed c_pref inside the corridor of the box's rear axle and front
    v0_ = vp[0]
    x = v0_ if v0_ > 0.0 else 0.0
    ab = tab_plan[min(int(x * 20.0), tab_plan.shape[0] - 1)]
    cum = 0.0
    for k in range(N + 1):
        if k > 0:
            cum += dpl[k - 1]
        la, ha = _corr1(st[k], wlo, whi, nf, fds, bx, bsd, w_h)
        lb, hb = _corr1(st[k] + BOX_FRONT, wlo, whi, nf, fds, bx, bsd, w_h)
        loc = la if la > lb else lb
        hic = ha if ha < hb else hb
        dr = dpath[k]
        if dr < loc + c_pref:
            dr = loc + c_pref
        if dr > hic - c_pref:
            dr = hic - c_pref
        if hic - loc < 2 * c_pref:
            dr = 0.5 * (loc + hic)
        vsq = v0_ * v0_ - 2.0 * ab * cum
        vreach = math.sqrt(vsq if vsq > 1.0 else 1.0)
        vlat = vp[k] if vp[k] < vreach else vreach
        wa = q_a * (vp[k] * vp[k])
        for i in range(5):
            for j in range(5):
                bQ[k, i, j] = 0.0
            bq[k, i] = 0.0
            for j in range(nl + 1):
                bC[k, i, j] = 0.0
        if k == 0:
            for i in range(5):
                bQ[0, i, i] = 2e-6
            for j in range(nl + 1):
                blg[0, j] = 0.0
                bug[0, j] = 0.0
            for i in range(5):
                blbx[0, i] = X0[i]
                bubx[0, i] = X0[i]
            continue
        bQ[k, 0, 0] = 2 * q_d
        bQ[k, 1, 1] = 2 * q_mu
        bQ[k, 2, 2] = 2e-6
        bQ[k, 3, 3] = 2 * wa
        bQ[k, 4, 4] = 2e-6
        bq[k, 0] = -2 * q_d * dr
        bq[k, 3] = -2 * wa * vp[k] * kr[k]
        # corridor rows of the collision-box points (d + lp cos(mu) mu, linearised at Xbar)
        mb = Xbar[k, 1]
        cmy = math.cos(mb)
        smy = math.sin(mb)
        Dk = 1.0 - kr[k] * Xbar[k, 0]
        if Dk < 0.2:
            Dk = 0.2
        keff = kr[k] / Dk
        for j in range(nl):
            lp = lpts[j]
            lo, hi = _corr1(st[k] + lp * cmy / Dk, wlo, whi, nf, fds, bx, bsd, w_h)
            const = lp * (smy - cmy * mb) - keff * lp * lp / 2
            lo = lo - const
            hi = hi - const
            if lo > hi:          # closed corridor: keep the problem well posed (slack does the rest)
                lo = hi = 0.5 * (lo + hi)
            cmu[k - 1, j] = lp * cmy
            lgc[k - 1, j] = lo
            ugc[k - 1, j] = hi
            bC[k, 0, j] = 1.0
            bC[k, 1, j] = lp * cmy
            blg[k, j] = lo
            bug[k, j] = hi
        bC[k, 3, nl] = vlat
        blg[k, nl] = -a_lat_hard
        bug[k, nl] = a_lat_hard
        blbx[k, 0] = -STEER_MAX * 0.98
        bubx[k, 0] = STEER_MAX * 0.98


@njit(cache=True)
def _hits_k(scan, lx, ly, cy, sy, ca, sa, r_max, ox, oy, res, W, H, edt, obs_edt, mres, MW):
    """memory-grid cells (sorted, unique) of the scan hits farther than obs_edt from any occupied map cell (the
    first part of CorridorMPC._perceive, same arithmetic; edt is float32 and compared in float32 like numpy)"""
    n = scan.shape[0]
    out = np.empty(n, np.int64)
    m = 0
    thr = np.float32(obs_edt)
    for i in range(n):
        r = scan[i]
        if not (r > 0.05 and r < r_max):
            continue
        ux = cy * ca[i] - sy * sa[i]
        uy = sy * ca[i] + cy * sa[i]
        px = lx + r * ux
        py = ly + r * uy
        col = int(math.floor((px - ox) / res))
        row = int(math.floor((py - oy) / res))
        if col < 0 or col >= W or row < 0 or row >= H:
            continue
        if not (edt[row, col] > thr):
            continue
        mc = int(math.floor((px - ox) / mres))
        mr = int(math.floor((py - oy) / mres))
        out[m] = mr * MW + mc
        m += 1
    srt = np.sort(out[:m])
    k = 0
    for i in range(m):
        if i == 0 or srt[i] != srt[i - 1]:
            srt[k] = srt[i]
            k += 1
    return srt[:k].copy()


@njit(cache=True)
def _merge_k(active, amem, ax, ay, a_s, ad, aok, cells, cell_x, cell_y, cell_s, cell_d, cell_ok):
    """add the new obstacle cells (sorted, unique) to the remembered set (sorted ids + per-cell arrays aligned with
    it: memory count, x, y, s, d, ok): +2 (max 12) for cells already remembered, 2 for new ones (their grid count
    was 0), the static per-cell attributes are copied once when a cell enters"""
    na, nc = active.shape[0], cells.shape[0]
    n = na + nc
    o_id = np.empty(n, np.int64)
    o_mem = np.empty(n, np.int16)
    o_x = np.empty(n, np.float32)
    o_y = np.empty(n, np.float32)
    o_s = np.empty(n, np.float32)
    o_d = np.empty(n, np.float32)
    o_ok = np.empty(n, np.bool_)
    i = j = k = 0
    while i < na or j < nc:
        if j >= nc or (i < na and active[i] < cells[j]):
            o_id[k], o_mem[k], o_x[k], o_y[k], o_s[k], o_d[k], o_ok[k] = active[i], amem[i], ax[i], ay[i], a_s[i], ad[i], aok[i]
            i += 1
        elif i < na and active[i] == cells[j]:
            o_id[k], o_x[k], o_y[k], o_s[k], o_d[k], o_ok[k] = active[i], ax[i], ay[i], a_s[i], ad[i], aok[i]
            o_mem[k] = min(amem[i] + 2, 12)
            i += 1
            j += 1
        else:
            c = cells[j]
            o_id[k], o_mem[k], o_x[k], o_y[k], o_s[k], o_d[k], o_ok[k] = c, 2, cell_x[c], cell_y[c], cell_s[c], cell_d[c], cell_ok[c]
            j += 1
        k += 1
    return o_id[:k].copy(), o_mem[:k].copy(), o_x[:k].copy(), o_y[:k].copy(), o_s[:k].copy(), o_d[:k].copy(), o_ok[:k].copy()


@njit(cache=True)
def _carve_k(amem, rng, b, scan):
    """free-space carving of CorridorMPC._perceive on the remembered cells: a cell 0.4..7 m away whose 5 beams
    around its bearing all return >= 0.2 m behind it loses one count; returns the mask of cells still remembered.
    rng is float32 and compared like numpy does (float32 constants, float32 rng + 0.2)"""
    n = amem.shape[0]
    keep = np.ones(n, np.bool_)
    lo, hi, mg = np.float32(0.4), np.float32(7.0), np.float32(0.2)
    for i in range(n):
        bi = b[i]
        if bi < 2 or bi > 1077 or not (rng[i] > lo) or not (rng[i] < hi):
            continue
        mn = min(min(min(min(scan[bi - 2], scan[bi - 1]), scan[bi]), scan[bi + 1]), scan[bi + 2])
        if mn != mn or scan[bi - 2] != scan[bi - 2] or scan[bi - 1] != scan[bi - 1] or scan[bi] != scan[bi] \
                or scan[bi + 1] != scan[bi + 1] or scan[bi + 2] != scan[bi + 2]:
            continue                  # an invalid (NaN) beam: no evidence of free space (np.minimum gave NaN)
        if mn > np.float64(np.float32(rng[i] + mg)):
            amem[i] -= 1
            if amem[i] <= 0:
                keep[i] = False
    return keep


@njit(cache=True)
def _fmin2(a, b):
    """np.fmin of two scalars: a NaN is ignored unless both are NaN"""
    if a != a:
        return b
    if b != b:
        return a
    return a if a <= b else b


@njit(cache=True)
def _frenet_k(fx, fy, x, y, guess):
    """nearest raceline sample (first minimum) in the window guess-12 .. guess+39 (guess < 0: whole raceline)"""
    nf = fx.shape[0]
    best = np.inf
    bi = 0
    if guess < 0:
        for i in range(nf):
            dd = (fx[i] - x) ** 2 + (fy[i] - y) ** 2
            if dd < best:
                best = dd
                bi = i
    else:
        for c in range(guess - 12, guess + 40):
            i = c % nf
            dd = (fx[i] - x) ** 2 + (fy[i] - y) ** 2
            if dd < best:
                best = dd
                bi = i
    return bi


@njit(cache=True)
def _targets_k(H, tt, tl, tb):
    """CorridorMPC._targets on the command history array H (rows: t_issue, delta, v_cmd); latency tl, buffer tb"""
    hd, dl, hv, vc = False, 0.0, False, 0.0
    ts0 = tt + 1e-6 - tl - tb
    tv0 = tt + 1e-6 - tl
    for i in range(H.shape[0]):
        if H[i, 0] <= ts0:
            dl = H[i, 1]
            hd = True
        if H[i, 0] <= tv0:
            vc = H[i, 2]
            hv = True
    return hd, dl, hv, vc


@njit(cache=True)
def _servo_k(delta, has, tgt):
    if not has:
        return delta
    diff = tgt - delta
    step = STEER_RATE * 0.01
    if abs(diff) > 1e-4:
        delta += max(-step, min(step, diff))
    return max(-STEER_MAX, min(STEER_MAX, delta))


@njit(cache=True)
def _p_accel_k(v_cmd, v):
    dv = v_cmd - v
    if v > 0.0:
        a = (KP_ACC if dv > 0 else KP_BRK) * dv
    else:
        a = (2.0 * 9.51 / 20.0 if dv > 0 else 2.0 * 9.51 / 5.0) * dv
    pos = A_MAX * V_SWITCH / v if v > V_SWITCH else A_MAX
    return max(-A_MAX, min(pos, a))


@njit(cache=True)
def _bicycle_s_k(v, a, c):
    a = -A_MAX if a < -A_MAX else (A_MAX if a > A_MAX else a)
    cf = c * MU_ * CSF_ * (G_ * LR_ - a * HCG_)
    cr = c * MU_ * CSR_ * (G_ * LF_ + a * HCG_)
    vl = v * WB
    return (-(cr + cf) / vl, (cr * LR_ - cf * LF_) / (v * vl) - 1.0, cf / vl,
            M_ * (cr * LR_ - cf * LF_) / (I_ * WB), -M_ * (LF_ * LF_ * cf + LR_ * LR_ * cr) / (v * I_ * WB),
            M_ * LF_ * cf / (I_ * WB))


@njit(cache=True)
def _obs_main_k(H, tl, tb, t0, n, vf0, delta):
    """loop 1 of CorridorMPC._observe: servo and speed model of the commands in flight over n 10 ms steps from the
    filtered speed vf0; returns the final delta and model speed, and per step delta, accel, model speed, vc, has vc"""
    m = max(n, 1)
    dmain = np.empty(m)
    amain = np.empty(m)
    vms = np.empty(m)
    vck = np.zeros(m)
    hvk = np.zeros(m, np.bool_)
    vm = vf0
    for k in range(n):
        hd, tgt, hv, vc = _targets_k(H, t0 + 0.01 * k, tl, tb)
        delta = _servo_k(delta, hd, tgt)
        a = _p_accel_k(vc, vm) if hv else 0.0
        dmain[k] = delta
        amain[k] = a
        vms[k] = vm
        vck[k] = vc
        hvk[k] = hv
        vm = max(0.0, vm + a * 0.01)
    return delta, vm, dmain, amain, vms, vck, hvk


@njit(cache=True)
def _obs_beta_k(n, vf0, vf, rf0, rf, yawf0, dyw, xf0, yf0, beta, dk, vck, hvk, c):
    """loop 4 of CorridorMPC._observe: sideslip (exact for the linear beta ODE over 10 ms, filtered speed / yaw rate)
    and the predicted course; returns beta, has_mid, beta_mid (middle of the interval), predicted x, y"""
    has_mid = False
    beta_mid = 0.0
    px_ = xf0
    py_ = yf0
    for k in range(n):
        b_before = beta
        f = (k + 0.5) / max(n, 1)
        vk = vf0 + (vf - vf0) * f
        rk = rf0 + (rf - rf0) * f
        a = _p_accel_k(vck[k], vk) if hvk[k] else 0.0
        yk = yawf0 + dyw * f
        if vk >= 0.5:
            a11, a12, b1, _, _, _ = _bicycle_s_k(vk, a, c)
            e = math.exp(a11 * 0.01)
            bss = -(a12 * rk + b1 * dk[k]) / a11
            bmid_ = bss + (beta - bss) * math.exp(a11 * 0.005)
            beta = bss + (beta - bss) * e
        else:
            bmid_ = 0.0
            beta = 0.0
        px_ += vk * math.cos(yk + bmid_) * 0.01
        py_ += vk * math.sin(yk + bmid_) * 0.01
        h2 = 2 * (k + 1)
        if h2 == n:
            beta_mid = beta
            has_mid = True
        elif h2 == n + 1:
            beta_mid = 0.5 * (b_before + beta)
            has_mid = True
    return beta, has_mid, beta_mid, px_, py_


@njit(cache=True)
def _mm_prop(bb, rr, dg, cv, lg, nc, H, tl, tb, t0, n, vf0, v, dmain, amain, vms):
    """parallel single-track models of the stiffness / delay estimator over n 10 ms steps (RK4, the simulator's
    integrator).  Model 0 is the estimator's own (steering dmain, acceleration amain); model 1 + l * nc + i has
    stiffness cv[1 + l * nc + i] and its own servo dg[l] fed with the command history H (t_issue, steer, v_cmd) delayed
    by the estimated latency tl (+ tb for steering) plus the extra delay lg[l] (the speed command too: it sets the
    load transfer).  vms: model speed."""
    nl = lg.shape[0]
    NH = H.shape[0]
    NM = bb.shape[0]
    dvec = np.empty(NM)
    avec = np.empty(NM)
    ag = np.empty(nl)
    h = 0.01
    for k in range(n):
        tt = t0 + 0.01 * k
        vk = vf0 + (v - vf0) * (k + 0.5) / max(n, 1)
        for l in range(nl):
            tq = tt - lg[l] + 1e-6
            ts0 = tq - tl - tb
            tv0 = tq - tl
            tg = dg[l]
            vc = vk
            for j in range(NH - 1, -1, -1):
                if H[j, 0] <= ts0:
                    tg = H[j, 1]
                    break
            for j in range(NH - 1, -1, -1):
                if H[j, 0] <= tv0:
                    vc = H[j, 2]
                    break
            diff = tg - dg[l]
            if abs(diff) > 1e-4:
                dg[l] += min(max(diff, -STEER_RATE * 0.01), STEER_RATE * 0.01)
            dg[l] = min(max(dg[l], -STEER_MAX), STEER_MAX)
            dvc = vc - vk
            a = ((KP_ACC if dvc > 0 else KP_BRK) * dvc) if vk > 0.0 else 0.0
            amax = A_MAX * V_SWITCH / vk if vk > V_SWITCH else A_MAX
            ag[l] = min(max(a, -A_MAX), amax)
        dvec[0] = dmain[k]
        avec[0] = amain[k]
        for l in range(nl):
            for i in range(nc):
                dvec[1 + l * nc + i] = dg[l]
                avec[1 + l * nc + i] = ag[l]
        vm = vms[k]
        if vm >= 0.5:
            for i in range(NM):
                cf = cv[i] * MU_ * CSF_ * (G_ * LR_ - avec[i] * HCG_)
                cr = cv[i] * MU_ * CSR_ * (G_ * LF_ + avec[i] * HCG_)
                a11 = -(cr + cf) / (vm * WB)
                a12 = (cr * LR_ - cf * LF_) / (vm * vm * WB) - 1.0
                a21 = M_ * (cr * LR_ - cf * LF_) / (I_ * WB)
                a22 = -M_ * (LF_ * LF_ * cf + LR_ * LR_ * cr) / (vm * I_ * WB)
                ub = cf / (vm * WB) * dvec[i]
                ur = M_ * LF_ * cf / (I_ * WB) * dvec[i]
                b0, r0 = bb[i], rr[i]
                k1b = a11 * b0 + a12 * r0 + ub
                k1r = a21 * b0 + a22 * r0 + ur
                b_, r_ = b0 + 0.5 * h * k1b, r0 + 0.5 * h * k1r
                k2b = a11 * b_ + a12 * r_ + ub
                k2r = a21 * b_ + a22 * r_ + ur
                b_, r_ = b0 + 0.5 * h * k2b, r0 + 0.5 * h * k2r
                k3b = a11 * b_ + a12 * r_ + ub
                k3r = a21 * b_ + a22 * r_ + ur
                b_, r_ = b0 + h * k3b, r0 + h * k3r
                k4b = a11 * b_ + a12 * r_ + ub
                k4r = a21 * b_ + a22 * r_ + ur
                bb[i] = b0 + h / 6.0 * (k1b + 2.0 * k2b + 2.0 * k3b + k4b)
                rr[i] = r0 + h / 6.0 * (k1r + 2.0 * k2r + 2.0 * k3r + k4r)
        else:
            for i in range(NM):
                bb[i] = 0.0
                rr[i] = vm * math.tan(dvec[i]) / WB


@njit(cache=True)
def _predict_k(H, tl, tb, t, x, y, yaw, v, r, t_pred, delta, beta, c):
    """CorridorMPC._predict (same arithmetic)"""
    n = int(round(t_pred / 0.01))
    a = 0.0
    for k in range(n):
        tt = t + 0.01 * k
        hd, tgt, hv, vc = _targets_k(H, tt, tl, tb)
        delta = _servo_k(delta, hd, tgt)
        a = _p_accel_k(vc, v) if hv else 0.0
        if v >= 0.5:
            a11, a12, b1, a21, a22, b2 = _bicycle_s_k(v, a, c)
            m = max(1, int(math.ceil(-0.01 * min(a11, a22) / 0.7)))
            h = 0.01 / m
            for _ in range(m):
                bd = a11 * beta + a12 * r + b1 * delta
                rd = a21 * beta + a22 * r + b2 * delta
                x += v * math.cos(yaw + beta) * h
                y += v * math.sin(yaw + beta) * h
                yaw += r * h
                beta += bd * h
                r += rd * h
        else:
            x += v * math.cos(yaw) * 0.01
            y += v * math.sin(yaw) * 0.01
            r = v * math.tan(delta) / WB
            yaw += r * 0.01
            beta = 0.0
        v = max(0.0, v + a * 0.01)
    return x, y, yaw, v, beta, r, delta, a


@njit(cache=True)
def _psum(a):
    """np.add.reduce of a short float64 array (numpy's pairwise order: sequential below 8, else 8 partial sums)"""
    n = a.shape[0]
    if n < 8:
        res = 0.0
        for i in range(n):
            res += a[i]
        return res
    r0, r1, r2, r3, r4, r5, r6, r7 = a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]
    i = 8
    m = n - (n % 8)
    while i < m:
        r0 += a[i]
        r1 += a[i + 1]
        r2 += a[i + 2]
        r3 += a[i + 3]
        r4 += a[i + 4]
        r5 += a[i + 5]
        r6 += a[i + 6]
        r7 += a[i + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        res += a[i]
        i += 1
    return res


@njit(cache=True)
def _dp_boxes_k(sd, dg, s0, bx, bprev, bpf, bhard, half, w_h, dp_near, p_near, c_pref, w_soft, Ds, gap_rule,
                min_gap_w, min_gap_keep, wlo, whi, cost_node, blocked):
    """the obstacle part of CorridorMPC._dp's node costs / blocked cells (same operations, same order per box);
    bprev: previous side (0: none), bpf: flip penalty, bhard: committed and close (wrong side blocked)"""
    M1 = sd.shape[0]
    J = dg.shape[0]
    ks = np.empty(M1, np.int64)
    for b in range(bx.shape[0]):
        nks = 0
        for k in range(M1):
            if sd[k] >= bx[b, 0] - half and sd[k] <= bx[b, 1] + half:
                ks[nks] = k
                nks += 1
        if nks == 0:
            continue
        lo = bx[b, 2] - w_h
        hi = bx[b, 3] + w_h
        nfar = 0
        mwhi = np.inf
        mwlo = -np.inf
        for t in range(nks):
            k = ks[t]
            if sd[k] - s0 >= dp_near:
                nfar += 1
                mwhi = min(mwhi, whi[k])
                mwlo = max(mwlo, wlo[k])
                for j in range(J):
                    if dg[j] > lo and dg[j] < hi:
                        blocked[k, j] = True
        if nfar > 0 and gap_rule:
            # a side whose corridor (obstacle .. wall) is narrower than min_gap_w is not offered; the side already
            # chosen down to min_gap_keep
            g_hi = min_gap_keep if bprev[b] == 1 else min_gap_w
            g_lo = min_gap_keep if bprev[b] == -1 else min_gap_w
            if mwhi - hi < g_hi:
                for t in range(nks):
                    k = ks[t]
                    if sd[k] - s0 >= dp_near:
                        for j in range(J):
                            if dg[j] >= hi:
                                blocked[k, j] = True
            if lo - mwlo < g_lo:
                for t in range(nks):
                    k = ks[t]
                    if sd[k] - s0 >= dp_near:
                        for j in range(J):
                            if dg[j] <= lo:
                                blocked[k, j] = True
        for t in range(nks):
            k = ks[t]
            if sd[k] - s0 < dp_near:
                for j in range(J):
                    if dg[j] > lo and dg[j] < hi:
                        cost_node[k, j] += p_near
        for t in range(nks):
            k = ks[t]
            for j in range(J):
                clr = max(lo - dg[j], dg[j] - hi)
                soft = max(0.0, c_pref - clr)
                cost_node[k, j] += w_soft * (soft * soft) * Ds
        if bprev[b] != 0:
            dc = 0.5 * (bx[b, 2] + bx[b, 3])
            kmid = ks[nks // 2]
            for j in range(J):
                wrong = (dg[j] < dc) if bprev[b] > 0 else (dg[j] > dc)
                if not wrong:
                    continue
                if bhard[b]:
                    for t in range(nks):
                        blocked[ks[t], j] = True
                else:
                    cost_node[kmid, j] += bpf[b]


@njit(cache=True)
def _dp_sides_k(sd, dpath_dp, reach, bx, half):
    """side of every box from the DP path (mean lateral offset over its reached stations vs its centre)"""
    nb = bx.shape[0]
    out = np.zeros(nb, np.int64)
    buf = np.empty(sd.shape[0])
    for b in range(nb):
        n = 0
        for k in range(min(reach + 1, sd.shape[0])):
            if sd[k] >= bx[b, 0] - half and sd[k] <= bx[b, 1] + half:
                buf[n] = dpath_dp[k]
                n += 1
        if n == 0:
            continue
        out[b] = 1 if _psum(buf[:n]) / n > 0.5 * (bx[b, 2] + bx[b, 3]) else -1
    return out


@njit(cache=True)
def _psum32_blk(a, lo, n):
    """np.add.reduce of float32 a[lo:lo+n], n <= 128, in numpy's order (sequential below 8, else 8 partial sums)"""
    if n < 8:
        res = np.float32(0.0)
        for i in range(lo, lo + n):
            res = np.float32(res + a[i])
        return res
    r = np.empty(8, np.float32)
    for j in range(8):
        r[j] = a[lo + j]
    i = 8
    m = n - (n % 8)
    while i < m:
        for j in range(8):
            r[j] = np.float32(r[j] + a[lo + i + j])
        i += 8
    res = np.float32(np.float32(np.float32(r[0] + r[1]) + np.float32(r[2] + r[3])) +
                     np.float32(np.float32(r[4] + r[5]) + np.float32(r[6] + r[7])))
    while i < n:
        res = np.float32(res + a[lo + i])
        i += 1
    return res


@njit(cache=True)
def _psum32(a, lo, n):
    """np.add.reduce of float32 a[lo:lo+n] (n <= 8192) in numpy's order: blocks of <= 128, larger ranges split in two
    halves (multiple of 8) and summed recursively; evaluated with an explicit stack (cached numba functions must
    not be recursive)"""
    if n <= 128:
        return _psum32_blk(a, lo, n)
    s_lo = np.empty(64, np.int64)
    s_n = np.empty(64, np.int64)
    s_state = np.zeros(64, np.int64)
    s_val = np.zeros(64, np.float32)
    s_lo[0] = lo
    s_n[0] = n
    sp = 1
    ret = np.float32(0.0)
    while sp > 0:
        t = sp - 1
        nn = s_n[t]
        if nn <= 128:
            ret = _psum32_blk(a, s_lo[t], nn)
            sp -= 1
            continue
        n2 = nn // 2
        n2 -= n2 % 8
        if s_state[t] == 0:
            s_state[t] = 1
            s_lo[sp] = s_lo[t]
            s_n[sp] = n2
            s_state[sp] = 0
            sp += 1
        elif s_state[t] == 1:
            s_val[t] = ret
            s_state[t] = 2
            s_lo[sp] = s_lo[t] + n2
            s_n[sp] = nn - n2
            s_state[sp] = 0
            sp += 1
        else:
            ret = np.float32(s_val[t] + ret)
            sp -= 1
    return ret


@njit(cache=True)
def _clusters_k(labels, nc, rel, dd, seen, brg, bi, rng, scan):
    """per cluster of _boxes: extent of the cells (rel s, d), and of the cells the current scan sees: count, mean
    bearing (numpy's float32 mean) and whether the beams just beyond the lowest / highest beam index return
    something closer (edge hidden, see _extent_fix)"""
    n = labels.shape[0]
    start = np.zeros(nc + 1, np.int64)
    for i in range(n):
        start[labels[i] + 1] += 1
    for c in range(nc):
        start[c + 1] += start[c]
    order = np.empty(n, np.int64)
    pos = start[:nc].copy()
    for i in range(n):
        c = labels[i]
        order[pos[c]] = i
        pos[c] += 1
    rmin = np.empty(nc, np.float32)
    rmax = np.empty(nc, np.float32)
    dmin = np.empty(nc, np.float32)
    dmax = np.empty(nc, np.float32)
    nseen = np.zeros(nc, np.int64)
    bmean = np.zeros(nc, np.float32)
    occl = np.zeros((nc, 2), np.int8)       # 0 edge seen free, 1 hidden, 2 no information (only NaN beams)
    buf = np.empty(max(n, 1), np.float32)
    for c in range(nc):
        i0 = order[start[c]]
        rmin[c] = rel[i0]
        rmax[c] = rel[i0]
        dmin[c] = dd[i0]
        dmax[c] = dd[i0]
        ns = 0
        ie_lo = -1
        ie_hi = -1
        for t in range(start[c], start[c + 1]):
            i = order[t]
            rmin[c] = min(rmin[c], rel[i])
            rmax[c] = max(rmax[c], rel[i])
            dmin[c] = min(dmin[c], dd[i])
            dmax[c] = max(dmax[c], dd[i])
            if seen[i]:
                buf[ns] = brg[i]
                ns += 1
                if ie_lo < 0 or bi[i] < bi[ie_lo]:
                    ie_lo = i
                if ie_hi < 0 or bi[i] > bi[ie_hi]:
                    ie_hi = i
        nseen[c] = ns
        if ns == 0:
            continue
        bmean[c] = np.float32(np.float64(_psum32(buf, 0, ns)) / ns)
        for side in range(2):
            ie = ie_lo if side == 0 else ie_hi
            step = -1 if side == 0 else 1
            b0 = bi[ie]
            cnt = 0
            nnan = 0
            mn = np.inf
            for k in range(1, 4):
                bb = b0 + step * k
                if 0 <= bb <= 1079:
                    cnt += 1
                    sv = scan[bb]
                    if sv != sv:
                        nnan += 1
                    elif sv < mn:
                        mn = sv
            if cnt == 3 and nnan == 3:
                occl[c, side] = 2
            elif cnt < 3 or mn < np.float64(rng[ie]) - 0.05:
                occl[c, side] = 1
    return rmin, rmax, dmin, dmax, nseen, bmean, occl


@njit(cache=True)
def _dp_k(s0, d0, chi0, v, N, ds, Ds, dd, Kmax, wlo_c, whi_c, wlo_p, whi_p, fds, dp_dmax, w_dev, w_wall, bx, bprev,
          bpf, bhard, w_h, dp_near, p_near, c_pref, w_soft, gap_rule, min_gap_w, dp_adec, dp_vmax, dp_vfloor, a_lat_hard,
          dp_kcap, w_cv, w_sl, c_ty, min_gap_keep, fk, L, dp_kgeo, w_geo):
    """CorridorMPC._dp after the per-box commitment logic, in one compiled pass (same formulas and order): grid,
    wall and obstacle node costs, curvature limit, _dp_core, back-tracking, sides.  Returns st, dpath, sides, reach,
    s_block (nan: complete path)"""
    st = np.empty(N + 1)
    for i in range(N + 1):
        st[i] = s0 + ds * i
    M = int(round(N * ds / Ds))
    sd = np.empty(M + 1)
    for i in range(M + 1):
        sd[i] = s0 + Ds * i
    sq = dd / Ds
    K = 2 * Kmax + 1
    kv = np.arange(-Kmax, Kmax + 1)
    nf = wlo_c.shape[0]
    wlo = np.empty(M + 1)
    whi = np.empty(M + 1)
    wlo2 = np.empty(M + 1)
    whi2 = np.empty(M + 1)
    for i in range(M + 1):
        k = int(np.rint(sd[i] / fds)) % nf
        wlo[i], whi[i], wlo2[i], whi2[i] = wlo_c[k], whi_c[k], wlo_p[k], whi_p[k]
    jlo = int(math.floor((min(max(wlo.min(), -dp_dmax), d0) + 3.5) / dd)) - 1
    jhi = int(math.ceil((max(min(whi.max(), dp_dmax), d0) + 3.5) / dd)) + 1
    j_a, j_b = max(jlo, 0), min(jhi, int(7.0 / dd))
    J = j_b - j_a + 1
    dg = np.empty(J)
    for j in range(J):
        dg[j] = -3.5 + dd * (j_a + j)
    blocked = np.empty((M + 1, J), np.bool_)
    cost_node = np.empty((M + 1, J))
    for i in range(M + 1):
        for j in range(J):
            blocked[i, j] = dg[j] < wlo[i] or dg[j] > whi[i]
            pen = max(0.0, wlo2[i] - dg[j]) + max(0.0, dg[j] - whi2[i])
            cost_node[i, j] = w_dev * (dg[j] * dg[j]) * Ds + w_wall * (pen * pen) * Ds
    half = 0.5 * Ds
    _dp_boxes_k(sd, dg, s0, bx, bprev, bpf, bhard, half, w_h, dp_near, p_near, c_pref, w_soft, Ds, gap_rule,
                min_gap_w, min_gap_keep, wlo, whi, cost_node, blocked)
    # curvature limit per DP station: the slowest speed the car can have there (planned braking)
    cst = np.empty(M + 1, np.int64)
    c = 0
    for i in range(M + 1):
        vr = min(math.sqrt(max(v * v - 2.0 * dp_adec * (sd[i] - s0), 0.0)), 100.0)
        vdp = max(min(vr, dp_vmax), dp_vfloor)
        kap = min(min(STEER_MAX / (WB + K_US0 / c_ty * vdp * vdp), a_lat_hard / (vdp * vdp)), dp_kcap)
        cst[i] = min(max(np.int64(math.floor(kap * Ds / sq + 1e-9)), 1), Kmax)
        c = max(c, cst[i])
    cv_cost = np.empty(2 * c + 1)
    for i in range(2 * c + 1):
        g = (i - c) * sq / Ds
        cv_cost[i] = w_cv * (g * g) * Ds
    sl_cost = np.empty(K)
    for k in range(K):
        g = kv[k] * sq
        sl_cost[k] = w_sl * (g * g) * Ds
    # geometric curvature kr / D + d'' / D^2 (D = 1 - kr d) within the steering limit (soft, see _dp)
    dlo = np.full((M + 1, J), -100.0)
    dhi = np.full((M + 1, J), 100.0)
    if dp_kgeo > 0.0:
        kg = dp_kgeo * STEER_MAX / (WB + K_US0 / c_ty * dp_vfloor ** 2)
        nfk = fk.shape[0]
        for i in range(M + 1):
            u = (sd[i] % L) / fds
            fu = math.floor(u)
            i0 = int(fu) % nfk
            f = u - fu
            kr_d = fk[i0] * (1 - f) + fk[(i0 + 1) % nfk] * f
            for j in range(J):
                Dm = max(1.0 - kr_d * dg[j], 0.05)
                dlo[i, j] = (-kg * Dm - kr_d) * Dm * (Ds / sq)
                dhi[i, j] = (kg * Dm - kr_d) * Dm * (Ds / sq)
    cost0 = np.full((J, K), np.inf)
    j0 = min(max(int(round((d0 - dg[0]) / dd)), 0), J - 1)
    k0 = min(max(int(round(math.tan(max(-1.2, min(1.2, chi0))) / sq)), -Kmax), Kmax) + Kmax
    for k in range(max(k0 - 1, 0), min(k0 + 2, K)):
        cost0[j0, k] = 0.0
    back, cost, reach = _dp_core(cost0, cost_node, blocked, kv, cst, cv_cost, sl_cost, dlo, dhi, w_geo)
    if reach < M:
        # cost at the last reached station: rerun to that station
        back, cost, reach2 = _dp_core(cost0, cost_node[:reach + 1], blocked[:reach + 1], kv, cst[:reach + 1], cv_cost,
                                      sl_cost, dlo, dhi, w_geo)
    flat = 0
    for q in range(J * K):
        if cost.flat[q] < cost.flat[flat]:
            flat = q
    jb, kb = flat // K, flat % K
    path_j = np.zeros(M + 1, np.int64)
    path_j[reach] = jb
    for i in range(reach, 0, -1):
        dl = np.int64(back[i - 1, jb, kb]) - c
        jp = jb - kv[kb]
        kb = kb - dl
        jb = jp
        path_j[i - 1] = jb
    for i in range(reach + 1, M + 1):
        path_j[i] = path_j[reach]
    dpath_dp = np.empty(M + 1)
    for i in range(M + 1):
        dpath_dp[i] = dg[path_j[i]]
    dpath = np.empty(N + 1)
    for i in range(N + 1):
        dpath[i] = _interp1(st[i], sd, dpath_dp)
    sides = _dp_sides_k(sd, dpath_dp, reach, bx, half)
    s_block = sd[reach + 1] if reach < M else np.nan
    return st, dpath, sides, s_block


@njit(cache=True)
def _plan_prep_k(st, fk, L, fds, sol_st, sol_X, dpath, X0, prof_s, prof_dpl, prof_vf, pv, a_cmd, a_acc_eff, jerk,
                 tab_emerg, t_call):
    """inputs of CorridorMPC._plan in the usual case (previous solution and speed profile exist, acc_model), same
    arithmetic: raceline curvature at the stations, linearisation trajectory (previous solution moved to the new
    stations, DP path beyond it), previous speed profile moved to the new stations and _cmd_sim on it"""
    N1 = st.shape[0]
    nf = fk.shape[0]
    kr = np.empty(N1)
    for i in range(N1):
        u = (st[i] % L) / fds
        fu = math.floor(u)
        i0 = int(fu) % nf
        f = u - fu
        kr[i] = fk[i0] * (1 - f) + fk[(i0 + 1) % nf] * f
    Xbar = np.empty((N1, 5))
    s_end = sol_st[sol_st.shape[0] - 1]
    for i in range(N1):
        if st[i] > s_end:
            Xbar[i, 0] = dpath[i]
            for k in range(1, 5):
                Xbar[i, k] = 0.0
        else:
            for k in range(5):
                Xbar[i, k] = _interp1(st[i], sol_st, sol_X[:, k])
    for k in range(5):
        Xbar[0, k] = X0[k]
    N = N1 - 1
    dpl = np.empty(N)
    cum0 = np.empty(N1)
    cum0[0] = 0.0
    vtarget = np.empty(N1)
    for i in range(N1):
        vtarget[i] = _interp1(st[i], prof_s, prof_vf)
        if i < N:
            dpl[i] = _interp1(st[i], prof_s[:N], prof_dpl)
            cum0[i + 1] = cum0[i] + dpl[i]
    vpred, acc = _cmd_sim(vtarget, cum0, dpl, pv, a_cmd, a_acc_eff, jerk, tab_emerg, t_call)
    return kr, Xbar, dpl, vpred, acc


@njit(cache=True)
def _boxes_k(active, act_ok, act_s, act_x, act_y, act_d, v_rng, v_brg, v_bi, scan, MW, W, s_car, L, win_hi, link):
    """the cell selection and clustering of CorridorMPC._boxes on the compact remembered set, same operations
    (float32 like numpy): cells with map ok, one per 2x2 block above 400 cells (first occurrence), planning window
    in s, single-linkage clusters numbered like np.unique of their roots; then _clusters_k"""
    n = active.shape[0]
    pos = np.empty(n, np.int64)
    m = 0
    for i in range(n):
        if act_ok[i]:
            pos[m] = i
            m += 1
    pos = pos[:m]
    if m > 400:
        blk = np.empty(m, np.int64)
        for i in range(m):
            a = active[pos[i]]
            blk[i] = (a // MW // 2) * W + (a % MW) // 2
        order = np.argsort(blk, kind='mergesort')
        keep = np.zeros(m, np.bool_)
        for t in range(m):
            if t == 0 or blk[order[t]] != blk[order[t - 1]]:
                keep[order[t]] = True
        pos = pos[keep]
        m = pos.shape[0]
    sc, h, Lf, lo, hi = np.float32(s_car), np.float32(L / 2), np.float32(L), np.float32(-1.5), np.float32(win_hi)
    rel = np.empty(m, np.float32)
    k = 0
    for i in range(m):
        v = np.float32(np.float32(act_s[pos[i]] - sc) + h)
        r = np.float32(np.float32(v % Lf) - h)
        if r > lo and r < hi:
            pos[k] = pos[i]
            rel[k] = r
            k += 1
    pos = pos[:k]
    rel = rel[:k]
    if k == 0:
        return _clusters_k(np.zeros(0, np.int64), 0, rel, rel, np.zeros(0, np.bool_), rel, np.zeros(0, np.int64), rel,
                           scan)
    xs = np.empty(k)
    ys = np.empty(k)
    dd = np.empty(k, np.float32)
    rng = np.empty(k, np.float32)
    brg = np.empty(k, np.float32)
    bi = np.empty(k, np.int64)
    seen = np.empty(k, np.bool_)
    for i in range(k):
        j = pos[i]
        xs[i] = act_x[j]
        ys[i] = act_y[j]
        dd[i] = act_d[j]
        rng[i] = v_rng[j]
        brg[i] = v_brg[j]
        bi[i] = v_bi[j]
        bic = min(max(bi[i], 0), 1079)
        seen[i] = bi[i] >= 0 and bi[i] <= 1079 and abs(scan[bic] - np.float64(rng[i])) < 0.08
    roots = _cluster_k(xs, ys, link) if k > 1 else np.zeros(1, np.int64)
    # labels as np.unique(roots, return_inverse=True): rank of the root among the distinct roots
    rs = np.unique(roots)
    nc = rs.shape[0]
    labels = np.empty(k, np.int64)
    for i in range(k):
        labels[i] = np.searchsorted(rs, roots[i])
    return _clusters_k(labels, nc, rel, dd, seen, brg, bi, rng, scan)


@njit(cache=True)
def _interp1(x, xp, fp):
    """np.interp at one point (numpy's special cases and formula; xp strictly increasing)"""
    n = xp.shape[0]
    if x < xp[0]:
        return fp[0]
    if x >= xp[n - 1]:
        return fp[n - 1]
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        if xp[mid] <= x:
            lo = mid
        else:
            hi = mid
    if xp[lo] == x:
        return fp[lo]
    slope = (fp[lo + 1] - fp[lo]) / (xp[lo + 1] - xp[lo])
    return slope * (x - xp[lo]) + fp[lo]


@njit(cache=True)
def _wrap1(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


@njit(cache=True)
def _speed_k(st, d, mu, beta, kr, v0, scan, lx, ly, yaw, s_block, kyaw, fx, fy, fnx, fny, fpsi, fvx, L, fds,
             ds, v_max, a_lat, v_end_scale, swerve_w, a_acc, gg, gg_min, a_acc_eff, vis_tol, vis_start, m_vis, m_blk,
             t_react, v_vis_safe, tab_plan, tab_emerg, tab_dstop, tab_v):
    """CorridorMPC._speed_profile (without vis_obs_stop) in one compiled pass, same formulas and order.
    s_block: nan when there is none; kyaw: empty when unused.  Returns vprof, vpred, acc, dpl, k_vis (-1: all
    visible), v[0] of the limited profile."""
    N1 = st.shape[0]
    N = N1 - 1
    nf = fx.shape[0]
    dpl = np.empty(N)
    cum = np.empty(N1)
    cum[0] = 0.0
    for i in range(N):
        sc = (1.0 - kr[i] * d[i]) / max(math.cos(mu[i] + beta[i]), 0.3)
        dpl[i] = ds * max(sc, 0.2)
        cum[i + 1] = cum[i] + dpl[i]
    # path curvature from the course angle (raceline heading + mu + beta), 3-point max
    hd = np.empty(N1)
    xs = np.empty(N1)
    ys = np.empty(N1)
    for i in range(N1):
        u = (st[i] % L) / fds
        fu = math.floor(u)
        i0 = int(fu) % nf
        i1 = (i0 + 1) % nf
        f = u - fu
        hd[i] = fpsi[i0] + _wrap1(fpsi[i1] - fpsi[i0]) * f + (mu[i] + beta[i])
        bx = fx[i0] * (1 - f) + fx[i1] * f
        by = fy[i0] * (1 - f) + fy[i1] * f
        nx = fnx[i0] * (1 - f) + fnx[i1] * f
        ny = fny[i0] * (1 - f) + fny[i1] * f
        xs[i] = bx + d[i] * nx
        ys[i] = by + d[i] * ny
    kg = np.empty(N1)
    for i in range(N):
        kg[i] = _wrap1(hd[i + 1] - hd[i]) / dpl[i]
    kg[N] = kg[N - 1]
    ka0 = np.empty(N1)
    for i in range(N1):
        ka0[i] = abs(kg[i])
        if swerve_w > 0.0:
            ka0[i] = ka0[i] + swerve_w * abs(kg[i] - kr[i])
        if kyaw.shape[0] > 0:
            ka0[i] = max(ka0[i], kyaw[i])
    ka = np.empty(N1)
    vlim = np.empty(N1)
    for i in range(N1):
        nb_ = max(ka0[min(i + 1, N)], ka0[max(i - 1, 0)])
        ka[i] = max(ka0[i], nb_)
        vlim[i] = min(v_max, math.sqrt(a_lat / max(ka[i], 1e-3)))
    u = (st[N] % L) / fds
    fu = math.floor(u)
    i0 = int(fu) % nf
    f = u - fu
    vx_end = fvx[i0] * (1 - f) + fvx[(i0 + 1) % nf] * f
    v_end = min(v_max, vx_end * math.sqrt(a_lat / 12.0) * v_end_scale)
    vlim[N] = min(vlim[N], v_end)
    v = _passes(vlim, dpl, tab_plan, v0, a_acc, ka, a_lat, gg, gg_min)
    # emergency envelope: stop before unseen path / closed corridor
    kv = -1
    for i in range(N1):
        dx = xs[i] - lx
        dy = ys[i] - ly
        rng = math.hypot(dx, dy)
        brg = _wrap1(math.atan2(dy, dx) - yaw)
        b = int(np.rint((brg - ANG_MIN) / ANG_INC))
        inview = b >= 1 and b <= 1078
        bb = min(max(b, 1), 1078)
        meas = _fmin2(_fmin2(scan[bb - 1], scan[bb]), scan[bb + 1])     # NaN beams ignored (all NaN: unseen)
        vis = (inview and meas >= rng - vis_tol) or cum[i] < vis_start
        if not vis:
            kv = i
            break
    has_stop = False
    stop_at = 0.0
    sb = 0.0
    if kv >= 0:
        stop_at = cum[kv] - BOX_FRONT - m_vis
        has_stop = True
    blk = not math.isnan(s_block)
    if blk:
        kb = 0
        while kb < N1 and st[kb] < s_block:
            kb += 1
        kb = min(kb, N)
        sb = cum[kb] - BOX_FRONT - m_blk
        stop_at = sb if not has_stop else min(stop_at, sb)
        has_stop = True
    if has_stop:
        vis_env = v_vis_safe > 0.0 and kv >= 0 and ((not blk) or st[kv] < s_block)
        ds_ = _interp1(v_vis_safe, tab_v, tab_dstop)
        for i in range(N1):
            if vis_env:
                reach_v = cum[kv] - BOX_FRONT - m_vis - cum[i] - v0 * t_react
                e = _interp1(max(reach_v, 0.0) + ds_, tab_dstop, tab_v)
                if reach_v <= 0.0:
                    e = min(e, v_vis_safe)
                if blk:
                    reach_b = sb - cum[i] - v0 * t_react
                    eb = _interp1(max(reach_b, 0.0), tab_dstop, tab_v)
                    if reach_b <= 0:
                        eb = 0.0
                    e = min(e, eb)
            else:
                reach = stop_at - cum[i] - v0 * t_react
                e = _interp1(max(reach, 0.0), tab_dstop, tab_v)
                if reach <= 0:
                    e = 0.0
            v[i] = min(v[i], e)
    vprof, vpred = _fwd(v, dpl, v0, a_acc_eff, tab_emerg)
    acc = np.empty(N)
    for i in range(N):
        a = (vpred[i + 1] * vpred[i + 1] - vpred[i] * vpred[i]) / (2 * dpl[i])
        acc[i] = min(max(a, -A_MAX), A_MAX)
    return vprof, vpred, acc, dpl, kv, v[0], cum


@njit(cache=True)
def _hp_slack_k(Xs, cmu, lgc, ugc):
    """corridor slack per station (largest violation of the collision-box point rows)"""
    N, nl = cmu.shape
    sl = np.zeros(N + 1)
    for k in range(N):
        m = 0.0
        for j in range(nl):
            val = Xs[k + 1, 0] + cmu[k, j] * Xs[k + 1, 1]
            if lgc[k, j] - val > m:
                m = lgc[k, j] - val
            if val - ugc[k, j] > m:
                m = val - ugc[k, j]
        sl[k + 1] = m
    return sl


class _Hpipm:
    """minimal ctypes wrapper of HPIPM's OCP QP IPM (the library CasADi ships), soft general constraints"""

    def __init__(self, N, nx, nu, nbx, nbu, ng, ns, idxbx, idxbu, idxs, Zl, Zu, zl, zu, R, iter_max=30,
                 tol=(1e-5, 1e-6, 1e-6, 1e-6), mode=1):
        import ctypes as C
        import casadi
        path = os.path.join(os.path.dirname(casadi.__file__), 'libhpipm.so')
        lib = C.CDLL(path)
        self.C, self.lib, self.N = C, lib, N
        vp, ip, sz = C.c_void_p, C.c_int, C.c_size_t
        for name in ('d_ocp_qp_dim_strsize', 'd_ocp_qp_strsize', 'd_ocp_qp_sol_strsize', 'd_ocp_qp_ipm_arg_strsize',
                     'd_ocp_qp_ipm_ws_strsize'):
            getattr(lib, name).restype = sz
        lib.d_ocp_qp_dim_memsize.restype = sz
        lib.d_ocp_qp_dim_memsize.argtypes = [ip]
        for name in ('d_ocp_qp_memsize', 'd_ocp_qp_sol_memsize', 'd_ocp_qp_ipm_arg_memsize'):
            getattr(lib, name).restype = sz
            getattr(lib, name).argtypes = [vp]
        lib.d_ocp_qp_ipm_ws_memsize.restype = sz
        lib.d_ocp_qp_ipm_ws_memsize.argtypes = [vp, vp]
        keep = []

        def buf(n):
            b = C.create_string_buffer(int(n) + 64)
            keep.append(b)
            return C.addressof(b)

        def iarr(lst):
            a = np.ascontiguousarray(np.array(lst, dtype=np.int32))
            keep.append(a)
            return a
        self.dim = buf(lib.d_ocp_qp_dim_strsize())
        lib.d_ocp_qp_dim_create.argtypes = [ip, vp, vp]
        lib.d_ocp_qp_dim_create(N, self.dim, buf(lib.d_ocp_qp_dim_memsize(N)))
        lib.d_ocp_qp_dim_set_all.argtypes = [vp] * 7
        arrs = [iarr(a) for a in (nx, nu, nbx, nbu, ng, ns)]
        lib.d_ocp_qp_dim_set_all(*[a.ctypes.data for a in arrs], self.dim)
        self.qp = buf(lib.d_ocp_qp_strsize())
        lib.d_ocp_qp_create.argtypes = [vp, vp, vp]
        lib.d_ocp_qp_create(self.dim, self.qp, buf(lib.d_ocp_qp_memsize(self.dim)))
        self.sol = buf(lib.d_ocp_qp_sol_strsize())
        lib.d_ocp_qp_sol_create.argtypes = [vp, vp, vp]
        lib.d_ocp_qp_sol_create(self.dim, self.sol, buf(lib.d_ocp_qp_sol_memsize(self.dim)))
        self.arg = buf(lib.d_ocp_qp_ipm_arg_strsize())
        lib.d_ocp_qp_ipm_arg_create.argtypes = [vp, vp, vp]
        lib.d_ocp_qp_ipm_arg_create(self.dim, self.arg, buf(lib.d_ocp_qp_ipm_arg_memsize(self.dim)))
        lib.d_ocp_qp_ipm_arg_set_default.argtypes = [ip, vp]
        lib.d_ocp_qp_ipm_arg_set_default(mode, self.arg)
        for nm, val, typ in (('iter_max', iter_max, C.c_int), ('tol_stat', tol[0], C.c_double),
                             ('tol_eq', tol[1], C.c_double), ('tol_ineq', tol[2], C.c_double),
                             ('tol_comp', tol[3], C.c_double)):
            f = getattr(lib, 'd_ocp_qp_ipm_arg_set_' + nm)
            f.argtypes = [vp, vp]
            v = typ(val)
            keep.append(v)
            f(C.addressof(v), self.arg)
        self.ws = buf(lib.d_ocp_qp_ipm_ws_strsize())
        lib.d_ocp_qp_ipm_ws_create.argtypes = [vp, vp, vp, vp]
        lib.d_ocp_qp_ipm_ws_create(self.dim, self.arg, self.ws, buf(lib.d_ocp_qp_ipm_ws_memsize(self.dim, self.arg)))
        for nm in ('A', 'B', 'b', 'Q', 'S', 'R', 'q', 'r', 'lbx', 'ubx', 'lbu', 'ubu', 'idxbx', 'idxbu', 'C', 'D', 'lg',
                   'ug', 'Zl', 'Zu', 'zl', 'zu', 'idxs', 'lls', 'lus'):
            f = getattr(lib, 'd_ocp_qp_set_' + nm)
            f.argtypes = [ip, vp, vp]
            f.restype = None
        for nm in ('x', 'u', 'sl', 'su'):
            f = getattr(lib, 'd_ocp_qp_sol_get_' + nm)
            f.argtypes = [ip, vp, vp]
            f.restype = None
        lib.d_ocp_qp_ipm_solve.argtypes = [vp, vp, vp, vp]
        lib.d_ocp_qp_ipm_get_status.argtypes = [vp, vp]
        lib.d_ocp_qp_ipm_get_iter.argtypes = [vp, vp]
        self.keep = keep
        self.nx, self.nu, self.ng, self.ns = list(nx), list(nu), list(ng), list(ns)
        # static data
        for k in range(N + 1):
            if nbx[k]:
                lib.d_ocp_qp_set_idxbx(k, iarr(idxbx[k]).ctypes.data, self.qp)
            if nbu[k]:
                lib.d_ocp_qp_set_idxbu(k, iarr(idxbu[k]).ctypes.data, self.qp)
            if nu[k]:
                Rk = np.ascontiguousarray(R[k], dtype=float)
                keep.append(Rk)
                lib.d_ocp_qp_set_R(k, Rk.ctypes.data, self.qp)
                Sk = np.zeros(nu[k] * nx[k])
                keep.append(Sk)
                lib.d_ocp_qp_set_S(k, Sk.ctypes.data, self.qp)
                rk = np.zeros(nu[k])
                keep.append(rk)
                lib.d_ocp_qp_set_r(k, rk.ctypes.data, self.qp)
            if ns[k]:
                lib.d_ocp_qp_set_idxs(k, iarr(idxs[k]).ctypes.data, self.qp)
                for nm, val in (('Zl', Zl[k]), ('Zu', Zu[k]), ('zl', zl[k]), ('zu', zu[k]), ('lls', np.zeros(ns[k])),
                                ('lus', np.zeros(ns[k]))):
                    a = np.ascontiguousarray(val, dtype=float)
                    keep.append(a)
                    getattr(lib, 'd_ocp_qp_set_' + nm)(k, a.ctypes.data, self.qp)
            if ng[k] and nu[k]:
                Dk = np.zeros(ng[k] * nu[k])
                keep.append(Dk)
                lib.d_ocp_qp_set_D(k, Dk.ctypes.data, self.qp)
        self.status = C.c_int(0)
        self.iters = C.c_int(0)
        self.out_x = np.zeros((N + 1, nx[0]))
        self.out_u = np.zeros((N, max(nu)))
        self.out_sl = np.zeros((N + 1, max(max(ns), 1)))
        self.out_su = np.zeros((N + 1, max(max(ns), 1)))

    def setup_all(self, nbx, nbu, idxbx, idxbu, idxs, Zl, zl, R):
        """persistent buffers + pointer arrays for d_ocp_qp_set_all / d_ocp_qp_sol_get_all (one C call each)"""
        C, N, lib = self.C, self.N, self.lib
        nx, nu, ng, ns = self.nx, self.nu, self.ng, self.ns
        P = lambda arrs: (C.c_void_p * (N + 1))(*[a.ctypes.data if a is not None and a.size else None for a in arrs])
        z = lambda n: np.zeros(max(int(n), 1))
        self.bA = np.zeros((N + 1, 5, 5))
        self.bB = np.zeros((N + 1, 1, 5))
        self.bb = np.zeros((N + 1, 5))
        self.bQ = np.zeros((N + 1, 5, 5))
        self.bq = np.zeros((N + 1, 5))
        mg = max(ng)
        self.bC = np.zeros((N + 1, 5, mg))          # column-major (ng x nx) blocks: C^T rows
        self.blg = np.zeros((N + 1, mg))
        self.bug = np.zeros((N + 1, mg))
        mbx = max(nbx)
        self.blbx = np.zeros((N + 1, mbx))
        self.bubx = np.zeros((N + 1, mbx))
        self.keep_all = []
        k_ = self.keep_all
        S_ = [z(nu[k] * nx[k]) for k in range(N + 1)]
        R_ = [np.ascontiguousarray(R[k], dtype=float).ravel() if nu[k] else z(1) for k in range(N + 1)]
        r_ = [z(nu[k]) for k in range(N + 1)]
        D_ = [z(ng[k] * nu[k]) for k in range(N + 1)]
        ibx = [np.array(idxbx[k], dtype=np.int32) for k in range(N + 1)]
        ibu = [np.array(idxbu[k], dtype=np.int32) for k in range(N + 1)]
        lbu = [np.full(max(nbu[k], 1), -STEER_RATE * 0.95) for k in range(N + 1)]
        ubu = [np.full(max(nbu[k], 1), STEER_RATE * 0.95) for k in range(N + 1)]
        Zl_ = [np.array(Zl[k] if ns[k] else [0.0], dtype=float) for k in range(N + 1)]
        zl_ = [np.array(zl[k] if ns[k] else [0.0], dtype=float) for k in range(N + 1)]
        isx = [np.array(idxs[k] if ns[k] else [0], dtype=np.int32) for k in range(N + 1)]
        isr = []
        for k in range(N + 1):
            nb = nbx[k] + nbu[k]
            a = -np.ones(nb + ng[k], dtype=np.int32)
            for j, c in enumerate(idxs[k]):
                a[c] = j
            isr.append(a if a.size else np.zeros(1, dtype=np.int32))
        ls_ = [z(ns[k]) for k in range(N + 1)]
        k_ += [S_, R_, r_, D_, ibx, ibu, lbu, ubu, Zl_, zl_, isx, isr, ls_]
        self.p_all = [P([self.bA[k] for k in range(N + 1)]), P([self.bB[k] for k in range(N + 1)]),
                      P([self.bb[k] for k in range(N + 1)]), P([self.bQ[k] for k in range(N + 1)]), P(S_), P(R_),
                      P([self.bq[k] for k in range(N + 1)]), P(r_), P(ibx), P([self.blbx[k] for k in range(N + 1)]),
                      P([self.bubx[k] for k in range(N + 1)]), P(ibu), P(lbu), P(ubu),
                      P([self.bC[k] for k in range(N + 1)]), P(D_), P([self.blg[k] for k in range(N + 1)]),
                      P([self.bug[k] for k in range(N + 1)]), P(Zl_), P(Zl_), P(zl_), P(zl_), P(isx), P(isr),
                      P(ls_), P(ls_)]
        k_ += self.p_all
        lib.d_ocp_qp_set_all.argtypes = [C.c_void_p] * 27
        lib.d_ocp_qp_set_all.restype = None
        # solution buffers
        self.sx = np.zeros((N + 1, 5))
        self.su = np.zeros((N + 1, 1))
        msz = 64
        self.sbuf = [np.zeros((N + 1, msz)) for _ in range(9)]
        self.p_sol = [P([self.su[k] for k in range(N + 1)]), P([self.sx[k] for k in range(N + 1)])] + \
                     [P([b[k] for k in range(N + 1)]) for b in self.sbuf]
        k_ += [self.sbuf, self.p_sol]
        lib.d_ocp_qp_sol_get_all.argtypes = [C.c_void_p] * 12
        lib.d_ocp_qp_sol_get_all.restype = None

    def solve_all(self):
        """buffers bA, bB, bb, bQ, bq, bC, blg, bug, blbx, bubx filled by the caller"""
        lib, C = self.lib, self.C
        lib.d_ocp_qp_set_all(*self.p_all, self.qp)
        lib.d_ocp_qp_ipm_solve(self.qp, self.sol, self.arg, self.ws)
        lib.d_ocp_qp_ipm_get_status(self.ws, C.addressof(self.status))
        lib.d_ocp_qp_ipm_get_iter(self.ws, C.addressof(self.iters))
        lib.d_ocp_qp_sol_get_all(self.sol, *self.p_sol)
        return self.status.value, self.iters.value, self.sx.copy()

    def solve(self, A, B, b, Q, q, Cm, lg, ug, x0, lbx, ubx, lbu, ubu):
        """A (N,nx,nx) B (N,nx,nu) b (N,nx) Q (N+1,nx,nx) q (N+1,nx) Cm (N+1,ng,nx) lg/ug (N+1,ng); x0 fixed at
        stage 0 through lbx/ubx of stage 0; lbx/ubx (N+1, nbx) for later stages, lbu/ubu (N, nbu)"""
        lib, qp, N = self.lib, self.qp, self.N
        At = np.ascontiguousarray(np.transpose(A, (0, 2, 1)))           # column-major blocks
        Bt = np.ascontiguousarray(np.transpose(B, (0, 2, 1)))
        Qt = np.ascontiguousarray(np.transpose(Q, (0, 2, 1)))
        Ct = np.ascontiguousarray(np.transpose(Cm, (0, 2, 1)))
        b = np.ascontiguousarray(b)
        q = np.ascontiguousarray(q)
        lg = np.ascontiguousarray(lg)
        ug = np.ascontiguousarray(ug)
        lbx = np.ascontiguousarray(lbx)
        ubx = np.ascontiguousarray(ubx)
        lbu = np.ascontiguousarray(lbu)
        ubu = np.ascontiguousarray(ubu)
        x0 = np.ascontiguousarray(x0, dtype=float)
        pA, pB, pQ, pC = At.ctypes.data, Bt.ctypes.data, Qt.ctypes.data, Ct.ctypes.data
        pb, pq, plg, pug = b.ctypes.data, q.ctypes.data, lg.ctypes.data, ug.ctypes.data
        plx, pux, plu, puu = lbx.ctypes.data, ubx.ctypes.data, lbu.ctypes.data, ubu.ctypes.data
        sA, sB, sQ, sC = At.strides[0], Bt.strides[0], Qt.strides[0], Ct.strides[0]
        sb, sq, sg, sx, su = b.strides[0], q.strides[0], lg.strides[0], lbx.strides[0], lbu.strides[0]
        for k in range(N + 1):
            if k < N:
                lib.d_ocp_qp_set_A(k, pA + k * sA, qp)
                lib.d_ocp_qp_set_B(k, pB + k * sB, qp)
                lib.d_ocp_qp_set_b(k, pb + k * sb, qp)
                lib.d_ocp_qp_set_lbu(k, plu + k * su, qp)
                lib.d_ocp_qp_set_ubu(k, puu + k * su, qp)
            lib.d_ocp_qp_set_Q(k, pQ + k * sQ, qp)
            lib.d_ocp_qp_set_q(k, pq + k * sq, qp)
            if k == 0:
                lib.d_ocp_qp_set_lbx(0, x0.ctypes.data, qp)
                lib.d_ocp_qp_set_ubx(0, x0.ctypes.data, qp)
            else:
                lib.d_ocp_qp_set_lbx(k, plx + k * sx, qp)
                lib.d_ocp_qp_set_ubx(k, pux + k * sx, qp)
            if self.ng[k]:
                lib.d_ocp_qp_set_C(k, pC + k * sC, qp)
                lib.d_ocp_qp_set_lg(k, plg + k * sg, qp)
                lib.d_ocp_qp_set_ug(k, pug + k * sg, qp)
        lib.d_ocp_qp_ipm_solve(qp, self.sol, self.arg, self.ws)
        C = self.C
        lib.d_ocp_qp_ipm_get_status(self.ws, C.addressof(self.status))
        lib.d_ocp_qp_ipm_get_iter(self.ws, C.addressof(self.iters))
        X = self.out_x
        px, sxx = X.ctypes.data, X.strides[0]
        get_x, sol = lib.d_ocp_qp_sol_get_x, self.sol
        for k in range(N + 1):
            get_x(k, sol, px + k * sxx)
        return self.status.value, self.iters.value, X.copy()


class CorridorMPC:
    def __init__(self, **kw):
        p = dict(
            # speed
            v_max=8.5, a_lat=7.5, a_lat_hard=7.5, a_acc=6.0, a_dec=9.0, c_plan=0.6, c_emerg=0.3, a_lat_fb=99.0,
            acc_corner_min=1.0, gain_comp=False,
            m_vis=0.15, m_blk=0.30, jerk=60.0, v_min_plan=0.8, vis_tol=0.12, vis_start=0.6,
            t_react=0.04, v_end_scale=1.0,
            # geometry / margins
            m_hard=0.04, m_hard_v=0.006, c_pref=0.12, wall_clear=0.16, wall_clear_v=0.02, wall_clear_vmax=4.0, wall_pref=0.28,
            obs_pad_s=0.06, obs_pad_d=0.02, hidden_depth=0.43, min_width=0.30, w_max=0.43, occl_ext=True, occl_fix=True,
            box_mono=False,
            # perception
            obs_edt=0.09, r_max=9.0, link=0.22, blind_after_wrap=0.07, carve=True,
            # DP
            ds=0.2, n=40, dd=0.05, dp_ds=0.5, dp_kmax=6, dp_dmax=1.8, dp_vmax=6.0, dp_adec=4.0, dp_near=0.6, p_near=30.0,
            w_dev=1.0, w_sl=2.0, w_cv=20.0, dp_kcap=0.8, dp_vfloor=2.0,
            w_soft=40.0, w_wall=10.0,
            p_flip=3.0, p_flip_hard=40.0, commit_t=0.45, commit_d=1.0, commit_ramp=1.5, young_t=0.15,
            dp_kgeo=0.95, w_geo=10.0,
            cluster_ds=3.0, release_margin=0.0, hard_block_v=1.5, min_gap_w=0.06, min_gap_keep=0.0, escape=False, gap_fallback=True,
            # MPC
            q_d=2.0, q_mu=0.05, q_a=0.02, r_w=0.02, rho1=200.0, rho2=2000.0, rhoa1=50.0, rhoa2=50.0,
            slack_brake=0.03, lpts=(0.0, 0.22, 0.435), slack_near=20.0, slack_len=1.5, slack_reach=2.0,
            slack_bad=0.15,
            osqp_eps=1e-3, osqp_iter=250, osqp_rho_int=25, osqp_rho=0.1, osqp_tlim=0.006, shift_ws=False,
            solver='hpipm', hp_iter=30, hp_all=True, split=True, hp_tol=(1e-4, 1e-5, 1e-5, 1e-5), hp_mode=1, reuse_prof=True, ref_dp=True,
            # estimation / timing
            k_obs=0.5, t_lat=0.03, t_buf=0.02, creep_after=1.0, creep_v=0.6,
            # online latency estimate (t_lat is only the prior): candidates lat_min..lat_max in 10 ms steps
            lat_est=True, lat_min=0.0, lat_max=0.15, lat_tau=4.0, lat_gain=0.5, lat_cap=0.3, lat_hyst=0.05,
            # speed command for the mean acceleration over the hold period (P-controller response)
            hold_comp=True,
            # tyre-stiffness scale c (mu * C_S relative to the nominal tyre): initial value, then estimated online
            c_tyre=1.0,
            # measurement filters (adaptive gains, see _gain) and online tyre-stiffness / extra-delay estimate
            pose_filt=True, noise_alpha=0.03, noise_n0=25, q_pos=0.002 ** 2, q_yaw=0.001 ** 2, q_v=0.02 ** 2,
            q_r=0.03 ** 2, beta_sig=0.02, pose_gate=0.3,
            c_adapt=True, c_grid=(0.6, 1.4, 9), c_tau=3.0, c_prior=1.0, c_vmin=2.0, l_grid=(-0.02, 0.05, 8),
            l_prior=100.0, c_tables=True,
            acc_model=True, t_call=0.02, em_jerk=400.0, yaw_lim=0.0, gg=0.0, gg_min=0.25, v_vis_safe=2.5, swerve_w=0.0, a_acc_launch=0.0, v_launch=3.0, resolve_da=2.0, resolve_n=10, vis_obs_stop=False,
            debug=False,
            # FIXUP (2026-11-12): soft emergency at low speed (slack-only emergency with slack <= soft_em becomes a
            # soft_em_v speed cap once vv <= soft_em_arm); lat_est switch rule relative to the cost above the speed-noise
            # floor; adaptive estimators gated on 'the car follows the commands' (fg_win s open-loop check, rollback);
            # pose / yaw outlier gate (pose_out sigmas + pose_out_floor); stall-adaptive lateral acceleration (stall_db)
            soft_em=0.08, soft_em_v=1.0, soft_em_arm=1.5, lat_noise_floor=True,
            fg_check=True, fg_win=0.5, fg_thr=0.3, fg_thr_r=0.35, fg_k=4.0, fg_n=3,
            pose_out=4.0, pose_out_floor=0.03, yaw_out_floor=0.02,
            stall_db=0.1, stall_min=0.85,
            line_ktrig=0.9, line_kmax=1.0, line_dkmax=2.0, line_clr=0.35, line_wdev=0.001,
            jit=True,              # compiled (numba) kernels for the per-call numerics; False: the numpy reference code
            map_name='hyhall12',   # any stack_master/maps/<name> with global_waypoints.json
            # reference line shipped with the driver (json: arrays s, x, y[, vx] or a wpnts list / map-style file);
            # None: the map raceline.  Relative paths are resolved against the avoid/ directory.
            raceline_file=None,
            # delay-adaptive a_lat: [t0, t1, a1] (see header); None = v2
            lat_alat=None,
        )
        for k, v in kw.items():
            if k not in p:
                raise KeyError(f'unknown parameter {k}')
            p[k] = v
        self.p = p
        for k, v in p.items():
            setattr(self, k, v)
        self.lpts = tuple(self.lpts)
        self.wc_eff = self.wall_clear
        self.mapdir = os.path.join(os.path.dirname(MAPDIR), self.map_name)
        self._load_occ()
        self._load_track()
        self._load_map()
        self._build_qp()
        self.hp = None
        if self.solver == 'hpipm':
            try:
                self.hp = self._build_hpipm()
            except Exception:          # library not available: fall back to OSQP (same QP, slower)
                self.hp = None
        ang = ANG_MIN + ANG_INC * np.arange(1080)
        self.ca, self.sa = np.cos(ang), np.sin(ang)
        _hits_k(np.ones(1080), 0.0, 0.0, 1.0, 0.0, self.ca, self.sa, 9.0, self.ox, self.oy, self.res, self.W, self.H,
                self.edt, 0.09, self.mres, self.MW)
        c1 = np.ones(1, dtype=np.int64)
        _merge_k(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int16), *(np.zeros(0, dtype=np.float32),) * 4,
                 np.zeros(0, dtype=np.bool_), c1, self.cell_x, self.cell_y, self.cell_s, self.cell_d, self.cell_ok)
        _carve_k(np.ones(1, dtype=np.int16), np.ones(1, dtype=np.float32), c1 * 500, np.ones(1080))
        # usable decelerations (positive): keep WB + k_us(a) v^2 >= c * WB (yaw-stable, steerable); one set of tables
        # per stiffness of the estimator grid, interpolated for the current estimate (_set_tables)
        self.tab_v = np.arange(0.0, 15.0001, 0.05)
        self.tabs_c = np.linspace(*self.c_grid[:2], int(self.c_grid[2]))
        self.tabs = np.array([self._decel_tables(c) for c in self.tabs_c])
        self.tab_c = None
        self._set_tables(self.c_tyre)
        # compile / load the numba kernels now, not inside act()
        _dp_core(np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3), dtype=np.bool_), np.array([-1, 0, 1]),
                 np.ones(3, dtype=np.int64), np.zeros(3), np.zeros(3), np.full((3, 3), -1.0), np.ones((3, 3)), 1.0)
        z2 = np.zeros(2)
        _obs_main_k(np.zeros((1, 3)), 0.03, 0.02, 0.0, 2, 1.0, 0.0)
        _obs_beta_k(2, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, z2, z2, np.zeros(2, dtype=np.bool_), 1.0)
        _mm_prop(np.zeros(3), np.zeros(3), np.zeros(1), np.ones(3), np.zeros(1), 2, np.zeros((1, 3)), 0.03, 0.02, 0.0,
                 1, 1.0, 1.0, np.zeros(1), np.zeros(1), np.ones(1))
        _frenet_k(self.fx, self.fy, 0.0, 0.0, -1)
        _predict_k(np.zeros((1, 3)), 0.03, 0.02, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.05, 0.0, 0.0, 1.0)
        z32 = np.zeros(2, dtype=np.float32)
        for ok_ in (np.ones(2, dtype=np.bool_), np.zeros(2, dtype=np.bool_)):
            _boxes_k(np.array([0, 1]), ok_, z32, z32, z32, z32, z32, z32, np.array([5, 1079]), np.ones(1080), self.MW,
                     self.W, 0.0, float(self.L), 9.0, 0.22)
        _passes(np.ones(3), np.ones(2), self.tab_plan, 1.0, 1.0, np.zeros(3), 7.0, 0.0, 0.2)
        _fwd(np.ones(3), np.ones(2), 1.0, 1.0, self.tab_plan)
        _stable_accel_k(np.ones(3), np.ones(2), self.tab_plan, 1.0)
        _cluster_k(np.zeros(3), np.zeros(3), 0.1)
        _cmd_sim(np.ones(3), np.arange(3.0), np.ones(2), 1.0, 0.0, 1.0, 1.0, self.tab_emerg, 0.02)
        _lat_sim_k(np.zeros(2), np.ones(2), np.array([0.0, 0.03]), np.ones(2), 0.0, 2)
        _expm_k(np.zeros((2, 7, 7)))
        _expm_fr_k(np.zeros((2, 7, 7)))
        # latency hypotheses and their forgotten innovation sums; kept over reset() (a property of the system)
        self.lat_c = np.round(np.arange(self.lat_min, self.lat_max + 1e-9, 0.01), 3)
        self.lat_J = np.zeros(self.lat_c.size)
        self.lat_j = int(np.argmin(np.abs(self.lat_c - self.t_lat)))
        self.lat_W = 0.0            # forgotten sample count of lat_J (noise floor of the switch rule)
        self.n_foreign = 0          # windows in which the car did not follow the commands (estimators rolled back)
        self.n_bad_scan = 0         # calls with a scan of the wrong length (handled like an invalid sample)
        # measured call period (median of the recent gaps between calls)
        self.t_period = float(self.t_call)
        self.gaps = []
        self.t_last_call = None
        # constant inputs of the compiled MPC construction
        self.lpts_a = np.array(self.lpts, dtype=np.float64)
        kc = int(round(min(max(self.wall_clear, 0.0), 0.6) / 0.01))
        self.wlo_c = np.ascontiguousarray(self.wlo_tab[kc])
        self.whi_c = np.ascontiguousarray(self.whi_tab[kc])
        kp = int(round(min(max(self.wall_pref, 0.0), 0.6) / 0.01))
        self.wlo_p = np.ascontiguousarray(self.wlo_tab[kp])
        self.whi_p = np.ascontiguousarray(self.whi_tab[kp])
        for bxw in (np.zeros((0, 4)), np.array([[1.0, 1.5, -0.2, 0.2]])):
            nb_ = bxw.shape[0]
            _dp_k(0.0, 0.0, 0.0, 3.0, int(self.n), float(self.ds), float(self.dp_ds), float(self.dd), int(self.dp_kmax),
                  self.wlo_c, self.whi_c, self.wlo_p, self.whi_p, float(self.fds), 1.8, 1.0, 10.0, bxw,
                  np.ones(nb_, dtype=np.int64), np.ones(nb_), np.zeros(nb_, dtype=np.bool_), 0.2, 0.6, 30.0, 0.12,
                  40.0, True, 0.06, 4.0, 6.0, 2.0, 7.5, 0.8, 20.0, 2.0, 1.0, 0.0, self.fk, float(self.L), 0.95, 10.0)
        if self.hp is not None:
            st0 = np.arange(self.n + 1) * self.ds
            z1, zn = np.zeros(self.n + 1), np.zeros(self.n)
            hp = self.hp
            _mpc_hp_k(st0, np.zeros(5), np.zeros((self.n + 1, 5)), z1, z1 + 1.0, zn, zn + self.ds, z1, self.lpts_a,
                      self.wlo_c, self.whi_c, self.nf, self.fds, np.zeros((1, 4)), np.ones(1, dtype=np.int64), 0.1,
                      0.8, 0.1, 1.0, 1.0, 1.0, 1.0, self.tab_plan, hp.bA, hp.bB, hp.bb, hp.bQ, hp.bq, hp.bC, hp.blg,
                      hp.bug, hp.blbx, hp.bubx, np.zeros((self.n, 3)), np.zeros((self.n, 3)), np.zeros((self.n, 3)),
                      np.zeros(self.n), 1.0)
            _hp_slack_k(np.zeros((self.n + 1, 5)), np.zeros((self.n, 3)), np.zeros((self.n, 3)), np.zeros((self.n, 3)))
        self._no_kyaw = np.zeros(0)
        stw = np.arange(self.n + 1) * self.ds
        _plan_prep_k(stw, self.fk, float(self.L), float(self.fds), stw, np.zeros((self.n + 1, 5)), stw * 0.0,
                     np.zeros(5), stw, stw[:-1] * 0.0 + self.ds, stw * 0.0 + 1.0, 1.0, 0.0, 6.0, 60.0,
                     self.tab_emerg, 0.02)
        z1 = np.zeros(self.n + 1)
        for sblk in (math.nan, 1.0):
            _speed_k(np.arange(self.n + 1) * self.ds, z1, z1, z1, z1, 1.0, np.ones(1080), 0.0, 0.0, 0.0, sblk,
                     self._no_kyaw, self.fx, self.fy, self.fnx, self.fny, self.fpsi, self.fvx, float(self.L),
                     float(self.fds), 0.2, 8.0, 7.0, 1.0, 0.0, 6.0, 0.0, 0.2, 6.0, 0.1, 0.5, 0.1, 0.1, 0.04, 2.5,
                     self.tab_plan, self.tab_emerg, self.tab_dstop, self.tab_v)
        self.reset()

    def _decel_tables(self, c):
        """planned / emergency deceleration limits and the emergency stopping distance over tab_v for tyre scale c"""
        ag = np.linspace(-A_MAX, 0.0, 952)
        fac = WB + k_us(ag, c)[None, :] * self.tab_v[:, None] ** 2
        out = []
        for cw in (self.c_plan, self.c_emerg):
            ok = fac >= cw * WB
            amin = np.where(ok.any(1), ag[np.argmax(ok, axis=1)], 0.0)
            out.append(np.maximum(np.minimum(self.a_dec, -amin), 1.0))
        integ = self.tab_v / out[1]
        out.append(np.concatenate([[0.0], np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(self.tab_v))]))
        return out

    def _set_tables(self, c):
        if self.tab_c is not None and abs(c - self.tab_c) < 0.005:
            return
        g = self.tabs_c
        i = min(max(int(np.searchsorted(g, c)) - 1, 0), g.size - 2)
        w = min(max((c - g[i]) / (g[i + 1] - g[i]), 0.0), 1.0)
        tp, te, td = (1.0 - w) * self.tabs[i] + w * self.tabs[i + 1]
        self.tab_plan, self.tab_emerg, self.tab_dstop = np.ascontiguousarray(tp), np.ascontiguousarray(te), \
            np.ascontiguousarray(td)
        self.tab_c = c

    # ------------------------------------------------------------------ static precompute
    def _load_track(self):
        w = json.load(open(f'{self.mapdir}/global_waypoints.json'))['global_traj_wpnts_iqp']['wpnts']
        s = np.array([q['s_m'] for q in w])
        x = np.array([q['x_m'] for q in w])
        y = np.array([q['y_m'] for q in w])
        vx = np.array([q['vx_mps'] for q in w])
        x[-1], y[-1] = x[0], y[0]
        if self.raceline_file:
            s, x, y, vx = self._custom_line(s, x, y, vx, float(w[0]['psi_rad']))
        self._set_line(s, x, y)
        self.fvx = np.interp(self.fs, s, vx)
        self.line_repaired = False
        if self.line_kmax > 0.0 and float(np.max(np.abs(self.fk))) > self.line_ktrig * math.tan(STEER_MAX) / WB:
            # the raceline of an imported map can be tighter than the car can steer at all (small_hall: kappa 5.5
            # against tan(delta_max) / WB = 1.35): every Frenet quantity (DP curvature, MPC linearisation, wall tables)
            # then works against an impossible reference.  Replace it by the nearest line with |kappa| <= line_kmax
            # and wall clearance.  A line that is only marginally tight (max kappa 1.0 .. 1.2) is kept: it is drivable.
            xo, yo = self.fx.copy(), self.fy.copy()
            xr, yr = self._repair_line()
            sr = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xr), np.diff(yr)))])
            self._set_line(sr, xr, yr)
            # speed of the original line at the nearest original point (only used for the horizon end speed)
            _, io = cKDTree(np.stack([xo, yo], 1)).query(np.stack([self.fx, self.fy], 1))
            self.fvx = np.interp(io * (s[-1] / xo.size), s, vx)
            self.line_repaired = True
        self.tree = cKDTree(np.stack([self.fx, self.fy], 1))

    def _custom_line(self, s_m, x_m, y_m, vx_m, psi0):
        """reference line from raceline_file, closed, re-anchored so that s = 0 lies on the map raceline's finish
        line (through map wpnt 0, normal to its heading); vx from the file or, if absent, the map raceline's vx at
        the nearest map point"""
        path = self.raceline_file
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
        d = json.load(open(path))
        if isinstance(d, dict) and 'global_traj_wpnts_iqp' in d:
            d = d['global_traj_wpnts_iqp']
        if isinstance(d, dict) and 'wpnts' in d:
            d = d['wpnts']
        if isinstance(d, list):
            x = np.array([q['x_m'] for q in d], dtype=float)
            y = np.array([q['y_m'] for q in d], dtype=float)
            v = np.array([q['vx_mps'] for q in d], dtype=float) if 'vx_mps' in d[0] else None
        else:
            x, y = np.asarray(d['x'], dtype=float), np.asarray(d['y'], dtype=float)
            v = np.asarray(d['vx'], dtype=float) if 'vx' in d else None
        if math.hypot(x[-1] - x[0], y[-1] - y[0]) < 1e-6:      # drop the closing duplicate
            x, y = x[:-1], y[:-1]
            v = v[:-1] if v is not None else None
        # finish line: through map point 0, normal to the map heading; find where the closed line crosses it going
        # forward (closest crossing to map point 0) and start there
        hx, hy = math.cos(psi0), math.sin(psi0)
        xc, yc = np.append(x, x[0]), np.append(y, y[0])
        f = (xc - x_m[0]) * hx + (yc - y_m[0]) * hy           # signed distance along the map heading
        best = None
        for i in np.flatnonzero((f[:-1] < 0.0) & (f[1:] >= 0.0)):
            a = -f[i] / (f[i + 1] - f[i])
            px, py = xc[i] + a * (xc[i + 1] - xc[i]), yc[i] + a * (yc[i + 1] - yc[i])
            dist = math.hypot(px - x_m[0], py - y_m[0])
            if best is None or dist < best[0]:
                best = (dist, i, a, px, py)
        if best is None or best[0] > 3.0:
            raise ValueError(f'raceline_file {path}: does not cross the finish line near map s = 0')
        _, i, a, px, py = best
        n = x.size
        order = (np.arange(n) + i + 1) % n
        x2 = np.concatenate([[px], x[order], [px]])
        y2 = np.concatenate([[py], y[order], [py]])
        seg = np.hypot(np.diff(x2), np.diff(y2))
        keep = np.concatenate([[True], seg > 1e-6])           # a crossing exactly on a sample: no zero-length step
        x2, y2 = x2[keep], y2[keep]
        s2 = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x2), np.diff(y2)))])
        if v is not None:
            vc = np.append(v, v[0])
            v0 = vc[i] + a * (vc[i + 1] - vc[i])
            v2 = np.concatenate([[v0], v[order], [v0]])[keep]
        else:
            _, io = cKDTree(np.stack([x_m, y_m], 1)).query(np.stack([x2, y2], 1))
            v2 = vx_m[io]
        self.line_file = path
        return s2, x2, y2, v2

    def _set_line(self, s, x, y):
        """closed reference line through (x, y) at arc length s (last point == first point), sampled every ~5 cm"""
        L = float(s[-1])
        csx = CubicSpline(s, x, bc_type='periodic')
        csy = CubicSpline(s, y, bc_type='periodic')
        nf = int(round(L / 0.05))      # 840 on hyhall12
        fs = np.arange(nf) * (L / nf)
        dx, dy = csx(fs, 1), csy(fs, 1)
        ddx, ddy = csx(fs, 2), csy(fs, 2)
        self.L, self.nf, self.fds = L, nf, L / nf
        self.fs = fs
        self.fx, self.fy = csx(fs), csy(fs)
        self.fpsi = np.arctan2(dy, dx)
        self.fk = (dx * ddy - dy * ddx) / np.power(dx * dx + dy * dy, 1.5)
        self.ftx, self.fty = np.cos(self.fpsi), np.sin(self.fpsi)
        self.fnx, self.fny = -self.fty, self.ftx

    def _clearance(self, x, y):
        """distance to the nearest occupied map cell (bilinear on the distance transform)"""
        return ndimage.map_coordinates(self.edt, [(y - self.oy) / self.res - 0.5, (x - self.ox) / self.res - 0.5],
                                       order=1, mode='constant', cval=0.0)

    def _repair_line(self):
        """least-squares line close to the raceline with |kappa| <= line_kmax (Menger curvature of 20 cm chords)
        and a wall clearance of line_clr (or the raceline's own, if smaller); returns closed x, y"""
        h = 0.2
        n = int(round(self.L / h))
        h = self.L / n
        i0 = np.rint(np.arange(n) * h / self.fds).astype(int) % self.nf
        x0, y0 = self.fx[i0], self.fy[i0]
        kmax = self.line_kmax
        clr = np.minimum(self.line_clr, self._clearance(x0, y0))

        def res(z):
            x, y = z[:n], z[n:]
            ax, ay = x - np.roll(x, 1), y - np.roll(y, 1)
            bx, by = np.roll(x, -1) - x, np.roll(y, -1) - y
            la = np.hypot(ax, ay)
            k = 2.0 * (ax * by - ay * bx) / np.maximum(la * np.hypot(bx, by) * np.hypot(ax + bx, ay + by), 1e-9)
            dk = (k - np.roll(k, 1)) / h          # curvature rate: the steering rate is limited too
            return np.concatenate([math.sqrt(h) * np.maximum(np.abs(k) - kmax, 0.0),
                                   math.sqrt(h) * 0.2 * np.maximum(np.abs(dk) - self.line_dkmax, 0.0),
                                   3.0 * np.maximum(clr - self._clearance(x, y), 0.0),
                                   math.sqrt(self.line_wdev) * (x - x0), math.sqrt(self.line_wdev) * (y - y0),
                                   la - h])
        rows, cols = [], []
        ii = np.arange(n)
        for dj in (-1, 0, 1):                       # curvature: 3 points
            rows += [ii, ii]
            cols += [(ii + dj) % n, n + (ii + dj) % n]
        for dj in (-2, -1, 0, 1):                   # curvature rate: 4 points
            rows += [n + ii, n + ii]
            cols += [(ii + dj) % n, n + (ii + dj) % n]
        rows += [2 * n + ii, 2 * n + ii, 3 * n + ii, 4 * n + ii]
        cols += [ii, n + ii, ii, n + ii]
        for dj in (-1, 0):                          # spacing: 2 points
            rows += [5 * n + ii, 5 * n + ii]
            cols += [(ii + dj) % n, n + (ii + dj) % n]
        jac = sp.csr_matrix((np.ones(sum(r.size for r in rows)), (np.concatenate(rows), np.concatenate(cols))),
                            shape=(6 * n, 2 * n))
        r = least_squares(res, np.concatenate([x0, y0]), jac_sparsity=jac, method='trf', max_nfev=600,
                          xtol=1e-8, ftol=1e-10)
        x, y = r.x[:n], r.x[n:]
        return np.append(x, x[0]), np.append(y, y[0])

    def _load_occ(self):
        # same rule as the simulator (f1tenth_gym_ros map_selection.prefer_orig_map_yaml)
        y = f'{self.mapdir}/{self.map_name}_orig.yaml'
        if not os.path.exists(y):
            y = f'{self.mapdir}/{self.map_name}.yaml'
        meta = yaml.safe_load(open(y))
        img = np.array(Image.open(f"{self.mapdir}/{meta['image']}").convert('L').transpose(Image.FLIP_TOP_BOTTOM))
        occ = img <= 128
        self.res = float(meta['resolution'])
        self.ox, self.oy = float(meta['origin'][0]), float(meta['origin'][1])
        self.H, self.W = occ.shape
        self.edt = (ndimage.distance_transform_edt(~occ) * self.res).astype(np.float32)

    def _load_map(self):
        # obstacle memory grid (finer than the map)
        self.mres = self.res / 2.0
        self.MH, self.MW = self.H * 2, self.W * 2
        cc, rr = np.meshgrid(np.arange(self.MW), np.arange(self.MH))
        mx = self.ox + (cc.ravel() + 0.5) * self.mres
        my = self.oy + (rr.ravel() + 0.5) * self.mres
        dist, idx = self.tree.query(np.stack([mx, my], 1))
        ddx, ddy = mx - self.fx[idx], my - self.fy[idx]
        self.cell_x = mx.astype(np.float32)
        self.cell_y = my.astype(np.float32)
        self.cell_s = (self.fs[idx] + ddx * self.ftx[idx] + ddy * self.fty[idx]).astype(np.float32)
        self.cell_d = (ddx * self.fnx[idx] + ddy * self.fny[idx]).astype(np.float32)
        self.cell_ok = dist < 4.0
        # wall clearance along the normals -> free interval around the raceline, per clearance value
        dg = np.arange(-4.0, 4.0001, 0.01)
        px = self.fx[:, None] + dg[None, :] * self.fnx[:, None]
        py = self.fy[:, None] + dg[None, :] * self.fny[:, None]
        rows = (py - self.oy) / self.res - 0.5
        cols = (px - self.ox) / self.res - 0.5
        clr = ndimage.map_coordinates(self.edt, [rows.ravel(), cols.ravel()], order=1, mode='constant', cval=0.0)
        clr = clr.reshape(px.shape) - 0.03
        cl = np.arange(0.0, 0.61, 0.01)
        j0 = int(np.argmin(np.abs(dg)))
        lo = np.zeros((len(cl), self.nf))
        hi = np.zeros((len(cl), self.nf))
        for k, c in enumerate(cl):
            bad = clr < c
            badl = bad[:, :j0][:, ::-1]           # walking left (towards negative d) from the raceline
            badr = bad[:, j0:]
            il = np.where(badl.any(1), np.argmax(badl, axis=1), badl.shape[1])
            ir = np.where(badr.any(1), np.argmax(badr, axis=1), badr.shape[1])
            lo[k] = dg[np.clip(j0 - il, 0, len(dg) - 1)]
            hi[k] = dg[np.clip(j0 + ir - 1, 0, len(dg) - 1)]
        self.wlo_tab, self.whi_tab = lo, hi

    # ------------------------------------------------------------------ QP structure
    def _build_qp(self):
        N = self.n
        nl = len(self.lpts)
        X = lambda i, k: 5 * i + k
        Wv = lambda i: 5 * (N + 1) + i
        S = lambda i: 5 * (N + 1) + N + i
        T = lambda i: 5 * (N + 1) + N + (N + 1) + i
        self.nv = 5 * (N + 1) + N + 2 * (N + 1)
        self.iW0, self.iS0, self.iT0 = Wv(0), S(0), T(0)
        trip = []
        r = 0
        # init
        for k in range(5):
            trip.append((r, X(0, k)))
            r += 1
        self.row_dyn, self.k_dyn = r, len(trip)
        for i in range(N):
            for k in range(5):
                trip.append((r, X(i + 1, k)))
                for j in range(5):
                    trip.append((r, X(i, j)))
                trip.append((r, Wv(i)))
                r += 1
        self.row_cor, self.k_cor = r, len(trip)
        for i in range(1, N + 1):
            for _ in range(nl):
                trip += [(r, X(i, 0)), (r, X(i, 1)), (r, S(i))]
                trip += [(r + 1, X(i, 0)), (r + 1, X(i, 1)), (r + 1, S(i))]
                r += 2
        self.row_del, self.k_del = r, len(trip)
        for i in range(1, N + 1):
            trip.append((r, X(i, 4)))
            r += 1
        self.row_w, self.k_w = r, len(trip)
        for i in range(N):
            trip.append((r, Wv(i)))
            r += 1
        self.row_s, self.k_s = r, len(trip)
        for i in range(N + 1):
            trip.append((r, S(i)))
            r += 1
        for i in range(N + 1):
            trip.append((r, T(i)))
            r += 1
        self.row_lat, self.k_lat = r, len(trip)
        for i in range(1, N + 1):
            trip += [(r, X(i, 3)), (r, T(i))]
            trip += [(r + 1, X(i, 3)), (r + 1, T(i))]
            r += 2
        self.nrow = r
        rows = np.array([t[0] for t in trip])
        cols = np.array([t[1] for t in trip])
        self.ntrip = len(trip)
        A = sp.csc_matrix((np.arange(1, self.ntrip + 1, dtype=float), (rows, cols)), shape=(self.nrow, self.nv))
        A.sort_indices()
        assert A.nnz == self.ntrip
        self.A_perm = (A.data - 1).astype(int)
        self.A_sp = A
        vals = np.zeros(self.ntrip)
        vals[0:5] = 1.0
        vals[self.k_del:self.k_lat] = 1.0          # delta, w, slack bounds rows
        self.A_vals = vals
        P = sp.identity(self.nv, format='csc')
        self.P_sp = P
        self.qp = None

    def _build_hpipm(self):
        N, nl = self.n, len(self.lpts)
        nx = [5] * (N + 1)
        nu = [1] * N + [0]
        nbx = [5] + [1] * N
        nbu = [1] * N + [0]
        ng = [0] + [nl + 1] * N
        ns = [0] + [nl + 1] * N
        idxbx = [[0, 1, 2, 3, 4]] + [[4]] * N
        idxbu = [[0]] * N + [[]]
        idxs = [[]] + [[2 + j for j in range(nl + 1)]] * (N - 1) + [[1 + j for j in range(nl + 1)]]
        Zl, zl = [[]], [[]]
        for k in range(1, N + 1):
            fk = 1.0 + self.slack_near * math.exp(-k * self.ds / self.slack_len)
            Zl.append([2 * self.rho2 * fk] * nl + [2 * self.rhoa2])
            zl.append([self.rho1 * fk] * nl + [self.rhoa1])
        R = [np.array([[2 * self.r_w]])] * N + [np.zeros((0, 0))]
        hp = _Hpipm(N, nx, nu, nbx, nbu, ng, ns, idxbx, idxbu, idxs, Zl, Zl, zl, zl, R, iter_max=self.hp_iter,
                    tol=tuple(self.hp_tol), mode=self.hp_mode)
        hp.setup_all(nbx, nbu, idxbx, idxbu, idxs, Zl, zl, R)
        return hp

    def _qp_setup(self, Pd, q, Av, l, u):
        P = sp.diags(Pd, format='csc')
        A = self.A_sp.copy()
        A.data = Av[self.A_perm]
        self.qp = osqp.OSQP()
        self.qp.setup(P, q, A, l, u, verbose=False, warm_starting=True, polishing=False,
                      eps_abs=self.osqp_eps, eps_rel=self.osqp_eps, max_iter=self.osqp_iter, adaptive_rho=True,
                      adaptive_rho_interval=self.osqp_rho_int, scaling=10, check_termination=5,
                      rho=self.osqp_rho, time_limit=self.osqp_tlim)

    # ------------------------------------------------------------------ helpers
    def _fidx(self, s):
        return np.mod(np.rint(np.asarray(s) / self.fds).astype(int), self.nf)

    def _finterp(self, arr, s):
        u = np.mod(np.asarray(s, dtype=float), self.L) / self.fds
        i0 = np.floor(u).astype(int) % self.nf
        f = u - np.floor(u)
        return arr[i0] * (1 - f) + arr[(i0 + 1) % self.nf] * f

    def _xy(self, s, d):
        u = np.mod(np.asarray(s, dtype=float), self.L) / self.fds
        i0 = np.floor(u).astype(int) % self.nf
        i1 = (i0 + 1) % self.nf
        f = u - np.floor(u)
        bx = self.fx[i0] * (1 - f) + self.fx[i1] * f
        by = self.fy[i0] * (1 - f) + self.fy[i1] * f
        nx = self.fnx[i0] * (1 - f) + self.fnx[i1] * f
        ny = self.fny[i0] * (1 - f) + self.fny[i1] * f
        return bx + d * nx, by + d * ny

    def _frenet(self, x, y, guess):
        if self.jit:
            i = int(_frenet_k(self.fx, self.fy, float(x), float(y), -1 if guess is None else int(guess)))
            if guess is not None and (i == (guess - 12) % self.nf or i == (guess + 39) % self.nf):
                return self._frenet(x, y, None)     # window edge: see below
            dx, dy = x - self.fx[i], y - self.fy[i]
            return self.fs[i] + dx * self.ftx[i] + dy * self.fty[i], dx * self.fnx[i] + dy * self.fny[i], i
        if guess is None:
            i = int(np.argmin((self.fx - x) ** 2 + (self.fy - y) ** 2))
        else:
            cand = np.arange(guess - 12, guess + 40) % self.nf
            j = int(np.argmin((self.fx[cand] - x) ** 2 + (self.fy[cand] - y) ** 2))
            if j == 0 or j == cand.size - 1:
                # nearest point on the edge of the window (-0.6 .. +2 m): the car is beyond it (stall of the node at
                # speed, pose jump); the edge point is a wrong local minimum (d off by its curvature, and it can stay
                # locked there) -> global search
                return self._frenet(x, y, None)
            i = int(cand[j])
        dx, dy = x - self.fx[i], y - self.fy[i]
        s = self.fs[i] + dx * self.ftx[i] + dy * self.fty[i]
        d = dx * self.fnx[i] + dy * self.fny[i]
        return s, d, i

    def _wall_bounds(self, s, c):
        k = int(round(min(max(c, 0.0), 0.6) / 0.01))
        i = self._fidx(s)
        return self.wlo_tab[k, i], self.whi_tab[k, i]

    # ------------------------------------------------------------------ public
    def reset(self):
        sn = getattr(self, 'fg_sn', None)
        if sn is not None:
            # what the adaptive estimators learned since the last committed check window is not verified (another
            # source may have been driving until this /initialpose): back to the committed state (see _fg_window)
            self.lat_J, self.lat_j, self.t_lat, self.lat_W = sn[0].copy(), sn[1], sn[2], sn[3]
            self.cg_J, self.c_tyre, self.lat_extra = sn[4].copy(), sn[5], sn[6]
        self.fg_sn = None
        self.fg_ok = 0
        self.fi = None
        self.s_prev = None
        # memory counts on the map-sized grid (numpy path); the jit path keeps them aligned with the remembered cells
        self.mem = None if self.jit else np.zeros(self.MH * self.MW, dtype=np.int16)
        self._mem_clear()
        self.blind_until = -1.0
        self.hist = []              # (t_issue, delta_target, v_cmd); they act t_lat (+ t_buf for steering) later
        self.lat_prev = None
        self.lat_v = None
        self.delta_est = 0.0
        self.beta_est = 0.0
        self.last = None            # (t, x, y, yaw, v, r)
        self.lastf = None           # filtered (x, y, yaw, v, r)
        # noise statistics and the stiffness estimate describe the car / sensors: kept across resets
        if not hasattr(self, 'nstat'):
            self.nstat = {k: [0.0, 0.0, None, 0] for k in ('v', 'r', 'yaw', 'pos')}
            self.c_grid_v = np.linspace(*self.c_grid[:2], int(self.c_grid[2]))
            self.l_grid_v = np.linspace(*self.l_grid[:2], int(self.l_grid[2]))
            self.cg_J = np.zeros(self.l_grid_v.size * self.c_grid_v.size)
            self.lat_extra = 0.0
            self.cg_prior = self.c_prior * (self.c_grid_v[None, :] - 1.0) ** 2 + self.l_prior * self.l_grid_v[:, None] ** 2
        for st in self.nstat.values():
            st[2] = None
        # models: [0] = the estimator's own (c_tyre, assumed delay), then the grid (extra steering delay x stiffness)
        nl_, nc_ = self.l_grid_v.size, self.c_grid_v.size
        self.c_vec = np.concatenate([[self.c_tyre], np.tile(self.c_grid_v, nl_)])
        self.cg_bb = np.zeros(self.c_vec.size)
        self.cg_rr = np.zeros(self.c_vec.size)
        self.cg_b = np.zeros(nl_ * nc_)
        self.cg_r = np.zeros(nl_ * nc_)
        self.cg_d = np.zeros(nl_)
        self.cg_on = False
        self.sol = None             # previous MPC solution: (stations, X (N+1,5), w (N))
        self.prev_sides = []
        self.seen_boxes = []
        self.box_ext = []
        self.box_conf = []
        self.conf_idx = {}          # s bucket -> indices into box_conf (lookup cost independent of lap length)
        self._ps_cache = [None, None, None]   # (list, bucket index, memo) of prev_sides / seen_boxes for _first_match
        self._sb_cache = [None, None, None]
        self.box_age = []
        self.qp = None
        self.z_prev = None
        self.qp_info = None
        self.prof = None
        self.t_stop = None
        self.dp_res = None
        self.plan = None
        self.phase = 0
        self.sol_bad = False
        if self.debug:
            self.dbg = getattr(self, 'dbg', None) or []
        else:
            self.dbg = None
        self.n_fail = 0
        self.n_resolve = 0
        self.a_acc_eff = self.a_acc
        self.a_cmd = 0.0
        self.t_prev_cmd = None
        self.t_off = None           # internal time base (see _clock)
        self.t_int = None
        self.pose_prev = None       # (t, x, y, v) of the previous valid call (pose-jump detection)
        self.n_invalid = 0          # consecutive calls with a non-finite state sample
        self.fg = None              # open-loop window of the command-following check (see _fg_window)
        self.stalls = []            # (t, extra hold time) of the call gaps > 1.75 periods in the last 5 s
        self.out_pos = False        # previous pose / yaw sample was rejected as an outlier (see _observe)
        self.out_yaw = False
        self._no_latch = False      # teleport call: perception skipped, the extent latches are not updated

    def _clock(self, t):
        """internal time: relative to the first call after reset() and never running backwards.  A clock that jumped
        back (restarted sim / bag, stepped wall clock) left blind_until, t_stop, first-seen times and the command
        history in the future: perception stayed blind for as long as the clock had jumped back.  Relative time also
        keeps the 1e-9 hand-off tolerances meaningful with wall-clock stamps (~1.7e9 s)."""
        if self.t_off is None:
            self.t_off = t
        ti = t - self.t_off
        if self.t_int is not None and ti < self.t_int:
            self.t_off -= self.t_int - ti          # continue from the last internal time (a repeated stamp)
            ti = self.t_int
        self.t_int = ti
        return ti

    # ------------------------------------------------------------------ actuator models
    def _hist_arr(self):
        if getattr(self, '_H', None) is not None:
            return self._H
        h = self.hist
        return np.array(h, dtype=np.float64).reshape(len(h), 3)

    def _targets(self, tt):
        """steering target and speed command active at time tt (command latency t_lat, steering + t_buf)"""
        dl, vc = None, None
        ts0, tv0 = tt + 1e-6 - self.t_lat - self.t_buf, tt + 1e-6 - self.t_lat
        for (ti, d_, v_) in self.hist:
            if ti <= ts0:
                dl = d_
            if ti <= tv0:
                vc = v_
        return dl, vc

    def _lat_update(self, t, v):
        """online estimate of the command latency (multiple-model): the simulator's speed P-controller (p_accel) is
        driven by this driver's own speed commands, delayed by each candidate latency; every hypothesis keeps its
        own speed (corrected by lat_gain of its innovation, like an observer) and the hypothesis with the smallest
        exponentially forgotten sum of squared innovations is the estimate.  Exact for the simulator's longitudinal
        model; needs no tyre model (independent of mu / cornering stiffness)."""
        prev, self.lat_prev = self.lat_prev, (t, v)
        if prev is None or not self.hist:
            return
        t0, v0 = prev
        n = int(round((t - t0) / 0.01))
        if n <= 0 or n > 20 or v0 < 0.5 or v < 0.5:
            self.lat_v = None
            return
        # speed-noise variance measured by the pose filter (innovation correlation; 0 for exact speeds)
        R = self._noise_R('v') if self.lat_noise_floor else 0.0
        cap = max(self.lat_cap, 3.0 * math.sqrt(R))
        H = (self._H if self._H is not None else np.array(self.hist, dtype=np.float64)).T.copy()
        vL = np.full(self.lat_c.size, float(v0)) if self.lat_v is None else self.lat_v
        vL, ok = _lat_sim_k(H[0], H[2], self.lat_c, vL, float(t0), n)
        e = np.minimum(np.maximum(v - vL, -cap), cap)
        self.lat_v = vL + self.lat_gain * (v - vL)
        if not ok:
            return
        lam = math.exp(-(t - t0) / self.lat_tau)
        self.lat_J = lam * self.lat_J + e * e
        self.lat_W = lam * self.lat_W + 1.0
        j = int(np.argmin(self.lat_J))
        cur = self.lat_J[self.lat_j]
        # switch rule relative to the part of the cost above the noise floor R W that every hypothesis carries: with
        # speed noise, lat_hyst of the whole cost was more than the signal difference between hypotheses and the first
        # estimate never changed (speed noise 0.4: 0.04 for a whole run at true 0.08).  R = 0: the old rule.
        if self.lat_J[j] + 1e-5 < cur - self.lat_hyst * max(cur - R * self.lat_W, 0.0):
            self.lat_j = j
        self.t_lat = float(self.lat_c[self.lat_j])

    def _noise_R(self, key):
        """measurement-noise variance of one filter channel (innovation correlation, see _gain); 0 until known"""
        st = self.nstat[key]
        return max(-st[1], 0.0) if st[3] >= self.noise_n0 else 0.0

    def _fg_snap(self):
        return (self.lat_J.copy(), self.lat_j, self.t_lat, self.lat_W, self.cg_J.copy(), self.c_tyre, self.lat_extra)

    def _fg_window(self, t, v):
        """does the car follow this driver's commands?  Open-loop check over windows of fg_win s: every latency
        hypothesis predicts the speed from the window start with the issued speed commands (the simulator's
        P-controller, no observer correction).  When none of them ends within fg_thr + fg_k sqrt(R_v) of the measured
        speed, the car was not driven by these commands in that window (operator / mux priority, a second /drive
        publisher, car held or pushed): the latency and tyre-stiffness estimates are rolled back to the window start.
        Without the check they learned from such data, ended at their grid limits and kept them through reset()."""
        g = self.fg
        if not self.hist or v < 0.5:
            self.fg = None
            return
        if g is None:
            self._fg_start(t, v)
            return
        t_s, t_l, vO, sn = g
        n = int(round((t - t_l) / 0.01))
        ok = 0 < n <= 20
        if ok:
            H = (self._H if self._H is not None else np.array(self.hist, dtype=np.float64)).T.copy()
            vO, ok = _lat_sim_k(H[0], H[2], self.lat_c, vO, float(t_l), n)
        if not ok:
            self._fg_start(t, v)
            return
        if t - t_s < self.fg_win - 1e-6:
            self.fg = [t_s, t, vO, sn]
            return
        ev = float(np.min(np.abs(v - vO)))
        er = math.sqrt(float(np.min(self.fg_e2)) / self.fg_ne) if self.fg_ne > 0 else 0.0
        self.fg_diag = (ev, er)
        thr = self.fg_thr + self.fg_k * math.sqrt(self._noise_R('v'))
        thr_r = self.fg_thr_r + self.fg_k * math.sqrt(self._noise_R('r'))
        if ev > thr or er > thr_r:
            # rolled back to the last committed state; learning is committed again only after fg_n clean windows in
            # a row (one of the latency / stiffness hypotheses can fit a single foreign window by chance)
            self.lat_J, self.lat_j, self.t_lat, self.lat_W = sn[0].copy(), sn[1], sn[2], sn[3]
            self.cg_J, self.c_tyre, self.lat_extra = sn[4].copy(), sn[5], sn[6]
            self.n_foreign += 1
            self.fg_ok = 0
        else:
            self.fg_ok += 1
            if self.fg_ok >= self.fg_n:
                sn = self._fg_snap()
        self._fg_start(t, v, sn)

    def _fg_start(self, t, v, sn=None):
        if sn is None:
            # (re)start after low speed / a gap: the committed state is kept, a fresh check is armed
            sn = self.fg_sn if getattr(self, 'fg_sn', None) is not None else self._fg_snap()
        self.fg_sn = sn
        self.fg = [t, t, np.full(self.lat_c.size, float(v)), sn]
        self.fg_e2 = np.zeros(self.cg_J.size)
        self.fg_ne = 0

    @staticmethod
    def _servo_step(delta, tgt):
        if tgt is None:
            return delta
        diff = tgt - delta
        step = STEER_RATE * 0.01
        if abs(diff) > 1e-4:
            delta += max(-step, min(step, diff))
        return max(-STEER_MAX, min(STEER_MAX, delta))

    def _gain(self, key, e, qmin):
        """adaptive gain of a scalar random-walk filter.  e = (measured - model) increment between two successive
        calls: white measurement noise of variance R makes successive e anti-correlated, E[e_k e_k-1] = -R, while the
        model error q (random walk) is E[e^2] - 2R (innovation correlation, Mehra 1970).  Steady-state Kalman gain;
        exact measurements (the simulator, a good localisation) give R = 0 and gain 1 (no filtering, no lag)."""
        st = self.nstat[key]                         # [E e^2, E e_k e_k-1, e_prev, count]; e: tuple of components
        a = self.noise_alpha
        if st[2] is not None:
            st[1] += a * (sum(u * w for u, w in zip(e, st[2])) / len(e) - st[1])
            st[3] += 1
        st[0] += a * (sum(u * u for u in e) / len(e) - st[0])
        st[2] = e
        R = -st[1]
        if st[3] < self.noise_n0 or R <= 0.0:
            return 1.0
        q = max(st[0] - 2.0 * R, qmin)
        P = 0.5 * (q + math.sqrt(q * q + 4.0 * q * R))
        return P / (P + R)

    def _observe(self, t, x, y, yaw, v, r):
        """advance steering / sideslip estimates from the previous call to now.  Returns the measurement filtered
        against the model prediction (x, y, yaw, v, r), with gains from the measured noise level (gain 1 = raw)."""
        if self.last is None:
            self.last = (t, x, y, yaw, v, r)
            self.lastf = (x, y, yaw, v, r)
            self.cg_b[:] = self.beta_est
            self.cg_r[:] = r
            self.cg_d[:] = self.delta_est
            return x, y, yaw, v, r
        t0, x0, y0, yaw0, v0, r0 = self.last
        xf0, yf0, yawf0, vf0, rf0 = self.lastf
        n = max(int(round((t - t0) / 0.01)), 0)
        dtc = max(t - t0, 1e-3)
        n_gap = n
        if n > 50:
            # long gap (node / sim stall): servo and sideslip settled long ago, only the last 0.5 s matter (simulating
            # the whole gap cost ~0.6 ms per second of gap: a 10 min pause -> one 0.35 s call)
            t0 += 0.01 * (n - 50)
            n = 50
        # after a gap > 0.25 s the increments (filters, course angle) and the parallel models are meaningless: take
        # the measurement as it is and restart the stiffness models
        short = n_gap <= 25
        # 1) model prediction of speed and yaw rate (commands in flight) for the filters, plus the parallel models
        #    of the tyre-stiffness estimator (one single-track model per stiffness scale, driven by the same inputs)
        cv = self.c_vec
        cv[0] = self.c_tyre
        bb = self.cg_bb
        rr = self.cg_rr
        bb[0], rr[0] = self.beta_est, rf0
        bb[1:], rr[1:] = self.cg_b, self.cg_r
        if self.jit:
            delta, vm, dmain, amain, vms, vck, hvk = _obs_main_k(self._hist_arr(), float(self.t_lat), float(self.t_buf),
                                                                 float(t0), int(n), float(vf0), float(self.delta_est))
        else:
            delta = self.delta_est
            vm = vf0
            m_ = max(n, 1)
            dmain, amain, vms, vck = np.empty(m_), np.empty(m_), np.empty(m_), np.zeros(m_)
            hvk = np.zeros(m_, dtype=np.bool_)
            for k in range(n):
                tgt, vc = self._targets(t0 + 0.01 * k)
                delta = self._servo_step(delta, tgt)
                a = p_accel(vc, vm) if vc is not None else 0.0
                dmain[k], amain[k], vms[k] = delta, a, vm
                vck[k], hvk[k] = (vc, True) if vc is not None else (0.0, False)
                vm = max(0.0, vm + a * 0.01)
        if short and n > 0:
            H = self._H if self._H is not None else np.array(self.hist, dtype=np.float64).reshape(len(self.hist), 3)
            _mm_prop(bb, rr, self.cg_d, cv, self.l_grid_v, self.c_grid_v.size, H, float(self.t_lat), float(self.t_buf),
                     float(t0), int(n), float(vf0), float(v), dmain, amain, vms)
        filt = self.pose_filt and n > 0 and short
        # 2) speed and yaw rate
        kv = self._gain('v', ((v - v0) - (vm - vf0),), self.q_v) if filt else 1.0
        vf = vm + kv * (v - vm)
        rp = float(rr[0])
        kr = self._gain('r', ((r - r0) - (rp - rf0),), self.q_r) if filt else 1.0
        rf = rp + kr * (r - rp)
        # 3) heading: integrated yaw rate
        dyaw = 0.5 * (rf0 + rf) * dtc
        inn_y = _wrap(yaw - yawf0 - dyaw)
        yaw_keep = yaw                               # what the next call takes as the previous raw heading
        if filt:
            jump_y = abs(inn_y) > 0.5 * self.pose_gate
            outl_y = (not jump_y and self.pose_out > 0.0
                      and abs(inn_y) > self.pose_out * self._innov_sd('yaw', self.q_yaw) + self.yaw_out_floor)
            if outl_y and not self.out_yaw:
                # one sample far outside the measured noise (glitch): keep the prediction, keep it out of the noise
                # statistics, and take the next increment from the predicted heading
                kyw = 0.0
                self.nstat['yaw'][2] = None
                yaw_keep = float(_wrap(yawf0 + dyaw))
            elif jump_y or outl_y:
                kyw = 1.0                            # heading jump (localisation reset, or two outliers): take it
                self.nstat['yaw'][2] = None
            else:
                kyw = self._gain('yaw', (_wrap(yaw - yaw0) - dyaw,), self.q_yaw)
            self.out_yaw = outl_y and not self.out_yaw
        else:
            kyw = 1.0
        yawf = float(_wrap(yawf0 + dyaw + kyw * inn_y))
        # 4) sideslip (exact for the linear beta ODE over 10 ms, filtered speed / yaw rate) and the predicted course
        dyw = float(_wrap(yawf - yawf0))
        if self.jit:
            beta, has_mid, bm, px_, py_ = _obs_beta_k(int(n), float(vf0), float(vf), float(rf0), float(rf),
                                                      float(yawf0), dyw, float(xf0), float(yf0), float(self.beta_est),
                                                      dmain, vck, hvk, float(self.c_tyre))
            beta_mid = bm if has_mid else None
        else:
            beta = self.beta_est
            beta_mid = None
            px_, py_ = xf0, yf0
            for k in range(n):
                b_before = beta
                f = (k + 0.5) / max(n, 1)
                vk = vf0 + (vf - vf0) * f
                rk = rf0 + (rf - rf0) * f
                a = p_accel(vck[k], vk) if hvk[k] else 0.0
                yk = yawf0 + dyw * f
                if vk >= 0.5:
                    a11, a12, b1, _, _, _ = bicycle_s(vk, a, self.c_tyre)
                    e = math.exp(a11 * 0.01)
                    bss = -(a12 * rk + b1 * dmain[k]) / a11
                    bmid_ = bss + (beta - bss) * math.exp(a11 * 0.005)
                    beta = bss + (beta - bss) * e
                else:
                    bmid_ = beta = 0.0
                px_ += vk * math.cos(yk + bmid_) * 0.01
                py_ += vk * math.sin(yk + bmid_) * 0.01
                # sideslip at the middle of the interval (the chord's course): after n/2 steps
                h2 = 2 * (k + 1)
                if h2 == n:
                    beta_mid = beta
                elif h2 == n + 1:
                    beta_mid = 0.5 * (b_before + beta)
        # 5) position
        x_keep, y_keep = x, y                        # what the next call takes as the previous raw position
        p_out = False
        if filt:
            inn = math.hypot(x - px_, y - py_)
            jump = inn > self.pose_gate
            outl = (not jump and self.pose_out > 0.0
                    and inn > self.pose_out * self._innov_sd('pos', self.q_pos) + self.pose_out_floor)
            if outl and not self.out_pos:
                # a pose far outside the measured noise but below the jump gate (localisation glitch): it passed with
                # gain 1, made the noise estimate jump and the filter then stayed near the glitched pose for ~10 calls.
                # Keep the prediction for this one sample; a second one in a row is a real jump and is taken.
                kp = 0.0
                p_out = True
                self.nstat['pos'][2] = None
                x_keep, y_keep = px_, py_
            elif jump or outl:
                kp = 1.0                             # localisation jump: take it, do not smooth it in
                self.nstat['pos'][2] = None
            else:
                kp = self._gain('pos', ((x - x0) - (px_ - xf0), (y - y0) - (py_ - yf0)), self.q_pos)
            self.out_pos = outl and not self.out_pos
            xf, yf = px_ + kp * (x - px_), py_ + kp * (y - py_)
        else:
            xf, yf = x, y
        # course-angle correction from the (raw) pose increment, weighted by its noise (2 R_pos / dist^2): the chord
        # direction of a 2-20 cm increment is only as good as the poses (2 cm noise: +-0.3 rad)
        # (the chord is the course at the arc midpoint only while the yaw change stays well inside +-pi: short gaps)
        dist = math.hypot(x - x0, y - y0)
        if self.k_obs > 0 and dist > 0.02 and v > 1.0 and beta_mid is not None and short and not p_out:
            chi = math.atan2(y - y0, x - x0)
            ymid = yawf0 + 0.5 * _wrap(yawf - yawf0)
            bmeas = _wrap(chi - ymid)
            Rp = max(-self.nstat['pos'][1], 0.0) if self.nstat['pos'][3] >= self.noise_n0 else 0.0
            kb = self.k_obs * self.beta_sig ** 2 / (self.beta_sig ** 2 + 2.0 * Rp / (dist * dist))
            beta += kb * (bmeas - beta_mid)
        # the linear-tyre model means nothing beyond this; keeps one bad pose pair from feeding a +-pi sideslip
        # into the prediction and the MPC start state
        beta = min(max(beta, -0.5), 0.5)
        self.delta_est, self.beta_est = delta, beta
        # 6) tyre stiffness: output error of the parallel models against the measured yaw rate (exponential memory)
        if self.c_adapt and n > 0:
            if short and vf0 > self.c_vmin and vf > self.c_vmin and self.cg_on:
                self.cg_b[:], self.cg_r[:] = bb[1:], rr[1:]
                err = r - self.cg_r
                if self.fg is not None:
                    self.fg_e2 += err * err
                    self.fg_ne += 1
                lam = math.exp(-dtc / self.c_tau)
                self.cg_J = lam * self.cg_J + err * err
                self.c_tyre = self._c_estimate()
            else:
                self.cg_b[:], self.cg_r[:] = beta, rf
                self.cg_d[:] = delta
                self.cg_on = vf > self.c_vmin
        self.last = (t, x_keep, y_keep, yaw_keep, v, r)
        self.lastf = (xf, yf, yawf, vf, rf)
        return xf, yf, yawf, vf, rf

    def _innov_sd(self, key, qmin):
        """standard deviation of the filter innovation (measurement - prediction) of one channel: prior variance of the
        steady-state random-walk filter plus the measurement noise (see _gain)"""
        st = self.nstat[key]
        R = self._noise_R(key)
        q = max(st[0] - 2.0 * R, qmin)
        P = 0.5 * (q + math.sqrt(q * q + 4.0 * q * R))
        return math.sqrt(P + R)

    @staticmethod
    def _parab_min(g, J):
        """minimiser of J over the uniform grid g: parabola through the best grid point and its neighbours"""
        k = min(max(int(np.argmin(J)), 1), g.size - 2)
        j0, j1, j2 = J[k - 1], J[k], J[k + 1]
        den = j0 - 2.0 * j1 + j2
        x = g[k] + (0.5 * (g[1] - g[0]) * (j0 - j2) / den if den > 1e-12 else 0.0)
        return float(min(max(x, g[0]), g[-1]))

    def _c_estimate(self):
        """stiffness scale minimising the yaw-rate output error (+ weak priors at the nominal tyre and latency).  The
        command delay is estimated jointly (profile over the delay hypotheses, relative to the lat_est latency): an
        unknown latency otherwise biases the stiffness (20 ms extra latency looked like 15-40 % stiffer tyres, mostly
        through the load transfer of the late speed command)."""
        J = self.cg_J.reshape(self.l_grid_v.size, self.c_grid_v.size) + self.cg_prior
        self.lat_extra = self._parab_min(self.l_grid_v, J.min(axis=1))
        return self._parab_min(self.c_grid_v, J.min(axis=0))

    def _predict(self, t, x, y, yaw, v, r, t_pred):
        """nonlinear single-track prediction to t + t_pred with the commands in flight"""
        delta, beta = self.delta_est, self.beta_est
        if self.jit:
            return _predict_k(self._hist_arr(), float(self.t_lat), float(self.t_buf), float(t), float(x), float(y),
                              float(yaw), float(v), float(r),
                              float(t_pred), float(delta), float(beta), float(self.c_tyre))
        n = int(round(t_pred / 0.01))
        a = 0.0
        for k in range(n):
            tt = t + 0.01 * k
            tgt, vc = self._targets(tt)
            delta = self._servo_step(delta, tgt)
            a = p_accel(vc, v) if vc is not None else 0.0
            if v >= 0.5:
                a11, a12, b1, a21, a22, b2 = bicycle_s(v, a, self.c_tyre)
                # explicit Euler on the (beta, r) modes (about -210 1/s at 0.5 m/s, ~1/v): unstable below ~0.53 m/s and
                # sign-flipping up to ~1 m/s with 10 ms steps (the simulator integrates with RK4) -> sub-steps keep
                # |lambda h| <= 0.7; one step above ~1.5 m/s as before
                m = max(1, int(math.ceil(-0.01 * min(a11, a22) / 0.7)))
                h = 0.01 / m
                for _ in range(m):
                    bd = a11 * beta + a12 * r + b1 * delta
                    rd = a21 * beta + a22 * r + b2 * delta
                    x += v * math.cos(yaw + beta) * h
                    y += v * math.sin(yaw + beta) * h
                    yaw += r * h
                    beta += bd * h
                    r += rd * h
            else:
                x += v * math.cos(yaw) * 0.01
                y += v * math.sin(yaw) * 0.01
                r = v * math.tan(delta) / WB
                yaw += r * 0.01
                beta = 0.0
            v = max(0.0, v + a * 0.01)
        return x, y, yaw, v, beta, r, delta, a

    # ------------------------------------------------------------------ perception
    def _perceive(self, scan, lx, ly, yaw, t):
        cy, sy = math.cos(yaw), math.sin(yaw)
        if self.jit:
            self._perceive_jit(scan, lx, ly, yaw, t, cy, sy)
            return
        if t >= self.blind_until:
            ok = (scan > 0.05) & (scan < self.r_max)
            r = scan[ok]
            ux = cy * self.ca[ok] - sy * self.sa[ok]
            uy = sy * self.ca[ok] + cy * self.sa[ok]
            px = lx + r * ux
            py = ly + r * uy
            col = np.floor((px - self.ox) / self.res).astype(int)
            row = np.floor((py - self.oy) / self.res).astype(int)
            inb = (col >= 0) & (col < self.W) & (row >= 0) & (row < self.H)
            px, py, col, row = px[inb], py[inb], col[inb], row[inb]
            isobs = self.edt[row, col] > self.obs_edt
            if np.any(isobs):
                mc = np.floor((px[isobs] - self.ox) / self.mres).astype(int)
                mr = np.floor((py[isobs] - self.oy) / self.mres).astype(int)
                cells = np.unique(mr * self.MW + mc)
                self.mem[cells] = np.minimum(self.mem[cells] + 2, 12)
                self.active = np.union1d(self.active, cells)
        # free-space carving of remembered cells the beams now see through
        if self.carve and self.active.size:
            a = self.active
            dx = self.cell_x[a] - lx
            dy = self.cell_y[a] - ly
            rng = np.hypot(dx, dy)
            brg = _wrap(np.arctan2(dy, dx) - yaw)
            b = np.rint((brg - ANG_MIN) / ANG_INC).astype(int)
            chk = (b >= 2) & (b <= 1077) & (rng > 0.4) & (rng < 7.0)
            if np.any(chk):
                bb = b[chk]
                mn = np.minimum.reduce([scan[bb - 2], scan[bb - 1], scan[bb], scan[bb + 1], scan[bb + 2]])
                seen = mn > rng[chk] + 0.20
                if np.any(seen):
                    cs = a[chk][seen]
                    self.mem[cs] -= 1
                    dead = cs[self.mem[cs] <= 0]
                    if dead.size:
                        self.mem[dead] = 0
                        self.active = np.setdiff1d(self.active, dead, assume_unique=True)

    def _mem_clear(self):
        """forget all remembered obstacle cells (lap wrap, reset)"""
        self.active = np.zeros(0, dtype=np.int64)
        self.act_mem = np.zeros(0, dtype=np.int16)
        self.act_x, self.act_y, self.act_s, self.act_d = (np.zeros(0, dtype=np.float32) for _ in range(4))
        self.act_ok = np.zeros(0, dtype=np.bool_)
        self._view = None

    def _view_of(self, lx, ly, yaw):
        """range, bearing and beam index of every remembered cell from the LiDAR (float32 numpy math, as before)"""
        dx = self.act_x - lx
        dy = self.act_y - ly
        rng = np.hypot(dx, dy)
        brg = _wrap(np.arctan2(dy, dx) - yaw)
        return rng, brg, np.rint((brg - ANG_MIN) / ANG_INC).astype(np.int64)

    def _perceive_jit(self, scan, lx, ly, yaw, t, cy, sy):
        """_perceive on the compact remembered set: the per-cell map attributes (x, y, s, d, ok) and memory counts are
        kept in arrays aligned with the sorted cell ids, so the per-call work touches only those (no gathers into the
        map-sized grids, no set operations by sorting); the cells' view of this scan is kept for _boxes"""
        self._view = None
        if t >= self.blind_until:
            cells = _hits_k(scan, float(lx), float(ly), cy, sy, self.ca, self.sa, float(self.r_max), self.ox, self.oy,
                            self.res, self.W, self.H, self.edt, float(self.obs_edt), self.mres, self.MW)
            if cells.size:
                (self.active, self.act_mem, self.act_x, self.act_y, self.act_s, self.act_d,
                 self.act_ok) = _merge_k(self.active, self.act_mem, self.act_x, self.act_y, self.act_s, self.act_d,
                                         self.act_ok, cells, self.cell_x, self.cell_y, self.cell_s, self.cell_d,
                                         self.cell_ok)
        if self.active.size == 0:
            return
        view = self._view_of(lx, ly, yaw)
        if self.carve:
            keep = _carve_k(self.act_mem, view[0], view[2], scan)
            if not keep.all():
                for nm in ('active', 'act_mem', 'act_x', 'act_y', 'act_s', 'act_d', 'act_ok'):
                    setattr(self, nm, getattr(self, nm)[keep])
                view = tuple(v[keep] for v in view)
        self._view = view

    def _boxes(self, s_car, scan, lx, ly, yaw):
        """cluster remembered cells near the planning window into Frenet boxes [s_lo, s_hi, d_lo, d_hi].
        The hidden back is added (hidden_depth); a lateral end that the current scan shows bounded by a closer
        return (occluded) is extended to the largest possible footprint width (w_max)."""
        a = self.active
        if a.size == 0:
            return []
        if self.jit and self.occl_fix and not self.box_mono:
            return self._boxes_jit(s_car, scan, lx, ly, yaw)
        a = a[self.cell_ok[a]]
        if a.size > 60:
            # one representative per 2x2 block keeps the extent (+- one fine cell) and quarters the work
            blk = (a // self.MW // 2) * self.W + (a % self.MW) // 2
            _, first = np.unique(blk, return_index=True)
            keep = np.zeros(a.size, dtype=bool)
            keep[first] = True
            # keep all cells on the outline of each block set: simply also keep extreme cells per block
            a = a[keep] if a.size > 400 else a
        rel = np.mod(self.cell_s[a] - s_car + self.L / 2, self.L) - self.L / 2
        win = (rel > -1.5) & (rel < self.n * self.ds + 1.0)
        a, rel = a[win], rel[win]
        if a.size == 0:
            return []
        xs, ys = self.cell_x[a], self.cell_y[a]
        dd = self.cell_d[a]
        if a.size == 1:
            labels = np.zeros(1, dtype=int)
            nc = 1
        else:
            roots = _cluster_k(xs.astype(np.float64), ys.astype(np.float64), float(self.link))
            _, labels = np.unique(roots, return_inverse=True)
            nc = int(labels.max()) + 1
        # current view of every cell
        dxl, dyl = xs - lx, ys - ly
        rng = np.hypot(dxl, dyl)
        brg = _wrap(np.arctan2(dyl, dxl) - yaw)
        bi = np.rint((brg - ANG_MIN) / ANG_INC).astype(int)
        inv = (bi >= 0) & (bi <= 1079)
        bic = np.minimum(np.maximum(bi, 0), 1079)
        seen = inv & (np.abs(scan[bic] - rng) < 0.08)
        out = []
        half = self.mres / 2
        for c in range(nc):
            m = labels == c
            ss = rel[m] + s_car
            dl = dd[m]
            s_lo, s_hi = float(ss.min()) - half, float(ss.max()) + half
            d_lo, d_hi = float(dl.min()) - half, float(dl.max()) + half
            s_hi = max(s_hi, s_lo + self.hidden_depth)
            if self.occl_fix:
                d_lo, d_hi = self._extent_fix(m, seen, brg, bi, rng, dd, scan, s_lo, s_hi, d_lo, d_hi)
            elif d_hi - d_lo < self.min_width:
                dc = 0.5 * (d_lo + d_hi)
                d_lo = min(d_lo, dc - self.min_width / 2)
                d_hi = max(d_hi, dc + self.min_width / 2)
            if self.occl_ext and not self.occl_fix and d_hi - d_lo < self.w_max:
                # per-obstacle latch: an edge seen once with free space beyond it is real (never extended again)
                smr, dcr = 0.5 * (s_lo + s_hi), 0.5 * (d_lo + d_hi)
                ent = self._conf_entry(smr, dcr, 2)
                ms = m & seen
                if np.any(ms) and abs(float(np.mean(brg[ms]))) < 1.1:
                    idx = np.flatnonzero(ms)
                    i_lo = idx[np.argmin(bi[idx])]
                    i_hi = idx[np.argmax(bi[idx])]
                    ext = []
                    for ie, step in ((i_lo, -1), (i_hi, 1)):
                        b0 = int(bi[ie])
                        nb = [b0 + step * k for k in (1, 2, 3) if 0 <= b0 + step * k <= 1079]
                        mnb = float(np.fmin.reduce(scan[nb])) if len(nb) == 3 else 0.0
                        if math.isnan(mnb):
                            continue          # only invalid beams beyond the edge: no information
                        occl = (len(nb) < 3) or (mnb < rng[ie] - 0.05)
                        ext.append((occl, float(dd[ie])))
                    for occl, dend in ext:
                        low_side = dend <= dcr
                        k_ = 2 if low_side else 3
                        if not occl:
                            ent[k_] = True
                            continue
                        if ent[k_]:
                            continue
                        if low_side:
                            d_lo = min(d_lo, d_hi - self.w_max)
                        else:
                            d_hi = max(d_hi, d_lo + self.w_max)
            out.append([s_lo - self.obs_pad_s, s_hi + self.obs_pad_s, d_lo - self.obs_pad_d, d_hi + self.obs_pad_d])
        # static obstacles never shrink: keep the union with the extent seen before in this lap
        if self.box_mono:
            for b in out:
                smid, dc = 0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3])
                hit = None
                for e in self.box_ext:
                    if abs(e[0] - smid) < 0.4 and abs(e[1] - dc) < 0.35:
                        hit = e
                        break
                if hit is None:
                    self.box_ext.append([smid, dc, b[0], b[1], b[2], b[3]])
                else:
                    b[0] = min(b[0], hit[2])
                    b[1] = max(b[1], hit[3])
                    b[2] = min(b[2], hit[4])
                    b[3] = max(b[3], hit[5])
                    hit[2:] = b[:]
        out.sort(key=lambda b: b[0])
        return out

    def _boxes_jit(self, s_car, scan, lx, ly, yaw):
        """_boxes (occl_fix) on the compact remembered set: selection, clustering and per-cluster statistics in
        _boxes_k with the cells' view of this scan from _perceive, then the latch logic of _extent_fix per cluster"""
        view = self._view if self._view is not None else self._view_of(lx, ly, yaw)
        rmin, rmax, dmin, dmax, nseen, bmean, occl = _boxes_k(
            self.active, self.act_ok, self.act_s, self.act_x, self.act_y, self.act_d, view[0], view[1], view[2], scan,
            self.MW, self.W, float(s_car), float(self.L), float(self.n * self.ds + 1.0), float(self.link))
        out = []
        half = self.mres / 2
        s32 = np.float32(s_car)           # ss = rel + s_car is float32 (numpy casting); min/max commute with it
        for c in range(rmin.shape[0]):
            s_lo, s_hi = float(rmin[c] + s32) - half, float(rmax[c] + s32) + half
            d_lo, d_hi = float(dmin[c]) - half, float(dmax[c]) + half
            s_hi = max(s_hi, s_lo + self.hidden_depth)
            d_lo, d_hi = self._extent_latch(s_lo, s_hi, d_lo, d_hi, bool(nseen[c] > 0 and abs(float(bmean[c])) < 1.1),
                                            occl[c])
            out.append([s_lo - self.obs_pad_s, s_hi + self.obs_pad_s, d_lo - self.obs_pad_d, d_hi + self.obs_pad_d])
        out.sort(key=lambda b: b[0])
        return out

    def _extent_fix(self, m, seen, brg, bi, rng, dd, scan, s_lo, s_hi, o_lo, o_hi):
        """lateral extent of one cluster from its observed cells [o_lo, o_hi]: an edge with free space beyond it
        (beams past the edge return farther) is real and latched; an edge with a closer return beyond it is hidden
        behind something and is extended to the largest footprint (w_max) until it is seen free.  Edges are told
        apart by beam order (lower beam index = right = lower d while the car looks along the track)."""
        ms = m & seen
        use = bool(np.any(ms) and abs(float(np.mean(brg[ms]))) < 1.1)
        occl = [False, False]
        if use:
            idx = np.flatnonzero(ms)
            for side, (ie, step) in enumerate(((idx[np.argmin(bi[idx])], -1), (idx[np.argmax(bi[idx])], 1))):
                b0 = int(bi[ie])
                nb = [b0 + step * k for k in (1, 2, 3) if 0 <= b0 + step * k <= 1079]
                # NaN beams (invalid returns) carry no information: np.min gave NaN, the comparison False and the edge
                # was latched 'seen free' for the rest of the lap (hidden side never extended again)
                mnb = float(np.fmin.reduce(scan[nb])) if len(nb) == 3 else 0.0
                if math.isnan(mnb):
                    occl[side] = 2            # only invalid beams beyond the edge: no information
                    continue
                occl[side] = (len(nb) < 3) or (mnb < rng[ie] - 0.05)
        return self._extent_latch(s_lo, s_hi, o_lo, o_hi, use, occl)

    def _conf_entry(self, smr, dcr, nflag):
        """the first box_conf entry (insertion order) within 0.4 m in s and 0.35 m in d of (smr, dcr), else a new one;
        entries are bucketed by s, so the lookup does not grow with the number of clusters seen in the lap"""
        kb = math.floor(smr / 0.4)
        best = None
        for k in range(kb - 2, kb + 3):
            for i in self.conf_idx.get(k, ()):
                e = self.box_conf[i]
                if abs(e[0] - smr) < 0.4 and abs(e[1] - dcr) < 0.35:
                    if best is None or i < best:
                        best = i
                    break
        if best is not None:
            return self.box_conf[best]
        ent = [smr, dcr] + [False] * nflag
        self.conf_idx.setdefault(kb, []).append(len(self.box_conf))
        self.box_conf.append(ent)
        return ent

    def _extent_latch(self, s_lo, s_hi, o_lo, o_hi, use, occl):
        """the latch of _extent_fix from its scan evidence (also precomputed by _clusters_k): use = some cells seen and
        the cluster ahead (mean bearing < 1.1 rad), occl = (low-index edge hidden, high-index edge hidden)"""
        ent = self._conf_entry(0.5 * (s_lo + s_hi), 0.5 * (o_lo + o_hi), 4)
        if use and not self._no_latch:
            for kc, oc in ((2, occl[0]), (3, occl[1])):
                if oc == 2:
                    continue                  # no information about this edge (only NaN beams beyond it)
                if not oc:
                    ent[kc] = True
                    ent[kc + 2] = False
                elif not ent[kc]:
                    ent[kc + 2] = True
        d_lo, d_hi = o_lo, o_hi
        if ent[5] and not ent[3]:
            d_hi = max(d_hi, o_lo + self.w_max)
        if ent[4] and not ent[2]:
            d_lo = min(d_lo, o_hi - self.w_max)
        if d_hi - d_lo < self.min_width:
            if ent[2] and not ent[3]:
                d_hi = d_lo + self.min_width
            elif ent[3] and not ent[2]:
                d_lo = d_hi - self.min_width
            else:
                dc = 0.5 * (d_lo + d_hi)
                d_lo, d_hi = dc - self.min_width / 2, dc + self.min_width / 2
        return d_lo, d_hi

    @staticmethod
    def _first_match(lst, cache, smid, dc, ts, td):
        """first entry (list order) of lst within ts in s and td in d of (smid, dc); the list is bucketed by s (0.4 m)
        once per list object, so a lookup does not scan all entries (noisy scans give dozens of boxes)"""
        if cache[0] is not lst:
            idx = {}
            for i, e in enumerate(lst):
                idx.setdefault(math.floor(e[0] / 0.4), []).append(i)
            cache[0], cache[1], cache[2] = lst, idx, {}
        memo = cache[2]               # the same boxes are looked up several times per call (repeated DP passes)
        hit = memo.get((smid, dc), memo)
        if hit is not memo:
            return hit
        idx = cache[1]
        kb = math.floor(smid / 0.4)
        best = None
        for k in range(kb - 2, kb + 3):
            for i in idx.get(k, ()):
                e = lst[i]
                if abs(e[0] - smid) < ts and abs(e[1] - dc) < td:
                    if best is None or i < best:
                        best = i
                    break
        res = None if best is None else lst[best]
        memo[(smid, dc)] = res
        return res

    def _prev_side(self, smid, dc):
        if self.jit:
            e = self._first_match(self.prev_sides, self._ps_cache, smid, dc, 0.35, 0.3)
            return None if e is None else e[2]
        for e in self.prev_sides:
            if abs(e[0] - smid) < 0.35 and abs(e[1] - dc) < 0.3:
                return e[2]
        return None

    def _first_seen(self, smid, dc, t):
        if self.jit:
            e = self._first_match(self.seen_boxes, self._sb_cache, smid, dc, 0.4, 0.35)
            return t if e is None else e[2]
        for e in self.seen_boxes:
            if abs(e[0] - smid) < 0.4 and abs(e[1] - dc) < 0.35:
                return e[2]
        return t

    # ------------------------------------------------------------------ DP side selection
    def _dp(self, s0, d0, chi0, boxes, v, w_h, relax=(), hard=True, gap=True):
        """2nd-order DP over (station, lateral offset, slope): picks the side of every obstacle with a
        curvature-limited path; returns MPC stations, DP path on them, sides, first blocked s"""
        N, ds = self.n, self.ds
        if self.jit:
            return self._dp_jit(s0, d0, chi0, boxes, v, w_h, relax, hard, gap)
        st = s0 + ds * np.arange(N + 1)
        Ds, dd = self.dp_ds, self.dd
        M = int(round(N * ds / Ds))
        sd = s0 + Ds * np.arange(M + 1)
        sq = dd / Ds                                   # slope quantum
        Kmax = self.dp_kmax
        kv = np.arange(-Kmax, Kmax + 1)
        K = kv.size
        wlo, whi = self._wall_bounds(sd, self.wc_eff)
        jlo = int(math.floor((min(max(wlo.min(), -self.dp_dmax), d0) + 3.5) / dd)) - 1
        jhi = int(math.ceil((max(min(whi.max(), self.dp_dmax), d0) + 3.5) / dd)) + 1
        dg = -3.5 + dd * np.arange(max(jlo, 0), min(jhi, int(7.0 / dd)) + 1)
        J = dg.size
        blocked = (dg[None, :] < wlo[:, None]) | (dg[None, :] > whi[:, None])
        cost_node = np.repeat((self.w_dev * dg ** 2 * Ds)[None, :], M + 1, axis=0)
        wlo2, whi2 = self._wall_bounds(sd, self.wall_pref)
        pen = np.maximum(0.0, wlo2[:, None] - dg[None, :]) + np.maximum(0.0, dg[None, :] - whi2[:, None])
        cost_node += self.w_wall * pen ** 2 * Ds
        half = 0.5 * Ds
        for bi_, b in enumerate(boxes):
            ks = np.flatnonzero((sd >= b[0] - half) & (sd <= b[1] + half))
            if ks.size == 0:
                continue
            lo, hi = b[2] - w_h, b[3] + w_h
            near = ks[(sd[ks] - s0) < self.dp_near]
            far = ks[(sd[ks] - s0) >= self.dp_near]
            inside = (dg > lo) & (dg < hi)
            if far.size:
                blocked[np.ix_(far, np.flatnonzero(inside))] = True
                if gap and self.min_gap_w > 0.0:
                    # a side whose corridor (obstacle .. wall) is narrower than min_gap_w is not offered: such a gap
                    # appears and disappears with every small change of the estimate (late side flips).  A side
                    # already chosen keeps being offered down to min_gap_keep (hysteresis: a narrow gap that shrinks
                    # by 3 cm while the car approaches at speed must not force a flip it cannot drive)
                    pv_ = self._prev_side(0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]))
                    g_hi = self.min_gap_keep if pv_ == 1 else self.min_gap_w
                    g_lo = self.min_gap_keep if pv_ == -1 else self.min_gap_w
                    if float(np.min(whi[far])) - hi < g_hi:
                        blocked[np.ix_(far, np.flatnonzero(dg >= hi))] = True
                    if lo - float(np.max(wlo[far])) < g_lo:
                        blocked[np.ix_(far, np.flatnonzero(dg <= lo))] = True
            if near.size:
                cost_node[np.ix_(near, np.flatnonzero(inside))] += self.p_near
            clr = np.maximum(lo - dg, dg - hi)
            soft = np.maximum(0.0, self.c_pref - clr)
            cost_node[ks] += self.w_soft * soft[None, :] ** 2 * Ds
            smid, dc = 0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3])
            prev = self._prev_side(smid, dc)
            if prev is not None:
                dist = b[0] - (s0 + BOX_FRONT)
                dnear = max(self.commit_d, self.commit_t * v)
                f = min(1.0, max(0.0, (dnear + self.commit_ramp - dist) / self.commit_ramp))
                # releasing a commitment (forced flip in the cluster, or a neighbour that just appeared) is only
                # allowed outside the hard-commit distance: near obstacles keep their side, the car brakes instead
                if dist > dnear + self.release_margin:
                    if bi_ in relax:
                        f = 0.0
                    for c2, age in zip(boxes, self.box_age):
                        if c2 is not b and age < self.young_t and abs(0.5 * (c2[0] + c2[1]) - smid) < self.cluster_ds:
                            f = 0.0
                            break
                pf = self.p_flip + (self.p_flip_hard - self.p_flip) * f
                kmid = ks[len(ks) // 2]
                wrong = (dg < dc) if prev > 0 else (dg > dc)
                if hard and f >= 1.0 and dist < dnear and v > self.hard_block_v:
                    # committed and close: never flip here (brake instead if the side looks closed)
                    blocked[np.ix_(ks, np.flatnonzero(wrong))] = True
                else:
                    cost_node[kmid, wrong] += pf
        # curvature limit per DP station: the slowest speed the car can have there (planned braking)
        dist = sd - s0
        vr = np.interp(np.maximum(v * v - 2.0 * self.dp_adec * dist, 0.0) ** 0.5, [0.0, 100.0], [0.0, 100.0])
        vdp = np.maximum(np.minimum(vr, self.dp_vmax), self.dp_vfloor)
        kap = np.minimum(np.minimum(STEER_MAX / (WB + K_US0 / self.c_tyre * vdp * vdp), self.a_lat_hard / (vdp * vdp)),
                         self.dp_kcap)
        cst = np.minimum(np.maximum(np.floor(kap * Ds / sq + 1e-9).astype(np.int64), 1), Kmax)
        c = int(cst.max())
        cv_cost = self.w_cv * (np.arange(-c, c + 1) * sq / Ds) ** 2 * Ds
        sl_cost = self.w_sl * (kv * sq) ** 2 * Ds
        if self.dp_kgeo > 0.0:
            # geometric curvature of the path kr / D + d'' / D^2 (D = 1 - kr d) within the steering limit: on the
            # inside of a tight turn the same d'' is a much tighter radius (hairpins of narrow tracks)
            kr_d = self._finterp(self.fk, sd)[:, None]
            kg = self.dp_kgeo * STEER_MAX / (WB + K_US0 / self.c_tyre * self.dp_vfloor ** 2)
            Dm = np.maximum(1.0 - kr_d * dg[None, :], 0.05)
            dlo = (-kg * Dm - kr_d) * Dm * (Ds / sq)
            dhi = (kg * Dm - kr_d) * Dm * (Ds / sq)
        else:
            dlo = np.full((M + 1, J), -100.0)
            dhi = np.full((M + 1, J), 100.0)
        cost0 = np.full((J, K), np.inf)
        j0 = min(max(int(round((d0 - dg[0]) / dd)), 0), J - 1)
        k0 = min(max(int(round(math.tan(max(-1.2, min(1.2, chi0))) / sq)), -Kmax), Kmax) + Kmax
        cost0[j0, max(k0 - 1, 0):min(k0 + 2, K)] = 0.0
        back, cost, reach = _dp_core(cost0, np.ascontiguousarray(cost_node), np.ascontiguousarray(blocked),
                                     kv.astype(np.int64), cst, cv_cost, sl_cost, dlo, dhi, float(self.w_geo))
        if reach < M:
            pass
        flat = int(np.argmin(cost)) if reach == M else None
        if flat is None:
            # recompute the cost at the last reached station: rerun to that station
            back, cost, reach2 = _dp_core(cost0, np.ascontiguousarray(cost_node[:reach + 1]),
                                          np.ascontiguousarray(blocked[:reach + 1]), kv.astype(np.int64),
                                          cst[:reach + 1], cv_cost, sl_cost, dlo, dhi, float(self.w_geo))
            flat = int(np.argmin(cost))
        jb, kb = divmod(flat, K)
        path_j = np.zeros(M + 1, dtype=int)
        path_j[reach] = jb
        for i in range(reach, 0, -1):
            dl = int(back[i - 1, jb, kb]) - c
            jp = jb - kv[kb]
            kb = kb - dl
            jb = jp
            path_j[i - 1] = jb
        path_j[reach + 1:] = path_j[reach]
        dpath_dp = dg[path_j]
        dpath = np.interp(st, sd, dpath_dp)
        sides = []
        for b in boxes:
            ks = np.flatnonzero((sd >= b[0] - half) & (sd <= b[1] + half) & (np.arange(M + 1) <= reach))
            if ks.size == 0:
                sides.append(0)
                continue
            sides.append(1 if np.mean(dpath_dp[ks]) > 0.5 * (b[2] + b[3]) else -1)
        s_block = sd[reach + 1] if reach < M else None
        return st, dpath, sides, s_block

    def _dp_jit(self, s0, d0, chi0, boxes, v, w_h, relax, hard, gap):
        """_dp with the numerics in _dp_k; the side commitments (previous sides, young neighbours) stay here"""
        nbx = len(boxes)
        bx = np.array(boxes, dtype=np.float64).reshape(-1, 4)
        bprev = np.zeros(nbx, dtype=np.int64)
        bpf = np.zeros(nbx)
        bhard = np.zeros(nbx, dtype=np.bool_)
        mids = [0.5 * (c2[0] + c2[1]) for c2 in boxes]
        young = [i for i, age in enumerate(self.box_age[:nbx]) if age < self.young_t]
        dnear = max(self.commit_d, self.commit_t * v)
        for bi_, b in enumerate(boxes):
            smid, dc = mids[bi_], 0.5 * (b[2] + b[3])
            prev = self._prev_side(smid, dc)
            if prev is None:
                continue
            dist = b[0] - (s0 + BOX_FRONT)
            f = min(1.0, max(0.0, (dnear + self.commit_ramp - dist) / self.commit_ramp))
            # releasing a commitment (forced flip in the cluster, or a neighbour that just appeared) is only
            # allowed outside the hard-commit distance: near obstacles keep their side, the car brakes instead
            if dist > dnear + self.release_margin:
                if bi_ in relax:
                    f = 0.0
                for i2 in young:
                    if i2 != bi_ and abs(mids[i2] - smid) < self.cluster_ds:
                        f = 0.0
                        break
            bprev[bi_] = prev
            bpf[bi_] = self.p_flip + (self.p_flip_hard - self.p_flip) * f
            # committed and close: never flip here (brake instead if the side looks closed)
            bhard[bi_] = hard and f >= 1.0 and dist < dnear and v > self.hard_block_v
        st, dpath, sides, s_block = _dp_k(
            float(s0), float(d0), float(chi0), float(v), int(self.n), float(self.ds), float(self.dp_ds), float(self.dd),
            int(self.dp_kmax), self.wlo_c, self.whi_c, self.wlo_p, self.whi_p, float(self.fds), float(self.dp_dmax),
            float(self.w_dev), float(self.w_wall), bx, bprev, bpf, bhard, float(w_h), float(self.dp_near),
            float(self.p_near), float(self.c_pref), float(self.w_soft), bool(gap and self.min_gap_w > 0.0),
            float(self.min_gap_w), float(self.dp_adec), float(self.dp_vmax), float(self.dp_vfloor),
            float(self.a_lat_hard), float(self.dp_kcap), float(self.w_cv), float(self.w_sl), float(self.c_tyre),
            float(self.min_gap_keep), self.fk, float(self.L), float(self.dp_kgeo), float(self.w_geo))
        return st, dpath, sides.tolist(), (None if math.isnan(s_block) else float(s_block))

    def _corridor(self, s, boxes, sides, w_h, s_block):
        lo, hi = self._wall_bounds(s, self.wc_eff)
        lo = lo.copy()
        hi = hi.copy()
        for b, sd in zip(boxes, sides):
            if sd == 0:
                continue
            m = (s >= b[0]) & (s <= b[1])
            if sd > 0:
                lo[m] = np.maximum(lo[m], b[3] + w_h)
            else:
                hi[m] = np.minimum(hi[m], b[2] - w_h)
        return lo, hi

    # ------------------------------------------------------------------ speed profile
    def _abrake(self, tab, v):
        return tab[min(int(max(v, 0.0) * 20.0), tab.size - 1)]

    def _speed_profile(self, st, d, mu, beta, kr, v0, scan, lx, ly, yaw, s_block, kyaw=None):
        """returns (vprof target speeds, vpred predicted speeds, accel per station, s_vis)"""
        N = self.n
        if self.jit and not self.vis_obs_stop:
            # compiled path (same computation as below)
            f64 = lambda a: np.ascontiguousarray(a, dtype=np.float64)
            vprof, vpred, acc, dpl, kv, v0_lim, self._sp_cum = _speed_k(
                f64(st), f64(d), f64(mu), f64(beta), f64(kr), float(v0), f64(scan), float(lx), float(ly), float(yaw),
                math.nan if s_block is None else float(s_block), self._no_kyaw if kyaw is None else f64(kyaw),
                self.fx, self.fy, self.fnx, self.fny, self.fpsi, self.fvx, float(self.L), float(self.fds),
                float(self.ds), float(self.v_max), float(self.a_lat), float(self.v_end_scale), float(self.swerve_w),
                float(self.a_acc), float(self.gg), float(self.gg_min), float(self.a_acc_eff), float(self.vis_tol),
                float(self.vis_start), float(self.m_vis), float(self.m_blk), float(self.t_react), float(self.v_vis_safe),
                self.tab_plan, self.tab_emerg, self.tab_dstop, self.tab_v)
            self.vlim0 = float(v0_lim)
            return vprof, vpred, acc, dpl, (float(st[kv]) if kv >= 0 else None)
        chi = mu + beta
        scale = (1.0 - kr * d) / np.maximum(np.cos(chi), 0.3)
        dpl = self.ds * np.maximum(scale[:-1], 0.2)
        cum = np.concatenate([[0.0], np.cumsum(dpl)])
        # geometric curvature of the path from the course angle
        heading = self._heading(st) + chi
        kg = np.zeros(N + 1)
        kg[:-1] = _wrap(np.diff(heading)) / dpl
        kg[-1] = kg[-2]
        ka = np.abs(kg)
        if self.swerve_w > 0.0:
            # curvature that is not the raceline's (swerves) excites larger yaw transients than steady cornering
            ka = ka + self.swerve_w * np.abs(kg - kr)
        if kyaw is not None:
            # the referee measures v * r: in trail braking / swerves the body yaws faster than the path turns
            ka = np.maximum(ka, kyaw)
        ka = np.maximum(ka, np.maximum(np.concatenate([ka[1:], ka[-1:]]), np.concatenate([ka[:1], ka[:-1]])))
        vlim = np.minimum(self.v_max, np.sqrt(self.a_lat / np.maximum(ka, 1e-3)))
        v_end = min(self.v_max, float(self._finterp(self.fvx, st[-1])) * math.sqrt(self.a_lat / 12.0) * self.v_end_scale)
        vlim[-1] = min(vlim[-1], v_end)
        # planned profile: backward pass with the steerable deceleration
        v = _passes(vlim, dpl, self.tab_plan, v0, self.a_acc, ka, float(self.a_lat), float(self.gg), float(self.gg_min))
        # emergency envelope: stop before unseen path / closed corridor with the larger stable deceleration
        stop_at = None
        s_vis = None
        xs, ys = self._xy(st, d)
        dx, dy = xs - lx, ys - ly
        rng = np.hypot(dx, dy)
        brg = _wrap(np.arctan2(dy, dx) - yaw)
        b = np.rint((brg - ANG_MIN) / ANG_INC).astype(int)
        inview = (b >= 1) & (b <= 1078)
        bb = np.minimum(np.maximum(b, 1), 1078)
        meas = np.fmin(np.fmin(scan[bb - 1], scan[bb]), scan[bb + 1])      # NaN beams ignored (all three NaN: unseen)
        vis = (inview & (meas >= rng - self.vis_tol)) | (cum < self.vis_start)
        bad = np.flatnonzero(~vis)
        if bad.size:
            kv = int(bad[0])
            s_vis = float(st[kv])
            stop_at = cum[kv] - BOX_FRONT - self.m_vis
        if s_block is not None:
            kb = min(int(np.searchsorted(st, s_block)), N)
            sb = cum[kb] - BOX_FRONT - self.m_blk
            stop_at = sb if stop_at is None else min(stop_at, sb)
        if stop_at is not None:
            env = np.full(N + 1, np.inf)
            reach = stop_at - cum - v0 * self.t_react
            # v such that stopping distance(v) <= reach; tabulated inverse of the stopping distance
            env = np.interp(np.maximum(reach, 0.0), self.tab_dstop, self.tab_v)
            env[reach <= 0] = 0.0
            obs_occl = False
            if self.vis_obs_stop and bad.size:
                # what hides the first unseen path point?  a wall (blind corner) or an obstacle (a chicane partner may be
                # right behind it): look at the occluding return of that beam in the wall distance transform
                bk = int(bb[kv])
                rk = float(meas[kv])
                ang = yaw + ANG_MIN + ANG_INC * bk
                hx, hy = lx + rk * math.cos(ang), ly + rk * math.sin(ang)
                col = int((hx - self.ox) / self.res)
                row = int((hy - self.oy) / self.res)
                if 0 <= col < self.W and 0 <= row < self.H:
                    obs_occl = bool(self.edt[row, col] > self.obs_edt)
            if self.v_vis_safe > 0.0 and bad.size and not obs_occl and (s_block is None or s_vis < s_block):
                # unseen path (not a closed corridor): arrive there no faster than v_vis_safe, from where the car can
                # still stop within ~d_stop(v_vis_safe) or swerve sharply around whatever appears
                vs = self.v_vis_safe
                ds_ = float(np.interp(vs, self.tab_v, self.tab_dstop))
                reach_v = cum[kv] - BOX_FRONT - self.m_vis - cum - v0 * self.t_react
                env_v = np.interp(np.maximum(reach_v, 0.0) + ds_, self.tab_dstop, self.tab_v)
                env_v = np.where(reach_v <= 0.0, np.minimum(env_v, vs), env_v)
                if s_block is not None:
                    reach_b = sb - cum - v0 * self.t_react
                    env_b = np.interp(np.maximum(reach_b, 0.0), self.tab_dstop, self.tab_v)
                    env_b[reach_b <= 0] = 0.0
                    env_v = np.minimum(env_v, env_b)
                env = env_v
            v = np.minimum(v, env)
        # forward pass from the current speed; predicted speed (braking limited by the emergency limit)
        vprof, vpred = _fwd(v, dpl, float(v0), float(self.a_acc_eff), self.tab_emerg)
        self.vlim0 = float(v[0])
        acc = np.clip((vpred[1:] ** 2 - vpred[:-1] ** 2) / (2 * dpl), -A_MAX, A_MAX)
        return vprof, vpred, acc, dpl, s_vis

    def _heading(self, s):
        """raceline heading at s (continuous interpolation)"""
        if self.jit and isinstance(s, float):
            u = (s % self.L) / self.fds
            fu = math.floor(u)
            i0 = int(fu) % self.nf
            i1 = (i0 + 1) % self.nf
            f = u - fu
            return float(self.fpsi[i0] + _wrap(float(self.fpsi[i1] - self.fpsi[i0])) * f)
        u = np.mod(np.asarray(s, dtype=float), self.L) / self.fds
        i0 = np.floor(u).astype(int) % self.nf
        i1 = (i0 + 1) % self.nf
        f = u - np.floor(u)
        return self.fpsi[i0] + _wrap(self.fpsi[i1] - self.fpsi[i0]) * f

    # ------------------------------------------------------------------ MPC
    def _mpc(self, st, X0, Xbar, kr, vpred, acc, dpl, boxes, sides, w_h, s_block, dpath=None):
        N, nl = self.n, len(self.lpts)
        if self.jit and self.hp is not None and self.hp_all:
            # compiled path (same QP): ~10x less time than the numpy construction below, which stays for OSQP
            hp = self.hp
            bx = np.array(boxes, dtype=np.float64).reshape(-1, 4)
            bsd = np.array(sides, dtype=np.int64).reshape(-1)
            base = dpath if (dpath is not None and self.ref_dp) else np.zeros(N + 1)
            cmu, lgc, ugc, dt = np.empty((N, nl)), np.empty((N, nl)), np.empty((N, nl)), np.empty(N)
            f64 = lambda a: np.ascontiguousarray(a, dtype=np.float64)
            _mpc_hp_k(f64(st), f64(X0), f64(Xbar), f64(kr), f64(vpred), f64(acc), f64(dpl), f64(base), self.lpts_a,
                      self.wlo_c, self.whi_c, self.nf, self.fds, bx, bsd, float(w_h), float(self.v_min_plan),
                      float(self.c_pref), float(self.q_d), float(self.q_mu), float(self.q_a), float(self.a_lat_hard),
                      self.tab_plan, hp.bA, hp.bB, hp.bb, hp.bQ, hp.bq, hp.bC, hp.blg, hp.bug, hp.blbx, hp.bubx,
                      cmu, lgc, ugc, dt, float(self.c_tyre))
            status, iters, Xs = hp.solve_all()
            self.qp_info = (iters, status)
            if status not in (0, 1) or not np.all(np.isfinite(Xs)):
                return None
            return Xs, np.diff(Xs[:, 4]) / dt, _hp_slack_k(Xs, cmu, lgc, ugc), dt
        vp = np.maximum(vpred, self.v_min_plan)
        vi = vp[:N]
        ai = acc
        dbar, mbar, bbar = Xbar[:, 0], Xbar[:, 1], Xbar[:, 2]
        chib = mbar + bbar
        cb, sb = np.cos(chib), np.sin(chib)
        D = np.maximum(1.0 - kr * dbar, 0.2)
        dt = dpl / vi
        a11, a12, b1, a21, a22, b2 = bicycle(vi, ai, self.c_tyre)
        krr = kr[:N]
        Ac = np.zeros((N, 7, 7))
        Ac[:, 0, 1] = vi * cb[:N]
        Ac[:, 0, 2] = vi * cb[:N]
        Ac[:, 0, 6] = vi * (sb[:N] - cb[:N] * chib[:N])
        cd = -krr * krr * vi * cb[:N] / D[:N] ** 2
        cm = krr * vi * sb[:N] / D[:N]
        Ac[:, 1, 0] = cd
        Ac[:, 1, 1] = cm
        Ac[:, 1, 2] = cm
        Ac[:, 1, 3] = 1.0
        Ac[:, 1, 6] = -krr * vi * cb[:N] / D[:N] - cd * dbar[:N] - cm * chib[:N]
        Ac[:, 2, 2] = a11
        Ac[:, 2, 3] = a12
        Ac[:, 2, 4] = b1
        Ac[:, 3, 2] = a21
        Ac[:, 3, 3] = a22
        Ac[:, 3, 4] = b2
        Ac[:, 4, 5] = 1.0
        E = _expm_k(np.ascontiguousarray(Ac * dt[:, None, None]))
        Ad = E[:, :5, :5]
        Bd = E[:, :5, 5]
        ed = E[:, :5, 6]
        Av = self.A_vals.copy()
        l = np.zeros(self.nrow)
        u = np.zeros(self.nrow)
        l[0:5] = X0
        u[0:5] = X0
        blk = np.zeros((N, 5, 7))
        blk[:, :, 0] = 1.0
        blk[:, :, 1:6] = -Ad
        blk[:, :, 6] = -Bd
        Av[self.k_dyn:self.k_dyn + 35 * N] = blk.ravel()
        r0 = self.row_dyn
        l[r0:r0 + 5 * N] = ed.ravel()
        u[r0:r0 + 5 * N] = ed.ravel()
        # corridor on the collision-box points
        ii = np.arange(1, N + 1)
        cmy, smy = np.cos(mbar[ii]), np.sin(mbar[ii])
        keff = kr[ii] / D[ii]
        vals = np.zeros((N, nl, 6))
        lo_all = np.zeros((N, nl))
        hi_all = np.zeros((N, nl))
        for j, lp in enumerate(self.lpts):
            sP = st[ii] + lp * cmy / D[ii]
            lo, hi = self._corridor(sP, boxes, sides, w_h, s_block)
            const = lp * (smy - cmy * mbar[ii]) - keff * lp * lp / 2
            vals[:, j, 0] = 1.0
            vals[:, j, 1] = lp * cmy
            vals[:, j, 2] = 1.0
            vals[:, j, 3] = 1.0
            vals[:, j, 4] = lp * cmy
            vals[:, j, 5] = -1.0
            lo_all[:, j] = lo - const
            hi_all[:, j] = hi - const
        Av[self.k_cor:self.k_cor + 6 * nl * N] = vals.ravel()
        rc = self.row_cor
        lv = np.zeros((N, nl, 2))
        uv = np.zeros((N, nl, 2))
        lv[:, :, 0] = lo_all
        uv[:, :, 0] = np.inf
        lv[:, :, 1] = -np.inf
        uv[:, :, 1] = hi_all
        l[rc:rc + 2 * nl * N] = lv.ravel()
        u[rc:rc + 2 * nl * N] = uv.ravel()
        # steering angle and rate
        rd = self.row_del
        l[rd:rd + N] = -STEER_MAX * 0.98
        u[rd:rd + N] = STEER_MAX * 0.98
        rw = self.row_w
        l[rw:rw + N] = -STEER_RATE * 0.95
        u[rw:rw + N] = STEER_RATE * 0.95
        rs = self.row_s
        l[rs:rs + 2 * (N + 1)] = 0.0
        u[rs:rs + 2 * (N + 1)] = np.inf
        # lateral acceleration |v r| <= a_lat_hard (soft)
        rl = self.row_lat
        vlat = np.zeros((N, 2, 2))
        vlat[:, 0, 0] = vp[ii]
        vlat[:, 0, 1] = 1.0
        vlat[:, 1, 0] = vp[ii]
        vlat[:, 1, 1] = -1.0
        Av[self.k_lat:self.k_lat + 4 * N] = vlat.ravel()
        ll = np.zeros((N, 2))
        ul = np.zeros((N, 2))
        ll[:, 0] = -self.a_lat_hard
        ul[:, 0] = np.inf
        ll[:, 1] = -np.inf
        ul[:, 1] = self.a_lat_hard
        l[rl:rl + 2 * N] = ll.ravel()
        u[rl:rl + 2 * N] = ul.ravel()
        # cost
        lo_a, hi_a = self._corridor(st, boxes, sides, w_h, s_block)
        lo_b, hi_b = self._corridor(st + BOX_FRONT, boxes, sides, w_h, s_block)
        lo_c, hi_c = np.maximum(lo_a, lo_b), np.minimum(hi_a, hi_b)
        base = dpath if (dpath is not None and self.ref_dp) else np.zeros_like(lo_c)
        dref = np.minimum(np.maximum(base, lo_c + self.c_pref), hi_c - self.c_pref)
        narrow = (hi_c - lo_c) < 2 * self.c_pref
        dref[narrow] = 0.5 * (lo_c[narrow] + hi_c[narrow])
        if self.hp is not None:
            # lateral-acceleration limit at the slowest speed reachable with planned braking: far away the speed
            # profile adapts to the path, so only the near part is really constrained
            v0_ = float(vp[0])
            cum = np.concatenate([[0.0], np.cumsum(dpl)])
            vreach = np.sqrt(np.maximum(v0_ * v0_ - 2.0 * self._abrake(self.tab_plan, v0_) * cum, 1.0))
            vlat = np.minimum(vp, vreach)
            return self._solve_hpipm(X0, Ad, Bd, ed, vals[:, :, 1], lo_all, hi_all, vp, dref, kr, dt, vlat)
        Pd = np.full(self.nv, 1e-6)
        q = np.zeros(self.nv)
        xi = 5 * ii
        Pd[xi + 0] = 2 * self.q_d
        q[xi + 0] = -2 * self.q_d * dref[ii]
        Pd[xi + 1] = 2 * self.q_mu
        wa = self.q_a * vp[ii] ** 2
        Pd[xi + 3] = 2 * wa
        q[xi + 3] = -2 * wa * vp[ii] * kr[ii]
        Pd[self.iW0:self.iW0 + N] = 2 * self.r_w
        Pd[self.iS0:self.iS0 + N + 1] = 2 * self.rho2
        q[self.iS0:self.iS0 + N + 1] = self.rho1
        Pd[self.iT0:self.iT0 + N + 1] = 2 * self.rhoa2
        q[self.iT0:self.iT0 + N + 1] = self.rhoa1
        if self.qp is None:
            self._qp_setup(Pd, q, Av, l, u)
        else:
            self.qp.update(Px=Pd, q=q, Ax=Av[self.A_perm], l=l, u=u)
            if self.shift_ws and self.sol is not None and self.z_prev is not None:
                # warm start: previous primal solution moved to the new stations
                ss_ = self.sol[0]
                zp = self.z_prev
                Xp = zp[:5 * (N + 1)].reshape(N + 1, 5)
                Xw = np.stack([np.interp(st, ss_, Xp[:, k]) for k in range(5)], 1)
                Xw[0] = X0
                ww = np.interp(st[:N], ss_[:N], zp[self.iW0:self.iW0 + N])
                zw = zp.copy()
                zw[:5 * (N + 1)] = Xw.ravel()
                zw[self.iW0:self.iW0 + N] = ww
                self.qp.warm_start(x=zw)
        res = self.qp.solve()
        status = res.info.status_val
        self.qp_info = (res.info.iter, status)
        if status not in (1, 2, 7, 8) or res.x is None or not np.all(np.isfinite(res.x)):
            self.qp = None
            self.z_prev = None
            return None
        self.z_prev = res.x.copy()
        z = res.x
        Xs = z[:5 * (N + 1)].reshape(N + 1, 5)
        w = z[self.iW0:self.iW0 + N]
        slack = z[self.iS0:self.iS0 + N + 1]
        return Xs, w, slack, dt

    def _solve_hpipm(self, X0, Ad, Bd, ed, cmu, lo_all, hi_all, vp, dref, kr, dt, vlat):
        N, nl = self.n, len(self.lpts)
        Q = np.zeros((N + 1, 5, 5))
        idx = np.arange(5)
        wa = self.q_a * vp ** 2
        diag = np.zeros((N + 1, 5))
        diag[:, 0] = 2 * self.q_d
        diag[:, 1] = 2 * self.q_mu
        diag[:, 2] = 2e-6
        diag[:, 3] = 2 * wa
        diag[:, 4] = 2e-6
        diag[0] = 2e-6
        Q[:, idx, idx] = diag
        q = np.zeros((N + 1, 5))
        q[1:, 0] = -2 * self.q_d * dref[1:]
        q[1:, 3] = -2 * wa[1:] * vp[1:] * kr[1:]
        Cm = np.zeros((N + 1, nl + 1, 5))
        Cm[1:, :nl, 0] = 1.0
        Cm[1:, :nl, 1] = cmu
        Cm[1:, nl, 3] = vlat[1:]
        lg = np.zeros((N + 1, nl + 1))
        ug = np.zeros((N + 1, nl + 1))
        lg[1:, :nl] = lo_all
        ug[1:, :nl] = hi_all
        lg[1:, nl] = -self.a_lat_hard
        ug[1:, nl] = self.a_lat_hard
        bad = lg[1:, :nl] > ug[1:, :nl]          # closed corridor: keep the problem well posed (slack does the rest)
        if np.any(bad):
            mid = 0.5 * (lg[1:, :nl] + ug[1:, :nl])
            lg[1:, :nl] = np.where(bad, mid, lg[1:, :nl])
            ug[1:, :nl] = np.where(bad, mid, ug[1:, :nl])
        lbx = np.full((N + 1, 1), -STEER_MAX * 0.98)
        ubx = np.full((N + 1, 1), STEER_MAX * 0.98)
        lbu = np.full((N, 1), -STEER_RATE * 0.95)
        ubu = np.full((N, 1), STEER_RATE * 0.95)
        if self.hp_all:
            hp = self.hp
            hp.bA[:N] = np.transpose(Ad, (0, 2, 1))
            hp.bB[:N, 0, :] = Bd
            hp.bb[:N] = ed
            hp.bQ[:] = Q
            hp.bq[:] = q
            hp.bC[:] = np.transpose(Cm, (0, 2, 1))
            hp.blg[:] = lg
            hp.bug[:] = ug
            hp.blbx[0, :5] = X0
            hp.bubx[0, :5] = X0
            hp.blbx[1:, 0] = lbx[1:, 0]
            hp.bubx[1:, 0] = ubx[1:, 0]
            status, iters, Xs = hp.solve_all()
        else:
            status, iters, Xs = self.hp.solve(Ad, Bd[:, :, None], ed, Q, q, Cm, lg, ug, X0, lbx, ubx, lbu, ubu)
        self.qp_info = (iters, status)
        if status not in (0, 1) or not np.all(np.isfinite(Xs)):
            return None
        sl = np.zeros(N + 1)
        sl[1:] = self._hp_slack(Xs, cmu, lg, ug)
        return Xs, np.diff(Xs[:, 4]) / dt, sl, dt

    def _hp_slack(self, Xs, cmu, lg, ug):
        nl = len(self.lpts)
        val = Xs[1:, 0:1] + cmu * Xs[1:, 1:2]
        return np.maximum(np.max(lg[1:, :nl] - val, axis=1), np.max(val - ug[1:, :nl], axis=1)).clip(0.0)

    # ------------------------------------------------------------------ main
    def act(self, scan, x, y, yaw, v, yaw_rate, t):
        x, y, yaw, v, yaw_rate, t = float(x), float(y), float(yaw), float(v), float(yaw_rate), float(t)
        scan = np.asarray(scan, dtype=float)
        # the compiled kernels assume one range per beam of the configured LiDAR (1080) and do not check bounds: a
        # scan of another length was read out of bounds silently (the numpy path raised IndexError)
        bad_scan = scan.shape != self.ca.shape
        if bad_scan:
            self.n_bad_scan += 1
            if self.n_bad_scan == 1:
                print(f'CorridorMPC: scan with {scan.size} ranges, expected {self.ca.size}: treated as an invalid '
                      f'sample (last command repeated, then stop)', file=sys.stderr, flush=True)
        if bad_scan or not (math.isfinite(x) and math.isfinite(y) and math.isfinite(yaw) and math.isfinite(v)
                            and math.isfinite(yaw_rate) and math.isfinite(t)):
            # invalid state sample: keep it out of every estimate (a NaN yaw rate poisoned beta_est for good, a NaN
            # pose / speed / stamp raised in the DP or _observe -> avoid_node stops the car for good).  Repeat the
            # last command for a few calls (like a skipped call), then stop.
            self.n_invalid += 1
            if not self.hist:
                return 0.0, 0.0
            steer_h, speed_h = self.hist[-1][1], (self.hist[-1][2] if self.n_invalid <= 5 else 0.0)
            if speed_h != self.hist[-1][2]:
                # the stop command is part of what the car does: the actuator models / estimators read the history
                if math.isfinite(t) and self.t_off is not None:
                    t_h = max(t - self.t_off, self.hist[-1][0])
                else:
                    t_h = self.hist[-1][0] + self.n_invalid * self.t_period
                self.hist.append((t_h, steer_h, speed_h))
            return steer_h, speed_h
        self.n_invalid = 0
        t = self._clock(t)
        if np.isneginf(scan).any():
            # REP 117: -inf = closer than range_min (a return at the sensor), +inf = no return (free), NaN = invalid
            scan = np.where(np.isneginf(scan), 0.0, scan)
        N = self.n
        # pose discontinuity (teleport without reset(): /initialpose handled after the next scan, localisation jump):
        # the local Frenet search, the course angle, the lap-wrap test and the previous plan across the jump are
        # meaningless, and the scan of a glitched pose would paint the walls as obstacles
        teleport = False
        if self.pose_prev is not None:
            tp_, xp_, yp_, vp_ = self.pose_prev
            teleport = math.hypot(x - xp_, y - yp_) > 0.5 + 1.5 * max(abs(v), abs(vp_)) * (t - tp_)
        jump_d = 0.0 if self.pose_prev is None else math.hypot(x - self.pose_prev[1], y - self.pose_prev[2])
        self.pose_prev = (t, x, y, v)
        # perception is skipped on a teleport call: the compiled _boxes then used the previous call's view of the
        # remembered cells (ranges / bearings from the old pose) and the extent latches were set from it
        self._no_latch = teleport
        if teleport:
            self.fi = None
            self.s_prev = None
            self.last = None
            self.sol_bad = True
            self.prof = None
            self.dp_res = None
            self._view = None
            if jump_d > 2.0:
                # relocated (not a glitch): what was remembered, the side commitments and the latches belong to the
                # old place / pose (v1's lap-wrap test cleared them on a backward jump, the teleport branch skips it)
                if self.mem is not None:
                    self.mem[self.active] = 0
                self._mem_clear()
                self.prev_sides = []
                self.seen_boxes = []
                self.box_ext = []
                self.box_conf = []
                self.conf_idx = {}
        # command history: keep what can still act (largest latency hypothesis) plus the command acting before it
        k_old = 0
        while k_old + 1 < len(self.hist) and self.hist[k_old + 1][0] < t - 0.3 - self.lat_max:
            k_old += 1
        self.hist = self.hist[k_old:]
        self._H = np.array(self.hist, dtype=np.float64).reshape(len(self.hist), 3) if self.jit else None
        self.a_acc_eff = self.a_acc_launch if (self.a_acc_launch > 0.0 and v < self.v_launch) else self.a_acc
        # call period: median of the recent gaps (a single stall does not change it)
        gap = None if self.t_last_call is None else t - self.t_last_call
        self.t_last_call = t
        if self.stall_db < 1.0:
            # stall-adaptive lateral acceleration: a gap > 1.75 measured periods means the previous command was held
            # that much longer (node / CPU stall); when such stalls take more than stall_db of the last 5 s the
            # lateral-acceleration target is lowered (heavy stalls: wall hits in fast corners after 2-4 stalls in a s)
            if gap is not None and 1.75 * self.t_period < gap:
                self.stalls.append((t, min(gap, 0.5) - self.t_period))
            while self.stalls and self.stalls[0][0] <= t - 5.0:
                self.stalls.pop(0)
            frac = sum(e[1] for e in self.stalls) / 5.0
            self.a_lat = self.p['a_lat'] * max(self.stall_min, 1.0 - max(frac - self.stall_db, 0.0))
        if self.lat_alat is not None:
            # delay-adaptive lateral acceleration (time-trial option, see header); t_lat from the previous call
            l0, l1, a1 = self.lat_alat
            f = min(max((self.t_lat - l0) / max(l1 - l0, 1e-6), 0.0), 1.0)
            ratio = 1.0 + f * (a1 / self.p['a_lat'] - 1.0)
            self.a_lat = min(self.a_lat if self.stall_db < 1.0 else self.p['a_lat'], self.p['a_lat'] * ratio)
            self.a_lat_hard = self.p['a_lat_hard'] * ratio
        if gap is not None and 0.001 < gap < 0.25:
            self.gaps = self.gaps[-14:] + [gap]
            self.t_period = sorted(self.gaps)[len(self.gaps) // 2]
        if self.lat_est:
            self._lat_update(t, v)
        v_raw = v
        x, y, yaw, v, yaw_rate = self._observe(t, x, y, yaw, v, yaw_rate)
        if self.fg_check and (self.lat_est or self.c_adapt):
            self._fg_window(t, v_raw)
        if self.c_tables:
            self._set_tables(self.c_tyre)
        t_bite = self.t_lat + self.t_buf
        px, py, pyaw, pv, pbeta, pr, pdelta, pa = self._predict(t, x, y, yaw, v, yaw_rate, t_bite)
        vv = max(v, 0.0)
        # ---- Frenet (current, for lap bookkeeping) and predicted
        s_c, d_c, self.fi = self._frenet(x, y, self.fi)
        wrapped = False
        if self.s_prev is not None and self.s_prev - s_c > self.L / 2:
            wrapped = True
            # crossed the finish line: new layout -> forget, ignore the next stale scan(s)
            if self.mem is not None:
                self.mem[self.active] = 0
            self._mem_clear()
            self.blind_until = t + self.blind_after_wrap
            self.prev_sides = []
            self.seen_boxes = []
            self.box_ext = []
            self.box_conf = []
            self.conf_idx = {}
            if self.sol is not None:
                self.sol = (self.sol[0] - self.L,) + self.sol[1:]
            if self.prof is not None:
                self.prof = (self.prof[0] - self.L,) + self.prof[1:]
        self.s_prev = s_c
        s0, d0, i0 = self._frenet(px, py, self.fi)
        if s0 - s_c < -self.L / 2:
            s0 += self.L
        elif s0 - s_c > self.L / 2:
            s0 -= self.L
        mu0 = float(_wrap(pyaw - self._heading(s0)))
        X0 = np.array([d0, mu0, pbeta, pr, pdelta])
        # ---- perception
        lx, ly = x + LIDAR_X * math.cos(yaw), y + LIDAR_X * math.sin(yaw)
        if not teleport:
            self._perceive(scan, lx, ly, yaw, t)
        # (after a long gap -- node stall, paused sim -- the car drove on with the last command: a DP-only call would
        # steer by the old plan read 0.3 .. 1 s ahead, far from where the car really is -> plan everything now)
        full = (self.dp_res is None or self.plan is None or not self.split or wrapped or t - self.plan['t'] > 0.2)
        do_dp = full or self.phase == 0
        # the MPC also runs when its plan is older than ~one period (a call after a stall)
        # (1.75: a one-tick slip of a 50 Hz node on a 100 Hz sim -- a 30 ms gap -- sat exactly on 1.5 periods, where
        # float rounding of the stamps decided)
        do_mpc = full or self.phase == 1 or t - self.plan['t'] > 1.75 * self.t_period
        self.phase = 1 - self.phase if not full else 1
        w_h = BOX_HW + self.m_hard + self.m_hard_v * vv
        # wall clearance grows with speed like the obstacle margin (tracking error in the tight S-bends of narrow tracks,
        # which the car drives along the walls); capped: at high speed a growing margin narrows obstacle gaps next to
        # the wall and makes the side choice depend on the current speed
        wc_run = self.wall_clear + self.wall_clear_v * min(vv, self.wall_clear_vmax)
        self._set_wall(wc_run)
        if do_dp:
            boxes = self._boxes(s0, scan, lx, ly, yaw)
            firsts = [self._first_seen(0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]), t) for b in boxes]
            self.box_age = [t - f for f in firsts]
            self.seen_boxes = [(0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]), f) for b, f in zip(boxes, firsts)]
            # ---- DP side choice + corridor
            st, dpath, sides, s_block = self._dp(s0, d0, mu0 + pbeta, boxes, vv, w_h)
            # a forced side change invalidates the commitments of its cluster: re-plan the cluster jointly
            mids = [0.5 * (b[0] + b[1]) for b in boxes]
            flipped = []
            for i, b in enumerate(boxes):
                p_ = self._prev_side(mids[i], 0.5 * (b[2] + b[3]))
                if p_ is not None and sides[i] != 0 and sides[i] != p_:
                    flipped.append(i)
            relax = ()
            if flipped:
                relax = {j for j in range(len(boxes)) for i in flipped if abs(mids[j] - mids[i]) < self.cluster_ds}
                st, dpath, sides, s_block = self._dp(s0, d0, mu0 + pbeta, boxes, vv, w_h, relax=relax)
            if s_block is not None and (self.gap_fallback or self.escape):
                # no complete path: first allow the narrow gaps, then (last resort) release the hard commitments
                tries = ([(True, False)] if self.gap_fallback else []) + ([(False, True), (False, False)] if self.escape else [])
                for hard_, gap_ in tries:
                    r_ = self._dp(s0, d0, mu0 + pbeta, boxes, vv, w_h, relax=relax, hard=hard_, gap=gap_)
                    if r_[3] is None:
                        st, dpath, sides, s_block = r_
                        break
            if s_block is not None:
                # no complete path: keep every obstacle in the corridor (previous side, else the side the car is on)
                for bi_, b in enumerate(boxes):
                    if sides[bi_] != 0:
                        continue
                    smid, dc = 0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3])
                    prev = self._prev_side(smid, dc)
                    sides[bi_] = prev if prev is not None else (1 if d0 > dc else -1)
            self.prev_sides = [(0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]), sd) for b, sd in zip(boxes, sides) if sd != 0]
            self.dp_res = (boxes, sides, s_block, st, dpath)
        else:
            boxes, sides, s_block, st_old, dpath_old = self.dp_res
            st = s0 + self.ds * np.arange(N + 1)
            dpath = np.interp(st, st_old, dpath_old)
        if do_mpc:
            new = self._plan(t, st, X0, dpath, boxes, sides, s_block, w_h, pv, scan, lx, ly, yaw, vv)
            old = self.plan
            if (new['failed'] and old is not None and not old['bad'] and self.n_fail <= 2 and not full
                    and t - old['t'] < 0.12):
                # a failed solve (HPIPM min-step / NaN status) used to mean an emergency brake + geometric path
                # following for that call; the previous plan is 1-2 calls old, exactly what every DP-only call of the
                # split schedule drives on -> keep it (emergency only after the third failure in a row)
                pass
            else:
                self.plan = new
        plan = self.plan
        # ---- commands from the latest plan (time-shifted when it was computed in an earlier call)
        Xs, dts, vprof, cum, emergency, slack, s_vis = (plan['Xs'], plan['dts'], plan['vprof'], plan['cum'],
                                                        plan['emergency'], plan['slack'], plan['s_vis'])
        shift_t = t - plan['t']
        shift_s = s0 - plan['st'][0]
        s_ahead = max(0.3, vv * 0.10)
        v_tgt = float(np.interp(s_ahead + shift_s, cum, vprof))
        a_em = self._abrake(self.tab_emerg, vv)
        # deadlock recovery: stopped for a while (e.g. reset inside an obstacle footprint) -> creep along the plan
        if vv < 0.15 and (emergency or s_block is not None):
            if self.t_stop is None:
                self.t_stop = t
        elif vv > 0.5:
            self.t_stop = None
        creep = self.t_stop is not None and t - self.t_stop > self.creep_after
        if creep:
            emergency = False
        if emergency:
            a_des = -a_em
        else:
            # feedback on the measured lateral acceleration (the referee's |v r|): cap the target speed
            kap_m = abs(yaw_rate) / max(vv, 0.5)
            if kap_m > 1e-3 and vv > 1.0:
                v_tgt = min(v_tgt, math.sqrt(self.a_lat_fb / kap_m))
            a_des = (v_tgt * v_tgt - pv * pv) / (2 * s_ahead)
            # accelerating in a corner grows the sideslip and the yaw rate: share the grip
            if a_des > 0:
                load = min(1.0, abs(vv * yaw_rate) / self.a_lat)
                a_des = min(a_des, self.a_acc * max(self.acc_corner_min, 1.0 - load * load))
            if creep:
                a_des = (self.creep_v - vv) * 4.0
            a_des = min(max(a_des, -a_em), self.a_acc_eff)
        dt_c = 0.02 if self.t_prev_cmd is None else min(0.1, max(0.005, t - self.t_prev_cmd))
        j_dn = self.jerk if (self.em_jerk is None or not emergency) else self.em_jerk
        a_des = min(max(a_des, self.a_cmd - j_dn * dt_c), self.a_cmd + self.jerk * dt_c)
        self.t_prev_cmd = t
        self.a_cmd = a_des
        v_at = vv + pa * self.t_lat           # speed when the command reaches the controller
        kp = KP_ACC if a_des >= 0 else KP_BRK
        if self.hold_comp:
            # the command is held for one call period: choose it so that the P-controller's mean acceleration over
            # the hold (not its initial one) is a_des; (v_cmd - v)(1 - exp(-kp T)) = a_des T
            speed = v_at + a_des * self.t_period / (1.0 - math.exp(-kp * self.t_period))
        else:
            speed = v_at + a_des / kp
        speed = max(0.0, speed)
        if vv < 0.05 and a_des <= 0:
            speed = 0.0
        # the command is the measured speed plus a P-controller offset (<= ~1.3 m/s): a glitched speed sample (fuzz:
        # v = 1e6) went straight to the car.  Never binds in normal driving.
        speed = min(speed, self.v_max + 2.0)
        # ---- steering: planned angle one control period after the bite time
        if plan['bad']:
            # infeasible corridor at this speed: brake and follow the (curvature limited) DP path geometrically
            ld = max(0.7, 0.3 * vv)
            sts = plan['st']
            tx, ty = self._xy(np.array([s0 + ld]), np.interp(np.array([s0 + ld]), sts, plan['dpath']))
            ax_, ay_ = px + LIDAR_X * 0 * math.cos(pyaw), py
            alpha = _wrap(math.atan2(float(ty[0]) - ay_, float(tx[0]) - ax_) - pyaw)
            dist = max(math.hypot(float(tx[0]) - ax_, float(ty[0]) - ay_), 0.3)
            kap = 2.0 * math.sin(alpha) / dist
            fac = max(WB + float(k_us(a_des, self.c_tyre)) * vv * vv, 0.3 * WB)
            steer = kap * fac
        else:
            tcum = np.concatenate([[0.0], np.cumsum(dts)])
            # the command is held until the next call: target the planned angle one (measured) period ahead
            tq = self.t_period + shift_t
            steer = float(np.interp(tq, tcum, Xs[:, 4]))
            if self.gain_comp:
                # the yaw gain depends strongly on the longitudinal acceleration (load transfer): when the commanded
                # acceleration differs from the one the plan assumed, scale the steering by the steady-state gain ratio
                a_pl = float(np.interp(tq, tcum[:-1], plan['acc']))
                vb2 = vv * vv
                f_cmd = max(WB + float(k_us(a_des, self.c_tyre)) * vb2, 0.3 * WB)
                f_pl = max(WB + float(k_us(a_pl, self.c_tyre)) * vb2, 0.3 * WB)
                steer *= min(max(f_cmd / f_pl, 0.4), 2.0)
        steer = min(max(steer, -STEER_MAX), STEER_MAX)
        self.hist.append((t, steer, speed))
        if self.dbg is not None:
            self.dbg.append(dict(t=t, x=x, y=y, v=v, r=yaw_rate, s=s0, d=d0, mu=mu0, beta=pbeta, steer=steer,
                                 speed=speed, a_des=a_des, boxes=[list(b) for b in boxes], sides=list(sides),
                                 s_block=s_block, s_vis=s_vis, emergency=emergency, dpath=dpath.tolist(),
                                 dS=Xs[:, 0].tolist(), st=plan['st'].tolist(), vprof=vprof.tolist(),
                                 slack=None if slack is None else float(np.max(slack)),
                                 pred=(px, py, pyaw, pv, pbeta, pr, pdelta), beta_est=self.beta_est,
                                 delta_est=self.delta_est, qp=self.qp_info, dS_full=Xs[:, 0].tolist(),
                                 dS_delta=Xs[:, 4].tolist(), mpc=do_mpc, t_lat=self.t_lat, c=self.c_tyre))
        return steer, speed

    def _set_wall(self, wc):
        """wall clearance of this call's DP / corridor (and the compiled wall rows of the same value)"""
        self.wc_eff = wc
        if self.jit:
            kc = int(round(min(max(wc, 0.0), 0.6) / 0.01))      # the row _wall_bounds uses
            self.wlo_c = np.ascontiguousarray(self.wlo_tab[kc])
            self.whi_c = np.ascontiguousarray(self.whi_tab[kc])

    def _plan(self, t, st, X0, dpath, boxes, sides, s_block, w_h, pv, scan, lx, ly, yaw, vv):
        """MPC + speed profile on the current stations; returns the plan used for the commands"""
        N = self.n
        if (self.jit and self.sol is not None and not self.sol_bad and self.reuse_prof and self.prof is not None
                and self.acc_model):
            sp_, vpp, dplp, vpf = self.prof
            kr, Xbar, dpl, vpred, acc = _plan_prep_k(st, self.fk, float(self.L), float(self.fds), self.sol[0],
                                                     self.sol[1], dpath, X0, sp_, dplp, vpf, float(pv),
                                                     float(self.a_cmd), float(self.a_acc_eff), float(self.jerk),
                                                     self.tab_emerg, float(self.t_period))
            return self._plan_rest(t, st, X0, dpath, boxes, sides, s_block, w_h, pv, scan, lx, ly, yaw, vv, kr, Xbar,
                                   dpl, vpred, acc)
        kr = self._finterp(self.fk, st)
        if self.sol is not None and not self.sol_bad:
            ss_, Xp, wp = self.sol
            Xbar = np.stack([np.interp(st, ss_, Xp[:, k]) for k in range(5)], 1)
            beyond = st > ss_[-1]
            Xbar[beyond, 0] = dpath[beyond]
            Xbar[beyond, 1:] = 0.0
        else:
            Xbar = np.zeros((N + 1, 5))
            Xbar[:, 0] = dpath
            Xbar[1:, 1] = np.arctan(np.gradient(dpath, st))[1:]
        Xbar[0] = X0
        # speed prediction for the MPC timing: previous plan's, moved to the new stations
        if self.reuse_prof and self.prof is not None:
            sp_, vpp, dplp, vpf = self.prof
            vpred = np.interp(st, sp_, vpp)
            vpred[0] = pv
            dpl = np.interp(st[:-1], sp_[:-1], dplp)
            vtarget = np.interp(st, sp_, vpf)
        else:
            vtarget, vpred, acc, dpl, s_vis = self._speed_profile(st, Xbar[:, 0], Xbar[:, 1], Xbar[:, 2], kr, pv,
                                                                  scan, lx, ly, yaw, s_block)
        if self.acc_model:
            # the speed / acceleration the command logic will really produce (jerk limited, from the current command)
            cum0 = np.concatenate([[0.0], np.cumsum(dpl)])
            vpred, acc = _cmd_sim(vtarget.astype(np.float64), cum0, dpl.astype(np.float64), float(pv), float(self.a_cmd),
                                  float(self.a_acc_eff), float(self.jerk), self.tab_emerg, float(self.t_period))
        else:
            vpred, acc = self._stable_accel(vpred, dpl)
        return self._plan_rest(t, st, X0, dpath, boxes, sides, s_block, w_h, pv, scan, lx, ly, yaw, vv, kr, Xbar, dpl,
                               vpred, acc)

    def _plan_rest(self, t, st, X0, dpath, boxes, sides, s_block, w_h, pv, scan, lx, ly, yaw, vv, kr, Xbar, dpl, vpred,
                   acc):
        N = self.n
        acc_mpc = acc.copy()
        out = self._mpc(st, X0, Xbar, kr, vpred, acc, dpl, boxes, sides, w_h, s_block, dpath)
        slack = None
        if out is not None:
            Xs, w, slack, dts = out
            self.sol = (st, Xs, w)
            self.n_fail = 0
        else:
            self.n_fail += 1
            Xs, dts = Xbar, dpl / np.maximum(vpred[:N], self.v_min_plan)
        kyaw = None
        if self.yaw_lim > 0.0 and out is not None:
            kyaw = self.yaw_lim * np.abs(Xs[:, 3]) / np.maximum(vpred, self.v_min_plan)
        vprof, vpred, acc, dpl, s_vis = self._speed_profile(st, Xs[:, 0], Xs[:, 1], Xs[:, 2], kr, pv,
                                                            scan, lx, ly, yaw, s_block, kyaw=kyaw)
        if self.acc_model and self.resolve_da > 0.0 and out is not None:
            # the new profile changes the acceleration the car will have near term (e.g. a new swerve starts the
            # braking now): the load transfer then changes the yaw response a lot -> solve once more with it
            cum1 = self._sp_cum if self.jit and not self.vis_obs_stop else np.concatenate([[0.0], np.cumsum(dpl)])
            v2, acc2 = _cmd_sim(vprof.astype(np.float64), cum1, dpl.astype(np.float64), float(pv), float(self.a_cmd),
                                float(self.a_acc_eff), float(self.jerk), self.tab_emerg, float(self.t_period))
            kn = self.resolve_n
            if float(np.max(np.abs(acc2[:kn] - acc_mpc[:kn]))) > self.resolve_da:
                out2 = self._mpc(st, X0, Xs, kr, v2, acc2, dpl, boxes, sides, w_h, s_block, dpath)
                if out2 is not None:
                    out = out2
                    Xs, w, slack, dts = out2
                    self.sol = (st, Xs, w)
                    acc_mpc = acc2.copy()
                    vprof, vpred, acc, dpl, s_vis = self._speed_profile(st, Xs[:, 0], Xs[:, 1], Xs[:, 2], kr, pv,
                                                                        scan, lx, ly, yaw, s_block)
                    self.n_resolve += 1
        self.prof = (st, vpred, dpl, vprof)
        cum = self._sp_cum if self.jit and not self.vis_obs_stop else np.concatenate([[0.0], np.cumsum(dpl)])
        emergency = False
        bad = out is None or (slack is not None and float(np.max(slack)) > self.slack_bad)
        self.sol_bad = bad
        if slack is not None:
            near = cum <= self._dstop(vv) + self.slack_reach
            if np.any(slack[near] > self.slack_brake):
                emergency = True
        if out is None and self.n_fail > 2:
            emergency = True
        if bad:
            emergency = True
        if (emergency and self.soft_em > 0.0 and not bad and out is not None and s_block is None
                and vv <= self.soft_em_arm and slack is not None):
            # soft emergency at low speed: a slack-only emergency with a small slack (the speed-dependent wall margin
            # narrowing an obstacle-wall gap by a few cm) stopped the car beside the obstacle, where the slack of the
            # current pose kept the emergency on -> stop / creep 0.6 m/s / stop at up to 10 s per metre.  Once slow,
            # drive on at soft_em_v instead; above soft_em_arm the emergency brake is unchanged.
            near = cum <= self._dstop(vv) + self.slack_reach
            mx = float(np.max(slack[near])) if np.any(near) else 0.0
            if mx <= self.soft_em:
                emergency = False
                vprof = np.minimum(vprof, self.soft_em_v)
        return dict(t=t, st=st, Xs=Xs, dts=dts, vprof=vprof, cum=cum, emergency=emergency, slack=slack, s_vis=s_vis,
                    bad=bad, dpath=dpath, acc=acc_mpc, failed=out is None)

    def _stable_accel(self, vpred, dpl):
        """the car cannot brake harder than the (yaw-stable) emergency limit: keep the speed prediction and the
        acceleration used for the load transfer of the MPC model inside that range"""
        return _stable_accel_k(vpred.astype(np.float64), dpl.astype(np.float64), self.tab_emerg, float(self.a_acc))

    def _dstop(self, v):
        if v <= 0:
            return 0.0
        return float(np.interp(v, self.tab_v, self.tab_dstop)) + v * self.t_react
