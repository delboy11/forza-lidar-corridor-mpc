"""Corridor MPC: one planner (DP side choice on a Frenet grid -> corridor) and one controller (spatial LTV-MPC on the
single-track model + speed profile) for a known track with or without static obstacles.

One class, CorridorMPC; the configurations are kwargs profiles: profiles/berlin_tt.json / berlin_tt_safe.json (clear
track), profiles/berlin_obs.json (random static obstacles: the obstacle options below on).  drivers.cmpc_berlin_obs
re-exports the class under its older import path.  Input per call: act(scan, x, y, yaw, v, yaw_rate, t) -> (steering
angle, speed); in the exam setup the pose arguments come from drivers.lidar_loc.LidarLocDriver (LiDAR scan-to-map
localisation), never from ground truth.

Pipeline per act():
  1. state estimation / delay compensation: the steering servo (3.2 rad/s, command latency + 2-step buffer) and the
     speed P-controller are simulated from the commands this driver issued; the CG sideslip (not measured) is
     estimated with the single-track model driven by the measured yaw rate and corrected by the course angle of
     successive poses; the state is predicted with the nonlinear single-track model to the time the new steering
     command bites.
  2. perception: scan hits -> map frame; hits farther than obs_edt from every occupied map cell (distance transform of
     the map) are obstacle cells.  Lap-scoped memory (cleared when s wraps, with a short blind window for a stale
     scan) and free-space carving of cells the beams see through.  Cells are clustered; each cluster becomes a Frenet
     box of its measured extent (see 'Boxes' below).
  3. DP over a Frenet grid picks one consistent side for every visible box (all of them, not only the closest), with a
     side-flip penalty (hysteresis) that becomes almost hard close to the box.  The chosen sides define a corridor
     lb(s) <= d <= ub(s) for the body line (box and wall bounds inflated by the collision-box half width + margin).
  4. spatial LTV-MPC (HPIPM) on the single-track model with linear tyres in Frenet coordinates: states d, mu (yaw
     relative to the reference line), beta, r, delta; input steering rate (|.| <= 3.2 rad/s); exact discretisation per
     station along the planned speed profile, load transfer from the acceleration the jerk-limited speed command will
     really produce.  Corridor constraints on three points of the collision box (soft, heavy penalty), soft
     lateral-acceleration limit |v r| <= a_lat_hard.  When the new speed profile changes the near-term acceleration by
     more than resolve_da, the MPC is solved a second time with it.
  5. speed profile on the MPC path: lateral-acceleration limit, planned braking only within the steerable
     (yaw-stable) deceleration, plus emergency envelopes: stop (with the larger, still stable deceleration) before a
     closed corridor, and reach the first path point the scan cannot see no faster than v_vis_safe.  Emergency brake
     (not jerk limited, em_jerk) when the MPC needs corridor slack near the car; a small slack at low speed is a speed
     cap instead (soft_em).

Estimation and timing (all measured online, nothing assumed about the platform):
  * lat_est: command latency by a multiple-model estimate (candidates lat_min..lat_max in 10 ms steps: the speed
    P-controller driven by this driver's own commands delayed by each candidate, compared with the measured speed);
    t_lat is only the prior.  Under the LiDAR wrapper the wrapper's own scan-based estimate is used instead.
  * t_period: the call period is measured (median of recent gaps) and used for the steering target (planned angle one
    period after the bite) and for the jerk limit per call; hold_comp: a speed command is held for one period, so it is
    chosen to give the desired mean acceleration over the hold.  The MPC is also solved when its plan is older than
    1.75 periods (a call after a stall).  stall_db: when held commands (call gaps > 1.75 periods) take more than
    stall_db of the last 5 s, a_lat is lowered.  lat_alat: a_lat / a_lat_hard scaled down when the estimated latency
    exceeds lat_alat[0].
  * c_adapt: online tyre-stiffness scale c (a grid of parallel single-track models driven by the issued commands; the
    output error against the measured yaw rate picks c); c scales the model everywhere (observer, prediction, MPC,
    deceleration tables, DP curvature).
  * pose_filt: x, y, yaw, v, r are filtered against the model prediction with steady-state Kalman gains from the
    measured noise level (exact inputs keep gain 1); outlier gates (pose_out); pose jumps larger than pose_gate
    (relocalisation) are taken as they are.  Invalid samples (non-finite values, a scan of the wrong length) repeat the
    last command for a few calls, then stop.
  * fg_check (off in the obstacle profile): open-loop check that the car follows the commands, which gates the
    adaptive estimators.

Reference line (time trial):
  * raceline_file: a reference line shipped with the driver replaces the map raceline inside the driver only (DP
    deviation cost, MPC reference, Frenet frame, wall tables, horizon end speed); its s = 0 is anchored on the map
    raceline's finish line, where the lap-scoped obstacle memory is cleared.
  * line repair: a raceline tighter than line_ktrig * tan(steer_max) / wheelbase is replaced at start-up by the
    nearest line with |kappa| <= line_kmax, a bounded curvature rate and wall clearance (sparse least squares).
  * dp_kgeo / w_geo: the DP charges slope changes that make the geometric path curvature kr / D + d'' / D^2
    (D = 1 - kr d) exceed the steering limit.
  * wall_clear_v: the wall clearance grows with speed up to wall_clear_vmax (tracking error grows with speed).
  * min_gap_w / min_gap_keep: a box side whose corridor to the wall is narrower than min_gap_w is not offered to the
    DP (such gaps open and close with every few cm of the estimate); the side already chosen stays offered down to
    a gap of min_gap_keep (hysteresis).  Fallback without the rule when no path exists.

Obstacle options (EXTRA; all off by default, i.e. the clear-track driver):
  * obstacle mode (obs_mem < 0: per lap): on in the first lap, in every lap after a lap in which a box was seen and
    from the first box of a lap on; obs_kw = parameter overrides in that mode (reaction time, sight envelope, margins,
    gap rule).  A clear track is driven with the base parameters.
  * width-gated horizon frontier (v_hz, m_hz, w_front, hz_look, front_ramp): what lies beyond the planning horizon is
    unknown; where the free width near the horizon end is below w_front (a box there could close the track), the car
    must be able to reach v_hz at the horizon end; a static ramped cap prepares for it.
  * em_frac: the closed-corridor stop envelope again with em_frac of the emergency deceleration.
  * narrow-side closing (g_narrow, g_diff): a box side narrower than g_narrow while the other side is wider by g_diff
    is closed for the DP (the wide side is taken early).
  * slow DP (v_slow, kcap_slow, kmax_slow, dp_adec_slow, slow_pre, slow_len, m_slow, slow_commit, a_lat_slow): when the
    DP finds no complete path, plan again with the curvature the car steers at v_slow (optionally releasing committed
    sides further away than slow_commit) and slow down to v_slow before the tight part instead of stopping.
  * checked creep and proximity guard (creep_slack, creep_chk, creep_m_obs, creep_m_wall, prox_v, prox_t): below prox_v
    the body points along the planned path, and during a soft drive-on along the steering arc, must keep creep_m_obs /
    creep_m_wall from the measured obstacle cells / map walls, else the car stops (the 3-point MPC corridor does not
    cover the box corners at large heading angles).
  * heading guard (mu_max): stop when the heading error to the line exceeds mu_max (the Frenet linearisation breaks
    down).
  * weave cap (a_weave, v_weave_min, weave_pre): two consecutive boxes passed on opposite sides -> reach
    sqrt(a_weave / (4 dd / l^2)) by the end of the first box (dd: lateral change of the body line, l: free length
    between the boxes as measured, at least one station ds).
  * forward arc search (arc_search, man_after, man_max, man_*): the low-speed recovery of the same driver.  It acts
    only on a car that has stood (|v| < 0.1) for man_after s without a drivable plan (emergency, closed corridor or
    guard veto).  Why it is not done by the MPC: the Frenet MPC plans a path along the reference line at driving speed
    with a 3-point corridor and margins that grow with speed; a car standing next to a box, possibly at a large heading
    angle, needs a short low-speed path whose whole body is checked against the measured obstacle points, which that
    corridor cannot represent.  Search: sequences of up to 3 forward circular arcs of the simulator's low-speed model
    (steering {-1, -0.5, 0, 0.5, 1} STEER_MAX x the man_lens lengths; all 1- and 2-arc sequences, the man_beam best
    2-arc sequences extended by one more arc).  Valid: the body points every 5 cm keep man_m_wall from the map walls
    and man_m_obs from the remembered obstacle cells (a car that starts closer may lose at most man_close).  Score:
    progress along the line (capped at man_goal) - man_w_len length - man_w_mu |heading error| + man_w_clr
    min(margin, 0.1).  Per arc: stop, turn the wheels at standstill, drive at man_vf, re-check the rest of the arc
    every call (abort: stop); the MPC takes over after the last arc, or once its plan has been drivable for man_happy
    s.  Nothing found: wait man_after s and search again.  man_max limits the manoeuvres per place (cleared once the
    car is 1.5 m ahead of or 3 m behind the line position of the first of them, after a lap wrap or a pose jump).
    Every arc drives forward: the driver never commands a negative speed.
  * creep (creep_after, creep_v): the clear-track profiles' recovery (stopped for creep_after s -> creep along the
    plan); the obstacle profile switches it off (creep_after 1e9) and uses the checked drive-on / arc search.

Boxes: a box is the measured extent of its remembered cells (+ half a memory cell, + obs_pad_s / obs_pad_d), at least ds
long (the MPC station spacing: its sampled corridor constraints could step over a shorter box).  No obstacle size is
assumed: what the scan has not seen is not filled in; a path point the scan cannot see (behind or beside an obstacle)
is limited by the sight envelope (v_vis_safe) like any other unseen part of the track, and the box grows as the beams
reach more of the object.

Removed parameters (REMOVED) raise KeyError when passed.
"""
import json
import math
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import yaml
from PIL import Image
from scipy import ndimage
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
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
# body points (CG frame, m) checked against the measured obstacle cells / map walls: the referee box (LiDAR +- 0.165 m
# along, +- 0.1016 m across, i.e. CG + 0.105 .. 0.435) densely, the rest of the car more coarsely
BODY_PTS = ((-0.30, 0.12), (-0.30, -0.12), (-0.30, 0.0), (-0.15, 0.12), (-0.15, -0.12), (0.0, 0.12), (0.0, -0.12),
            (0.105, 0.11), (0.105, -0.11), (0.2, 0.11), (0.2, -0.11), (0.3, 0.11), (0.3, -0.11), (BOX_FRONT, 0.11),
            (BOX_FRONT, -0.11), (BOX_FRONT, 0.0))
FRONT_PTS = tuple(p for p in BODY_PTS if p[0] >= 0.0)


def k_us(a, c=1.0):
    """understeer gradient under longitudinal acceleration a (load transfer); c scales the tyre stiffness"""
    a = np.minimum(np.maximum(a, -A_MAX), A_MAX)
    return (LR_ / (MU_ * CSF_ * (G_ * LR_ - a * HCG_)) - LF_ / (MU_ * CSR_ * (G_ * LF_ + a * HCG_))) / c


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
def _passes(vlim, dpl, tab_plan):
    """backward pass with the speed dependent (steerable) deceleration"""
    N1 = vlim.shape[0]
    v = vlim.copy()
    nt = tab_plan.shape[0]
    for i in range(N1 - 2, -1, -1):
        vn = v[i + 1]
        ab = tab_plan[min(int(vn * 20.0), nt - 1)]
        g1 = math.sqrt(vn * vn + 2.0 * ab * dpl[i])
        ab = tab_plan[min(int(g1 * 20.0), nt - 1)]
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
def _expm_fr_k(M):
    """batched matrix exponential (scaling and squaring + Taylor 10) of the augmented Frenet single-track Jacobian of
    _mpc_hp_k (bitwise the same result as the generic _expm_k of the reference copy): rows 5, 6
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
def _clusters_k(labels, nc, rel, dd):
    """per cluster of _boxes: extent of the cells (rel s, d)"""
    n = labels.shape[0]
    rmin = np.full(nc, np.inf, np.float32)
    rmax = np.full(nc, -np.inf, np.float32)
    dmin = np.full(nc, np.inf, np.float32)
    dmax = np.full(nc, -np.inf, np.float32)
    for i in range(n):
        c = labels[i]
        rmin[c] = min(rmin[c], rel[i])
        rmax[c] = max(rmax[c], rel[i])
        dmin[c] = min(dmin[c], dd[i])
        dmax[c] = max(dmax[c], dd[i])
    return rmin, rmax, dmin, dmax


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
def _boxes_k(active, act_ok, act_s, act_x, act_y, act_d, MW, W, s_car, L, win_hi, link):
    """the cell selection and clustering of CorridorMPC._boxes on the compact remembered set (float32 like numpy):
    cells with map ok, one per 2x2 block above 400 cells (first occurrence), planning window in s, single-linkage
    clusters numbered like np.unique of their roots; then the extent per cluster (_clusters_k)"""
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
        return _clusters_k(np.zeros(0, np.int64), 0, rel, rel)
    xs = np.empty(k)
    ys = np.empty(k)
    dd = np.empty(k, np.float32)
    for i in range(k):
        j = pos[i]
        xs[i] = act_x[j]
        ys[i] = act_y[j]
        dd[i] = act_d[j]
    roots = _cluster_k(xs, ys, link) if k > 1 else np.zeros(1, np.int64)
    # labels as np.unique(roots, return_inverse=True): rank of the root among the distinct roots
    rs = np.unique(roots)
    nc = rs.shape[0]
    labels = np.empty(k, np.int64)
    for i in range(k):
        labels[i] = np.searchsorted(rs, roots[i])
    return _clusters_k(labels, nc, rel, dd)


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
def _speed_k(st, d, mu, beta, kr, v0, scan, lx, ly, yaw, s_block, fx, fy, fnx, fny, fpsi, fvx, L, fds, ds, v_max,
             a_lat, v_end_scale, a_acc, vis_tol, vis_start, m_vis, m_blk, t_react, v_vis_safe, tab_plan, tab_emerg,
             tab_dstop, tab_v):
    """speed profile on the MPC path in one compiled pass: lateral-acceleration limit, backward pass with the
    steerable deceleration, emergency envelope (sight / closed corridor), forward pass.  s_block: nan when there is
    none.  Returns vprof, vpred, acc, dpl, k_vis (-1: all visible), v[0] of the limited profile, cum."""
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
    v = _passes(vlim, dpl, tab_plan)
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
    vprof, vpred = _fwd(v, dpl, v0, a_acc, tab_emerg)
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


# obstacle options (switched on by the obstacle profile), all off by default (see the module docstring)
EXTRA = dict(
    # obstacle mode (obs_kw: parameter overrides while the mode is on, scalars read per call only)
    obs_mem=0.0,            # 0: the mode is always on (obs_kw always applied); < 0: per lap: on in the first lap, in
                            # every lap after a lap with a box, and from the first box of a lap on (a clear track is
                            # driven with the base parameters)
    obs_kw=None,
    # horizon-frontier envelope (obstacle mode): whatever lies beyond the planning horizon is unknown -> be able to
    # reach v_hz at the horizon end (path length cum[N]) minus BOX_FRONT + m_hz with the emergency deceleration,
    # t_lag seconds of command lag included.  v_hz < 0: off
    v_hz=-1.0, m_hz=0.15, t_lag=0.0,
    # w_front > 0: the frontier envelope only where the track (free width from the map along the line normals) is
    # narrower than w_front within [horizon end - 0.5, horizon end + hz_look]; a static speed cap (the envelope's value
    # for a straight 8 m horizon, ramped with front_ramp m/s^2) prepares for it.  Wide sections need no stop for an
    # obstacle appearing at the horizon end: every single obstacle or chicane there can be passed by a swerve
    w_front=0.0, hz_look=2.5, front_ramp=3.0,
    # em_frac < 1: the closed-corridor stop envelope (s_block) again with em_frac of the emergency deceleration (a car
    # that finds itself above it can get back under it)
    em_frac=1.0,
    # DP: a side of a box whose corridor (box .. wall, after margins) is narrower than g_narrow while the other side
    # is wider by g_diff is closed for the DP (the wide side is taken early instead of the narrow one).  <0: off
    g_narrow=-1.0, g_diff=0.25,
    # creep / soft emergency guard: the plan is 'blocked' while it needs more than creep_slack of corridor slack within
    # creep_dist of path (arc-search trigger, checked creep).  <0: off
    creep_slack=-1.0, creep_dist=0.8,
    # checked creep (creep_chk > 0, required with creep_slack >= 0): the creep / soft drive-on of a blocked plan is
    # allowed while the body (front / middle / rear corners) moved creep_chk m along the kinematic arc of the commanded
    # steering keeps creep_m_obs from the remembered obstacle cells (measured surface points, not the padded boxes)
    # and creep_m_wall from the map walls; otherwise the car stops (and the arc search may take over)
    creep_chk=0.0, creep_m_obs=0.04, creep_m_wall=0.03,
    # proximity guard (prox_v > 0): below prox_v m/s the planned path of the next max(creep_chk, prox_t v) m must keep
    # the body creep_m_obs from the measured obstacle cells and creep_m_wall from the walls, else the car stops (the
    # 3-point corridor of the MPC misses the box corners at large heading angles next to an obstacle corner)
    prox_v=0.0, prox_t=0.4,
    # heading guard (mu_max > 0): heading error to the line above mu_max rad in obstacle mode -> stop (the Frenet MPC
    # linearisation breaks down; a full-lock swerve can turn the car around); with arc_search the arc search runs once
    # the car has stood man_after s (its score charges the heading error, man_w_mu)
    mu_max=0.0,
    # weave cap (a_weave > 0): for two consecutive boxes passed on opposite sides, the S between them needs the
    # curvature 4 dd / l^2 (dd: lateral change of the body line, l: free length between the boxes as measured, at
    # least one station ds: no assumed obstacle depth, no added length); reach
    # sqrt(a_weave / kappa) (>= v_weave_min) by the end of the first box (planned from the boxes and DP sides, not
    # from the MPC path: the MPC assumes the braking the car then lags)
    a_weave=0.0, v_weave_min=1.0, weave_pre=0.3,
    # slow DP: when the DP finds no complete path, plan again with the curvature the car can steer at v_slow (DP
    # curvature cap kcap_slow, braking dp_adec_slow from now) before falling back to narrow gaps / a stop; the speed
    # profile then reaches v_slow at s_block_normal - slow_pre and keeps it for slow_len m.  v_slow < 0: off
    v_slow=-1.0, kcap_slow=1.1, kmax_slow=6, dp_adec_slow=5.0, slow_pre=2.0, slow_len=4.0, m_slow=0.1,
    slow_commit=None,       # [commit_d, commit_t] of a second slow-DP try that may release committed sides
    a_lat_slow=0.0,         # > 0: lateral acceleration of the slow DP's curvature limit (truly slow manoeuvres)
    # low-speed manoeuvre (forward arc search, see the module docstring): arc_search switches it on; a search after
    # the car has stood man_after s without a drivable plan.  man_max = manoeuvres (successful searches) per place; a
    # failed search is not counted and is repeated every man_after s while the car stands
    arc_search=False, man_after=1.0, man_max=3,
    man_vf=0.45,                # arc speed (m/s; < 0.5: the simulator's kinematic regime)
    man_lens=(0.35, 0.7, 1.2),  # arc lengths (m)
    man_m_wall=0.04, man_m_obs=0.06,
    man_tol=0.015,              # re-check while driving: clearance may fall this much below the search's floor
    man_goal=2.5,               # progress (m along the line) beyond which more is not rewarded
    man_w_len=0.1, man_w_mu=0.3, man_w_clr=1.0,
    man_beam=40,
    man_min_prog=0.05,          # a sequence must gain at least this much progress
    man_happy=0.2,              # early hand-back once the MPC plan was drivable this long (s) during a manoeuvre, < 0 off
    man_close=0.02,             # a car that starts closer than the margins may lose this much clearance (floor >= this)
)


# parameters that no longer exist (options without effect in any profile, the earlier low-speed rules and reverse
# arcs of the manoeuvre, and assumed obstacle sizes): passing one raises a clear KeyError instead of 'unknown'
REMOVED = ('a_lat_fb', 'gain_comp', 'yaw_lim', 'gg', 'gg_min', 'swerve_w', 'a_acc_launch', 'v_launch', 'escape',
           'vis_obs_stop', 'box_mono', 'occl_ext', 'occl_fix', 'debug', 'jit', 'solver', 'hp_all', 'acc_model',
           'shift_ws', 'osqp_eps', 'osqp_iter', 'osqp_rho_int', 'osqp_rho', 'osqp_tlim', 'vis_copy', 'back_ref_min',
           'back_grow', 'back_v', 'back_k', 'back_tmax', 'back_chk', 'back_m_wall', 'back_m_obs', 'tight_kw',
           'hidden_depth', 'w_max', 'min_width', 'back_dist', 'back_after', 'back_max', 'man_v')


class CorridorMPC:
    def __init__(self, **kw):
        p = dict(
            # speed
            v_max=8.5, a_lat=7.5, a_lat_hard=7.5, a_acc=6.0, a_dec=9.0, c_plan=0.6, c_emerg=0.3,
            acc_corner_min=1.0,
            m_vis=0.15, m_blk=0.30, jerk=60.0, v_min_plan=0.8, vis_tol=0.12, vis_start=0.6,
            t_react=0.04, v_end_scale=1.0,
            # geometry / margins
            m_hard=0.04, m_hard_v=0.006, c_pref=0.12, wall_clear=0.16, wall_clear_v=0.02, wall_clear_vmax=4.0, wall_pref=0.28,
            obs_pad_s=0.06, obs_pad_d=0.02,
            # perception
            obs_edt=0.09, r_max=9.0, link=0.22, blind_after_wrap=0.07, carve=True,
            # DP
            ds=0.2, n=40, dd=0.05, dp_ds=0.5, dp_kmax=6, dp_dmax=1.8, dp_vmax=6.0, dp_adec=4.0, dp_near=0.6, p_near=30.0,
            w_dev=1.0, w_sl=2.0, w_cv=20.0, dp_kcap=0.8, dp_vfloor=2.0,
            w_soft=40.0, w_wall=10.0,
            p_flip=3.0, p_flip_hard=40.0, commit_t=0.45, commit_d=1.0, commit_ramp=1.5, young_t=0.15,
            dp_kgeo=0.95, w_geo=10.0,
            cluster_ds=3.0, release_margin=0.0, hard_block_v=1.5, min_gap_w=0.06, min_gap_keep=0.0, gap_fallback=True,
            # MPC
            q_d=2.0, q_mu=0.05, q_a=0.02, r_w=0.02, rho1=200.0, rho2=2000.0, rhoa1=50.0, rhoa2=50.0,
            slack_brake=0.03, lpts=(0.0, 0.22, 0.435), slack_near=20.0, slack_len=1.5, slack_reach=2.0,
            slack_bad=0.15,
            hp_iter=30, split=True, hp_tol=(1e-4, 1e-5, 1e-5, 1e-5), hp_mode=1, reuse_prof=True, ref_dp=True,
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
            t_call=0.02, em_jerk=400.0, v_vis_safe=2.5, resolve_da=2.0, resolve_n=10,
            # soft emergency at low speed (slack-only emergency with slack <= soft_em becomes a
            # soft_em_v speed cap once vv <= soft_em_arm); lat_est switch rule relative to the cost above the speed-noise
            # floor; adaptive estimators gated on 'the car follows the commands' (fg_win s open-loop check, rollback);
            # pose / yaw outlier gate (pose_out sigmas + pose_out_floor); stall-adaptive lateral acceleration (stall_db)
            soft_em=0.08, soft_em_v=1.0, soft_em_arm=1.5, lat_noise_floor=True,
            fg_check=True, fg_win=0.5, fg_thr=0.3, fg_thr_r=0.35, fg_k=4.0, fg_n=3,
            pose_out=4.0, pose_out_floor=0.03, yaw_out_floor=0.02,
            stall_db=0.1, stall_min=0.85,
            line_ktrig=0.9, line_kmax=1.0, line_dkmax=2.0, line_clr=0.35, line_wdev=0.001,
            map_name='hyhall12',   # any stack_master/maps/<name> with global_waypoints.json
            # reference line shipped with the driver (json: arrays s, x, y[, vx] or a wpnts list / map-style file);
            # None: the map raceline.  Relative paths are resolved against the avoid/ directory.
            raceline_file=None,
            # delay-adaptive a_lat: [t0, t1, a1] (see the module docstring); None: off
            lat_alat=None,
        )
        p.update(EXTRA)
        for k, v in kw.items():
            if k in REMOVED:
                raise KeyError(f'parameter {k} was removed (see REMOVED in drivers/cmpc_berlin_tt.py)')
            if k not in p:
                raise KeyError(f'unknown parameter {k}')
            p[k] = v
        self.p = p
        for k, v in p.items():
            setattr(self, k, v)
        if self.obs_mem > 0.0:
            raise ValueError('obs_mem > 0 (distance-based obstacle mode) was removed: use 0 (always on) or < 0 (per lap)')
        if self.creep_slack >= 0.0 and self.creep_chk <= 0.0:
            raise ValueError('creep_slack >= 0 needs creep_chk > 0 (the unchecked creep-forbidden mode was removed)')
        self.lpts = tuple(self.lpts)
        self.wc_eff = self.wall_clear
        self.mapdir = os.path.join(os.path.dirname(MAPDIR), self.map_name)
        self._load_occ()
        self._load_track()
        self._load_map()
        self.hp = self._build_hpipm()        # HPIPM of the CasADi install
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
            _boxes_k(np.array([0, 1]), ok_, z32, z32, z32, z32, self.MW, self.W, 0.0, float(self.L), 9.0, 0.22)
        _passes(np.ones(3), np.ones(2), self.tab_plan)
        _fwd(np.ones(3), np.ones(2), 1.0, 1.0, self.tab_plan)
        _cluster_k(np.zeros(3), np.zeros(3), 0.1)
        _cmd_sim(np.ones(3), np.arange(3.0), np.ones(2), 1.0, 0.0, 1.0, 1.0, self.tab_emerg, 0.02)
        _lat_sim_k(np.zeros(2), np.ones(2), np.array([0.0, 0.03]), np.ones(2), 0.0, 2)
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
        st0 = np.arange(self.n + 1) * self.ds
        z1, zn = np.zeros(self.n + 1), np.zeros(self.n)
        hp = self.hp
        _mpc_hp_k(st0, np.zeros(5), np.zeros((self.n + 1, 5)), z1, z1 + 1.0, zn, zn + self.ds, z1, self.lpts_a,
                  self.wlo_c, self.whi_c, self.nf, self.fds, np.zeros((1, 4)), np.ones(1, dtype=np.int64), 0.1,
                  0.8, 0.1, 1.0, 1.0, 1.0, 1.0, self.tab_plan, hp.bA, hp.bB, hp.bb, hp.bQ, hp.bq, hp.bC, hp.blg,
                  hp.bug, hp.blbx, hp.bubx, np.zeros((self.n, 3)), np.zeros((self.n, 3)), np.zeros((self.n, 3)),
                  np.zeros(self.n), 1.0)
        _hp_slack_k(np.zeros((self.n + 1, 5)), np.zeros((self.n, 3)), np.zeros((self.n, 3)), np.zeros((self.n, 3)))
        stw = np.arange(self.n + 1) * self.ds
        _plan_prep_k(stw, self.fk, float(self.L), float(self.fds), stw, np.zeros((self.n + 1, 5)), stw * 0.0,
                     np.zeros(5), stw, stw[:-1] * 0.0 + self.ds, stw * 0.0 + 1.0, 1.0, 0.0, 6.0, 60.0,
                     self.tab_emerg, 0.02)
        z1 = np.zeros(self.n + 1)
        for sblk in (math.nan, 1.0):
            _speed_k(np.arange(self.n + 1) * self.ds, z1, z1, z1, z1, 1.0, np.ones(1080), 0.0, 0.0, 0.0, sblk,
                     self.fx, self.fy, self.fnx, self.fny, self.fpsi, self.fvx, float(self.L), float(self.fds), 0.2,
                     8.0, 7.0, 1.0, 6.0, 0.1, 0.5, 0.1, 0.1, 0.04, 2.5, self.tab_plan, self.tab_emerg, self.tab_dstop,
                     self.tab_v)
        self.reset()
        # obstacle mode: obs_kw overrides (validated; EXTRA options cannot be switched), frontier tables
        self.obs_kw = dict(self.obs_kw or {})
        for k in self.obs_kw:
            if k not in self.p or k in EXTRA:
                raise KeyError(f'obs_kw: unknown or non-switchable parameter {k}')
        self.base_kw = {k: self.p[k] for k in self.obs_kw}
        self.obs_on = None
        self._set_mode(True)
        self._front_tables()
        # arc search: body points and the forward primitives (steering, length)
        bp = np.array(BODY_PTS, dtype=float)
        self._bpx, self._bpy = bp[:, 0], bp[:, 1]
        st = np.array([-1.0, -0.5, 0.0, 0.5, 1.0]) * STEER_MAX
        self._prims = np.array([(dl, ln) for dl in st for ln in self.man_lens])
        self.man_stats = []         # one entry per search (time, candidates, best sequence)

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
            # the raceline of an imported map can be tighter than the car can steer at all (kappa far above
            # tan(delta_max) / WB = 1.35): every Frenet quantity (DP curvature, MPC linearisation, wall tables)
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
        nf = int(round(L / 0.05))      # 5 cm stations
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
        i = int(_frenet_k(self.fx, self.fy, float(x), float(y), -1 if guess is None else int(guess)))
        if guess is not None and (i == (guess - 12) % self.nf or i == (guess + 39) % self.nf):
            # nearest point on the edge of the window (-0.6 .. +2 m): the car is beyond it (stall of the node at
            # speed, pose jump); the edge point is a wrong local minimum (d off by its curvature, and it can stay
            # locked there) -> global search
            return self._frenet(x, y, None)
        dx, dy = x - self.fx[i], y - self.fy[i]
        return self.fs[i] + dx * self.ftx[i] + dy * self.fty[i], dx * self.fnx[i] + dy * self.fny[i], i

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
        self._ps_cache = [None, None, None]   # (list, bucket index, memo) of prev_sides / seen_boxes for _first_match
        self._sb_cache = [None, None, None]
        self.box_age = []
        self.qp_info = None
        self.prof = None
        self.t_stop = None
        self.dp_res = None
        self.plan = None
        self.phase = 0
        self.sol_bad = False
        self.n_fail = 0
        self.n_resolve = 0
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
        self.odo = 0.0              # metres driven (|v| dt) since the start
        self.lap_boxes = 0          # DP calls with boxes in this lap
        self.lap_clear = False
        if getattr(self, 'base_kw', None) is not None and self.obs_mem != 0.0:
            self._set_mode(True)    # after a reset (re-placed car): obstacle mode until a clear lap
        self.t_odo = None
        self.tr_extra = None
        self.slow_at = None
        self.weaves = []
        self.man = None             # running manoeuvre (arc sequence, see _manoeuvre)
        self.tr_mode = None         # diagnostics: ('settle' | 'drive', arc index) while a manoeuvre commands
        self.n_man = 0
        self.man_s = 0.0
        self._creep_blocked = False
        if not hasattr(self, 'n_man_tot'):
            self.n_man_tot = 0

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
        H = self._H.T.copy()
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
        # estimate never changed.  R = 0: switch on the whole cost.
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
            H = self._H.T.copy()
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
        delta, vm, dmain, amain, vms, vck, hvk = _obs_main_k(self._H, float(self.t_lat), float(self.t_buf), float(t0),
                                                             int(n), float(vf0), float(self.delta_est))
        if short and n > 0:
            _mm_prop(bb, rr, self.cg_d, cv, self.l_grid_v, self.c_grid_v.size, self._H, float(self.t_lat), float(self.t_buf),
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
        beta, has_mid, bm, px_, py_ = _obs_beta_k(int(n), float(vf0), float(vf), float(rf0), float(rf), float(yawf0),
                                                  dyw, float(xf0), float(yf0), float(self.beta_est), dmain, vck, hvk,
                                                  float(self.c_tyre))
        beta_mid = bm if has_mid else None          # sideslip at the middle of the interval (the chord's course)
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
        return _predict_k(self._H, float(self.t_lat), float(self.t_buf), float(t), float(x), float(y), float(yaw),
                          float(v), float(r), float(t_pred), float(self.delta_est), float(self.beta_est),
                          float(self.c_tyre))

    # ------------------------------------------------------------------ perception
    def _mem_clear(self):
        """forget all remembered obstacle cells (lap wrap, reset)"""
        self.active = np.zeros(0, dtype=np.int64)
        self.act_mem = np.zeros(0, dtype=np.int16)
        self.act_x, self.act_y, self.act_s, self.act_d = (np.zeros(0, dtype=np.float32) for _ in range(4))
        self.act_ok = np.zeros(0, dtype=np.bool_)

    def _view_of(self, lx, ly, yaw):
        """range, bearing and beam index of every remembered cell from the LiDAR (float32 numpy math, as before)"""
        dx = self.act_x - lx
        dy = self.act_y - ly
        rng = np.hypot(dx, dy)
        brg = _wrap(np.arctan2(dy, dx) - yaw)
        return rng, brg, np.rint((brg - ANG_MIN) / ANG_INC).astype(np.int64)

    def _perceive(self, scan, lx, ly, yaw, t):
        """scan hits farther than obs_edt from the map -> remembered obstacle cells; free-space carving.  The per-cell
        map attributes (x, y, s, d, ok) and memory counts are kept in arrays aligned with the sorted cell ids, so the
        per-call work touches only those"""
        cy, sy = math.cos(yaw), math.sin(yaw)
        if t >= self.blind_until:
            cells = _hits_k(scan, float(lx), float(ly), cy, sy, self.ca, self.sa, float(self.r_max), self.ox, self.oy,
                            self.res, self.W, self.H, self.edt, float(self.obs_edt), self.mres, self.MW)
            if cells.size:
                (self.active, self.act_mem, self.act_x, self.act_y, self.act_s, self.act_d,
                 self.act_ok) = _merge_k(self.active, self.act_mem, self.act_x, self.act_y, self.act_s, self.act_d,
                                         self.act_ok, cells, self.cell_x, self.cell_y, self.cell_s, self.cell_d,
                                         self.cell_ok)
        if self.active.size == 0 or not self.carve:
            return
        view = self._view_of(lx, ly, yaw)
        keep = _carve_k(self.act_mem, view[0], view[2], scan)
        if not keep.all():
            for nm in ('active', 'act_mem', 'act_x', 'act_y', 'act_s', 'act_d', 'act_ok'):
                setattr(self, nm, getattr(self, nm)[keep])

    def _boxes(self, s_car, scan, lx, ly, yaw):
        """cluster remembered cells near the planning window into Frenet boxes [s_lo, s_hi, d_lo, d_hi], sorted by s_lo:
        the measured extent of the cells, at least ds long, padded by obs_pad_s / obs_pad_d.  No obstacle size is
        assumed: unseen parts are left to the sight envelope (see the module docstring)"""
        if self.active.size == 0:
            return []
        rmin, rmax, dmin, dmax = _boxes_k(
            self.active, self.act_ok, self.act_s, self.act_x, self.act_y, self.act_d, self.MW, self.W, float(s_car),
            float(self.L), float(self.n * self.ds + 1.0), float(self.link))
        out = []
        half = self.mres / 2
        s32 = np.float32(s_car)           # ss = rel + s_car is float32 (numpy casting); min/max commute with it
        for c in range(rmin.shape[0]):
            s_lo, s_hi = float(rmin[c] + s32) - half, float(rmax[c] + s32) + half
            d_lo, d_hi = float(dmin[c]) - half, float(dmax[c]) + half
            s_hi = max(s_hi, s_lo + self.ds)
            out.append([s_lo - self.obs_pad_s, s_hi + self.obs_pad_s, d_lo - self.obs_pad_d, d_hi + self.obs_pad_d])
        out.sort(key=lambda b: b[0])
        return out

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
        e = self._first_match(self.prev_sides, self._ps_cache, smid, dc, 0.35, 0.3)
        return None if e is None else e[2]

    def _first_seen(self, smid, dc, t):
        e = self._first_match(self.seen_boxes, self._sb_cache, smid, dc, 0.4, 0.35)
        return t if e is None else e[2]

    # ------------------------------------------------------------------ DP side selection
    def _dp(self, s0, d0, chi0, boxes, v, w_h, relax=(), hard=True, gap=True):
        """2nd-order DP over (station, lateral offset, slope): picks the side of every obstacle with a
        curvature-limited path; returns MPC stations, DP path on them, sides, first blocked s.  The numerics run in
        _dp_k; the side commitments (previous sides, young neighbours) are set here"""
        if self.g_narrow >= 0.0 and boxes and self.obs_on and gap:
            # (the narrow-side closing belongs to the gap rule: the fallbacks without it may use a narrow side)
            boxes = self._close_narrow(s0, boxes, v, w_h)
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

    # ------------------------------------------------------------------ speed profile
    def _abrake(self, tab, v):
        return tab[min(int(max(v, 0.0) * 20.0), tab.size - 1)]

    def _speed_profile(self, st, d, mu, beta, kr, v0, scan, lx, ly, yaw, s_block):
        """returns (vprof target speeds, vpred predicted speeds, accel per station, dpl, s_vis); compiled in _speed_k,
        then (obstacle mode) the frontier, em_frac, weave and slow-DP envelopes"""
        f64 = lambda a: np.ascontiguousarray(a, dtype=np.float64)
        vprof, vpred, acc, dpl, kv, v0_lim, self._sp_cum = _speed_k(
            f64(st), f64(d), f64(mu), f64(beta), f64(kr), float(v0), f64(scan), float(lx), float(ly), float(yaw),
            math.nan if s_block is None else float(s_block), self.fx, self.fy, self.fnx, self.fny, self.fpsi, self.fvx,
            float(self.L), float(self.fds), float(self.ds), float(self.v_max), float(self.a_lat), float(self.v_end_scale),
            float(self.a_acc), float(self.vis_tol), float(self.vis_start), float(self.m_vis), float(self.m_blk),
            float(self.t_react), float(self.v_vis_safe), self.tab_plan, self.tab_emerg, self.tab_dstop, self.tab_v)
        self.vlim0 = float(v0_lim)
        s_vis = float(st[kv]) if kv >= 0 else None
        if not self.obs_on:
            return vprof, vpred, acc, dpl, s_vis
        cum = self._sp_cum
        envs = []                   # the speed envelopes that apply now
        if self.v_hz >= 0.0:
            gated = False
            if self.front_flag is not None:
                ii = (np.floor(np.mod(st, self.L) / self.fds).astype(np.int64)) % self.nf
                envs.append(self.front_cap[ii])
                gated = not bool(self.front_flag[ii[0]])
            if not gated:
                envs.append(self._env(cum, cum.size - 1, v0, self.v_hz, self.m_hz))
        if self.em_frac < 1.0 and s_block is not None:
            # the closed-corridor envelope again (lag t_react + t_lag, reduced deceleration)
            kb = min(int(np.searchsorted(st, s_block)), cum.size - 1)
            envs.append(self._env(cum, kb, v0, 0.0, self.m_blk, self.em_frac))
        for s_w, s_w2, v_w in self.weaves:
            k = min(int(np.searchsorted(st, s_w)), cum.size - 1)
            k2 = int(np.searchsorted(st, s_w2))
            if k2 > 0:
                e3 = self._env(cum, k, v0, v_w, -BOX_FRONT)
                e3[k2 + 1:] = np.inf
                envs.append(e3)
        if self.v_slow > 0.0 and self.slow_at is not None:
            s_t = self.slow_at - self.slow_pre
            k = min(int(np.searchsorted(st, s_t)), cum.size - 1)
            e2 = self._env(cum, k, v0, self.v_slow, self.m_slow - BOX_FRONT)
            e2[(st > s_t + self.slow_len)] = np.inf
            envs.append(e2)
        env = np.min(envs, axis=0) if envs else None
        if env is not None and np.any(env < vprof):
            vl = np.minimum(vprof, env)
            vprof, vpred = _fwd(np.ascontiguousarray(vl), np.ascontiguousarray(dpl, dtype=np.float64), float(v0),
                                float(self.a_acc), self.tab_emerg)
            acc = np.clip((vpred[1:] ** 2 - vpred[:-1] ** 2) / (2 * dpl), -A_MAX, A_MAX)
            self.vlim0 = min(self.vlim0, float(vl[0]))
        return vprof, vpred, acc, dpl, s_vis

    def _heading(self, s):
        """raceline heading at the station s (scalar; continuous interpolation)"""
        u = (s % self.L) / self.fds
        fu = math.floor(u)
        i0 = int(fu) % self.nf
        i1 = (i0 + 1) % self.nf
        f = u - fu
        return float(self.fpsi[i0] + _wrap(float(self.fpsi[i1] - self.fpsi[i0])) * f)

    # ------------------------------------------------------------------ MPC
    def _mpc(self, st, X0, Xbar, kr, vpred, acc, dpl, boxes, sides, w_h, dpath):
        """QP data written by _mpc_hp_k straight into HPIPM's buffers; returns (Xs, steering rate, slack, dt) or None"""
        N, nl = self.n, len(self.lpts)
        hp = self.hp
        bx = np.array(boxes, dtype=np.float64).reshape(-1, 4)
        bsd = np.array(sides, dtype=np.int64).reshape(-1)
        base = dpath if self.ref_dp else np.zeros(N + 1)
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
        if self.t_odo is not None and t > self.t_odo:
            self.odo += abs(v) * min(t - self.t_odo, 0.2)
        self.t_odo = t
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
        # perception is skipped on a teleport call
        if teleport:
            self.fi = None
            self.s_prev = None
            self.last = None
            self.sol_bad = True
            self.prof = None
            self.dp_res = None
            if jump_d > 2.0:
                # relocated (not a glitch): what was remembered and the side commitments belong to the previous
                # place / pose (the lap-wrap test, which clears them otherwise, is skipped on a teleport call)
                self._mem_clear()
                self.prev_sides = []
                self.seen_boxes = []
        # command history: keep what can still act (largest latency hypothesis) plus the command acting before it
        k_old = 0
        while k_old + 1 < len(self.hist) and self.hist[k_old + 1][0] < t - 0.3 - self.lat_max:
            k_old += 1
        self.hist = self.hist[k_old:]
        self._H = np.array(self.hist, dtype=np.float64).reshape(len(self.hist), 3)
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
            self._mem_clear()
            self.blind_until = t + self.blind_after_wrap
            self.prev_sides = []
            self.seen_boxes = []
            if self.obs_mem < 0.0:
                self.lap_clear = self.lap_boxes == 0 and self.odo > 0.5 * self.L
                self.lap_boxes = 0
                self._set_mode(not self.lap_clear)
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
            if boxes:
                self.lap_boxes += 1
                if self.obs_mem != 0.0:
                    self._set_mode(True)
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
            self.slow_at = None
            if s_block is not None and self.v_slow > 0.0 and self.obs_on:
                sb0 = s_block
                for rel_, gap_ in ((False, True), (True, True), (False, False), (True, False)):
                    if rel_ and self.slow_commit is None:
                        continue
                    r_ = self._dp_slow(s0, d0, mu0 + pbeta, boxes, vv, w_h, relax, gap_, rel_)
                    if r_[3] is None:
                        st, dpath, sides, s_block = r_
                        self.slow_at = sb0
                        break
            if s_block is not None and self.gap_fallback:
                # no complete path: allow the narrow gaps
                r_ = self._dp(s0, d0, mu0 + pbeta, boxes, vv, w_h, relax=relax, hard=True, gap=False)
                if r_[3] is None:
                    st, dpath, sides, s_block = r_
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
            self.tr_extra = (int(bool(self.obs_on)), None if self.slow_at is None else round(self.slow_at, 2))
            self.weaves = self._weaves(boxes, sides, w_h, s0) if (self.a_weave > 0.0 and self.obs_on) else []
        else:
            boxes, sides, s_block, st_old, dpath_old = self.dp_res
            st = s0 + self.ds * np.arange(N + 1)
            dpath = np.interp(st, st_old, dpath_old)
        if do_mpc:
            new = self._plan(t, st, X0, dpath, boxes, sides, s_block, w_h, pv, scan, lx, ly, yaw, vv)
            old = self.plan
            if (new['failed'] and old is not None and not old['bad'] and self.n_fail <= 2 and not full
                    and t - old['t'] < 0.12):
                # a failed solve (HPIPM min-step / NaN status): the previous plan is 1-2 calls old, exactly what
                # every DP-only call of the split schedule drives on -> keep it (emergency only after the third
                # failure in a row)
                pass
            else:
                self.plan = new
        plan = self.plan
        # ---- commands from the latest plan (time-shifted when it was computed in an earlier call)
        Xs, dts, vprof, cum, emergency, slack = (plan['Xs'], plan['dts'], plan['vprof'], plan['cum'],
                                                 plan['emergency'], plan['slack'])
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
        self._creep_blocked = bool(plan['bad'])
        if self.creep_slack >= 0.0 and slack is not None:
            near_ = cum <= self.creep_dist
            if np.any(near_) and float(np.max(slack[near_])) > self.creep_slack:
                self._creep_blocked = True
        if creep:
            emergency = False
        if emergency:
            a_des = -a_em
        else:
            a_des = (v_tgt * v_tgt - pv * pv) / (2 * s_ahead)
            # accelerating in a corner grows the sideslip and the yaw rate: share the grip
            if a_des > 0:
                load = min(1.0, abs(vv * yaw_rate) / self.a_lat)
                a_des = min(a_des, self.a_acc * max(self.acc_corner_min, 1.0 - load * load))
            if creep:
                a_des = (self.creep_v - vv) * 4.0
            a_des = min(max(a_des, -a_em), self.a_acc)
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
        # ---- steering: planned angle one control period after the bite time (on a bad plan too: the bad flag brakes,
        # see _plan_rest; the command is held until the next call: target the planned angle one measured period ahead)
        tcum = np.concatenate([[0.0], np.cumsum(dts)])
        steer = float(np.interp(self.t_period + shift_t, tcum, Xs[:, 4]))
        steer = min(max(steer, -STEER_MAX), STEER_MAX)
        veto = self.mu_max > 0.0 and self.obs_on and speed > 0.0 and abs(mu0) > self.mu_max
        if not veto and self.prox_v > 0.0 and self.obs_on and speed > 0.0 and vv < self.prox_v and not plan['bad']:
            veto = not self._plan_clear(plan, max(self.creep_chk, self.prox_t * vv), shift_s)
        if (self.creep_chk > 0.0 and self.creep_slack >= 0.0 and self.obs_on and self._creep_blocked and speed > 0.0
                and vv < 1.5 and not veto):
            # checked creep / soft drive-on: only while the measured surfaces stay clear of the body
            veto = not self._fwd_clear(x, y, yaw, steer, max(self.creep_chk, 0.3 * vv))
        if veto:
            self._creep_blocked = True        # no drivable plan (the arc search may take over)
            speed = 0.0
            self.a_cmd = min(self.a_cmd, 0.0)
            if self.t_stop is None and vv < 0.15:
                self.t_stop = t
        if self.arc_search:
            if self.n_man and (wrapped or teleport or s_c - self.man_s > 1.5 or s_c < self.man_s - 3.0):
                self.n_man = 0
            rc = self._manoeuvre(t, x, y, yaw, v, mu0, s_c)
            if rc is not None:
                steer, speed = rc
                self.a_cmd = 0.0
        self.hist.append((t, steer, speed))
        return steer, speed

    def _set_wall(self, wc):
        """wall clearance of this call's DP / corridor (and the compiled wall rows of the same value)"""
        self.wc_eff = wc
        kc = int(round(min(max(wc, 0.0), 0.6) / 0.01))      # the row _wall_bounds uses
        self.wlo_c = np.ascontiguousarray(self.wlo_tab[kc])
        self.whi_c = np.ascontiguousarray(self.whi_tab[kc])

    def _plan(self, t, st, X0, dpath, boxes, sides, s_block, w_h, pv, scan, lx, ly, yaw, vv):
        """MPC + speed profile on the current stations; returns the plan used for the commands"""
        N = self.n
        if self.sol is not None and not self.sol_bad and self.reuse_prof and self.prof is not None:
            sp_, vpp, dplp, vpf = self.prof
            kr, Xbar, dpl, vpred, acc = _plan_prep_k(st, self.fk, float(self.L), float(self.fds), self.sol[0],
                                                     self.sol[1], dpath, X0, sp_, dplp, vpf, float(pv),
                                                     float(self.a_cmd), float(self.a_acc), float(self.jerk),
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
        # the speed / acceleration the command logic will really produce (jerk limited, from the current command)
        cum0 = np.concatenate([[0.0], np.cumsum(dpl)])
        vpred, acc = _cmd_sim(vtarget.astype(np.float64), cum0, dpl.astype(np.float64), float(pv), float(self.a_cmd),
                              float(self.a_acc), float(self.jerk), self.tab_emerg, float(self.t_period))
        return self._plan_rest(t, st, X0, dpath, boxes, sides, s_block, w_h, pv, scan, lx, ly, yaw, vv, kr, Xbar, dpl,
                               vpred, acc)

    def _plan_rest(self, t, st, X0, dpath, boxes, sides, s_block, w_h, pv, scan, lx, ly, yaw, vv, kr, Xbar, dpl, vpred,
                   acc):
        N = self.n
        acc_mpc = acc.copy()
        out = self._mpc(st, X0, Xbar, kr, vpred, acc, dpl, boxes, sides, w_h, dpath)
        slack = None
        if out is not None:
            Xs, w, slack, dts = out
            self.sol = (st, Xs, w)
            self.n_fail = 0
        else:
            self.n_fail += 1
            Xs, dts = Xbar, dpl / np.maximum(vpred[:N], self.v_min_plan)
        vprof, vpred, acc, dpl, s_vis = self._speed_profile(st, Xs[:, 0], Xs[:, 1], Xs[:, 2], kr, pv,
                                                            scan, lx, ly, yaw, s_block)
        if self.resolve_da > 0.0 and out is not None:
            # the new profile changes the acceleration the car will have near term (e.g. a new swerve starts the
            # braking now): the load transfer then changes the yaw response a lot -> solve once more with it
            v2, acc2 = _cmd_sim(vprof.astype(np.float64), self._sp_cum, dpl.astype(np.float64), float(pv),
                                float(self.a_cmd), float(self.a_acc), float(self.jerk), self.tab_emerg,
                                float(self.t_period))
            kn = self.resolve_n
            if float(np.max(np.abs(acc2[:kn] - acc_mpc[:kn]))) > self.resolve_da:
                out2 = self._mpc(st, X0, Xs, kr, v2, acc2, dpl, boxes, sides, w_h, dpath)
                if out2 is not None:
                    out = out2
                    Xs, w, slack, dts = out2
                    self.sol = (st, Xs, w)
                    acc_mpc = acc2.copy()
                    vprof, vpred, acc, dpl, s_vis = self._speed_profile(st, Xs[:, 0], Xs[:, 1], Xs[:, 2], kr, pv,
                                                                        scan, lx, ly, yaw, s_block)
                    self.n_resolve += 1
        self.prof = (st, vpred, dpl, vprof)
        cum = self._sp_cum
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
                    bad=bad, acc=acc_mpc, failed=out is None)

    def _dstop(self, v):
        if v <= 0:
            return 0.0
        return float(np.interp(v, self.tab_v, self.tab_dstop)) + v * self.t_react

    # ------------------------------------------------------------------ obstacle options (EXTRA)
    def _front_tables(self):
        """static part of the width-gated frontier envelope (see w_front)"""
        self.front_flag = None
        if self.w_front <= 0.0 or self.v_hz < 0.0:
            return
        W = self.whi_tab[0] - self.wlo_tab[0] + 0.06
        nf, fds = self.nf, self.fds
        H = self.n * self.ds
        nar = W < self.w_front
        k0, k1 = int(round((H - 0.5) / fds)), int(round((H + self.hz_look) / fds))
        idx = (np.arange(nf)[:, None] + np.arange(k0, k1 + 1)[None, :]) % nf
        self.front_flag = nar[idx].any(1)
        # the envelope's value with a straight horizon of length H (obstacle-mode t_react)
        tr = self.obs_kw.get('t_react', self.t_react)
        vs = self.tab_v
        reach = H - BOX_FRONT - self.m_hz - vs * (tr + self.t_lag) + float(np.interp(self.v_hz, vs, self.tab_dstop))
        ok = self.tab_dstop <= reach
        v_fc = float(vs[ok][-1]) if ok.any() else self.v_hz
        cap = np.where(self.front_flag, v_fc, np.inf)
        for _ in range(2):
            for i in range(nf - 1, -1, -1):
                cap[i] = min(cap[i], math.sqrt(cap[(i + 1) % nf] ** 2 + 2.0 * self.front_ramp * fds))
        self.front_cap = cap
        self.v_fc = v_fc

    def _set_mode(self, on):
        if on == self.obs_on:
            return
        self.obs_on = on
        for k, v in (self.obs_kw if on else self.base_kw).items():
            setattr(self, k, v)
            self.p[k] = v

    def _env(self, cum, k, v0, vt, m, frac=1.0):
        """speed envelope: reach vt at path length cum[k] - BOX_FRONT - m with the emergency deceleration (command lag
        included), vt beyond"""
        reach = cum[k] - BOX_FRONT - m - cum - v0 * (self.t_react + self.t_lag)
        tds = self.tab_dstop / frac
        ds_ = float(np.interp(vt, self.tab_v, tds))
        env = np.interp(np.maximum(reach, 0.0) + ds_, tds, self.tab_v)
        return np.where(reach <= 0.0, np.minimum(env, vt), env)

    def _weaves(self, boxes, sides, w_h, s0):
        """(s, v): speed caps at the end of the first box of every opposite-side pair (see a_weave)"""
        out = []
        bs = sorted((b, sd) for b, sd in zip(boxes, sides) if sd != 0 and b[1] > s0)
        for (b1, s1), (b2, s2) in zip(bs[:-1], bs[1:]):
            if s1 == s2 or b2[0] - b1[1] > 4.0:
                continue
            # body-line bound past box 1 and the one needed at box 2 (s1 = +1: d >= b[3] + w_h, -1: d <= b[2] - w_h)
            d1 = b1[3] + w_h if s1 > 0 else b1[2] - w_h
            d2 = b2[3] + w_h if s2 > 0 else b2[2] - w_h
            dd = (d2 - d1) * s2
            if dd <= 0.0:
                continue
            # l: the measured free length b2[0] - b1[1] between the boxes (measured extents + obs_pad_s, no obstacle
            # size).  The MPC corridor holds the CG point to box 1's bound up to b1[1] and from b2[0] on (its other
            # points lie ahead on the tangent, which is on the safe side), so the planned S fits exactly into this
            # length.  Floor: one MPC station ds (the S cannot be resolved shorter)
            ln = max(b2[0] - b1[1], self.ds)
            kap = 4.0 * dd / (ln * ln)
            v_w = max(math.sqrt(self.a_weave / kap), self.v_weave_min)
            out.append((b1[1] - self.weave_pre, b2[0], v_w))
        return out

    def _dp_slow(self, s0, d0, chi0, boxes, v, w_h, relax, gap, release=False):
        sv = (self.dp_vfloor, self.dp_vmax, self.dp_kcap, self.dp_adec, self.dp_kmax, self.commit_d, self.commit_t)
        self.dp_vfloor, self.dp_vmax, self.dp_kcap, self.dp_adec = self.v_slow, 99.0, self.kcap_slow, self.dp_adec_slow
        self.dp_kmax = self.kmax_slow
        sv_alh = self.a_lat_hard
        if self.a_lat_slow > 0.0:
            self.a_lat_hard = min(self.a_lat_hard, self.a_lat_slow)
        if release:
            # shorter hard-commit distance: a committed side may be released further out than commit_d / commit_t v
            self.commit_d, self.commit_t = self.slow_commit
        try:
            return self._dp(s0, d0, chi0, boxes, v, w_h, relax=relax, hard=True, gap=gap)
        finally:
            (self.dp_vfloor, self.dp_vmax, self.dp_kcap, self.dp_adec, self.dp_kmax, self.commit_d,
             self.commit_t) = sv
            self.a_lat_hard = sv_alh

    def _close_narrow(self, s0, boxes, v, w_h):
        """boxes for the DP with the narrow side of a box closed when the other side is clearly wider (g_narrow,
        g_diff; DP only: the corridor of the chosen side is built from the real box)"""
        dnear = max(self.commit_d, self.commit_t * v)
        bx2 = []
        for b in boxes:
            b2 = list(b)
            ss = np.linspace(b[0], b[1], 5)
            wlo, whi = self._wall_bounds(ss, self.wc_eff)
            g_hi = float(np.min(whi)) - (b[3] + w_h)
            g_lo = (b[2] - w_h) - float(np.max(wlo))
            prev = self._prev_side(0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]))
            dist = b[0] - (s0 + BOX_FRONT)
            near = dist < dnear and v > self.hard_block_v
            if g_hi < self.g_narrow and g_lo > g_hi + self.g_diff and not (near and prev == 1):
                b2[3] = float(np.max(whi)) + 0.5
            elif g_lo < self.g_narrow and g_hi > g_lo + self.g_diff and not (near and prev == -1):
                b2[2] = float(np.min(wlo)) - 0.5
            bx2.append(b2)
        return bx2

    def _body_clr(self, x, y, yaw, dl, dist):
        """smallest clearance of the body points (rear / middle / front corners) to the map walls and to the
        remembered obstacle cells after driving dist m forward with steering dl (kinematic arc; dist 0: current pose)"""
        yw = yaw + dist * math.tan(dl) / WB
        ym = 0.5 * (yaw + yw)
        cx, cy = x + dist * math.cos(ym), y + dist * math.sin(ym)
        c, s_ = math.cos(yw), math.sin(yw)
        P = np.array([(cx + lx_ * c - ly_ * s_, cy + lx_ * s_ + ly_ * c)
                      for lx_, ly_ in BODY_PTS])
        cw = float(np.min(self._clearance(P[:, 0], P[:, 1])))
        co = math.inf
        if self.active.size:
            dx = self.act_x[:, None] - P[None, :, 0]
            dy = self.act_y[:, None] - P[None, :, 1]
            co = math.sqrt(float(np.min(dx * dx + dy * dy)))
        return cw, co

    def _plan_clear(self, plan, dist, shift):
        """body points along the planned path (d, mu at the stations within dist m of path ahead) keep creep_m_obs /
        creep_m_wall from the measured obstacle cells / walls"""
        st, Xs, cum = plan['st'], plan['Xs'], plan['cum']
        c0 = float(np.interp(shift, st - st[0], cum)) if shift > 0 else 0.0
        k = np.flatnonzero((cum >= c0) & (cum <= c0 + dist))
        if k.size == 0:
            return True
        xs, ys = self._xy(st[k], Xs[k, 0])
        hd = np.array([self._heading(float(q)) for q in st[k]]) + Xs[k, 1]
        c, s_ = np.cos(hd), np.sin(hd)
        pts = []
        for lx_, ly_ in FRONT_PTS:
            pts.append(np.stack([xs + lx_ * c - ly_ * s_, ys + lx_ * s_ + ly_ * c], 1))
        P = np.concatenate(pts)
        if float(np.min(self._clearance(P[:, 0], P[:, 1]))) < self.creep_m_wall:
            return False
        if self.active.size:
            dx = self.act_x[:, None] - P[None, :, 0]
            dy = self.act_y[:, None] - P[None, :, 1]
            if float(np.min(dx * dx + dy * dy)) < self.creep_m_obs ** 2:
                return False
        return True

    def _fwd_clear(self, x, y, yaw, dl, dist):
        """the body stays creep_m_obs / creep_m_wall clear when driving dist m forward with steering dl"""
        for i in (1, 2, 3):
            cw, co = self._body_clr(x, y, yaw, dl, dist * i / 3)
            if cw < self.creep_m_wall or co < self.creep_m_obs:
                return False
        return True

    # ---- low-speed manoeuvre (arc search)
    @staticmethod
    def _arc(x, y, yaw, dl, u):
        """pose after driving the arc length u forward with steering dl (exact circle; broadcasting)"""
        th = u * np.tan(dl) / WB
        c = u * np.sinc(0.5 * th / np.pi)
        return x + c * np.cos(yaw + 0.5 * th), y + c * np.sin(yaw + 0.5 * th), yaw + th

    def _obs_field(self, x, y):
        """distance field of the remembered obstacle cells, 7 m square around the car, aligned with the memory grid"""
        res, half = self.mres, 3.5
        n = int(2 * half / res)
        gx = self.ox + math.floor((x - half - self.ox) / res) * res
        gy = self.oy + math.floor((y - half - self.oy) / res) * res
        occ = np.zeros((n, n), dtype=bool)
        if self.active.size:
            ci = np.floor((self.act_x - gx) / res).astype(np.int64)
            ri = np.floor((self.act_y - gy) / res).astype(np.int64)
            ok = (ci >= 0) & (ci < n) & (ri >= 0) & (ri < n)
            occ[ri[ok], ci[ok]] = True
        if not occ.any():
            return None
        return (ndimage.distance_transform_edt(~occ) * res).astype(np.float32), gx, gy, res

    def _clr(self, X, Y, YAW, fld):
        """smallest wall and obstacle clearance of the body points of every pose"""
        c, s = np.cos(YAW)[..., None], np.sin(YAW)[..., None]
        PX = X[..., None] + self._bpx * c - self._bpy * s
        PY = Y[..., None] + self._bpx * s + self._bpy * c
        cw = self._clearance(PX.ravel(), PY.ravel()).reshape(PX.shape).min(-1)
        if fld is None:
            return cw, np.full(cw.shape, 9.0)
        d, gx, gy, res = fld
        co = ndimage.map_coordinates(d, [(PY.ravel() - gy) / res - 0.5, (PX.ravel() - gx) / res - 0.5], order=1,
                                     mode='constant', cval=9.0).reshape(PX.shape).min(-1)
        return cw, co

    def _track(self, I0, X, Y):
        """line index of the last sample of every rollout, tracked sample by sample from I0 (+-0.3 m per 5 cm step)"""
        w = np.arange(-6, 7)
        I = np.broadcast_to(I0[:, None], X.shape[:2]).copy()
        for k in range(X.shape[2]):
            cand = (I[..., None] + w) % self.nf
            d2 = (X[..., k, None] - self.fx[cand]) ** 2 + (Y[..., k, None] - self.fy[cand]) ** 2
            I = np.take_along_axis(cand, np.argmin(d2, -1)[..., None], -1)[..., 0]
        return I

    def _s_at(self, i, x, y):
        return self.fs[i] + (x - self.fx[i]) * self.ftx[i] + (y - self.fy[i]) * self.fty[i]

    def _man_search(self, x, y, yaw):
        t0 = time.perf_counter()
        fld = self._obs_field(x, y)
        P = self._prims
        cw0, co0 = self._clr(np.array([x]), np.array([y]), np.array([yaw]), fld)
        fw = min(self.man_m_wall, max(float(cw0[0]) - self.man_close, self.man_close))
        fo = min(self.man_m_obs, max(float(co0[0]) - self.man_close, self.man_close))
        g = self.fi if self.fi is not None else int(np.argmin((self.fx - x) ** 2 + (self.fy - y) ** 2))
        win = (g + np.arange(-40, 41)) % self.nf
        i0 = int(win[np.argmin((self.fx[win] - x) ** 2 + (self.fy[win] - y) ** 2)])
        s0 = float(self._s_at(i0, x, y))
        F = dict(x=np.array([x]), y=np.array([y]), w=np.array([yaw]), i=np.array([i0]), m=np.array([np.inf]),
                 ln=np.array([0.0]))
        seqs, cands, n_eval = [()], [], 0
        K = int(math.ceil(float(P[:, 1].max()) / 0.05))
        u = np.minimum(np.arange(1, K + 1)[None, :] * 0.05, P[:, 1:2])
        for depth in range(3):
            X, Y, W = self._arc(F['x'][:, None, None], F['y'][:, None, None], F['w'][:, None, None],
                                P[None, :, 0:1], u[None])
            n_eval += X.shape[0] * X.shape[1]
            cw, co = self._clr(X, Y, W, fld)
            m = np.minimum(np.minimum(cw - fw, co - fo).min(-1), F['m'][:, None])
            a, b = np.nonzero(m >= 0.0)
            if not a.size:
                break
            I = self._track(F['i'][a], X[a, b][:, None], Y[a, b][:, None])[:, 0]
            xe, ye, we, me = X[a, b, -1], Y[a, b, -1], W[a, b, -1], m[a, b]
            ln = F['ln'][a] + P[b, 1]
            ds = (self._s_at(I, xe, ye) - s0 + 0.5 * self.L) % self.L - 0.5 * self.L
            mu = _wrap(we - self.fpsi[I])
            sc = (np.minimum(ds, self.man_goal) - self.man_w_len * ln - self.man_w_mu * np.abs(mu)
                  + self.man_w_clr * np.minimum(me, 0.1))
            for k in np.flatnonzero(ds >= self.man_min_prog):
                cands.append((float(sc[k]), float(ds[k]), seqs[a[k]] + (int(b[k]),)))
            keep = np.arange(a.size) if depth == 0 else np.argsort(-ds)[:self.man_beam]
            F = dict(x=xe[keep], y=ye[keep], w=we[keep], i=I[keep], m=me[keep], ln=ln[keep])
            seqs = [seqs[a[k]] + (int(b[k]),) for k in keep]
        best = max(cands) if cands else None
        self.man_stats.append(dict(ms=round((time.perf_counter() - t0) * 1e3, 1), n=n_eval, valid=len(cands),
                                   fw=round(fw, 3), fo=round(fo, 3),
                                   best=None if best is None else (round(best[0], 3), round(best[1], 3),
                                                                   [tuple(round(float(v), 3) for v in P[j]) for j in best[2]])))
        return None if best is None else ([tuple(float(v) for v in P[j]) for j in best[2]], fw, fo)

    def _arc_clear(self, x, y, yaw, dl, rem):
        """the rest of the current arc keeps the search's floors (- man_tol) with the cells seen now"""
        if rem <= 0.02:
            return True
        u = np.minimum(np.arange(1, int(math.ceil(rem / 0.05)) + 1) * 0.05, rem)
        cw, co = self._clr(*self._arc(x, y, yaw, dl, u), self._obs_field(x, y))
        M = self.man
        return bool(np.all(cw >= M['fw'] - self.man_tol) and np.all(co >= M['fo'] - self.man_tol))

    def _end(self, t_stop):
        self.man = None
        self.tr_mode = None
        self.t_stop = t_stop

    def _manoeuvre(self, t, x, y, yaw, v, mu, s_c):
        """arc manoeuvre of a car that has stood man_after s without a drivable plan: (steer, speed) while it
        commands, None when the MPC plan drives"""
        M = self.man
        if M is None:
            self.tr_mode = None
            if not (self.t_stop is not None and t - self.t_stop > self.man_after and abs(v) < 0.1
                    and self.n_man < self.man_max):
                return None
            res = self._man_search(x, y, yaw)
            if res is None:
                self.t_stop = t               # nothing found: wait man_after s, then search again
                return None
            self.man = M = dict(seq=res[0], k=0, phase=0, fw=res[1], fo=res[2])
            if self.n_man == 0:
                self.man_s = s_c
            self.n_man += 1
            self.n_man_tot += 1
        dl, ln = M['seq'][M['k']]
        if M['phase'] == 0:
            # stop, then turn the wheels at standstill (servo 3.2 rad/s + latency)
            if abs(v) > 0.03:
                return dl, 0.0
            prev = self.hist[-1][1] if self.hist else 0.0
            M.setdefault('ts', t + abs(dl - prev) / STEER_RATE + 0.08)
            if t < M['ts']:
                self.tr_mode = ('settle', M['k'])
                return dl, 0.0
            M.update(phase=1, x0=x, y0=y, yaw0=yaw, t1=t)
        k = math.tan(dl) / WB
        rem = ln - (abs(_wrap(yaw - M['yaw0'])) / abs(k) if abs(k) > 0.05 else math.hypot(x - M['x0'], y - M['y0']))
        if t - M['t1'] > 3.0 * ln / self.man_vf + 1.5:
            self._end(t)                      # timeout (3 x the nominal arc time + 1.5 s): stop, search again
            return dl, 0.0
        if rem <= 0.01:
            M.update(k=M['k'] + 1, phase=0)
            M.pop('ts', None)
            M.pop('t_happy', None)
            if M['k'] >= len(M['seq']):
                self._end(None)               # last arc done: the MPC takes over
                return None
            return self._manoeuvre(t, x, y, yaw, v, mu, s_c)
        if self.man_happy >= 0.0:
            # early hand-back: the MPC (planned every call from the current state) has a drivable plan again
            if not self._creep_blocked and not self.plan['emergency']:
                if t - M.setdefault('t_happy', t) >= self.man_happy:
                    self._end(None)
                    return None
            else:
                M.pop('t_happy', None)
        if not self._arc_clear(x, y, yaw, dl, rem):
            self._end(t)
            self._creep_blocked = True
            return dl, 0.0
        self.tr_mode = ('drive', M['k'])
        return dl, self.man_vf
