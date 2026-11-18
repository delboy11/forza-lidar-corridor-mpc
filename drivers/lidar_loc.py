"""LiDAR + map + own-command state estimation for the MPC drivers ("LiDAR-FAST"): no ground truth after the initial
pose.

Why: the time-trial driver (drivers.cmpc_berlin_tt:CorridorMPC, profile profiles/berlin_tt.json = "FAST") was fed the
simulator's ground truth (/car_state/pose_GT, /car_state/odom_GT).  LidarLocDriver wraps it (or any driver with the
act(scan, x, y, yaw, v, yaw_rate, t) interface, e.g. drivers.corridor_mpc_v2:CorridorMPC for obstacles) and feeds it its
own estimate of (x, y, yaw, v, yaw_rate) instead.  Results and design notes: work/lidar_loc/REPORT.md.

What the estimator uses (and nothing else):
  * the LiDAR scan (1080 beams, 4.7 rad FOV, mounted LIDAR_X = 0.27 m ahead of the pose point along the heading),
  * the known occupancy map (stack_master/maps/<map>; the same image / threshold the simulator ray-casts against),
  * the commands this driver issued (steering angle, speed) and the inner driver's actuator / vehicle model: 2-step
    steering buffer, steering servo 3.2 rad/s, the simulator's speed P-controller, the single-track model of
    sim_params.yaml (tyre scale c = the inner driver's online estimate); the command latency is estimated here
    (multiple-model, scan evidence) and handed to the inner driver, whose own estimator is switched off,
  * ONE initial pose per reset(): the operator's start / re-placement pose (= /initialpose in ROS, which any particle
    filter / AMCL also needs).  Sources, in this order: set_initial_pose(x, y, yaw) (avoid_node passes /initialpose),
    else init='gt': the x, y, yaw arguments of the FIRST act() after reset() (the harness / pose_GT, read once), or
    init='start': the map's start pose (raceline point 0, what run_avoid.sh publishes), or init='global': a global
    scan-to-map search (no outside pose at all).  The pose is refined by a local search; v and r start at 0.
After that, act() never reads its x, y, yaw, v, yaw_rate arguments (work/lidar_loc/test_gt_free.py feeds NaN / random
numbers after the initial pose and checks that every command is bit-identical; forbid_gt=True refuses init='gt',
raises if GT would be read, and makes avoid_node pass NaN instead of GT).
eval_log=True (evaluation only) records estimate - GT AFTER the command has been computed.
A given initial pose (operator / GT / map start) is authoritative: it is refined by local searches only (a scan still in
flight from before a teleport once matched the old place); if no candidate passes the checks for init_force scans (the
car stands right in front of an obstacle, a stale scan), the pose is taken as given.  act() never raises.

Inner drivers: any act(scan, x, y, yaw, v, yaw_rate, t) driver; tested with cmpc_berlin_tt (FAST, berlin_tt.json),
corridor_mpc_v2 and cmpc_berlin_obs (berlin_obs.json: creeps at 0.6 m/s, backs off at -0.3 m/s).
DART / DAgger: note_applied(steer, speed) after every act() tells the expert the command that was actually sent
(rewrites the localizer's and the inner driver's command history and the inner's a_cmd).

Estimator (per act(), ~0.5-1 ms):
  0. clock: the filter, the command history, the latency estimate and the inner driver's t run on the simulator's
     clock (10 ms per physics step), not on the stamps: the live gym_bridge steps the physics with a jittery wall-clock
     timer that runs slower (or faster) than real time, so 1-3 steps fall between two 50 Hz calls whose stamps differ by
     6-49 ms.  The number of steps since the last scan is chosen per scan by the scan evidence (every count from 1 to the
     stamp-based guess + 1 is predicted and registered; Laplace objective + a learned prior around stamp difference x
     estimated step rate); an identical scan means no step (nothing updated, last command repeated).  loc clock='wall'
     restores round(stamp difference / 10 ms) (v3; offline harness only).
  1. prediction: EKF state s = [x, y, yaw, v, r, beta] at the pose point (the simulator's CG reference point), steering
     angle tracked deterministically; propagated in 10 ms steps over the interval since the last scan with the issued
     commands and RK4 on the single-track model (v >= 0.5 m/s; kinematic model below and for every negative speed,
     i.e. reversing); Jacobian by one-sided finite differences (kinks of the speed controller at v = 0 and of the
     kinematic switch at 0.5 m/s).  One EKF per latency hypothesis (0..150 ms); unselected hypotheses that diverge are
     re-anchored on the selected one.
  2. scan-to-map registration as the measurement update (iterated EKF = Gauss-Newton on prior + robust scan cost) of the
     selected hypothesis: every `stride`-th valid beam endpoint (LiDAR offset modelled) against the signed distance
     field of the map (bilinear), residual = SDF - (sdf_off + sdf_off_b |cos incidence|) (the simulator's ray-marcher
     stops inside the first occupied pixel), Huber weights, gate 0.25 m in the first iteration and ~5 sigma later
     (obstacles >= 0.15 m from the walls, unmapped objects, max-range / inf / NaN beams are ignored).  The prior keeps
     the along-corridor direction defined where the walls alone do not; the scan information is deflated by
     info_scale (beam errors are correlated through the map pixels); beam sigma adapts to the measured residuals.
  3. speed and yaw rate are EKF states: the command-driven model carries them between scans, the registered poses
     correct them (no differentiation of poses).
  4. latency: the other hypotheses are updated with the quadratic model of the scan cost; their evidence (Laplace
     approximation) is summed with lat_tau forgetting while driving, the best one (with hysteresis) is selected.
  5. quality = inliers / (valid beams - free-space endpoints in compact, obstacle-like runs); endpoints behind walls or
     in long free-space runs count against the pose.  Bad scans are not used; after `lost_n` bad scans in a row (or one
     very bad one) the pose is searched again: local correlative search around the prediction, then a global search
     over the map (~0.3 s); the estimate is re-initialised from the best match (speed / yaw rate kept).
"""
import importlib
import json
import math
import os
import sys
import time

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage

try:
    from numba import njit
except Exception:                                    # pragma: no cover - numba ships with the simulator
    def njit(*a, **k):
        return (lambda f: f) if not (len(a) == 1 and callable(a[0])) else a[0]

AVOID = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPS = os.path.expanduser('~/forza_ws/race_stack/stack_master/maps')

# vehicle and sensor (sim_params.yaml / sim.yaml of the race stack; the same values the inner driver uses)
MU_, CSF_, CSR_, LF_, LR_, HCG_, G_ = 1.0489, 4.718, 5.4562, 0.15875, 0.17145, 0.074, 9.81
M_, I_ = 3.47, 0.04712
WB = LF_ + LR_
STEER_MAX = 0.4189
STEER_RATE = 3.2
KP_ACC, KP_BRK, A_MAX, V_SWITCH = 10.0 * 9.51 / 20.0, 10.0 * 9.51 / 5.0, 9.51, 7.319
LIDAR_X = 0.27
ANG_MIN = -2.35
ANG_INC = 4.7 / 1079
NB = 1080

# registry of driver instances in this process (evaluation scripts read the loc. statistics after an episode)
INSTANCES = []
# evaluation only: callable returning the simulator's obstacles [(x, y, theta, L, W), ...] (set by the evaluation
# script; used with eval_log for the error near obstacles and for phantom-obstacle counting, never for driving)
EVAL_OBSTACLES = None


# ---------------------------------------------------------------------------------------------------- numba kernels
@njit(cache=True)
def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@njit(cache=True)
def _targets(Ht, Hs, Hv, nh, tt, tl, tb):
    """steering target / speed command acting at time tt (issued <= tt - tl - tb resp. tt - tl)"""
    ts0 = tt + 1e-6 - tl - tb
    tv0 = tt + 1e-6 - tl
    hs, st, hv, vc = False, 0.0, False, 0.0
    for j in range(nh - 1, -1, -1):
        if Ht[j] <= ts0:
            hs, st = True, Hs[j]
            break
    for j in range(nh - 1, -1, -1):
        if Ht[j] <= tv0:
            hv, vc = True, Hv[j]
            break
    return hs, st, hv, vc


@njit(cache=True)
def _p_accel(v_cmd, v):
    dv = v_cmd - v
    if v > 0.0:
        a = (KP_ACC if dv > 0 else KP_BRK) * dv
    else:
        a = (2.0 * 9.51 / 20.0 if dv > 0 else 2.0 * 9.51 / 5.0) * dv
    pos = A_MAX * V_SWITCH / v if v > V_SWITCH else A_MAX
    return max(-A_MAX, min(pos, a))


@njit(cache=True)
def _st_rhs(v, r, beta, psi, delta, a, c):
    """single-track model of the simulator (linear tyres with load transfer), tyre stiffness scaled by c"""
    aa = -A_MAX if a < -A_MAX else (A_MAX if a > A_MAX else a)
    cf = c * MU_ * CSF_ * (G_ * LR_ - aa * HCG_)
    cr = c * MU_ * CSR_ * (G_ * LF_ + aa * HCG_)
    vl = v * WB
    a11 = -(cr + cf) / vl
    a12 = (cr * LR_ - cf * LF_) / (v * vl) - 1.0
    b1 = cf / vl
    a21 = M_ * (cr * LR_ - cf * LF_) / (I_ * WB)
    a22 = -M_ * (LF_ * LF_ * cf + LR_ * LR_ * cr) / (v * I_ * WB)
    b2 = M_ * LF_ * cf / (I_ * WB)
    return (v * math.cos(psi + beta), v * math.sin(psi + beta), r, a,
            a21 * beta + a22 * r + b2 * delta, a11 * beta + a12 * r + b1 * delta)


@njit(cache=True)
def _prop(Ht, Hs, Hv, nh, tl, tb, t0, n, s, delta, c):
    """state s = [x, y, yaw, v, r, beta] and steering angle delta from t0 over n 10 ms steps with the issued commands"""
    x, y, psi, v, r, beta = s[0], s[1], s[2], s[3], s[4], s[5]
    h = 0.01
    for k in range(n):
        hs, tgt, hv, vc = _targets(Ht, Hs, Hv, nh, t0 + 0.01 * k, tl, tb)
        d0 = delta
        if hs:
            diff = tgt - delta
            if abs(diff) > 1e-4:
                delta += max(-STEER_RATE * h, min(STEER_RATE * h, diff))
            delta = max(-STEER_MAX, min(STEER_MAX, delta))
        dm = 0.5 * (d0 + delta)
        a = _p_accel(vc, v) if hv else 0.0
        if v >= 0.5:
            # RK4 (the simulator's integrator), inputs held over the step
            k1 = _st_rhs(v, r, beta, psi, dm, a, c)
            k2 = _st_rhs(v + 0.5 * h * k1[3], r + 0.5 * h * k1[4], beta + 0.5 * h * k1[5], psi + 0.5 * h * k1[2], dm, a, c)
            k3 = _st_rhs(v + 0.5 * h * k2[3], r + 0.5 * h * k2[4], beta + 0.5 * h * k2[5], psi + 0.5 * h * k2[2], dm, a, c)
            k4 = _st_rhs(v + h * k3[3], r + h * k3[4], beta + h * k3[5], psi + h * k3[2], dm, a, c)
            x += h / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
            y += h / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
            psi += h / 6.0 * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
            v += h / 6.0 * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3])
            r += h / 6.0 * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4])
            beta += h / 6.0 * (k1[5] + 2.0 * k2[5] + 2.0 * k3[5] + k4[5])
        else:
            # kinematic model below 0.5 m/s and for every negative speed (the simulator switches back to the
            # single-track model below -0.5 m/s, which is unstable in reverse: a latency hypothesis pulled to -1.5 m/s
            # during a full-lock back-off diverged; the drivers reverse at <= 0.3 m/s)
            x += v * math.cos(psi) * h
            y += v * math.sin(psi) * h
            r = v * math.tan(dm) / WB
            psi += r * h
            beta = 0.0
            v += a * h
    out = np.empty(6)
    out[0], out[1], out[2], out[3], out[4], out[5] = x, y, psi, v, r, beta
    return out, delta


@njit(cache=True)
def _predict_ekf(Ht, Hs, Hv, nh, tl, tb, t0, n, s, P, delta, c, Qd, Pmax):
    """EKF prediction: mean by _prop, covariance with the finite-difference Jacobian; Qd = process noise per step"""
    s1, d1 = _prop(Ht, Hs, Hv, nh, tl, tb, t0, n, s, delta, c)
    F = np.eye(6)
    eps = np.array([1e-4, 1e-4, 1e-5, 1e-4, 1e-4, 1e-5])
    for j in range(6):
        sp = s.copy()
        # one-sided difference on the side of the model branch the state is in (P-controller kink at v = 0,
        # kinematic / dynamic switch at 0.5 m/s): a difference across a branch made F[v, v] ~ 800 at the launch
        sg = 1.0
        if j == 3 and (s[3] <= 0.0 or (s[3] < 0.5 and s[3] + eps[3] >= 0.5)):
            sg = -1.0
        sp[j] += sg * eps[j]
        s2, _ = _prop(Ht, Hs, Hv, nh, tl, tb, t0, n, sp, delta, c)
        for i in range(6):
            F[i, j] = (s2[i] - s1[i]) / (sg * eps[j])
    P1 = F @ P @ F.T
    for i in range(6):
        P1[i, i] += Qd[i] * n
    P1 = 0.5 * (P1 + P1.T)
    # keep the covariance in a physical range (a model-branch switch inside the interval can still inflate it)
    for i in range(6):
        if P1[i, i] > Pmax[i]:
            f = math.sqrt(Pmax[i] / P1[i, i])
            for j in range(6):
                P1[i, j] *= f
                P1[j, i] *= f
    return s1, P1, d1


@njit(cache=True)
def _sdf_at(sdf, ox, oy, res, W, H, px, py):
    """bilinear signed distance (m, + free) and its gradient; ok = inside the map"""
    u = (px - ox) / res - 0.5
    v = (py - oy) / res - 0.5
    fu = math.floor(u)
    fv = math.floor(v)
    c0 = int(fu)
    r0 = int(fv)
    if c0 < 0 or r0 < 0 or c0 + 1 >= W or r0 + 1 >= H:
        return 0.0, 0.0, 0.0, False
    fx = u - fu
    fy = v - fv
    v00 = sdf[r0, c0]
    v01 = sdf[r0, c0 + 1]
    v10 = sdf[r0 + 1, c0]
    v11 = sdf[r0 + 1, c0 + 1]
    val = (1.0 - fy) * ((1.0 - fx) * v00 + fx * v01) + fy * ((1.0 - fx) * v10 + fx * v11)
    gx = ((1.0 - fy) * (v01 - v00) + fy * (v11 - v10)) / res
    gy = ((1.0 - fx) * (v10 - v00) + fx * (v11 - v01)) / res
    return val, gx, gy, True


@njit(cache=True)
def _register(s_pred, P, ranges, cb, sb, stride, r_min, r_max, sdf, ox, oy, res, W, H, sig_b, sig_ang, off, off_b,
              hub, gate, gate2, inl_thr, info_scale, n_iter, use_prior, link, obs_ext):
    """iterated EKF update with the scan (Gauss-Newton on prior + robust scan cost).  Returns the posterior mean and
    covariance, (valid beams, inliers, rms of the inliers), the pose information matrix of the scan and iterations.
    Residual of a beam: SDF(endpoint) - (off + off_b |cos incidence|) (the ray-marcher stops inside the first occupied
    pixel: ~0.5 px deep for a perpendicular beam, less for a grazing one).  Outliers (obstacles, unmapped objects):
    Huber weights, residuals beyond `gate` (first iteration) / `gate2` (later iterations, close to the optimum) are
    ignored; nfree = valid beams whose endpoint lies in free space beyond inl_thr (could be an obstacle), as opposed to
    endpoints behind a wall (contradict the pose); nsmall = free-space endpoints in compact runs (consecutive endpoints
    <= link apart, first to last <= obs_ext: obstacle-like), counted at the posterior pose.  use_prior False: the scan alone (with a weak prior from P only for conditioning)."""
    Pinv = np.linalg.inv(P)
    s = s_pred.copy()
    M = Pinv.copy()
    nval = 0
    ninl = 0
    nfree = 0
    ss = 0.0
    A = np.zeros((3, 3))
    b = np.zeros(3)
    p0 = s_pred[:3].copy()
    it_done = 0
    for it in range(n_iter):
        A[:, :] = 0.0
        b = np.zeros(3)
        p0 = s[:3].copy()
        nval = 0
        ninl = 0
        nfree = 0
        ss = 0.0
        gt_ = gate if it == 0 else gate2
        cy = math.cos(s[2])
        sy = math.sin(s[2])
        lx = s[0] + LIDAR_X * cy
        ly = s[1] + LIDAR_X * sy
        for i in range(0, ranges.shape[0], stride):
            rg = ranges[i]
            if not (rg > r_min and rg < r_max):
                continue
            bx = rg * cb[i]
            by = rg * sb[i]
            px = lx + cy * bx - sy * by
            py = ly + sy * bx + cy * by
            val, gx, gy, ok = _sdf_at(sdf, ox, oy, res, W, H, px, py)
            if not ok:
                continue
            nval += 1
            gn = math.sqrt(gx * gx + gy * gy)
            ci = abs(gx * (px - lx) + gy * (py - ly)) / (gn * rg) if gn > 1e-9 else 1.0
            e = val - (off + off_b * ci)
            ae = abs(e)
            if ae < inl_thr:
                ninl += 1
                ss += e * e
            elif e > 0.0:
                nfree += 1
            if ae > gt_:
                continue
            sg = math.sqrt(sig_b * sig_b + (sig_ang * rg) ** 2)
            w = 1.0 if ae <= hub * sg else hub * sg / ae
            w /= sg * sg * info_scale
            jt = -gx * (py - s[1]) + gy * (px - s[0])
            A[0, 0] += w * gx * gx
            A[0, 1] += w * gx * gy
            A[0, 2] += w * gx * jt
            A[1, 1] += w * gy * gy
            A[1, 2] += w * gy * jt
            A[2, 2] += w * jt * jt
            b[0] += w * gx * e
            b[1] += w * gy * e
            b[2] += w * jt * e
        A[1, 0] = A[0, 1]
        A[2, 0] = A[0, 2]
        A[2, 1] = A[1, 2]
        M = Pinv.copy()
        for i in range(3):
            for j in range(3):
                M[i, j] += A[i, j]
        dsv = s - s_pred
        dsv[2] = _wrap(dsv[2])
        g = Pinv @ dsv if use_prior else np.zeros(6)
        for i in range(3):
            g[i] += b[i]
        dx = -np.linalg.solve(M, g)
        for i in range(6):
            s[i] += dx[i]
        it_done = it + 1
        if abs(dx[0]) < 1e-5 and abs(dx[1]) < 1e-5 and abs(dx[2]) < 1e-6:
            break
    Pp = np.linalg.inv(M)
    Pp = 0.5 * (Pp + Pp.T)
    s[2] = _wrap(s[2])
    rms = math.sqrt(ss / ninl) if ninl > 0 else 1.0
    # classification at the posterior pose: inliers, free-space endpoints, and those of them in compact runs; and the
    # objective there (robust scan cost truncated at gate2 + prior term + log det(M P): Laplace evidence, for comparing
    # hypotheses that differ in the prediction, e.g. the number of simulator steps since the last scan)
    jcost = 0.0
    cy = math.cos(s[2])
    sy = math.sin(s[2])
    lx = s[0] + LIDAR_X * cy
    ly = s[1] + LIDAR_X * sy
    nval = 0
    ninl = 0
    nfree = 0
    nsmall = 0
    run_n = 0
    x0r = 0.0
    y0r = 0.0
    xl = 0.0
    yl = 0.0
    for i in range(0, ranges.shape[0], stride):
        rg = ranges[i]
        if not (rg > r_min and rg < r_max):
            continue
        bx = rg * cb[i]
        by = rg * sb[i]
        px = lx + cy * bx - sy * by
        py = ly + sy * bx + cy * by
        val, gx, gy, ok = _sdf_at(sdf, ox, oy, res, W, H, px, py)
        if not ok:
            continue
        nval += 1
        gn = math.sqrt(gx * gx + gy * gy)
        ci = abs(gx * (px - lx) + gy * (py - ly)) / (gn * rg) if gn > 1e-9 else 1.0
        e = val - (off + off_b * ci)
        sg = math.sqrt(sig_b * sig_b + (sig_ang * rg) ** 2)
        ae = min(abs(e), gate2)
        if ae <= hub * sg:
            jcost += (ae / sg) ** 2 / info_scale
        else:
            jcost += (2.0 * hub * ae / sg - hub * hub) / info_scale
        free = False
        if abs(e) < inl_thr:
            ninl += 1
        elif e > 0.0:
            nfree += 1
            free = True
        if free and run_n > 0 and math.hypot(px - xl, py - yl) <= link:
            run_n += 1
        else:
            if run_n > 0 and math.hypot(xl - x0r, yl - y0r) <= obs_ext:
                nsmall += run_n
            run_n = 0
            if free:
                run_n = 1
                x0r = px
                y0r = py
        if free:
            xl = px
            yl = py
    if run_n > 0 and math.hypot(xl - x0r, yl - y0r) <= obs_ext:
        nsmall += run_n
    if use_prior:
        dsv = s - s_pred
        dsv[2] = _wrap(dsv[2])
        jcost += dsv @ (Pinv @ dsv)
        sg1, ld1 = np.linalg.slogdet(M)
        sg2, ld2 = np.linalg.slogdet(P)
        jcost += ld1 + ld2
    return s, Pp, nval, ninl, nfree, nsmall, rms, A, it_done, p0, b, jcost


@njit(cache=True)
def _mm_predict(Ht, Hs, Hv, nh, lats, tb, t0, n, S, PP, D, c, Qd, Pmax):
    """EKF prediction of every latency hypothesis k (state S[k], covariance PP[k], steering D[k])"""
    K = lats.shape[0]
    S1 = np.empty_like(S)
    P1 = np.empty_like(PP)
    D1 = np.empty_like(D)
    for k in range(K):
        s1, p1, d1 = _predict_ekf(Ht, Hs, Hv, nh, lats[k], tb, t0, n, S[k], PP[k], D[k], c, Qd, Pmax)
        S1[k] = s1
        P1[k] = p1
        D1[k] = d1
    return S1, P1, D1


@njit(cache=True)
def _mm_update(S, PP, p0, A, b):
    """scan update of every hypothesis with the quadratic model of the scan cost around p0 (one Gauss-Newton step:
    gradient b + A (pose - p0), Hessian A; the robust weights of the full registration are kept) and its evidence
    J = -2 log p(scan | hypothesis) + const (Laplace approximation: minimum of prior + scan cost + log det(I + P A))"""
    K = S.shape[0]
    S1 = np.empty_like(S)
    P1 = np.empty_like(PP)
    J = np.empty(K)
    for k in range(K):
        Pi = np.linalg.inv(PP[k])
        M = Pi.copy()
        for i in range(3):
            for j in range(3):
                M[i, j] += A[i, j]
        u = np.empty(3)
        u[0] = S[k, 0] - p0[0]
        u[1] = S[k, 1] - p0[1]
        u[2] = _wrap(S[k, 2] - p0[2])
        gq = b + A @ u
        g = np.zeros(6)
        g[0], g[1], g[2] = gq[0], gq[1], gq[2]
        Minv = np.linalg.inv(M)
        dx = -(Minv @ g)
        S1[k] = S[k] + dx
        S1[k, 2] = _wrap(S1[k, 2])
        P1[k] = 0.5 * (Minv + Minv.T)
        f0 = 2.0 * (b @ u) + u @ (A @ u)
        # det(I + P A_ext) = det(M) det(P)
        sgn1, ld1 = np.linalg.slogdet(M)
        sgn2, ld2 = np.linalg.slogdet(PP[k])
        J[k] = f0 - g @ (Minv @ g) + ld1 + ld2
    return S1, P1, J


@njit(cache=True)
def _corr_scores(lf, ox, oy, res, W, H, qx, qy, cx, cy, out):
    """correlative search: out[k, j] = sum of the likelihood field over the points (qx[j], qy[j]) (car frame rotated
    by yaw j) placed at candidate position k (nearest pixel)"""
    nk = cx.shape[0]
    nj = qx.shape[0]
    nb = qx.shape[1]
    for k in range(nk):
        x0 = (cx[k] - ox) / res
        y0 = (cy[k] - oy) / res
        for j in range(nj):
            acc = 0.0
            for i in range(nb):
                c = int(x0 + qx[j, i])
                r = int(y0 + qy[j, i])
                if 0 <= c < W and 0 <= r < H:
                    acc += lf[r, c]
            out[k, j] = acc


# ---------------------------------------------------------------------------------------------------- map model
class MapModel:
    """occupancy (same image / rule as the simulator: grey <= 128 occupied), signed distance field to the
    occupied-pixel boundary (m, + in free space), likelihood fields for the correlative search"""

    _cache = {}

    def __init__(self, map_name):
        d = os.path.join(MAPS, map_name)
        y = f'{d}/{map_name}_orig.yaml'
        if not os.path.exists(y):
            y = f'{d}/{map_name}.yaml'
        meta = yaml.safe_load(open(y))
        img = np.array(Image.open(f"{d}/{meta['image']}").convert('L').transpose(Image.FLIP_TOP_BOTTOM))
        occ = img <= 128
        self.res = float(meta['resolution'])
        self.ox, self.oy = float(meta['origin'][0]), float(meta['origin'][1])
        self.H, self.W = occ.shape
        d_free = ndimage.distance_transform_edt(~occ)          # free pixel -> nearest occupied pixel centre (px)
        d_occ = ndimage.distance_transform_edt(occ)            # occupied pixel -> nearest free pixel centre (px)
        self.sdf = np.ascontiguousarray(np.where(occ, -(d_occ - 0.5), d_free - 0.5) * self.res, dtype=np.float64)
        self.occ = occ
        self._lf = {}
        # start pose of the map (finish line, first raceline point), used by init='start'
        try:
            w = json.load(open(f'{d}/global_waypoints.json'))['global_traj_wpnts_iqp']['wpnts']
            self.start = (w[0]['x_m'], w[0]['y_m'], w[0]['psi_rad'])
            self.line = np.array([[p['x_m'], p['y_m'], p['psi_rad']] for p in w])
        except Exception:
            self.start, self.line = None, None

    @classmethod
    def get(cls, map_name):
        if map_name not in cls._cache:
            cls._cache[map_name] = MapModel(map_name)
        return cls._cache[map_name]

    def lfield(self, tau, off, grow=0):
        """likelihood field exp(-(sdf - off)^2 / 2 tau^2) (float32), optionally max-filtered by `grow` pixels"""
        key = (round(tau, 4), round(off, 4), grow)
        if key not in self._lf:
            lf = np.exp(-0.5 * ((self.sdf - off) / tau) ** 2)
            if grow > 0:
                lf = ndimage.maximum_filter(lf, size=2 * grow + 1)
            self._lf[key] = np.ascontiguousarray(lf, dtype=np.float32)
        return self._lf[key]

    def free_points(self, step, clear):
        """candidate positions for the global search: free space with clearance >= clear on a `step` m grid"""
        k = max(1, int(round(step / self.res)))
        rr, cc = np.mgrid[k // 2:self.H:k, k // 2:self.W:k]
        ok = self.sdf[rr, cc] >= clear
        return (self.ox + (cc[ok] + 0.5) * self.res, self.oy + (rr[ok] + 0.5) * self.res)


# ---------------------------------------------------------------------------------------------------- estimator
class LidarLocalizer:
    """EKF on [x, y, yaw, v, r, beta] with the scan-to-map registration as measurement (see module header)"""

    DEFAULTS = dict(
        stride=2, r_min=0.06, r_max=20.0, sig_b=0.014, sig_ang=0.0005, sdf_off=-0.005, sdf_off_b=-0.015, huber=2.0,
        gate=0.25, gate2_min=0.06, gate2_k=5.0,
        inl_thr=0.08, info_scale=3.0, n_iter=4,
        # process noise per 10 ms step (std): x, y, yaw, v, r, beta (replay grid on GT-driven recordings, REPORT.md)
        q_xy=0.002, q_yaw=0.001, q_v=0.03, q_r=0.15, q_beta=0.005,
        # initial covariance (std)
        p0_xy=0.05, p0_yaw=0.03, p0_v=0.1, p0_r=0.1, p0_beta=0.02,
        # tracking quality / relocalisation
        # quality q = inliers / (valid beams - endpoints in free space): obstacles do not lower it, endpoints behind
        # walls do.  Normal tracking: q >= 0.97 also with 3 cm scan noise; a mirrored pose in a corridor reached 0.84.
        # q_min_inl: the inliers must also be this fraction of all valid beams
        # Free-space endpoints count as explained only in compact runs (obstacle-like: consecutive endpoints <= obs_link
        # apart, first to last <= obs_ext; the obstacles are 0.30 m cubes): a displaced pose had 46 % inliers and 50 %
        # free-space endpoints along walls; real obstacles covered up to 27 % of the beams.
        # q_bad 0.9: tracking stays >= 0.97 (3 cm scan noise, obstacles); a wrong step count gave 0.81-0.83 fits
        # q_min_inl / q_min_inl_reloc: floors on the inlier fraction of all valid beams (the compact-run rule rejects
        # displaced poses by itself; an obstacle 9 cm in front of the LiDAR covered 75 % of the beams)
        q_bad=0.9, q_lost=0.3, lost_n=3, q_ok=0.95, jump_gate=0.5, q_min_inl=0.2, q_min_inl_reloc=0.25,
        obs_link=0.2, obs_ext=0.55,
        # an init / relocalisation candidate must also fit as well as tracking does: inlier RMS <= max(rms_ok_k ref,
        # ref + rms_ok_add), ref = beam noise (EMA of the tracking RMS).  A spurious along-track minimum 0.12 m off had
        # 100 % inliers but RMS 0.0195 against 0.0136 at the true pose
        rms_ok_k=1.3, rms_ok_add=0.004,
        loc_win=0.8, loc_yaw=0.4, loc_step=0.05, loc_ystep=0.0175,
        glob_step=0.12, glob_ystep=0.0524, glob_clear=0.10, glob_beams=90, glob_tau=0.12, glob_top=40,
        retry_global=0.5,
        t_buf=0.02,
        # beam noise used in the update: max(sig_b, EMA of the inlier residual RMS) (scan noise is not known a priori)
        sig_adapt=True, sig_alpha=0.05,
        # command latency: 'mm' = own multiple-model estimate (one EKF per hypothesis lat_min..lat_max in lat_step,
        # scored by the scan evidence, forgetting lat_tau s, switch when better by lat_hyst), 'inner' = the value
        # passed to step() (the inner driver's estimate)
        lat_mode='mm', lat_min=0.0, lat_max=0.15, lat_step=0.01, lat_tau=2.0, lat_hyst=2.0, lat_vmin=1.0, lat0=0.03,
        # clock: 'sim' = number of physics steps per scan from the scan evidence (prior: Gaussian around stamp
        # difference x rate, variance n_var + (n_sig1 n_exp)^2, n_var = EMA of the squared deviation of decisive
        # choices, >= n_var_min); rate (steps / stamp second) from choices better than the runner-up by rate_margin
        # while v > rate_vmin, forgetting rate_lam per update; 'wall' = round(dt / 10 ms) (v3)
        clock='sim', n_var0=0.36, n_var_min=0.04, n_var_alpha=0.02, n_sig1=0.1, rate0=100.0, rate_margin=9.0,
        rate_vmin=1.5, rate_lam=0.99,
    )

    def __init__(self, map_name, **kw):
        p = dict(self.DEFAULTS)
        for k, v in kw.items():
            if k not in p:
                raise KeyError(f'unknown localizer parameter {k}')
            p[k] = v
        self.p = p
        for k, v in p.items():
            setattr(self, k, v)
        self.map = MapModel.get(map_name)
        ang = ANG_MIN + ANG_INC * np.arange(NB)
        self.cb, self.sb = np.cos(ang), np.sin(ang)
        self.ang = ang
        self.Qd = np.array([self.q_xy, self.q_xy, self.q_yaw, self.q_v, self.q_r, self.q_beta]) ** 2
        self.Pmax = np.array([1.0, 1.0, 0.3, 2.0, 2.0, 0.2]) ** 2
        self.P0 = np.diag(np.array([self.p0_xy, self.p0_xy, self.p0_yaw, self.p0_v, self.p0_r, self.p0_beta]) ** 2)
        self.lats = np.round(np.arange(self.lat_min, self.lat_max + 1e-9, self.lat_step), 4)
        self.K = self.lats.size if self.lat_mode == 'mm' else 1
        self.Jsum = np.zeros(self.K)            # evidence of the latency hypotheses: a property of the system, kept
        self.j = int(np.argmin(np.abs(self.lats - self.lat0))) if self.K > 1 else 0
        self.t_lat_est = float(self.lats[self.j]) if self.K > 1 else self.lat0
        m = self.map
        # compile now, not inside act()
        z = np.zeros(1)
        _predict_ekf(z, z, z, 0, 0.03, 0.02, 0.0, 2, np.array([0.0, 0, 0, 1.0, 0, 0]), np.eye(6), 0.0, 1.0, self.Qd,
                     self.Pmax)
        _register(np.zeros(6), np.eye(6), np.ones(NB), self.cb, self.sb, 2, 0.06, 20.0, m.sdf, m.ox, m.oy, m.res,
                  m.W, m.H, 0.015, 0.0015, -0.02, 0.0, 2.0, 0.25, 0.08, 0.08, 10.0, 2, True, 0.2, 0.55)
        S0 = np.zeros((2, 6))
        S0[:, 3] = 1.0
        S1, P1, _ = _mm_predict(z, z, z, 0, np.array([0.02, 0.03]), 0.02, 0.0, 2, S0, np.stack([np.eye(6)] * 2),
                                np.zeros(2), 1.0, self.Qd, self.Pmax)
        _mm_update(S1, P1, np.zeros(3), np.eye(3), np.zeros(3))
        lf = m.lfield(self.glob_tau, self.sdf_off + 0.7 * self.sdf_off_b)
        _corr_scores(lf, m.ox, m.oy, m.res, m.W, m.H, np.zeros((1, 1)), np.zeros((1, 1)), np.zeros(1), np.zeros(1),
                     np.zeros((1, 1)))
        self.reset()

    def reset(self):
        self.s = None               # EKF mean [x, y, yaw, v, r, beta] of the selected latency hypothesis
        self.P = None
        self.delta = 0.0            # steering angle (deterministic, from the command history)
        self.S = self.PP = self.D = None    # all hypotheses (K, 6), (K, 6, 6), (K,)
        self.t = None               # time of the estimate
        self.Ht, self.Hs, self.Hv = [], [], []
        self.n_bad = 0
        self.lost = False
        self.t_glob = -1e9
        self.q = 1.0
        self.last_info = None
        if not hasattr(self, 'rms_ema'):
            self.rms_ema = self.sig_b          # a property of the sensor: kept over resets
        if not hasattr(self, 'rate'):          # physics steps per stamp second: a property of the system, kept
            self.rate = float(self.rate0)
            self.rate_N, self.rate_T = 20.0, 20.0 / self.rate0
            self.n_var = float(self.n_var0)
        self.t_wall = None                     # stamp of the last scan used (self.t is the simulator clock)
        self.debug_scores = getattr(self, 'debug_scores', False)
        self.last_r = None
        self.n_last = 0
        self.n_dup = getattr(self, 'n_dup', 0)

    # -- commands
    def add_command(self, t, steer, speed):
        self.Ht.append(float(t))
        self.Hs.append(float(steer))
        self.Hv.append(float(speed))

    def _hist(self, t_keep):
        k = 0
        while k + 1 < len(self.Ht) and self.Ht[k + 1] < t_keep:
            k += 1
        if k:
            del self.Ht[:k], self.Hs[:k], self.Hv[:k]
        n = len(self.Ht)
        return (np.array(self.Ht, dtype=np.float64), np.array(self.Hs, dtype=np.float64),
                np.array(self.Hv, dtype=np.float64), n)

    # -- initialisation
    def init_pose(self, t, x, y, yaw, scan, refine=True):
        """start the filter at (x, y, yaw), standing car; refine the pose with a local search + registration.
        Returns the scan quality at the refined pose."""
        s = np.array([x, y, yaw, 0.0, 0.0, 0.0])
        q = 1.0
        if refine and scan is not None:
            s3, q = self.local_search(scan, s[:3], win=0.3, yaw_win=0.15)
            s[:3] = s3
        self._set_all(s, self.P0.copy(), 0.0)
        self.t = t
        self.t_wall = t
        self.last_r = self._ranges(scan) if scan is not None else None
        self.n_bad = 0
        self.lost = False
        self.q = q
        return q

    def initialise(self, t, scan, pose0):
        """start the filter (standing car) from pose0 = (x, y, yaw) or, pose0 None, from a global search.  pose0 is
        refined by a local search (+-0.3 m, +-0.15 rad); when no candidate fits (quality q_ok and RMS, see refine) by a
        wider one (+-1 m, +-0.5 rad); never by a global search.  Returns (ok, quality); the filter is only set when ok."""
        if pose0 is None:
            pz, q = self.global_search(scan)
        else:
            # a given pose is authoritative: no global search (a scan taken before the teleport, still in flight
            # after /initialpose, matched the OLD place 2.1 m away); the caller retries with the next scan and finally
            # takes the pose as given (LidarLocDriver.init_force)
            pz, q = self.local_search(scan, pose0, win=0.3, yaw_win=0.15)
            if q < self.q_ok:
                pz2, q2 = self.local_search(scan, pose0, win=1.0, yaw_win=0.5)
                if q2 > q:
                    pz, q = pz2, q2
        ok = pz is not None and q >= self.q_ok
        if ok:
            self.init_pose(t, pz[0], pz[1], pz[2], scan, refine=False)
            self.q = q
        return ok, q

    def _set_all(self, s, P, delta):
        self.S = np.tile(s, (self.K, 1))
        self.PP = np.tile(P, (self.K, 1, 1))
        self.D = np.full(self.K, float(delta))
        self._select()

    def _select(self):
        self.s, self.P, self.delta = self.S[self.j], self.PP[self.j], float(self.D[self.j])

    def _ranges(self, scan):
        r = np.asarray(scan, dtype=np.float64)
        if r.shape[0] != NB:
            raise ValueError(f'scan with {r.shape[0]} beams, expected {NB}')
        return np.where(np.isfinite(r), r, -1.0)

    def _gate2(self, sig):
        return max(self.gate2_min, self.gate2_k * sig)

    def _quality(self, nval, ninl, nsmall, min_inl=None):
        """tracking quality: inliers among the beams that do not end in free space in a compact (obstacle-like) run;
        0 when the inliers are fewer than q_min_inl of all valid beams"""
        if ninl < (self.q_min_inl if min_inl is None else min_inl) * max(nval, 1):
            return 0.0
        return ninl / max(nval - nsmall, 1)

    def refine(self, scan, pose3, iters=8):
        """scan-only registration from pose3 (weak prior), gates from coarse to fine; returns pose, ranking key
        (inlier fraction of all valid beams to 1 %, then -RMS of the inliers: a shallow local minimum 10 cm along the
        track had all beams within the 8 cm inlier band too), tracking quality (accepts a candidate)"""
        m = self.map
        r = self._ranges(scan)
        s = np.array([pose3[0], pose3[1], pose3[2], 0.0, 0.0, 0.0])
        P = np.diag(np.array([1.0, 1.0, 0.5, 1.0, 1.0, 1.0]) ** 2)
        nval, ninl, nsmall = 1, 0, 0
        sig = max(self.sig_b, self.rms_ema) if self.sig_adapt else self.sig_b
        # the last stage runs to convergence: in a shallow direction the robust Gauss-Newton moves slowly (8 iterations
        # stopped 2-6 cm short of the optimum at 2 of 98 start poses)
        for gate, nit in ((0.5, iters), (0.25, 30)):
            s, _, nval, ninl, nfree, nsmall, rms, A, _, _, _, _ = _register(
                s, P, r, self.cb, self.sb, 1, self.r_min, self.r_max, m.sdf, m.ox, m.oy, m.res, m.W, m.H, sig,
                self.sig_ang, self.sdf_off, self.sdf_off_b, self.huber, gate, self._gate2(sig) if gate < 0.3 else gate,
                self.inl_thr, self.info_scale, nit, False, self.obs_link, self.obs_ext)
        ref = max(self.sig_b, self.rms_ema)
        qt = self._quality(nval, ninl, nsmall, self.q_min_inl_reloc)
        if rms > max(self.rms_ok_k * ref, ref + self.rms_ok_add):
            qt = 0.0                          # fits worse than a tracked scan: not accepted (the search widens)
        return s[:3].copy(), (round(ninl / max(nval, 1), 2), -rms), qt

    def _car_points(self, scan, nbeam):
        r = self._ranges(scan)
        idx = np.arange(0, NB, max(1, NB // nbeam))
        idx = idx[(r[idx] > self.r_min) & (r[idx] < self.r_max)]
        return LIDAR_X + r[idx] * self.cb[idx], r[idx] * self.sb[idx]

    def _scores(self, scan, cx, cy, yaws, tau, nbeam, grow=0):
        m = self.map
        bx, by = self._car_points(scan, nbeam)
        cyw, syw = np.cos(yaws)[:, None], np.sin(yaws)[:, None]
        qx = (cyw * bx[None, :] - syw * by[None, :]) / m.res
        qy = (syw * bx[None, :] + cyw * by[None, :]) / m.res
        out = np.zeros((cx.size, yaws.size))
        _corr_scores(m.lfield(tau, self.sdf_off + 0.7 * self.sdf_off_b, grow), m.ox, m.oy, m.res, m.W, m.H, qx, qy,
                     np.ascontiguousarray(cx, dtype=np.float64), np.ascontiguousarray(cy, dtype=np.float64), out)
        return out / max(bx.size, 1)

    def local_search(self, scan, pose3, win=None, yaw_win=None):
        """correlative search in a window around pose3, then registration; returns pose, quality"""
        win = self.loc_win if win is None else win
        yaw_win = self.loc_yaw if yaw_win is None else yaw_win
        g = np.arange(-win, win + 1e-9, self.loc_step)
        gx, gy = np.meshgrid(g, g)
        cx, cy = pose3[0] + gx.ravel(), pose3[1] + gy.ravel()
        yaws = pose3[2] + np.arange(-yaw_win, yaw_win + 1e-9, self.loc_ystep)
        sc = self._scores(scan, cx, cy, yaws, 0.06, 120)
        best = []
        flat = np.argsort(sc.ravel())[::-1][:5]
        for f in flat:
            k, j = divmod(int(f), yaws.size)
            best.append(self.refine(scan, (cx[k], cy[k], yaws[j])))
        # the starting pose itself too
        best.append(self.refine(scan, pose3))
        b = max(best, key=lambda b: b[1])
        b = self._polish(scan, b)
        return b[0], b[2]

    def _polish(self, scan, b):
        """repeat the refine from the chosen candidate until it moves < 1 mm: along a flat direction (berlin s ~ 30 m)
        the robust Gauss-Newton stalled 35-55 mm short of the optimum at the same RMS"""
        for _ in range(3):
            b2 = self.refine(scan, b[0])
            moved = math.hypot(b2[0][0] - b[0][0], b2[0][1] - b[0][1])
            b = b2
            if moved < 1e-3:
                break
        return b

    def global_search(self, scan):
        """correlative search over all free positions and headings, top candidates refined; returns pose, quality"""
        cx, cy = self.map.free_points(self.glob_step, self.glob_clear)
        yaws = np.arange(-math.pi, math.pi, self.glob_ystep)
        sc = self._scores(scan, cx, cy, yaws, self.glob_tau, self.glob_beams, grow=1)
        bj = np.argmax(sc, axis=1)
        bs = sc[np.arange(cx.size), bj]
        order = np.argsort(bs)[::-1]
        picked = []
        for k in order:
            if all(math.hypot(cx[k] - cx[q], cy[k] - cy[q]) > 0.3 for q in picked):
                picked.append(k)
            if len(picked) >= self.glob_top:
                break
        best = (None, (-1.0, 0.0), 0.0)
        good = 0
        for k in picked:
            pz, key, qt = self.refine(scan, (cx[k], cy[k], yaws[bj[k]]))
            if key > best[1]:
                best = (pz, key, qt)
            good += key[0] > 0.97
            if good >= 3:              # a few candidates that fit: the best of them (not the first)
                break
        if best[0] is not None:
            best = self._polish(scan, best)
        return best[0], best[2]

    # -- one scan
    def step(self, t, scan, t_lat, c_tyre=1.0):
        """one scan with stamp t (wall clock in ROS): predict with the issued commands, update with the scan; returns
        (x, y, yaw, v, r).  The filter runs on the simulator's clock (self.t, 10 ms per physics step): clock='sim'
        chooses the number of physics steps since the last scan among candidates around (t - t_wall) * rate by the
        scan evidence (the ROS bridge steps the physics with a jittery wall-clock timer, slower than real time, and
        publishes on another timer: 1-3 steps between 50 Hz calls whose stamps differ by 6-49 ms); rate (steps per
        stamp second) is estimated from the decisive choices.  clock='wall': round((t - t_wall) / 10 ms) (v3).
        An identical scan (republished, no physics step in between) changes nothing (self.n_last = 0)."""
        m = self.map
        r = self._ranges(scan)
        if self.last_r is not None and np.array_equal(r, self.last_r):
            self.n_last = 0
            self.n_dup += 1
            s = self.s
            return s[0], s[1], s[2], s[3], s[4]
        dtw = t - self.t_wall
        if self.clock == 'wall':
            n0 = int(round(dtw / 0.01))
            if n0 <= 0:
                s = self.s
                return s[0], s[1], s[2], s[3], s[4]
            n_exp = float(n0)
        else:
            # a late physics timer does not catch up and short steps follow long ones (live: 1-3 steps between calls
            # whose stamps differ by 6-49 ms): the stamp difference is only a weak prior, the scan evidence decides
            n_exp = max(dtw, 0.0) * self.rate
        self.last_r = r
        self.t_wall = max(self.t_wall, t)
        Ht, Hs, Hv, nh = self._hist(self.t - 0.5)
        j = self.j
        lat_j = float(self.lats[j]) if self.K > 1 else float(t_lat)
        sig = max(self.sig_b, self.rms_ema) if self.sig_adapt else self.sig_b
        sig_n = math.sqrt(self.n_var + (self.n_sig1 * n_exp) ** 2)
        self.dbg_sc = []
        cache = {}

        def evaluate(n):
            s_pred, P_pred, d1 = _predict_ekf(Ht, Hs, Hv, nh, lat_j, float(self.t_buf), float(self.t), n, self.S[j],
                                              self.PP[j], float(self.D[j]), float(c_tyre), self.Qd, self.Pmax)
            reg = _register(s_pred, P_pred, r, self.cb, self.sb, self.stride, self.r_min, self.r_max, m.sdf, m.ox,
                            m.oy, m.res, m.W, m.H, sig, self.sig_ang, self.sdf_off, self.sdf_off_b, self.huber,
                            self.gate, self._gate2(sig), self.inl_thr, self.info_scale, self.n_iter, True,
                            self.obs_link, self.obs_ext)
            score = reg[11] + ((n - n_exp) / sig_n) ** 2
            cache[n] = (score, n, s_pred, P_pred, d1, reg)
            if self.debug_scores:
                self.dbg_sc.append((n, score, reg[11], reg[3] / max(reg[2], 1), reg[6],
                                    math.hypot(reg[0][0] - s_pred[0], reg[0][1] - s_pred[1])))

        if self.clock == 'wall':
            evaluate(min(int(round(n_exp)), 500))
        else:
            # every count from 1 to the guess + 1 (far from the truth the registration converges to wrong local fits:
            # the evidence is flat there and a hill-climb from a too-large guess after a stall got stuck), then upward
            # while the best is the largest one tried
            hi = min(max(2, int(round(n_exp)) + 1), 500)
            for nn in range(1, hi + 1):
                evaluate(nn)
            while min(cache.values(), key=lambda c: c[0])[1] == hi and hi < 500:
                hi += 1
                evaluate(hi)
        scores = sorted(c[0] for c in cache.values())
        best = min(cache.values(), key=lambda c: c[0])
        _, n, s_pred, P_pred, d1, reg = best
        s_new, P_new, nval, ninl, nfree, nsmall, rms, A, it, p0, bq, jc = reg
        self.n_last = n
        if len(scores) > 1:
            sc = scores
            if sc[1] - sc[0] > self.rate_margin and s_new[3] > self.rate_vmin:
                # a decisive choice while driving: update the step rate (forgetting rate_lam per update) and the
                # spread of the true count around the stamp-based guess (the prior's width: ~0 on an exact clock)
                self.n_var += self.n_var_alpha * ((n - n_exp) ** 2 - self.n_var)
                self.n_var = max(self.n_var, self.n_var_min)
                self.rate_N = self.rate_lam * self.rate_N + n
                self.rate_T = self.rate_lam * self.rate_T + max(dtw, 1e-3)
                self.rate = min(max(self.rate_N / self.rate_T, 50.0), 150.0)
        if self.K > 1:
            lats = self.lats
            S1, P1, D1 = _mm_predict(Ht, Hs, Hv, nh, lats, float(self.t_buf), float(self.t), n, self.S, self.PP,
                                     self.D, float(c_tyre), self.Qd, self.Pmax)
        else:
            S1, P1, D1 = s_pred[None, :].copy(), P_pred[None, :, :].copy(), np.array([d1])
        dt = n * 0.01
        self.D = D1
        self.t = self.t + dt
        t = self.t
        q = self._quality(nval, ninl, nsmall)
        self.free_frac = nfree / max(nval, 1)
        self.free_unexpl = (nfree - nsmall) / max(nval, 1)
        jump = math.hypot(s_new[0] - s_pred[0], s_new[1] - s_pred[1])
        self.q = q
        self.last_info = (nval, ninl, rms, it, jump)
        if q >= self.q_bad and jump < self.jump_gate:
            if self.K > 1:
                try:
                    S2, P2, J = _mm_update(S1, P1, p0, A, bq)
                    # unselected hypotheses that diverged (non-finite, or speed > 1 m/s from the selected one) are
                    # re-anchored on the selected one
                    bad = ~np.isfinite(S2).all(axis=1) | ~np.isfinite(P2).all(axis=(1, 2)) | \
                        (np.abs(S2[:, 3] - s_new[3]) > 1.0) | ~np.isfinite(J)
                    if bad.any():
                        S2[bad] = s_new
                        P2[bad] = P_new
                        J = np.where(bad, np.max(np.where(bad, -np.inf, J)) if (~bad).any() else 0.0, J)
                        self.n_reanchor = getattr(self, 'n_reanchor', 0) + int(bad.sum())
                except np.linalg.LinAlgError:
                    S2, P2, J = np.tile(s_new, (self.K, 1)), np.tile(P_new, (self.K, 1, 1)), np.zeros(self.K)
                    self.n_reanchor = getattr(self, 'n_reanchor', 0) + self.K
                S2[j], P2[j] = s_new, P_new          # the selected hypothesis: the full (iterated, robust) update
                if s_new[3] > self.lat_vmin and n <= 10:
                    # evidence only while driving (a standing car tells nothing about the delay) and over short gaps
                    lam = math.exp(-dt / self.lat_tau)
                    self.Jsum = lam * self.Jsum + (J - J.min())
                    k = int(np.argmin(self.Jsum))
                    if self.Jsum[k] < self.Jsum[j] - self.lat_hyst:
                        self.j = k
                        self.t_lat_est = float(self.lats[k])
                self.S, self.PP = S2, P2
            else:
                self.S, self.PP = s_new[None, :].copy(), P_new[None, :, :].copy()
            self.n_bad = 0
            self.rms_ema += self.sig_alpha * (rms - self.rms_ema)
            self.lost = False
        else:
            # bad scan / tracking lost: keep the prediction (dead reckoning) and search again when it persists
            self.S, self.PP = S1, P1
            self.n_bad += 1
            if self.n_bad >= self.lost_n or q < self.q_lost:
                self._select()
                self._relocalise(t, scan)
        self._select()
        s = self.s
        return s[0], s[1], s[2], s[3], s[4]

    def _relocalise(self, t, scan):
        pz, q = self.local_search(scan, self.s[:3])
        if q < self.q_ok and t - self.t_glob >= self.retry_global:
            self.t_glob = t
            pg, qg = self.global_search(scan)
            if pg is not None and qg > q:
                pz, q = pg, qg
        self.lost = q < self.q_ok
        self.n_reloc = getattr(self, 'n_reloc', 0) + 1
        if not self.lost:
            for k in range(self.K):
                self.S[k, :3] = pz
                P = self.PP[k]
                P[:3, :] = 0.0
                P[:, :3] = 0.0
                P[0, 0] = P[1, 1] = self.p0_xy ** 2
                P[2, 2] = self.p0_yaw ** 2
            self._select()
            self.n_bad = 0
        self.q = q


# ---------------------------------------------------------------------------------------------------- driver wrapper
def _load_kwargs(profile):
    if profile is None:
        return {}
    if isinstance(profile, dict):
        return dict(profile)
    path = profile if os.path.isabs(profile) else os.path.join(AVOID, profile)
    return json.load(open(path))


class LidarLocDriver:
    """inner driver (default: FAST = cmpc_berlin_tt:CorridorMPC + profiles/berlin_tt.json) fed with the LiDAR estimate.

    kwargs:
      inner         'module:Class' of the driver to wrap
      profile       json file (relative to ~/avoid) or dict with the inner driver's kwargs
      inner_kwargs  extra / overriding inner kwargs
      init          where the ONE initial pose per reset() comes from when set_initial_pose() was not called:
                    'gt' (x, y, yaw of the first act() after reset), 'global' (global scan search, no outside pose),
                    'start' (the map's start pose = raceline point 0, refined by a local search)
      forbid_gt     raise if act() would read its GT arguments (use with init 'global' / 'start' / set_initial_pose)
      shadow        evaluation only: the inner driver gets GT, the estimator runs alongside (tuning / replay)
      eval_log      evaluation only: record estimate - GT after the command is computed
      loc           LidarLocalizer parameters
    """

    def __init__(self, inner='drivers.cmpc_berlin_tt:CorridorMPC', profile='profiles/berlin_tt.json',
                 inner_kwargs=None, init='gt', forbid_gt=False, shadow=False, eval_log=False, record=False,
                 loc=None, map_name=None, use_inner_c=True, init_force=10, **extra):
        kw = _load_kwargs(profile)
        kw.update(inner_kwargs or {})
        kw.update(extra)                     # unknown keys go to the inner driver (e.g. map_name from run_avoid.sh)
        if map_name is not None:
            kw['map_name'] = map_name
        loc = dict(loc or {})
        self.lat_own = loc.get('lat_mode', LidarLocalizer.DEFAULTS['lat_mode']) == 'mm' and not shadow
        if self.lat_own:
            # one latency estimate: the localizer's (scan evidence); the inner driver's own estimator compared its model
            # with the measured speed, which is now the localizer's model-driven estimate (it locked on its prior)
            kw['lat_est'] = False
        self.map_name = kw.get('map_name', 'hyhall12')
        mod, cls = inner.split(':')
        self.inner = getattr(importlib.import_module(mod), cls)(**kw)
        self.loc = LidarLocalizer(self.map_name, **loc)
        if init not in ('gt', 'global', 'start'):
            raise ValueError(f'init must be gt, global or start, not {init}')
        if forbid_gt and init == 'gt':
            raise ValueError('forbid_gt=True needs init global / start (or set_initial_pose before every act)')
        self.init, self.forbid_gt, self.shadow = init, forbid_gt, shadow
        self.eval_log, self.record = eval_log, record
        self.use_inner_c = use_inner_c
        self.init_force = init_force
        self.pending_pose = None
        self.n_gt_reads = 0
        self.n_resets = 0
        self.stats = dict(err=[], loc_ms=[], q=[], reloc=0, lost_calls=0, init_q=[], init_ms=[], tlat=[], c=[], free=[],
                          dobs=[], boxes=0, phantom=0, phantom_calls=0, calls=0, dup=0)
        self.rec = []
        INSTANCES.append(self)
        self.reset()

    # ---- operator interface
    def reset(self):
        self.inner.reset()
        self.loc.reset()
        self.need_init = True
        self.init_tries = 0
        self.t0 = None
        self.last_cmd = None
        self._issued = False
        self.n_resets += 1

    def set_initial_pose(self, x, y, yaw):
        """operator's pose (/initialpose): the next act() starts the estimate here (call after reset())"""
        self.pending_pose = (float(x), float(y), float(yaw))
        self.need_init = True
        self.init_tries = 0

    @property
    def gt_free(self):
        """True: this driver never reads GT (avoid_node then passes NaN and does not wait for GT messages)"""
        return self.forbid_gt

    @property
    def v_est(self):
        return float(self.loc.s[3]) if self.loc.s is not None else math.nan

    def _read_gt_once(self, x, y, yaw):
        """the ONLY place where act() reads its GT arguments: the initial pose after reset() when no operator pose was
        given and init == 'gt'"""
        if self.forbid_gt:
            raise RuntimeError('LidarLocDriver: GT read requested with forbid_gt=True')
        self.n_gt_reads += 1
        return float(x), float(y), float(yaw)

    # ---- per call
    def _initialise(self, t, scan, x, y, yaw):
        t_a = time.perf_counter()
        if self.pending_pose is not None:
            pose0, src = self.pending_pose, 'operator'
        elif self.init == 'gt':
            pose0, src = self._read_gt_once(x, y, yaw), 'gt'
        elif self.init == 'start':
            pose0, src = self.loc.map.start, 'start'
        else:
            pose0, src = None, 'global'
        self.pending_pose = None
        ok, q = self.loc.initialise(t, scan, pose0)
        if not ok and pose0 is not None:
            self.pending_pose = pose0       # nothing fits this scan (stale scan after a teleport?): retry with the next
        self.stats['init_q'].append(round(float(q), 3))
        self.stats['init_ms'].append(round((time.perf_counter() - t_a) * 1e3, 1))
        self.init_src = src
        return ok

    def act(self, scan, x, y, yaw, v, yaw_rate, t):
        t = float(t)
        if self.t0 is None:
            self.t0 = t
        tr = t - self.t0                   # relative time (wall-clock stamps are ~1.7e9 s)
        t_a = time.perf_counter()
        if self.need_init:
            self.init_tries += 1
            if not self._initialise(tr, scan, x, y, yaw):
                pose0 = self.pending_pose
                if pose0 is not None and self.init_tries >= self.init_force:
                    # the operator's pose does not pass the checks for init_force scans (e.g. the car stands right in
                    # front of an obstacle): take it as given (/initialpose is authoritative); tracking refines it
                    self.loc.init_pose(tr, pose0[0], pose0[1], pose0[2], scan, refine=False)
                    self.pending_pose = None
                    self.stats['init_forced'] = self.stats.get('init_forced', 0) + 1
                    print(f'LidarLocDriver: initial pose {pose0} taken without a scan match after {self.init_tries} '
                          f'scans', file=sys.stderr, flush=True)
                else:
                    if self.init_tries % 50 == 0:
                        print(f'LidarLocDriver: no pose fits the scan yet ({self.init_tries} scans)', file=sys.stderr,
                              flush=True)
                    return 0.0, 0.0              # standing car: wait for a scan that fits
            self.need_init = False
            est = (self.loc.s[0], self.loc.s[1], self.loc.s[2], 0.0, 0.0)
        else:
            t_lat = getattr(self.inner, 't_lat', self.loc.lat0)
            c = getattr(self.inner, 'c_tyre', 1.0) if self.use_inner_c else 1.0
            try:
                est = self.loc.step(tr, scan, t_lat, c)
            except (np.linalg.LinAlgError, ValueError, ZeroDivisionError) as exc:
                # numerical failure (never seen after v6): restart the filter around the current estimate
                s_keep = self.loc.s.copy() if self.loc.s is not None else None
                self.stats['loc_exc'] = self.stats.get('loc_exc', 0) + 1
                if self.stats['loc_exc'] <= 3:
                    print(f'LidarLocDriver: estimator exception {exc!r}: filter restarted', file=sys.stderr, flush=True)
                if s_keep is None or not np.isfinite(s_keep).all():
                    self.need_init = True
                    self.pending_pose = None
                    return self.last_cmd if self.last_cmd is not None else (0.0, 0.0)
                self.loc._set_all(s_keep, self.loc.P0.copy(), self.loc.delta)
                self.loc.t_wall = tr
                est = (s_keep[0], s_keep[1], s_keep[2], s_keep[3], s_keep[4])
            if self.loc.lost:
                self.stats['lost_calls'] += 1
            if self.loc.n_last == 0 and self.last_cmd is not None and not self.shadow:
                # the same scan again (no physics step since the last call): nothing new, keep the command
                self.stats['dup'] += 1
                self._issued = False
                return self.last_cmd
        loc_ms = (time.perf_counter() - t_a) * 1e3
        self.stats['loc_ms'].append(loc_ms)
        self.stats['q'].append(self.loc.q)
        self.stats['free'].append((getattr(self.loc, 'free_frac', 0.0), getattr(self.loc, 'free_unexpl', 0.0)))
        if self.lat_own:
            self.inner.t_lat = self.loc.t_lat_est
        if self.shadow:
            steer, speed = self.inner.act(scan, x, y, yaw, v, yaw_rate, t)          # evaluation only
            self.loc.add_command(self.loc.t, steer, speed)
        else:
            # the inner driver runs on the localizer's simulator clock (its actuator / delay models count 10 ms steps)
            steer, speed = self.inner.act(scan, *est, self.loc.t)
            self.loc.add_command(self.loc.t, steer, speed)
        self.last_cmd = (steer, speed)
        self._issued = True
        # ---- evaluation only (after the command): estimate - GT
        if self.eval_log:
            ey = math.atan2(math.sin(est[2] - yaw), math.cos(est[2] - yaw))
            self.stats['err'].append((tr, est[0] - x, est[1] - y, ey, est[3] - v, est[4] - yaw_rate, v))
            self.stats['tlat'].append(getattr(self.inner, 't_lat', float('nan')))
            self.stats['c'].append(getattr(self.inner, 'c_tyre', float('nan')))
            self._eval_obstacles(x, y)
        if self.record:
            self.rec.append((t, np.asarray(scan, dtype=np.float32).copy(), float(x), float(y), float(yaw), float(v),
                             float(yaw_rate), float(steer), float(speed), getattr(self.inner, 't_lat', float('nan')),
                             getattr(self.inner, 'c_tyre', float('nan'))))
        self.stats['reloc'] = getattr(self.loc, 'n_reloc', 0)
        return steer, speed

    def _eval_obstacles(self, x, y):
        """evaluation only: distance from the true pose to the nearest real obstacle, and the inner driver's obstacle
        boxes whose centre is more than 0.6 m from every real obstacle (phantoms)"""
        obs = EVAL_OBSTACLES() if EVAL_OBSTACLES is not None else []
        d = min((math.hypot(o[0] - x, o[1] - y) for o in obs), default=math.inf)
        self.stats['dobs'].append(d)
        self.stats['calls'] += 1
        dp = getattr(self.inner, 'dp_res', None)
        if dp is None or not dp[0] or not hasattr(self.inner, '_xy'):
            return
        n_ph = 0
        for b in dp[0]:
            bx, by = self.inner._xy(np.array([0.5 * (b[0] + b[1])]), np.array([0.5 * (b[2] + b[3])]))
            if min((math.hypot(o[0] - float(bx[0]), o[1] - float(by[0])) for o in obs), default=math.inf) > 0.6:
                n_ph += 1
        self.stats['boxes'] += len(dp[0])
        self.stats['phantom'] += n_ph
        self.stats['phantom_calls'] += n_ph > 0

    def note_applied(self, steer, speed, a_cmd='implied'):
        """DART / DAgger hook: tell the expert which command was ACTUALLY sent after the last act() (it differs from
        the one act() returned).  Call once after every act(), before the next one.  Rewrites
          * the last entry of the localizer's command history (its process model: servo, speed controller, latency
            hypotheses),
          * the last entry of the inner driver's command history (inner.hist: servo / sideslip / latency / tyre models),
          * the inner driver's a_cmd (jerk-limit state): a_cmd='implied' (default) = the acceleration the applied speed
            command produces under the inner's command law (hold-compensated P-controller inverse, as
            tln/berlin/collect.py sync_expert), 'expert' = keep the expert's own, or a number.
        Uses no ground truth (the estimated speed only where the expert's speed command was clipped).  Do not set
        attributes on the wrapper itself (driver.a_cmd = ... would not reach the inner driver): use this hook or
        driver.inner."""
        steer, speed = float(steer), float(speed)
        if self.last_cmd is None:
            return                          # nothing issued yet (initialising): the car stands
        se, ve = self.last_cmd
        if self._issued:
            if self.loc.Hs:
                self.loc.Hs[-1] = steer
                self.loc.Hv[-1] = speed
            inn = self.inner
            if getattr(inn, 'hist', None):
                h = inn.hist[-1]
                inn.hist[-1] = (h[0], steer, speed)
            if a_cmd == 'expert' or not hasattr(inn, 'a_cmd'):
                pass
            elif a_cmd != 'implied':
                inn.a_cmd = min(max(float(a_cmd), -A_MAX), A_MAX)
            elif speed != ve:
                T = float(getattr(inn, 't_period', 0.02))

                def gain(kp):               # speed offset per m/s^2 of the inner's command law
                    return T / (1.0 - math.exp(-kp * T)) if getattr(inn, 'hold_comp', True) else 1.0 / kp
                a_e = float(inn.a_cmd)
                clipped = ve <= 0.0 or ve >= getattr(inn, 'v_max', 1e9) + 2.0 - 1e-9
                v_est = float(self.loc.s[3]) if self.loc.s is not None else 0.0
                v_at = max(v_est, 0.0) if clipped else ve - a_e * gain(KP_ACC if a_e >= 0 else KP_BRK)
                dv = speed - v_at
                inn.a_cmd = min(max(dv / gain(KP_ACC if dv >= 0 else KP_BRK), -A_MAX), A_MAX)
        elif self.loc.s is not None:
            # the last act() repeated a command without calling the inner driver (identical scan): the applied one
            # still enters the process model
            self.loc.add_command(self.loc.t, steer, speed)
        self.last_cmd = (steer, speed)
        self.n_applied = getattr(self, 'n_applied', 0) + 1

    def __getattr__(self, name):
        # attributes of the inner driver (t_lat, c_tyre, ...) for tools that inspect the driver
        if name == 'inner' or name.startswith('__'):
            raise AttributeError(name)
        return getattr(self.inner, name)

    def loc_summary(self, skip_s=0.0):
        """localisation error statistics (evaluation only)"""
        out = dict(gt_reads=self.n_gt_reads, resets=self.n_resets, reloc=self.stats['reloc'],
                   reanchor=getattr(self.loc, 'n_reanchor', 0), loc_exc=self.stats.get('loc_exc', 0),
                   init_forced=self.stats.get('init_forced', 0),
                   lost_calls=self.stats['lost_calls'], init_q=self.stats['init_q'][:5], init_ms=self.stats['init_ms'][:5])
        lm = np.array(self.stats['loc_ms'])
        if lm.size:
            out.update(loc_ms_mean=round(float(lm.mean()), 3), loc_ms_p99=round(float(np.percentile(lm, 99)), 3),
                       loc_ms_max=round(float(lm.max()), 1))
        qq = np.array(self.stats['q'])
        if qq.size:
            out.update(q_min=round(float(qq.min()), 3), q_p1=round(float(np.percentile(qq, 1)), 3),
                       free_max=round(float(np.max(np.array(self.stats['free'])[:, 0])), 3),
                       unexpl_max=round(float(np.max(np.array(self.stats['free'])[:, 1])), 3))
        e = np.array(self.stats['err'])
        if e.size:
            e = e[e[:, 0] >= skip_s]
            pos = np.hypot(e[:, 1], e[:, 2])
            out.update(n=int(e.shape[0]),
                       pos_rms=float(np.sqrt(np.mean(pos ** 2))), pos_max=float(pos.max()),
                       yaw_rms=float(np.sqrt(np.mean(e[:, 3] ** 2))), yaw_max=float(np.abs(e[:, 3]).max()),
                       v_rms=float(np.sqrt(np.mean(e[:, 4] ** 2))), v_max=float(np.abs(e[:, 4]).max()),
                       r_rms=float(np.sqrt(np.mean(e[:, 5] ** 2))), r_max=float(np.abs(e[:, 5]).max()),
                       yaw_bias=float(np.mean(e[:, 3])), v_bias=float(np.mean(e[:, 4])))
            tl = np.array(self.stats['tlat'])
            if tl.size:
                out['tlat_mean'] = float(np.nanmean(tl))
            do = np.array(self.stats['dobs'])
            if do.size == len(self.stats['err']):
                do = do[np.array(self.stats['err'])[:, 0] >= skip_s]
                for lim in (0.8, 1.5):
                    s = do < lim
                    if s.any():
                        out[f'pos_rms_obs{lim}'] = float(np.sqrt(np.mean(pos[s] ** 2)))
                        out[f'pos_max_obs{lim}'] = float(pos[s].max())
                        out[f'yaw_max_obs{lim}'] = float(np.abs(e[s, 3]).max())
                        out[f'n_obs{lim}'] = int(s.sum())
            out.update(calls=self.stats['calls'], boxes=self.stats['boxes'], phantom=self.stats['phantom'],
                       phantom_calls=self.stats['phantom_calls'])
            # by motion state (true speed): reversing, creeping, standing; and standing / creeping near an obstacle
            vt = e[:, 6]
            near = do < 0.8 if do.size == e.shape[0] else np.zeros(e.shape[0], dtype=bool)
            cls = dict(rev=vt < -0.05, creep=(vt >= 0.05) & (vt < 1.0), stand=np.abs(vt) < 0.05,
                       slow_near=(np.abs(vt) < 1.0) & near)
            for k, sel in cls.items():
                if sel.any():
                    out[f'n_{k}'] = int(sel.sum())
                    out[f'pos_rms_{k}'] = float(np.sqrt(np.mean(pos[sel] ** 2)))
                    out[f'pos_max_{k}'] = float(pos[sel].max())
                    out[f'yaw_max_{k}'] = float(np.abs(e[sel, 3]).max())
                    out[f'v_max_{k}'] = float(np.abs(e[sel, 4]).max())
        return out
