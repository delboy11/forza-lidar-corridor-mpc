"""berlin with random static obstacles: CorridorMPC = cmpc_berlin_tt (frozen) + obstacle fixes (work/berlin_obs/REPORT.md).

Subclass of drivers.cmpc_berlin_tt.CorridorMPC: the numba kernels and constants are imported from that module (its
numba cache stays valid); act() and _plan_rest() are copies with the additions below.  Every addition has a kwarg in
EXTRA; with all of them at their default ('off') the driver is bit-identical to cmpc_berlin_tt.
Profile: profiles/berlin_obs.json (= the time-trial profile berlin_tt.json + the obstacle additions).

Additions (mechanisms they fix: work/berlin_obs/REPORT.md, M1-M6):
  * obstacle mode (obs_mem < 0: per lap): on in the first lap, in every lap after a lap in which a box was seen and
    from the first box of a lap on; obs_kw = parameter overrides in that mode (t_react, sight envelope, low-speed
    margins, gap rule).  A clear track is driven with the time-trial parameters from the second lap on.
  * width-gated horizon frontier (v_hz, m_hz, w_front, hz_look, front_ramp): where the track near the horizon end is
    narrower than w_front, the car must be able to reach v_hz there (an obstacle chicane can appear at the horizon
    end and is not passable at speed); static ramped cap before it (M1).
  * em_frac: the closed-corridor stop envelope again with em_frac of the emergency deceleration (M1 / M2 tracking).
  * narrow-side closing in the DP (g_narrow, g_diff; part of the gap rule) (M3).
  * slow DP (v_slow, kcap_slow, kmax_slow, dp_adec_slow, slow_pre, slow_len, m_slow, slow_commit): when the DP finds no
    path, plan the curvature the car steers at v_slow (optionally releasing committed sides further than slow_commit)
    and slow down to v_slow before the tight part instead of stopping (M1 / M4).
  * checked creep and proximity guard (creep_slack, creep_chk, creep_m_obs, creep_m_wall, prox_v, prox_t): below
    prox_v the body points along the planned path, and during a creep / soft drive-on along the steering arc, must
    keep creep_m_obs / creep_m_wall from the measured obstacle cells / map walls, else the car stops (M4 / M6).
  * back-off (back_*): stopped without a drivable plan -> reverse at back_v < 0.5 m/s (kinematic regime of the
    simulator) with the steering of largest body clearance, growing distance, then retry with tight_kw (smaller
    corridor margins) until the place is passed (M4: infeasible hairpin chicanes).
  * heading guard (mu_max): stop when the heading error to the line exceeds mu_max (a car turning around).
  * weave cap (a_weave, v_weave_min, weave_pre): two consecutive boxes passed on opposite sides -> reach
    sqrt(a_weave / (4 dd / l^2)) by the end of the first box and hold it to the second (M7: weaves on the narrow
    straight planned by the coarse DP that the car could not brake for in time).
Known gaps (work/berlin_obs/REPORT.md): hairpin chicanes at s 23-25 can still end in a stall or, after long creeping /
back-off manoeuvres, in a low-speed touch (creep_m_obs / creep_m_wall 0.04 / 0.03 are small for 2 cm cells and a 4 cm
map); 50 ms command latency and pose noise were tested but not tuned for.
"""
import math
import sys

import numpy as np

import drivers.cmpc_berlin_tt as TT
from drivers.cmpc_berlin_tt import (A_MAX, BOX_FRONT, BOX_HW, KP_ACC, KP_BRK, LIDAR_X, STEER_MAX, WB, _cmd_sim, _fwd, _wrap,
                                    k_us)

# body points (CG frame, m) checked against the measured obstacle cells / map walls: the referee box (LiDAR +- 0.165 m
# along, +- 0.1016 m across, i.e. CG + 0.105 .. 0.435) densely, the rest of the car more coarsely
BODY_PTS = ((-0.30, 0.12), (-0.30, -0.12), (-0.30, 0.0), (-0.15, 0.12), (-0.15, -0.12), (0.0, 0.12), (0.0, -0.12),
            (0.105, 0.11), (0.105, -0.11), (0.2, 0.11), (0.2, -0.11), (0.3, 0.11), (0.3, -0.11), (BOX_FRONT, 0.11),
            (BOX_FRONT, -0.11), (BOX_FRONT, 0.0))
FRONT_PTS = tuple(p for p in BODY_PTS if p[0] >= 0.0)

EXTRA = dict(
    # obstacle mode: on from the start and whenever an obstacle box was seen within the last obs_mem metres of
    # driving; off after obs_mem metres without any box (a clear track is driven with the base parameters).
    # obs_kw: parameter overrides while the mode is on (scalars read per call only)
    obs_mem=0.0,            # 0: the mode is always on (obs_kw always applied); < 0: per lap: on in the first lap, in
                            # every lap after a lap with a box, and from the first box of a lap on
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
    # that finds itself above it can get back under it); vis_copy: the sight envelope again with this class's lag
    em_frac=1.0, vis_copy=False,
    # DP: a side of a box whose corridor (box .. wall, after margins) is narrower than g_narrow while the other side
    # is wider by g_diff is closed for the DP (the wide side is taken early instead of the narrow one).  <0: off
    g_narrow=-1.0, g_diff=0.25,
    # creep / soft emergency guard: no creep / soft drive-on while the plan needs more than creep_slack of corridor
    # slack within creep_dist of path.  <0: off
    creep_slack=-1.0, creep_dist=0.8,
    # checked creep (creep_chk > 0, with creep_slack >= 0): instead of forbidding the creep / soft drive-on, it is
    # allowed while the body (front / middle / rear corners) moved creep_chk m along the kinematic arc of the commanded
    # steering keeps creep_m_obs from the remembered obstacle cells (measured surface points, not the padded boxes)
    # and creep_m_wall from the map walls; otherwise the car stops (and may back off)
    creep_chk=0.0, creep_m_obs=0.04, creep_m_wall=0.03,
    # proximity guard (prox_v > 0): below prox_v m/s the planned path of the next max(creep_chk, prox_t v) m must keep
    # the body creep_m_obs from the measured obstacle cells and creep_m_wall from the walls, else the car stops (the
    # 3-point corridor of the MPC misses the box corners at large heading angles next to an obstacle corner)
    prox_v=0.0, prox_t=0.4,
    # heading guard (mu_max > 0): heading error to the line above mu_max rad in obstacle mode -> stop (the Frenet MPC
    # linearisation breaks down; a full-lock swerve once turned the car around); the back-off then straightens it
    mu_max=0.0,
    # weave cap (a_weave > 0): for two consecutive boxes passed on opposite sides, the S between them needs the
    # curvature 4 dd / l^2 (dd: lateral change of the body line, l: free length between the boxes); reach
    # sqrt(a_weave / kappa) (>= v_weave_min) by the end of the first box (planned from the boxes and DP sides, not
    # from the MPC path: the MPC assumes the braking the car then lags)
    a_weave=0.0, v_weave_min=1.0, weave_pre=0.3,
    # slow DP: when the DP finds no complete path, plan again with the curvature the car can steer at v_slow (DP
    # curvature cap kcap_slow, braking dp_adec_slow from now) before falling back to narrow gaps / a stop; the speed
    # profile then reaches v_slow at s_block_normal - slow_pre and keeps it for slow_len m.  v_slow < 0: off
    v_slow=-1.0, kcap_slow=1.1, kmax_slow=6, dp_adec_slow=5.0, slow_pre=2.0, slow_len=4.0, m_slow=0.1,
    slow_commit=None,       # [commit_d, commit_t] of a second slow-DP try that may release committed sides
    a_lat_slow=0.0,         # > 0: lateral acceleration of the slow DP's curvature limit (truly slow manoeuvres)
    # back-off: stopped for back_after s with no drivable plan (creep blocked) -> reverse at back_v (< 0.5 m/s: the
    # simulator's kinematic regime) for back_dist m, steering the heading back toward the line, while the body
    # (rear and front corners, back_chk m ahead along the reverse arc) keeps back_m_wall from the map walls and
    # back_m_obs from remembered obstacle cells; at most back_max times per place.  back_dist <= 0: off
    back_dist=0.0, back_grow=0.0, back_v=0.3, back_after=1.0, back_max=3, back_k=1.5, back_tmax=3.0, back_chk=0.15,
    back_m_wall=0.05, back_m_obs=0.08, back_ref_min=0.0,
    # after a back-off, until the car has passed the place: these overrides (smaller corridor margins; the proximity
    # guard / checked creep on the measured surfaces stay the safety net) -> the next attempt hugs the first obstacle
    tight_kw=None,
)


class CorridorMPC(TT.CorridorMPC):
    def __init__(self, **kw):
        ex = {k: kw.pop(k) for k in list(kw) if k in EXTRA}
        for k, v in EXTRA.items():
            setattr(self, k, ex.get(k, v))
        super().__init__(**kw)
        self.p.update({k: getattr(self, k) for k in EXTRA})
        self.obs_kw = dict(self.obs_kw or {})
        for k in self.obs_kw:
            if k not in self.p or k in EXTRA:
                raise KeyError(f'obs_kw: unknown or non-switchable parameter {k}')
        self.base_kw = {k: self.p[k] for k in self.obs_kw}
        self.obs_on = None
        self._set_mode(True)
        self._front_tables()
        self.tight_kw = dict(self.tight_kw or {})
        for k in self.tight_kw:
            if k not in self.p or k in EXTRA:
                raise KeyError(f'tight_kw: unknown or non-switchable parameter {k}')
        self.tight_save = None

    def _tight(self, on):
        if on and self.tight_save is None and self.tight_kw:
            self.tight_save = {k: getattr(self, k) for k in self.tight_kw}
            for k, v in self.tight_kw.items():
                setattr(self, k, v)
                self.p[k] = v
        elif not on and self.tight_save is not None:
            for k, v in self.tight_save.items():
                setattr(self, k, v)
                self.p[k] = v
            self.tight_save = None

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

    def reset(self):
        if getattr(self, 'tight_save', None) is not None:
            self._tight(False)
        super().reset()
        self.odo = 0.0              # metres driven (|v| dt) since the start
        self.odo_box = None         # odometer when a box was last seen
        self.lap_boxes = 0          # DP calls with boxes in this lap
        self.lap_clear = False
        if getattr(self, 'base_kw', None) is not None and self.obs_mem != 0.0:
            self._set_mode(True)    # after a reset (re-placed car): obstacle mode until a clear lap
        self.t_odo = None
        self.tr_extra = None
        self.slow_at = None
        self.weaves = []
        self.rev = None
        self.n_back = 0
        self.back_s = 0.0
        self._creep_blocked = False
        if not hasattr(self, 'n_back_tot'):
            self.n_back_tot = 0

    def _env(self, cum, k, v0, vt, m, frac=1.0):
        """speed envelope: reach vt at path length cum[k] - BOX_FRONT - m with the emergency deceleration (command lag
        included), vt beyond"""
        reach = cum[k] - BOX_FRONT - m - cum - v0 * (self.t_react + self.t_lag)
        tds = self.tab_dstop / frac
        ds_ = float(np.interp(vt, self.tab_v, tds))
        env = np.interp(np.maximum(reach, 0.0) + ds_, tds, self.tab_v)
        return np.where(reach <= 0.0, np.minimum(env, vt), env)

    def _speed_profile(self, st, d, mu, beta, kr, v0, scan, lx, ly, yaw, s_block, kyaw=None):
        vprof, vpred, acc, dpl, s_vis = super()._speed_profile(st, d, mu, beta, kr, v0, scan, lx, ly, yaw, s_block,
                                                                 kyaw)
        if not self.obs_on:
            return vprof, vpred, acc, dpl, s_vis
        cum = self._sp_cum if (self.jit and not self.vis_obs_stop) else np.concatenate([[0.0], np.cumsum(dpl)])
        env = np.full(cum.size, np.inf)
        if self.v_hz >= 0.0:
            gated = False
            if self.front_flag is not None:
                ii = (np.floor(np.mod(st, self.L) / self.fds).astype(np.int64)) % self.nf
                env = np.minimum(env, self.front_cap[ii])
                gated = not bool(self.front_flag[ii[0]])
            if not gated:
                env = np.minimum(env, self._env(cum, cum.size - 1, v0, self.v_hz, self.m_hz))
        if self.vis_copy or self.em_frac < 1.0:
            # the sight / closed-corridor envelopes again (lag t_react + t_lag, reduced deceleration for s_block)
            if s_vis is not None and self.vis_copy:
                kv = min(int(np.searchsorted(st, s_vis - 1e-9)), cum.size - 1)
                ev = self._env(cum, kv, v0, self.v_vis_safe, self.m_vis)
                if s_block is not None and s_vis >= s_block:
                    ev = np.full(cum.size, np.inf)
                env = np.minimum(env, ev)
            if s_block is not None:
                kb = min(int(np.searchsorted(st, s_block)), cum.size - 1)
                env = np.minimum(env, self._env(cum, kb, v0, 0.0, self.m_blk, self.em_frac))
        for s_w, s_w2, v_w in getattr(self, 'weaves', ()):
            k = min(int(np.searchsorted(st, s_w)), cum.size - 1)
            k2 = int(np.searchsorted(st, s_w2))
            if k2 > 0:
                e3 = self._env(cum, k, v0, v_w, -BOX_FRONT)
                e3[k2 + 1:] = np.inf
                env = np.minimum(env, e3)
        if self.v_slow > 0.0 and getattr(self, 'slow_at', None) is not None:
            s_t = self.slow_at - self.slow_pre
            k = min(int(np.searchsorted(st, s_t)), cum.size - 1)
            e2 = self._env(cum, k, v0, self.v_slow, self.m_slow - BOX_FRONT)
            e2[(st > s_t + self.slow_len)] = np.inf
            env = np.minimum(env, e2)
        if np.any(env < vprof):
            vl = np.minimum(vprof, env)
            vprof, vpred = _fwd(np.ascontiguousarray(vl), np.ascontiguousarray(dpl, dtype=np.float64), float(v0),
                                float(self.a_acc_eff), self.tab_emerg)
            acc = np.clip((vpred[1:] ** 2 - vpred[:-1] ** 2) / (2 * dpl), -A_MAX, A_MAX)
            self.vlim0 = min(self.vlim0, float(vl[0]))
        return vprof, vpred, acc, dpl, s_vis

    def _body_clr(self, x, y, yaw, dl, dist):
        """smallest clearance of the body points (rear / middle / front corners) to the map walls and to the
        remembered obstacle cells after reversing dist m with steering dl (kinematic arc; dist 0: current pose)"""
        sb = dist
        yw = yaw - sb * math.tan(dl) / WB
        ym = 0.5 * (yaw + yw)
        cx, cy = x - sb * math.cos(ym), y - sb * math.sin(ym)
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
            cw, co = self._body_clr(x, y, yaw, dl, -dist * i / 3)
            if cw < self.creep_m_wall or co < self.creep_m_obs:
                return False
        return True

    def _body_margin(self, x, y, yaw, dl, dist, ref):
        """smallest margin (m) over the reverse of dist m with steering dl between the body clearance and its floor:
        back_m_wall / back_m_obs, or, where the body already was closer at the start (ref), ~the start clearance"""
        fw = min(self.back_m_wall, max(ref[0] - 0.01, min(ref[0], 0.05)))
        fo = min(self.back_m_obs, max(ref[1] - 0.01, min(ref[1], 0.05)))
        m = math.inf
        for i in (1, 2, 3):
            cw, co = self._body_clr(x, y, yaw, dl, dist * i / 3)
            m = min(m, cw - fw, co - fo)
        return m

    def _body_clear(self, x, y, yaw, dl, dist, ref):
        return self._body_margin(x, y, yaw, dl, dist, ref) >= 0.0

    def _back_steer(self, x, y, yaw, mu, ref, prev=None):
        """reverse steering with the largest clearance margin among: heading correction (back_k mu), half of it,
        straight, the opposite ones and full lock; the previous choice is kept unless another is 1 cm better (no
        chattering: the servo needs ~0.25 s for a full swing).  None: nothing stays clear."""
        dl = min(max(self.back_k * mu, -STEER_MAX), STEER_MAX)
        if abs(dl) < 0.1:
            dl = 0.1 if dl >= 0 else -0.1
        cands = [min(max(f * dl, -STEER_MAX), STEER_MAX) for f in (1.0, 0.5, 0.0, -0.5, -1.0)] + [STEER_MAX, -STEER_MAX]
        best, bm = None, -math.inf
        for d_ in cands:
            m = self._body_margin(x, y, yaw, d_, self.back_chk, ref)
            if m > bm + 1e-4:
                best, bm = d_, m
        if prev is not None:
            mp = self._body_margin(x, y, yaw, prev, self.back_chk, ref)
            if mp >= 0.0 and mp >= bm - 0.01:
                return prev
        return best if bm >= 0.0 else None

    def _backoff(self, t, x, y, yaw, v, mu, s_c):
        R = self.rev
        if R is None:
            if not (self.obs_on and self.t_stop is not None and t - self.t_stop > self.back_after and abs(v) < 0.1
                    and self._creep_blocked and self.n_back < self.back_max):
                return None
            ref = self._body_clr(x, y, yaw, 0.0, 0.0)
            if min(ref[0] / max(self.back_m_wall, 1e-3), ref[1] / max(self.back_m_obs, 1e-3)) < self.back_ref_min:
                return None               # already too close to something to move blind (no rear view)
            dl = self._back_steer(x, y, yaw, mu, ref)
            if dl is None:
                return None
            self.rev = R = dict(t0=t, x0=x, y0=y, phase=0, dl=dl, ref=ref)
            if self.n_back == 0:
                self.back_s = s_c
            self.n_back += 1
            self.n_back_tot += 1
            self._tight(True)
        if R['phase'] == 0:
            moved = math.hypot(x - R['x0'], y - R['y0'])
            bd = self.back_dist * (1.0 + self.back_grow * (self.n_back - 1))
            dl = self._back_steer(x, y, yaw, mu, R['ref'], R['dl']) if moved < bd and t - R['t0'] < self.back_tmax else None
            if dl is not None:
                R['dl'] = dl
                return dl, -self.back_v
            R['phase'] = 1
            R['t1'] = t
        if v < -0.03 and t - R['t1'] < 1.0:
            return R['dl'], 0.8           # stop the reverse motion (the simulator brakes a reversing car gently)
        self.rev = None
        self.t_stop = t
        return None

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
            ln = max(b2[0] - b1[1] + 0.3, 0.3)
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

    def _dp_jit(self, s0, d0, chi0, boxes, v, w_h, relax, hard, gap):
        if self.g_narrow < 0.0 or not boxes or not self.obs_on or not gap:
            # (the narrow-side closing belongs to the gap rule: the fallbacks without it may use a narrow side)
            return super()._dp_jit(s0, d0, chi0, boxes, v, w_h, relax, hard, gap)
        # close the narrow side of a box when the other side is clearly wider (DP only; the corridor of the chosen
        # side is built from the real box)
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
        st, dpath, sides, s_block = super()._dp_jit(s0, d0, chi0, bx2, v, w_h, relax, hard, gap)
        # sides relative to the real boxes (the closed side moved the box centre)
        out = []
        for b, sd in zip(boxes, sides):
            out.append(sd)
        return st, dpath, out, s_block

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
        if self.obs_mem > 0.0:
            self._set_mode(self.odo < self.obs_mem if self.odo_box is None else self.odo - self.odo_box < self.obs_mem)
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
                self.odo_box = self.odo
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
        self._creep_blocked = bool(plan['bad'])
        if self.creep_slack >= 0.0 and slack is not None:
            near_ = cum <= self.creep_dist
            if np.any(near_) and float(np.max(slack[near_])) > self.creep_slack:
                self._creep_blocked = True
        if creep and self.creep_slack >= 0.0 and self._creep_blocked and self.creep_chk <= 0.0:
            creep = False             # the corridor ahead is too narrow for the body: stay stopped
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
        veto = self.mu_max > 0.0 and self.obs_on and speed > 0.0 and abs(mu0) > self.mu_max
        if not veto and self.prox_v > 0.0 and self.obs_on and speed > 0.0 and vv < self.prox_v and not plan['bad']:
            veto = not self._plan_clear(plan, max(self.creep_chk, self.prox_t * vv), shift_s)
        if (self.creep_chk > 0.0 and self.creep_slack >= 0.0 and self.obs_on and self._creep_blocked and speed > 0.0
                and vv < 1.5 and not veto):
            # checked creep / soft drive-on: only while the measured surfaces stay clear of the body
            veto = not self._fwd_clear(x, y, yaw, steer, max(self.creep_chk, 0.3 * vv))
        if veto:
            self._creep_blocked = True        # no drivable plan: the back-off may take over
            if True:
                speed = 0.0
                self.a_cmd = min(self.a_cmd, 0.0)
                if self.t_stop is None and vv < 0.15:
                    self.t_stop = t
        if self.back_dist > 0.0:
            if self.n_back and (wrapped or teleport or s_c - self.back_s > 1.5 or s_c < self.back_s - 3.0):
                self.n_back = 0
                self._tight(False)
            rc = self._backoff(t, x, y, yaw, v, mu0, s_c)
            if rc is not None:
                steer, speed = rc
                self.a_cmd = 0.0
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
            if mx <= self.soft_em and not (self.creep_slack >= 0.0 and mx > self.creep_slack and self.creep_chk <= 0.0):
                emergency = False
                vprof = np.minimum(vprof, self.soft_em_v)
        return dict(t=t, st=st, Xs=Xs, dts=dts, vprof=vprof, cum=cum, emergency=emergency, slack=slack, s_vis=s_vis,
                    bad=bad, dpath=dpath, acc=acc_mpc, failed=out is None)

