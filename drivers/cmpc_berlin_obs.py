"""berlin with random static obstacles: drivers.cmpc_berlin_tt.CorridorMPC with profiles/berlin_obs.json.

The obstacle options (obstacle mode, frontier / em_frac / weave / slow envelopes, narrow-side closing, slow DP, checked
creep, proximity guard, low-speed manoeuvre, heading guard) were a subclass of cmpc_berlin_tt.CorridorMPC here, with
its own copies of act() and _plan_rest().  On 2026-11-19 they were merged into that class (EXTRA there, all off by
default; work/advice_fix/refactor).  The same day the low-speed manoeuvre became one arc search in place of the
back-off, the tight-margin retry and (in profiles/berlin_obs.json) creep, a bad plan is steered by the MPC instead of a
pure-pursuit law, and the obstacle profile turned the command-following check (fg_check) off
(work/advice_fix/behaviour, work/advice_fix/integrate; details and known gaps in the cmpc_berlin_tt docstring).
This module keeps the import path drivers.cmpc_berlin_obs:CorridorMPC (run_avoid.sh, lidar_loc profiles, TLN
collectors, subclasses) and the names other modules import from it.
"""
from drivers.cmpc_berlin_tt import (A_MAX, BODY_PTS, BOX_FRONT, BOX_HW, EXTRA, FRONT_PTS, KP_ACC, KP_BRK, LIDAR_X,
                                    STEER_MAX, WB, CorridorMPC, _cmd_sim, _fwd, _wrap, k_us)

__all__ = ['A_MAX', 'BODY_PTS', 'BOX_FRONT', 'BOX_HW', 'EXTRA', 'FRONT_PTS', 'KP_ACC', 'KP_BRK', 'LIDAR_X', 'STEER_MAX',
           'WB', 'CorridorMPC', '_cmd_sim', '_fwd', '_wrap', 'k_us']
