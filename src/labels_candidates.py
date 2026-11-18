"""Per-point label derivation (route alignment) and candidate generation.

.route is the PATH (sequence of arc ids), not one label per GPS point
(len(route) != len(track) in every trip), so per-point labels are derived by
aligning the time-ordered GPS points to the route in order.

Two aligners are implemented:
  * greedy_align  – the pointer method from the team brief: keep a pointer on the
                    route; for each point pick the closest arc among
                    route[p : p+lookahead]; pointer only moves forward.
  * dp_align      – globally optimal monotone alignment (dynamic programming):
                    choose non-decreasing route indices j_1 <= j_2 <= ... <= j_N
                    minimising sum_i dist(point_i, route[j_i]). Same constraint as
                    the pointer method but cannot get "stuck" or jump too early.
We use dp_align for labels and report agreement with greedy_align.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from common import point_segment


def route_dist_matrix(trip, idx):
    a = trip.arcs[trip.route]
    A_, B_ = trip.nodes_xy[a[:, 0]], trip.nodes_xy[a[:, 1]]
    xy = trip.track_xy[idx]
    D = np.empty((len(xy), len(a)), dtype=np.float32)
    for s in range(0, len(a), 256):
        d, _ = point_segment(xy[:, None, 0], xy[:, None, 1], A_[None, s:s+256, 0], A_[None, s:s+256, 1],
                             B_[None, s:s+256, 0], B_[None, s:s+256, 1])
        D[:, s:s+256] = d
    return D


def dp_align(D):
    """D: (N points, M route arcs). Returns route index per point (non-decreasing)."""
    N, M = D.shape
    back = np.empty((N, M), dtype=np.int32)
    prev = D[0].astype(np.float64)
    back[0] = np.arange(M)
    for i in range(1, N):
        # prefix minimum of prev and its argmin
        pm = np.minimum.accumulate(prev)
        # argmin of prefix: index where prefix min was last updated
        upd = np.r_[True, prev[1:] < pm[:-1]]
        arg = np.maximum.accumulate(np.where(upd, np.arange(M), 0))
        back[i] = arg
        prev = pm + D[i]
    j = np.empty(N, dtype=np.int64)
    j[-1] = int(np.argmin(prev))
    for i in range(N - 1, 0, -1):
        j[i - 1] = back[i, j[i]]
    return j


def greedy_align(D, lookahead=5):
    N, M = D.shape
    p = 0
    out = np.empty(N, dtype=np.int64)
    for i in range(N):
        w = D[i, p:min(M, p + lookahead + 1)]
        p = p + int(np.argmin(w))
        out[i] = p
    return out


def derive_labels(trip, idx):
    D = route_dist_matrix(trip, idx)
    j_dp = dp_align(D)
    j_gr = greedy_align(D)
    true_arc = trip.route[j_dp]
    label_dist = D[np.arange(len(idx)), j_dp]
    nearest_route_dist = D.min(axis=1)
    info = {"agree_dp_greedy": float(np.mean(j_dp == j_gr)),
            "dp_mean_dist": float(label_dist.mean()), "greedy_mean_dist": float(D[np.arange(len(idx)), j_gr].mean()),
            "frac_label_dist_gt30": float(np.mean(label_dist > 30)),
            "frac_dp_not_nearest_route_arc": float(np.mean(label_dist > nearest_route_dist + 1e-3)),
            "route_arcs_with_no_point": int(len(trip.route) - len(np.unique(j_dp)))}
    return true_arc, j_dp, label_dist, info


class ArcIndex:
    """KD-tree over arc samples (every `step` m) for radius queries; exact
    point-to-segment distance is then computed for the retrieved arcs."""

    def __init__(self, trip, idx, R_max, step=10.0):
        a = trip.arcs
        A_, B_ = trip.nodes_xy[a[:, 0]], trip.nodes_xy[a[:, 1]]
        xy = trip.track_xy[idx]
        lo, hi = xy.min(0) - R_max - 1, xy.max(0) + R_max + 1
        box = ((np.minimum(A_, B_) <= hi).all(1) & (np.maximum(A_, B_) >= lo).all(1))
        self.arc_ids = np.flatnonzero(box)
        A, B = A_[self.arc_ids], B_[self.arc_ids]
        L = np.hypot(*(B - A).T)
        n = np.maximum(1, np.ceil(L / step).astype(int))
        owner = np.repeat(np.arange(len(A)), n + 1)
        frac = np.concatenate([np.linspace(0, 1, k + 1) for k in n])
        pts = A[owner] + frac[:, None] * (B - A)[owner]
        self.tree = cKDTree(pts)
        self.owner = owner
        self.A, self.B = A, B
        self.half_step = step / 2 + 0.01

    def query(self, xy, R):
        lists = self.tree.query_ball_point(xy, R + self.half_step)
        pi, ai = [], []
        for i, l in enumerate(lists):
            if l:
                u = np.unique(self.owner[l])
                pi.append(np.full(len(u), i)); ai.append(u)
        if not pi:
            return np.empty(0, int), np.empty(0, int), np.empty(0), np.empty(0)
        pi, ai = np.concatenate(pi), np.concatenate(ai)
        d, t = point_segment(xy[pi, 0], xy[pi, 1], self.A[ai, 0], self.A[ai, 1], self.B[ai, 0], self.B[ai, 1])
        m = d <= R
        return pi[m], self.arc_ids[ai[m]], d[m], t[m]
