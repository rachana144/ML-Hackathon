"""HMM / Viterbi map-matching on top of the candidate classifier.

Model (Newson & Krumm 2009 style, with learned emissions)
  hidden state at GPS point i : one of its candidate arcs (directed), with the point's
                                projection position s on that arc (metres from the arc start)
  emission     log P(z_i | a) : log(classifier probability + eps)
  transition   log P(a -> b)  : -| route_dist(a,s_a -> b,s_b) - gps_dist(z_i, z_i+1) | / beta
     route_dist = driving distance along the DIRECTED network:
        same arc     : s_b - s_a if >= 0; backward < BACK_TOL m = 0 m (stationary jitter); more = unreachable
        other arc    : (len_a - s_a) + shortest_path(to(a) -> from(b)) + s_b   (Dijkstra, arc lengths)
     unreachable  : finite penalty, worse than any reachable transition (decoding never fails)
  segments     : sequence is cut at GPS gaps > gap_s seconds and at points without candidates;
                 each segment is decoded independently (no transition is trusted across a cut).
Decoding is exact Viterbi in log space with back-pointers.

API usage:
    from hmm import decode_trip
    arcs = decode_trip(trip, idx, cands)   # np.ndarray, one arc_id per cleaned point, -1 = none
    (trip = common.load_trip_dir(...), idx = clean_track(trip)[0], cands = CandidateScorer output)
"""
from __future__ import annotations

import heapq
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS, clean_track, point_segment

EPS = 1e-6          # emission floor: log(0) would forbid a state forever
BACK_TOL = 25.0     # metres of backwards movement on the same arc tolerated as GPS jitter
V_MAX = 60.0        # m/s, Dijkstra search limit = dt * V_MAX + 200 m
LIMIT_PAD = 200.0

def _load_defaults():
    """Tuned values from results/hmm/hmm_params.json (written by run_hmm.py tune on TRAINING trips)."""
    d = dict(beta=0.5, gap_s=300.0, top_k=10)
    try:
        import json
        prm = json.loads((RESULTS / "hmm" / "hmm_params.json").read_text())
        d.update({k: prm[k] for k in ("beta", "gap_s", "top_k") if k in prm})
    except (OSError, ValueError):
        pass
    return d


DEFAULTS = _load_defaults()


class RoadGraph:
    """Directed arc graph of one trip with a cached, distance-limited Dijkstra per source node."""

    def __init__(self, trip):
        self.arcs = trip.arcs
        a = trip.arcs
        xy = trip.nodes_xy
        self.arc_len = np.hypot(*(xy[a[:, 1]] - xy[a[:, 0]]).T)
        order = np.argsort(a[:, 0], kind="stable")
        self.out_to = a[order, 1]
        self.out_w = self.arc_len[order]
        self.indptr = np.searchsorted(a[order, 0], np.arange(len(trip.nodes) + 1))
        self._cache: dict[int, tuple[float, dict]] = {}

    def sp_from(self, src: int, limit: float) -> dict:
        hit = self._cache.get(src)
        if hit is not None and hit[0] >= limit:
            return hit[1]
        dist = {src: 0.0}
        pq = [(0.0, src)]
        done = set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in done:
                continue
            done.add(u)
            for k in range(self.indptr[u], self.indptr[u + 1]):
                v = int(self.out_to[k]); nd = d + self.out_w[k]
                if nd <= limit and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        self._cache[src] = (limit, dist)
        return dist


def _states(g: pd.DataFrame, graph: RoadGraph, trip, row: int):
    arc = g.arc_id.to_numpy().astype(int)
    a = graph.arcs[arc]
    A, B = trip.nodes_xy[a[:, 0]], trip.nodes_xy[a[:, 1]]
    p = trip.track_xy[row]
    _, t = point_segment(p[0], p[1], A[:, 0], A[:, 1], B[:, 0], B[:, 1])
    s = t * graph.arc_len[arc]
    loge = np.log(g.prob.to_numpy().astype(float) + EPS)
    return arc, s, loge


def transition_logp(graph: RoadGraph, arc_a, s_a, arc_b, s_b, gps_d: float, dt: float, beta: float):
    """log P for all (prev, cur) pairs -> matrix (len(arc_a), len(arc_b))."""
    limit = max(dt, 1.0) * V_MAX + LIMIT_PAD
    unreach = -(limit + gps_d) / beta - 10.0          # strictly worse than any reachable option
    T = np.empty((len(arc_a), len(arc_b)))
    from_b = graph.arcs[arc_b, 0]
    for i, (a, sa) in enumerate(zip(arc_a, s_a)):
        rest = graph.arc_len[a] - sa
        sp = graph.sp_from(int(graph.arcs[a, 1]), limit)
        spd = np.fromiter((sp.get(int(n), np.inf) for n in from_b), float, len(from_b))
        route = rest + spd + s_b
        same = arc_b == a
        if same.any():
            d = s_b[same] - sa
            # forward along the arc = d metres driven; small backward motion can only be "stood still
            # + GPS jitter", so it counts as 0 m driven (cost |0 - gps_d|/beta: cheap when stopped,
            # expensive when moving - this is what stops the decoder sliding backwards along a twin arc)
            route[same] = np.where(d >= 0, d, np.where(d > -BACK_TOL, 0.0, np.inf))
        route[route > limit] = np.inf
        lp = -np.abs(route - gps_d) / beta
        lp[~np.isfinite(route)] = unreach
        T[i] = lp
    return T


def viterbi(emis: list[np.ndarray], trans: list[np.ndarray]) -> np.ndarray:
    """emis[k]: (n_k,) log emissions; trans[k]: (n_k, n_{k+1}) log transitions. Returns best state index per step."""
    delta = emis[0].copy()
    back = []
    for k, T in enumerate(trans):
        scores = delta[:, None] + T                 # (n_k, n_{k+1})
        bp = scores.argmax(axis=0)
        delta = scores[bp, np.arange(T.shape[1])] + emis[k + 1]
        back.append(bp)
    path = np.empty(len(emis), dtype=int)
    path[-1] = int(delta.argmax())
    for k in range(len(trans) - 1, -1, -1):
        path[k] = back[k][path[k + 1]]
    return path


def decode_trip(trip, idx, cands: pd.DataFrame, beta: float | None = None, gap_s: float | None = None,
                top_k: int | None = None, graph: RoadGraph | None = None) -> np.ndarray:
    """API entry point (api/main.py).
    trip  : common.Trip from load_trip_dir (needs .arcs, .nodes_xy, .track_xy, .track)
    idx   : cleaned GPS row indices from clean_track(trip)
    cands : CandidateScorer.score_trip_dir() output (needs point_idx, arc_id, prob)
    returns np.ndarray of arc_id, one per entry of idx, -1 where no prediction."""
    return decode_trip_df(trip, cands, beta, gap_s, top_k, graph, idx=idx).arc_id.to_numpy()


def decode_trip_df(trip, cands: pd.DataFrame, beta: float | None = None, gap_s: float | None = None,
                   top_k: int | None = None, graph: RoadGraph | None = None, idx=None) -> pd.DataFrame:
    """Same decoding, returns one row per cleaned GPS point: point_idx, arc_id (-1 = none), segment."""
    beta = DEFAULTS["beta"] if beta is None else beta
    gap_s = DEFAULTS["gap_s"] if gap_s is None else gap_s
    top_k = DEFAULTS["top_k"] if top_k is None else top_k
    if idx is None:
        idx, _ = clean_track(trip)
    idx = np.asarray(idx, dtype=np.int64)
    if cands is None or len(cands) == 0:
        return pd.DataFrame({"point_idx": idx, "arc_id": np.full(len(idx), -1), "segment": np.full(len(idx), -1)})
    graph = graph or RoadGraph(trip)
    c = cands[["point_idx", "arc_id", "prob"]]
    if top_k:
        c = c.sort_values(["point_idx", "prob"], ascending=[True, False], kind="mergesort")
        c = c[c.groupby("point_idx").cumcount() < top_k]
    groups = {p: g for p, g in c.groupby("point_idx", sort=False)}
    row_of = {int(p): i for i, p in enumerate(idx)}  # point_idx == original track row
    t = trip.track[:, 2]
    out_arc = np.full(len(idx), -1, dtype=np.int64)
    seg_id = np.full(len(idx), -1, dtype=np.int64)
    seg, cur = 0, []

    def flush(points):
        nonlocal seg
        if not points:
            return
        st = [_states(groups[p], graph, trip, p) for p in points]
        trans = []
        for k in range(len(points) - 1):
            p0, p1 = points[k], points[k + 1]
            gps_d = float(np.hypot(*(trip.track_xy[p1] - trip.track_xy[p0])))
            trans.append(transition_logp(graph, st[k][0], st[k][1], st[k + 1][0], st[k + 1][1],
                                         gps_d, float(t[p1] - t[p0]), beta))
        path = viterbi([s[2] for s in st], trans)
        for p, s, j in zip(points, st, path):
            out_arc[row_of[p]] = s[0][j]; seg_id[row_of[p]] = seg
        seg += 1

    prev = None
    for p in idx:
        p = int(p)
        if p not in groups:                       # no candidate: cut, no prediction
            flush(cur); cur = []; prev = None; continue
        if prev is not None and t[p] - t[prev] > gap_s:
            flush(cur); cur = []
        cur.append(p); prev = p
    flush(cur)
    return pd.DataFrame({"point_idx": idx, "arc_id": out_arc, "segment": seg_id})


def decode_trip_dir(trip_dir, scorer=None, **kw):
    """Convenience for the API: raw trip folder -> (classifier candidates, HMM path)."""
    from inference import CandidateScorer
    scorer = scorer or CandidateScorer()
    cands = scorer.score_trip_dir(trip_dir)
    return cands, decode_trip(scorer.last_trip, scorer.last_idx, cands, **kw)
