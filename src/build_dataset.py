"""Build the shared candidate dataset.

Outputs (data/processed/):
  features.parquet  one row per (trip, GPS point, candidate arc):
                    trip_id, partition, point_idx, arc_id, <features>, label
  points.parquet    one row per GPS point: trip_id, partition, point_idx, t, lon, lat,
                    true_arc, route_pos, label_dist, n_cands, true_in_cands
  folds.json        GroupKFold assignment of the 80 training trips
  dataset_config.json
Only the train and student_test partitions are processed.
Run: python src/build_dataset.py [--R 50 --K 30]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import (ROOT, SEED, angle_diff, bearing_deg, clean_track, list_trips, load_trip,
                    point_segment, save_json)
from labels_candidates import ArcIndex, derive_labels

PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "dist_m",            # perpendicular distance GPS point -> arc segment (core emission signal)
    "dist_rank",         # rank of the arc by distance among this point's candidates (0 = nearest)
    "dist_gap_min",      # dist - min dist at this point (relative closeness, removes per-point GPS bias)
    "n_cands",           # number of candidates at this point (junction / dense-area indicator)
    "heading_diff",      # |GPS motion bearing - arc bearing| in [0,180]; only feature separating twin directed arcs
    "heading_reliable",  # 1 if the vehicle moved >= 2 m around this point (else heading is carried forward)
    "speed_ms",          # GPS speed (central difference); low speed => heading unreliable, high => major roads
    "turn_rate",         # change of GPS heading per second (turning near junctions)
    "arc_len_m",         # arc length; long arcs are major roads, very short arcs are junction connectors
    "along_t",           # projection parameter on arc in [0,1]; 0/1 = point projects on an endpoint
    "dt_prev",           # time gap to previous point (s); gaps make context features less trustworthy
    "dist_prev_pt",      # distance of previous GPS point to the same arc (temporal context)
    "dist_next_pt",      # distance of next GPS point to the same arc
    "dist_win5",         # mean distance of points i-2..i+2 to the same arc (smoothed distance)
    "heading_diff_win5", # heading diff using a longer (i-2 -> i+2) motion vector
    "node_deg_sum",      # total degree of the arc's two end nodes (junction complexity / road class proxy)
    "is_two_way",        # reverse arc exists (two-way street => twin candidate with opposite heading)
]


def motion(xy, t, k):
    n = len(xy)
    i0 = np.clip(np.arange(n) - k, 0, n - 1)
    i1 = np.clip(np.arange(n) + k, 0, n - 1)
    d = xy[i1] - xy[i0]
    dist = np.hypot(d[:, 0], d[:, 1])
    dt = t[i1] - t[i0]
    return bearing_deg(d[:, 0], d[:, 1]), dist, np.divide(dist, dt, out=np.zeros_like(dist), where=dt > 0)


def ffill_bfill(vals, ok):
    s = pd.Series(np.where(ok, vals, np.nan))
    s = s.ffill().bfill()
    return s.fillna(0).to_numpy()


def candidate_features(tr, idx, R, K):
    """Candidates + features for one trip. Uses ONLY .track/.nodes/.arcs (never .route),
    so the exact same function is used for training and for API inference."""
    xy = tr.track_xy[idx]
    tt = tr.track[idx, 2]
    n = len(idx)
    # point-level kinematics
    brg1, mv1, spd = motion(xy, tt, 1)
    rel1 = mv1 >= 2.0
    head1 = ffill_bfill(brg1, rel1)
    brg2, mv2, _ = motion(xy, tt, 2)
    head2 = ffill_bfill(brg2, mv2 >= 4.0)
    dt_prev = np.r_[np.nan, np.diff(tt)]
    dt_prev[0] = np.nanmedian(dt_prev[1:]) if n > 1 else 1.0
    dh = np.r_[0.0, angle_diff(head1[1:], head1[:-1])]
    turn_rate = dh / np.maximum(dt_prev, 1e-6)

    # candidates
    ai = ArcIndex(tr, idx, R)
    pi, arc, d, along = ai.query(xy, R)
    df = pd.DataFrame({"pi": pi, "arc_id": arc, "dist_m": d, "along_t": along})
    df["rk"] = df.groupby("pi").dist_m.rank(method="first") - 1
    df = df[df.rk < K].drop(columns="rk")
    df["dist_rank"] = df.groupby("pi").dist_m.rank(method="min") - 1
    df["dist_gap_min"] = df.dist_m - df.groupby("pi").dist_m.transform("min")
    df["n_cands"] = df.groupby("pi").arc_id.transform("size")
    pi = df.pi.to_numpy(); arc = df.arc_id.to_numpy()

    a = tr.arcs
    A_, B_ = tr.nodes_xy[a[arc, 0]], tr.nodes_xy[a[arc, 1]]
    arc_brg = bearing_deg(B_[:, 0] - A_[:, 0], B_[:, 1] - A_[:, 1])
    df["heading_diff"] = angle_diff(head1[pi], arc_brg)
    df["heading_diff_win5"] = angle_diff(head2[pi], arc_brg)
    df["heading_reliable"] = rel1[pi].astype(np.int8)
    df["speed_ms"] = spd[pi]
    df["turn_rate"] = turn_rate[pi]
    df["arc_len_m"] = np.hypot(*(B_ - A_).T)
    df["dt_prev"] = dt_prev[pi]
    ctx = {}
    for off in (-2, -1, 1, 2):
        q = np.clip(pi + off, 0, n - 1)
        ctx[off], _ = point_segment(xy[q, 0], xy[q, 1], A_[:, 0], A_[:, 1], B_[:, 0], B_[:, 1])
    df["dist_prev_pt"] = ctx[-1]
    df["dist_next_pt"] = ctx[1]
    df["dist_win5"] = (ctx[-2] + ctx[-1] + df.dist_m.to_numpy() + ctx[1] + ctx[2]) / 5
    deg = np.bincount(a.ravel(), minlength=len(tr.nodes))
    df["node_deg_sum"] = deg[a[arc, 0]] + deg[a[arc, 1]]
    pair_key = a[:, 0].astype(np.int64) * (len(tr.nodes) + 1) + a[:, 1]
    rev_key = a[arc, 1].astype(np.int64) * (len(tr.nodes) + 1) + a[arc, 0]
    df["is_two_way"] = np.isin(rev_key, pair_key).astype(np.int8)
    df["point_idx"] = idx[pi]
    df["trip_id"] = tr.trip_id
    df["partition"] = tr.partition
    df = df[["trip_id", "partition", "point_idx", "arc_id"] + FEATURES].reset_index(drop=True)
    return df, pi


def build_trip(partition, t, R, K):
    tr = load_trip(partition, t)
    idx, cinfo = clean_track(tr)
    n = len(idx)
    tt = tr.track[idx, 2]
    true_arc, route_pos, label_dist, ainfo = derive_labels(tr, idx)
    df, pi = candidate_features(tr, idx, R, K)
    df["label"] = (df.arc_id.to_numpy() == true_arc[pi]).astype(np.int8)
    n_c = np.bincount(pi, minlength=n)
    hit = np.bincount(pi, weights=df.label.to_numpy(), minlength=n) > 0
    pts = pd.DataFrame({"trip_id": t, "partition": partition, "point_idx": idx, "t": tt,
                        "lon": tr.track[idx, 0], "lat": tr.track[idx, 1], "true_arc": true_arc,
                        "route_pos": route_pos, "label_dist": label_dist, "n_cands": n_c, "true_in_cands": hit})
    stat = {"trip_id": t, "partition": partition, "n_points": n, "rows": len(df),
            "miss_rate": float(1 - hit.mean()), "zero_cand_rate": float((n_c == 0).mean()), **ainfo, **cinfo}
    return df, pts, stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=float, default=50)
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    t0 = time.time()
    feats, pts, stats = [], [], []
    for part in ["train", "student_test"]:
        for t in list_trips(part):
            f, p, s = build_trip(part, t, args.R, args.K)
            feats.append(f); pts.append(p); stats.append(s)
            print(part, t, s["rows"], f"miss={s['miss_rate']:.3f}", flush=True)
    F = pd.concat(feats, ignore_index=True)
    for c in F.columns:
        if F[c].dtype == np.float64:
            F[c] = F[c].astype(np.float32)
    P = pd.concat(pts, ignore_index=True)
    S = pd.DataFrame(stats)
    F.to_parquet(PROC / "features.parquet", index=False)
    P.to_parquet(PROC / "points.parquet", index=False)
    S.to_csv(ROOT / "results" / "candidates" / "dataset_trip_stats.csv", index=False)

    # GroupKFold on training trips only (group = trip_id, fixed seed)
    tr_trips = np.array(sorted(P.loc[P.partition == "train", "trip_id"].unique()))
    gkf = GroupKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    folds = {}
    for k, (_, te) in enumerate(gkf.split(tr_trips, groups=tr_trips)):
        for t in tr_trips[te]:
            folds[str(t)] = k
    save_json(folds, PROC / "folds.json")
    summ = S.groupby("partition").apply(lambda g: pd.Series({
        "trips": len(g), "points": g.n_points.sum(), "rows": g.rows.sum(),
        "pooled_miss_rate": (g.miss_rate * g.n_points).sum() / g.n_points.sum(),
        "trip_mean_miss_rate": g.miss_rate.mean(),
        "zero_cand_rate": (g.zero_cand_rate * g.n_points).sum() / g.n_points.sum()}), include_groups=False)
    print(summ)
    summ.to_csv(ROOT / "results" / "candidates" / "candidate_summary.csv")
    save_json({"R_m": args.R, "K": args.K, "n_folds": args.folds, "seed": SEED, "features": FEATURES,
               "label_method": "dp_monotone_alignment", "pos_rate": float(F.label.mean()),
               "build_seconds": time.time() - t0}, PROC / "dataset_config.json")


if __name__ == "__main__":
    main()
