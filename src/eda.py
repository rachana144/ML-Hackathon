"""Exploratory data analysis on the TRAIN and STUDENT_TEST partitions.

Outputs (results/eda/):
  inventory.csv        per-trip file availability, sizes, dtypes, row counts
  trip_stats.csv       per-trip GPS quality, timing, extent, network + route stats
  partition_summary.csv
  *.png                plots
Run:  python src/eda.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).parent))
from common import (DATA_DIR, PARTITIONS, RESULTS, clean_track, haversine_m,
                    list_trips, load_trip, point_segment)

OUT = RESULTS / "eda"
OUT.mkdir(parents=True, exist_ok=True)


def inventory():
    rows = []
    for part in PARTITIONS:
        for t in list_trips(part):
            d = DATA_DIR / PARTITIONS[part] / t
            r = {"partition": part, "trip_id": t, "n_files": len(list(d.iterdir()))}
            for ext in ["track", "nodes", "arcs", "route"]:
                p = d / f"{t}.{ext}"
                r[f"{ext}_exists"] = p.exists()
                r[f"{ext}_kb"] = round(p.stat().st_size / 1024, 1) if p.exists() else np.nan
                if p.exists():
                    first = p.open().readline().split()
                    r[f"{ext}_ncols"] = len(first)
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "inventory.csv", index=False)
    return df


def trip_stats(part, t):
    tr = load_trip(part, t)
    raw = tr.track
    keep, cinfo = clean_track(tr)
    trk = raw[keep]
    dt = np.diff(trk[:, 2])
    step = haversine_m(trk[:-1, 0], trk[:-1, 1], trk[1:, 0], trk[1:, 1])
    speed = np.divide(step, dt, out=np.zeros_like(step), where=dt > 0)
    M, A = len(tr.nodes), len(tr.arcs)
    a = tr.arcs
    # connectivity
    g = coo_matrix((np.ones(A), (a[:, 0], a[:, 1])), shape=(M, M))
    n_weak, lab_w = connected_components(g, directed=True, connection="weak")
    n_strong, lab_s = connected_components(g, directed=True, connection="strong")
    deg = np.bincount(a.ravel(), minlength=M)
    largest_weak = np.bincount(lab_w).max() / M
    largest_strong = np.bincount(lab_s).max() / M
    arc_len = haversine_m(tr.nodes[a[:, 0], 0], tr.nodes[a[:, 0], 1],
                          tr.nodes[a[:, 1], 0], tr.nodes[a[:, 1], 1])
    pairs = set(map(tuple, a.tolist()))
    two_way = np.mean([(y, x) in pairs for x, y in a.tolist()])
    dup_arcs = A - len(pairs)
    # route
    ro = tr.route
    r = {}
    if ro is not None and len(ro):
        valid_ids = (ro >= 0) & (ro < A)
        ro_v = ro[valid_ids]
        cont = np.mean(a[ro_v[:-1], 1] == a[ro_v[1:], 0]) if len(ro_v) > 1 else np.nan
        # GPS point -> nearest route arc distance (proxy for GPS noise + route quality)
        xy = tr.track_xy[keep]
        A_ = tr.nodes_xy[a[ro_v, 0]]; B_ = tr.nodes_xy[a[ro_v, 1]]
        dmin = np.full(len(xy), np.inf)
        for s in range(0, len(ro_v), 200):
            d, _ = point_segment(xy[:, None, 0], xy[:, None, 1], A_[None, s:s+200, 0], A_[None, s:s+200, 1],
                                 B_[None, s:s+200, 0], B_[None, s:s+200, 1])
            dmin = np.minimum(dmin, d.min(axis=1))
        r = {"route_len": len(ro), "route_invalid_ids": int((~valid_ids).sum()),
             "route_unique_arcs": len(np.unique(ro_v)), "route_continuity": cont,
             "route_length_km": arc_len[ro_v].sum() / 1000,
             "gps_to_route_median_m": np.median(dmin), "gps_to_route_p95_m": np.percentile(dmin, 95),
             "gps_to_route_max_m": dmin.max(), "frac_gps_gt50m_from_route": np.mean(dmin > 50)}
    return {
        "partition": part, "trip_id": t,
        "n_points_raw": cinfo["n_raw"], "n_invalid": cinfo["n_invalid"], "n_nonmonotone": cinfo["n_nonmonotone"],
        "n_points": cinfo["n_kept"], "n_nan": int(np.isnan(raw).sum()),
        "duration_s": trk[-1, 2] - trk[0, 2], "t_start": trk[0, 2],
        "frac_dt_1s": np.mean(np.isclose(dt, 1.0, atol=1e-3)), "dt_max": dt.max(), "dt_p99": np.percentile(dt, 99),
        "n_gaps_gt5s": int((dt > 5).sum()), "n_gaps_gt30s": int((dt > 30).sum()),
        "gps_path_km": step.sum() / 1000, "speed_median_ms": np.median(speed), "speed_p99_ms": np.percentile(speed, 99),
        "speed_max_ms": speed.max(), "n_jumps_gt70ms": int((speed > 70).sum()),
        "lon_min": trk[:, 0].min(), "lon_max": trk[:, 0].max(), "lat_min": trk[:, 1].min(), "lat_max": trk[:, 1].max(),
        "lon_c": trk[:, 0].mean(), "lat_c": trk[:, 1].mean(),
        "track_in_network_bbox": bool((trk[:, 0].min() >= tr.nodes[:, 0].min()) & (trk[:, 0].max() <= tr.nodes[:, 0].max())
                                       & (trk[:, 1].min() >= tr.nodes[:, 1].min()) & (trk[:, 1].max() <= tr.nodes[:, 1].max())),
        "utm_epsg": tr.crs.to_epsg(),
        "n_nodes": M, "n_arcs": A, "isolated_nodes": int((deg == 0).sum()), "self_loops": int((a[:, 0] == a[:, 1]).sum()),
        "duplicate_arcs": int(dup_arcs), "zero_len_arcs": int((arc_len < 0.01).sum()),
        "arc_len_median_m": np.median(arc_len), "arc_len_max_m": arc_len.max(), "two_way_frac": two_way,
        "weak_components": n_weak, "strong_components": n_strong,
        "largest_weak_frac": largest_weak, "largest_strong_frac": largest_strong,
        **r,
    }


def plots(st: pd.DataFrame):
    tr_ = st[st.partition == "train"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for p, c in [("train", "tab:blue"), ("student_test", "tab:orange")]:
        s = st[st.partition == p]
        ax[0, 0].hist(s.n_points, bins=30, alpha=.6, label=p, color=c)
        ax[0, 1].hist(s.n_arcs, bins=30, alpha=.6, label=p, color=c)
        ax[0, 2].hist(s.gps_to_route_median_m, bins=30, alpha=.6, label=p, color=c)
        ax[1, 0].hist(s.speed_median_ms * 3.6, bins=30, alpha=.6, label=p, color=c)
        ax[1, 1].hist(s.route_len, bins=30, alpha=.6, label=p, color=c)
        ax[1, 2].scatter(s.route_len, s.n_points, alpha=.7, label=p, color=c)
    for a_, t_ in zip(ax.ravel(), ["GPS points per trip", "Road arcs per trip", "Median GPS→route dist (m)",
                                   "Median speed (km/h)", "Route length (#arcs)", "Route arcs vs GPS points"]):
        a_.set_title(t_); a_.legend()
    lim = st[["route_len", "n_points"]].max().max()
    ax[1, 2].plot([0, lim], [0, lim], "k--", lw=1, label="y = x"); ax[1, 2].set_xlabel("#route arcs"); ax[1, 2].set_ylabel("#GPS points")
    fig.suptitle("Per-trip distributions (train + student_test partitions)")
    fig.tight_layout(); fig.savefig(OUT / "trip_distributions.png", dpi=120); plt.close(fig)

    # sampling interval distribution (train partition)
    dts = []
    for t in tr_.trip_id:
        trk = load_trip("train", t, project=False).track
        dts.append(np.diff(trk[:, 2]))
    dts = np.concatenate(dts)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    vals, cnts = np.unique(np.round(dts), return_counts=True)
    ax[0].bar(vals[:10].astype(str), cnts[:10]); ax[0].set_yscale("log"); ax[0].set_title("Sampling interval Δt (s), train, log count")
    ax[1].hist(dts[dts > 1.5], bins=60); ax[1].set_yscale("log"); ax[1].set_title("Gaps Δt > 1.5 s (train)")
    fig.tight_layout(); fig.savefig(OUT / "sampling_intervals.png", dpi=120); plt.close(fig)
    pd.DataFrame({"dt_s": vals, "count": cnts}).to_csv(OUT / "dt_value_counts_train.csv", index=False)

    # global trip locations
    fig, ax = plt.subplots(figsize=(10, 5))
    for p, c in [("train", "tab:blue"), ("student_test", "tab:orange")]:
        s = st[st.partition == p]
        ax.scatter(s.lon_c, s.lat_c, s=20, c=c, label=p)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.set_title("Trip centroids (lon, lat)"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "trip_locations.png", dpi=120); plt.close(fig)

    # GPS trajectory vs route + network, 6 train trips
    sel = tr_.sort_values("n_points").trip_id.iloc[[0, 15, 30, 45, 60, 79]].tolist()
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    for ax, t in zip(axs.ravel(), sel):
        tr = load_trip("train", t)
        xy = tr.track_xy; x0, y0 = xy.min(0) - 300; x1, y1 = xy.max(0) + 300
        A_ = tr.nodes_xy[tr.arcs[:, 0]]; B_ = tr.nodes_xy[tr.arcs[:, 1]]
        m = ((A_[:, 0] > x0) & (A_[:, 0] < x1) & (A_[:, 1] > y0) & (A_[:, 1] < y1))
        segs = np.stack([A_[m], B_[m]], 1)
        from matplotlib.collections import LineCollection
        ax.add_collection(LineCollection(segs, colors="lightgrey", lw=.5))
        rs = np.stack([tr.nodes_xy[tr.arcs[tr.route, 0]], tr.nodes_xy[tr.arcs[tr.route, 1]]], 1)
        ax.add_collection(LineCollection(rs, colors="green", lw=2.5))
        ax.scatter(xy[:, 0], xy[:, 1], s=1.5, c="blue")
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.set_title(f"train trip {t} ({len(xy)} pts)")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("GPS points (blue), reference route (green), road network (grey) — UTM metres")
    fig.tight_layout(); fig.savefig(OUT / "gps_trajectories.png", dpi=110); plt.close(fig)


def main():
    inv = inventory()
    rows = []
    for part in PARTITIONS:
        for t in list_trips(part):
            rows.append(trip_stats(part, t))
            print(part, t, "done", flush=True)
    st = pd.DataFrame(rows)
    st.to_csv(OUT / "trip_stats.csv", index=False)
    agg = st.groupby("partition").agg(
        trips=("trip_id", "count"), gps_points=("n_points", "sum"), pts_min=("n_points", "min"),
        pts_median=("n_points", "median"), pts_max=("n_points", "max"), invalid=("n_invalid", "sum"),
        nonmonotone=("n_nonmonotone", "sum"), nan=("n_nan", "sum"), frac_dt_1s=("frac_dt_1s", "mean"),
        gaps_gt5s=("n_gaps_gt5s", "sum"), gaps_gt30s=("n_gaps_gt30s", "sum"), max_gap=("dt_max", "max"),
        nodes_median=("n_nodes", "median"), arcs_median=("n_arcs", "median"), arcs_max=("n_arcs", "max"),
        isolated_nodes=("isolated_nodes", "sum"), weak_comp_median=("weak_components", "median"),
        strong_comp_median=("strong_components", "median"), route_cont_min=("route_continuity", "min"),
        route_len_median=("route_len", "median"), gps_route_med=("gps_to_route_median_m", "median"),
        gps_route_p95=("gps_to_route_p95_m", "median"),
    ).T
    agg.to_csv(OUT / "partition_summary.csv")
    print(agg)
    plots(st)


if __name__ == "__main__":
    main()
