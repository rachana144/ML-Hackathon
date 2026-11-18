"""Choose candidate radius R (and cap K) on the TRAIN partition only.

For each R in {20,30,40,50,75,100} m and cap K in {inf, 30, 20, 10} we report
candidate miss rate (true arc not in candidate set) and mean #candidates/point.
Also writes label-alignment diagnostics (DP vs greedy pointer).
Run: python src/radius_study.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS, clean_track, list_trips, load_trip
from labels_candidates import ArcIndex, derive_labels

OUT = RESULTS / "candidates"
OUT.mkdir(parents=True, exist_ok=True)
RS = [20, 30, 40, 50, 75, 100]
KS = [np.inf, 30, 20, 10]


def main(partition="train"):
    recs, align = [], []
    for t in list_trips(partition):
        tr = load_trip(partition, t)
        idx, _ = clean_track(tr)
        true_arc, _, label_dist, info = derive_labels(tr, idx)
        align.append({"trip_id": t, "n_points": len(idx), "route_len": len(tr.route), **info})
        ai = ArcIndex(tr, idx, max(RS))
        pi, arc, d, _ = ai.query(tr.track_xy[idx], max(RS))
        # rank of true arc by distance (ties -> min rank)
        true_d = np.full(len(idx), np.inf)
        is_true = arc == true_arc[pi]
        true_d[pi[is_true]] = d[is_true]
        # number of candidates strictly closer than the true arc
        closer = np.bincount(pi, weights=(d < true_d[pi]).astype(float), minlength=len(idx))
        for R in RS:
            cnt = np.bincount(pi, weights=(d <= R).astype(float), minlength=len(idx))
            for K in KS:
                hit = (true_d <= R) & (closer < K)
                recs.append({"trip_id": t, "R": R, "K": K, "n": len(idx), "hits": int(hit.sum()),
                             "cands": float(np.minimum(cnt, K).sum()), "zero_cand": int((cnt == 0).sum())})
        print(t, info, flush=True)
    df = pd.DataFrame(recs)
    g = df.groupby(["R", "K"]).agg(n=("n", "sum"), hits=("hits", "sum"), cands=("cands", "sum"),
                                     zero=("zero_cand", "sum")).reset_index()
    g["miss_rate"] = 1 - g.hits / g.n
    g["mean_cands"] = g.cands / g.n
    g["zero_cand_rate"] = g.zero / g.n
    # per-trip miss rate (macro) for R with K=inf
    pt = df[df.K == np.inf].assign(m=lambda x: 1 - x.hits / x.n).groupby("R").m.agg(["mean", "median", "max"])
    g = g.merge(pt.rename(columns=lambda c: f"trip_miss_{c}"), left_on="R", right_index=True)
    g.to_csv(OUT / f"radius_study_{partition}.csv", index=False)
    pd.DataFrame(align).to_csv(OUT / f"label_alignment_{partition}.csv", index=False)
    print(g.to_string())
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for K in KS:
        s = g[g.K == K]
        ax[0].plot(s.R, 100 * s.miss_rate, "o-", label=f"K={K}")
        ax[1].plot(s.R, s.mean_cands, "o-", label=f"K={K}")
    ax[0].set_xlabel("radius R (m)"); ax[0].set_ylabel("candidate miss rate (%)"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].set_xlabel("radius R (m)"); ax[1].set_ylabel("mean candidates / point"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.suptitle(f"Candidate radius study ({partition} partition)")
    fig.tight_layout(); fig.savefig(OUT / f"radius_study_{partition}.png", dpi=120)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "train")
