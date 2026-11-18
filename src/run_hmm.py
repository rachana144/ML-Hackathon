"""Evaluate classifier-only vs classifier + HMM, and tune beta / gap threshold on TRAINING trips.

  python src/run_hmm.py tune     # out-of-fold probs on fold-0 training trips, grid over beta then gap_s
  python src/run_hmm.py test     # 10 student-test trips with the chosen parameters
  python src/run_hmm.py map 00000077
Outputs in results/hmm/ and results/maps/. Evaluator trips are never read.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS, SEED, load_trip
from evaluate import FEATURES, load_data, route_continuity, top1_metrics, top1_table
from hmm import DEFAULTS, RoadGraph, decode_trip_df

OUT = RESULTS / "hmm"
OUT.mkdir(parents=True, exist_ok=True)
BETAS = [0.25, 0.5, 1, 2, 5, 10, 20, 50]
GAPS = [10, 30, 60, 300]


# ------------------------------------------------------------------ metrics
def categorise(m: pd.DataFrame, arcs) -> np.ndarray:
    cat = []
    for p, gt, cov in zip(m.arc_id, m.true_arc, m.true_in_cands):
        if pd.isna(p) or p < 0:
            cat.append("no_prediction"); continue
        p = int(p)
        if p == gt: cat.append("correct")
        elif not cov: cat.append("not_a_candidate")
        elif arcs[p, 0] == arcs[gt, 1] and arcs[p, 1] == arcs[gt, 0]: cat.append("reverse_twin")
        elif {*arcs[p]} & {*arcs[gt]}: cat.append("adjacent")
        else: cat.append("other")
    return np.array(cat)


def distinct_continuity(arc_seq: np.ndarray, graph: RoadGraph, max_m=250.0):
    """Collapse consecutive repeats; for each change a->b: 1-hop (to(a)==from(b)) and
    network-reachable (shortest path to(a)->from(b) <= max_m)."""
    a = arc_seq[arc_seq >= 0]
    if len(a) < 2:
        return np.nan, np.nan, 0
    keep = np.r_[True, a[1:] != a[:-1]]
    d = a[keep]
    if len(d) < 2:
        return 1.0, 1.0, 0
    hop = graph.arcs[d[:-1], 1] == graph.arcs[d[1:], 0]
    reach = np.array([graph.sp_from(int(graph.arcs[x, 1]), max_m).get(int(graph.arcs[y, 0]), np.inf) <= max_m
                      for x, y in zip(d[:-1], d[1:])])
    return hop.mean(), reach.mean(), len(d) - 1


def eval_trips(partition, trips, probs, points, beta, gap_s, top_k=DEFAULTS["top_k"], with_detail=True):
    per_trip, merged_all = [], []
    for t in trips:
        tr = load_trip(partition, t)
        graph = RoadGraph(tr)
        c = probs[probs.trip_id == t]
        P = points[points.trip_id == t]
        t0 = time.perf_counter()
        hmm = decode_trip_df(tr, c, beta=beta, gap_s=gap_s, top_k=top_k, graph=graph)
        sec = time.perf_counter() - t0
        clf = top1_table(c.assign(partition=partition), c.prob.to_numpy())[["point_idx", "arc_id"]]
        m = P.merge(clf.rename(columns={"arc_id": "clf_arc"}), on="point_idx", how="left") \
             .merge(hmm.rename(columns={"arc_id": "hmm_arc"}), on="point_idx", how="left")
        m["hmm_arc"] = m.hmm_arc.where(m.hmm_arc >= 0)
        r = {"trip_id": t, "n_points": len(m), "hmm_seconds": sec, "segments": int(hmm.segment.max() + 1),
             "top1_clf": float((m.clf_arc == m.true_arc).mean()), "top1_hmm": float((m.hmm_arc == m.true_arc).mean())}
        if with_detail:
            for k in ("clf", "hmm"):
                mm = m[["trip_id", "partition", "point_idx"]].assign(arc_id=m[f"{k}_arc"])
                r[f"cont_{k}"] = route_continuity(mm)
                h, rc, n = distinct_continuity(m[f"{k}_arc"].fillna(-1).astype(int).to_numpy(), graph)
                r[f"distinct_1hop_{k}"], r[f"distinct_reach250_{k}"], r[f"distinct_changes_{k}"] = h, rc, n
                m[f"cat_{k}"] = categorise(m.assign(arc_id=m[f"{k}_arc"]), tr.arcs)
        per_trip.append(r); merged_all.append(m)
    return pd.DataFrame(per_trip), pd.concat(merged_all, ignore_index=True)


def pooled(df):
    w = df.n_points
    out = {"points": int(w.sum()),
           "top1_clf": float((df.top1_clf * w).sum() / w.sum()), "top1_hmm": float((df.top1_hmm * w).sum() / w.sum())}
    for c in [c for c in df.columns if c.startswith(("cont_", "distinct_1hop", "distinct_reach"))]:
        out[c] = float((df[c] * w).sum() / w.sum())
    out["hmm_seconds_total"] = float(df.hmm_seconds.sum())
    return out


# ------------------------------------------------------------------ tuning (training trips only)
def oof_probs_fold0():
    """LightGBM (tuned params) fitted on training folds 1-4, scored on fold-0 trips -> unbiased probs."""
    p = OUT / "oof_fold0_probs.parquet"
    if p.exists():
        return pd.read_parquet(p)
    from lightgbm import LGBMClassifier
    best = json.loads((RESULTS / "optuna" / "lightgbm_best.json").read_text())["best_params"]
    F, _ = load_data()
    tr = F[F.partition == "train"]
    fit, val = tr[tr.fold != 0], tr[tr.fold == 0]
    m = LGBMClassifier(**best, subsample_freq=1, random_state=SEED, n_jobs=-1, verbose=-1)
    m.fit(fit[FEATURES].to_numpy(), fit.label.to_numpy())
    out = val[["trip_id", "point_idx", "arc_id", "label"]].copy()
    out["prob"] = m.predict_proba(val[FEATURES].to_numpy())[:, 1].astype(np.float32)
    out.to_parquet(p, index=False)
    return out


def tune():
    probs = oof_probs_fold0()
    _, P = load_data()
    P = P[P.partition == "train"]
    trips = sorted(probs.trip_id.unique())
    rows = []
    for b in BETAS:
        d, _ = eval_trips("train", trips, probs, P, beta=b, gap_s=60, with_detail=False)
        rows.append({"stage": "beta", "beta": b, "gap_s": 60, **pooled(d)}); print(rows[-1], flush=True)
    best_b = max([r for r in rows if r["stage"] == "beta"], key=lambda r: r["top1_hmm"])["beta"]
    for g in GAPS:
        d, _ = eval_trips("train", trips, probs, P, beta=best_b, gap_s=g, with_detail=False)
        rows.append({"stage": "gap", "beta": best_b, "gap_s": g, **pooled(d)}); print(rows[-1], flush=True)
    best = max([r for r in rows if r["stage"] == "gap"], key=lambda r: (r["top1_hmm"], r["gap_s"] == 60))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "tuning_train_fold0.csv", index=False)
    chosen = {"beta": best["beta"], "gap_s": best["gap_s"], "top_k": DEFAULTS["top_k"],
              "tuned_on": f"{len(trips)} training trips of fold 0 (out-of-fold LightGBM probs)", "trips": trips,
              "selection_metric": "pooled HMM Top-1"}
    (OUT / "hmm_params.json").write_text(json.dumps(chosen, indent=2))
    print(res.to_string(index=False)); print(chosen)


# ------------------------------------------------------------------ student-test evaluation
def test():
    prm = json.loads((OUT / "hmm_params.json").read_text())
    probs = pd.read_parquet(RESULTS / "preds" / "lightgbm_tuned_student_test.parquet")
    _, P = load_data()
    P = P[P.partition == "student_test"]
    trips = sorted(P.trip_id.unique())
    d, m = eval_trips("student_test", trips, probs, P, beta=prm["beta"], gap_s=prm["gap_s"], top_k=prm["top_k"])
    d.to_csv(OUT / "hmm_per_trip_student_test.csv", index=False)
    pl = pooled(d)
    summ = pd.DataFrame([
        {"method": "classifier only (LightGBM tuned)", "top1": pl["top1_clf"], "top1_trip_mean": d.top1_clf.mean(),
         "route_continuity": pl["cont_clf"], "distinct_1hop": pl["distinct_1hop_clf"],
         "distinct_reach250m": pl["distinct_reach250_clf"]},
        {"method": f"classifier + HMM (beta={prm['beta']}, gap={prm['gap_s']}s, top_k={prm['top_k']})",
         "top1": pl["top1_hmm"], "top1_trip_mean": d.top1_hmm.mean(), "route_continuity": pl["cont_hmm"],
         "distinct_1hop": pl["distinct_1hop_hmm"], "distinct_reach250m": pl["distinct_reach250_hmm"],
         "hmm_seconds_total": pl["hmm_seconds_total"],
         "hmm_ms_per_1k_points": 1e6 * pl["hmm_seconds_total"] / pl["points"]}])
    summ.to_csv(OUT / "hmm_vs_classifier.csv", index=False)
    cats = pd.concat([m.cat_clf.value_counts().rename("classifier"), m.cat_hmm.value_counts().rename("hmm")], axis=1).fillna(0).astype(int)
    cats["classifier_%"] = (100 * cats.classifier / len(m)).round(2); cats["hmm_%"] = (100 * cats.hmm / len(m)).round(2)
    cats.to_csv(OUT / "error_breakdown_student_test.csv")
    tr = pd.crosstab(m.cat_clf, m.cat_hmm); tr.to_csv(OUT / "error_transitions_student_test.csv")
    m[["trip_id", "point_idx", "true_arc", "clf_arc", "hmm_arc", "cat_clf", "cat_hmm"]].to_parquet(OUT / "student_test_paths.parquet", index=False)
    pd.set_option("display.width", 220)
    print(summ.round(4).to_string(index=False)); print(d.round(4).to_string(index=False)); print(cats); print(tr)


def make_map(trip_id):
    from folium_maps import trip_map
    m = pd.read_parquet(OUT / "student_test_paths.parquet")
    m = m[m.trip_id == trip_id]
    print(trip_map("student_test", trip_id, point_step=2,
                   pred_layers=[(m.clf_arc.dropna().astype(int).to_numpy(), "Classifier only (Top-1)", "#d62728", "6", 3),
                                (m.hmm_arc.dropna().astype(int).to_numpy(), "Classifier + HMM (Viterbi)", "#7b2cbf", None, 4)],
                   fname=RESULTS / "maps" / f"student_test_{trip_id}_hmm.html"))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    {"tune": tune, "test": test}.get(cmd, lambda: make_map(sys.argv[2]))()
