"""Shared evaluation harness used by BOTH team members.

    from evaluate import evaluate
    evaluate(model, name="LogReg", family="Linear", config={...})

`model` is any sklearn-compatible estimator with fit / predict_proba (wrap
scalers inside a Pipeline so they are fitted on training folds only).

Protocol (identical for every model):
  1. 5-fold GroupKFold on the 80 training trips (folds fixed in data/processed/folds.json)
     -> CV macro F1 (mean ± std), CV Top-1, CV log loss
  2. refit on all 80 training trips -> train macro F1 (over/underfitting evidence)
  3. predict the 10 student-test trips -> Top-1, macro F1, precision, recall, accuracy,
     log loss, confusion matrix, classifier-only route continuity, inference time
Writes:
  results/results.csv                       one row per evaluate() call
  results/models/<name>.json                fold scores, confusion matrix, config
  results/preds/<name>_student_test.parquet candidate probabilities (input to HMM)
  artifacts/models/<name>.joblib            model refitted on all training trips (if save_model)
"""
from __future__ import annotations

import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss, precision_score,
                             recall_score)

sys.path.insert(0, str(Path(__file__).parent))
from common import ARTIFACTS, RESULTS, ROOT, SEED, load_trip

PROC = ROOT / "data" / "processed"
CFG = json.loads((PROC / "dataset_config.json").read_text())
FEATURES = CFG["features"]
RESULTS_CSV = RESULTS / "results.csv"
EPS = 1e-7


@lru_cache(maxsize=1)
def load_data():
    F = pd.read_parquet(PROC / "features.parquet")
    F = F.sort_values(["trip_id", "point_idx", "dist_m"], kind="mergesort").reset_index(drop=True)
    P = pd.read_parquet(PROC / "points.parquet")
    folds = json.loads((PROC / "folds.json").read_text())
    F["fold"] = F.trip_id.map(folds).fillna(-1).astype(int)
    P["fold"] = P.trip_id.map(folds).fillna(-1).astype(int)
    assert not F[FEATURES].isna().any().any(), "NaN in features"
    return F, P


@lru_cache(maxsize=None)
def _arcs(partition, trip_id):
    return load_trip(partition, trip_id, project=False).arcs


def top1_table(rows: pd.DataFrame, prob: np.ndarray) -> pd.DataFrame:
    """Per GPS point, the highest-scoring candidate (ties -> nearest, rows are distance-sorted)."""
    r = rows[["trip_id", "partition", "point_idx", "arc_id"]].copy()
    r["p"] = prob
    best = r.loc[r.groupby(["trip_id", "point_idx"], sort=False).p.idxmax()]
    return best


def top1_metrics(points: pd.DataFrame, best: pd.DataFrame) -> dict:
    m = points.merge(best[["trip_id", "point_idx", "arc_id", "p"]], on=["trip_id", "point_idx"], how="left")
    correct = (m.arc_id == m.true_arc)          # points without candidates -> NaN -> False
    cov = m.true_in_cands
    per_trip = correct.groupby(m.trip_id).mean()
    return {"top1": float(correct.mean()), "top1_covered": float(correct[cov].mean()),
            "top1_trip_mean": float(per_trip.mean()), "n_points": int(len(m))}, m


def route_continuity(m: pd.DataFrame) -> float:
    """Classifier-only route continuity: fraction of consecutive GPS-point pairs (both with a
    prediction) whose predicted arcs are identical or directly connected (to(a) == from(b))."""
    ok_tot, n_tot = 0, 0
    for (part, t), g in m.sort_values(["trip_id", "point_idx"]).groupby(["partition", "trip_id"]):
        arcs = _arcs(part, t)
        a = g.arc_id.to_numpy()
        a = a[~np.isnan(a)].astype(int)
        if len(a) < 2:
            continue
        same = a[1:] == a[:-1]
        conn = arcs[a[:-1], 1] == arcs[a[1:], 0]
        ok_tot += int((same | conn).sum()); n_tot += len(a) - 1
    return ok_tot / max(n_tot, 1)


def cls_metrics(y, p, thr=0.5) -> dict:
    yhat = (p >= thr).astype(int)
    pc = np.clip(p, EPS, 1 - EPS)
    return {"macro_f1": f1_score(y, yhat, average="macro"), "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat, zero_division=0), "accuracy": accuracy_score(y, yhat),
            "log_loss": log_loss(y, pc, labels=[0, 1]), "cm": confusion_matrix(y, yhat, labels=[0, 1]).tolist()}


def _subsample(rows: pd.DataFrame, frac: float | None, seed=SEED) -> pd.DataFrame:
    """Subsample whole GPS points (all their candidates) to keep per-point structure."""
    if not frac or frac >= 1:
        return rows
    key = rows.trip_id.astype(str) + "_" + rows.point_idx.astype(str)
    u = key.unique()
    rng = np.random.default_rng(seed)
    keep = set(rng.choice(u, size=int(len(u) * frac), replace=False))
    return rows[key.isin(keep)]


def _proba(model, X):
    return model.predict_proba(X)[:, 1]


def evaluate(model, name: str, family: str, config: dict | str = "", train_frac: float | None = None,
             features: list[str] | None = None, save_model: bool = True, log: bool = True,
             cv: bool = True, author: str = "") -> dict:
    feats = features or FEATURES
    F, P = load_data()
    tr = F[F.partition == "train"]
    te = F[F.partition == "student_test"]
    res = {"model": name, "family": family, "config": json.dumps(config) if isinstance(config, dict) else str(config),
           "features": len(feats), "train_frac": train_frac or 1.0, "author": author}
    # ---------- 1. GroupKFold CV
    fold_scores = []
    if cv:
        for k in sorted(tr.fold.unique()):
            fit_rows = _subsample(tr[tr.fold != k], train_frac)
            val_rows = tr[tr.fold == k]
            m = clone(model)
            m.fit(fit_rows[feats].to_numpy(), fit_rows.label.to_numpy())
            p = _proba(m, val_rows[feats].to_numpy())
            cm = cls_metrics(val_rows.label.to_numpy(), p)
            t1, _ = top1_metrics(P[(P.partition == "train") & (P.fold == k)], top1_table(val_rows, p))
            fold_scores.append({"fold": int(k), **{x: cm[x] for x in ["macro_f1", "log_loss", "precision", "recall"]},
                                "top1": t1["top1"], "top1_covered": t1["top1_covered"]})
            print(f"  [{name}] fold {k}: macroF1={cm['macro_f1']:.4f} top1={t1['top1']:.4f}", flush=True)
        fs = pd.DataFrame(fold_scores)
        res.update({"cv_macro_f1_mean": fs.macro_f1.mean(), "cv_macro_f1_std": fs.macro_f1.std(ddof=0),
                    "cv_top1_mean": fs.top1.mean(), "cv_top1_std": fs.top1.std(ddof=0),
                    "cv_top1_covered_mean": fs.top1_covered.mean(), "cv_log_loss_mean": fs.log_loss.mean()})
    # ---------- 2. refit on all training trips
    fit_rows = _subsample(tr, train_frac)
    m = clone(model)
    t0 = time.perf_counter()
    m.fit(fit_rows[feats].to_numpy(), fit_rows.label.to_numpy())
    res["train_time_s"] = time.perf_counter() - t0
    trs = fit_rows.sample(n=min(len(fit_rows), 300_000), random_state=SEED)
    res["train_macro_f1"] = cls_metrics(trs.label.to_numpy(), _proba(m, trs[feats].to_numpy()))["macro_f1"]
    # ---------- 3. student-test
    Xte = te[feats].to_numpy()
    t0 = time.perf_counter()
    p = _proba(m, Xte)
    inf = time.perf_counter() - t0
    cm = cls_metrics(te.label.to_numpy(), p)
    best = top1_table(te, p)
    t1, merged = top1_metrics(P[P.partition == "student_test"], best)
    res.update({"test_top1": t1["top1"], "test_top1_covered": t1["top1_covered"], "test_top1_trip_mean": t1["top1_trip_mean"],
                "test_macro_f1": cm["macro_f1"], "test_precision": cm["precision"], "test_recall": cm["recall"],
                "test_accuracy": cm["accuracy"], "test_log_loss": cm["log_loss"],
                "test_route_continuity": route_continuity(merged),
                "inference_time_s": inf, "inference_ms_per_1k_points": 1000 * inf / max(t1["n_points"], 1) * 1000,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})
    if log:
        (RESULTS / "models").mkdir(parents=True, exist_ok=True)
        (RESULTS / "preds").mkdir(parents=True, exist_ok=True)
        detail = {**res, "fold_scores": fold_scores, "test_confusion_matrix": cm["cm"], "features_used": feats,
                  "params": {k: str(v) for k, v in m.get_params().items()}}
        (RESULTS / "models" / f"{name}.json").write_text(json.dumps(detail, indent=2, default=float))
        out = te[["trip_id", "point_idx", "arc_id", "label"]].copy(); out["prob"] = p.astype(np.float32)
        out.to_parquet(RESULTS / "preds" / f"{name}_student_test.parquet", index=False)
        row = pd.DataFrame([res])
        if RESULTS_CSV.exists():
            old = pd.read_csv(RESULTS_CSV)
            old = old[old.model != name]           # re-running a model replaces its row
            row = pd.concat([old, row], ignore_index=True)
        row.to_csv(RESULTS_CSV, index=False)
    if save_model:
        (ARTIFACTS / "models").mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": m, "features": feats, "dataset_config": CFG}, ARTIFACTS / "models" / f"{name}.joblib")
    show = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k != "config"}
    print(json.dumps(show, indent=1, default=str))
    return res


class NearestArcBaseline:
    """Non-learned reference: score = -distance (picks the nearest arc). Used to show what the
    classifier adds over pure geometry. predict_proba maps distance rank 0 -> 1."""

    def __init__(self, dist_col=0, rank_col=1):
        self.dist_col, self.rank_col = dist_col, rank_col

    def fit(self, X, y):
        return self

    def get_params(self, deep=True):
        return {"dist_col": self.dist_col, "rank_col": self.rank_col}

    def set_params(self, **p):
        self.__dict__.update(p); return self

    def predict_proba(self, X):
        s = np.exp(-X[:, self.dist_col] / 10.0) * (X[:, self.rank_col] == 0)
        s = np.clip(s + 1e-4 * np.exp(-X[:, self.dist_col] / 10.0), 0, 1)
        return np.column_stack([1 - s, s])
