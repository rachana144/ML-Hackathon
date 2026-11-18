"""Model comparison plots from results/results.csv + per-model JSON (no metric is re-typed by hand).

Run: python src/plots.py  -> results/plots/*.png, results/comparison_table.csv/.md
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import ARTIFACTS, RESULTS

OUT = RESULTS / "plots"
OUT.mkdir(parents=True, exist_ok=True)
C = {"cv": "#2a6fdb", "test": "#e8833a", "train": "#7a7a7a", "tuned": "#2ca02c"}


def _short_cfg(c):
    try:
        d = json.loads(c)
    except Exception:
        return str(c)
    return ", ".join(f"{k}={round(v, 3) if isinstance(v, float) else v}" for k, v in d.items())


def main():
    r = pd.read_csv(RESULTS / "results.csv")
    r = r.sort_values("cv_macro_f1_mean", ascending=True).reset_index(drop=True)
    n = r.model.tolist()
    y = np.arange(len(r))

    # 1. CV macro F1 (mean ± std) vs train macro F1 vs student-test macro F1
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(r) + 1.5))
    ax.barh(y - .25, r.train_macro_f1, .25, color=C["train"], label="train (refit, all 80 trips)")
    ax.barh(y, r.cv_macro_f1_mean, .25, xerr=r.cv_macro_f1_std, color=C["cv"], label="5-fold GroupKFold CV (mean ± std)")
    ax.barh(y + .25, r.test_macro_f1, .25, color=C["test"], label="student-test")
    ax.set_yticks(y, n); ax.set_xlim(max(0, r[["cv_macro_f1_mean", "test_macro_f1"]].min().min() - .05), 1)
    ax.set_xlabel("macro F1 (candidate classification, threshold 0.5)"); ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Macro F1: train vs CV vs student-test"); ax.grid(axis="x", alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "macro_f1_comparison.png", dpi=130); plt.close(fig)

    # 2. Top-1 accuracy + route continuity
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(r) + 1.5))
    ax.barh(y - .2, r.cv_top1_mean, .2, xerr=r.cv_top1_std, color=C["cv"], label="CV Top-1 (all points)")
    ax.barh(y, r.test_top1, .2, color=C["test"], label="student-test Top-1")
    ax.barh(y + .2, r.test_route_continuity, .2, color=C["tuned"], label="student-test route continuity (classifier-only)")
    ax.set_yticks(y, n); ax.set_xlim(.4, 1); ax.legend(loc="lower right", fontsize=8); ax.grid(axis="x", alpha=.3)
    ax.set_title("Observation-level Top-1 accuracy and route continuity"); ax.set_xlabel("fraction")
    fig.tight_layout(); fig.savefig(OUT / "top1_comparison.png", dpi=130); plt.close(fig)

    # 3. log loss + inference time
    fig, ax = plt.subplots(1, 2, figsize=(12, 0.5 * len(r) + 1.5))
    ax[0].barh(y - .2, r.cv_log_loss_mean, .4, color=C["cv"], label="CV"); ax[0].barh(y + .2, r.test_log_loss, .4, color=C["test"], label="student-test")
    ax[0].set_yticks(y, n); ax[0].set_xscale("log"); ax[0].set_title("Log loss (lower = better calibrated)"); ax[0].legend(fontsize=8)
    ax[1].barh(y, r.inference_ms_per_1k_points, color="#8c6bb1"); ax[1].set_yticks(y, n); ax[1].set_xscale("log")
    ax[1].set_title("Inference time (ms per 1,000 GPS points, student-test)")
    for a in ax: a.grid(axis="x", alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "logloss_inference.png", dpi=130); plt.close(fig)

    # 4. per-fold macro F1 (stability)
    fig, ax = plt.subplots(figsize=(9, 4))
    for m in n:
        d = json.loads((RESULTS / "models" / f"{m}.json").read_text())
        fs = pd.DataFrame(d["fold_scores"])
        if len(fs):
            ax.plot(fs.fold, fs.macro_f1, "o-", label=m)
    ax.set_xlabel("fold"); ax.set_ylabel("macro F1"); ax.set_xticks(range(5)); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.3)
    ax.set_title("Per-fold CV macro F1 (GroupKFold by trip)")
    fig.tight_layout(); fig.savefig(OUT / "cv_folds.png", dpi=130); plt.close(fig)

    # 5. confusion matrices (student-test, threshold 0.5)
    k = len(n); cols = 4; rows = int(np.ceil(k / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.1 * rows))
    for a, m in zip(np.ravel(axs), n[::-1]):
        cm = np.array(json.loads((RESULTS / "models" / f"{m}.json").read_text())["test_confusion_matrix"])
        a.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(cm):
            a.text(j, i, f"{v:,}", ha="center", va="center", fontsize=9,
                   color="white" if cm[i, j] / cm[i].sum() > .5 else "black")
        a.set_xticks([0, 1], ["pred 0", "pred 1"]); a.set_yticks([0, 1], ["true 0", "true 1"]); a.set_title(m, fontsize=9)
    for a in np.ravel(axs)[k:]: a.axis("off")
    fig.suptitle("Student-test confusion matrices (candidate level, row-normalised colour)")
    fig.tight_layout(); fig.savefig(OUT / "confusion_matrices.png", dpi=130); plt.close(fig)

    # 6. feature importance of tree models
    for m in ["lightgbm", "lightgbm_tuned", "xgboost", "random_forest"]:
        p = ARTIFACTS / "models" / f"{m}.joblib"
        if not p.exists():
            continue
        art = joblib.load(p); mod = art["model"]
        if m.startswith("lightgbm"):
            imp = mod.booster_.feature_importance("gain")
        else:
            imp = mod.feature_importances_
        imp = imp / imp.sum()
        o = np.argsort(imp)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(np.array(art["features"])[o], imp[o], color=C["cv"]); ax.set_title(f"Feature importance – {m}" + (" (gain)" if "lightgbm" in m else ""))
        fig.tight_layout(); fig.savefig(OUT / f"feature_importance_{m}.png", dpi=130); plt.close(fig)

    # comparison table for report 3.1
    t = r.sort_values("cv_macro_f1_mean", ascending=False)
    tab = pd.DataFrame({
        "Model": t.model, "Algorithm family": t.family, "Key configuration": t.config,
        "CV macro F1 (mean±std)": [f"{a:.4f} ± {b:.4f}" for a, b in zip(t.cv_macro_f1_mean, t.cv_macro_f1_std)],
        "CV Top-1": t.cv_top1_mean.round(4), "Student-test Top-1": t.test_top1.round(4),
        "Route continuity (clf-only)": t.test_route_continuity.round(4),
        "Student-test macro F1": t.test_macro_f1.round(4), "Precision": t.test_precision.round(4),
        "Recall": t.test_recall.round(4), "Log loss": t.test_log_loss.round(4),
        "Train macro F1": t.train_macro_f1.round(4), "Train time (s)": t.train_time_s.round(1),
        "Inference (ms / 1k pts)": t.inference_ms_per_1k_points.round(2)})
    tab.to_csv(RESULTS / "comparison_table.csv", index=False)
    md = tab.copy()
    md["Key configuration"] = md["Key configuration"].map(_short_cfg)
    (RESULTS / "comparison_table.md").write_text(md.to_markdown(index=False))
    print(tab.drop(columns="Key configuration").to_string(index=False))


if __name__ == "__main__":
    main()
