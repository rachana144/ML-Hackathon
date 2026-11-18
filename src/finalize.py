"""Export the final model for the API and run consistency checks.

  python src/finalize.py [--name lightgbm_tuned]
1. copies artifacts/models/<name>.joblib -> artifacts/final_model.joblib (+ metadata)
2. checks that inference.py (raw folder, no .route) reproduces the evaluate() probabilities
   on every student-test trip (max abs diff must be ~0)
3. writes results/optuna/tuned_vs_untuned.csv
4. Folium map of classifier Top-1 predictions vs reference route for one student-test trip
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import ARTIFACTS, DATA_DIR, RESULTS, list_trips
from folium_maps import trip_map
from inference import CandidateScorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="lightgbm_tuned")
    ap.add_argument("--map_trip", default="00000077")
    a = ap.parse_args()
    art = joblib.load(ARTIFACTS / "models" / f"{a.name}.joblib")
    art["name"] = a.name
    res = pd.read_csv(RESULTS / "results.csv").set_index("model")
    art["metrics"] = res.loc[a.name].to_dict()
    joblib.dump(art, ARTIFACTS / "final_model.joblib")
    (ARTIFACTS / "final_model_meta.json").write_text(json.dumps(
        {"name": a.name, "features": art["features"], "R_m": art["dataset_config"]["R_m"], "K": art["dataset_config"]["K"],
         "metrics": art["metrics"]}, indent=2, default=str))

    # consistency check: inference path == training/evaluation path
    sc = CandidateScorer()
    ref = pd.read_parquet(RESULTS / "preds" / f"{a.name}_student_test.parquet")
    diffs, times = [], []
    for t in list_trips("student_test"):
        d = sc.score_trip_dir(DATA_DIR / "student_test" / t)
        m = ref[ref.trip_id == t].merge(d[["point_idx", "arc_id", "prob"]], on=["point_idx", "arc_id"], how="outer",
                                        suffixes=("_eval", "_inf"), indicator=True)
        diffs.append({"trip_id": t, "rows_eval": int((m._merge != "right_only").sum()),
                      "rows_inf": int((m._merge != "left_only").sum()), "unmatched": int((m._merge != "both").sum()),
                      "max_abs_prob_diff": float((m.prob_eval - m.prob_inf).abs().max()),
                      "seconds": sc.last_timing["seconds"], "points": sc.last_timing["points"]})
    chk = pd.DataFrame(diffs)
    chk.to_csv(RESULTS / "inference_consistency_check.csv", index=False)
    print(chk.to_string(index=False))
    assert chk.unmatched.sum() == 0 and chk.max_abs_prob_diff.max() < 1e-5, "inference path differs from training path!"

    # tuned vs untuned
    base = a.name.replace("_tuned", "")
    if base in res.index and a.name != base:
        cols = ["cv_macro_f1_mean", "cv_macro_f1_std", "cv_top1_mean", "cv_log_loss_mean", "train_macro_f1",
                "test_top1", "test_macro_f1", "test_precision", "test_recall", "test_log_loss",
                "test_route_continuity", "train_time_s", "inference_ms_per_1k_points"]
        tv = res.loc[[base, a.name], cols].T
        tv["delta"] = tv[a.name] - tv[base]
        tv.to_csv(RESULTS / "optuna" / "tuned_vs_untuned.csv")
        print(tv.round(4).to_string())

    # prediction map
    d = sc.score_trip_dir(DATA_DIR / "student_test" / a.map_trip)
    top = sc.top1(d)
    print(trip_map("student_test", a.map_trip, pred_arcs=top.arc_id.to_numpy(),
                   pred_label=f"{a.name} Top-1 (classifier only)",
                   fname=RESULTS / "maps" / f"student_test_{a.map_trip}_predictions.html"))


if __name__ == "__main__":
    main()
