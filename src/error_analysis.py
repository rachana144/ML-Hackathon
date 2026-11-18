"""Categorise classifier-only Top-1 errors on student-test.

  python src/error_analysis.py [model_name]
Categories for a wrong prediction p (true arc g):
  reverse_twin   p = (v,u) when g = (u,v)  -> direction confusion on a two-way street
  adjacent       p shares a node with g     -> junction / arc-boundary confusion
  parallel_other anything else (parallel road, wrong street)
  no_candidate   point had no candidates within R
  miss           true arc not among candidates
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS, load_trip
from evaluate import load_data


def main(name):
    _, P = load_data()
    P = P[P.partition == "student_test"]
    pr = pd.read_parquet(RESULTS / "preds" / f"{name}_student_test.parquet")
    best = pr.loc[pr.groupby(["trip_id", "point_idx"]).prob.idxmax(), ["trip_id", "point_idx", "arc_id"]]
    m = P.merge(best, on=["trip_id", "point_idx"], how="left")
    cats = []
    for t, g in m.groupby("trip_id"):
        arcs = load_trip("student_test", t, project=False).arcs
        for p, gt, cov in zip(g.arc_id, g.true_arc, g.true_in_cands):
            if pd.isna(p):
                cats.append("no_candidate"); continue
            p = int(p)
            if p == gt:
                cats.append("correct")
            elif not cov:
                cats.append("miss")
            elif arcs[p, 0] == arcs[gt, 1] and arcs[p, 1] == arcs[gt, 0]:
                cats.append("reverse_twin")
            elif len({*arcs[p]} & {*arcs[gt]}) > 0:
                cats.append("adjacent")
            else:
                cats.append("parallel_other")
    m["cat"] = cats
    s = m.cat.value_counts().rename("points").to_frame()
    s["share_of_points"] = s.points / len(m)
    s["share_of_errors"] = s.points / (m.cat != "correct").sum()
    s.loc["correct", "share_of_errors"] = float("nan")
    out = RESULTS / f"error_analysis_{name}.csv"
    s.to_csv(out)
    print(s.round(4))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lightgbm")
