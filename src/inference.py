"""Inference entry point for the API (Person 2): raw trip folder -> candidate probabilities.

    from inference import CandidateScorer
    scorer = CandidateScorer()                      # loads artifacts/final_model.joblib
    cands = scorer.score_trip_dir("path/to/00000123")   # .route is NOT read
    # cands: DataFrame trip_id, point_idx, arc_id, <features>, prob   (+ scorer.last_trip for nodes/arcs)

Exactly the same cleaning, candidate generation (R, K) and features as training
(build_dataset.candidate_features); parameters come from the saved artifact.
"""
import sys
import time
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from build_dataset import candidate_features
from common import ARTIFACTS, clean_track, load_trip_dir


class CandidateScorer:
    def __init__(self, path=ARTIFACTS / "final_model.joblib"):
        art = joblib.load(path)
        self.model = art["model"]
        self.features = art["features"]
        self.R = art["dataset_config"]["R_m"]
        self.K = art["dataset_config"]["K"]
        self.name = art.get("name", Path(path).stem)
        self.last_trip = None

    def score_trip_dir(self, trip_dir):
        t0 = time.perf_counter()
        tr = load_trip_dir(trip_dir, read_route=False)
        idx, info = clean_track(tr)
        df, _ = candidate_features(tr, idx, self.R, self.K)
        df["prob"] = self.model.predict_proba(df[self.features].to_numpy())[:, 1].astype(np.float32) if len(df) else []
        self.last_trip, self.last_idx = tr, idx
        self.last_timing = {"seconds": time.perf_counter() - t0, "points": len(idx), "candidates": len(df), **info}
        return df

    def top1(self, df):
        """Classifier-only prediction: best candidate per GPS point."""
        df = df.sort_values(["point_idx", "dist_m"], kind="mergesort")
        return df.loc[df.groupby("point_idx").prob.idxmax(), ["point_idx", "arc_id", "prob"]]


if __name__ == "__main__":
    s = CandidateScorer()
    d = s.score_trip_dir(sys.argv[1])
    print(s.top1(d).head(), s.last_timing)
