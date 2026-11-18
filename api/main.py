"""PathSnap API - runs on LAPTOP 1.

Run from the repo root:
    PATHSNAP_EVAL_DIR=data/raw/student_test uvicorn api.main:app --host 0.0.0.0 --port 8000   # development
    PATHSNAP_EVAL_DIR=data/evaluator        uvicorn api.main:app --host 0.0.0.0 --port 8000   # demo day
(--host 0.0.0.0 lets Laptop 2 connect; the default 127.0.0.1 only accepts requests from this laptop.)

Endpoints
    GET  /health            server up + which model / decoder is loaded
    GET  /trips             the trip ids in PATHSNAP_EVAL_DIR
    POST /predict {trip_id} full pipeline server-side -> predicted route + accuracy metrics

Pipeline for one trip (all on this laptop):
    1. CandidateScorer.score_trip_dir()  -> candidates + tuned-LightGBM probability per candidate   (.route NOT read)
    2. classifier-only prediction        -> best candidate per GPS point
    3. HMM / Viterbi (src/hmm.py)        -> connected route, if src/hmm.py exists (else classifier-only)
    4. ground truth, server-side only    -> load .route separately, DP-align it to the GPS points (same function
                                            as training labels), compute Top-1 / continuity / miss rate
Only aggregate metrics leave the server: the reference route and per-point labels are never in the response.
"""
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from inference import CandidateScorer                      # noqa: E402  (teammate's inference entry point)
from common import load_trip_dir, clean_track               # noqa: E402
from labels_candidates import derive_labels                 # noqa: E402  (same DP alignment as training labels)

try:                                                         # HMM is optional until src/hmm.py exists
    from hmm import decode_trip                              # contract: decode_trip(trip, idx, cands) -> arc per point
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", ROOT / "data" / "evaluator"))
NEARBY_ROAD_M = 150      # draw roads within this distance of the GPS track
MAX_ROAD_SEGMENTS = 5000

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap")

app = FastAPI(title="PathSnap API")
scorer = CandidateScorer()                                   # loads artifacts/final_model.joblib once at startup
log.info("model loaded: %s | HMM available: %s | eval dir: %s", scorer.name, HMM_AVAILABLE, EVAL_DIR)


class PredictRequest(BaseModel):
    trip_id: str


def list_eval_trips():
    if not EVAL_DIR.exists():
        return []
    return sorted(p.name for p in EVAL_DIR.iterdir() if p.is_dir() and (p / f"{p.name}.track").exists())


def continuity(pred, arcs):
    """Same definition as src/evaluate.route_continuity: share of consecutive point pairs (both predicted)
    whose arcs are identical or directly connected (to(a) == from(b))."""
    a = pred[pred >= 0]
    if len(a) < 2:
        return None
    ok = (a[1:] == a[:-1]) | (arcs[a[:-1], 1] == arcs[a[1:], 0])
    return float(ok.mean())


def to_point_positions(cands, idx):
    """Map the scorer's point_idx onto positions 0..len(idx)-1 (positions line up with derive_labels output).
    If point_idx already are positions this is the identity; if they are original track rows we translate them."""
    pid = cands["point_idx"].to_numpy()
    if len(pid) and pid.max() >= len(idx):                   # original row numbers -> positions
        pos_of_row = {int(r): i for i, r in enumerate(idx)}
        return np.array([pos_of_row[int(p)] for p in pid])
    return pid


def arc_segments(trip, arc_ids):
    """[[lat, lon], [lat, lon]] per arc - Folium wants lat first, the files store lon first."""
    n = trip.nodes
    return [[[float(n[trip.arcs[a, 0], 1]), float(n[trip.arcs[a, 0], 0])],
             [float(n[trip.arcs[a, 1], 1]), float(n[trip.arcs[a, 1], 0])]] for a in arc_ids]


def ordered_unique(arr):
    out = []
    for a in arr:
        if a >= 0 and (not out or out[-1] != a):
            out.append(int(a))
    return out


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    log.info("REQUEST %s %s from %s -> %d (%.2fs)", request.method, request.url.path,
             request.client.host if request.client else "?", response.status_code, time.perf_counter() - t0)
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model": scorer.name, "hmm": HMM_AVAILABLE,
            "eval_dir": str(EVAL_DIR), "n_trips": len(list_eval_trips())}


@app.get("/trips")
def trips():
    return {"trips": list_eval_trips()}


@app.post("/predict")
def predict(req: PredictRequest):
    if req.trip_id not in list_eval_trips():                 # also blocks path tricks like "../train/x"
        raise HTTPException(404, f"unknown trip {req.trip_id}")
    trip_dir = EVAL_DIR / req.trip_id
    t0 = time.perf_counter()

    # 1) candidates + probabilities (scorer reads .track/.nodes/.arcs only)
    cands = scorer.score_trip_dir(trip_dir)
    trip, idx = scorer.last_trip, scorer.last_idx
    n = len(idx)
    pos = to_point_positions(cands, idx)

    # 2) classifier-only: best candidate per point
    pred_cls = np.full(n, -1, dtype=np.int64)
    if len(cands):
        best = scorer.top1(cands)
        pred_cls[to_point_positions(best, idx)] = best["arc_id"].to_numpy()
    t_cls = time.perf_counter() - t0

    # 3) HMM / Viterbi
    pred_hmm, decoder = None, "classifier-only"
    if HMM_AVAILABLE and len(cands):
        pred_hmm = np.asarray(decode_trip(trip, idx, cands), dtype=np.int64)
        decoder = "classifier + HMM/Viterbi"
    pred = pred_hmm if pred_hmm is not None else pred_cls
    t_total = time.perf_counter() - t0

    # 4) server-side evaluation - ground truth never leaves this function
    metrics = {"decoder": decoder, "n_points": int(n), "n_candidates": int(len(cands)),
               "points_without_candidates": int(n - len(np.unique(pos))),
               "inference_time_s": round(t_total, 3), "classifier_time_s": round(t_cls, 3),
               "continuity_classifier": continuity(pred_cls, trip.arcs)}
    if pred_hmm is not None:
        metrics["continuity_hmm"] = continuity(pred_hmm, trip.arcs)
    gt_trip = load_trip_dir(trip_dir, read_route=True)
    if gt_trip.route is not None and len(gt_trip.route):
        gt_idx, _ = clean_track(gt_trip)
        true_arc, _, _, _ = derive_labels(gt_trip, gt_idx)
        cand_sets = {}
        for p, a in zip(pos, cands["arc_id"].to_numpy()):
            cand_sets.setdefault(int(p), set()).add(int(a))
        metrics["top1_classifier"] = float(np.mean(pred_cls == true_arc))
        if pred_hmm is not None:
            metrics["top1_hmm"] = float(np.mean(pred_hmm == true_arc))
        metrics["candidate_miss_rate"] = float(np.mean([true_arc[i] not in cand_sets.get(i, ()) for i in range(n)]))
    else:
        metrics["note"] = "no .route file for this trip - accuracy not computed"

    # map payload: GPS, predicted route, nearby roads (no ground truth)
    gps = trip.track[idx]
    near_nodes = cKDTree(trip.track_xy[idx]).query(trip.nodes_xy, distance_upper_bound=NEARBY_ROAD_M)[0] < np.inf
    road_ids = np.flatnonzero(near_nodes[trip.arcs[:, 0]] | near_nodes[trip.arcs[:, 1]])[:MAX_ROAD_SEGMENTS]

    log.info("PREDICT trip=%s points=%d decoder=%s top1_cls=%s top1_hmm=%s time=%.2fs", req.trip_id, n, decoder,
             metrics.get("top1_classifier"), metrics.get("top1_hmm"), t_total)
    return {
        "trip_id": req.trip_id,
        "model": scorer.name,
        "gps_points": [[float(lat), float(lon)] for lon, lat in gps[:, :2]],
        "predicted_arc_ids": [int(a) for a in pred],
        "predicted_route_segments": arc_segments(trip, ordered_unique(pred)),
        "road_network_segments": arc_segments(trip, road_ids),
        "metrics": metrics,
    }
