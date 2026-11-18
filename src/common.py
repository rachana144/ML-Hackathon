"""Shared I/O, projection and geometry helpers for PathSnap (Person 1).

All distances are computed in a per-trip UTM projection (metres). Raw files
store coordinates as (longitude, latitude); Folium needs [lat, lon].
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("PATHSNAP_DATA", ROOT / "data" / "raw"))
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"
SEED = 42

# Only these two partitions are ever read by Person 1's code.
# The evaluator partition is deliberately NOT listed here.
PARTITIONS = {"train": "train", "student_test": "student_test"}


def list_trips(partition: str) -> list[str]:
    if partition not in PARTITIONS:
        raise ValueError(f"partition {partition!r} not allowed")
    d = DATA_DIR / PARTITIONS[partition]
    return sorted(p.name for p in d.iterdir() if p.is_dir())


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: np.ndarray      # (N,3) lon, lat, t
    nodes: np.ndarray      # (M,2) lon, lat
    arcs: np.ndarray       # (A,2) from, to
    route: np.ndarray | None  # (R,) arc ids
    crs: CRS
    track_xy: np.ndarray   # (N,2) metres
    nodes_xy: np.ndarray   # (M,2) metres


def utm_crs_for(lon: float, lat: float) -> CRS:
    zone = int((lon + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


def _read(path: Path, dtype=float, ncols=None) -> np.ndarray:
    if not path.exists():
        return None
    arr = pd.read_csv(path, sep=r"\s+", header=None, dtype=dtype, engine="c").to_numpy()
    if ncols is not None and arr.ndim == 2 and arr.shape[1] != ncols:
        raise ValueError(f"{path}: expected {ncols} cols, got {arr.shape[1]}")
    return arr


def load_trip(partition: str, trip_id: str, project: bool = True) -> Trip:
    return load_trip_dir(DATA_DIR / PARTITIONS[partition] / trip_id, partition, project)


def load_trip_dir(d: Path, partition: str = "inference", project: bool = True, read_route: bool = True) -> Trip:
    """Load a trip folder <d>/<id>.{track,nodes,arcs[,route]}. API inference passes read_route=False."""
    d = Path(d)
    trip_id = d.name
    track = _read(d / f"{trip_id}.track", float, 3)
    nodes = _read(d / f"{trip_id}.nodes", float, 2)
    arcs = _read(d / f"{trip_id}.arcs", np.int64, 2)
    route = _read(d / f"{trip_id}.route", np.int64, 1) if read_route else None
    route = None if route is None else route.ravel()
    crs = utm_crs_for(float(np.nanmedian(track[:, 0])), float(np.nanmedian(track[:, 1])))
    if project:
        tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        tx, ty = tf.transform(track[:, 0], track[:, 1])
        nx, ny = tf.transform(nodes[:, 0], nodes[:, 1])
        track_xy = np.column_stack([tx, ty])
        nodes_xy = np.column_stack([nx, ny])
    else:
        track_xy = nodes_xy = None
    return Trip(trip_id, partition, track, nodes, arcs, route, crs, track_xy, nodes_xy)


def clean_track(trip: Trip) -> tuple[np.ndarray, dict]:
    """Return indices of GPS rows kept after basic validity cleaning.

    Rules (all derived from EDA, no labels used):
      * drop rows with NaN / inf coordinates or timestamps
      * drop rows outside valid lon/lat ranges or at (0,0)
      * drop rows whose timestamp is not strictly greater than the previous kept one
    """
    tr = trip.track
    ok = np.isfinite(tr).all(axis=1)
    ok &= (np.abs(tr[:, 0]) <= 180) & (np.abs(tr[:, 1]) <= 90)
    ok &= ~((tr[:, 0] == 0) & (tr[:, 1] == 0))
    idx = np.flatnonzero(ok)
    keep = []
    last_t = -np.inf
    for i in idx:
        if tr[i, 2] > last_t:
            keep.append(i)
            last_t = tr[i, 2]
    keep = np.asarray(keep, dtype=np.int64)
    info = {"n_raw": len(tr), "n_invalid": int((~ok).sum()),
            "n_nonmonotone": int(ok.sum() - len(keep)), "n_kept": len(keep)}
    return keep, info


# ---------------------------------------------------------------- geometry
def haversine_m(lon1, lat1, lon2, lat2):
    R = 6371008.8
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def point_segment(px, py, ax, ay, bx, by):
    """Vectorised point-to-segment distance. Returns (dist, t in [0,1])."""
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    with np.errstate(invalid="ignore", divide="ignore"):
        t = ((px - ax) * vx + (py - ay) * vy) / L2
    t = np.where(L2 > 0, np.clip(t, 0.0, 1.0), 0.0)
    cx, cy = ax + t * vx, ay + t * vy
    return np.hypot(px - cx, py - cy), t


def bearing_deg(dx, dy):
    """Compass bearing (0 = north, clockwise) of a vector in projected metres."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff(a, b):
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
