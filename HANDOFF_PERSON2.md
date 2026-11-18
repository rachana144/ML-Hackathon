# Hand-off: Person 1 → Person 2

Everything below is reproducible with `python src/build_dataset.py` (≈45 s on 2 cores).

## 1. Files you get

| File | What it is |
|---|---|
| `data/processed/features.parquet` | one row per (GPS point, candidate arc). Columns: `trip_id, partition, point_idx, arc_id, <17 features>, label` |
| `data/processed/points.parquet` | one row per GPS point: `trip_id, partition, point_idx, t, lon, lat, true_arc, route_pos, label_dist, n_cands, true_in_cands` |
| `data/processed/folds.json` | `{trip_id: fold}` for the 80 training trips (5-fold GroupKFold, seed 42, 16 trips per fold) |
| `data/processed/dataset_config.json` | R = 50 m, K = 30, feature list, label method |
| `src/evaluate.py` | **the shared evaluator**. Use it for MLP / KNN / SVM so all models are comparable |
| `results/preds/<model>_student_test.parquet` | candidate probabilities on student-test (`trip_id, point_idx, arc_id, label, prob`): the HMM emission input |
| `artifacts/models/<model>.joblib` | `{"model", "features", "dataset_config"}`, refitted on all 80 training trips |

`partition` is `train` or `student_test`. Evaluator trips are **not** in any file.

## 2. How to use evaluate() for your models

```python
import sys; sys.path.insert(0, "src")
from evaluate import evaluate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                                                    random_state=42, max_iter=200))
evaluate(mlp, name="mlp", family="Neural network", config={"hidden": [64, 32]}, author="person2")

# KNN / SVM are O(n^2)-ish: train on a subsample of whole GPS points (same folds)
from sklearn.neighbors import KNeighborsClassifier
knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance"))
evaluate(knn, name="knn", family="Instance-based", config={"k": 25}, train_frac=0.1, author="person2")
```

* Put scalers **inside** the Pipeline: `evaluate()` clones and fits them on training folds only.
* `train_frac` subsamples whole GPS points (all their candidates), seed 42. It applies to CV fits and the final refit.
* For SVM use `SVC(probability=True)` or `CalibratedClassifierCV(LinearSVC())` with `train_frac≈0.05`. Top-1 needs a score per candidate.
* Each call appends or replaces a row in `results/results.csv` and writes `results/models/<name>.json` (fold scores, confusion matrix).
* For the MLP you can plot the train/val loss curve: `mlp[-1].loss_curve_` and `mlp[-1].validation_scores_`. It is the only model in the project that trains in epochs.

## 3. Facts the HMM needs

* **Hidden state** = candidate `arc_id` at each GPS point (directed arc; `arcs[a] = (from_node, to_node)`).
* **Emission**: use the classifier `prob` from `results/preds/…` (or `model.predict_proba` at inference).
* **Transition**: arcs `a → b` are connected iff `arcs[a,1] == arcs[b,0]`, plus `a == b` (staying on the arc). Every network is per-trip: load `.nodes/.arcs` from the trip folder (`src/common.load_trip`). Use network shortest-path distance vs GPS great-circle distance for multi-hop transitions (Newson & Krumm style); the GPS steps are mostly 1 s, so most transitions are 0–1 hops.
* **Two-way streets**: arcs `(u,v)` and `(v,u)` are geometrically identical. The classifier separates them via `heading_diff`. Pure nearest-arc only reaches 54% Top-1 on student-test, mostly because of this tie.
* **GPS gaps**: `dt_prev` > 5 s happens 1,216 times in train (max 153,064 s in trip 00000090). Widen or reset transitions across large gaps.
* **Points with no candidates** (R = 50 m): 8.7% of train points (mostly trip 00000054, which sat parked off-network for hours) and 0% on student-test. The HMM must skip or bridge them.
* Classifier-only route continuity is already computed by `evaluate()` (`test_route_continuity`). Same definition for the HMM: consecutive predictions are identical or directly connected.

## 4. Inference pipeline for the API

`src/inference.py` → `predict_trip(trip_dir)` loads raw `.track/.nodes/.arcs` (no `.route`), builds candidates and features exactly as in training, and returns per-candidate probabilities from the final model in `artifacts/final_model.joblib`. Ground truth is never read by this function.
