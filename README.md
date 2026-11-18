# PathSnap – ML map-matching (UE24CS352A hackathon)

Noisy GPS tracks → candidate road arcs → binary classifier (P(arc is the true arc)) → HMM/Viterbi → route, served by FastAPI + UI.
This README covers the **ML pipeline (Person 1)**. HMM, API and UI are documented in their own folders.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# put the course data here (NOT committed):
#   data/raw/train/<trip_id>/<trip_id>.{track,nodes,arcs,route}
#   data/raw/student_test/<trip_id>/...
# or point to it: export PATHSNAP_DATA=/path/to/"Student folder"
```

The evaluator-testing trips are never read by any script in `src/`. `common.PARTITIONS` only allows `train` and `student_test`.

## Reproduce everything (in order)

| Step | Command | Time (2 cores) | Output |
|---|---|---|---|
| 1. EDA | `python src/eda.py` | 15 s | `results/eda/*.csv, *.png` |
| 2. Folium maps | `python src/folium_maps.py` | 25 s | `results/maps/*.html` |
| 3. Radius study + label alignment | `python src/radius_study.py` | 40 s | `results/candidates/radius_study_train.*`, `label_alignment_train.csv` |
| 4. Dataset + folds | `python src/build_dataset.py --R 50 --K 30` | 45 s | `data/processed/features.parquet, points.parquet, folds.json` |
| 5. Models | `python src/train_models.py` (or `... logreg lightgbm`) | ~15 min | `results/results.csv`, `results/models/*.json`, `results/preds/`, `artifacts/models/` |
| 6. Bayesian optimisation | `python src/optuna_tune.py --model lightgbm --trials 40 --timeout 2700` | 45 min | `results/optuna/*` + row `lightgbm_tuned` |
| 7. Final artifact | `python src/finalize.py` | 1 min | `artifacts/final_model.joblib`, prediction map |
| 8. Plots + comparison table | `python src/plots.py` | 10 s | `results/plots/*.png`, `results/comparison_table.{csv,md}` |

Seeds are fixed (`SEED = 42`, in `src/common.py`). Every number in the report comes from `results/`.

## Layout

```
src/
  common.py            I/O, per-trip UTM projection, geometry, track cleaning
  eda.py               dataset inventory, quality, timing, connectivity, plots
  folium_maps.py       interactive maps (GPS + network + route [+ predictions])
  labels_candidates.py route->per-point label alignment (DP and greedy), KD-tree candidate search
  radius_study.py      candidate radius / cap selection (train only)
  build_dataset.py     candidates + 17 features + labels -> features.parquet; GroupKFold folds
  evaluate.py          SHARED evaluator (CV + student-test metrics, logs results.csv)
  train_models.py      nearest-arc reference, LogReg, GaussianNB, RandomForest, LightGBM, XGBoost
  optuna_tune.py       TPE Bayesian optimisation with group-aware CV
  finalize.py          exports the final model artifact for the API
  inference.py         raw trip folder -> candidate probabilities (no .route access)
  plots.py             comparison plots and table
results/               all logs, tables, plots, maps (committed)
artifacts/             saved models (final_model.joblib is what the API loads)
report/                Person 1 report sections (2.2, 3.1–3.3, 4)
```

## Key decisions (details in `report/person1_sections.md`)

* **Distances**: per-trip UTM zone (pyproj). Trips cover 25 UTM zones, so a single projection is impossible. Euclidean distance in degrees is never used.
* **Labels**: `.route` is a path (≈5 GPS points per route arc), so labels come from a **monotone DP alignment** of GPS points to the route. The greedy pointer from the brief failed on 5/80 trips (e.g. trip 93: mean error 4.3 km vs 8.7 m for DP).
* **Candidates**: all arcs within **R = 50 m**, nearest **K = 30** kept. Train miss rate is 11.2% pooled (3.2% without parked trip 54), 2.5% per-trip mean; student-test 0.14%.
* **CV**: 5-fold GroupKFold by trip (seed 42), with scalers inside sklearn Pipelines.
