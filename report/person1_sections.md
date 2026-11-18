# PathSnap – Technical report: Person 1 sections

All numbers come from files in `results/` and can be regenerated with the commands in `README.md`.
Partitions: **train** = 80 trips, **student-test** = 10 trips. The evaluator partition was never read.

---

## 2.2 Dataset Understanding and Exploratory Data Analysis

### 2.2.1 How the raw data is organised (train + student-test, `results/eda/inventory.csv`)

Each trip folder holds four headerless, whitespace-separated files. We verified that all 90 trips have all four files with the expected number of columns.

| File | Columns (verified) | Meaning |
|---|---|---|
| `.track` | lon, lat, t (float) | GPS observations, ≈1 Hz |
| `.nodes` | lon, lat (float) | road-network nodes, `node_id` = 0-based row |
| `.arcs` | from_node, to_node (int) | **directed** road segments, `arc_id` = 0-based row |
| `.route` | arc_id (int) | chronological **path** of reference arcs, *not* one label per GPS point |

Each trip has its **own** road network (a local extract around the trip). Networks differ between trips: different md5 hashes and 7k–307k arcs. Trips are spread over **25 UTM zones** (Europe, Russia, Central Asia; `trip_locations.png`).

### 2.2.2 Data-quality findings (`results/eda/partition_summary.csv`, `trip_stats.csv`)

| Check | Train (80 trips) | Student-test (10 trips) |
|---|---|---|
| GPS points | 208,417 (361 – 19,313 per trip, median 1,529) | 19,160 (262 – 3,958, median 1,420) |
| Missing / NaN / out-of-range coordinates | 0 | 0 |
| Non-increasing timestamps | 0 | 0 |
| Steps exactly 1 s apart | 93.8% (trip min 53%) | 98.2% |
| Gaps > 5 s / > 30 s | 1,216 / 133 | 44 / 2 |
| Largest gap | 153,064 s (trip 00000090, ≈42 h) | 210 s |
| Speed spikes > 70 m/s | 148 (95 in trip 00000013) | 1 |
| Nodes / arcs per trip (median) | 9,608 / 19,527 | 6,597 / 13,145 |
| Isolated nodes | 0 | 0 |
| Weakly / strongly connected components (median) | 20 / 127 | 11 / 54 |
| Largest weak component (mean share of nodes) | 96.9% | 96.4% |
| Two-way arcs (reverse arc exists) | 92% | 93% |
| Duplicate arcs / self-loops / zero-length arcs | in 60 / 9 / some trips | present |
| Route continuity (to(aᵢ) = from(aᵢ₊₁)) | min 0.981 (10 trips < 1) | min 0.997 (1 trip < 1) |
| GPS → nearest route arc, median / p95 | 4.5 m / 14.2 m | 4.6 m / 15.8 m |

**Patterns and anomalies**
* `len(.route) < len(.track)` in **every** trip (median ≈ 5.3 GPS points per route arc), so `.route` cannot be used as per-point labels directly (§2.2.3).
* Every directed arc on a two-way street has a geometrically identical twin. Distance alone cannot tell them apart, so heading has to.
* **Trip 00000054** (19,313 points, 9% of all training points): the vehicle stood still in a parking area off the network for hours before and after the route (`gps_trajectories.png`, `maps/train_00000054_anomaly.html`). 85% of its points have no arc within 50 m.
* **Trip 00000033**: 59% of GPS points are more than 50 m from every reference-route arc, although GPS is on roads (0.8% without candidates). The reference route and the GPS disagree for long stretches, so its labels are unreliable.
* Small route breaks (continuity < 1) in 11 trips show that the reference route itself has errors. As instructed, we treat it as reference information, not verified ground truth.

### 2.2.3 Preprocessing

1. **Coordinates**: raw order is (lon, lat). Each trip is projected to its own UTM zone (`pyproj`, EPSG 326xx/327xx chosen from the trip's median lon/lat), so all distances and bearings are in metres or degrees on a conformal projection. Distortion inside one zone is < 0.1%, far below GPS noise. Distances in degrees are never used. Haversine is used only in EDA summaries.
2. **Cleaning** (`common.clean_track`): drop NaN, out-of-range, (0,0) and non-increasing-time rows. None occurred in train or student-test, but the API needs the guard. Speed spikes and long gaps are **kept**: dropping points would remove positions the API must still match, so we expose them as features (`dt_prev`, `speed_ms`) and let the HMM handle them.
3. **Per-point labels from the route path**: we align time-ordered GPS points to the route with a **monotone dynamic-programming alignment**. It picks route indices j₁ ≤ j₂ ≤ … ≤ j_N that minimise Σ dist(pointᵢ, route[jᵢ]), the same "pointer only moves forward" constraint as the greedy method but globally optimal. Compared with the greedy pointer (look-ahead 5) on train (`results/candidates/label_alignment_train.csv`): median agreement 96.9%, but greedy failed badly on 5/80 trips (trip 93: mean label distance 4,350 m vs 8.7 m for DP; trip 34: 602 m vs 7.2 m) because it cannot recover once it falls behind.
   *Limitations*: (i) at junctions, a point between two consecutive route arcs is assigned to whichever is closer, so labels near arc boundaries are ambiguous; (ii) when the route is wrong (trips 33, 54, small breaks), the labels inherit the error; (iii) route arcs shorter than one GPS step get no point (median 23 per trip). We report all metrics against these derived labels and do not claim they are perfect ground truth.
4. **Candidate generation** (`labels_candidates.ArcIndex`): a KD-tree over points sampled every 10 m along each arc, followed by exact point-to-segment distance. Candidates = all arcs within **R = 50 m**, keeping the nearest **K = 30**.

**Radius study (train only, `results/candidates/radius_study_train.csv`, K = 30):**

| R (m) | candidate miss rate (pooled) | per-trip mean miss | points with 0 candidates | mean candidates / point |
|---|---|---|---|---|
| 20 | 14.5% | 5.3% | 11.2% | 3.6 |
| 30 | 12.5% | 3.3% | 9.7% | 5.2 |
| 40 | 11.7% | 2.8% | 9.0% | 6.8 |
| **50** | **11.2%** | **2.5%** | **8.8%** | **8.5** |
| 75 | 10.5% | 2.1% | 6.5% | 12.3 |
| 100 | 10.2% | 2.0% | 0.7% | 15.6 |

Since the GPS → route p95 distance is about 15 m, R = 50 m covers more than 3× the typical noise. Going to 100 m cuts the miss rate by only 1 pp but almost doubles the candidates (and HMM states), adding mostly parallel-street distractors. Nearly all remaining misses come from trip 54's parking cluster: without it, the pooled train miss rate is 3.2%; the median trip has 0%. The K = 30 cap costs only 0.02 pp. **Student-test miss rate at R = 50: 0.14%** (`candidate_summary.csv`).

Result: 1,772,015 training rows (10.4% positive) and 200,052 student-test rows (9.6% positive).

### 2.2.4 Features (17 per (point, candidate); `build_dataset.FEATURES`)

| Feature | Why |
|---|---|
| `dist_m` | perpendicular distance from GPS point to segment; the basic emission signal |
| `dist_rank`, `dist_gap_min` | distance *relative to the other candidates*; removes per-point GPS bias (a 20 m distance means different things in an open and a dense area) |
| `n_cands` | candidate density: junctions and dense city blocks are harder |
| `heading_diff` | \|GPS bearing − arc bearing\| ∈ [0°,180°]; the **only** signal that separates a directed arc from its reverse twin |
| `heading_diff_win5` | the same with a ±2-point motion vector, less sensitive to jitter |
| `heading_reliable`, `speed_ms` | a heading is meaningless while stationary (< 2 m moved), so the last reliable heading is carried forward and flagged; speed also hints at road class |
| `turn_rate` | heading change per second, high near junctions where adjacent arcs compete |
| `arc_len_m` | long arcs = major roads; very short arcs = junction connectors |
| `along_t` | where the projection falls on the arc (0/1 = endpoint: the point may belong to the next arc) |
| `dt_prev` | time since previous fix; after gaps, context features are less reliable |
| `dist_prev_pt`, `dist_next_pt`, `dist_win5` | distance of neighbouring fixes to the **same** arc: temporal smoothing that suppresses single-point noise |
| `node_deg_sum` | degree of the arc's end nodes: junction complexity / road-class proxy |
| `is_two_way` | whether a reverse twin exists, so the model knows when heading matters |

No feature uses the route or labels, so the same code (`build_dataset.candidate_features`) runs at API inference time. We verified on all 10 student-test trips that the raw-folder inference path reproduces the training-path probabilities exactly (`results/inference_consistency_check.csv`).

**Visualisations**: `results/eda/trip_distributions.png`, `sampling_intervals.png`, `trip_locations.png`, `gps_trajectories.png` (6 train trips, GPS + route + network), `results/candidates/radius_study_train.png`. **Folium maps**: `results/maps/train_00000001.html` (GPS points, network within 150 m, reference route, start/end markers, legend, layer control, minimap), `train_00000054_anomaly.html` (parking-cluster anomaly), `student_test_00000007.html`, and `student_test_00000077_predictions.html` (classifier predictions vs reference).

---

## 3. Model Experimentation and Comparison

**Protocol (identical for every model, `src/evaluate.py`)**
* 5-fold **GroupKFold by trip** on the 80 training trips (seed 42, 16 trips per fold, `folds.json`). Scalers and transforms sit inside sklearn Pipelines, so they are fitted on training folds only.
* Refit on all 80 training trips, then evaluate on the 10 student-test trips.
* **Primary selection metric: CV macro F1** (mean over folds), with CV log loss, fold stability and inference time as secondary criteria. Selection never uses the student-test set; it is for comparison only.
* **Top-1** = for each GPS point, the highest-probability candidate is compared with the aligned reference arc. Points whose true arc is not a candidate count as wrong.
* **Route continuity (classifier-only)** = share of consecutive GPS-point pairs whose Top-1 arcs are identical or directly connected (`to(a) = from(b)`). The HMM is evaluated on the same definition by Person 2.
* Macro F1, precision, recall and the confusion matrix use the candidate-level threshold 0.5. Log loss uses the candidate probabilities.

### 3.1 Overall model comparison

| Model          | Algorithm family                                 | Key configuration                                                                                                                                     | CV macro F1 (mean±std)   |   CV Top-1 |   Student-test Top-1 |   Route continuity (clf-only) |   Student-test macro F1 |   Precision |   Recall |   Log loss |   Train macro F1 |   Train time (s) |   Inference (ms / 1k pts) |
|:---------------|:-------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------|-----------:|---------------------:|------------------------------:|------------------------:|------------:|---------:|-----------:|-----------------:|-----------------:|--------------------------:|
| lightgbm_tuned | Boosting (gradient boosted trees) – Optuna tuned | n_estimators=450, learning_rate=0.039, num_leaves=76, min_child_samples=59, subsample=0.997, colsample_bytree=0.694, reg_lambda=0.03, reg_alpha=3.795 | 0.9267 ± 0.0154          |     0.7921 |               0.8832 |                        0.9402 |                  0.9323 |      0.8935 |   0.8617 |     0.0575 |           0.9507 |             42.6 |                    181.63 |
| lightgbm       | Boosting (gradient boosted trees)                | n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=50, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0                        | 0.9262 ± 0.0158          |     0.7918 |               0.8814 |                        0.939  |                  0.9311 |      0.889  |   0.8619 |     0.058  |           0.9507 |             33.4 |                    129.11 |
| xgboost        | Boosting (gradient boosted trees)                | n_estimators=400, learning_rate=0.05, max_depth=7, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method=hist          | 0.9256 ± 0.0155          |     0.7918 |               0.8831 |                        0.9406 |                  0.9328 |      0.8917 |   0.8651 |     0.0572 |           0.9481 |             33   |                     30.88 |
| random_forest  | Ensemble (bagging)                               | n_estimators=100, max_depth=None, min_samples_leaf=5, max_features=sqrt, max_samples=0.2                                                              | 0.9235 ± 0.0158          |     0.7875 |               0.8776 |                        0.9344 |                  0.9297 |      0.8977 |   0.8485 |     0.0602 |           0.9421 |             98   |                     43.94 |
| logreg         | Linear / probabilistic                           | scaler=StandardScaler, C=1.0, penalty=l2, solver=lbfgs, max_iter=1000, class_weight=None                                                              | 0.9013 ± 0.0139          |     0.7814 |               0.8695 |                        0.9311 |                  0.9118 |      0.8448 |   0.836  |     0.0869 |           0.9053 |              4.3 |                      0.89 |
| gaussian_nb    | Linear / probabilistic (generative)              | transform=QuantileTransformer(normal), var_smoothing=0.0                                                                                              | 0.8055 ± 0.0104          |     0.7477 |               0.8388 |                        0.8961 |                  0.8066 |      0.5248 |   0.902  |     0.6234 |           0.8054 |              4.1 |                     18.55 |
| nearest_arc    | Heuristic (non-learned reference)                | rule=pick candidate with smallest perpendicular distance                                                                                              | 0.7508 ± 0.0099          |     0.5087 |               0.5442 |                        0.8201 |                  0.7475 |      0.5113 |   0.5878 |     0.2573 |           0.7527 |              0.1 |                      0.14 |

*Notes.* `nearest_arc` is a non-learned reference (pick the closest arc), not one of the five models. *CV Top-1* is lower than *student-test Top-1* for every model because training folds contain trips 54 and 33, where most true arcs are outside the candidate set (fold 2 Top-1 ≈ 0.57 for all models). On covered points, CV Top-1 is 0.880 for LightGBM, matching student-test 0.881. Route continuity is reported here for classifier-only predictions; the HMM continuity is in Person 2's section.

### 3.2 Per-model answers (Person 1's models)

#### Logistic Regression (baseline): `logreg`
**A1. Why suitable.** The task is a binary "is this the true arc?" decision on a few strongly monotone signals (closer is better, smaller heading difference is better). LR is the simplest model that gives **calibrated probabilities**, which the HMM needs as emission scores.
**A2. Strengths / limits.** It is fast (4 s to train, 0.9 ms per 1k points to infer), its coefficients are interpretable, and it is stable across folds (std 0.014). It cannot model interactions: heading should only matter when `is_two_way` = 1 and `heading_reliable` = 1, and that conditional logic is unavailable to a linear model. The largest standardised coefficients were `dist_rank` −4.31, `dist_gap_min` −2.67, `heading_diff_win5` −1.39 and `dist_win5` −1.15. The model leans on relative distance and smoothed heading.
**B5. Hyperparameters.** StandardScaler; L2 penalty, C = 1.0, solver lbfgs, max_iter 1000, no class weighting (to keep probabilities calibrated for the HMM).
**B6. Objective.** Binary cross-entropy (log loss) + L2 penalty.
**C7/C8. Student-test.** Top-1 **0.8695**, macro F1 **0.9118**, precision 0.845, recall 0.836, log loss **0.0869**, route continuity 0.931.
**C9. Over/underfitting.** Train macro F1 0.9053 ≈ CV 0.9013 ≈ test 0.9118, so there is no overfitting but there is mild **underfitting**: every tree model beats it by about 0.02 macro F1 and has lower log loss (0.058 vs 0.087). That gap is the cost of the linear boundary.

#### Gaussian Naive Bayes: `gaussian_nb`
**A1. Why suitable.** It is a generative probabilistic model: it estimates class-conditional densities P(features | true arc), which is conceptually the HMM emission density. It also trains in seconds on 1.8 M rows.
**A2. Strengths / limits.** It is fast and needs very little data, but it assumes features are independent given the class. Ours are strongly correlated by design (`dist_m`, `dist_prev_pt`, `dist_next_pt`, `dist_win5`; `heading_diff`, `heading_diff_win5`), so NB counts the same evidence several times and becomes over-confident.
**B5. Hyperparameters.** QuantileTransformer to a normal output (1000 quantiles, fitted on training folds) so the skewed distance features are closer to Gaussian; `var_smoothing` = 1e-9.
**B6. Objective.** Maximum likelihood of the per-class Gaussian means and variances (closed form).
**C7/C8. Student-test.** Top-1 **0.8388**, macro F1 **0.8066**, precision **0.525**, recall **0.902**, log loss **0.6234**, route continuity 0.896.
**C9. Over/underfitting.** Train 0.8054 ≈ CV 0.8055 ≈ test 0.8066: no overfitting, clear **underfitting / model bias**. The confusion matrix shows 15,626 false positives against 17,258 true positives: double-counted evidence pushes many wrong candidates above 0.5. Its log loss is 7× LR's, so its probabilities are poor HMM emissions. The ranking is better than its F1 suggests (Top-1 0.839), because Top-1 only needs the order of candidates, not their calibration.

#### Random Forest: `random_forest`
**A1. Why suitable.** It learns non-linear rules and interactions (e.g. "heading matters only on two-way arcs while moving") with no scaling needed, and bagging gives stable averaged probabilities.
**A2. Strengths / limits.** It is robust to outliers (speed spikes, the 42 h gap) and to correlated features. On the other hand it is the slowest to train (98 s even with 20% bootstrap samples), the saved model is 82 MB (excluded from git), and its averaged leaf frequencies are less sharp than boosting's.
**B5. Hyperparameters.** n_estimators 100, max_depth None, min_samples_leaf 5, max_features "sqrt", max_samples 0.2 (bootstrap fraction, for speed on 1.77 M rows).
**B6. Objective.** Each tree greedily minimises Gini impurity; the forest averages tree probabilities.
**C7/C8. Student-test.** Top-1 **0.8776**, macro F1 **0.9297**, precision 0.898, recall 0.849, log loss **0.0602**, route continuity 0.934.
**C9. Over/underfitting.** Train 0.9421 vs CV 0.9235 (gap 0.019): **mild overfitting**, held in check by `min_samples_leaf` = 5 and bootstrap subsampling. CV and test agree (0.9235 vs 0.9297), so it generalises across trips.

#### LightGBM: `lightgbm` (selected for tuning)
**A1. Why suitable.** Gradient-boosted trees model the feature interactions the linear models miss, and are among the strongest learners for tabular data. Histogram training handles 1.8 M rows in about 33 s on 2 cores, and the log-loss objective gives well-calibrated probabilities for the HMM.
**A2. Strengths / limits.** It had the best CV macro F1, CV log loss and covered Top-1 of all models. It has many interacting hyperparameters (hence Bayesian optimisation) and can overfit with large `num_leaves`. Its inference (129 ms per 1k points on 2 threads) is slower than XGBoost's, but a 4,000-point trip still takes about 0.5 s. By gain importance, the top features are `dist_rank` 0.42, `heading_diff_win5` 0.23, `dist_gap_min` 0.14, `along_t` 0.05 and `dist_win5` 0.04 (`plots/feature_importance_lightgbm.png`): relative distance plus smoothed heading dominate, as expected.
**B5. Initial hyperparameters.** n_estimators 400, learning_rate 0.05, num_leaves 63, min_child_samples 50, subsample 0.8 (freq 1), colsample_bytree 0.8, reg_lambda 1.0.
**B6. Objective.** Binary log loss, minimised by gradient boosting (second-order, leaf-wise growth).
**C7/C8. Student-test.** Top-1 **0.8814**, macro F1 **0.9311**, precision 0.889, recall 0.862, log loss **0.0580**, route continuity 0.939.
**C9. Over/underfitting.** Train 0.9507 vs CV 0.9262 (gap 0.025): **mild overfitting**, the largest among our models, but CV, test (0.9311) and covered Top-1 all agree, so it is not hurting generalisation. Fold scores range 0.909–0.955 (std 0.016, `plots/cv_folds.png`). The spread comes from trip difficulty, not model instability; every model shows the same fold pattern.

#### XGBoost: `xgboost`
**A1. Why suitable.** It is a second boosting implementation (level-wise trees with `max_depth` instead of leaf-wise growth), which checks whether LightGBM's lead is about the algorithm family or the library.
**A2. Strengths / limits.** It matches LightGBM within noise and has the fastest tree inference (31 ms per 1k points). It has a similar hyperparameter burden.
**B5. Hyperparameters.** n_estimators 400, learning_rate 0.05, max_depth 7, min_child_weight 5, subsample 0.8, colsample_bytree 0.8, reg_lambda 1.0, tree_method "hist".
**B6. Objective.** `binary:logistic` (log loss) with L2 regularisation on leaf weights.
**C7/C8. Student-test.** Top-1 **0.8831**, macro F1 **0.9328**, precision 0.892, recall 0.865, log loss **0.0572**, route continuity 0.941.
**C9. Over/underfitting.** Train 0.9481 vs CV 0.9256 (gap 0.023): mild overfitting, same picture as LightGBM.

### Model selection (Person 1 + Person 2 to confirm with MLP/KNN/SVM results)
LightGBM has the highest **CV macro F1** (0.9262 ± 0.0158), the lowest **CV log loss** (0.0725) and the highest **CV covered Top-1** (0.880). XGBoost is within 0.001 on all three, far inside one fold-std. We chose LightGBM for Bayesian optimisation on these CV criteria alone, without looking at student-test (where XGBoost happens to be 0.002 ahead, also within noise). Both produce well-calibrated probabilities (test log loss ≈ 0.057) suitable as HMM emissions. GNB is ruled out by its calibration (log loss 0.62), and LR is a solid, fast fallback. If inference speed became critical, XGBoost is an equivalent drop-in: `optuna_tune.py --model xgboost`.

### Error analysis (student-test, classifier-only, `results/error_analysis_*.csv`)
| | nearest_arc | LightGBM |
|---|---|---|
| correct | 54.4% | 88.1% |
| reverse twin (wrong direction) | 34.7% | 5.5% (47% of errors) |
| adjacent arc (shares a node, junction boundary) | 6.2% | 4.0% (34%) |
| other / parallel road | 4.6% | 2.2% (18%) |
| true arc not a candidate | 0.14% | 0.14% |

Heading features remove 84% of the direction confusions. What remains is mostly direction at stops (no reliable heading) and junction-boundary ambiguity, which is in part label ambiguity (§2.2.3). Both are what the HMM's connectivity constraint is designed to fix.

### 3.3 Visualisations
* `results/plots/macro_f1_comparison.png`: train vs CV (± std) vs student-test macro F1 per model (over/underfitting at a glance)
* `results/plots/top1_comparison.png`: CV/test Top-1 and classifier-only route continuity
* `results/plots/logloss_inference.png`: probability quality and inference cost
* `results/plots/cv_folds.png`: per-fold stability
* `results/plots/confusion_matrices.png`: student-test confusion matrices for all models
* `results/plots/feature_importance_{lightgbm,lightgbm_tuned,xgboost,random_forest}.png`
* `results/maps/student_test_00000077_predictions.html`: Folium map of tuned-LightGBM Top-1 arcs (red, dashed) over the reference route (green) and GPS (blue)
* No train/validation loss curves are shown for our models: none of them trains in epochs. Boosting iterations could be plotted, but CV curves were not recorded, so we do not draw any. The MLP curve is in Person 2's section.

---

## 4. Bayesian Hyperparameter Optimisation

**Model tuned:** LightGBM, chosen on CV criteria (§3, Model selection). Code: `src/optuna_tune.py`; all trials: `results/optuna/lightgbm_trials.csv`; best: `lightgbm_best.json`.

**Search strategy.** Optuna TPE sampler (multivariate, seed 42, 8 random start-up trials), MedianPruner on the running per-fold mean (stop a trial after fold ≥ 2 if it is below the median of earlier trials at the same fold). Budget: 40 trials or 45 min, whichever came first. Trial 0 was the **untuned configuration**, so tuned and untuned are compared inside the same search.

**Objective.** Maximise the mean macro F1 over the same fixed 5-fold GroupKFold on the 80 training trips that `evaluate()` uses. Student-test and evaluator trips were never used inside the search. CV log loss and CV Top-1 were recorded per trial as user attributes.

**Search space and justification**

| Hyperparameter | Range | Why |
|---|---|---|
| n_estimators | 150–800 (step 50) | trades off with learning_rate; around the untuned 400 |
| learning_rate | 0.02–0.2 (log) | smaller = smoother fit, needs more trees |
| num_leaves | 15–255 (log) | main complexity knob of leaf-wise trees; the untuned model overfits mildly (train−CV gap 0.025) |
| min_child_samples | 10–400 (log) | minimum leaf size: regularises against fitting single noisy GPS points |
| subsample | 0.5–1.0 | row bagging (regulariser) |
| colsample_bytree | 0.4–1.0 | feature bagging: several features are highly correlated |
| reg_lambda | 1e-3–30 (log) | L2 on leaf values |
| reg_alpha | 1e-3–10 (log) | L1 on leaf values |

**Trials.** 21 trials ran in 46.7 min on 2 CPU cores: 13 completed and 8 pruned. Completed trials scored CV macro F1 0.9239–0.9267 (mean 0.9259); a full 5-fold trial took 182 s on average. Plots: `results/optuna/lightgbm_history.png`, `lightgbm_param_importance.png`. Hyperparameter importance (fANOVA): reg_lambda 0.67, min_child_samples 0.18, all others ≤ 0.06.

**Best configuration (trial 15).** n_estimators 450, learning_rate 0.0394, num_leaves 76, min_child_samples 59, subsample 0.997, colsample_bytree 0.694, reg_lambda 0.0303, reg_alpha 3.795. It was refitted on all 80 training trips and evaluated once on student-test via `evaluate()` (row `lightgbm_tuned`). This is the model exported to `artifacts/final_model.joblib`.

**Tuned vs untuned (`results/optuna/tuned_vs_untuned.csv`)**

| Metric | Untuned | Tuned | Δ |
|---|---|---|---|
| CV macro F1 | 0.9262 | 0.9267 | +0.0005 |
| CV macro F1 std | 0.0158 | 0.0154 | -0.0003 |
| CV Top-1 | 0.7918 | 0.7921 | +0.0003 |
| CV log loss | 0.0725 | 0.0721 | -0.0003 |
| Train macro F1 | 0.9507 | 0.9507 | +0.0000 |
| Student-test Top-1 | 0.8814 | 0.8832 | +0.0018 |
| Student-test macro F1 | 0.9311 | 0.9323 | +0.0011 |
| Precision | 0.8890 | 0.8935 | +0.0045 |
| Recall | 0.8619 | 0.8617 | -0.0003 |
| Student-test log loss | 0.0580 | 0.0575 | -0.0005 |
| Route continuity (clf-only) | 0.9390 | 0.9402 | +0.0011 |
| Training time (s) | 33.4496 | 42.5904 | +9.1408 |
| Inference (ms / 1k points) | 129.1126 | 181.6341 | +52.5215 |

**Interpretation.** Tuning gave a small but consistent improvement: CV macro F1 +0.0005 with lower fold std, CV log loss −0.0003, student-test Top-1 +0.18 pp, macro F1 +0.0011 and precision +0.45 pp. All of these are well inside one fold standard deviation (0.015), so the honest conclusion is that **the untuned LightGBM was already near this feature set's ceiling**. The remaining errors are direction-at-stops and junction-boundary cases (§3 error analysis), which need sequence context (HMM) or better labels rather than different tree hyperparameters. The search favoured stronger L1 regularisation (reg_alpha ≈ 3.8) and feature subsampling (0.69), consistent with the correlated feature groups. The train macro F1 is unchanged (0.9507), so the train−CV gap shrank slightly to 0.024.

**Compute cost.** Search: 46.7 min wall-clock on 2 cores (≈ 81 LightGBM fits; pruning saved roughly 8 × 3 = 24 min). The final refit took 42.6 s versus 33.4 s untuned. Inference is slower (182 vs 129 ms per 1k points: more and deeper trees), but a 4,000-point trip is still scored end-to-end, including candidate generation, in about 1 s (`results/inference_consistency_check.csv`).

