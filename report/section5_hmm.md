# 5. HMM / Viterbi route reconstruction

Code: `src/hmm.py` (model + Viterbi, importable by `api/main.py`), `src/run_hmm.py` (tuning, evaluation, maps).
Reproduce: `python src/run_hmm.py tune && python src/run_hmm.py test && python src/run_hmm.py map 00000091`.
All numbers below are taken from `results/hmm/`.

## 5.1 Why an HMM on top of the classifier
The classifier scores each GPS point **independently**, so it cannot know that a car on arc *a* one second ago must now be on *a* or on an arc reachable from *a*. Its remaining errors on student-test were 47% wrong-direction twins and 34% junction-boundary arcs (§3). Both are errors that **sequence consistency** can fix. The HMM combines the classifier's evidence at each point with a model of how a car can actually move through the directed road network.

## 5.2 Model

**Hidden states.** At GPS point *i*, the hidden state is one of its candidate arcs (directed). We also store where the point projects onto that arc, *s* = metres from the arc's start. Only the 10 highest-probability candidates are kept (`top_k = 10`). This loses the true arc for only 11 of 19,133 student-test points (0.06%) and bounds the work per step to at most 10 × 10 transitions.

**Emissions.** log P(zᵢ | a) = log(p̂ᵢ(a) + 10⁻⁶), where p̂ is the tuned LightGBM probability. We can use it directly as an emission score because its log loss is low (0.0575), i.e. the probabilities are well calibrated. The 10⁻⁶ floor stops a single bad score from making a state permanently impossible (log 0 = −∞).

**Transitions (Newson & Krumm 2009).** For consecutive points *i → i+1* and candidate pair (*a*, s_a) → (*b*, s_b):

  log P(a → b) = − | route_dist − gps_dist | / β

* `gps_dist` = straight-line distance between the two GPS fixes, in UTM metres (the same projection as `common.py`).
* `route_dist` = how far a car must **drive along the directed network** to get from the first matched position to the second:
  * *Different arcs:* (len_a − s_a) + shortest_path(to(a) → from(b)) + s_b. The shortest path comes from Dijkstra over the directed arc graph with arc-length weights. It is cached per source node and the search stops at dt × 60 m/s + 200 m.
  * *Same arc, forward:* s_b − s_a.
  * *Same arc, backward by less than 25 m:* 0, i.e. "the car stood still and GPS jittered". The cost is then |0 − gps_dist|/β: cheap when the car is stopped, expensive when it is moving.
  * *Further backwards, or beyond the search limit:* unreachable.
* **Unreachable** pairs get a finite penalty −(limit + gps_dist)/β − 10. That is worse than any reachable transition, but not −∞, so Viterbi always returns a path even if the network is disconnected around a point.

*Why this form:* if the car really drove from *a* to *b*, the distance it drove should roughly equal the straight-line GPS displacement over one second. A large mismatch means the pair is implausible, for example a 300 m detour or a U-turn. β sets how quickly plausibility drops with the mismatch: smaller β gives stricter routes.

*Why shortest paths and not just "same or directly connected arc":* on highway trips the car covers 50–60 m per 1 s step, while the median arc is about 20 m. One step can cross 2–3 arcs, and a one-hop rule would wrongly forbid the true path. The Dijkstra limit (260 m at dt = 1 s) covers 60 m/s with margin.

*How it fixes direction errors:* a twin pair (u,v)/(v,u) gets identical distance-based features. Switching from (u,v) to (v,u) requires driving to *v* and coming back (a U-turn, route ≫ gps), so it is heavily penalised. Staying on the wrong twin while moving means "driving backwards" along it, which the same-arc rule above also penalises. The decoder therefore keeps the direction consistent with the motion before and after a stop. That is exactly the case where the classifier's heading feature is unreliable.

**Viterbi recurrence (log space).** δ₀(j) = log E₀(j); for each step k:

  δₖ₊₁(j) = maxᵢ [ δₖ(i) + log Tₖ(i, j) ] + log Eₖ₊₁(j),  ψₖ₊₁(j) = argmaxᵢ [ … ]

The final state is argmaxⱼ δ_N(j), and we backtrack with ψ. This is exact dynamic programming: it finds the single most probable state sequence in O(N · K²) time for N points and K = 10 candidates. Log space turns products of many small probabilities into sums, so nothing underflows.

**GPS gaps and missing candidates.** A transition over a long gap is meaningless: after 42 hours, any arc is reachable. The sequence is therefore **cut into segments** wherever dt > `gap_s`, and wherever a point has no candidate within 50 m. Each segment is decoded independently. Points without candidates get no prediction and count as wrong in Top-1. On student-test every trip is a single segment: the largest gap there is 210 s and no point lacks candidates. Segmentation matters in training trips 90 (42 h gap) and 54 (parking cluster without candidates).

## 5.3 Tuning (training trips only)
Only β and `gap_s` were tuned. We used the **16 training trips of fold 0** (39,460 points). For them the LightGBM probabilities were **out-of-fold**: the model was refitted with the tuned hyperparameters on folds 1–4 only, so the HMM saw realistic, not over-confident, emissions. Selection metric: pooled HMM Top-1. Out-of-fold classifier-only Top-1 on these trips is 0.8449. Grid and results: `results/hmm/tuning_train_fold0.csv`.

| β | 0.25 | **0.5** | 1 | 2 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|---|---|
| HMM Top-1 (gap 60 s) | 0.9054 | **0.9058** | 0.8970 | 0.8973 | 0.8908 | 0.8854 | 0.8775 | 0.8677 |

| gap_s (β = 0.5) | 10 | 30 | 60 | **300** |
|---|---|---|---|---|
| HMM Top-1 | 0.9026 | 0.9053 | 0.9058 | **0.9084** |

Chosen: **β = 0.5 m, gap_s = 300 s**. The β optimum lies inside the grid; the first grid started at β = 1, ended at its edge, and was extended downwards. A small β works because at 1 Hz true transitions match the GPS displacement within a few metres. `gap_s` = 300 s is at the edge of its grid, but the curve is flat (±0.3 pp) and student-test has no gap that large, so it does not affect test results.

## 5.4 Results on the 10 student-test trips (`results/hmm/hmm_vs_classifier.csv`, `hmm_per_trip_student_test.csv`)

| Method | Top-1 (pooled) | Top-1 (trip mean) | Route continuity† | Distinct-arc 1-hop‡ | Distinct-arc reachable ≤ 250 m‡ |
|---|---|---|---|---|---|
| Classifier only (tuned LightGBM) | 0.8832 | 0.9008 | 0.9402 | 0.7667 | 0.9370 |
| **Classifier + HMM** | **0.9238** | **0.9375** | **0.9825** | **0.9240** | **1.0000** |

† `evaluate.route_continuity`: consecutive GPS-point predictions are the same arc or directly connected (to(a) = from(b)).
‡ Computed on the sequence of **distinct** predicted arcs (repeats collapsed). *1-hop* = consecutive distinct arcs directly connected. *Reachable* = a directed network path to(a) → from(b) of at most 250 m exists. We report both because on highways two **correct** consecutive predictions can legitimately be several short arcs apart, which a 1-hop test counts as a "break".

* **The HMM improved Top-1 on all 10 trips; none got worse.** Per-trip gains range from +0.3 pp (trip 09) to +5.6 pp (trip 07).
* Every jump in the HMM path is drivable within 250 m. The classifier-only sequence had 6.3% impossible jumps.
* **Run time:** 7.1 s for all 10 trips (19,160 points), i.e. ≈ 0.37 s per 1,000 points and at most 1.5 s per trip, on top of the classifier.

**Error breakdown, student-test (`error_breakdown_student_test.csv`)**

| Category | Classifier | Classifier + HMM |
|---|---|---|
| correct | 88.32% | **92.38%** |
| reverse twin (wrong direction) | 5.38% | 4.15% |
| adjacent arc (junction boundary) | 3.99% | 2.77% |
| other / parallel road | 2.17% | 0.55% |
| true arc not a candidate | 0.14% | 0.14% |

From the transition matrix (`error_transitions_student_test.csv`): the HMM fixed 567 reverse-twin, 469 adjacent and 345 "other" errors. It also **broke 603 points the classifier had right** (302 → reverse twin, 251 → adjacent, 50 → other). The net gain is +778 points.

## 5.5 Where the HMM did not help (honest notes)
* **Trip 00000077 stays the weakest** (0.735 → 0.780). Its reference route contains two U-turns, and the HMM still holds the wrong direction over long stretches. A strongly penalised U-turn plus a wrong early choice can keep a whole stretch on the twin chain. A dedicated U-turn transition probability would be the next step.
* **Trip 00000009 barely changes** (0.883 → 0.886). Its errors are mostly adjacent/parallel arcs where both options are network-consistent, so sequence information cannot separate them.
* **New errors (603 points):** a sequence model trades local accuracy for consistency. Near junctions it sometimes switches arc a few points early or late. Some of these are label-boundary ambiguities (§2.2.3), not true mistakes.
* **Points with no true candidate** (0.14%) cannot be fixed by any decoder.
* **A bug found during development:** in the first version, any backward movement under 25 m on the same arc cost |d| instead of 0. A car driving forward could then "slide backwards" along the wrong-direction twin for free. We found it by inspecting trip 77 on student-test; the student-test set is meant for development comparisons. The fix changes no tuned parameter's role. After the fix, β and gap_s were **re-tuned on the training trips**: the selection was again β = 0.5 and gap_s = 300 s. Student-test Top-1 went from 0.9220 with the buggy rule to 0.9238 after the fix.

## 5.6 Visualisation
* `results/maps/student_test_00000091_hmm.html`: reference route (green), classifier-only Top-1 arcs (red, dashed), HMM path (purple), GPS points (blue), with a layer switcher.
* `results/maps/student_test_00000077_hmm.html`: the hardest trip, showing where wrong-direction errors remain.
