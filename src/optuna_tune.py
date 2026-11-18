"""Bayesian hyperparameter optimisation (Optuna TPE) of the chosen model.

Objective : mean macro F1 over the fixed 5-fold GroupKFold on the 80 TRAINING trips
            (student-test and evaluator trips are never touched inside the search).
Sampler   : TPESampler(seed=42, multivariate=True); MedianPruner on per-fold scores.
Outputs   : results/optuna/<model>_trials.csv, <model>_best.json, plots; then the tuned
            model is refitted on all training trips and evaluated on student-test via
            evaluate() (row "<model>_tuned" in results.csv) and saved to artifacts/.
Run: python src/optuna_tune.py --model lightgbm --trials 40 --timeout 2700
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, log_loss
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS, SEED
from evaluate import FEATURES, evaluate, load_data, top1_metrics, top1_table

OUT = RESULTS / "optuna"
OUT.mkdir(parents=True, exist_ok=True)


def space_lightgbm(trial):
    # Ranges centred on the untuned config (lr 0.05, 63 leaves, 400 trees) and widened in both
    # directions: num_leaves / min_child_samples control tree complexity (over/underfitting),
    # subsample / colsample / L1 / L2 are the regularisers; lr x n_estimators trade-off.
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 150, 800, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 400, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        random_state=SEED, n_jobs=-1, verbose=-1)


def space_xgboost(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 150, 800, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        min_child_weight=trial.suggest_float("min_child_weight", 1, 50, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
        gamma=trial.suggest_float("gamma", 1e-3, 5, log=True),
        tree_method="hist", eval_metric="logloss", random_state=SEED, n_jobs=-1)


BUILD = {"lightgbm": (space_lightgbm, LGBMClassifier), "xgboost": (space_xgboost, XGBClassifier)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lightgbm", choices=list(BUILD))
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=2700)
    args = ap.parse_args()
    space, Cls = BUILD[args.model]
    F, P = load_data()
    tr = F[F.partition == "train"]
    folds = sorted(tr.fold.unique())
    splits = [(np.flatnonzero(tr.fold.to_numpy() != k), np.flatnonzero(tr.fold.to_numpy() == k)) for k in folds]
    X, y = tr[FEATURES].to_numpy(), tr.label.to_numpy()

    def objective(trial):
        params = space(trial)
        f1s, lls, t1s = [], [], []
        t0 = time.perf_counter()
        for step, (a, b) in enumerate(splits):
            m = Cls(**params).fit(X[a], y[a])
            p = m.predict_proba(X[b])[:, 1]
            f1s.append(f1_score(y[b], p >= 0.5, average="macro"))
            lls.append(log_loss(y[b], np.clip(p, 1e-7, 1 - 1e-7)))
            k = folds[step]
            t1, _ = top1_metrics(P[(P.partition == "train") & (P.fold == k)], top1_table(tr.iloc[b], p))
            t1s.append(t1["top1"])
            trial.report(float(np.mean(f1s)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("cv_f1_std", float(np.std(f1s)))
        trial.set_user_attr("cv_log_loss", float(np.mean(lls)))
        trial.set_user_attr("cv_top1", float(np.mean(t1s)))
        trial.set_user_attr("seconds", time.perf_counter() - t0)
        return float(np.mean(f1s))

    study = optuna.create_study(direction="maximize", study_name=f"{args.model}_cv_macro_f1",
                                sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True, n_startup_trials=8),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=1))
    # enqueue the untuned configuration (train_models.py) as trial 0 -> tuned vs untuned in one search
    untuned = {"lightgbm": {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 50,
                            "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 0.001},
               "xgboost": {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 7, "min_child_weight": 5,
                           "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "gamma": 0.001}}
    study.enqueue_trial(untuned[args.model])
    t0 = time.time()
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout, gc_after_trial=True)
    total = time.time() - t0
    df = study.trials_dataframe()
    df.to_csv(OUT / f"{args.model}_trials.csv", index=False)
    best = {"model": args.model, "best_value_cv_macro_f1": study.best_value, "best_params": study.best_params,
            "best_trial": study.best_trial.number, "best_user_attrs": study.best_trial.user_attrs,
            "n_trials": len(study.trials),
            "n_complete": sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials),
            "n_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
            "search_seconds": total, "sampler": "TPE(multivariate, seed=42)", "pruner": "MedianPruner",
            "objective": "mean macro F1, 5-fold GroupKFold on 80 training trips"}
    (OUT / f"{args.model}_best.json").write_text(json.dumps(best, indent=2, default=float))
    print(json.dumps(best, indent=2, default=float))
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
        ax = plot_optimization_history(study); ax.figure.set_size_inches(8, 4); ax.figure.tight_layout()
        ax.figure.savefig(OUT / f"{args.model}_history.png", dpi=120); plt.close("all")
        ax = plot_param_importances(study); ax.figure.set_size_inches(8, 4); ax.figure.tight_layout()
        ax.figure.savefig(OUT / f"{args.model}_param_importance.png", dpi=120); plt.close("all")
    except Exception as e:  # plotting is optional
        print("plot failed:", e)
    tuned = Cls(**{**space(optuna.trial.FixedTrial(study.best_params))})
    evaluate(tuned, name=f"{args.model}_tuned", family="Boosting (gradient boosted trees) – Optuna tuned",
             config=study.best_params, author="person1")


if __name__ == "__main__":
    main()
