"""Person 1 model experiments. Every model uses the SAME features, folds and evaluate().

Run all:   python src/train_models.py
Run some:  python src/train_models.py logreg lgbm
Initial (untuned) hyperparameters are listed here and logged to results.csv.
"""
import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from common import SEED
from evaluate import NearestArcBaseline, evaluate

MODELS = {
    "nearest_arc": dict(
        model=NearestArcBaseline(), family="Heuristic (non-learned reference)",
        config={"rule": "pick candidate with smallest perpendicular distance"}),
    "logreg": dict(
        model=make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
        family="Linear / probabilistic",
        config={"scaler": "StandardScaler", "C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 1000,
                "class_weight": None}),
    "gaussian_nb": dict(
        # NB assumes per-class Gaussian features; a quantile->normal transform makes skewed
        # distance features closer to Gaussian (fitted on training folds only, inside the pipeline).
        model=make_pipeline(QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                                                subsample=200_000, random_state=SEED), GaussianNB()),
        family="Linear / probabilistic (generative)",
        config={"transform": "QuantileTransformer(normal)", "var_smoothing": 1e-9}),
    "random_forest": dict(
        model=RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_leaf=5, max_features="sqrt",
                                     max_samples=0.2, n_jobs=-1, random_state=SEED),
        family="Ensemble (bagging)",
        config={"n_estimators": 100, "max_depth": None, "min_samples_leaf": 5, "max_features": "sqrt",
                "max_samples": 0.2}),
    "lightgbm": dict(
        model=LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=50,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                             random_state=SEED, n_jobs=-1, verbose=-1),
        family="Boosting (gradient boosted trees)",
        config={"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 50,
                "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0}),
    "xgboost": dict(
        model=XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=7, min_child_weight=5,
                            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist",
                            eval_metric="logloss", random_state=SEED, n_jobs=-1),
        family="Boosting (gradient boosted trees)",
        config={"n_estimators": 400, "learning_rate": 0.05, "max_depth": 7, "min_child_weight": 5,
                "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "tree_method": "hist"}),
}


if __name__ == "__main__":
    names = sys.argv[1:] or list(MODELS)
    for n in names:
        spec = MODELS[n]
        print(f"=== {n}", flush=True)
        evaluate(spec["model"], name=n, family=spec["family"], config=spec["config"], author="person1")
