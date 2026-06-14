import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.config import SEED


def _lgbm_objective(trial, X_tr, X_val, y_tr, y_val, seed):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 4000, step=200),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "num_leaves": trial.suggest_int("num_leaves", 15, 95, step=8),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 0.5, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 0.5, log=True),
        "random_state": seed,
        "early_stopping_round": 100,
        "verbose": -1,
    }
    model = LGBMClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="logloss")
    proba = model.predict_proba(X_val)[:, 1]
    return f1_score(y_val, proba > 0.5)


def _xgb_objective(trial, X_tr, X_val, y_tr, y_val, seed):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 4000, step=200),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 0.5, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 0.5, log=True),
        "random_state": seed,
        "early_stopping_rounds": 100,
        "eval_metric": "logloss",
        "verbosity": 0,
    }
    model = XGBClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
    proba = model.predict_proba(X_val)[:, 1]
    return f1_score(y_val, proba > 0.5)


def run_optuna(
    features: pd.DataFrame,
    labels: pd.Series,
    models: list[str],
    n_trials: int = 50,
    seed: int = SEED,
) -> tuple[dict | None, dict | None]:
    """Run Optuna hyperparameter search for LGBM and/or XGBoost.

    Uses a 20% holdout from the full dataset for validation.
    Returns (lgbm_best_params, xgb_best_params) dicts with best params or None.
    """
    import optuna  # lazy import — optuna is an optional dependency

    X_tr, X_val, y_tr, y_val = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=seed + 777,
    )

    lgbm_best = None
    xgb_best = None

    if "lgbm" in models:
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(
            lambda trial: _lgbm_objective(trial, X_tr, X_val, y_tr, y_val, seed),
            n_trials=n_trials,
            show_progress_bar=True,
        )
        lgbm_best = study.best_params
        print(f"  [optuna] lgbm best_f1={study.best_value:.4f}")
        print(f"  [optuna] lgbm best_params={lgbm_best}")

    if "xgb" in models:
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed + 1))
        study.optimize(
            lambda trial: _xgb_objective(trial, X_tr, X_val, y_tr, y_val, seed),
            n_trials=n_trials,
            show_progress_bar=True,
        )
        xgb_best = study.best_params
        print(f"  [optuna] xgb best_f1={study.best_value:.4f}")
        print(f"  [optuna] xgb best_params={xgb_best}")

    return lgbm_best, xgb_best
