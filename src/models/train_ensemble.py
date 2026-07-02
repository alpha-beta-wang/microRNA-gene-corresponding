"""集成模型训练模块，支持多模型、特征选择、stacking 和阈值搜索。"""

from typing import Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.config import MODEL_DIR, OOF_DIR, SEED, N_SPLITS

import joblib


class ScaledModel:
    """Wrapper that applies StandardScaler before model predict_proba/predict."""
    def __init__(self, scaler: StandardScaler, model):
        self.scaler = scaler
        self.model = model

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X))

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))


def _selector_lgbm(seed: int) -> LGBMClassifier:
    """Lightweight LGBM for feature importance — no early stopping needed."""
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        random_state=seed,
        verbose=-1,
    )


def _base_lgbm(seed: int, params: dict | None = None) -> LGBMClassifier:
    defaults = dict(
        n_estimators=3000,
        learning_rate=0.01,
        max_depth=6,
        num_leaves=31,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=seed,
        metric="binary_logloss",
        early_stopping_round=100,
        verbose=-1,
        scale_pos_weight=scale_pos_weight,
    )
    if params:
        defaults.update(params)
    return LGBMClassifier(**defaults)


def _base_xgb(seed: int, params: dict | None = None) -> XGBClassifier:
    defaults = dict(
        n_estimators=3000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=seed,
        early_stopping_rounds=100,
        eval_metric="logloss",
        verbosity=0,
        scale_pos_weight=scale_pos_weight,
    )
    if params:
        defaults.update(params)
    return XGBClassifier(**defaults)


def _base_svm(seed: int, params: dict | None = None) -> SVC:
    defaults = dict(
        probability=True,
        kernel="rbf",
        C=1.0,
        gamma="scale",
        random_state=seed,
    )
    if params:
        defaults.update(params)
    return SVC(**defaults)


def _base_rf(seed: int, params: dict | None = None) -> RandomForestClassifier:
    defaults = dict(
        n_estimators=500,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=seed,
        verbose=0,
    )
    if params:
        defaults.update(params)
    return RandomForestClassifier(**defaults)


def _base_et(seed: int, params: dict | None = None) -> ExtraTreesClassifier:
    defaults = dict(
        n_estimators=500,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=seed,
        verbose=0,
    )
    if params:
        defaults.update(params)
    return ExtraTreesClassifier(**defaults)


def _base_fm(seed: int, params: dict | None = None):
    from src.models.factorization_machine import FactorizationMachineClassifier
    defaults = dict(
        n_factors=8,
        learning_rate=0.01,
        epochs=200,
        batch_size=64,
        reg_w=0.001,
        reg_v=0.001,
        random_state=seed,
        verbose=0,
    )
    if params:
        defaults.update(params)
    return FactorizationMachineClassifier(**defaults)


# --- model registry ---
MODEL_REGISTRY: dict[str, dict] = {
    "lgbm": {"factory": _base_lgbm, "needs_eval_set": True, "needs_scaling": False},
    "xgb": {"factory": _base_xgb, "needs_eval_set": True, "needs_scaling": False},
    "svm": {"factory": _base_svm, "needs_eval_set": False, "needs_scaling": True},
    "rf": {"factory": _base_rf, "needs_eval_set": False, "needs_scaling": False},
    "extratrees": {"factory": _base_et, "needs_eval_set": False, "needs_scaling": False},
    "fm": {"factory": _base_fm, "needs_eval_set": False, "needs_scaling": True},
}


def _select_features(
    X_tr: pd.DataFrame,
    X_val: pd.DataFrame,
    y_tr: pd.Series,
    selector_model: object,
    strategy: str,
    top_n: int,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Fit selector on training fold, return reduced train/val DataFrames."""
    selector_model.fit(X_tr, y_tr)
    if hasattr(selector_model, "feature_importances_"):
        importances = selector_model.feature_importances_
    elif hasattr(selector_model, "coef_"):
        importances = np.abs(selector_model.coef_).flatten()
    else:
        return X_tr, X_val, list(X_tr.columns)

    feat_imp = pd.Series(importances, index=X_tr.columns).sort_values(ascending=False)

    if strategy == "top_n":
        selected = feat_imp.head(top_n).index.tolist()
    elif strategy == "select_from_model":
        selected = feat_imp[feat_imp >= threshold].index.tolist()
        max_n = max(5, len(X_tr.columns) // 3)
        if len(selected) > max_n:
            selected = feat_imp.head(max_n).index.tolist()
        if len(selected) < 3:
            selected = feat_imp.head(3).index.tolist()
    else:
        selected = list(X_tr.columns)

    return X_tr[selected], X_val[selected], selected


def _best_threshold(y_true, proba) -> float:
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.1, 0.9, 81):
        cur = f1_score(y_true, (proba >= t).astype(int))
        if cur > best_f1:
            best_f1 = cur
            best_t = float(t)
    return best_t


def train_ensemble(
    features: pd.DataFrame,
    labels: pd.Series,
    models: list[str] | None = None,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
    feature_selection: str = "none",
    feature_top_n: int = 50,
    feature_selection_threshold: str = "median",
    model_params: dict[str, dict] | None = None,
) -> dict:
    if models is None:
        models = ["lgbm", "xgb"]
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = {}
    results["selected_features_per_fold"] = []

    if model_params is None:
        model_params = {}

    # Validate all model names are in registry
    for m in models:
        if m not in MODEL_REGISTRY:
            raise ValueError(f"unknown model '{m}'; available: {list(MODEL_REGISTRY)}")

    # Pre-allocate OOF / models / scores for each model
    for m in models:
        results[f"{m}_oof"] = pd.Series(
            np.full(len(labels), 0.5), index=labels.index, name=f"{m}_oof",
        )
        results[f"{m}_models"] = []
        results[f"{m}_fold_scores"] = []

    for fold_idx, (train_idx, val_idx) in enumerate(split_iter):
        X_tr, X_val = features.iloc[train_idx], features.iloc[val_idx]
        y_tr, y_val = labels.iloc[train_idx], labels.iloc[val_idx]

        # Feature selection within fold (avoids leakage)
        if feature_selection != "none":
            fs_selector = _selector_lgbm(seed + fold_idx)
            X_tr, X_val, selected_cols = _select_features(
                X_tr, X_val, y_tr, fs_selector,
                strategy=feature_selection,
                top_n=feature_top_n,
                threshold=feature_selection_threshold,
            )
            results["selected_features_per_fold"].append(selected_cols)
            print(f"  [fold {fold_idx}] feature_selection: {len(selected_cols)}/{len(features.columns)} cols")

        for m in models:
            cfg = MODEL_REGISTRY[m]
            model = cfg["factory"](seed + fold_idx, model_params.get(m))

            if cfg["needs_scaling"]:
                scaler = StandardScaler()
                X_tr_m = scaler.fit_transform(X_tr)
                X_val_m = scaler.transform(X_val)
                model_for_store = ScaledModel(scaler, model)
            else:
                X_tr_m, X_val_m = X_tr, X_val
                model_for_store = model

            if cfg["needs_eval_set"]:
                model.fit(X_tr_m, y_tr, eval_set=[(X_val_m, y_val)])
            else:
                model.fit(X_tr_m, y_tr)

            proba = model.predict_proba(X_val_m)[:, 1]
            results[f"{m}_oof"].iloc[val_idx] = proba
            results[f"{m}_fold_scores"].append(f1_score(y_val, proba > 0.5))
            results[f"{m}_models"].append(model_for_store)
            results.setdefault(f"{m}_feature_cols", []).append(list(X_tr.columns))

    # Compute ensemble OOF (mean of all trained model OOFs) if at least 2 models
    trained_oofs = [results[f"{m}_oof"] for m in models if f"{m}_oof" in results]
    if len(trained_oofs) >= 2:
        results["ensemble_oof"] = sum(trained_oofs) / len(trained_oofs)
        results["ensemble_f1_05"] = f1_score(labels, results["ensemble_oof"] > 0.5)

    return results


def build_stacking_ensemble(
    results: dict,
    labels: pd.Series,
    seed: int = SEED,
) -> dict:
    """Train a logistic regression on OOF probabilities as a stacking meta-learner."""
    oof_df = pd.DataFrame(index=labels.index)
    model_keys = []
    for key, value in results.items():
        if key.endswith("_oof") and isinstance(value, pd.Series) and key != "ensemble_oof":
            model_key = key[:-4]  # strip "_oof" suffix
            oof_df[model_key] = value
            model_keys.append(model_key)

    if len(model_keys) < 2:
        results["ensemble_mode"] = "mean"
        return results

    stacker = LogisticRegression(random_state=seed, max_iter=1000)
    stacker.fit(oof_df, labels)
    stacking_proba = stacker.predict_proba(oof_df)[:, 1]

    results["ensemble_oof"] = pd.Series(stacking_proba, index=labels.index, name="ensemble_oof")
    results["ensemble_f1_05"] = f1_score(labels, stacking_proba > 0.5)
    results["stacker"] = stacker
    results["stacker_coef"] = dict(zip(model_keys, stacker.coef_[0]))
    results["ensemble_mode"] = "stacking"
    return results


def predict_ensemble_proba(
    model_probas: dict[str, np.ndarray],
    results: dict,
) -> np.ndarray:
    """Combine model probabilities using the configured ensemble mode."""
    mode = results.get("ensemble_mode", "mean")
    if mode == "stacking" and "stacker" in results:
        if not model_probas:
            raise RuntimeError("no model probabilities for stacking")
        df = pd.DataFrame(model_probas)
        return results["stacker"].predict_proba(df)[:, 1]
    # Default: mean
    if not model_probas:
        raise RuntimeError("no model probabilities for ensemble")
    probas = list(model_probas.values())
    return sum(probas) / len(probas)


def optimize_thresholds(results: dict, labels: pd.Series) -> dict:
    oof_keys = [k for k in results
                if k.endswith("_oof") and isinstance(results[k], pd.Series)]
    for key in oof_keys:
        best_t = 0.5
        best_f1 = 0.0
        for t in np.linspace(0.1, 0.9, 101):
            cur = f1_score(labels, (results[key] >= t).astype(int))
            if cur > best_f1:
                best_f1 = cur
                best_t = t
        results[f"{key}_best_threshold"] = best_t
        results[f"{key}_best_f1"] = best_f1

    for prefix in ["lgbm", "xgb", "ensemble"]:
        ts = results[f"{prefix}_fold_thresholds"]
        results[f"{prefix}_nested_threshold"] = float(np.median(ts))
    return results


def save_ensemble_artifacts(results: dict, run_tag: str | None = None) -> None:
    prefix = f"{run_tag}_" if run_tag else ""
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # OOF CSVs: any key ending in _oof
    for key, value in results.items():
        if key.endswith("_oof") and isinstance(value, pd.Series):
            value.to_csv(OOF_DIR / f"{prefix}{key}.csv", index=True, header=True)

    # Model .pkl: any key ending in _models
    for key, model_list in results.items():
        if key.endswith("_models") and isinstance(model_list, list):
            model_name = key[:-7]  # strip "_models"
            for fold_idx, model in enumerate(model_list):
                joblib.dump(model, MODEL_DIR / f"{prefix}{model_name}_fold{fold_idx}.pkl")

    # Stacker
    if "stacker" in results:
        joblib.dump(results["stacker"], MODEL_DIR / f"{prefix}stacker.pkl")

    # Feature columns: any key ending in _feature_cols
    for key, cols in results.items():
        if key.endswith("_feature_cols") and isinstance(cols, list):
            import json
            with open(MODEL_DIR / f"{prefix}{key}.json", "w") as fc:
                json.dump(cols, fc, indent=2)

    # Info file
    with open(MODEL_DIR / f"{prefix}ensemble_info.txt", "w") as f:
        for key in sorted(results):
            if key.endswith("_fold_scores"):
                f.write(f"{key}={results[key]}\n")
        for suffix in ["best_threshold", "best_f1"]:
            for key in sorted(results):
                if key.endswith(f"_oof_{suffix}"):
                    f.write(f"{key}={results[key]:.4f}\n")
        if "stacker_coef" in results:
            f.write(f"stacker_coef={results['stacker_coef']}\n")
        if "selected_features_per_fold" in results:
            f.write(f"n_selected_per_fold={[len(s) for s in results['selected_features_per_fold']]}\n")
"""集成模型训练模块，支持多模型、特征选择、stacking 和阈值搜索。"""
