import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.config import MODEL_DIR, OOF_DIR, SEED, N_SPLITS

import joblib


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
        early_stopping_round=100,
        verbose=-1,
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
    )
    if params:
        defaults.update(params)
    return XGBClassifier(**defaults)


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


def train_ensemble(
    features: pd.DataFrame,
    labels: pd.Series,
    models: list[str] | None = None,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
    feature_selection: str = "none",
    feature_top_n: int = 50,
    feature_selection_threshold: str = "median",
    lgbm_params: dict | None = None,
    xgb_params: dict | None = None,
) -> dict:
    if models is None:
        models = ["lgbm", "xgb"]
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = {}
    results["selected_features_per_fold"] = []

    use_lgbm = "lgbm" in models
    use_xgb = "xgb" in models

    if use_lgbm:
        results["lgbm_oof"] = pd.Series(np.full(len(labels), 0.5), index=labels.index, name="lgbm_oof")
        results["lgbm_models"] = []
        results["lgbm_fold_scores"] = []
    if use_xgb:
        results["xgb_oof"] = pd.Series(np.full(len(labels), 0.5), index=labels.index, name="xgb_oof")
        results["xgb_models"] = []
        results["xgb_fold_scores"] = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds.split(features, labels)):
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

        if use_lgbm:
            lgbm = _base_lgbm(seed + fold_idx, lgbm_params)
            lgbm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="logloss")
            lgbm_proba = lgbm.predict_proba(X_val)[:, 1]
            results["lgbm_oof"].iloc[val_idx] = lgbm_proba
            results["lgbm_fold_scores"].append(f1_score(y_val, lgbm_proba > 0.5))
            results["lgbm_models"].append(lgbm)
            results.setdefault("lgbm_feature_cols", []).append(list(X_tr.columns))

        if use_xgb:
            xgb = _base_xgb(seed + fold_idx, xgb_params)
            xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
            xgb_proba = xgb.predict_proba(X_val)[:, 1]
            results["xgb_oof"].iloc[val_idx] = xgb_proba
            results["xgb_fold_scores"].append(f1_score(y_val, xgb_proba > 0.5))
            results["xgb_models"].append(xgb)
            results.setdefault("xgb_feature_cols", []).append(list(X_tr.columns))

    if use_lgbm and use_xgb:
        ensemble_proba = (results["lgbm_oof"] + results["xgb_oof"]) / 2
        results["ensemble_oof"] = ensemble_proba
        results["ensemble_f1_05"] = f1_score(labels, ensemble_proba > 0.5)

    return results


def build_stacking_ensemble(
    results: dict,
    labels: pd.Series,
    seed: int = SEED,
) -> dict:
    """Train a logistic regression on OOF probabilities as a stacking meta-learner."""
    oof_df = pd.DataFrame(index=labels.index)
    model_keys = []
    if "lgbm_oof" in results:
        oof_df["lgbm"] = results["lgbm_oof"]
        model_keys.append("lgbm")
    if "xgb_oof" in results:
        oof_df["xgb"] = results["xgb_oof"]
        model_keys.append("xgb")

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
    lgbm_proba: np.ndarray | None,
    xgb_proba: np.ndarray | None,
    results: dict,
) -> np.ndarray:
    """Combine model probabilities using the configured ensemble mode."""
    mode = results.get("ensemble_mode", "mean")
    if mode == "stacking" and "stacker" in results:
        parts = {}
        if lgbm_proba is not None:
            parts["lgbm"] = lgbm_proba
        if xgb_proba is not None:
            parts["xgb"] = xgb_proba
        if not parts:
            raise RuntimeError("no model probabilities for stacking")
        df = pd.DataFrame(parts)
        return results["stacker"].predict_proba(df)[:, 1]
    # Default: mean
    probas = [p for p in [lgbm_proba, xgb_proba] if p is not None]
    if not probas:
        raise RuntimeError("no model probabilities for ensemble")
    return sum(probas) / len(probas)


def optimize_thresholds(results: dict, labels: pd.Series) -> dict:
    oof_keys = [k for k in ["lgbm_oof", "xgb_oof", "ensemble_oof"] if k in results]
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
    return results


def save_ensemble_artifacts(results: dict, run_tag: str | None = None) -> None:
    prefix = f"{run_tag}_" if run_tag else ""
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if "lgbm_oof" in results:
        results["lgbm_oof"].to_csv(OOF_DIR / f"{prefix}oof_lgbm.csv", index=True, header=True)
    if "xgb_oof" in results:
        results["xgb_oof"].to_csv(OOF_DIR / f"{prefix}oof_xgb.csv", index=True, header=True)
    if "ensemble_oof" in results:
        results["ensemble_oof"].to_csv(OOF_DIR / f"{prefix}oof_ensemble.csv", index=True, header=True)

    for fold_idx, model in enumerate(results.get("lgbm_models", [])):
        joblib.dump(model, MODEL_DIR / f"{prefix}lgbm_fold{fold_idx}.pkl")
    for fold_idx, model in enumerate(results.get("xgb_models", [])):
        joblib.dump(model, MODEL_DIR / f"{prefix}xgb_fold{fold_idx}.pkl")

    if "stacker" in results:
        joblib.dump(results["stacker"], MODEL_DIR / f"{prefix}stacker.pkl")

    for key in ["lgbm_feature_cols", "xgb_feature_cols"]:
        if key in results:
            import json
            with open(MODEL_DIR / f"{prefix}{key}.json", "w") as fc:
                json.dump(results[key], fc, indent=2)

    with open(MODEL_DIR / f"{prefix}ensemble_info.txt", "w") as f:
        for key in ["lgbm_fold_scores", "xgb_fold_scores"]:
            if key in results:
                f.write(f"{key}={results[key]}\n")
        for suffix in ["best_threshold", "best_f1"]:
            for model in ["lgbm", "xgb", "ensemble"]:
                k = f"{model}_oof_{suffix}"
                if k in results:
                    f.write(f"{k}={results[k]:.4f}\n")
        if "stacker_coef" in results:
            f.write(f"stacker_coef={results['stacker_coef']}\n")
        if "selected_features_per_fold" in results:
            f.write(f"n_selected_per_fold={[len(s) for s in results['selected_features_per_fold']]}\n")
