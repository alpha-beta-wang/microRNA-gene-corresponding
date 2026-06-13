import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.config import MODEL_DIR, OOF_DIR, SEED, N_SPLITS

import joblib


def _base_lgbm(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
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


def _base_xgb(seed: int) -> XGBClassifier:
    return XGBClassifier(
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


def train_ensemble(
    features: pd.DataFrame,
    labels: pd.Series,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
) -> dict:
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = {
        "lgbm_oof": pd.Series(np.full(len(labels), 0.5), index=labels.index, name="lgbm_oof"),
        "xgb_oof": pd.Series(np.full(len(labels), 0.5), index=labels.index, name="xgb_oof"),
        "lgbm_models": [],
        "xgb_models": [],
        "lgbm_fold_scores": [],
        "xgb_fold_scores": [],
    }

    for fold_idx, (train_idx, val_idx) in enumerate(folds.split(features, labels)):
        X_tr, X_val = features.iloc[train_idx], features.iloc[val_idx]
        y_tr, y_val = labels.iloc[train_idx], labels.iloc[val_idx]

        lgbm = _base_lgbm(seed + fold_idx)
        lgbm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="logloss")
        lgbm_proba = lgbm.predict_proba(X_val)[:, 1]
        results["lgbm_oof"].iloc[val_idx] = lgbm_proba
        results["lgbm_fold_scores"].append(f1_score(y_val, lgbm_proba > 0.5))
        results["lgbm_models"].append(lgbm)

        xgb = _base_xgb(seed + fold_idx)
        xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        xgb_proba = xgb.predict_proba(X_val)[:, 1]
        results["xgb_oof"].iloc[val_idx] = xgb_proba
        results["xgb_fold_scores"].append(f1_score(y_val, xgb_proba > 0.5))
        results["xgb_models"].append(xgb)

    ensemble_proba = (results["lgbm_oof"] + results["xgb_oof"]) / 2
    results["ensemble_oof"] = ensemble_proba
    results["ensemble_f1_05"] = f1_score(labels, ensemble_proba > 0.5)

    return results


def optimize_thresholds(results: dict, labels: pd.Series) -> dict:
    for key in ["lgbm_oof", "xgb_oof", "ensemble_oof"]:
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


def save_ensemble_artifacts(results: dict) -> None:
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    results["lgbm_oof"].to_csv(OOF_DIR / "oof_lgbm.csv", index=True, header=True)
    results["xgb_oof"].to_csv(OOF_DIR / "oof_xgb.csv", index=True, header=True)
    results["ensemble_oof"].to_csv(OOF_DIR / "oof_ensemble.csv", index=True, header=True)

    for fold_idx, model in enumerate(results["lgbm_models"]):
        joblib.dump(model, MODEL_DIR / f"lgbm_fold{fold_idx}.pkl")
    for fold_idx, model in enumerate(results["xgb_models"]):
        joblib.dump(model, MODEL_DIR / f"xgb_fold{fold_idx}.pkl")

    with open(MODEL_DIR / "ensemble_info.txt", "w") as f:
        f.write(f"lgbm_fold_scores={results['lgbm_fold_scores']}\n")
        f.write(f"xgb_fold_scores={results['xgb_fold_scores']}\n")
        f.write(f"lgbm_best_threshold={results['lgbm_oof_best_threshold']:.4f}\n")
        f.write(f"lgbm_best_f1={results['lgbm_oof_best_f1']:.4f}\n")
        f.write(f"xgb_best_threshold={results['xgb_oof_best_threshold']:.4f}\n")
        f.write(f"xgb_best_f1={results['xgb_oof_best_f1']:.4f}\n")
        f.write(f"ensemble_best_threshold={results['ensemble_oof_best_threshold']:.4f}\n")
        f.write(f"ensemble_best_f1={results['ensemble_oof_best_f1']:.4f}\n")
