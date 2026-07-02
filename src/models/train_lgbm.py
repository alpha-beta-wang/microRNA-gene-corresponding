"""LightGBM 训练模块，包含交叉验证、阈值优化和产物保存。"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from src.config import MODEL_DIR, OOF_DIR, SEED, N_SPLITS


def train_with_cv(
    features: pd.DataFrame,
    labels: pd.Series,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
) -> tuple[pd.Series, float, list[float], list[LGBMClassifier]]:
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = pd.Series(np.full(len(labels), 0.5), index=labels.index, name="oof")
    fold_scores: list[float] = []
    models: list[LGBMClassifier] = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds.split(features, labels)):
        X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
        y_train, y_val = labels.iloc[train_idx], labels.iloc[val_idx]

        model = LGBMClassifier(
            n_estimators=3000,
            learning_rate=0.01,
            max_depth=6,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=seed + fold_idx,
            early_stopping_round=100,
            verbose=-1,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="logloss",
        )

        val_proba = model.predict_proba(X_val)[:, 1]
        oof.iloc[val_idx] = val_proba
        fold_score = f1_score(y_val, val_proba > 0.5)
        fold_scores.append(fold_score)
        models.append(model)

    return oof, float(np.mean(fold_scores)), fold_scores, models


def optimize_threshold(oof: pd.Series, labels: pd.Series, n_steps: int = 101) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = 0.0
    for t in np.linspace(0.1, 0.9, n_steps):
        current_f1 = f1_score(labels, (oof >= t).astype(int))
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = t
    return best_threshold, best_f1


def save_artifacts(
    oof: pd.Series,
    models: list[LGBMClassifier],
    fold_scores: list[float],
    best_threshold: float,
) -> None:
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    oof.to_csv(OOF_DIR / "oof_lgbm.csv", index=True, header=True)

    import joblib

    for fold_idx, model in enumerate(models):
        joblib.dump(model, MODEL_DIR / f"lgbm_fold{fold_idx}.pkl")

    with open(MODEL_DIR / "cv_info.txt", "w") as f:
        f.write(f"fold_scores={fold_scores}\n")
        f.write(f"mean_cv_f1={float(np.mean(fold_scores)):.6f}\n")
        f.write(f"best_threshold={best_threshold:.4f}\n")
"""LightGBM 训练模块，包含交叉验证、阈值优化和产物保存。"""
