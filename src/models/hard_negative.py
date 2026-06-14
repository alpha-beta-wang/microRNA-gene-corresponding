import numpy as np
import pandas as pd

from src.config import SEED, N_SPLITS
from src.models.train_ensemble import train_ensemble, optimize_thresholds


def select_hard_negatives(
    oof_proba: pd.Series,
    labels: pd.Series,
    threshold: float = 0.8,
    top_fraction: float | None = None,
) -> pd.Series:
    """Select hard negatives from true negatives using OOF probabilities.

    Returns a boolean mask selecting: all positives + selected hard negatives.
    """
    pos_mask = labels == 1
    neg_mask = labels == 0

    if top_fraction is not None:
        neg_proba = oof_proba[neg_mask]
        n_select = max(1, int(neg_mask.sum() * top_fraction))
        thresh = neg_proba.nlargest(n_select).iloc[-1]
        hard_neg = neg_mask & (oof_proba >= thresh)
    else:
        hard_neg = neg_mask & (oof_proba >= threshold)

    n_hard = hard_neg.sum()
    print(f"  hard_negatives selected: {n_hard} / {neg_mask.sum()} "
          f"(positives: {pos_mask.sum()}, stage2_total: {pos_mask.sum() + n_hard})")
    return pos_mask | hard_neg


def train_two_stage(
    features: pd.DataFrame,
    labels: pd.Series,
    models: list[str] | None = None,
    hn_threshold: float = 0.8,
    hn_top_fraction: float | None = None,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
) -> dict:
    if models is None:
        models = ["lgbm", "xgb"]

    # Stage 1: normal CV
    print("  [stage1] training on full dataset")
    s1_results = train_ensemble(features, labels, models=models, n_splits=n_splits, seed=seed)
    s1_results = optimize_thresholds(s1_results, labels)

    # Select hard negatives
    s1_oof = s1_results.get("ensemble_oof")
    if s1_oof is None:
        use_lgbm = "lgbm" in models
        use_xgb = "xgb" in models
        if use_lgbm and use_xgb:
            s1_oof = (s1_results["lgbm_oof"] + s1_results["xgb_oof"]) / 2
        elif use_lgbm:
            s1_oof = s1_results["lgbm_oof"]
        else:
            s1_oof = s1_results["xgb_oof"]

    stage2_mask = select_hard_negatives(s1_oof, labels, hn_threshold, hn_top_fraction)

    # Stage 2: train on positives + hard negatives
    print("  [stage2] training on positives + hard negatives")
    s2_features = features.loc[stage2_mask].copy()
    s2_labels = labels.loc[stage2_mask]

    meta_col = "meta__stage1_oof"
    s2_features[meta_col] = s1_oof.loc[stage2_mask].values

    s2_results = train_ensemble(s2_features, s2_labels, models=models, n_splits=n_splits, seed=seed + 100)
    s2_results = optimize_thresholds(s2_results, s2_labels)

    return {
        "stage1": s1_results,
        "stage2": s2_results,
        "stage2_mask": stage2_mask,
        "meta_column": meta_col,
    }


def predict_two_stage(
    test_features: pd.DataFrame,
    two_stage_results: dict,
) -> np.ndarray:
    """Predict test set using two-stage models.

    Stage-1 predictions become a meta-feature for stage-2.
    """
    from src.models.predict import _predict_with_models

    s1 = two_stage_results["stage1"]
    s2 = two_stage_results["stage2"]
    meta_col = two_stage_results["meta_column"]

    # Stage 1: get model lists
    s1_models = []
    s1_feature_cols = []
    for m in ["lgbm", "xgb"]:
        key = f"{m}_models"
        cols_key = f"{m}_feature_cols"
        if key in s1:
            s1_models.extend(s1[key])
            if cols_key in s1:
                s1_feature_cols.extend(s1[cols_key])

    if not s1_models:
        raise RuntimeError("no stage-1 models found")

    s1_cols = s1_feature_cols if s1_feature_cols else None
    s1_proba = _predict_with_models(test_features, s1_models, s1_cols)

    # Stage 2: add meta-feature
    s2_features = test_features.copy()
    s2_features[meta_col] = s1_proba

    # Align columns to what stage-2 was trained on
    s2_models = []
    s2_feature_cols = []
    for m in ["lgbm", "xgb"]:
        key = f"{m}_models"
        cols_key = f"{m}_feature_cols"
        if key in s2:
            s2_models.extend(s2[key])
            if cols_key in s2:
                s2_feature_cols.extend(s2[cols_key])

    if not s2_models:
        raise RuntimeError("no stage-2 models found")

    # Align feature columns to what s2 models expect (add meta column)
    if s2_feature_cols:
        s2_feature_cols = [cols + [meta_col] for cols in s2_feature_cols]

    return _predict_with_models(s2_features, s2_models, s2_feature_cols if s2_feature_cols else None)
