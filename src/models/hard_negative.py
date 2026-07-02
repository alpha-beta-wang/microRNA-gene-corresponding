"""困难负样本挖掘模块，用于筛选高置信负样本再训练。"""

import numpy as np
import pandas as pd

from src.config import SEED, N_SPLITS
from src.models.train_ensemble import train_ensemble, optimize_thresholds


def _collect_models_and_cols(results: dict) -> tuple[list, list[list[str]] | None]:
    """Collect all fold models and their feature columns from generic results dict."""
    models = []
    feature_cols_list = []
    for key, value in results.items():
        if key.endswith("_models") and isinstance(value, list):
            models.extend(value)
            base = key[:-7]  # strip "_models"
            fk = f"{base}_feature_cols"
            if fk in results:
                feature_cols_list.extend(results[fk])
    if not models:
        raise RuntimeError("no models found in results")
    return models, feature_cols_list if feature_cols_list else None


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
    feature_selection: str = "none",
    feature_top_n: int = 50,
    feature_selection_threshold: str = "median",
    model_params: dict[str, dict] | None = None,
) -> dict:
    if models is None:
        models = ["lgbm", "xgb"]

    # Stage 1: normal CV
    print("  [stage1] training on full dataset")
    s1_results = train_ensemble(
        features, labels, models=models, n_splits=n_splits, seed=seed,
        feature_selection=feature_selection, feature_top_n=feature_top_n,
        feature_selection_threshold=feature_selection_threshold,
        model_params=model_params,
    )
    s1_results = optimize_thresholds(s1_results, labels)

    # Select hard negatives
    s1_oof = s1_results.get("ensemble_oof")
    if s1_oof is None:
        oofs = []
        for m in models:
            oof_key = f"{m}_oof"
            if oof_key in s1_results:
                oofs.append(s1_results[oof_key])
        if oofs:
            s1_oof = sum(oofs) / len(oofs)
        else:
            raise RuntimeError("no OOF predictions found in stage-1 results")

    stage2_mask = select_hard_negatives(s1_oof, labels, hn_threshold, hn_top_fraction)

    # Stage 2: train on positives + hard negatives
    print("  [stage2] training on positives + hard negatives")
    s2_features = features.loc[stage2_mask].copy()
    s2_labels = labels.loc[stage2_mask]

    meta_col = "meta__stage1_oof"
    s2_features[meta_col] = s1_oof.loc[stage2_mask].values

    s2_results = train_ensemble(
        s2_features, s2_labels, models=models, n_splits=n_splits, seed=seed + 100,
        feature_selection=feature_selection, feature_top_n=feature_top_n,
        feature_selection_threshold=feature_selection_threshold,
        model_params=model_params,
    )
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

    # Stage 1
    s1_models, s1_cols = _collect_models_and_cols(s1)
    s1_proba = _predict_with_models(test_features, s1_models, s1_cols)

    # Stage 2: add meta-feature
    s2_features = test_features.copy()
    s2_features[meta_col] = s1_proba

    s2_models, s2_cols = _collect_models_and_cols(s2)
    if s2_cols is not None:
        s2_cols = [cols + [meta_col] for cols in s2_cols]

    return _predict_with_models(s2_features, s2_models, s2_cols)
"""困难负样本挖掘模块，用于筛选高置信负样本再训练。"""
