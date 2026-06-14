from typing import Union

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src.config import SUBMISSION_DIR, SUBMIT_EXAMPLE_FILE, GENE_COLUMN, MIRNA_COLUMN, TARGET_COLUMN


def _predict_with_models(
    features: pd.DataFrame, models: list[Union[LGBMClassifier, XGBClassifier]],
    feature_cols: list[list[str]] | None = None,
) -> np.ndarray:
    if feature_cols is None:
        probas = np.array([m.predict_proba(features)[:, 1] for m in models])
    else:
        probas = np.array([
            m.predict_proba(features[feature_cols[i]])[:, 1]
            for i, m in enumerate(models)
        ])
    return probas.mean(axis=0)


def predict_and_submit(
    test_features: pd.DataFrame,
    test_meta: pd.DataFrame,
    models: list[Union[LGBMClassifier, XGBClassifier]],
    threshold: float,
    output_name: str = "submission.csv",
    model_feature_cols: list[list[str]] | None = None,
) -> pd.DataFrame:
    avg_proba = _predict_with_models(test_features, models, model_feature_cols)
    predictions = (avg_proba >= threshold).astype(int)

    template = pd.read_csv(SUBMIT_EXAMPLE_FILE)
    submission = test_meta[[GENE_COLUMN, MIRNA_COLUMN]].copy()
    submission[TARGET_COLUMN] = predictions
    submission = submission[list(template.columns)]

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = SUBMISSION_DIR / output_name
    submission.to_csv(submission_path, index=False)
    print(f"submission saved to {submission_path} rows={len(submission)}")
    return submission


def predict_ensemble(
    test_features: pd.DataFrame,
    test_meta: pd.DataFrame,
    lgbm_models: list[LGBMClassifier],
    xgb_models: list[XGBClassifier],
    threshold: float,
    output_name: str = "submission_ensemble.csv",
) -> pd.DataFrame:
    lgbm_proba = _predict_with_models(test_features, lgbm_models)
    xgb_proba = _predict_with_models(test_features, xgb_models)
    avg_proba = (lgbm_proba + xgb_proba) / 2
    predictions = (avg_proba >= threshold).astype(int)

    template = pd.read_csv(SUBMIT_EXAMPLE_FILE)
    submission = test_meta[[GENE_COLUMN, MIRNA_COLUMN]].copy()
    submission[TARGET_COLUMN] = predictions
    submission = submission[list(template.columns)]

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = SUBMISSION_DIR / output_name
    submission.to_csv(submission_path, index=False)
    print(f"submission saved to {submission_path} rows={len(submission)}")
    return submission
