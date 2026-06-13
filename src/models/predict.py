import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src.config import SUBMISSION_DIR, SUBMIT_EXAMPLE_FILE, GENE_COLUMN, MIRNA_COLUMN, TARGET_COLUMN


def predict_and_submit(
    test_features: pd.DataFrame,
    test_meta: pd.DataFrame,
    models: list[LGBMClassifier],
    threshold: float,
    output_name: str = "submission_lgbm.csv",
) -> pd.DataFrame:
    probas = np.array([m.predict_proba(test_features)[:, 1] for m in models])
    avg_proba = probas.mean(axis=0)
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
