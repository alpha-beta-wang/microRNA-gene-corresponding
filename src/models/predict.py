"""预测与提交文件生成模块。"""

from typing import Any

import numpy as np
import pandas as pd

from src.config import SUBMISSION_DIR, SUBMIT_EXAMPLE_FILE, GENE_COLUMN, MIRNA_COLUMN, TARGET_COLUMN


def _predict_with_models(
    features: pd.DataFrame, models: list[Any],
    feature_cols: list[list[str]] | None = None,
) -> np.ndarray:
    """对多折模型预测概率并取平均。"""
    if feature_cols is None:
        probas = np.array([m.predict_proba(features)[:, 1] for m in models])
    else:
        probas = np.array([
            m.predict_proba(features[feature_cols[i]])[:, 1]  # 每个模型使用各自训练时的特征列
            for i, m in enumerate(models)
        ])
    return probas.mean(axis=0)  # 折间概率做均值集成


def predict_and_submit(
    test_features: pd.DataFrame,
    test_meta: pd.DataFrame,
    models: list[Any],
    threshold: float,
    output_name: str = "submission.csv",
    model_feature_cols: list[list[str]] | None = None,
) -> pd.DataFrame:
    """生成预测结果并按提交模板写出 CSV。"""
    avg_proba = _predict_with_models(test_features, models, model_feature_cols)
    predictions = (avg_proba >= threshold).astype(int)  # 按给定阈值二值化

    template = pd.read_csv(SUBMIT_EXAMPLE_FILE)
    submission = test_meta[[GENE_COLUMN, MIRNA_COLUMN]].copy()
    submission[TARGET_COLUMN] = predictions
    submission = submission[list(template.columns)]  # 列顺序与官方模板保持一致

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = SUBMISSION_DIR / output_name
    submission.to_csv(submission_path, index=False)
    print(f"submission saved to {submission_path} rows={len(submission)}")
    return submission
