import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from src.config import SEED, N_SPLITS, GENE_COLUMN, MIRNA_COLUMN


def _run_group_cv(
    features: pd.DataFrame,
    labels: pd.Series,
    groups: pd.Series,
    label: str,
) -> None:
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = pd.Series(np.full(len(labels), 0.5), index=labels.index)
    fold_scores = []

    for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(features, labels, groups)):
        X_tr, X_val = features.iloc[tr_idx], features.iloc[val_idx]
        y_tr, y_val = labels.iloc[tr_idx], labels.iloc[val_idx]

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
            random_state=SEED + fold_idx,
            early_stopping_round=100,
            verbose=-1,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="logloss")
        val_proba = model.predict_proba(X_val)[:, 1]
        oof.iloc[val_idx] = val_proba
        fold_scores.append(f1_score(y_val, val_proba > 0.5))

    best_t = 0.5
    best_f1 = 0.0
    for t in np.linspace(0.1, 0.9, 101):
        cur = f1_score(labels, (oof >= t).astype(int))
        if cur > best_f1:
            best_f1 = cur
            best_t = t

    print(f"  group_by={label} fold_scores={[round(s, 4) for s in fold_scores]}")
    print(f"  mean_f1@{0.5}={np.mean(fold_scores):.4f} best_threshold={best_t:.3f} best_f1={best_f1:.4f}")


def run_group_validation(
    features: pd.DataFrame,
    labels: pd.Series,
    train: pd.DataFrame,
) -> None:
    print("=== group validation ===")
    _run_group_cv(features, labels, train[GENE_COLUMN], "gene")
    _run_group_cv(features, labels, train[MIRNA_COLUMN], "miRNA")
    _run_group_cv(
        features,
        labels,
        train[GENE_COLUMN].astype(str) + "_" + train[MIRNA_COLUMN].astype(str),
        "pair",
    )


if __name__ == "__main__":
    from src.config import TARGET_COLUMN
    from src.data.load_data import build_dataset_bundle
    from src.features.basic_features import compute_sequence_features
    from src.features.sequence_match_features import compute_match_features

    bundle = build_dataset_bundle()
    basic = compute_sequence_features(bundle.train)
    match = compute_match_features(bundle.train)
    feats = pd.concat([basic, match], axis=1)
    feats = feats.loc[:, ~feats.columns.duplicated()]
    run_group_validation(feats, bundle.train[TARGET_COLUMN], bundle.train)
