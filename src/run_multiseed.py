"""多随机种子实验入口，用于提升交叉验证和提交结果稳定性。"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.config import (
    GENE_COLUMN, MIRNA_COLUMN, N_SPLITS, SUBMISSION_DIR, SUBMIT_EXAMPLE_FILE,
    TARGET_COLUMN, MODEL_DIR, OOF_DIR,
)
from src.data.load_data import build_dataset_bundle
from src.features.advanced_features import compute_advanced_features
from src.features.basic_features import compute_sequence_features
from src.features.kmer_features import compute_kmer_features
from src.features.kmer_interaction_features import compute_kmer_interaction_features
from src.features.sequence_match_features import compute_match_features

SEEDS = [42, 123, 456, 789, 2024]


def _build_lgbm(seed: int, scale_pos_weight: float) -> LGBMClassifier:
    """Create a LightGBM base model for the given seed."""
    return LGBMClassifier(
        n_estimators=3000, learning_rate=0.01, max_depth=6, num_leaves=31,
        min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, random_state=seed,
        early_stopping_round=100, verbose=-1, scale_pos_weight=scale_pos_weight,
    )


def _build_xgb(seed: int, scale_pos_weight: float) -> XGBClassifier:
    """Create an XGBoost base model for the given seed."""
    return XGBClassifier(
        n_estimators=3000, learning_rate=0.01, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
        random_state=seed, early_stopping_rounds=100, eval_metric="logloss",
        verbosity=0, scale_pos_weight=scale_pos_weight,
    )


def main():
    """Train models with multiple seeds and generate thresholded submissions."""
    print("=== loading data ===")
    bundle = build_dataset_bundle()
    print(f"train={len(bundle.train)} test={len(bundle.test)}")

    print("=== building features ===")
    train_X = pd.concat([
        compute_sequence_features(bundle.train),
        compute_match_features(bundle.train),
        compute_advanced_features(bundle.train),
    ], axis=1)
    test_X = pd.concat([
        compute_sequence_features(bundle.test),
        compute_match_features(bundle.test),
        compute_advanced_features(bundle.test),
    ], axis=1)
    train_X = train_X.loc[:, ~train_X.columns.duplicated()]
    test_X = test_X.loc[:, ~test_X.columns.duplicated()]
    test_X = test_X[train_X.columns]

    kmer_train, kmer_test = compute_kmer_features(bundle.train, bundle.test)
    kmi_train = compute_kmer_interaction_features(bundle.train)
    kmi_test  = compute_kmer_interaction_features(bundle.test)
    train_X = pd.concat([train_X, kmer_train, kmi_train], axis=1)
    test_X  = pd.concat([test_X,  kmer_test,  kmi_test],  axis=1)
    test_X = test_X[train_X.columns]
    print(f"  base+kmer+kmi features: base=26 kmer={kmer_train.shape[1]} kmi={kmi_train.shape[1]} total={train_X.shape[1]}")
    y = bundle.train[TARGET_COLUMN].copy()
    pos = int(y.sum()); neg = int((1 - y).sum())
    spw = neg / pos
    print(f"feature_count={len(train_X.columns)} pos={pos} neg={neg} spw={spw:.4f}")

    n = len(bundle.test)
    test_proba_sum = np.zeros(n, dtype=float)
    test_proba_count = 0

    oof_proba = np.full(len(y), np.nan)
    oof_proba_count = np.zeros(len(y), dtype=int)

    for seed in SEEDS:
        print(f"=== seed={seed} ===")
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(train_X, y)):
            X_tr, X_val = train_X.iloc[tr_idx], train_X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            lgbm = _build_lgbm(seed + fold_idx, spw)
            lgbm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="logloss")
            xgb = _build_xgb(seed + fold_idx, spw)
            xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])

            val_p = (lgbm.predict_proba(X_val)[:, 1] + xgb.predict_proba(X_val)[:, 1]) / 2
            mask = np.isnan(oof_proba[val_idx])
            new = oof_proba[val_idx]
            new[mask] = 0.0
            new = (new * oof_proba_count[val_idx] + val_p) / (oof_proba_count[val_idx] + 1)
            oof_proba[val_idx] = new
            oof_proba_count[val_idx] += 1

            test_p = (lgbm.predict_proba(test_X)[:, 1] + xgb.predict_proba(test_X)[:, 1]) / 2
            test_proba_sum += test_p
            test_proba_count += 1

    test_proba = test_proba_sum / test_proba_count

    print("=== OOF F1 grid (multi-seed averaged) ===")
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.30, 0.65, 36):
        cur = f1_score(y, (oof_proba >= t).astype(int))
        if cur > best_f1:
            best_f1, best_t = cur, float(t)
    print(f"oof best_f1={best_f1:.4f} at thr={best_t:.3f}")
    for t in [0.40, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52]:
        f1 = f1_score(y, (oof_proba >= t).astype(int))
        print(f"  oof t={t:.2f} f1={f1:.4f}")

    print("=== test proba quantiles ===")
    for q in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
        print(f"  q{int(q*100)}={np.quantile(test_proba, q):.4f}")

    template = pd.read_csv(SUBMIT_EXAMPLE_FILE)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    test_meta = bundle.test[[GENE_COLUMN, MIRNA_COLUMN]]

    print("=== submissions ===")
    for t in [0.43, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52]:
        sub = test_meta.copy()
        sub[TARGET_COLUMN] = (test_proba >= t).astype(int)
        sub = sub[list(template.columns)]
        name = f"submission_kmi_t{int(round(t*100))}.csv"
        sub.to_csv(SUBMISSION_DIR / name, index=False)
        pp = int((sub[TARGET_COLUMN] == 1).sum())
        print(f"{name}: pos={pp} neg={len(sub)-pp} pos_rate={pp/len(sub):.3f}")

    OOF_DIR.mkdir(parents=True, exist_ok=True)
    pd.Series(test_proba, name="test_proba").to_csv(OOF_DIR / "test_proba_kmi.csv", index=True, header=True)
    pd.Series(oof_proba, name="oof_proba").to_csv(OOF_DIR / "oof_kmi.csv", index=True, header=True)


if __name__ == "__main__":
    main()
