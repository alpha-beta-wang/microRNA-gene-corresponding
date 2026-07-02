"""基线实验入口，用于快速训练基础模型并生成初始结果。"""

import pandas as pd

from src.config import TARGET_COLUMN, GENE_COLUMN, MIRNA_COLUMN
from src.data.load_data import build_dataset_bundle
from src.features.basic_features import compute_sequence_features
from src.features.sequence_match_features import compute_match_features
from src.features.advanced_features import compute_advanced_features
from src.models.train_ensemble import train_ensemble, optimize_thresholds, save_ensemble_artifacts
from src.models.predict import predict_and_submit, predict_ensemble


def main():
    print("=== loading data ===")
    bundle = build_dataset_bundle()
    print(
        f"train={len(bundle.train)} test={len(bundle.test)} "
        f"gene_miss={bundle.train_missing_gene_sequences} "
        f"mirna_miss={bundle.train_missing_mirna_sequences}"
    )

    print("=== building features ===")
    basic = compute_sequence_features(bundle.train)
    match = compute_match_features(bundle.train)
    advanced = compute_advanced_features(bundle.train)
    train_features = pd.concat([basic, match, advanced], axis=1)

    test_basic = compute_sequence_features(bundle.test)
    test_match = compute_match_features(bundle.test)
    test_advanced = compute_advanced_features(bundle.test)
    test_features = pd.concat([test_basic, test_match, test_advanced], axis=1)

    train_features = train_features.loc[:, ~train_features.columns.duplicated()]
    test_features = test_features.loc[:, ~test_features.columns.duplicated()]
    test_features = test_features[train_features.columns]

    print(f"feature_count={len(train_features.columns)}")

    labels = bundle.train[TARGET_COLUMN].copy()
    test_meta = bundle.test[[GENE_COLUMN, MIRNA_COLUMN]]

    pos = int(labels.sum()); neg = int((1 - labels).sum())
    spw = neg / pos
    print(f"=== training ensemble (StratifiedKFold, scale_pos_weight={spw:.4f}) ===")
    results = train_ensemble(train_features, labels, scale_pos_weight=spw)
    results = optimize_thresholds(results, labels)

    lt = results["lgbm_nested_threshold"]
    xt = results["xgb_nested_threshold"]
    et = results["ensemble_nested_threshold"]

    print(f"lgbm fold_scores={[round(s, 4) for s in results['lgbm_fold_scores']]}")
    print(f"lgbm fold_thresholds={[round(t, 3) for t in results['lgbm_fold_thresholds']]}")
    print(f"lgbm nested_t={lt:.3f} oof_best_t={results['lgbm_oof_best_threshold']:.3f} oof_best_f1={results['lgbm_oof_best_f1']:.4f}")
    print(f"xgb  fold_scores={[round(s, 4) for s in results['xgb_fold_scores']]}")
    print(f"xgb  fold_thresholds={[round(t, 3) for t in results['xgb_fold_thresholds']]}")
    print(f"xgb  nested_t={xt:.3f} oof_best_t={results['xgb_oof_best_threshold']:.3f} oof_best_f1={results['xgb_oof_best_f1']:.4f}")
    print(f"ens  fold_thresholds={[round(t, 3) for t in results['ensemble_fold_thresholds']]}")
    print(f"ens  nested_t={et:.3f} oof_best_t={results['ensemble_oof_best_threshold']:.3f} oof_best_f1={results['ensemble_oof_best_f1']:.4f}")

    save_ensemble_artifacts(results)

    print("=== generating submissions ===")
    predict_and_submit(test_features, test_meta, results["lgbm_models"], lt, "submission_lgbm.csv")
    predict_and_submit(test_features, test_meta, results["xgb_models"], xt, "submission_xgb.csv")
    predict_ensemble(
        test_features,
        test_meta,
        results["lgbm_models"],
        results["xgb_models"],
        et,
        "submission_ensemble.csv",
    )


if __name__ == "__main__":
    main()
"""基线实验入口，用于快速训练基础模型并生成初始结果。"""
