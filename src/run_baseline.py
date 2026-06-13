import pandas as pd

from src.config import TARGET_COLUMN, GENE_COLUMN, MIRNA_COLUMN
from src.data.load_data import build_dataset_bundle
from src.features.basic_features import compute_sequence_features
from src.features.sequence_match_features import compute_match_features
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
    train_features = pd.concat([basic, match], axis=1)

    test_basic = compute_sequence_features(bundle.test)
    test_match = compute_match_features(bundle.test)
    test_features = pd.concat([test_basic, test_match], axis=1)

    train_features = train_features.loc[:, ~train_features.columns.duplicated()]
    test_features = test_features.loc[:, ~test_features.columns.duplicated()]
    test_features = test_features[train_features.columns]

    print(f"feature_count={len(train_features.columns)}")

    labels = bundle.train[TARGET_COLUMN].copy()
    test_meta = bundle.test[[GENE_COLUMN, MIRNA_COLUMN]]

    print("=== training ensemble ===")
    results = train_ensemble(train_features, labels)
    results = optimize_thresholds(results, labels)

    lt = results["lgbm_oof_best_threshold"]
    xt = results["xgb_oof_best_threshold"]
    et = results["ensemble_oof_best_threshold"]

    print(f"lgbm fold_scores={[round(s, 4) for s in results['lgbm_fold_scores']]}")
    print(f"lgbm best_f1={results['lgbm_oof_best_f1']:.4f} threshold={lt:.3f}")
    print(f"xgb  fold_scores={[round(s, 4) for s in results['xgb_fold_scores']]}")
    print(f"xgb  best_f1={results['xgb_oof_best_f1']:.4f} threshold={xt:.3f}")
    print(f"ensemble best_f1={results['ensemble_oof_best_f1']:.4f} threshold={et:.3f}")

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
