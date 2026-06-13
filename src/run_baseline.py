import pandas as pd

from src.config import TARGET_COLUMN, GENE_COLUMN, MIRNA_COLUMN
from src.data.load_data import build_dataset_bundle
from src.features.basic_features import compute_sequence_features
from src.features.sequence_match_features import compute_match_features
from src.models.train_lgbm import train_with_cv, optimize_threshold, save_artifacts
from src.models.predict import predict_and_submit


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

    print("=== training ===")
    oof, mean_f1, fold_scores, models = train_with_cv(train_features, labels)
    print(f"fold_scores={[round(s, 4) for s in fold_scores]}")
    print(f"mean_cv_f1@{0.5}={mean_f1:.4f}")

    best_threshold, best_f1 = optimize_threshold(oof, labels)
    print(f"best_threshold={best_threshold:.3f} best_f1={best_f1:.4f}")

    save_artifacts(oof, models, fold_scores, best_threshold)

    print("=== generating submission ===")
    test_meta = bundle.test[[GENE_COLUMN, MIRNA_COLUMN]]
    predict_and_submit(test_features, test_meta, models, best_threshold)


if __name__ == "__main__":
    main()
