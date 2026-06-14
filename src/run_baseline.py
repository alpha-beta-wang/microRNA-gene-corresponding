import argparse

import pandas as pd

from src.config import TARGET_COLUMN, GENE_COLUMN, MIRNA_COLUMN
from src.data.load_data import build_dataset_bundle
from src.features.build_features import build_features
from src.models.hard_negative import train_two_stage, predict_two_stage
from src.models.train_ensemble import train_ensemble, optimize_thresholds, save_ensemble_artifacts
from src.models.predict import predict_and_submit, predict_ensemble


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="microRNA-gene MTI prediction pipeline")
    p.add_argument(
        "--feature-blocks", default="basic,match",
        help="comma-separated feature block names (default: basic,match)",
    )
    p.add_argument(
        "--models", default="lgbm,xgb",
        help="comma-separated model names: lgbm, xgb (default: lgbm,xgb)",
    )
    p.add_argument(
        "--run-tag", default=None,
        help="optional experiment name for isolated outputs",
    )
    p.add_argument(
        "--threshold-search", default="on", choices=["on", "off"],
        help="whether to search for best OOF threshold (default: on)",
    )
    p.add_argument(
        "--second-stage", default="none", choices=["none", "hard_negative"],
        help="second-stage training strategy (default: none)",
    )
    p.add_argument(
        "--hard-negative-threshold", type=float, default=0.8,
        help="OOF probability threshold for hard-negative selection (default: 0.8)",
    )
    p.add_argument(
        "--hard-negative-top-fraction", type=float, default=None,
        help="top fraction of negatives to select as hard (overrides threshold)",
    )
    p.add_argument(
        "--imbalance-strategy", default="none", choices=["none"],
        help="class imbalance strategy (default: none; focal not implemented — data is 70/30)",
    )
    p.add_argument(
        "--missing-external-policy", default="error", choices=["error", "skip"],
        help="behavior when an external tool is missing (default: error)",
    )
    p.add_argument(
        "--ensemble-mode", default="mean",
        help="ensemble combination mode (default: mean)",
    )
    return p.parse_args()


def _print_results(results: dict, model_names: list[str]) -> None:
    for m in model_names:
        fs = f"{m}_fold_scores"
        bt = f"{m}_oof_best_threshold"
        bf = f"{m}_oof_best_f1"
        if fs in results:
            print(f"{m} fold_scores={[round(s, 4) for s in results[fs]]}")
        if bt in results:
            print(f"{m} best_f1={results[bf]:.4f} threshold={results[bt]:.3f}")
    if "ensemble_oof_best_f1" in results:
        print(f"ensemble best_f1={results['ensemble_oof_best_f1']:.4f} "
              f"threshold={results['ensemble_oof_best_threshold']:.3f}")


def _submit_standard(
    test_features: pd.DataFrame,
    test_meta: pd.DataFrame,
    results: dict,
    model_names: list[str],
    tag_prefix: str,
) -> None:
    if "lgbm" in model_names and "lgbm_models" in results:
        lt = results.get("lgbm_oof_best_threshold", 0.5)
        predict_and_submit(test_features, test_meta, results["lgbm_models"], lt,
                           f"{tag_prefix}submission_lgbm.csv")

    if "xgb" in model_names and "xgb_models" in results:
        xt = results.get("xgb_oof_best_threshold", 0.5)
        predict_and_submit(test_features, test_meta, results["xgb_models"], xt,
                           f"{tag_prefix}submission_xgb.csv")

    if "ensemble_oof" in results and "lgbm_models" in results and "xgb_models" in results:
        et = results.get("ensemble_oof_best_threshold", 0.5)
        predict_ensemble(test_features, test_meta,
                         results["lgbm_models"], results["xgb_models"], et,
                         f"{tag_prefix}submission_ensemble.csv")


def main():
    args = parse_args()

    feature_blocks = [s.strip() for s in args.feature_blocks.split(",")]
    model_names = [s.strip() for s in args.models.split(",")]

    print("=== loading data ===")
    bundle = build_dataset_bundle()
    print(
        f"train={len(bundle.train)} test={len(bundle.test)} "
        f"gene_miss={bundle.train_missing_gene_sequences} "
        f"mirna_miss={bundle.train_missing_mirna_sequences}"
    )

    print(f"=== building features (blocks: {feature_blocks}) ===")
    train_features, test_features = build_features(
        bundle.train, bundle.test, feature_blocks,
        missing_external_policy=args.missing_external_policy,
    )

    labels = bundle.train[TARGET_COLUMN].copy()
    test_meta = bundle.test[[GENE_COLUMN, MIRNA_COLUMN]]
    tag_prefix = f"{args.run_tag}_" if args.run_tag else ""

    if args.second_stage == "hard_negative":
        print(f"=== two-stage training (models: {model_names}) ===")
        ts_results = train_two_stage(
            train_features, labels, models=model_names,
            hn_threshold=args.hard_negative_threshold,
            hn_top_fraction=args.hard_negative_top_fraction,
        )
        s1 = ts_results["stage1"]
        s2 = ts_results["stage2"]

        print("--- stage 1 results ---")
        _print_results(s1, model_names)
        print("--- stage 2 results ---")
        _print_results(s2, model_names)

        save_ensemble_artifacts(s1, run_tag=f"{args.run_tag}_stage1" if args.run_tag else "stage1")
        save_ensemble_artifacts(s2, run_tag=f"{args.run_tag}_stage2" if args.run_tag else "stage2")

        print("=== generating submissions ===")
        _submit_standard(test_features, test_meta, s1, model_names, f"{tag_prefix}stage1_")

        # Two-stage submission
        ts_proba = predict_two_stage(test_features, ts_results)
        et = s2.get("ensemble_oof_best_threshold", 0.5)
        ts_preds = (ts_proba >= et).astype(int)
        import pandas as pd
        from src.config import SUBMISSION_DIR, SUBMIT_EXAMPLE_FILE
        template = pd.read_csv(SUBMIT_EXAMPLE_FILE)
        sub = test_meta[[GENE_COLUMN, MIRNA_COLUMN]].copy()
        sub[TARGET_COLUMN] = ts_preds
        sub = sub[list(template.columns)]
        SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
        out = SUBMISSION_DIR / f"{tag_prefix}submission_hard_negative.csv"
        sub.to_csv(out, index=False)
        print(f"submission saved to {out} rows={len(sub)}")
    else:
        print(f"=== training (models: {model_names}) ===")
        results = train_ensemble(train_features, labels, models=model_names)

        if args.threshold_search == "on":
            results = optimize_thresholds(results, labels)

        _print_results(results, model_names)
        save_ensemble_artifacts(results, run_tag=args.run_tag)

        print("=== generating submissions ===")
        _submit_standard(test_features, test_meta, results, model_names, tag_prefix)


if __name__ == "__main__":
    main()
