import argparse

import pandas as pd

from src.config import TARGET_COLUMN, GENE_COLUMN, MIRNA_COLUMN, SUBMISSION_DIR, SUBMIT_EXAMPLE_FILE
from src.data.load_data import build_dataset_bundle
from src.features.build_features import build_features
from src.models.hard_negative import train_two_stage, predict_two_stage

from src.models.train_ensemble import (
    train_ensemble, build_stacking_ensemble, predict_ensemble_proba,
    optimize_thresholds, save_ensemble_artifacts,
)
from src.models.predict import predict_and_submit


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
        help="OOF probability threshold for hard-negative selection",
    )
    p.add_argument(
        "--hard-negative-top-fraction", type=float, default=None,
        help="top fraction of negatives to select as hard (overrides threshold)",
    )
    p.add_argument(
        "--imbalance-strategy", default="none", choices=["none"],
        help="class imbalance strategy (default: none)",
    )
    p.add_argument(
        "--missing-external-policy", default="error", choices=["error", "skip"],
        help="behavior when an external tool is missing (default: error)",
    )
    p.add_argument(
        "--ensemble-mode", default="mean", choices=["mean", "stacking"],
        help="ensemble combination mode (default: mean)",
    )
    p.add_argument(
        "--feature-selection", default="none", choices=["none", "top_n"],
        help="feature selection within CV folds (default: none)",
    )
    p.add_argument(
        "--feature-top-n", type=int, default=50,
        help="top N features to keep when --feature-selection top_n (default: 50)",
    )
    p.add_argument(
        "--feature-selection-threshold", type=str, default="median",
        help="threshold for select_from_model (default: median)",
    )
    p.add_argument(
        "--hyperopt", default="none", choices=["none", "optuna"],
        help="hyperparameter optimization mode (default: none)",
    )
    p.add_argument(
        "--hyperopt-trials", type=int, default=50,
        help="number of Optuna trials per model (default: 50)",
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
        mode = results.get("ensemble_mode", "mean")
        print(f"ensemble ({mode}) best_f1={results['ensemble_oof_best_f1']:.4f} "
              f"threshold={results['ensemble_oof_best_threshold']:.3f}")
    if "stacker_coef" in results:
        print(f"stacker weights: {results['stacker_coef']}")


def _submit_standard(
    test_features: pd.DataFrame,
    test_meta: pd.DataFrame,
    results: dict,
    model_names: list[str],
    tag_prefix: str,
) -> None:
    for m in model_names:
        models_key = f"{m}_models"
        if models_key in results:
            cols_key = f"{m}_feature_cols"
            cols = results.get(cols_key)
            t = results.get(f"{m}_oof_best_threshold", 0.5)
            predict_and_submit(test_features, test_meta, results[models_key], t,
                               f"{tag_prefix}submission_{m}.csv",
                               model_feature_cols=cols)

    # Ensemble submission (if at least one model and ensemble OOF exists)
    if "ensemble_oof" in results and any(f"{m}_models" in results for m in model_names):
        et = results.get("ensemble_oof_best_threshold", 0.5)
        _submit_ensemble(test_features, test_meta, results, et,
                         f"{tag_prefix}submission_ensemble.csv")


def _submit_ensemble(
    test_features: pd.DataFrame,
    test_meta: pd.DataFrame,
    results: dict,
    threshold: float,
    output_name: str,
) -> None:
    from src.models.predict import _predict_with_models

    model_probas = {}
    for key in list(results.keys()):
        if key.endswith("_models") and isinstance(results[key], list):
            m = key[:-7]  # strip "_models"
            cols_key = f"{m}_feature_cols"
            cols = results.get(cols_key)
            model_probas[m] = _predict_with_models(test_features, results[key], cols)

    avg_proba = predict_ensemble_proba(model_probas, results)
    predictions = (avg_proba >= threshold).astype(int)

    template = pd.read_csv(SUBMIT_EXAMPLE_FILE)
    submission = test_meta[[GENE_COLUMN, MIRNA_COLUMN]].copy()
    submission[TARGET_COLUMN] = predictions
    submission = submission[list(template.columns)]

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = SUBMISSION_DIR / output_name
    submission.to_csv(submission_path, index=False)
    print(f"submission saved to {submission_path} rows={len(submission)}")


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

    if args.hyperopt == "optuna":
        from src.models.hyperopt import run_optuna
        print(f"=== hyperparameter optimization (optuna, {args.hyperopt_trials} trials) ===")
        model_params = run_optuna(
            train_features, labels, model_names, args.hyperopt_trials,
        )
    else:
        model_params = {}

    train_kwargs = dict(
        models=model_names,
        feature_selection=args.feature_selection,
        feature_top_n=args.feature_top_n,
        feature_selection_threshold=args.feature_selection_threshold,
        model_params=model_params,
    )

    if args.second_stage == "hard_negative":
        print(f"=== two-stage training (models: {model_names}) ===")
        ts_results = train_two_stage(train_features, labels, **train_kwargs,
                                     hn_threshold=args.hard_negative_threshold,
                                     hn_top_fraction=args.hard_negative_top_fraction)
        s1 = ts_results["stage1"]
        s2 = ts_results["stage2"]

        if args.ensemble_mode == "stacking":
            s1 = build_stacking_ensemble(s1, labels)
            s2_labels = labels[ts_results["stage2_mask"]]
            s2 = build_stacking_ensemble(s2, s2_labels)

        print("--- stage 1 results ---")
        _print_results(s1, model_names)
        print("--- stage 2 results ---")
        _print_results(s2, model_names)

        save_ensemble_artifacts(s1, run_tag=f"{args.run_tag}_stage1" if args.run_tag else "stage1")
        save_ensemble_artifacts(s2, run_tag=f"{args.run_tag}_stage2" if args.run_tag else "stage2")

        print("=== generating submissions ===")
        _submit_standard(test_features, test_meta, s1, model_names, f"{tag_prefix}stage1_")

        ts_proba = predict_two_stage(test_features, ts_results)
        et = s2.get("ensemble_oof_best_threshold", 0.5)
        ts_preds = (ts_proba >= et).astype(int)
        template = pd.read_csv(SUBMIT_EXAMPLE_FILE)
        sub = test_meta[[GENE_COLUMN, MIRNA_COLUMN]].copy()
        sub[TARGET_COLUMN] = ts_preds
        sub = sub[list(template.columns)]
        SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
        out = SUBMISSION_DIR / f"{tag_prefix}submission_hard_negative.csv"
        sub.to_csv(out, index=False)
        print(f"submission saved to {out} rows={len(sub)}")
    else:
        print(f"=== training (models: {model_names}, "
              f"feature_selection={args.feature_selection}, "
              f"ensemble={args.ensemble_mode}) ===")
        results = train_ensemble(train_features, labels, **train_kwargs)

        if args.ensemble_mode == "stacking":
            results = build_stacking_ensemble(results, labels)

        if args.threshold_search == "on":
            results = optimize_thresholds(results, labels)

        _print_results(results, model_names)
        save_ensemble_artifacts(results, run_tag=args.run_tag)

        print("=== generating submissions ===")
        _submit_standard(test_features, test_meta, results, model_names, tag_prefix)


if __name__ == "__main__":
    main()
