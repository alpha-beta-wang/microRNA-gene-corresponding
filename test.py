"""
生成提交文件，默认输出为根目录下的 `submit.csv`。

使用方式：
1. `dataset_root` 是输入数据文件夹路径，不是单个 csv 文件路径。
2. `dataset_root` 的目录结构须与赛题提供的数据文件夹的结构一致，且不支持多层嵌套：

   <dataset_root>/
   ├── test_dataset.csv
   ├── submit_example.csv
   └── train_dataset/
       ├── Train.csv
       ├── gene_seq.csv
       └── mirna_seq.csv

3. 不提供值时，默认使用仓库自带的 `data/` 目录
4. 本脚本方法为：
   - 特征：basic + match + advanced + kmer
   - 模型：lgbm + xgb
   - 融合：5 个随机种子平均
   - 阈值：0.46
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import GENE_COLUMN, MIRNA_COLUMN, TARGET_COLUMN
from src.data.load_data import build_dataset_bundle_from_root
from src.experiment_runner import (
    DEFAULT_CONFIG,
    _scale_pos_weight,
    _train_with_method,
    build_features,
    load_config,
)

BEST_THRESHOLD = 0.46
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parent / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the current best online method and write submit.csv")
    parser.add_argument(
        "--dataset_root",
        default=str(DEFAULT_DATASET_ROOT),
        help=(
            "Input data directory. It must follow the same layout as the repo's data/ folder: "
            "test_dataset.csv, submit_example.csv, and train_dataset/{Train.csv,gene_seq.csv,mirna_seq.csv}."
        ),
    )
    parser.add_argument(
        "--output",
        default="submit.csv",
        help="Output submission CSV path (default: submit.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_path = Path(args.output)

    # dataset_root is an INPUT folder path, not a single CSV file.
    # Its internal structure must match the repo's data/ directory:
    #   <dataset_root>/test_dataset.csv
    #   <dataset_root>/submit_example.csv
    #   <dataset_root>/train_dataset/Train.csv
    #   <dataset_root>/train_dataset/gene_seq.csv
    #   <dataset_root>/train_dataset/mirna_seq.csv
    bundle = build_dataset_bundle_from_root(dataset_root)

    config = load_config(DEFAULT_CONFIG)
    train_X, test_X, y, train_meta, test_meta, feature_counts = build_features(config, bundle=bundle)
    spw = _scale_pos_weight(y, config)
    enabled_models = [name for name, enabled in config.get("models", {}).items() if enabled]
    model_params = config.get("model_params", {})
    seeds = [int(seed) for seed in config.get("training", {}).get("seeds", [42])]

    print(f"dataset_root={dataset_root}")
    print(f"features={feature_counts}")
    print(f"models={enabled_models} threshold={BEST_THRESHOLD:.2f}")

    train_result = _train_with_method(
        train_X,
        test_X,
        y,
        train_meta,
        config,
        enabled_models,
        spw,
        model_params,
        seeds,
    )
    test_proba = train_result["test_proba"]

    template = pd.read_csv(dataset_root / "submit_example.csv")
    submission = test_meta[[GENE_COLUMN, MIRNA_COLUMN]].copy()
    submission[TARGET_COLUMN] = (test_proba >= BEST_THRESHOLD).astype(int)
    submission = submission[list(template.columns)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"saved submission to {output_path} rows={len(submission)}")


if __name__ == "__main__":
    main()
