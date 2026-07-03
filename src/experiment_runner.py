"""可配置实验运行器，负责特征、模型、融合、输出和飞书记录。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.config import (
    GENE_COLUMN,
    MIRNA_COLUMN,
    OOF_DIR,
    PROJECT_ROOT,
    SUBMISSION_DIR,
    SUBMIT_EXAMPLE_FILE,
    TARGET_COLUMN,
)
from src.data.load_data import DatasetBundle, build_dataset_bundle
from src.features.advanced_features import compute_advanced_features
from src.features.basic_features import compute_sequence_features
from src.features.dinucleotide_features import compute_dinucleotide_features
from src.features.embedding_features import EmbeddingFeaturizer
from src.features.entity_features import compute_entity_count_features
from src.features.kmer_features import compute_kmer_features
from src.features.kmer_interaction_features import compute_kmer_interaction_features
from src.features.position_features import compute_position_features
from src.features.seed_type_features import compute_seed_type_features
from src.features.seed_variant_features import compute_seed_variant_features
from src.features.sequence_match_features import compute_match_features
from src.features.targetscan_features import compute_targetscan_features


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default_experiment.json"
DEFAULT_LARK_TOKEN = "JTGqsBzSlhpTcPtlJ6scOlbenwQ"
DEFAULT_LARK_SHEET_ID = "c8da76"

FEATURE_CN = {
    "basic": "??????",
    "match": "??????",
    "advanced": "??????",
    "kmer": "?? k-mer TF-IDF ??",
    "rna_energy": "RNA ??????",
    "entity_counts": "gene/miRNA ??????",
    "seed_variants": "seed ????????",
    "dinucleotide": "??????????",
    "targetscan": "TargetScan ??????",
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Read an experiment JSON configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Write an experiment configuration and create the parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")


def list_score_factors() -> list[str]:
    """Return the score-factor descriptions shown in reports."""
    return [
        "????????/???????????????????? gene/miRNA ????",
        "?????gene/miRNA ????????GC ??????",
        "?????seed ??????? seed ???miRNA ??????????????",
        "??????????? seed ??????????? AT ??",
        "k-mer ???miRNA k ??gene k ??TF-IDF min_df?gene ?????",
        "RNA ????????????seed ?????????????????????",
        "???????gene/miRNA ?????seed ?????????????",
        "?????LightGBM?XGBoost?ExtraTrees??????????",
        "???????????????????????????????????",
        "?????? seed/? seed?StratifiedKFold/StratifiedGroupKFold?????",
        "?????????????????????????????",
        "??????? OOF ??????????????????????",
    ]


def _feature_flag(config: dict[str, Any], name: str) -> bool:
    """Return whether a named feature block is enabled."""
    return bool(config.get("features", {}).get(name, False))


def enabled_feature_names(config: dict[str, Any]) -> list[str]:
    """List feature blocks enabled by the current configuration."""
    return [name for name in FEATURE_CN if _feature_flag(config, name)]


FEATURE_CN.update(
    {
        "basic": "??????",
        "match": "??????",
        "advanced": "??????",
        "kmer": "?? k-mer TF-IDF ??",
        "rna_energy": "RNA ??????",
        "entity_counts": "gene/miRNA ??????",
        "seed_variants": "seed ????????",
        "dinucleotide": "??????????",
        "targetscan": "TargetScan ??????",
        "kmer_interaction": "miRNA-gene ???? k-mer ????",
        "alignment": "??/????????",
        "position": "3' ???????",
        "seed_type": "?? seed ??? GU wobble ??",
        "embedding": "????? k-mer ?? SVD ????",
    }
)


def list_score_factors() -> list[str]:
    """Return the score-factor descriptions shown in reports."""
    return [
        "????????/???????????????????? gene/miRNA ????",
        "?????gene/miRNA ????????GC ??????",
        "?????seed ??????? seed ???miRNA ??????????????",
        "??????????? seed ??????????? AT ??",
        "k-mer ???miRNA k ??gene k ??TF-IDF min_df?gene ?????",
        "?????????? seed ???GU wobble?3' ????????/????????? k-mer ??",
        "RNA ????????????seed ??????????????????????",
        "??????????k-mer ?? PPMI + TruncatedSVD embedding ?????????????",
        "?????LightGBM?XGBoost?ExtraTrees?RandomForest?SVM?KNN?Factorization Machine ???????",
        "????????????????????????????????????SVM C/gamma?FM ???",
        "?????? seed/? seed?StratifiedKFold/StratifiedGroupKFold?????",
        "?????????????????????????????",
        "??????? OOF ??????????????????????",
    ]


FEATURE_CN.update(
    {
        "basic": "基础序列特征",
        "match": "序列匹配特征",
        "advanced": "高级命中特征",
        "kmer": "字符 k-mer TF-IDF 特征",
        "rna_energy": "RNA 能量特征",
        "entity_counts": "gene/miRNA 出现频次特征",
        "seed_variants": "seed 位置变体匹配特征",
        "dinucleotide": "二核苷酸组成差异特征",
        "targetscan": "TargetScan 风格位点特征",
        "kmer_interaction": "miRNA-gene 反向互补 k-mer 交互特征",
        "alignment": "局部/全局序列比对特征",
        "position": "3' 端位置权重特征",
        "seed_type": "经典 seed 类型与 GU wobble 特征",
        "embedding": "非深度学习 k-mer 共现 SVD 表征特征",
    }
)


def list_score_factors() -> list[str]:
    """Return the score-factor descriptions shown in reports."""
    return [
        "数据与切分：训练/测试分布、交叉验证折数、随机种子、是否按 gene/miRNA 分组切分",
        "基础特征：gene/miRNA 长度、碱基比例、GC 含量、长度比",
        "匹配特征：seed 命中、反向互补 seed 命中、miRNA 命中、连续匹配长度、命中次数",
        "高级命中特征：反向互补 seed 精确命中次数、命中局部 AT 含量",
        "k-mer 特征：miRNA k 值、gene k 值、TF-IDF min_df、gene 最大特征数",
        "生物学增强特征：经典 seed 类型、GU wobble、3' 端位置权重、局部/全局比对、反向互补 k-mer 交互",
        "RNA 能量特征：ViennaRNA duplex/MFE 能量、seed 能量、配对比例",
        "非深度学习表征特征：k-mer 共现 PPMI + TruncatedSVD embedding 维度、上下文窗口、最小词频",
        "模型集合：LightGBM、XGBoost、ExtraTrees、RandomForest、SVM、KNN、Factorization Machine 及平均/stacking 集成方式",
        "模型超参：树数量、学习率、深度、叶子数、采样比例、正负样本权重、正则化、SVM C/gamma、FM 因子数",
        "训练方法：单 seed/多 seed、StratifiedKFold/StratifiedGroupKFold、hard negative、hyperopt",
        "后处理：分类阈值、预测正样本比例、是否按固定正样本数量截断",
        "提交选择：本地 OOF 与线上测试集分布差异、提交次数和阈值试探顺序",
    ]


def build_features(
    config: dict[str, Any],
    bundle: DatasetBundle | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Build aligned train and test feature matrices from the configuration."""
    if bundle is None:
        bundle = build_dataset_bundle()
    train_blocks = []
    test_blocks = []
    feature_counts: dict[str, int] = {}

    def add_block(name: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Append one feature block after aligning train and test columns."""
        train_blocks.append(train_df)
        test_blocks.append(test_df)
        feature_counts[name] = train_df.shape[1]

    if _feature_flag(config, "basic"):
        add_block("basic", compute_sequence_features(bundle.train), compute_sequence_features(bundle.test))
    if _feature_flag(config, "match"):
        add_block("match", compute_match_features(bundle.train), compute_match_features(bundle.test))
    if _feature_flag(config, "advanced"):
        add_block("advanced", compute_advanced_features(bundle.train), compute_advanced_features(bundle.test))
    if _feature_flag(config, "rna_energy"):
        from src.features.rna_energy_features import compute_rna_energy_features

        add_block("rna_energy", compute_rna_energy_features(bundle.train), compute_rna_energy_features(bundle.test))
    if _feature_flag(config, "entity_counts"):
        train_entity, test_entity = compute_entity_count_features(bundle.train, bundle.test)
        add_block("entity_counts", train_entity, test_entity)
    if _feature_flag(config, "seed_variants"):
        add_block("seed_variants", compute_seed_variant_features(bundle.train), compute_seed_variant_features(bundle.test))
    if _feature_flag(config, "dinucleotide"):
        add_block("dinucleotide", compute_dinucleotide_features(bundle.train), compute_dinucleotide_features(bundle.test))
    if _feature_flag(config, "targetscan"):
        add_block("targetscan", compute_targetscan_features(bundle.train), compute_targetscan_features(bundle.test))
    if _feature_flag(config, "kmer_interaction"):
        add_block(
            "kmer_interaction",
            compute_kmer_interaction_features(bundle.train),
            compute_kmer_interaction_features(bundle.test),
        )
    if _feature_flag(config, "alignment"):
        from src.features.alignment_features import compute_alignment_features

        add_block("alignment", compute_alignment_features(bundle.train), compute_alignment_features(bundle.test))
    if _feature_flag(config, "position"):
        add_block("position", compute_position_features(bundle.train), compute_position_features(bundle.test))
    if _feature_flag(config, "seed_type"):
        add_block("seed_type", compute_seed_type_features(bundle.train), compute_seed_type_features(bundle.test))
    if _feature_flag(config, "embedding"):
        embedding_cfg = config.get("embedding", {})
        embedding = EmbeddingFeaturizer()
        train_embedding = embedding(
            bundle.train,
            k=int(embedding_cfg.get("k", 3)),
            dim=int(embedding_cfg.get("dim", 12)),
            context_radius=int(embedding_cfg.get("context_radius", 2)),
            min_count=int(embedding_cfg.get("min_count", 2)),
        )
        test_embedding = embedding(bundle.test)
        add_block("embedding", train_embedding, test_embedding)
    if _feature_flag(config, "kmer"):
        kmer_cfg = config.get("kmer", {})
        train_kmer, test_kmer = compute_kmer_features(
            bundle.train,
            bundle.test,
            mirna_k=int(kmer_cfg.get("mirna_k", 3)),
            gene_k=int(kmer_cfg.get("gene_k", 3)),
            gene_max_features=int(kmer_cfg.get("gene_max_features", 256)),
        )
        add_block("kmer", train_kmer, test_kmer)

    if not train_blocks:
        raise ValueError("No features selected in config.")

    train_X = pd.concat(train_blocks, axis=1)
    test_X = pd.concat(test_blocks, axis=1)
    train_X = train_X.loc[:, ~train_X.columns.duplicated()]
    test_X = test_X.loc[:, ~test_X.columns.duplicated()]
    test_X = test_X[train_X.columns]
    y = bundle.train[TARGET_COLUMN].copy()
    train_meta = bundle.train[[GENE_COLUMN, MIRNA_COLUMN]]
    test_meta = bundle.test[[GENE_COLUMN, MIRNA_COLUMN]]
    feature_counts["total"] = train_X.shape[1]
    return train_X, test_X, y, train_meta, test_meta, feature_counts


def _scale_pos_weight(y: pd.Series, config: dict[str, Any]) -> float:
    """Compute the positive-class weight from labels and configuration."""
    value = config.get("training", {}).get("scale_pos_weight", "auto")
    if value == "auto":
        pos = int(y.sum())
        neg = int((1 - y).sum())
        return neg / pos
    return float(value)


def _build_model(name: str, seed: int, scale_pos_weight: float, params: dict[str, Any]):
    """Create a classifier by model name, seed, and parameter overrides."""
    if name == "lgbm":
        return LGBMClassifier(
            n_estimators=int(params.get("n_estimators", 3000)),
            learning_rate=float(params.get("learning_rate", 0.01)),
            max_depth=int(params.get("max_depth", 6)),
            num_leaves=int(params.get("num_leaves", 31)),
            min_child_samples=int(params.get("min_child_samples", 10)),
            subsample=float(params.get("subsample", 0.8)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            reg_alpha=float(params.get("reg_alpha", 0.1)),
            reg_lambda=float(params.get("reg_lambda", 0.1)),
            random_state=seed,
            early_stopping_round=int(params.get("early_stopping_round", 100)),
            verbose=-1,
            scale_pos_weight=scale_pos_weight,
        )
    if name == "xgb":
        return XGBClassifier(
            n_estimators=int(params.get("n_estimators", 3000)),
            learning_rate=float(params.get("learning_rate", 0.01)),
            max_depth=int(params.get("max_depth", 6)),
            subsample=float(params.get("subsample", 0.8)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            reg_alpha=float(params.get("reg_alpha", 0.1)),
            reg_lambda=float(params.get("reg_lambda", 0.1)),
            random_state=seed,
            early_stopping_rounds=int(params.get("early_stopping_rounds", 100)),
            eval_metric="logloss",
            verbosity=0,
            scale_pos_weight=scale_pos_weight,
        )
    if name in {"extra_trees", "extratrees"}:
        return ExtraTreesClassifier(
            n_estimators=int(params.get("n_estimators", 800)),
            max_depth=params.get("max_depth"),
            min_samples_leaf=int(params.get("min_samples_leaf", 2)),
            max_features=params.get("max_features", "sqrt"),
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 500)),
            max_depth=params.get("max_depth", 12),
            min_samples_split=int(params.get("min_samples_split", 5)),
            min_samples_leaf=int(params.get("min_samples_leaf", 2)),
            max_features=params.get("max_features", "sqrt"),
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "svm":
        return make_pipeline(
            StandardScaler(),
            SVC(
                probability=True,
                kernel=params.get("kernel", "rbf"),
                C=float(params.get("C", 1.0)),
                gamma=params.get("gamma", "scale"),
                class_weight=params.get("class_weight", "balanced"),
                random_state=seed,
            ),
        )
    if name == "fm":
        from src.models.factorization_machine import FactorizationMachineClassifier

        return make_pipeline(
            StandardScaler(),
            FactorizationMachineClassifier(
                n_factors=int(params.get("n_factors", 8)),
                learning_rate=float(params.get("learning_rate", 0.001)),
                epochs=int(params.get("epochs", 300)),
                batch_size=int(params.get("batch_size", 128)),
                reg_w=float(params.get("reg_w", 0.001)),
                reg_v=float(params.get("reg_v", 0.001)),
                random_state=seed,
                verbose=int(params.get("verbose", 0)),
            ),
        )
    if name == "knn":
        return make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
                n_neighbors=int(params.get("n_neighbors", 35)),
                weights=params.get("weights", "distance"),
                p=int(params.get("p", 2)),
            ),
        )
    raise ValueError(f"Unsupported model: {name}")


def _fit_predict(model_name: str, model, X_tr, y_tr, X_val, y_val, test_X) -> tuple[np.ndarray, np.ndarray]:
    """Fit one fold model and return validation and test probabilities."""
    if model_name == "lgbm":
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="logloss")
    elif model_name == "xgb":
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_tr, y_tr)
    return model.predict_proba(X_val)[:, 1], model.predict_proba(test_X)[:, 1]


def _split_iterator(config: dict[str, Any], train_X: pd.DataFrame, y: pd.Series, train_meta: pd.DataFrame, seed: int):
    """Yield regular or grouped cross-validation splits from the configuration."""
    training = config.get("training", {})
    n_splits = int(training.get("n_splits", 5))
    strategy = training.get("fold_strategy", "stratified")
    group_column = training.get("group_column", "")
    if strategy == "stratified_group" and group_column:
        if group_column not in train_meta.columns:
            raise ValueError(f"group_column not found: {group_column}")
        groups = train_meta[group_column]
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return splitter.split(train_X, y, groups)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return splitter.split(train_X, y)


def _lark_cli_base() -> list[str]:
    """Build the base Lark CLI command."""
    appdata = os.environ.get("APPDATA", "")
    os.environ["PATH"] = f"C:\\Program Files\\nodejs;{appdata}\\npm;{os.environ.get('PATH', '')}"
    return ["lark-cli.cmd"]


def _next_lark_row(token: str, sheet_id: str) -> int:
    """Query the next writable row in the Lark sheet."""
    cmd = _lark_cli_base() + [
        "sheets",
        "+csv-get",
        "--spreadsheet-token",
        token,
        "--sheet-id",
        sheet_id,
        "--range",
        "A1:Q500",
    ]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    payload = json.loads(result.stdout)
    non_empty_rows = []
    for line in payload.get("data", {}).get("annotated_csv", "").splitlines():
        row_match = re.match(r"\[row=(\d+)\]\s*(.*)", line)
        if row_match and row_match.group(2).strip().replace(",", ""):
            non_empty_rows.append(int(row_match.group(1)))
    if non_empty_rows:
        return max(non_empty_rows) + 1
    current_region = payload.get("data", {}).get("current_region", "")
    match = re.search(r":?[A-Z]+(\d+)$", current_region)
    if match:
        return int(match.group(1)) + 1
    rows = [int(m.group(1)) for m in re.finditer(r"\[row=(\d+)\]", result.stdout)]
    return max(non_empty_rows or rows or [1]) + 1


def _write_lark_rows(token: str, sheet_id: str, start_row: int, rows: list[list[Any]]) -> None:
    """Write experiment result rows to the Lark sheet."""
    fd, temp_path = tempfile.mkstemp(prefix="lark_experiment_", suffix=".csv", dir=PROJECT_ROOT, text=True)
    os.close(fd)
    temp_file = Path(temp_path)
    try:
        with open(temp_file, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        cmd = _lark_cli_base() + [
            "sheets",
            "+csv-put",
            "--spreadsheet-token",
            token,
            "--sheet-id",
            sheet_id,
            "--start-cell",
            f"A{start_row}",
            "--csv",
            f"@{temp_file.name}",
        ]
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        if result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout)
    finally:
        temp_file.unlink(missing_ok=True)


def record_experiment_to_lark(config: dict[str, Any], summary: dict[str, Any]) -> None:
    """Record experiment settings, metrics, and outputs to the Lark sheet."""
    lark_cfg = config.get("lark", {})
    if not lark_cfg.get("enabled", True):
        print("lark_record=disabled")
        return

    token = lark_cfg.get("spreadsheet_token", DEFAULT_LARK_TOKEN)
    sheet_id = lark_cfg.get("sheet_id", DEFAULT_LARK_SHEET_ID)
    feature_names = enabled_feature_names(config)
    feature_combo = " + ".join(FEATURE_CN.get(name, name) for name in feature_names)
    kmer_cfg = config.get("kmer", {})
    training = config.get("training", {})
    seeds = "?".join(str(seed) for seed in training.get("seeds", []))
    fold_strategy = f"{training.get('fold_strategy', 'stratified')} {training.get('n_splits', 5)}?"
    model_plan = " + ".join(summary.get("models", []))
    spw = f"{summary.get('scale_pos_weight', 0):.4f}"

    rows = []
    for item in summary.get("submissions", []):
        rows.append(
            [
                summary.get("experiment_name", ""),
                feature_combo,
                "????????GC??????" if "basic" in feature_names else "",
                "seed???????????????????" if "match" in feature_names else "",
                "??AT???????seed????" if "advanced" in feature_names else "",
                f"miRNA k={kmer_cfg.get('mirna_k', '')}?gene k={kmer_cfg.get('gene_k', '')}?gene????={kmer_cfg.get('gene_max_features', '')}"
                + ("??TargetScan????/??AU/3???" if "targetscan" in feature_names else "")
                if "kmer" in feature_names or "targetscan" in feature_names
                else "",
                model_plan,
                seeds,
                fold_strategy,
                spw,
                item.get("threshold", ""),
                item.get("file", ""),
                item.get("pos", ""),
                item.get("neg", ""),
                f"{item.get('pos_rate', 0):.4f}",
                "",
                f"{item.get('train_f1', 0):.4f}",
            ]
        )

    if not rows:
        print("lark_record=no_rows")
        return
    start_row = _next_lark_row(token, sheet_id)
    _write_lark_rows(token, sheet_id, start_row, rows)
    print(f"lark_record=ok start_row={start_row} rows={len(rows)}")


def record_experiment_to_lark(config: dict[str, Any], summary: dict[str, Any]) -> None:
    """Record experiment settings, metrics, and outputs to the Lark sheet."""
    lark_cfg = config.get("lark", {})
    if not lark_cfg.get("enabled", True):
        print("lark_record=disabled")
        return

    token = lark_cfg.get("spreadsheet_token", DEFAULT_LARK_TOKEN)
    sheet_id = lark_cfg.get("sheet_id", DEFAULT_LARK_SHEET_ID)
    feature_names = enabled_feature_names(config)
    feature_combo = " + ".join(FEATURE_CN.get(name, name) for name in feature_names)
    kmer_cfg = config.get("kmer", {})
    embedding_cfg = config.get("embedding", {})
    training = config.get("training", {})
    seeds = "?".join(str(seed) for seed in training.get("seeds", []))
    fold_strategy = f"{training.get('fold_strategy', 'stratified')} {training.get('n_splits', 5)}?"
    model_plan = " + ".join(summary.get("models", []))
    spw = f"{summary.get('scale_pos_weight', 0):.4f}"

    extra_feature_notes = []
    if "kmer_interaction" in feature_names:
        extra_feature_notes.append("???? 3-mer ????")
    if "alignment" in feature_names:
        extra_feature_notes.append("????/seed ??????")
    if "position" in feature_names:
        extra_feature_notes.append("seed ? 3' ????????")
    if "seed_type" in feature_names:
        extra_feature_notes.append("8mer/7mer/6mer ??? GU wobble")
    if "embedding" in feature_names:
        extra_feature_notes.append(
            "k-mer ?? SVD: "
            f"k={embedding_cfg.get('k', 3)}, dim={embedding_cfg.get('dim', 12)}, "
            f"context={embedding_cfg.get('context_radius', 2)}, min_count={embedding_cfg.get('min_count', 2)}"
        )

    kmer_note = ""
    if "kmer" in feature_names:
        kmer_note = (
            f"miRNA k={kmer_cfg.get('mirna_k', '')}?"
            f"gene k={kmer_cfg.get('gene_k', '')}?"
            f"gene ?????={kmer_cfg.get('gene_max_features', '')}"
        )
    if "targetscan" in feature_names:
        kmer_note = (kmer_note + "?" if kmer_note else "") + "TargetScan ????/?? AU/3' ???"
    if extra_feature_notes:
        kmer_note = (kmer_note + "?" if kmer_note else "") + "?".join(extra_feature_notes)

    rows = []
    for item in summary.get("submissions", []):
        rows.append(
            [
                summary.get("experiment_name", ""),
                feature_combo,
                "????????GC ??????" if "basic" in feature_names else "",
                "seed ???????????????????" if "match" in feature_names else "",
                "?? AT ??????? seed ????" if "advanced" in feature_names else "",
                kmer_note,
                model_plan,
                seeds,
                fold_strategy,
                spw,
                item.get("threshold", ""),
                item.get("file", ""),
                item.get("pos", ""),
                item.get("neg", ""),
                f"{item.get('pos_rate', 0):.4f}",
                "",
                f"{item.get('train_f1', 0):.4f}",
            ]
        )

    if not rows:
        print("lark_record=no_rows")
        return
    start_row = _next_lark_row(token, sheet_id)
    _write_lark_rows(token, sheet_id, start_row, rows)
    print(f"lark_record=ok start_row={start_row} rows={len(rows)}")


def _run_optuna_search(
    train_X: pd.DataFrame,
    y: pd.Series,
    enabled_models: list[str],
    config: dict[str, Any],
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Run Optuna hyperparameter search for the enabled models."""
    training = config.get("training", {})
    if not training.get("hyperopt_enabled", False):
        return {}
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("hyperopt_enabled=true requires installing optuna") from exc

    n_trials = int(training.get("hyperopt_trials", 20))
    X_tr, X_val, y_tr, y_val = train_test_split(
        train_X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=seed + 777,
    )
    tuned: dict[str, dict[str, Any]] = {}

    def lgbm_objective(trial):
        """Evaluate one LightGBM Optuna trial with validation F1."""
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 3000, step=250),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "num_leaves": trial.suggest_int("num_leaves", 15, 95, step=8),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 1.0, log=True),
            "early_stopping_round": 100,
        }
        model = _build_model("lgbm", seed, _scale_pos_weight(y, config), params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="logloss")
        return f1_score(y_val, model.predict_proba(X_val)[:, 1] >= 0.5)

    def xgb_objective(trial):
        """Evaluate one XGBoost Optuna trial with validation F1."""
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 3000, step=250),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 1.0, log=True),
            "early_stopping_rounds": 100,
        }
        model = _build_model("xgb", seed, _scale_pos_weight(y, config), params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return f1_score(y_val, model.predict_proba(X_val)[:, 1] >= 0.5)

    objectives = {"lgbm": lgbm_objective, "xgb": xgb_objective}
    for model_name in enabled_models:
        if model_name not in objectives:
            print(f"hyperopt skip model={model_name} reason=no_search_space")
            continue
        print(f"=== hyperopt model={model_name} trials={n_trials} ===")
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        study.optimize(objectives[model_name], n_trials=n_trials, show_progress_bar=False)
        tuned[model_name] = dict(study.best_params)
        print(f"hyperopt model={model_name} best_f1={study.best_value:.4f} params={tuned[model_name]}")
    return tuned


def _train_cv_predictions(
    train_X: pd.DataFrame,
    test_X: pd.DataFrame,
    y: pd.Series,
    train_meta: pd.DataFrame,
    config: dict[str, Any],
    enabled_models: list[str],
    spw: float,
    model_params: dict[str, dict[str, Any]],
    seeds: list[int],
) -> dict[str, Any]:
    """Train enabled models across folds and seeds and collect predictions."""
    oof_by_model = {name: np.zeros(len(y), dtype=float) for name in enabled_models}
    oof_count_by_model = {name: np.zeros(len(y), dtype=int) for name in enabled_models}
    test_sum_by_model = {name: np.zeros(len(test_X), dtype=float) for name in enabled_models}
    test_count_by_model = {name: 0 for name in enabled_models}

    for seed in seeds:
        print(f"=== seed={seed} ===")
        for fold_idx, (tr_idx, val_idx) in enumerate(_split_iterator(config, train_X, y, train_meta, seed)):
            X_tr, X_val = train_X.iloc[tr_idx], train_X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            fold_model_probas = []
            for model_name in enabled_models:
                model = _build_model(model_name, seed + fold_idx, spw, model_params.get(model_name, {}))
                val_p, test_p = _fit_predict(model_name, model, X_tr, y_tr, X_val, y_val, test_X)
                oof_by_model[model_name][val_idx] += val_p
                oof_count_by_model[model_name][val_idx] += 1
                test_sum_by_model[model_name] += test_p
                test_count_by_model[model_name] += 1
                fold_model_probas.append(val_p)
            val_fold = np.mean(fold_model_probas, axis=0)
            print(f"fold={fold_idx} f1@0.5={f1_score(y_val, val_fold >= 0.5):.4f}")

    model_oof = {}
    model_test = {}
    for name in enabled_models:
        model_oof[name] = oof_by_model[name] / np.maximum(oof_count_by_model[name], 1)
        model_test[name] = test_sum_by_model[name] / max(test_count_by_model[name], 1)

    training = config.get("training", {})
    ensemble_mode = training.get("ensemble_mode", "mean")
    if ensemble_mode == "stacking" and len(enabled_models) >= 2:
        stack_train = pd.DataFrame(model_oof)
        stack_test = pd.DataFrame(model_test)
        stacker = LogisticRegression(random_state=seeds[0] if seeds else 42, max_iter=1000)
        stacker.fit(stack_train, y)
        oof_proba = stacker.predict_proba(stack_train)[:, 1]
        test_proba = stacker.predict_proba(stack_test)[:, 1]
        print(f"stacking_weights={dict(zip(enabled_models, stacker.coef_[0]))}")
        mode_used = "stacking"
    else:
        oof_proba = np.mean([model_oof[name] for name in enabled_models], axis=0)
        test_proba = np.mean([model_test[name] for name in enabled_models], axis=0)
        mode_used = "mean"
        if ensemble_mode == "stacking":
            print("stacking skipped: at least two models are required")

    return {
        "oof_proba": oof_proba,
        "test_proba": test_proba,
        "model_oof": model_oof,
        "model_test": model_test,
        "ensemble_mode": mode_used,
    }


def _train_with_method(
    train_X: pd.DataFrame,
    test_X: pd.DataFrame,
    y: pd.Series,
    train_meta: pd.DataFrame,
    config: dict[str, Any],
    enabled_models: list[str],
    spw: float,
    model_params: dict[str, dict[str, Any]],
    seeds: list[int],
) -> dict[str, Any]:
    """Dispatch to regular CV or hard-negative training based on config."""
    method = config.get("training", {}).get("training_method", "normal")
    if method != "hard_negative":
        result = _train_cv_predictions(train_X, test_X, y, train_meta, config, enabled_models, spw, model_params, seeds)
        result["training_method"] = method
        return result

    print("=== hard_negative stage1 ===")
    stage1_cfg = json.loads(json.dumps(config))
    stage1_cfg.setdefault("training", {})["ensemble_mode"] = "mean"
    stage1 = _train_cv_predictions(train_X, test_X, y, train_meta, stage1_cfg, enabled_models, spw, model_params, seeds)
    threshold = float(config.get("training", {}).get("hard_negative_threshold", 0.8))
    top_fraction_raw = config.get("training", {}).get("hard_negative_top_fraction", "")
    top_fraction = float(top_fraction_raw) if str(top_fraction_raw).strip() else None

    neg_mask = y == 0
    if top_fraction is not None:
        n_select = max(1, int(neg_mask.sum() * top_fraction))
        cutoff = pd.Series(stage1["oof_proba"], index=y.index)[neg_mask].nlargest(n_select).iloc[-1]
        hard_neg = neg_mask & (stage1["oof_proba"] >= cutoff)
    else:
        hard_neg = neg_mask & (stage1["oof_proba"] >= threshold)
    stage2_mask = (y == 1) | hard_neg
    print(
        f"hard_negative selected={int(hard_neg.sum())}/{int(neg_mask.sum())} "
        f"stage2_total={int(stage2_mask.sum())}"
    )

    print("=== hard_negative stage2 ===")
    stage2_train_X = train_X.loc[stage2_mask].copy()
    stage2_y = y.loc[stage2_mask].copy()
    stage2_meta = train_meta.loc[stage2_mask].copy()
    result = _train_cv_predictions(
        stage2_train_X,
        test_X,
        stage2_y,
        stage2_meta,
        config,
        enabled_models,
        spw,
        model_params,
        seeds,
    )
    result["stage1_oof_proba"] = stage1["oof_proba"]
    result["stage2_mask"] = stage2_mask
    result["training_method"] = "hard_negative"
    return result


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Run the full experiment and produce metrics, submissions, and summary."""
    print("=== factors ===")
    for item in list_score_factors():
        print(f"- {item}")

    print("=== loading data and building features ===")
    train_X, test_X, y, train_meta, test_meta, feature_counts = build_features(config)
    spw = _scale_pos_weight(y, config)
    enabled_models = [name for name, enabled in config.get("models", {}).items() if enabled]
    if not enabled_models:
        raise ValueError("No models selected in config.")

    print(f"features={feature_counts}")
    print(f"models={enabled_models} spw={spw:.4f}")

    seeds = [int(seed) for seed in config.get("training", {}).get("seeds", [42])]
    model_params = config.get("model_params", {})
    tuned_params = _run_optuna_search(train_X, y, enabled_models, config, seeds[0] if seeds else 42)
    for model_name, params in tuned_params.items():
        model_params.setdefault(model_name, {})
        model_params[model_name].update(params)

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
    oof_proba = train_result["oof_proba"]
    test_proba = train_result["test_proba"]
    score_y = y.loc[train_result["stage2_mask"]] if "stage2_mask" in train_result else y
    score_oof = oof_proba if "stage2_mask" in train_result else pd.Series(oof_proba, index=y.index)
    if "stage2_mask" in train_result:
        score_oof = pd.Series(oof_proba, index=score_y.index)

    best_t = 0.5
    best_f1 = 0.0
    for t in np.linspace(0.30, 0.65, 36):
        cur = f1_score(score_y, score_oof >= t)
        if cur > best_f1:
            best_f1 = cur
            best_t = float(t)

    print("=== OOF F1 grid ===")
    print(f"best_f1={best_f1:.4f} best_threshold={best_t:.3f}")
    threshold_metrics = {}
    for t in config.get("thresholds", []):
        t = float(t)
        threshold_metrics[f"{t:.2f}"] = float(f1_score(score_y, score_oof >= t))
        print(f"t={t:.2f} f1={threshold_metrics[f'{t:.2f}']:.4f}")

    print("=== test proba quantiles ===")
    for q in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
        print(f"q{int(q * 100)}={np.quantile(test_proba, q):.4f}")

    output_cfg = config.get("output", {})
    base_run_name = config.get("experiment_name", "experiment")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    add_run_id = bool(output_cfg.get("add_run_id", True))
    run_name = f"{base_run_name}_{run_id}" if add_run_id else base_run_name
    prefix_base = output_cfg.get("submission_prefix", base_run_name)
    prefix = f"{prefix_base}_{run_id}" if add_run_id else prefix_base
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    template = pd.read_csv(SUBMIT_EXAMPLE_FILE)
    submission_rows = []
    for t in config.get("thresholds", []):
        t = float(t)
        sub = test_meta.copy()
        sub[TARGET_COLUMN] = (test_proba >= t).astype(int)
        sub = sub[list(template.columns)]
        name = f"{prefix}_t{int(round(t * 100))}.csv"
        sub.to_csv(SUBMISSION_DIR / name, index=False)
        pos = int((sub[TARGET_COLUMN] == 1).sum())
        row = {
            "file": name,
            "threshold": t,
            "pos": pos,
            "neg": int(len(sub) - pos),
            "pos_rate": float(pos / len(sub)),
            "train_f1": float(threshold_metrics.get(f"{t:.2f}", f1_score(score_y, score_oof >= t))),
        }
        submission_rows.append(row)
        print(f"{name}: pos={row['pos']} neg={row['neg']} pos_rate={row['pos_rate']:.3f} train_f1={row['train_f1']:.4f}")

    OOF_DIR.mkdir(parents=True, exist_ok=True)
    if output_cfg.get("save_oof", True):
        pd.Series(oof_proba, name="oof_proba").to_csv(OOF_DIR / f"oof_{run_name}.csv", index=True, header=True)
    if output_cfg.get("save_test_proba", True):
        pd.Series(test_proba, name="test_proba").to_csv(OOF_DIR / f"test_proba_{run_name}.csv", index=True, header=True)

    summary = {
        "experiment_name": run_name,
        "base_experiment_name": base_run_name,
        "run_id": run_id,
        "feature_counts": feature_counts,
        "models": enabled_models,
        "training_method": train_result.get("training_method", "normal"),
        "ensemble_mode": train_result.get("ensemble_mode", "mean"),
        "hyperopt_enabled": bool(config.get("training", {}).get("hyperopt_enabled", False)),
        "tuned_params": tuned_params,
        "scale_pos_weight": spw,
        "best_threshold": best_t,
        "best_train_f1": float(best_f1),
        "threshold_metrics": threshold_metrics,
        "submissions": submission_rows,
    }
    summary_path = OOF_DIR / f"summary_{run_name}.json"
    save_config(summary, summary_path)
    print(f"summary={summary_path}")

    try:
        record_experiment_to_lark(config, summary)
    except Exception as exc:
        print(f"lark_record=failed error={exc}")

    print("=== RESULT SUMMARY ===")
    print(f"experiment_name={summary['experiment_name']}")
    print(f"best_train_f1={summary['best_train_f1']:.4f} best_threshold={summary['best_threshold']:.3f}")
    print("submission_files:")
    for row in submission_rows:
        print(
            f"- {row['file']} threshold={row['threshold']:.2f} "
            f"train_f1={row['train_f1']:.4f} pos={row['pos']} neg={row['neg']}"
        )

    return summary


def main() -> None:
    """Parse command-line arguments and run one configured experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to experiment JSON config.")
    parser.add_argument("--list-factors", action="store_true", help="Print score factors and exit.")
    args = parser.parse_args()

    if args.list_factors:
        for item in list_score_factors():
            print(f"- {item}")
        return
    run_experiment(load_config(args.config))


if __name__ == "__main__":
    main()
