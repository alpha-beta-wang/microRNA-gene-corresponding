"""数据加载与合并模块，构造训练集、测试集和序列元数据。"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import (
    FEATURE_DIR,
    GENE_COLUMN,
    GENE_SEQUENCE_COLUMN,
    GENE_SEQUENCE_FILE,
    MIRNA_COLUMN,
    MIRNA_SEQUENCE_COLUMN,
    MIRNA_SEQUENCE_FILE,
    TARGET_COLUMN,
    TEST_FILE,
    TRAIN_FILE,
)

_SEQUENCE_CLEAN_PATTERN = re.compile(r"\s+")


@dataclass
class DatasetBundle:
    train: pd.DataFrame
    test: pd.DataFrame
    train_missing_gene_sequences: int
    train_missing_mirna_sequences: int
    test_missing_gene_sequences: int
    test_missing_mirna_sequences: int


def clean_sequence(value: Optional[str]) -> Optional[str]:
    """清洗单条序列字符串并统一为大写。"""
    if value is None or pd.isna(value):
        return None
    cleaned = _SEQUENCE_CLEAN_PATTERN.sub("", str(value)).upper().replace('"', "")
    return cleaned or None


def load_train_frame() -> pd.DataFrame:
    """读取默认训练集配对表。"""
    return pd.read_csv(TRAIN_FILE)


def load_test_frame() -> pd.DataFrame:
    """读取默认测试集配对表。"""
    return pd.read_csv(TEST_FILE)


def _load_gene_sequences_from_path(path: str | Path) -> pd.DataFrame:
    """从指定路径读取 gene 序列表。"""
    frame = pd.read_csv(path).rename(
        columns={"label": GENE_COLUMN, "sequence": GENE_SEQUENCE_COLUMN}
    )
    frame[GENE_COLUMN] = frame[GENE_COLUMN].astype(str).str.strip()
    frame[GENE_SEQUENCE_COLUMN] = frame[GENE_SEQUENCE_COLUMN].map(clean_sequence)
    return frame.drop_duplicates(subset=[GENE_COLUMN], keep="first")  # 同名 gene 只保留首条序列


def load_gene_sequences() -> pd.DataFrame:
    """读取默认 gene 序列表。"""
    return _load_gene_sequences_from_path(GENE_SEQUENCE_FILE)


def _load_mirna_sequences_from_path(path: str | Path) -> pd.DataFrame:
    """从指定路径读取 miRNA 序列表。"""
    frame = pd.read_csv(path).rename(
        columns={"mirna": MIRNA_COLUMN, "seq": MIRNA_SEQUENCE_COLUMN}
    )
    frame[MIRNA_COLUMN] = frame[MIRNA_COLUMN].astype(str).str.strip()
    frame[MIRNA_SEQUENCE_COLUMN] = frame[MIRNA_SEQUENCE_COLUMN].map(clean_sequence)
    return frame.drop_duplicates(subset=[MIRNA_COLUMN], keep="first")  # 同名 miRNA 只保留首条序列


def load_mirna_sequences() -> pd.DataFrame:
    """读取默认 miRNA 序列表。"""
    return _load_mirna_sequences_from_path(MIRNA_SEQUENCE_FILE)


_LABEL_MAP = {"Functional MTI": 1, "Non-Functional MTI": 0}


def _normalize_pair_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """统一配对表中的主键列与标签列格式。"""
    normalized = frame.copy()
    normalized[GENE_COLUMN] = normalized[GENE_COLUMN].astype(str).str.strip()
    normalized[MIRNA_COLUMN] = normalized[MIRNA_COLUMN].astype(str).str.strip()
    if "label" in normalized.columns:
        normalized = normalized.rename(columns={"label": TARGET_COLUMN})  # 统一标签列名
    if TARGET_COLUMN in normalized.columns:
        col = normalized[TARGET_COLUMN]
        if col.dtype == object or not pd.api.types.is_integer_dtype(col):
            normalized[TARGET_COLUMN] = col.astype(str).str.strip().map(_LABEL_MAP).astype(int)  # 文本标签映射为 0/1
    return normalized


def _merge_sequences(frame: pd.DataFrame, gene_sequences: pd.DataFrame, mirna_sequences: pd.DataFrame) -> pd.DataFrame:
    """将配对表与 gene、miRNA 序列表做左连接。"""
    merged = frame.merge(gene_sequences, on=GENE_COLUMN, how="left")
    merged = merged.merge(mirna_sequences, on=MIRNA_COLUMN, how="left")
    return merged


def build_dataset_bundle() -> DatasetBundle:
    """读取默认数据目录并构造完整数据包。"""
    train = _normalize_pair_columns(load_train_frame())
    test = _normalize_pair_columns(load_test_frame())
    gene_sequences = load_gene_sequences()
    mirna_sequences = load_mirna_sequences()

    train_merged = _merge_sequences(train, gene_sequences, mirna_sequences)
    test_merged = _merge_sequences(test, gene_sequences, mirna_sequences)

    return DatasetBundle(
        train=train_merged,
        test=test_merged,
        train_missing_gene_sequences=int(train_merged[GENE_SEQUENCE_COLUMN].isna().sum()),
        train_missing_mirna_sequences=int(train_merged[MIRNA_SEQUENCE_COLUMN].isna().sum()),
        test_missing_gene_sequences=int(test_merged[GENE_SEQUENCE_COLUMN].isna().sum()),
        test_missing_mirna_sequences=int(test_merged[MIRNA_SEQUENCE_COLUMN].isna().sum()),
    )


def build_dataset_bundle_from_root(dataset_root: str | Path) -> DatasetBundle:
    """从指定数据目录读取并构造完整数据包。"""
    root = Path(dataset_root)
    train_dir = root / "train_dataset"

    train = _normalize_pair_columns(pd.read_csv(train_dir / "Train.csv"))
    test = _normalize_pair_columns(pd.read_csv(root / "test_dataset.csv"))
    gene_sequences = _load_gene_sequences_from_path(train_dir / "gene_seq.csv")
    mirna_sequences = _load_mirna_sequences_from_path(train_dir / "mirna_seq.csv")

    train_merged = _merge_sequences(train, gene_sequences, mirna_sequences)
    test_merged = _merge_sequences(test, gene_sequences, mirna_sequences)

    return DatasetBundle(
        train=train_merged,
        test=test_merged,
        train_missing_gene_sequences=int(train_merged[GENE_SEQUENCE_COLUMN].isna().sum()),
        train_missing_mirna_sequences=int(train_merged[MIRNA_SEQUENCE_COLUMN].isna().sum()),
        test_missing_gene_sequences=int(test_merged[GENE_SEQUENCE_COLUMN].isna().sum()),
        test_missing_mirna_sequences=int(test_merged[MIRNA_SEQUENCE_COLUMN].isna().sum()),
    )


def save_merged_frames(bundle: DatasetBundle) -> None:
    """将合并后的训练集与测试集持久化为 parquet。"""
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    bundle.train.to_parquet(FEATURE_DIR / "train_merged.parquet", index=False)
    bundle.test.to_parquet(FEATURE_DIR / "test_merged.parquet", index=False)
