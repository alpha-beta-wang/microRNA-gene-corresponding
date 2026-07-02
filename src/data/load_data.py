"""数据加载与合并模块，构造训练集、测试集和序列元数据。"""

import re
from dataclasses import dataclass
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
    if value is None or pd.isna(value):
        return None
    cleaned = _SEQUENCE_CLEAN_PATTERN.sub("", str(value)).upper().replace('"', "")
    return cleaned or None


def load_train_frame() -> pd.DataFrame:
    return pd.read_csv(TRAIN_FILE)


def load_test_frame() -> pd.DataFrame:
    return pd.read_csv(TEST_FILE)


def load_gene_sequences() -> pd.DataFrame:
    frame = pd.read_csv(GENE_SEQUENCE_FILE).rename(
        columns={"label": GENE_COLUMN, "sequence": GENE_SEQUENCE_COLUMN}
    )
    frame[GENE_COLUMN] = frame[GENE_COLUMN].astype(str).str.strip()
    frame[GENE_SEQUENCE_COLUMN] = frame[GENE_SEQUENCE_COLUMN].map(clean_sequence)
    return frame.drop_duplicates(subset=[GENE_COLUMN], keep="first")


def load_mirna_sequences() -> pd.DataFrame:
    frame = pd.read_csv(MIRNA_SEQUENCE_FILE).rename(
        columns={"mirna": MIRNA_COLUMN, "seq": MIRNA_SEQUENCE_COLUMN}
    )
    frame[MIRNA_COLUMN] = frame[MIRNA_COLUMN].astype(str).str.strip()
    frame[MIRNA_SEQUENCE_COLUMN] = frame[MIRNA_SEQUENCE_COLUMN].map(clean_sequence)
    return frame.drop_duplicates(subset=[MIRNA_COLUMN], keep="first")


_LABEL_MAP = {"Functional MTI": 1, "Non-Functional MTI": 0}


def _normalize_pair_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized[GENE_COLUMN] = normalized[GENE_COLUMN].astype(str).str.strip()
    normalized[MIRNA_COLUMN] = normalized[MIRNA_COLUMN].astype(str).str.strip()
    if "label" in normalized.columns:
        normalized = normalized.rename(columns={"label": TARGET_COLUMN})
    if TARGET_COLUMN in normalized.columns:
        col = normalized[TARGET_COLUMN]
        if col.dtype == object or not pd.api.types.is_integer_dtype(col):
            normalized[TARGET_COLUMN] = col.astype(str).str.strip().map(_LABEL_MAP).astype(int)
    return normalized


def _merge_sequences(frame: pd.DataFrame, gene_sequences: pd.DataFrame, mirna_sequences: pd.DataFrame) -> pd.DataFrame:
    merged = frame.merge(gene_sequences, on=GENE_COLUMN, how="left")
    merged = merged.merge(mirna_sequences, on=MIRNA_COLUMN, how="left")
    return merged


def build_dataset_bundle() -> DatasetBundle:
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


def save_merged_frames(bundle: DatasetBundle) -> None:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    bundle.train.to_parquet(FEATURE_DIR / "train_merged.parquet", index=False)
    bundle.test.to_parquet(FEATURE_DIR / "test_merged.parquet", index=False)
"""数据加载与合并模块，构造训练集、测试集和序列元数据。"""
