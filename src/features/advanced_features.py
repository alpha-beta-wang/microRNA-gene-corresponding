"""高级命中特征模块，统计精确 seed 命中和局部 AT 环境。"""

import pandas as pd

from src.config import GENE_SEQUENCE_COLUMN, MIRNA_SEQUENCE_COLUMN
from src.features.sequence_match_features import _reverse_complement


def _count_substring(needle: str, haystack: str) -> int:
    if not needle or not haystack:
        return 0
    count = 0
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            break
        count += 1
        start = idx + 1
    return count


def _first_hit_local_at(needle: str, haystack: str, window: int = 20) -> float:
    if not needle or not haystack:
        return 0.0
    idx = haystack.find(needle)
    if idx == -1:
        return 0.0
    start = max(0, idx - window)
    end = min(len(haystack), idx + len(needle) + window)
    region = haystack[start:end]
    if not region:
        return 0.0
    return (region.count("A") + region.count("T")) / len(region)


def compute_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    gene_seq = df[GENE_SEQUENCE_COLUMN].fillna("")
    mirna_seq = df[MIRNA_SEQUENCE_COLUMN].fillna("")
    features = pd.DataFrame(index=df.index)

    seed_rc = [
        _reverse_complement(m[1:8]) if len(m) >= 8 else ""
        for m in mirna_seq
    ]

    features["seed_revcomp_exact_count"] = [
        _count_substring(rc, g) for rc, g in zip(seed_rc, gene_seq)
    ]

    features["seed_hit_local_at"] = [
        _first_hit_local_at(rc, g) for rc, g in zip(seed_rc, gene_seq)
    ]

    return features.fillna(0)
"""高级命中特征模块，统计精确 seed 命中和局部 AT 环境。"""
