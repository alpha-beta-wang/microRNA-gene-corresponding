"""种子区变体特征模块，比较不同 miRNA seed 窗口的匹配情况。"""

import pandas as pd

from src.config import GENE_SEQUENCE_COLUMN, MIRNA_SEQUENCE_COLUMN
from src.features.sequence_match_features import _reverse_complement


def _count_substring(needle: str, haystack: str) -> int:
    """Count occurrences of a substring in a target sequence."""
    if not needle or not haystack:
        return 0
    count = 0
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            return count
        count += 1
        start = idx + 1


def _max_consecutive_match(short: str, long: str) -> int:
    """Compute the longest consecutive match between two sequences."""
    best = 0
    for i in range(len(long) - len(short) + 1):
        cur = 0
        for j, base in enumerate(short):
            if long[i + j] == base:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
    return best


def compute_seed_variant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute seed variant, offset-match, and match-strength features."""
    gene_seq = df[GENE_SEQUENCE_COLUMN].fillna("").str.upper().str.replace("U", "T", regex=False)
    mirna_seq = df[MIRNA_SEQUENCE_COLUMN].fillna("").str.upper()
    features = pd.DataFrame(index=df.index)

    seed_specs = {
        "seed_2_7": (1, 7),
        "seed_2_8": (1, 8),
        "seed_3_8": (2, 8),
        "seed_3_9": (2, 9),
        "seed_4_10": (3, 10),
    }
    for name, (start, end) in seed_specs.items():
        seeds = [m[start:end] if len(m) >= end else "" for m in mirna_seq]
        rc_seeds = [_reverse_complement(seed) for seed in seeds]
        features[f"{name}_revcomp_hit"] = [int(seed in gene) if seed else 0 for seed, gene in zip(rc_seeds, gene_seq)]
        features[f"{name}_revcomp_count"] = [_count_substring(seed, gene) for seed, gene in zip(rc_seeds, gene_seq)]
        features[f"{name}_revcomp_max_consecutive"] = [
            _max_consecutive_match(seed, gene) if seed else 0 for seed, gene in zip(rc_seeds, gene_seq)
        ]

    return features.fillna(0)
