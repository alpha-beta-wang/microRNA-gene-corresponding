"""位置特征模块，刻画种子命中相对 gene 3' 端的位置。"""

import pandas as pd

from src.config import GENE_SEQUENCE_COLUMN, MIRNA_SEQUENCE_COLUMN
from src.features.sequence_match_features import COMPLEMENT


def _reverse_complement(seq: str) -> str:
    """Return the reverse complement of an RNA or DNA sequence."""
    return seq.translate(COMPLEMENT)[::-1]


def _first_seed_position_from_3prime(seed: str, gene: str) -> float:
    """Normalized distance of first seed occurrence from 3' end (0=3' end, 1=5' end). -1 if not found."""
    if len(seed) < 7 or len(gene) == 0:
        return -1.0
    idx = gene.find(seed)
    if idx == -1:
        return -1.0
    return 1.0 - (idx / len(gene))


def _best_match_position_from_3prime(short: str, long: str) -> float:
    """Position (normalized from 3' end) of window with max consecutive match."""
    if not short or not long or len(long) < len(short):
        return -1.0
    best_pos = -1
    best_val = 0
    for i in range(len(long) - len(short) + 1):
        cur = 0
        for j in range(len(short)):
            if long[i + j] == short[j]:
                cur += 1
            else:
                if cur > best_val:
                    best_val = cur
                    best_pos = i
                cur = 0
        if cur > best_val:
            best_val = cur
            best_pos = i
    if best_pos < 0:
        return -1.0
    return 1.0 - (best_pos / len(long))


def _seed_count_weighted_3prime(seed: str, gene: str) -> float:
    """Count seed occurrences weighted by proximity to 3' end. Each match contributes (1 - position_from_3prime)."""
    if len(seed) < 7 or len(gene) == 0:
        return 0.0
    total = 0.0
    start = 0
    glen = len(gene)
    while True:
        idx = gene.find(seed, start)
        if idx == -1:
            break
        weight = 1.0 - (idx / glen)  # higher near 3' end
        total += weight
        start = idx + 1
    return total


def _seed_in_3prime_region(seed: str, gene: str, fraction: float = 0.3) -> int:
    """Whether seed occurs in the last `fraction` of the gene (3' region)."""
    if len(seed) < 7 or len(gene) == 0:
        return 0
    cutoff = int(len(gene) * (1.0 - fraction))
    region = gene[cutoff:]
    return int(seed in region)


def compute_position_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute seed-match position, window distribution, and coverage features."""
    gene_seq = df[GENE_SEQUENCE_COLUMN].fillna("")
    mirna_seq = df[MIRNA_SEQUENCE_COLUMN].fillna("")
    features = pd.DataFrame(index=df.index)

    seeds = [m[1:8] if len(m) >= 8 else "" for m in mirna_seq]
    seeds_rc = [_reverse_complement(s) if s else "" for s in seeds]

    features["position__seed_first_from_3prime"] = [
        _first_seed_position_from_3prime(s, g) for s, g in zip(seeds, gene_seq)
    ]
    features["position__seed_rc_first_from_3prime"] = [
        _first_seed_position_from_3prime(s, g) for s, g in zip(seeds_rc, gene_seq)
    ]
    features["position__seed_best_match_from_3prime"] = [
        _best_match_position_from_3prime(s, g) for s, g in zip(seeds, gene_seq)
    ]
    features["position__mirna_best_match_from_3prime"] = [
        _best_match_position_from_3prime(m, g) for m, g in zip(mirna_seq, gene_seq)
    ]
    features["position__seed_count_3prime_weighted"] = [
        _seed_count_weighted_3prime(s, g) for s, g in zip(seeds, gene_seq)
    ]
    features["position__seed_rc_count_3prime_weighted"] = [
        _seed_count_weighted_3prime(s, g) for s, g in zip(seeds_rc, gene_seq)
    ]
    features["position__seed_in_3prime_region"] = [
        _seed_in_3prime_region(s, g, 0.3) for s, g in zip(seeds, gene_seq)
    ]
    features["position__seed_rc_in_3prime_region"] = [
        _seed_in_3prime_region(s, g, 0.3) for s, g in zip(seeds_rc, gene_seq)
    ]

    return features.fillna(-1.0)
