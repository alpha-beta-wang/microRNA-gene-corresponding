"""miRNA seed type classification features.

The four canonical seed types are the strongest biological signal for
functional miRNA-target interactions (Bartel 2009):

  8mer    — seed(2-8) RC match in gene + A at position matching miRNA pos1
  7mer-m8 — seed(2-8) RC match in gene (no A1 requirement)
  7mer-A1 — seed(2-7) RC match in gene + A at position matching miRNA pos1
  6mer    — seed(2-7) RC match in gene

All positions are 1-indexed per convention; in Python slices:
  seed(2-8) = m[1:8]   (7 bases)
  seed(2-7) = m[1:7]   (6 bases)
"""

import pandas as pd

from src.config import GENE_SEQUENCE_COLUMN, MIRNA_SEQUENCE_COLUMN
from src.features.sequence_match_features import _reverse_complement


def _classify_seed(mirna: str, gene: str) -> tuple[int, int, int, int]:
    """Return (is_8mer, is_7mer_m8, is_7mer_a1, is_6mer) for one pair."""
    if len(mirna) < 8 or len(gene) < 7:
        return 0, 0, 0, 0

    seed_7rc = _reverse_complement(mirna[1:8])  # 7-base RC for 7mer-m8 / 8mer
    seed_6rc = _reverse_complement(mirna[1:7])  # 6-base RC for 6mer / 7mer-A1

    is_8mer = 0
    is_7mer_m8 = 0
    is_7mer_a1 = 0
    is_6mer = 0

    # Scan gene for seed_7rc matches
    start = 0
    while True:
        idx = gene.find(seed_7rc, start)
        if idx == -1:
            break
        is_7mer_m8 = 1
        # 8mer: the base immediately after the 7-mer match must be A
        if idx + 7 < len(gene) and gene[idx + 7] == "A":
            is_8mer = 1
        start = idx + 1

    # Scan gene for seed_6rc matches
    start = 0
    while True:
        idx = gene.find(seed_6rc, start)
        if idx == -1:
            break
        is_6mer = 1
        # 7mer-A1: the base immediately after the 6-mer match must be A
        if idx + 6 < len(gene) and gene[idx + 6] == "A":
            is_7mer_a1 = 1
        start = idx + 1

    return is_8mer, is_7mer_m8, is_7mer_a1, is_6mer


def compute_seed_type_features(df: pd.DataFrame) -> pd.DataFrame:
    gene_seq = df[GENE_SEQUENCE_COLUMN].fillna("")
    mirna_seq = df[MIRNA_SEQUENCE_COLUMN].fillna("")
    features = pd.DataFrame(index=df.index)

    results = [_classify_seed(m, g) for m, g in zip(mirna_seq, gene_seq)]
    is_8mer, is_7mer_m8, is_7mer_a1, is_6mer = zip(*results) if results else ([], [], [], [])

    features["seed_type__is_8mer"] = list(is_8mer)
    features["seed_type__is_7mer_m8"] = list(is_7mer_m8)
    features["seed_type__is_7mer_a1"] = list(is_7mer_a1)
    features["seed_type__is_6mer"] = list(is_6mer)

    # Ordinal encoding: 0=none < 1=6mer < 2=7mer-a1 < 3=7mer-m8 < 4=8mer
    best = []
    for a, b, c, d in zip(is_8mer, is_7mer_m8, is_7mer_a1, is_6mer):
        if a:
            best.append(4)
        elif b:
            best.append(3)
        elif c:
            best.append(2)
        elif d:
            best.append(1)
        else:
            best.append(0)
    features["seed_type__best_type"] = best

    features["seed_type__has_canonical"] = [
        int(a or b or c or d)
        for a, b, c, d in zip(is_8mer, is_7mer_m8, is_7mer_a1, is_6mer)
    ]

    return features.fillna(0)
