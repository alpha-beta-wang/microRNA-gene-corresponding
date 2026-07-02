"""二核苷酸组成特征模块，比较 miRNA 与 gene 的二联体比例差异。"""

from itertools import product

import pandas as pd

from src.config import GENE_SEQUENCE_COLUMN, MIRNA_SEQUENCE_COLUMN

DINUCS = ["".join(p) for p in product("ACGT", repeat=2)]


def _normalize(seq: str) -> str:
    return seq.upper().replace("U", "T") if isinstance(seq, str) else ""


def _ratio(seq: str, token: str) -> float:
    seq = _normalize(seq)
    denom = max(len(seq) - 1, 1)
    return seq.count(token) / denom


def compute_dinucleotide_features(df: pd.DataFrame) -> pd.DataFrame:
    gene_seq = df[GENE_SEQUENCE_COLUMN].fillna("")
    mirna_seq = df[MIRNA_SEQUENCE_COLUMN].fillna("")
    features = pd.DataFrame(index=df.index)

    for token in DINUCS:
        gene_col = f"gene_dinuc_{token}"
        mirna_col = f"mirna_dinuc_{token}"
        features[gene_col] = [_ratio(seq, token) for seq in gene_seq]
        features[mirna_col] = [_ratio(seq, token) for seq in mirna_seq]
        features[f"dinuc_absdiff_{token}"] = (features[gene_col] - features[mirna_col]).abs()

    return features.fillna(0)
"""二核苷酸组成特征模块，比较 miRNA 与 gene 的二联体比例差异。"""
