"""k-mer 交互特征模块，统计 miRNA 与 gene 反向互补 k-mer 的相互作用。"""

from itertools import product

import numpy as np
import pandas as pd

from src.config import GENE_SEQUENCE_COLUMN, MIRNA_SEQUENCE_COLUMN

_DNA = "ACGT"
_RC = str.maketrans("ACGT", "TGCA")

ALL_3MERS = ["".join(t) for t in product(_DNA, repeat=3)]  # 64 fixed columns
_RC_MAP = {k: k.translate(_RC)[::-1] for k in ALL_3MERS}


def _normalize(seq: str) -> str:
    if not isinstance(seq, str):
        return ""
    return seq.upper().replace("U", "T")


def _kmer_counts(seq: str, k: int = 3) -> dict:
    counts = {}
    for i in range(len(seq) - k + 1):
        km = seq[i : i + k]
        counts[km] = counts.get(km, 0) + 1
    return counts


def _interaction_vector(mirna: str, gene: str) -> np.ndarray:
    m = _normalize(mirna)
    g = _normalize(gene)
    if not m or not g:
        return np.zeros(len(ALL_3MERS), dtype=np.float32)

    mirna_counts = _kmer_counts(m)
    gene_counts = _kmer_counts(g)
    gene_len_kb = max(len(g), 1) / 1000.0

    vec = np.zeros(len(ALL_3MERS), dtype=np.float32)
    for idx, km in enumerate(ALL_3MERS):
        mc = mirna_counts.get(km, 0)
        if mc == 0:
            continue
        rc = _RC_MAP[km]
        gc = gene_counts.get(rc, 0)
        # product of miRNA k-mer count × gene rc occurrence density
        vec[idx] = mc * (gc / gene_len_kb)
    return vec


def compute_kmer_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    mirna_col = df[MIRNA_SEQUENCE_COLUMN].fillna("")
    gene_col = df[GENE_SEQUENCE_COLUMN].fillna("")

    cols = [f"kmi3_{km}" for km in ALL_3MERS]
    mat = np.vstack([
        _interaction_vector(m, g)
        for m, g in zip(mirna_col, gene_col)
    ])
    return pd.DataFrame(mat, index=df.index, columns=cols)
"""k-mer 交互特征模块，统计 miRNA 与 gene 反向互补 k-mer 的相互作用。"""
