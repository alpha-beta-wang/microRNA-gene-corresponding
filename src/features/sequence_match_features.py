import numpy as np
import pandas as pd

from src.config import GENE_SEQUENCE_COLUMN, MIRNA_SEQUENCE_COLUMN

COMPLEMENT = str.maketrans("ACGUT", "UGCAA")


def _reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def compute_match_features(df: pd.DataFrame) -> pd.DataFrame:
    gene_seq = df[GENE_SEQUENCE_COLUMN].fillna("")
    mirna_seq = df[MIRNA_SEQUENCE_COLUMN].fillna("")
    features = pd.DataFrame(index=df.index)

    features["seed_in_gene"] = [
        int(m[1:8] in g) if len(m) >= 8 and len(g) > 0 else 0
        for m, g in zip(mirna_seq, gene_seq)
    ]

    features["seed_revcomp_in_gene"] = [
        int(_reverse_complement(m[1:8]) in g) if len(m) >= 8 and len(g) > 0 else 0
        for m, g in zip(mirna_seq, gene_seq)
    ]

    features["mirna_in_gene"] = [
        int(m in g) if len(m) > 0 and len(g) > 0 else 0
        for m, g in zip(mirna_seq, gene_seq)
    ]

    mirrors = pd.DataFrame({
        MIRNA_SEQUENCE_COLUMN: mirna_seq,
        GENE_SEQUENCE_COLUMN: gene_seq,
    })
    features["mirna_revcomp_in_gene"] = mirrors.apply(
        lambda r: int(
            _reverse_complement(r[MIRNA_SEQUENCE_COLUMN]) in r[GENE_SEQUENCE_COLUMN]
        )
        if len(r[MIRNA_SEQUENCE_COLUMN]) > 0 and len(r[GENE_SEQUENCE_COLUMN]) > 0
        else 0,
        axis=1,
    )

    features["gene_N_ratio"] = [
        g.count("N") / len(g) if len(g) > 0 else 0.0 for g in gene_seq
    ]

    features["seed_gc"] = [
        (m[1:8].count("G") + m[1:8].count("C")) / 7 if len(m) >= 8 else 0.0
        for m in mirna_seq
    ]

    features["seed_A_count"] = [
        m[1:8].count("A") if len(m) >= 8 else 0 for m in mirna_seq
    ]
    features["seed_U_count"] = [
        m[1:8].count("U") if len(m) >= 8 else 0 for m in mirna_seq
    ]

    return features.fillna(0)
