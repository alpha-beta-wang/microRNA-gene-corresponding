import itertools

import pandas as pd

from src.config import GENE_SEQUENCE_COLUMN, MIRNA_SEQUENCE_COLUMN

GENE_ALPHABET = "ACGT"
MIRNA_ALPHABET = "ACGU"


def _all_kmers(alphabet: str, k: int) -> list[str]:
    return ["".join(p) for p in itertools.product(alphabet, repeat=k)]


def _kmer_freqs(seq: str, kmers: list[str]) -> dict[str, float]:
    n = len(seq)
    counts = {kmer: 0 for kmer in kmers}
    if n == 0:
        return counts
    for i in range(n - len(kmers[0]) + 1):
        sub = seq[i : i + len(kmers[0])]
        if sub in counts:
            counts[sub] += 1
    total = sum(counts.values())
    return {k: v / total if total > 0 else 0.0 for k, v in counts.items()}


def compute_kmer_features(
    df: pd.DataFrame, k_sizes: tuple[int, ...] = (2, 3)
) -> pd.DataFrame:
    gene_seq = df[GENE_SEQUENCE_COLUMN].fillna("")
    mirna_seq = df[MIRNA_SEQUENCE_COLUMN].fillna("")
    features = pd.DataFrame(index=df.index)

    for k in k_sizes:
        gene_kmers = _all_kmers(GENE_ALPHABET, k)
        mirna_kmers = _all_kmers(MIRNA_ALPHABET, k)

        gene_rows = {}
        mirna_rows = {}
        for kmer in gene_kmers:
            gene_rows[f"kmer__gene_k{k}_{kmer}"] = 0.0
        for kmer in mirna_kmers:
            mirna_rows[f"kmer__mirna_k{k}_{kmer}"] = 0.0

        gene_data = []
        mirna_data = []
        for gs, ms in zip(gene_seq, mirna_seq):
            gf = _kmer_freqs(gs, gene_kmers)
            mf = _kmer_freqs(ms, mirna_kmers)
            gene_data.append({f"kmer__gene_k{k}_{km}": gf[km] for km in gene_kmers})
            mirna_data.append({f"kmer__mirna_k{k}_{km}": mf[km] for km in mirna_kmers})

        gene_df = pd.DataFrame(gene_data, index=df.index)
        mirna_df = pd.DataFrame(mirna_data, index=df.index)
        features = pd.concat([features, gene_df, mirna_df], axis=1)

    return features
