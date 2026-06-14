import pandas as pd

from src.config import GENE_SEQUENCE_COLUMN, MIRNA_SEQUENCE_COLUMN

COMPLEMENT = str.maketrans("ACGUT", "UGCAA")


def _reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def _max_consecutive_match(short: str, long: str) -> int:
    if not short or not long:
        return 0
    best = 0
    for i in range(len(long) - len(short) + 1):
        cur = 0
        for j in range(len(short)):
            if long[i + j] == short[j]:
                cur += 1
            else:
                best = max(best, cur)
                cur = 0
        best = max(best, cur)
    return best


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

    features["mirna_revcomp_in_gene"] = [
        int(_reverse_complement(m) in g) if len(m) > 0 and len(g) > 0 else 0
        for m, g in zip(mirna_seq, gene_seq)
    ]

    # gene_N_ratio — 注释掉，原始数据中基因序列不含未知碱基 N，该特征恒为 0
    # features["gene_N_ratio"] = [
    #     g.count("N") / len(g) if len(g) > 0 else 0.0 for g in gene_seq
    # ]

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

    features["seed_occurrence_count"] = [
        _count_substring(m[1:8], g) if len(m) >= 8 else 0
        for m, g in zip(mirna_seq, gene_seq)
    ]

    features["seed_max_consecutive"] = [
        _max_consecutive_match(m[1:8], g) if len(m) >= 8 else 0
        for m, g in zip(mirna_seq, gene_seq)
    ]

    features["mirna_max_consecutive"] = [
        _max_consecutive_match(m, g)
        for m, g in zip(mirna_seq, gene_seq)
    ]

    return features.fillna(0)
