from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import GENE_SEQUENCE_COLUMN, MIRNA_SEQUENCE_COLUMN
from src.features.sequence_match_features import _reverse_complement


def _norm_gene(seq: str) -> str:
    return seq.upper().replace("U", "T") if isinstance(seq, str) else ""


def _norm_mirna(seq: str) -> str:
    return seq.upper().replace("T", "U") if isinstance(seq, str) else ""


def _find_all(haystack: str, needle: str) -> list[int]:
    if not haystack or not needle:
        return []
    out = []
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx < 0:
            return out
        out.append(idx)
        start = idx + 1


def _au_ratio(seq: str) -> float:
    if not seq:
        return 0.0
    return (seq.count("A") + seq.count("T") + seq.count("U")) / len(seq)


def _wc_pair(m_base: str, g_base: str) -> bool:
    return (m_base, g_base) in {
        ("A", "T"),
        ("A", "U"),
        ("U", "A"),
        ("T", "A"),
        ("G", "C"),
        ("C", "G"),
    }


def _wobble_pair(m_base: str, g_base: str) -> bool:
    return (m_base, g_base) in {("G", "T"), ("G", "U"), ("U", "G"), ("T", "G")}


def _pairing_score(mirna_region: str, target_region: str) -> tuple[int, int, int]:
    wc = 0
    wobble = 0
    longest = 0
    cur = 0
    for m_base, g_base in zip(mirna_region, target_region):
        if _wc_pair(m_base, g_base):
            wc += 1
            cur += 1
            longest = max(longest, cur)
        elif _wobble_pair(m_base, g_base):
            wobble += 1
            cur = 0
        else:
            cur = 0
    return wc, wobble, longest


def _site_metrics(gene: str, mirna: str) -> dict[str, float]:
    gene = _norm_gene(gene)
    mirna = _norm_mirna(mirna)
    if len(gene) == 0 or len(mirna) < 8:
        return {
            "ts_8mer_count": 0,
            "ts_7mer_m8_count": 0,
            "ts_7mer_a1_count": 0,
            "ts_6mer_count": 0,
            "ts_any_canonical_count": 0,
            "ts_best_site_score": 0,
            "ts_best_local_au": 0,
            "ts_mean_local_au": 0,
            "ts_best_relative_pos": 0,
            "ts_nearest_end_distance": 0,
            "ts_best_3p_wc": 0,
            "ts_best_3p_wobble": 0,
            "ts_best_3p_longest_wc": 0,
            "ts_weighted_site_sum": 0,
        }

    seed_2_8 = mirna[1:8]
    seed_2_7 = mirna[1:7]
    rc_2_8 = _reverse_complement(seed_2_8)
    rc_2_7 = _reverse_complement(seed_2_7)

    sites: list[tuple[str, int, int, int]] = []
    for pos in _find_all(gene, rc_2_8):
        a1 = gene[pos - 1] == "A" if pos > 0 else False
        sites.append(("8mer" if a1 else "7mer_m8", pos, 8 if a1 else 7, 4 if a1 else 3))
    for pos in _find_all(gene, rc_2_7):
        a1 = gene[pos - 1] == "A" if pos > 0 else False
        if a1:
            sites.append(("7mer_a1", pos, 7, 2))
        else:
            sites.append(("6mer", pos, 6, 1))

    counts = {
        "ts_8mer_count": sum(1 for s in sites if s[0] == "8mer"),
        "ts_7mer_m8_count": sum(1 for s in sites if s[0] == "7mer_m8"),
        "ts_7mer_a1_count": sum(1 for s in sites if s[0] == "7mer_a1"),
        "ts_6mer_count": sum(1 for s in sites if s[0] == "6mer"),
    }
    if not sites:
        return {
            **counts,
            "ts_any_canonical_count": 0,
            "ts_best_site_score": 0,
            "ts_best_local_au": 0,
            "ts_mean_local_au": 0,
            "ts_best_relative_pos": 0,
            "ts_nearest_end_distance": 0,
            "ts_best_3p_wc": 0,
            "ts_best_3p_wobble": 0,
            "ts_best_3p_longest_wc": 0,
            "ts_weighted_site_sum": 0,
        }

    local_aus = []
    rel_positions = []
    end_distances = []
    three_prime_scores = []
    weighted_sum = 0
    gene_len = len(gene)
    mirna_3p = mirna[12:17] if len(mirna) >= 17 else mirna[12:]

    for _site_type, pos, site_len, score in sites:
        start = max(0, pos - 30)
        end = min(gene_len, pos + site_len + 30)
        local_aus.append(_au_ratio(gene[start:end]))
        rel_positions.append((pos + site_len / 2) / gene_len)
        end_distances.append(min(pos, max(0, gene_len - pos - site_len)) / gene_len)
        downstream = gene[pos + site_len : pos + site_len + len(mirna_3p)]
        three_prime_scores.append(_pairing_score(mirna_3p, downstream))
        weighted_sum += score

    best_idx = int(np.argmax([site[3] for site in sites]))
    best_3p = three_prime_scores[best_idx] if three_prime_scores else (0, 0, 0)
    return {
        **counts,
        "ts_any_canonical_count": len(sites),
        "ts_best_site_score": max(site[3] for site in sites),
        "ts_best_local_au": max(local_aus),
        "ts_mean_local_au": float(np.mean(local_aus)),
        "ts_best_relative_pos": rel_positions[best_idx],
        "ts_nearest_end_distance": end_distances[best_idx],
        "ts_best_3p_wc": best_3p[0],
        "ts_best_3p_wobble": best_3p[1],
        "ts_best_3p_longest_wc": best_3p[2],
        "ts_weighted_site_sum": weighted_sum,
    }


def compute_targetscan_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        _site_metrics(gene, mirna)
        for gene, mirna in zip(df[GENE_SEQUENCE_COLUMN].fillna(""), df[MIRNA_SEQUENCE_COLUMN].fillna(""))
    ]
    return pd.DataFrame(rows, index=df.index).fillna(0)
