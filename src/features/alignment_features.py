"""序列比对特征模块，基于局部或全局比对度量 miRNA-gene 匹配质量。"""

import pandas as pd
from Bio.Align import PairwiseAligner

from src.config import GENE_SEQUENCE_COLUMN, MIRNA_SEQUENCE_COLUMN

_COMPLEMENT = str.maketrans("ACGUT", "UGCAA")


def _reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def _aligner_local() -> PairwiseAligner:
    a = PairwiseAligner()
    a.mode = "local"
    a.match_score = 2
    a.mismatch_score = -1
    a.open_gap_score = -2
    a.extend_gap_score = -1
    return a


def _aligner_global() -> PairwiseAligner:
    a = PairwiseAligner()
    a.mode = "global"
    a.match_score = 2
    a.mismatch_score = -1
    a.open_gap_score = -2
    a.extend_gap_score = -1
    return a


def _best_seed_window(mirna: str, gene: str, window_margin: int = 30) -> str | None:
    """Find the gene window where miRNA seed (positions 1-7) has the longest consecutive match."""
    if len(mirna) < 8 or len(gene) < 8:
        return None
    seed = mirna[1:8]
    best_start = 0
    best_consec = 0
    for i in range(len(gene) - len(seed) + 1):
        cur = 0
        for j in range(len(seed)):
            if gene[i + j] == seed[j]:
                cur += 1
            else:
                if cur > best_consec:
                    best_consec = cur
                    best_start = i
                cur = 0
        if cur > best_consec:
            best_consec = cur
            best_start = i
    win_start = max(0, best_start - window_margin)
    win_end = min(len(gene), best_start + len(seed) + window_margin)
    return gene[win_start:win_end]


def compute_alignment_features(df: pd.DataFrame) -> pd.DataFrame:
    gene_seq = df[GENE_SEQUENCE_COLUMN].fillna("")
    mirna_seq = df[MIRNA_SEQUENCE_COLUMN].fillna("")
    features = pd.DataFrame(index=df.index)

    local_aligner = _aligner_local()
    global_aligner = _aligner_global()

    local_scores = []
    local_scores_norm = []
    local_coverages = []
    global_scores = []
    global_scores_norm = []

    for m, g in zip(mirna_seq, gene_seq):
        if len(m) == 0 or len(g) == 0:
            local_scores.append(0.0)
            local_scores_norm.append(0.0)
            local_coverages.append(0.0)
            global_scores.append(0.0)
            global_scores_norm.append(0.0)
            continue

        # Local alignment (Smith-Waterman)
        local = local_aligner.align(m, g)
        best_local = local[0] if local else None
        if best_local is not None:
            ls = float(best_local.score)
            aligned = best_local.aligned
            # total aligned bases on the miRNA (target) side
            target_aligned = sum(end - start for start, end in aligned[0]) if len(aligned) > 0 else 0
            lc = target_aligned / len(m)
        else:
            ls = 0.0
            lc = 0.0
        local_scores.append(ls)
        local_scores_norm.append(ls / len(m) if len(m) > 0 else 0.0)
        local_coverages.append(lc)

        # Global alignment on best seed window
        window = _best_seed_window(m, g)
        if window and len(window) > 0 and len(m) > 0:
            g_aln = global_aligner.align(m, window)
            best_global = g_aln[0] if g_aln else None
            gs = float(best_global.score) if best_global is not None else 0.0
        else:
            gs = 0.0
        global_scores.append(gs)
        global_scores_norm.append(gs / len(m) if len(m) > 0 else 0.0)

    features["align__local_score"] = local_scores
    features["align__local_score_norm"] = local_scores_norm
    features["align__local_coverage"] = local_coverages
    features["align__global_score"] = global_scores
    features["align__global_score_norm"] = global_scores_norm

    # Reverse complement alignment: miRNA_rc vs gene local alignment
    rc_local_scores = []
    rc_local_scores_norm = []
    for m, g in zip(mirna_seq, gene_seq):
        if len(m) == 0 or len(g) == 0:
            rc_local_scores.append(0.0)
            rc_local_scores_norm.append(0.0)
        else:
            m_rc = _reverse_complement(m)
            rc_aln = local_aligner.align(m_rc, g)
            best_rc = rc_aln[0] if rc_aln else None
            rcs = float(best_rc.score) if best_rc is not None else 0.0
            rc_local_scores.append(rcs)
            rc_local_scores_norm.append(rcs / len(m))
    features["align__rc_local_score"] = rc_local_scores
    features["align__rc_local_score_norm"] = rc_local_scores_norm

    return features
"""序列比对特征模块，基于局部或全局比对度量 miRNA-gene 匹配质量。"""
