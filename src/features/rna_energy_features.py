"""RNA 能量特征模块，调用 ViennaRNA 估计折叠和双链能量。"""

import pandas as pd
import RNA

from src.config import GENE_SEQUENCE_COLUMN, MIRNA_SEQUENCE_COLUMN


def _detect_rna_backend() -> str:
    """Detect available RNA backend. Returns 'viennarna' if RNA module works."""
    try:
        RNA.fold("A")
        RNA.duplexfold("A", "U")
        return "viennarna"
    except Exception:
        pass
    raise RuntimeError(
        "No RNA backend available. Install ViennaRNA:\n"
        "  pip install viennarna\n"
        "Or install command-line tools:\n"
        "  conda install -c bioconda viennarna\n"
        "Then re-run with --feature-blocks including rna_energy."
    )


def _mirna_mfe(mirna: str) -> float:
    """MFE of miRNA folding alone (T->U already handled, miRNA uses U)."""
    if len(mirna) == 0:
        return 0.0
    _, mfe = RNA.fold(mirna)
    return mfe


def _duplex_energy(mirna: str, target_window: str) -> float:
    """Duplex folding energy between miRNA and a target window."""
    if len(mirna) == 0 or len(target_window) == 0:
        return 0.0
    dup = RNA.duplexfold(mirna, target_window)
    return dup.energy


def _seed_match_windows(
    mirna: str, gene: str, window_size: int = 80, max_windows: int = 5
) -> list[str]:
    """Extract gene windows around strong seed matches (T->U converted)."""
    if len(mirna) < 8 or len(gene) < 8:
        return []
    seed = mirna[1:8]
    gene_u = gene.replace("T", "U")
    windows = []
    for i in range(len(gene_u) - len(seed) + 1):
        # count consecutive matches from start of seed
        match_count = 0
        for j in range(len(seed)):
            if gene_u[i + j] == seed[j]:
                match_count += 1
            else:
                break
        if match_count >= 4:
            win_start = max(0, i - window_size // 2)
            win_end = min(len(gene_u), i + len(seed) + window_size // 2)
            windows.append(gene_u[win_start:win_end])
        if len(windows) >= max_windows:
            break
    return windows


def compute_rna_energy_features(
    df: pd.DataFrame,
    window_size: int = 80,
    max_windows: int = 5,
    compute_accessibility: bool = False,
) -> pd.DataFrame:
    """Compute simplified binding energy and accessibility features."""
    _detect_rna_backend()

    gene_seq = df[GENE_SEQUENCE_COLUMN].fillna("")
    mirna_seq = df[MIRNA_SEQUENCE_COLUMN].fillna("")
    features = pd.DataFrame(index=df.index)

    mirna_mfes = []
    best_duplex_energies = []
    best_duplex_energies_norm = []
    candidate_counts = []
    duplex_per_window = []
    target_window_avg_mfes = []
    target_window_min_mfes = []
    target_window_max_mfes = []

    for m, g in zip(mirna_seq, gene_seq):
        mfe = _mirna_mfe(m)
        mirna_mfes.append(mfe)

        windows = _seed_match_windows(m, g, window_size, max_windows)
        candidate_counts.append(len(windows))

        if windows:
            d_energies = [_duplex_energy(m, w) for w in windows]
            best_de = min(d_energies)  # more negative = stronger binding
            avg_de = sum(d_energies) / len(d_energies)
            if compute_accessibility:
                w_mfes = [_mirna_mfe(w) for w in windows]
                target_window_avg_mfes.append(sum(w_mfes) / len(w_mfes))
                target_window_min_mfes.append(min(w_mfes))
                target_window_max_mfes.append(max(w_mfes))
        else:
            best_de = 0.0
            avg_de = 0.0
            if compute_accessibility:
                target_window_avg_mfes.append(0.0)
                target_window_min_mfes.append(0.0)
                target_window_max_mfes.append(0.0)

        best_duplex_energies.append(best_de)
        best_duplex_energies_norm.append(best_de / len(m) if len(m) > 0 else 0.0)
        duplex_per_window.append(avg_de)

    features["rna__mirna_mfe"] = mirna_mfes
    features["rna__mirna_mfe_norm"] = [
        mfe / len(seq) if len(seq) > 0 else 0.0
        for mfe, seq in zip(mirna_mfes, mirna_seq)
    ]
    features["rna__duplex_mfe"] = best_duplex_energies
    features["rna__duplex_mfe_norm"] = best_duplex_energies_norm
    features["rna__binding_energy"] = [
        d - m if d < 0 else 0.0
        for d, m in zip(best_duplex_energies, mirna_mfes)
    ]
    features["rna__candidate_count"] = candidate_counts
    features["rna__duplex_per_window"] = duplex_per_window

    if compute_accessibility:
        features["rna__target_window_avg_mfe"] = target_window_avg_mfes
        features["rna__target_window_min_mfe"] = target_window_min_mfes
        features["rna__target_window_max_mfe"] = target_window_max_mfes
        features["rna__accessibility_energy"] = [
            d - w if d < 0 and w < 0 else 0.0
            for d, w in zip(best_duplex_energies, target_window_min_mfes)
        ]

    return features
