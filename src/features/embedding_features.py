"""3-mer co-occurrence → PPMI → TruncatedSVD embedding features.

Non-neural "word embedding" for biological sequences using classical
distributional semantics: k-mer tokens that co-occur in similar contexts
get similar low-dimensional vectors via PPMI matrix factorization.

Only fitted on training data; test data is projected through the learned
embedding space to avoid leakage.
"""

from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from src.config import GENE_SEQUENCE_COLUMN, MIRNA_SEQUENCE_COLUMN


def _tokenize(seq: str, k: int) -> list[str]:
    """Sliding-window k-mer tokens from *seq*."""
    return [seq[i : i + k] for i in range(len(seq) - k + 1)]


def _best_seed_window(mirna: str, gene: str, window_margin: int = 30) -> str | None:
    """Copy of alignment_features._best_seed_window — no biopython import needed."""
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


def _build_ppmi_embedding(
    tokens_list: list[list[str]],
    dim: int,
    context_radius: int,
    min_count: int,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, int], list[str]]:
    """Build PPMI matrix from token co-occurrence, then reduce with TruncatedSVD."""
    token_counts: Counter[str] = Counter()
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)

    for tokens in tokens_list:
        for i, t in enumerate(tokens):
            token_counts[t] += 1
            start = max(0, i - context_radius)
            end = min(len(tokens), i + context_radius + 1)
            for j in range(start, end):
                if i != j:
                    pair = (t, tokens[j])
                    pair_counts[pair] += 1

    # Filter vocabulary by min_count
    vocab = sorted(t for t, c in token_counts.items() if c >= min_count)
    token_to_idx = {t: i for i, t in enumerate(vocab)}
    n_vocab = len(vocab)

    if n_vocab == 0:
        return np.zeros((1, dim)), {}, []

    total_pairs = sum(pair_counts.values())
    total_tokens = sum(token_counts[t] for t in vocab)

    # Build PPMI matrix
    ppmi = np.zeros((n_vocab, n_vocab), dtype=np.float64)
    for (t1, t2), count in pair_counts.items():
        i = token_to_idx.get(t1)
        j = token_to_idx.get(t2)
        if i is not None and j is not None:
            pij = count / total_pairs
            pi = token_counts[t1] / total_tokens
            pj = token_counts[t2] / total_tokens
            pmi = np.log2(pij / (pi * pj) + 1e-10)
            ppmi[i, j] = max(0.0, pmi)

    # Reduce dimensions
    if n_vocab <= 1:
        embedding = np.full((max(1, n_vocab), dim), 1e-10, dtype=np.float64)
    else:
        n_components = min(dim, n_vocab - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=seed)
        emb = svd.fit_transform(ppmi)
        embedding = np.zeros((n_vocab, dim), dtype=np.float64)
        embedding[:, :n_components] = emb

    return embedding, token_to_idx, vocab


def _aggregate(tokens: list[str], embedding: np.ndarray, token_to_idx: dict[str, int], dim: int) -> np.ndarray:
    """Mean-pool k-mer embedding vectors for a list of tokens."""
    vectors = [embedding[token_to_idx[t]] for t in tokens if t in token_to_idx]
    return np.mean(vectors, axis=0) if vectors else np.zeros(dim)


class EmbeddingFeaturizer:
    """Stateful callable: first call fits on train, subsequent calls transform.

    Register via ``_make_embedding`` factory so ``build_features`` resolves
    one instance per pipeline run, naturally achieving fit(train)/transform(test).
    """

    def __init__(self) -> None:
        self._fitted = False
        self._k = 3
        self._dim = 12
        self._context_radius = 2
        self._min_count = 2

        self._embedding: np.ndarray = np.zeros((1, 12))
        self._token_to_idx: dict[str, int] = {}
        self._vocab: list[str] = []

    def __call__(
        self,
        df: pd.DataFrame,
        k: int | None = None,
        dim: int | None = None,
        context_radius: int | None = None,
        min_count: int | None = None,
    ) -> pd.DataFrame:
        if not self._fitted:
            if k is not None:
                self._k = k
            if dim is not None:
                self._dim = dim
            if context_radius is not None:
                self._context_radius = context_radius
            if min_count is not None:
                self._min_count = min_count
            self._fit(df)
        return self._transform(df)

    # ── fitting ───────────────────────────────────────────────

    def _fit(self, df: pd.DataFrame) -> None:
        gene_seq = df[GENE_SEQUENCE_COLUMN].fillna("")
        mirna_seq = df[MIRNA_SEQUENCE_COLUMN].fillna("")

        tokens_list: list[list[str]] = []
        for m, g in zip(mirna_seq, gene_seq):
            toks: list[str] = []
            # miRNA full sequence
            toks.extend(_tokenize(m, self._k))
            # miRNA seed (positions 1-7 relative to 0-indexed miRNA)
            if len(m) >= 8:
                toks.extend(_tokenize(m[:8], self._k))
            # gene window around best seed match
            window = _best_seed_window(m, g)
            if window:
                toks.extend(_tokenize(window, self._k))
            tokens_list.append(toks)

        self._embedding, self._token_to_idx, self._vocab = _build_ppmi_embedding(
            tokens_list,
            dim=self._dim,
            context_radius=self._context_radius,
            min_count=self._min_count,
        )
        self._fitted = True

    # ── transforming ──────────────────────────────────────────

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        gene_seq = df[GENE_SEQUENCE_COLUMN].fillna("")
        mirna_seq = df[MIRNA_SEQUENCE_COLUMN].fillna("")

        dim = self._dim
        seed_embs: list[np.ndarray] = []
        window_embs: list[np.ndarray] = []

        for m, g in zip(mirna_seq, gene_seq):
            seed_tokens = _tokenize(m[:8], self._k) if len(m) >= 8 else []
            window = _best_seed_window(m, g)
            window_tokens = _tokenize(window, self._k) if window else []

            seed_embs.append(_aggregate(seed_tokens, self._embedding, self._token_to_idx, dim))
            window_embs.append(_aggregate(window_tokens, self._embedding, self._token_to_idx, dim))

        seed_arr = np.array(seed_embs)
        window_arr = np.array(window_embs)

        features = pd.DataFrame(index=df.index)

        for d in range(dim):
            features[f"embed__mirna_seed_{d}"] = seed_arr[:, d]
            features[f"embed__gene_window_{d}"] = window_arr[:, d]
            features[f"embed__absdiff_{d}"] = np.abs(seed_arr[:, d] - window_arr[:, d])

        # Pairwise similarity
        dot = np.sum(seed_arr * window_arr, axis=1)
        seed_norm = np.linalg.norm(seed_arr, axis=1)
        window_norm = np.linalg.norm(window_arr, axis=1)
        denom = seed_norm * window_norm
        features["embed__cosine"] = np.where(denom > 0, dot / denom, 0.0)
        features["embed__dot"] = dot
        features["embed__l2"] = np.sqrt(np.sum((seed_arr - window_arr) ** 2, axis=1))

        return features.fillna(0.0)
