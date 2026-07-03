"""k-mer TF-IDF 特征模块，从 miRNA 和 gene 序列提取字符 n-gram 表示。"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import GENE_SEQUENCE_COLUMN, MIRNA_SEQUENCE_COLUMN


def _normalize(seq: str) -> str:
    """Normalize a sequence to an uppercase string."""
    if not isinstance(seq, str):
        return ""
    return seq.upper().replace("U", "T")


def _fit_transform_char_kmer(
    train_seqs: pd.Series,
    test_seqs: pd.Series,
    k: int,
    prefix: str,
    min_df: int,
    max_features: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit character k-mer TF-IDF and transform train and test sequences."""
    train_text = train_seqs.fillna("").map(_normalize)
    test_text = test_seqs.fillna("").map(_normalize)

    vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(k, k),
        lowercase=False,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
        norm="l2",
    )
    train_mat = vec.fit_transform(train_text)
    test_mat = vec.transform(test_text)
    cols = [f"{prefix}{tok}" for tok in vec.get_feature_names_out()]

    train_df = pd.DataFrame(
        train_mat.toarray().astype(np.float32),
        index=train_seqs.index,
        columns=cols,
    )
    test_df = pd.DataFrame(
        test_mat.toarray().astype(np.float32),
        index=test_seqs.index,
        columns=cols,
    )
    return train_df, test_df


def compute_kmer_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    mirna_k: int = 3,
    gene_k: int = 3,
    gene_max_features: int = 256,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate miRNA and gene k-mer TF-IDF feature matrices."""
    train_blocks = []
    test_blocks = []

    mirna_train, mirna_test = _fit_transform_char_kmer(
        train[MIRNA_SEQUENCE_COLUMN],
        test[MIRNA_SEQUENCE_COLUMN],
        k=mirna_k,
        prefix=f"mirna_kmer{mirna_k}_",
        min_df=2,
        max_features=4 ** mirna_k,
    )
    train_blocks.append(mirna_train)
    test_blocks.append(mirna_test)

    gene_train, gene_test = _fit_transform_char_kmer(
        train[GENE_SEQUENCE_COLUMN],
        test[GENE_SEQUENCE_COLUMN],
        k=gene_k,
        prefix=f"gene_kmer{gene_k}_",
        min_df=5,
        max_features=gene_max_features,
    )
    train_blocks.append(gene_train)
    test_blocks.append(gene_test)

    return (
        pd.concat(train_blocks, axis=1),
        pd.concat(test_blocks, axis=1),
    )
