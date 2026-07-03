"""基础序列统计特征模块，计算长度、碱基比例和 GC 等指标。"""

import numpy as np
import pandas as pd

from src.config import GENE_SEQUENCE_COLUMN, MIRNA_SEQUENCE_COLUMN

NUCLEOTIDES = ["A", "C", "G", "T"]


def compute_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算 gene 与 miRNA 的基础序列统计特征。"""
    gene_seq = df[GENE_SEQUENCE_COLUMN].fillna("")
    mirna_seq = df[MIRNA_SEQUENCE_COLUMN].fillna("")

    features = pd.DataFrame(index=df.index)
    features["gene_length"] = gene_seq.str.len().astype(int)
    features["mirna_length"] = mirna_seq.str.len().astype(int)

    for nt in ["A", "C", "G", "T"]:
        features[f"gene_{nt}_ratio"] = gene_seq.str.count(nt) / gene_seq.str.len().replace(0, np.nan)  # 避免空序列分母为 0

    for nt in ["A", "C", "G", "U"]:
        features[f"mirna_{nt}_ratio"] = mirna_seq.str.count(nt) / mirna_seq.str.len().replace(0, np.nan)  # miRNA 使用 RNA 字母表

    features["gene_gc"] = features["gene_G_ratio"] + features["gene_C_ratio"]
    features["mirna_gc"] = features["mirna_G_ratio"] + features["mirna_C_ratio"]
    features["length_ratio"] = features["gene_length"] / features["mirna_length"].replace(0, np.nan)  # 长度比同样跳过 0 分母

    return features.fillna(0)  # 空值统一回填为 0
