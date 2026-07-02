"""实体频次特征模块，统计训练集中 miRNA 和 gene 出现次数。"""

import pandas as pd

from src.config import GENE_COLUMN, MIRNA_COLUMN


def compute_entity_count_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat(
        [
            train[[GENE_COLUMN, MIRNA_COLUMN]].assign(_is_train=1),
            test[[GENE_COLUMN, MIRNA_COLUMN]].assign(_is_train=0),
        ],
        axis=0,
        ignore_index=True,
    )
    gene_all = combined[GENE_COLUMN].value_counts()
    mirna_all = combined[MIRNA_COLUMN].value_counts()
    gene_train = train[GENE_COLUMN].value_counts()
    mirna_train = train[MIRNA_COLUMN].value_counts()
    gene_test = test[GENE_COLUMN].value_counts()
    mirna_test = test[MIRNA_COLUMN].value_counts()

    def build(frame: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=frame.index)
        features["gene_all_count"] = frame[GENE_COLUMN].map(gene_all).fillna(0).astype(int)
        features["mirna_all_count"] = frame[MIRNA_COLUMN].map(mirna_all).fillna(0).astype(int)
        features["gene_train_count"] = frame[GENE_COLUMN].map(gene_train).fillna(0).astype(int)
        features["mirna_train_count"] = frame[MIRNA_COLUMN].map(mirna_train).fillna(0).astype(int)
        features["gene_test_count"] = frame[GENE_COLUMN].map(gene_test).fillna(0).astype(int)
        features["mirna_test_count"] = frame[MIRNA_COLUMN].map(mirna_test).fillna(0).astype(int)
        features["gene_mirna_count_product"] = features["gene_all_count"] * features["mirna_all_count"]
        return features

    return build(train), build(test)
"""实体频次特征模块，统计训练集中 miRNA 和 gene 出现次数。"""
