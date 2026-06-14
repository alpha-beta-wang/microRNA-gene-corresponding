from typing import Callable

import pandas as pd

from src.features.basic_features import compute_sequence_features
from src.features.sequence_match_features import compute_match_features

_FEATURE_BLOCKS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "basic": compute_sequence_features,
    "match": compute_match_features,
}


def register_block(name: str, fn: Callable[[pd.DataFrame], pd.DataFrame]) -> None:
    _FEATURE_BLOCKS[name] = fn


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame, blocks: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build train/test feature tables from a list of enabled block names."""
    if not blocks:
        raise ValueError("at least one feature block required")

    train_parts = []
    test_parts = []
    for name in blocks:
        fn = _FEATURE_BLOCKS.get(name)
        if fn is None:
            raise KeyError(f"unknown feature block '{name}'; available: {list(_FEATURE_BLOCKS.keys())}")
        tr = fn(train_df)
        te = fn(test_df)
        print(f"  [{name}] train_cols={len(tr.columns)} test_cols={len(te.columns)}")
        train_parts.append(tr)
        test_parts.append(te)

    train_features = pd.concat(train_parts, axis=1)
    test_features = pd.concat(test_parts, axis=1)

    train_features = train_features.loc[:, ~train_features.columns.duplicated()]
    test_features = test_features.loc[:, ~test_features.columns.duplicated()]
    test_features = test_features[train_features.columns]

    print(f"  total feature columns: {len(train_features.columns)}")
    return train_features, test_features
