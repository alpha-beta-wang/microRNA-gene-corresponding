"""特征构建注册表模块，按名称解析并组合多个特征块。"""

from typing import Any, Callable

import pandas as pd

from src.features.basic_features import compute_sequence_features
from src.features.sequence_match_features import compute_match_features

_LAZY_BLOCKS: dict[str, Callable[[], Callable[[pd.DataFrame], pd.DataFrame]]] = {}


def _register(name: str, factory: Callable[[], Callable[[pd.DataFrame], pd.DataFrame]]) -> None:
    """注册一个按需加载的特征块工厂。"""
    _LAZY_BLOCKS[name] = factory


def _resolve(name: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """按名称解析特征块实现。"""
    if name not in _LAZY_BLOCKS:
        raise KeyError(f"unknown feature block '{name}'; available: {list(_LAZY_BLOCKS.keys())}")
    return _LAZY_BLOCKS[name]()  # 仅在真正使用时导入对应模块


_register("basic", lambda: compute_sequence_features)
_register("match", lambda: compute_match_features)


def _make_kmer():
    """延迟导入 k-mer 特征实现。"""
    from src.features.kmer_features import compute_kmer_features
    return compute_kmer_features


_register("kmer", _make_kmer)


def _make_alignment():
    """延迟导入比对特征实现。"""
    from src.features.alignment_features import compute_alignment_features
    return compute_alignment_features


_register("alignment", _make_alignment)


def _make_rna_energy():
    """延迟导入 RNA 能量特征实现。"""
    from src.features.rna_energy_features import compute_rna_energy_features
    return compute_rna_energy_features


_register("rna_energy", _make_rna_energy)


def _make_position():
    """延迟导入位置特征实现。"""
    from src.features.position_features import compute_position_features
    return compute_position_features


_register("position", _make_position)


def _make_seed_type():
    """延迟导入 seed 类型特征实现。"""
    from src.features.seed_type_features import compute_seed_type_features
    return compute_seed_type_features


_register("seed_type", _make_seed_type)


def _make_embedding():
    """延迟创建 embedding 特征器。"""
    from src.features.embedding_features import EmbeddingFeaturizer
    return EmbeddingFeaturizer()


_register("embedding", _make_embedding)


def _make_rna_accessibility():
    """延迟导入 RNA 可及性特征实现。"""
    from functools import partial
    from src.features.rna_energy_features import compute_rna_energy_features
    return partial(compute_rna_energy_features, compute_accessibility=True)


_register("rna_accessibility", _make_rna_accessibility)


def build_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    blocks: list[str],
    block_kwargs: dict[str, dict[str, Any]] | None = None,
    missing_external_policy: str = "error",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """根据启用的特征块拼装训练集与测试集特征表。"""
    if not blocks:
        raise ValueError("at least one feature block required")
    if block_kwargs is None:
        block_kwargs = {}

    train_parts = []
    test_parts = []
    for name in blocks:
        try:
            fn = _resolve(name)
        except (ImportError, ModuleNotFoundError) as e:
            if missing_external_policy == "skip":
                print(f"  [{name}] SKIPPED — {e}")
                continue
            raise
        kwargs = block_kwargs.get(name, {})
        try:
            tr = fn(train_df, **kwargs) if kwargs else fn(train_df)
            te = fn(test_df, **kwargs) if kwargs else fn(test_df)
        except RuntimeError as e:
            if missing_external_policy == "skip":
                print(f"  [{name}] SKIPPED — {e}")
                continue
            raise
        print(f"  [{name}] train_cols={len(tr.columns)} test_cols={len(te.columns)}")
        train_parts.append(tr)
        test_parts.append(te)

    if not train_parts:
        raise RuntimeError("no feature blocks were successfully loaded")

    train_features = pd.concat(train_parts, axis=1)
    test_features = pd.concat(test_parts, axis=1)

    train_features = train_features.loc[:, ~train_features.columns.duplicated()]
    test_features = test_features.loc[:, ~test_features.columns.duplicated()]
    test_features = test_features[train_features.columns]  # 强制测试列顺序与训练列完全一致

    print(f"  total feature columns: {len(train_features.columns)}")
    return train_features, test_features