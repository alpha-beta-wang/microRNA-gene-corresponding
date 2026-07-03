"""预测结果融合工具，用于合并多个提交或概率文件。"""

from __future__ import annotations

import argparse
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.config import GENE_COLUMN, MIRNA_COLUMN, OOF_DIR, SUBMISSION_DIR, SUBMIT_EXAMPLE_FILE, TARGET_COLUMN
from src.data.load_data import build_dataset_bundle
from src.experiment_runner import DEFAULT_LARK_SHEET_ID, DEFAULT_LARK_TOKEN, _next_lark_row, _write_lark_rows


DEFAULT_INPUTS = [
    ("kmer", OOF_DIR / "oof_kmer_baseline_config.csv", OOF_DIR / "test_proba_kmer_baseline_config.csv"),
    ("seedfeat", OOF_DIR / "oof_seed_variant_entity_dinuc.csv", OOF_DIR / "test_proba_seed_variant_entity_dinuc.csv"),
    ("seedfeat_extra", OOF_DIR / "oof_seed_variant_entity_dinuc_extra.csv", OOF_DIR / "test_proba_seed_variant_entity_dinuc_extra.csv"),
    ("targetscan", OOF_DIR / "oof_targetscan_kmer_20260623_214326.csv", OOF_DIR / "test_proba_targetscan_kmer_20260623_214326.csv"),
    ("targetscan_reg", OOF_DIR / "oof_targetscan_regularized_20260623_214426.csv", OOF_DIR / "test_proba_targetscan_regularized_20260623_214426.csv"),
]


def _read_series(path) -> np.ndarray:
    """Load a single-column prediction file as a probability array."""
    frame = pd.read_csv(path)
    return frame.iloc[:, -1].to_numpy(dtype=float)


def _load_inputs():
    """Load labels, submission template, metadata, and available prediction files."""
    loaded = []
    for name, oof_path, test_path in DEFAULT_INPUTS:
        if oof_path.exists() and test_path.exists():
            loaded.append((name, _read_series(oof_path), _read_series(test_path)))
    if len(loaded) < 2:
        raise RuntimeError("Need at least two OOF/test probability pairs to blend.")
    return loaded


def _weight_grid(n: int, step: float = 0.1):
    """Generate normalized weight combinations at the requested step size."""
    slots = int(round(1.0 / step))
    for parts in product(range(slots + 1), repeat=n):
        if sum(parts) == slots:
            yield np.array(parts, dtype=float) / slots


def _best_blends(y: pd.Series, loaded, top_n: int = 8):
    """Search weighted blends and return the best OOF F1 candidates."""
    names = [x[0] for x in loaded]
    oofs = np.vstack([x[1] for x in loaded])
    results = []
    for weights in _weight_grid(len(loaded), step=0.1):
        if np.count_nonzero(weights) == 0:
            continue
        proba = np.average(oofs, axis=0, weights=weights)
        best_t = 0.5
        best_f1 = 0.0
        for threshold in np.linspace(0.35, 0.55, 41):
            cur = f1_score(y, proba >= threshold)
            if cur > best_f1:
                best_f1 = cur
                best_t = float(threshold)
        results.append((best_f1, best_t, weights))
    results.sort(key=lambda x: x[0], reverse=True)
    unique = []
    seen = set()
    for best_f1, best_t, weights in results:
        key = tuple(np.round(weights, 2))
        if key in seen:
            continue
        seen.add(key)
        unique.append((best_f1, best_t, weights))
        if len(unique) >= top_n:
            break
    print("=== blend candidates ===")
    for rank, (best_f1, best_t, weights) in enumerate(unique, start=1):
        desc = " + ".join(f"{name}:{weight:.1f}" for name, weight in zip(names, weights) if weight > 0)
        print(f"{rank}. f1={best_f1:.4f} thr={best_t:.3f} {desc}")
    return unique


def _make_topk_submission(test_meta, template, test_proba, name: str, pos_count: int) -> dict:
    """Create a submission by marking the top-K probabilities as positives."""
    threshold = float(np.sort(test_proba)[-pos_count])
    labels = (test_proba >= threshold).astype(int)
    if labels.sum() > pos_count:
        order = np.argsort(-test_proba)
        labels[:] = 0
        labels[order[:pos_count]] = 1
    sub = test_meta.copy()
    sub[TARGET_COLUMN] = labels
    sub = sub[list(template.columns)]
    sub.to_csv(SUBMISSION_DIR / name, index=False)
    pos = int(labels.sum())
    return {
        "file": name,
        "threshold": threshold,
        "pos": pos,
        "neg": int(len(labels) - pos),
        "pos_rate": float(pos / len(labels)),
    }


def _record_blend_to_lark(experiment_name: str, rows: list[dict], weights_desc: str, train_f1: float) -> None:
    """Append the blend experiment summary to the Lark sheet."""
    start = _next_lark_row(DEFAULT_LARK_TOKEN, DEFAULT_LARK_SHEET_ID)
    lark_rows = []
    for row in rows:
        lark_rows.append(
            [
                experiment_name,
                "?????" + weights_desc,
                "??????OOF/????",
                "",
                "",
                "????????????????? top-K ??",
                "OOF ??????",
                "",
                "???? 5?/?seed OOF",
                "",
                f"top{row['pos']}",
                row["file"],
                row["pos"],
                row["neg"],
                f"{row['pos_rate']:.4f}",
                "",
                f"{train_f1:.4f}",
            ]
        )
    _write_lark_rows(DEFAULT_LARK_TOKEN, DEFAULT_LARK_SHEET_ID, start, lark_rows)
    print(f"lark_record=ok start_row={start} rows={len(lark_rows)}")


def main() -> None:
    """Run blend search, write submissions, and record the best result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=1, help="Which blend candidate rank to materialize.")
    parser.add_argument("--pos-counts", default="148,150,151,152,153,154,155,156,158", help="Comma-separated positive counts.")
    args = parser.parse_args()

    bundle = build_dataset_bundle()
    y = bundle.train[TARGET_COLUMN]
    test_meta = bundle.test[[GENE_COLUMN, MIRNA_COLUMN]]
    template = pd.read_csv(SUBMIT_EXAMPLE_FILE)
    loaded = _load_inputs()
    candidates = _best_blends(y, loaded)
    rank_idx = max(0, min(args.rank - 1, len(candidates) - 1))
    best_f1, best_t, weights = candidates[rank_idx]
    names = [x[0] for x in loaded]
    test_stack = np.vstack([x[2] for x in loaded])
    test_proba = np.average(test_stack, axis=0, weights=weights)
    weights_desc = " + ".join(f"{name}:{weight:.1f}" for name, weight in zip(names, weights) if weight > 0)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"blend_rank{args.rank}_{run_id}"
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    print(f"selected={experiment_name} f1={best_f1:.4f} thr={best_t:.3f} {weights_desc}")
    rows = []
    for pos_count in [int(x.strip()) for x in args.pos_counts.split(",") if x.strip()]:
        name = f"submission_{experiment_name}_pos{pos_count}.csv"
        row = _make_topk_submission(test_meta, template, test_proba, name, pos_count)
        rows.append(row)
        print(f"{name}: pos={row['pos']} neg={row['neg']} pos_rate={row['pos_rate']:.3f}")

    pd.Series(test_proba, name="test_proba").to_csv(OOF_DIR / f"test_proba_{experiment_name}.csv", index=True, header=True)
    summary = {
        "experiment_name": experiment_name,
        "weights": {name: float(weight) for name, weight in zip(names, weights)},
        "best_train_f1": float(best_f1),
        "best_threshold": float(best_t),
        "submissions": rows,
    }
    with open(OOF_DIR / f"summary_{experiment_name}.json", "w", encoding="utf-8") as f:
        import json

        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    _record_blend_to_lark(experiment_name, rows, weights_desc, best_f1)


if __name__ == "__main__":
    main()
