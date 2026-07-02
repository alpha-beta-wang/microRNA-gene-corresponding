"""数据模块命令行入口，用于检查或加载项目数据。"""

from src.data.load_data import build_dataset_bundle, save_merged_frames


if __name__ == "__main__":
    bundle = build_dataset_bundle()
    save_merged_frames(bundle)

    print(f"train_rows={len(bundle.train)}")
    print(f"test_rows={len(bundle.test)}")
    print(f"train_missing_gene_sequences={bundle.train_missing_gene_sequences}")
    print(f"train_missing_mirna_sequences={bundle.train_missing_mirna_sequences}")
    print(f"test_missing_gene_sequences={bundle.test_missing_gene_sequences}")
    print(f"test_missing_mirna_sequences={bundle.test_missing_mirna_sequences}")
"""数据模块命令行入口，用于检查或加载项目数据。"""
