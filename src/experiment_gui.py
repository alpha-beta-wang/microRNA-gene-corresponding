"""实验配置图形界面，封装参数编辑、配置保存和后台运行。"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

from src.config import PROJECT_ROOT
from src.experiment_runner import DEFAULT_CONFIG, load_config, save_config


FEATURE_LABELS = {
    "basic": "??????",
    "match": "??????",
    "advanced": "??????",
    "kmer": "k-mer TF-IDF",
    "rna_energy": "RNA ????",
    "entity_counts": "gene/miRNA ????",
    "seed_variants": "seed ????",
    "dinucleotide": "??????",
    "targetscan": "TargetScan ????",
    "kmer_interaction": "???? k-mer ??",
    "alignment": "??/??????",
    "position": "3' ?????",
    "seed_type": "?? seed ??/GU wobble",
    "embedding": "k-mer ?? SVD ??",
}

MODEL_LABELS = {
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "extra_trees": "ExtraTrees",
    "rf": "RandomForest",
    "svm": "SVM",
    "knn": "KNN ????",
    "fm": "Factorization Machine",
}


class ExperimentGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("DataFountain 534 ?????")
        self.geometry("1160x800")
        self.minsize(980, 680)

        self.config_path = PROJECT_ROOT / "configs" / "gui_experiment.json"
        self.config_data = load_config(DEFAULT_CONFIG)
        self.feature_vars: dict[str, tk.BooleanVar] = {}
        self.model_vars: dict[str, tk.BooleanVar] = {}
        self.entries: dict[str, tk.Entry] = {}
        self.combo_vars: dict[str, tk.StringVar] = {}
        self.bool_vars: dict[str, tk.BooleanVar] = {}
        self.run_button: ttk.Button | None = None

        self._build_ui()
        self._load_to_ui()

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self, padding=(10, 8))
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="????", command=self._save_from_ui).pack(side=tk.LEFT, padx=(0, 8))
        self.run_button = ttk.Button(toolbar, text="????", command=self._run_experiment)
        self.run_button.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(toolbar, text="??????", command=self._show_config).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(toolbar, text="????????????????F1?csv ???????????").pack(
            side=tk.LEFT,
            padx=(12, 0),
        )

        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        left_outer = ttk.Frame(main)
        right = ttk.Frame(main)
        main.add(left_outer, weight=0)
        main.add(right, weight=1)

        canvas = tk.Canvas(left_outer, width=380, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_outer, orient=tk.VERTICAL, command=canvas.yview)
        self.form = ttk.Frame(canvas)
        self.form.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.form, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(self.form, text="????", font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w", pady=(0, 4))
        for name, label in FEATURE_LABELS.items():
            var = tk.BooleanVar()
            self.feature_vars[name] = var
            ttk.Checkbutton(self.form, text=f"{label} ({name})", variable=var).pack(anchor="w", pady=2)

        ttk.Label(self.form, text="????", font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w", pady=(14, 4))
        for name, label in MODEL_LABELS.items():
            var = tk.BooleanVar()
            self.model_vars[name] = var
            ttk.Checkbutton(self.form, text=f"{label} ({name})", variable=var).pack(anchor="w", pady=2)

        ttk.Label(self.form, text="????", font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w", pady=(14, 4))
        self.combo_vars["training_method"] = tk.StringVar()
        ttk.Label(self.form, text="????").pack(anchor="w", pady=(4, 0))
        ttk.Combobox(
            self.form,
            textvariable=self.combo_vars["training_method"],
            values=["normal", "hard_negative"],
            width=43,
            state="readonly",
        ).pack(anchor="w", fill=tk.X, pady=(0, 3))

        self.combo_vars["ensemble_mode"] = tk.StringVar()
        ttk.Label(self.form, text="????").pack(anchor="w", pady=(4, 0))
        ttk.Combobox(
            self.form,
            textvariable=self.combo_vars["ensemble_mode"],
            values=["mean", "stacking"],
            width=43,
            state="readonly",
        ).pack(anchor="w", fill=tk.X, pady=(0, 3))

        self.bool_vars["hyperopt_enabled"] = tk.BooleanVar()
        ttk.Checkbutton(self.form, text="?? hyperopt/Optuna ????", variable=self.bool_vars["hyperopt_enabled"]).pack(
            anchor="w",
            pady=2,
        )

        fields = [
            ("experiment_name", "???"),
            ("submission_prefix", "??????"),
            ("seeds", "?????????"),
            ("n_splits", "??"),
            ("thresholds", "???????"),
            ("mirna_k", "miRNA k-mer k"),
            ("gene_k", "gene k-mer k"),
            ("gene_max_features", "gene k-mer ?????"),
            ("embedding_k", "embedding k-mer k"),
            ("embedding_dim", "embedding SVD ??"),
            ("embedding_context_radius", "embedding ?????"),
            ("embedding_min_count", "embedding ????"),
            ("scale_pos_weight", "?????? auto/??"),
            ("hard_negative_threshold", "hard negative ??"),
            ("hard_negative_top_fraction", "hard negative top?????"),
            ("hyperopt_trials", "hyperopt trial ?"),
        ]
        ttk.Label(self.form, text="??", font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w", pady=(14, 4))
        for key, label in fields:
            ttk.Label(self.form, text=label).pack(anchor="w", pady=(4, 0))
            entry = ttk.Entry(self.form, width=48)
            entry.pack(anchor="w", fill=tk.X, pady=(0, 3))
            self.entries[key] = entry

        ttk.Label(right, text="????", font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w", pady=(0, 4))
        self.log = scrolledtext.ScrolledText(right, font=("Consolas", 10), wrap=tk.WORD)
        self.log.pack(fill=tk.BOTH, expand=True)
        self.log.insert(
            tk.END,
            "????????????????????????????? F1???? csv ???????????\n",
        )

    def _load_to_ui(self) -> None:
        cfg = self.config_data
        training = cfg.get("training", {})
        for name, var in self.feature_vars.items():
            var.set(bool(cfg.get("features", {}).get(name, False)))
        for name, var in self.model_vars.items():
            var.set(bool(cfg.get("models", {}).get(name, False)))
        self.combo_vars["training_method"].set(training.get("training_method", "normal"))
        self.combo_vars["ensemble_mode"].set(training.get("ensemble_mode", "mean"))
        self.bool_vars["hyperopt_enabled"].set(bool(training.get("hyperopt_enabled", False)))

        values = {
            "experiment_name": cfg.get("experiment_name", "gui_experiment"),
            "submission_prefix": cfg.get("output", {}).get("submission_prefix", "submission_gui"),
            "seeds": ",".join(str(x) for x in training.get("seeds", [42])),
            "n_splits": str(training.get("n_splits", 5)),
            "thresholds": ",".join(str(x) for x in cfg.get("thresholds", [0.46])),
            "mirna_k": str(cfg.get("kmer", {}).get("mirna_k", 3)),
            "gene_k": str(cfg.get("kmer", {}).get("gene_k", 3)),
            "gene_max_features": str(cfg.get("kmer", {}).get("gene_max_features", 256)),
            "embedding_k": str(cfg.get("embedding", {}).get("k", 3)),
            "embedding_dim": str(cfg.get("embedding", {}).get("dim", 12)),
            "embedding_context_radius": str(cfg.get("embedding", {}).get("context_radius", 2)),
            "embedding_min_count": str(cfg.get("embedding", {}).get("min_count", 2)),
            "scale_pos_weight": str(training.get("scale_pos_weight", "auto")),
            "hard_negative_threshold": str(training.get("hard_negative_threshold", 0.8)),
            "hard_negative_top_fraction": str(training.get("hard_negative_top_fraction", "")),
            "hyperopt_trials": str(training.get("hyperopt_trials", 20)),
        }
        for key, value in values.items():
            self.entries[key].delete(0, tk.END)
            self.entries[key].insert(0, value)

    def _config_from_ui(self) -> dict:
        cfg = json.loads(json.dumps(self.config_data))
        cfg.setdefault("training", {})
        cfg.setdefault("kmer", {})
        cfg.setdefault("embedding", {})
        cfg.setdefault("output", {})

        training = cfg["training"]
        cfg["experiment_name"] = self.entries["experiment_name"].get().strip() or "gui_experiment"
        cfg["features"] = {name: var.get() for name, var in self.feature_vars.items()}
        cfg["models"] = {name: var.get() for name, var in self.model_vars.items()}
        training["seeds"] = [int(x.strip()) for x in self.entries["seeds"].get().split(",") if x.strip()]
        training["n_splits"] = int(self.entries["n_splits"].get())
        training["training_method"] = self.combo_vars["training_method"].get() or "normal"
        training["ensemble_mode"] = self.combo_vars["ensemble_mode"].get() or "mean"
        training["hyperopt_enabled"] = bool(self.bool_vars["hyperopt_enabled"].get())
        training["hyperopt_trials"] = int(self.entries["hyperopt_trials"].get())
        training["hard_negative_threshold"] = float(self.entries["hard_negative_threshold"].get())
        hnf = self.entries["hard_negative_top_fraction"].get().strip()
        training["hard_negative_top_fraction"] = hnf
        spw = self.entries["scale_pos_weight"].get().strip()
        training["scale_pos_weight"] = spw if spw == "auto" else float(spw)

        cfg["thresholds"] = [float(x.strip()) for x in self.entries["thresholds"].get().split(",") if x.strip()]
        cfg["kmer"]["mirna_k"] = int(self.entries["mirna_k"].get())
        cfg["kmer"]["gene_k"] = int(self.entries["gene_k"].get())
        cfg["kmer"]["gene_max_features"] = int(self.entries["gene_max_features"].get())
        cfg["embedding"]["k"] = int(self.entries["embedding_k"].get())
        cfg["embedding"]["dim"] = int(self.entries["embedding_dim"].get())
        cfg["embedding"]["context_radius"] = int(self.entries["embedding_context_radius"].get())
        cfg["embedding"]["min_count"] = int(self.entries["embedding_min_count"].get())
        cfg["output"]["submission_prefix"] = self.entries["submission_prefix"].get().strip() or cfg["experiment_name"]
        cfg.setdefault("lark", {})
        cfg["lark"].setdefault("enabled", True)
        return cfg

    def _save_from_ui(self) -> None:
        try:
            cfg = self._config_from_ui()
            save_config(cfg, self.config_path)
            self._append_log(f"??????{self.config_path}\n")
        except Exception as exc:
            messagebox.showerror("????", str(exc))

    def _show_config(self) -> None:
        try:
            cfg = self._config_from_ui()
            self._append_log(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
        except Exception as exc:
            messagebox.showerror("????", str(exc))

    def _run_experiment(self) -> None:
        try:
            cfg = self._config_from_ui()
            save_config(cfg, self.config_path)
        except Exception as exc:
            messagebox.showerror("????", str(exc))
            return
        if self.run_button is not None:
            self.run_button.configure(state=tk.DISABLED)
        threading.Thread(target=self._run_worker, daemon=True).start()

    def _run_worker(self) -> None:
        cmd = [sys.executable, "-m", "src.experiment_runner", "--config", str(self.config_path)]
        self._append_log("?????" + " ".join(cmd) + "\n")
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            self._append_log(line)
        exit_code = proc.wait()
        self._append_log(f"?????????{exit_code}\n")
        self.after(0, self._enable_run_button)

    def _append_log(self, text: str) -> None:
        self.after(0, lambda: self._append_log_in_ui(text))

    def _append_log_in_ui(self, text: str) -> None:
        self.log.insert(tk.END, text)
        self.log.see(tk.END)

    def _enable_run_button(self) -> None:
        if self.run_button is not None:
            self.run_button.configure(state=tk.NORMAL)


def main() -> None:
    ExperimentGui().mainloop()


if __name__ == "__main__":
    main()
