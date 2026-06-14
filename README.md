# microRNA-gene-corresponding

microRNA 与 gene 功能性 MTI (MicroRNA-Target Interaction) 二分类预测。

赛题来源：[DataFountain 竞赛 #534](https://www.datafountain.cn/competitions/534)

---

## 赛题背景

microRNA 在基因表达调控中起重要作用。microRNA 通过与 gene 的 mRNA 结合来调控基因的表达。预测潜在的 microRNA 与 gene 的关联关系可以帮助科学家更好地理解疾病的发生机制，并为精准医疗提供潜在的靶点。

**任务**：给定一对 (gene, miRNA) 及其序列，判断该配对是否为 Functional MTI（二分类）。

**评估指标**：F1-score。

---

## 数据集

| 文件 | 路径 | 说明 |
|------|------|------|
| 训练集 | `data/train_dataset/Train.csv` | 训练集，标签为 Functional MTI / Non-Functional MTI |
| 测试集 | `data/test_dataset.csv` | 测试集，无标签 |
| Gene 序列 | `data/train_dataset/gene_seq.csv` | Gene 名称与完整序列映射 |
| miRNA 序列 | `data/train_dataset/mirna_seq.csv` | miRNA 名称与序列映射 |
| 提交样例 | `data/submit_example.csv` | 提交格式：gene, miRNA, results (0/1) |

**训练集规模**：738 条 | **测试集规模**：185 条

---

## Quickstart

### 环境要求

- Python 3.11+
- Windows / Linux / macOS

### 1. 创建虚拟环境

```bash
python -m venv .venv
```

### 2. 激活虚拟环境

**Windows (bash)**：
```bash
source .venv/Scripts/activate
```

**Windows (PowerShell)**：
```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS**：
```bash
source .venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 运行基线流水线

```bash
python -m src.run_baseline
```

一键完成：数据加载 → 特征构建 → 双模型训练 → 阈值优化 → 提交生成。

### 5. 运行泄漏检查

```bash
python -m src.models.validate_groups
```

### 6. 查看产出

```
outputs/
├── features/          # 合并后的特征表 (parquet)
├── oof/               # 交叉验证 OOF 预测
│   ├── oof_lgbm.csv
│   ├── oof_xgb.csv
│   └── oof_ensemble.csv
├── models/            # 训练好的模型文件
│   ├── lgbm_fold*.pkl
│   ├── xgb_fold*.pkl
│   └── ensemble_info.txt
└── submissions/       # 提交文件
    ├── submission_lgbm.csv
    ├── submission_xgb.csv
    └── submission_ensemble.csv
```

---

## 项目结构

```
.
├── data/                          # 原始赛题数据
│   ├── test_dataset.csv
│   ├── submit_example.csv
│   └── train_dataset/
│       ├── Train.csv
│       ├── gene_seq.csv
│       └── mirna_seq.csv
├── src/                           # 源代码
│   ├── config.py                  # 路径、列名、随机种子等全局常量
│   ├── run_baseline.py            # 一键入口：数据→特征→训练→提交
│   ├── data/
│   │   ├── load_data.py           # 数据读取、清洗、merge
│   │   └── __main__.py            # 数据加载自检入口
│   ├── features/
│   │   ├── basic_features.py      # 基础序列特征（长度、GC、核苷酸比例）
│   │   └── sequence_match_features.py  # 配对特征（seed 匹配、连续配对等）
│   └── models/
│       ├── train_ensemble.py      # LGBM + XGBoost 双模型训练与交叉验证
│       ├── predict.py             # 推理与提交文件生成
│       └── validate_groups.py     # GroupKFold 实体泄漏检查
├── outputs/                       # 生成的中间文件和提交（不提交 git）
│   ├── features/
│   ├── oof/
│   ├── models/
│   └── submissions/
├── requirements.txt               # Python 依赖
├── .gitignore                     # 排除 .venv/ 和 outputs/
└── README.md
```

---

## 模块介绍

### `src/config.py` — 全局配置

管理所有路径、列名和超参，其他模块从这里引用，避免散落魔数

```python
TARGET_COLUMN = "results"    # 预测目标
SEED = 42                    # 随机种子
N_SPLITS = 5                 # 交叉验证折数
```

### `src/data/load_data.py` — 数据加载与清洗

**数据流**：

```
Train.csv (gene, miRNA, label) ──┐
gene_seq.csv (label, sequence) ──┤── merge → train_merged
mirna_seq.csv (mirna, seq)    ──┘

test_dataset.csv (gene, miRNA) ──┐
gene_seq.csv                     ──┤── merge → test_merged
mirna_seq.csv                    ──┘
```

**处理步骤**：
1. 读取 CSV 文件，统一列名（`label` → `gene`，`mirna` → `miRNA`）
2. 序列清洗：去除换行符、引号、空白字符，统一大写
3. 标签映射：`"Functional MTI"` → 1，`"Non-Functional MTI"` → 0
4. 列名统一：`label` → `results`
5. Left-join 关联序列，统计命中率（当前 100%）

### `src/features/basic_features.py` — 基础序列特征

对每条 (gene, miRNA) 对计算：

| 特征 | 维度 | 说明 |
|------|------|------|
| gene_length | 1 | gene 序列长度 |
| mirna_length | 1 | miRNA 序列长度 |
| gene_{A,C,G,T}_ratio | 4 | gene 各碱基占比 |
| mirna_{A,C,G,U}_ratio | 4 | miRNA 各碱基占比 |
| gene_gc | 1 | gene GC 含量 |
| mirna_gc | 1 | miRNA GC 含量 |
| length_ratio | 1 | gene 长度 / miRNA 长度 |

**原理**：GC 含量影响序列互补配对的结合强度与热力学稳定性，是 miRNA-target 相互作用的基础物理化学特征。

### `src/features/sequence_match_features.py` — 序列配对特征

围绕 miRNA **seed 区**（第 2-8 位碱基，这是 miRNA 识别靶基因的核心区域）进行字符串匹配：

| 特征 | 说明 |
|------|------|
| seed_in_gene | seed 区是否出现在 gene 序列中（正向） |
| seed_revcomp_in_gene | seed 区反向互补是否出现在 gene 序列中 |
| mirna_in_gene | 完整 miRNA 序列是否出现在 gene 中 |
| mirna_revcomp_in_gene | 完整 miRNA 反向互补是否出现在 gene 中 |
| seed_occurrence_count | seed 区在 gene 中出现的次数 |
| seed_max_consecutive | seed 区与 gene 窗口的最长连续匹配碱基数 |
| mirna_max_consecutive | 完整 miRNA 与 gene 窗口的最长连续匹配 |
| seed_gc | seed 区 GC 含量 |
| seed_A_count / seed_U_count | seed 区 A/U 计数 |

> `gene_N_ratio` ~~gene 序列中未知碱基 (N) 的比例~~ — **已注释掉**。原始数据经赛题官方清洗后不含未知碱基 N，该特征恒为 0，无预测价值。详见提交记录。

**原理**：miRNA 通过 seed 区与靶 mRNA 的 3'UTR 互补配对实现调控。seed 区的出现方式、连续匹配程度和互补方向是决定是否存在功能性 MTI 的关键信号。

反向互补计算使用标准碱基配对规则：A↔U、C↔G、G↔C、U↔A、T↔A。

### `src/models/train_ensemble.py` — 模型训练

**模型选择**：

| 模型 | 类型 | 来源 |
|------|------|------|
| LightGBM 4.6.0 | 梯度提升决策树 (GBDT) | 微软开源，PyPI 预编译 wheel |
| XGBoost 3.2.0 | 梯度提升决策树 (GBDT) | 开源，PyPI 预编译 wheel |

两个模型是本地安装的开源库，训练和推理均在本机 CPU 完成

**训练流程**：

1. **5 折分层交叉验证** (`StratifiedKFold`)：保证每折正负样本比例一致
2. 每折分别训练 LGBM 和 XGBoost
3. 验证集概率存入 OOF (Out-of-Fold) 数组
4. 在 OOF 上搜索最优阈值（0.1 ~ 0.9，101 步），最大化 F1

**超参选择**（小数据集防过拟合）：
- `learning_rate=0.01`：小学习率提高泛化能力
- `max_depth=6`：限制树深度
- `subsample=0.8, colsample_bytree=0.8`：行/列采样增加模型多样性
- `reg_alpha=0.1, reg_lambda=0.1`：L1/L2 正则化
- `early_stopping_round=100`：验证集 loss 不降则提前停止

### `src/models/predict.py` — 推理与提交

1. K 折模型分别对测试集预测概率
2. 取 K 折平均 → 更鲁棒的预测
3. 用最优阈值二值化 → 0/1 标签
4. 按 `submit_example.csv` 格式写出提交文件

支持三种提交模式：
- 单模型提交（LGBM / XGBoost）
- 集成提交（LGBM + XGBoost 概率平均）

### `src/models/validate_groups.py` — 泄漏检查

用 `GroupKFold` 分别按 gene、miRNA、gene+miRNA 对分组做交叉验证。如果分组得分显著低于分层 CV，说明模型在记忆实体身份而非学习真实配对模式

### `src/features/kmer_features.py` — k-mer 频率特征

对 gene (ACGT) 和 miRNA (ACGU) 分别统计 k-mer 归一化频率（k=2,3），产生 160 列带 `kmer__` 前缀的特征。无外部依赖。

### `src/features/alignment_features.py` — 生物信息学比对特征

使用 Biopython 计算每对序列的 Smith-Waterman (local) 和 Needleman-Wunsch (global) 比对得分及归一化版本，共 7 列带 `align__` 前缀的特征。需 `pip install biopython`。

### `src/features/rna_energy_features.py` — RNA 热力学 MFE 特征

使用 ViennaRNA 计算 miRNA MFE、duplex 结合能、候选窗口数量等热力学特征，共 7 列带 `rna__` 前缀的特征。需 `pip install viennarna` 或安装 ViennaRNA 命令行工具，启用时后台不可用则直接报错。

### `src/models/hard_negative.py` — 二阶段难负样本挖掘

Stage-1 正常 CV 训练，用 OOF 概率从真实负样本中选出预测概率偏高的 "hard negatives"；Stage-2 在"全部正样本 + hard negatives"上训练，并将 Stage-1 OOF 作为 meta-feature 注入。推理时两阶段串联输出。

### `src/features/build_features.py` — 特征块注册与统一拼装

根据 `--feature-blocks` 参数懒加载对应的特征计算函数，拼接 train/test 特征表，去重列名并对齐测试集列到训练集。

### `src/run_baseline.py` — 可配置流水线入口

通过 `argparse` 提供完整的 CLI 控制，支持特征块选择、模型组合、训练策略切换。默认零参数运行等效于原 baseline：

```bash
python -m src.run_baseline
# 等价于
python -m src.run_baseline --feature-blocks basic,match --models lgbm,xgb --threshold-search on
```

**完整 CLI 参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--feature-blocks` | `basic,match` | 特征块：basic, match, kmer, alignment, rna_energy |
| `--models` | `lgbm,xgb` | 模型：lgbm, xgb |
| `--run-tag` | — | 实验标签，用于隔离输出文件 |
| `--threshold-search` | `on` | OOF 最优阈值搜索 |
| `--second-stage` | `none` | 二阶段策略：hard_negative |
| `--hard-negative-threshold` | `0.8` | hard-negative 概率阈值 |
| `--hard-negative-top-fraction` | — | 以 top fraction 选 hard negatives（覆盖阈值） |
| `--missing-external-policy` | `error` | 外部工具缺失行为：error（报错）/ skip（跳过） |
| `--ensemble-mode` | `mean` | 集成方式 |

流水线串联：

```
加载数据 → 构建特征 → [二阶段训练] → 阈值搜索 → 生成提交
```

---

## 基线结果

数据集：738 训练样本，185 测试样本

### Baseline（basic+match，23 特征）

| 模型 | 5 折 CV F1 (0.5 阈值) | OOF 最优 F1 | 最优阈值 |
|------|----------------------|-------------|----------|
| LightGBM | 0.8271 | 0.8305 | 0.452 |
| XGBoost | 0.8248 | 0.8316 | 0.388 |
| Ensemble (平均) | — | 0.8299 | 0.460 |

### basic+match+kmer（183 特征）

| 模型 | OOF 最优 F1 | 最优阈值 |
|------|-------------|----------|
| LightGBM | 0.8361 | 0.484 |
| XGBoost | 0.8346 | 0.476 |
| Ensemble | 0.8362 | 0.484 |

### 全特征（basic+match+kmer+alignment+rna_energy，197 特征）

| 模型 | OOF 最优 F1 | 最优阈值 |
|------|-------------|----------|
| LightGBM | 0.8332 | 0.468 |
| XGBoost | 0.8311 | 0.460 |
| Ensemble | 0.8310 | 0.476 |

### 泄漏检查结果

| 分组依据 | 5 折 CV F1 | vs 分层 CV 差异 |
|----------|-----------|----------------|
| 分层 (无分组) | 0.8250 | — |
| gene | 0.8202 | -0.005 |
| miRNA | 0.8223 | -0.003 |
| gene+miRNA | 0.8217 | -0.003 |

分组得分与分层 CV 基本一致，无显著实体泄漏

---

## 常用命令

```bash
# 运行基线流水线（默认 23 特征）
python -m src.run_baseline

# 启用 k-mer 频率特征（183 特征）
python -m src.run_baseline --feature-blocks basic,match,kmer

# 启用比对特征（需 biopython）
python -m src.run_baseline --feature-blocks basic,match,alignment

# 启用 RNA 热力学特征（需 viennarna）
python -m src.run_baseline --feature-blocks basic,match,rna_energy

# 全部特征（197 特征）
python -m src.run_baseline --feature-blocks basic,match,kmer,alignment,rna_energy

# 单模型（仅 XGBoost）
python -m src.run_baseline --models xgb

# 二阶段 hard-negative 挖掘
python -m src.run_baseline --feature-blocks basic,match,kmer --second-stage hard_negative

# 实验运行（带 run-tag，避免覆盖 baseline 产物）
python -m src.run_baseline --feature-blocks basic,match,kmer --run-tag kmer_exp

# 仅测试数据加载
python -m src.data

# 运行泄漏检查
python -m src.models.validate_groups

# 查看数据统计
python -c "from src.data.load_data import build_dataset_bundle; b = build_dataset_bundle(); print(b.train.describe())"
```

---

## 依赖

### 核心依赖（必装）

```
pandas>=2.2.0
numpy>=1.26.0
scikit-learn>=1.5.0
lightgbm>=4.5.0
xgboost>=2.1.0
pyarrow>=17.0.0
joblib>=1.4.0
```

### 可选依赖（按需安装）

| 依赖 | 用途 | 安装命令 |
|------|------|----------|
| `biopython>=1.80` | alignment 特征块（Smith-Waterman / Needleman-Wunsch） | `pip install biopython` |
| `viennarna>=2.6.0` | rna_energy 特征块（MFE / duplex 热力学） | `pip install viennarna` |

当 `--feature-blocks` 包含 `alignment` 或 `rna_energy` 但对应包未安装时，默认会报错并给出安装指引。可通过 `--missing-external-policy skip` 跳过缺失的 block。

所有核心依赖均为开源库，通过 `pip install` 从 PyPI 下载预编译 wheel。Biopython 和 ViennaRNA 也提供 Windows/Linux/macOS 预编译 wheel。

