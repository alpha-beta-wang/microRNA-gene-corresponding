# 实验 GUI 使用指南

本文档说明如何安装、启动、配置并查看本仓库中可配置实验 GUI 的输出结果。

## 1. 安装依赖

在项目根目录运行：

```powershell
python -m pip install -r requirements.txt
python -m pip install optuna biopython
```

说明：

- 只有启用 `hyperopt_enabled` 时才需要 `optuna`。
- 只有启用 `alignment` 特征模块时才需要 `biopython`。
- 只有启用 `rna_energy` 特征模块时才需要 `viennarna`。

## 2. 启动 GUI

运行：

```powershell
python -m src.experiment_gui
```

左侧面板包含特征开关、模型开关、训练方式选项和参数配置。
右侧面板显示运行日志，包括折进度、F1 分数、生成的 CSV 文件以及飞书表格写入状态。

## 3. 命令行运行

默认可配置实验：

```powershell
python -m src.experiment_runner --config configs/default_experiment.json
```

基于已整合同学分支的配置：

```powershell
python -m src.experiment_runner --config configs/harry_integrated_experiment.json
```

## 4. 流水线

运行器会执行以下流程：

```text
加载数据
  -> 构建选中的特征模块
  -> 使用指定随机种子和折数训练选中的模型
  -> 生成训练集 OOF 概率和测试集概率
  -> 按阈值生成提交 CSV 文件
  -> 保存 OOF、test_proba 和 summary 文件
  -> 启用时将实验记录写入飞书表格
```

`stacking`、`hard_negative`、`hyperopt` 等训练方式不会替代特征或模型选择。它们只改变所选模型的调参、融合或训练方式。

## 5. 特征模块

| 名称 | 含义 |
|---|---|
| `basic` | 基础序列统计特征，如长度、碱基比例、GC 含量和长度比 |
| `match` | 种子区和反向互补匹配特征 |
| `advanced` | 额外命中统计和局部 AT 含量 |
| `kmer` | miRNA/gene 字符 k-mer TF-IDF 特征 |
| `rna_energy` | 基于 ViennaRNA 的能量特征，需要 `viennarna` |
| `entity_counts` | 训练集中 gene/miRNA 实体出现频次 |
| `seed_variants` | 不同种子位置变体的匹配特征 |
| `dinucleotide` | 二核苷酸组成差异特征 |
| `targetscan` | TargetScan 风格位点特征 |
| `kmer_interaction` | 反向互补 3-mer 交互密度 |
| `alignment` | 局部/全局比对特征，需要 `biopython` |
| `position` | 3' 端位置相关种子特征 |
| `seed_type` | 8mer、7mer、6mer 和 GU wobble 特征 |
| `embedding` | 非深度学习 k-mer 共现 SVD 表征特征 |

## 6. 模型

| 名称 | 模型 |
|---|---|
| `lgbm` | LightGBM |
| `xgb` | XGBoost |
| `extra_trees` | ExtraTrees |
| `rf` | RandomForest |
| `svm` | 带标准化的 SVM |
| `knn` | 带标准化的 KNN |
| `fm` | 带标准化的因子分解机 |

## 7. 训练选项

### `training_method`

允许值：

```text
normal
hard_negative
```

`normal` 表示直接使用交叉验证训练选中的模型。

`hard_negative` 会先训练第一阶段模型，再根据 OOF 概率挑选困难负样本，最后使用全部正样本和选中的困难负样本重新训练。

相关参数：

| 参数 | 说明 |
|---|---|
| `hard_negative_threshold` | 选择 OOF 概率高于该阈值的真实负样本 |
| `hard_negative_top_fraction` | 非空时按 OOF 概率选择真实负样本中排名靠前的比例 |

### `ensemble_mode`

允许值：

```text
mean
stacking
```

`mean` 对选中模型的概率取平均。

`stacking` 会在各模型 OOF 概率上训练第二层 `LogisticRegression`。至少需要两个模型；否则运行器会回退到 `mean`。

### `hyperopt_enabled`

启用后，Optuna 会在交叉验证训练前搜索超参数。

当前支持搜索空间的模型：

```text
lgbm
xgb
```

不支持的模型会被跳过，并在日志中提示。

## 8. 主要参数

| 参数 | 说明 |
|---|---|
| `experiment_name` | summary 和 OOF 文件使用的实验名称 |
| `submission_prefix` | 生成提交 CSV 文件时使用的前缀 |
| `seeds` | 逗号分隔的随机种子 |
| `n_splits` | 交叉验证折数 |
| `thresholds` | 逗号分隔的提交阈值 |
| `mirna_k` | miRNA k-mer 长度 |
| `gene_k` | gene k-mer 长度 |
| `gene_max_features` | gene TF-IDF 特征最大数量 |
| `embedding_k` | embedding 特征使用的 k-mer 长度 |
| `embedding_dim` | embedding 特征的 SVD 维度 |
| `embedding_context_radius` | 共现上下文半径 |
| `embedding_min_count` | embedding 词表的最小 k-mer 计数 |
| `scale_pos_weight` | 使用 `auto` 或数值类别权重 |
| `hyperopt_trials` | 每个支持模型的 Optuna trial 数量 |

## 9. 配置文件

GUI 会将当前配置保存到：

```text
configs/gui_experiment.json
```

仓库提供的配置文件：

```text
configs/default_experiment.json
configs/harry_integrated_experiment.json
```

## 10. 输出文件

提交文件目录：

```text
outputs/submissions/
```

OOF、测试集概率和 summary 文件目录：

```text
outputs/oof/
```

典型文件名：

```text
outputs/submissions/<submission_prefix>_<run_time>_t<threshold>.csv
outputs/oof/oof_<experiment_name>_<run_time>.csv
outputs/oof/test_proba_<experiment_name>_<run_time>.csv
outputs/oof/summary_<experiment_name>_<run_time>.json
```

## 11. 飞书日志

默认通过配置中的 `lark` 段启用飞书日志。

本地测试时可以这样关闭：

```json
"lark": {
  "enabled": false
}
```

## 12. 故障排查

检查 GUI 导入：

```powershell
python -c "import src.experiment_gui; print('ok')"
```

如果 `alignment` 失败，安装：

```powershell
python -m pip install biopython
```

如果 `hyperopt` 失败，安装：

```powershell
python -m pip install optuna
```

如果 `rna_energy` 失败，安装：

```powershell
python -m pip install viennarna
```
