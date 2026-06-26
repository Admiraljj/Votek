# Phase 1：Vote-K 论文复现

本目录用于复现论文 *Selective Annotation Makes Language Models Better Few-Shot Learners* (Su et al., EMNLP 2022) 中的 Vote-K 算法。

## ⚠️ 代码来源说明

| 文件 | 来源 |
|------|------|
| `test_votek.py` | ✅ **本项目编写**——复现实验的驱动脚本 |
| `get_task.py` | 论文官方仓库 [HKUNLP/icl-selective-annotation](https://github.com/HKUNLP/icl-selective-annotation) |
| `two_steps.py` | 同上（Vote-K 算法核心实现）|
| `utils.py` | 同上 |
| `bridge_content_encoder.py` | 同上 |
| `main.py` | 同上 |
| `MetaICL/` | 同上（MetaICL 框架）|
| `selective_annotation.yml` | 同上（conda 环境配置）|

> 这些文件来自论文官方仓库，版权归原论文作者所有。我们仅基于它们运行复现实验，未修改其核心算法逻辑。

## 复现内容

- **数据集**：DBpedia-14（14 类维基百科实体分类）
- **任务**：对比 Random vs Vote-K 选择性注释对 ICL 效果的影响
- **注释预算**：100 个示例
- **模型**：GPT-2-XL（1.5B）与 Qwen3-4B（4B）

## 复现结果

| 模型 | Random | Vote-K | 提升 |
|------|--------|--------|------|
| GPT-2-XL (1.5B) | 8.98% | 7.03% | -1.95% |
| **Qwen3-4B (4B)** | **50.39%** | **60.16%** | **+9.77%** |

Qwen3-4B 上的 +9.77% 提升与论文报告的 +11.4% 基本一致，成功复现论文结论。

## 如何运行

### 1. 环境配置

建议使用论文官方提供的 conda 环境：

```bash
conda env create -f selective_annotation.yml
conda activate selective_annotation
```

或参考根目录 `requirements.txt` 中的 Phase 1 依赖。

### 2. 准备本地模型

`test_votek.py` 默认从 `models/` 目录加载本地模型。请下载以下模型之一：

```bash
mkdir models
# 例如下载 Qwen3-4B（从 ModelScope 或 HuggingFace）
# 放到 models/Qwen3-4B/
```

在 `test_votek.py` 中修改 `model_cache_dir` 指向你的模型路径。

### 3. 运行复现

```bash
cd phase1_reproduction
python test_votek.py --method votek --annotation_size 100
python test_votek.py --method random --annotation_size 100
```

## 相关文档

- 完整复现报告：[`../docs/EXPERIMENT_REPORT.md`](../docs/EXPERIMENT_REPORT.md)
- 论文：Su et al., *Selective Annotation Makes Language Models Better Few-Shot Learners*, EMNLP 2022
