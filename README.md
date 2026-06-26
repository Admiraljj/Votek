# Vote-K：从选择性注释到 RAG 上下文选择

> CS240 课程项目 · 复现并拓展 *Selective Annotation Makes Language Models Better Few-Shot Learners* (Su et al., 2022, HKU NLP)

本项目包含两个阶段的工作：

| 阶段 | 内容 | 是否原创 |
|------|------|---------|
| **Phase 1** | 复现 Vote-K 选择性注释算法，验证其在 ICL 任务中的效果 | 复现（基于论文官方代码）|
| **Phase 2** | 将 Vote-K 的核心思想迁移到 RAG 上下文选择场景，提出 Modified Vote-K 与 Hybrid 管线 | ✅ 原创工作 |

---

## 📌 核心结论

**Phase 1（论文复现）**：Vote-K 让 Qwen3-4B 在 DBpedia-14 上的准确率提升 **+9.77%**，与论文报告的 +11.4% 基本一致。

**Phase 2（RAG 拓展）**：在 HotpotQA 500 题大规模实验中，我们提出的 **Hybrid 管线（Reranker + Vote-K）以 23% 更少的 Token 实现了超越 Top-k 基线 +4.1% 的 F1，F1/1k Tokens 领先所有基线 28.8%**。

---

## 📁 项目结构

```
.
├── phase2_rag/          # Phase 2: RAG 上下文选择（原创）
│   ├── config.py           # 全局配置（API、模型、预算参数）
│   ├── data_loader.py      # HotpotQA 数据加载
│   ├── embedding_client.py # Qwen3-Embedding-8B 客户端
│   ├── reranker_client.py  # Qwen3-Reranker-4B 客户端
│   ├── llm_client.py       # Qwen3-4B 答案生成客户端
│   ├── faiss_index.py      # FAISS 向量检索
│   ├── modified_votek.py   # ⭐ 改造后的 Vote-K 核心算法
│   ├── pipeline_topk.py    # 管线 A: Top-k 基线
│   ├── pipeline_reranker.py# 管线 B: Reranker 基线
│   ├── pipeline_votek.py   # 管线 C: Modified Vote-K
│   ├── pipeline_hybrid.py  # 管线 D: Hybrid（Reranker + Vote-K）
│   ├── evaluator.py        # 评测指标（Coverage/F1/EM/Redundancy）
│   ├── run_experiment.py   # 实验主入口
│   ├── plot_results.py     # 可视化图表生成
│   └── figures/            # 实验图表
│
├── phase1_reproduction/    # Phase 1: Vote-K 论文复现
│   ├── test_votek.py       # 我们的复现驱动脚本
│   ├── get_task.py         # ↓ 以下来自论文官方仓库 HKUNLP/icl-selective-annotation
│   ├── two_steps.py        #   论文 Vote-K 算法实现
│   ├── utils.py            #   工具函数
│   ├── MetaICL/            #   MetaICL 框架
│   └── ...
│
├── results/                # 实验结果（最终 JSON）
├── docs/                   # 实验报告
└── requirements.txt
```

---

## 🔧 环境配置

### 1. Python 依赖

```bash
pip install -r requirements.txt
```

### 2. API Key 配置

本项目使用 **Gitee AI 平台**提供的模型 API（国内可直连，免费额度）：

```bash
# Windows
set GITEE_API_KEY=你的key

# Linux / macOS
export GITEE_API_KEY=你的key
```

API Key 申请：https://ai.gitee.com/

### 3. 所需模型

本项目通过 API 调用以下模型，**无需本地下载模型文件**：

| 用途 | 模型 | 调用方式 |
|------|------|---------|
| 文本向量化（Embedding）| Qwen3-Embedding-8B | Gitee API |
| 重排序（Reranker）| Qwen3-Reranker-4B | Gitee API |
| 答案生成（LLM）| Qwen3-4B | Gitee API |

> Phase 1 的 ICL 复现可选本地模型（如 Qwen3-4B、GPT-2-XL），下载到 `phase1_reproduction/models/` 即可。详见 `phase1_reproduction/README.md`。

### 4. GPU（可选）

- Phase 2 完全通过 API，**不需要 GPU**
- Phase 1 若用本地模型推理，需要 GPU（建议 ≥12GB 显存）

---

## 🚀 快速开始

### Phase 2：RAG 上下文选择实验

```bash
# 进入项目根目录
cd Votek_release

# 设置 API Key
export GITEE_API_KEY=你的key

# 运行 4 管线对比（50 样本，B=1500）
python -m phase2_rag.run_experiment --sample_size 50 --budget 1500 \
    --votek_rho 1.05 --votek_k 3 --min_tokens 0 \
    --output_dir results

# 大规模验证（500 样本）
python -m phase2_rag.run_experiment --sample_size 500 --budget 1500 \
    --votek_rho 1.05 --votek_k 3 --min_tokens 0 \
    --output_dir results_500

# 生成可视化图表
python -m phase2_rag.plot_results
```

**关键参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--sample_size` | 50 | 实验样本数 |
| `--budget` | 2000 | Token 预算 B |
| `--skip_llm` | False | 加上则跳过 LLM 答案生成（只测检索质量）|
| `--votek_rho` | 1.05 | Vote-K 衰减系数 |
| `--votek_k` | 3 | Vote-K 投票邻居数 |
| `--min_tokens` | 0 | 最小句子长度阈值 |

### Phase 1：Vote-K 论文复现

详见 [`phase1_reproduction/README.md`](phase1_reproduction/README.md)。

---

## 📊 实验结果

### Phase 2 主实验（500 样本，B=1500）

| 指标 | Top-k (A) | Reranker (B) | **Hybrid (D)** | Vote-K (C) |
|------|-----------|-------------|----------------|------------|
| F1-Score | 0.391 | **0.409** | **0.407** | 0.393 |
| Exact Match | 0.270 | **0.286** | 0.278 | 0.262 |
| Evidence Coverage | 0.998 | **1.000** | 0.996 | 0.997 |
| Context Redundancy | 0.346 | 0.344 | 0.365 | **0.342** |
| Avg Tokens Used | 1242 | 1242 | **959** | 1238 |
| **F1 per 1k Tokens** | 0.315 | 0.329 | **0.424** ⭐ | 0.317 |
| **Eff. per 1k Tokens** | 0.857 | 0.859 | **1.075** ⭐ | 0.858 |

**核心发现**：在高预算场景下（B=1500），所有管线 Coverage ≈ 100%，此时多样性成为关键区分因素。Hybrid 用更少 Token 达到更好答题效果，F1/1k Tokens 领先所有基线 28.8%。

完整结果与图表见 [`docs/`](docs/)。

---

## 📖 算法说明

### Modified Vote-K 核心公式

**原 Vote-K 增益函数**（论文）：

```
Δ(u | S) = Σ_{v ∈ N(u)} ρ^(-c_v)
```

**我们改造后的价值函数**（增加查询相关性 + Token 性价比）：

```
Value(u | S, q) = sim(u, q) × Σ_{v ∈ N(u)} ρ^(-c_v)

选择规则: u* = argmax  Value(u | S, q) / w_u     （0-1 背包贪心）
约束:     Σ w_u ≤ B                              （Token 预算）
```

三处关键改造：
1. **查询感知**：乘以 `sim(u, q)`，只选和当前问题相关的证据
2. **Token 预算**：从"选满 N 个"改为"塞满 B 个 Token"（0-1 背包）
3. **性价比**：除以 `w_u`（字数），优先选短小精悍的证据

详见 [`docs/FULL_EXPERIMENT_REPORT.md`](docs/FULL_EXPERIMENT_REPORT.md)。

---

## 🙏 致谢

- Phase 1 复现基于论文官方代码仓库 [HKUNLP/icl-selective-annotation](https://github.com/HKUNLP/icl-selective-annotation)
- 原论文：Su et al., *Selective Annotation Makes Language Models Better Few-Shot Learners*, EMNLP 2022
- 模型 API 由 [Gitee AI](https://ai.gitee.com/) 提供

---

## 📄 License

本项目采用 MIT License。Phase 1 复现代码的版权归原论文作者所有（见 `phase1_reproduction/`）。
