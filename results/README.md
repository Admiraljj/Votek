# 实验结果

本目录存放 Phase 2（RAG 上下文选择）的最终实验结果。

## 文件说明

| 文件 | 说明 |
|------|------|
| `500samples_B1500_results.json` | 500 样本、B=1500 主实验的最终评测结果 |

### JSON 字段说明

```json
{
  "topk":     { "F1": ..., "EM": ..., "Evidence Coverage": ..., ... },
  "reranker": { ... },
  "hybrid":   { ... },
  "votek":    { ... }
}
```

每个管线包含以下指标：

| 字段 | 含义 |
|------|------|
| `count` | 样本数 |
| `F1` | F1-Score（LLM 答案与标准答案的词级 F1）|
| `EM` | Exact Match（完全匹配率）|
| `Evidence Coverage` | 证据覆盖率（选中上下文包含的金标准事实比例）|
| `Context Redundancy` | 上下文冗余度（两两余弦相似度均值，越低越好）|
| `Avg Tokens` | 平均使用的 Token 数 |
| `F1 per 1k Tokens` | 每 1000 Token 的 F1（信息密度）|
| `Evidence Efficiency (per 1k tokens)` | 每 1000 Token 覆盖的金标准句子数 |

## 如何重新生成

```bash
# 在项目根目录
export GITEE_API_KEY=你的key

python -m phase2_rag.run_experiment --sample_size 500 --budget 1500 \
    --votek_rho 1.05 --votek_k 3 --min_tokens 0 \
    --output_dir results/run_xxx
```

结果会保存到 `results/run_xxx/experiment_results.json`。

> 注：完整运行 500 样本约需 5 小时（含 Embedding/Reranker/LLM 的 API 调用）。
> 运行过程中会生成 `embedding_cache.json` 缓存文件（已被 `.gitignore` 排除）。
