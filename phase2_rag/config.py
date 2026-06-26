"""
全局配置文件：API keys、模型名、预算参数等

使用前请设置环境变量 GITEE_API_KEY
Windows: set GITEE_API_KEY=你的key
Linux/Mac: export GITEE_API_KEY=你的key
"""

import os

# ============================================================
# API 配置 (Gitee AI 平台)
# ============================================================
GITEE_API_BASE = "https://ai.gitee.com/v1"
GITEE_API_KEY = os.environ.get("GITEE_API_KEY", "")

# Embedding 模型
EMBEDDING_MODEL = "Qwen3-Embedding-8B"
EMBEDDING_DIM = 1024
EMBEDDING_INSTRUCTION = "为以下文本生成语义嵌入，用于检索与问题相关的证据段落。"

# Reranker 模型
RERANKER_MODEL = "Qwen3-Reranker-4B"
RERANKER_INSTRUCTION = "按照与问题的相关性和信息量对候选文档进行排序"

# ============================================================
# FAISS 配置
# ============================================================
FAISS_TOP_K = 100  # 向量召回数量

# ============================================================
# 魔改 Vote-K 参数
# ============================================================
VOTEK_K = 5         # 投票邻居数（调低：减少对同文章多事实的惩罚）
VOTEK_RHO = 1.1     # 衰减系数（调低：温和衰减，保留更多相关事实）
MIN_TOKENS = 3      # 最小句子长度阈值（调低：避免过滤掉短金标准句子）

# ============================================================
# 预算与实验参数
# ============================================================
BUDGET_TOKENS = 2000       # 给 LLM 的上下文 Token 预算
EVAL_SAMPLE_SIZE = 500     # 评测用的题目数量（从 HotpotQA 验证集采样）
LLM_MAX_NEW_TOKENS = 128   # LLM 答案生成最大长度

# ============================================================
# 数据集配置
# ============================================================
HOTPOTQA_DATASET = "hotpotqa/hotpot_qa"
HOTPOTQA_CONFIG = "distractor"
HOTPOTQA_SPLIT = "validation"  # 用验证集做评测（有金标准答案）

# ============================================================
# API 调用配置
# ============================================================
API_MAX_RETRIES = 3
API_RETRY_DELAY = 2        # 秒，exponential backoff 初始延迟
EMBEDDING_BATCH_SIZE = 32   # 每次批量编码的文本数
RERANKER_BATCH_SIZE = 100   # Reranker 单次最大文档数
