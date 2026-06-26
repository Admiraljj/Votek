"""
Pipeline B: Reranker Baseline (工业主流流)
FAISS 向量召回 -> Qwen3-Reranker-4B 重排 -> 截断到 Token 预算 B -> 喂给 LLM
"""

from phase2_rag.faiss_index import FaissIndex
from phase2_rag.reranker_client import RerankerClient
from phase2_rag.config import BUDGET_TOKENS, FAISS_TOP_K


def run_pipeline_reranker(
    question,
    question_emb,
    candidate_embs,
    candidates_text,
    token_counts,
    budget_B=BUDGET_TOKENS,
    top_k=FAISS_TOP_K,
):
    """
    Pipeline B: 向量召回 + Reranker 重排

    Returns:
        selected_indices, selected_texts, used_budget
    """
    # 1. FAISS 召回 Top-k
    index = FaissIndex(dim=candidate_embs.shape[1])
    index.add(candidate_embs, candidates_text)
    distances, indices = index.search(question_emb, k=min(top_k, len(candidates_text)))

    top_indices = [int(i) for i in indices[0]]
    top_texts = [candidates_text[i] for i in top_indices]

    # 2. Reranker 重排
    reranker = RerankerClient()
    ranked = reranker.rerank(question, top_texts)

    # 3. 按重排分数排序，截断到预算
    selected_indices = []
    selected_texts = []
    used_budget = 0

    for r in ranked:
        orig_local_idx = r["index"]
        orig_idx = top_indices[orig_local_idx]
        tc = token_counts[orig_idx]
        if used_budget + tc > budget_B:
            continue
        selected_indices.append(orig_idx)
        selected_texts.append(candidates_text[orig_idx])
        used_budget += tc

    return selected_indices, selected_texts, used_budget
