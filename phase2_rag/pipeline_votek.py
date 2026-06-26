"""
Pipeline C: Modified Vote-K (高信息密度流)
FAISS 向量召回 -> 魔改 Vote-K -> 榨干预算 B -> 喂给 LLM
"""

from phase2_rag.faiss_index import FaissIndex
from phase2_rag.modified_votek import modified_votek, modified_votek_simple
from phase2_rag.config import BUDGET_TOKENS, FAISS_TOP_K, VOTEK_K, VOTEK_RHO, MIN_TOKENS


def run_pipeline_votek(
    question,
    question_emb,
    candidate_embs,
    candidates_text,
    token_counts,
    budget_B=BUDGET_TOKENS,
    top_k=FAISS_TOP_K,
    k=VOTEK_K,
    rho=VOTEK_RHO,
    min_tokens=MIN_TOKENS,
    use_simple=False,
):
    """
    Pipeline C: 向量召回 + 魔改 Vote-K 选择

    Returns:
        selected_indices, selected_texts, used_budget
    """
    # 1. FAISS 召回 Top-k
    index = FaissIndex(dim=candidate_embs.shape[1])
    index.add(candidate_embs, candidates_text)
    _, faiss_indices = index.search(question_emb, k=min(top_k, len(candidates_text)))
    top_k_indices = [int(i) for i in faiss_indices[0]]

    # 获取子集
    top_k_embs = candidate_embs[top_k_indices]
    top_k_texts = [candidates_text[i] for i in top_k_indices]
    top_k_tokens = [token_counts[i] for i in top_k_indices]

    # 2. 魔改 Vote-K
    votek_fn = modified_votek_simple if use_simple else modified_votek
    local_selected, used = votek_fn(
        question_emb, top_k_embs, top_k_texts, top_k_tokens,
        budget_B=budget_B, k=k, rho=rho, min_tokens=min_tokens,
    )

    # 映射回原始索引
    selected_indices = [top_k_indices[i] for i in local_selected]
    selected_texts = [candidates_text[i] for i in selected_indices]

    return selected_indices, selected_texts, used
