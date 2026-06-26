"""
Pipeline A: Top-k Baseline
FAISS -> Top-k -> Token budget截断 -> 喂给 LLM
"""

from phase2_rag.faiss_index import FaissIndex
from phase2_rag.config import BUDGET_TOKENS, FAISS_TOP_K


def run_pipeline_topk(
    question,
    question_emb,
    candidate_embs,
    candidates_text,
    token_counts,
    budget_B=BUDGET_TOKENS,
    top_k=FAISS_TOP_K,
):
    """
    Pipeline A: Top-k (去重 + budget截断)
    """
    index = FaissIndex(dim=candidate_embs.shape[1])
    index.add(candidate_embs, candidates_text)
    distances, indices = index.search(question_emb, k=min(top_k, len(candidates_text)))

    top_indices = [int(i) for i in indices[0]]
    seen = set()
    selected_indices = []
    selected_texts = []
    used_budget = 0

    for idx in top_indices:
        if idx in seen:
            continue
        seen.add(idx)
        tc = token_counts[idx]
        if used_budget + tc > budget_B:
            continue
        selected_indices.append(idx)
        selected_texts.append(candidates_text[idx])
        used_budget += tc

    return selected_indices, selected_texts, used_budget
