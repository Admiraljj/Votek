"""
Pipeline D: Reranker + Vote-K Hybrid
Reranker filters for Coverage, then Vote-K optimizes for Diversity
"""

from phase2_rag.faiss_index import FaissIndex
from phase2_rag.reranker_client import RerankerClient
from phase2_rag.modified_votek import modified_votek, modified_votek_simple
from phase2_rag.config import BUDGET_TOKENS, FAISS_TOP_K, VOTEK_K, VOTEK_RHO, MIN_TOKENS
import numpy as np


def run_pipeline_hybrid(
    question,
    question_emb,
    candidate_embs,
    candidates_text,
    token_counts,
    budget_B=BUDGET_TOKENS,
    top_k=FAISS_TOP_K,
    reranker_top_n=30,
    k=VOTEK_K,
    rho=VOTEK_RHO,
    min_tokens=MIN_TOKENS,
    use_simple=False,
):
    """
    Pipeline D: Reranker (Coverage) + Vote-K (Diversity)

    Step 1: FAISS recall Top-100
    Step 2: Reranker rerank, keep top-N (high Coverage)
    Step 3: Modified Vote-K on top-N (low Redundancy)
    Step 4: Budget truncation

    Returns:
        selected_indices, selected_texts, used_budget
    """
    # Step 1: FAISS recall
    index = FaissIndex(dim=candidate_embs.shape[1])
    index.add(candidate_embs, candidates_text)
    _, faiss_indices = index.search(question_emb, k=min(top_k, len(candidates_text)))
    faiss_top = [int(i) for i in faiss_indices[0]]
    faiss_texts = [candidates_text[i] for i in faiss_top]

    # Step 2: Reranker rerank, keep top-N
    reranker = RerankerClient()
    ranked = reranker.rerank(question, faiss_texts, top_n=min(reranker_top_n, len(faiss_texts)))

    reranked_indices = []
    reranked_texts = []
    reranked_embs_list = []
    reranked_tokens = []

    for r in ranked:
        local_idx = r["index"]
        orig_idx = faiss_top[local_idx]
        reranked_indices.append(orig_idx)
        reranked_texts.append(candidates_text[orig_idx])
        reranked_embs_list.append(candidate_embs[orig_idx])
        reranked_tokens.append(token_counts[orig_idx])

    reranked_embs = np.array(reranked_embs_list)

    # Step 3: Vote-K on reranked candidates
    votek_fn = modified_votek_simple if use_simple else modified_votek
    local_selected, used = votek_fn(
        question_emb, reranked_embs, reranked_texts, reranked_tokens,
        budget_B=budget_B, k=k, rho=rho, min_tokens=min_tokens,
    )

    # Step 4: Map back to original indices
    selected_indices = [reranked_indices[i] for i in local_selected]
    selected_texts = [candidates_text[i] for i in selected_indices]

    return selected_indices, selected_texts, used
