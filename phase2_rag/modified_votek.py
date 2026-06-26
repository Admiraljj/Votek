"""
魔改 Vote-K 核心算法：性价比最优的上下文选择

将原始 Vote-K（选代表）改造为 RAG 场景下的（选干货）：
- 综合价值 = 提问相关性 x 代表性（含衰减）
- 性价比 = 综合价值 / Token 数
- 贪心选择直到 Token 预算用完

参考:
- 原始 Vote-K: Selective Annotation Makes LMs Better Few-Shot Learners
- Knapsack 改进: Tang et al., 2021
"""

import numpy as np
from phase2_rag.config import VOTEK_K, VOTEK_RHO, MIN_TOKENS, BUDGET_TOKENS


def cosine_similarity_matrix(A, B=None):
    """
    计算余弦相似度矩阵

    Args:
        A: (n, d) numpy array
        B: (m, d) numpy array (可选，默认 B=A)

    Returns:
        sim: (n, m) 余弦相似度矩阵
    """
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
    if B is None:
        return A_norm @ A_norm.T
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    return A_norm @ B_norm.T


def modified_votek(
    question_emb,
    candidate_embs,
    candidates_text,
    token_counts,
    budget_B=BUDGET_TOKENS,
    k=VOTEK_K,
    rho=VOTEK_RHO,
    min_tokens=MIN_TOKENS,
    top_k_indices=None,
):
    """
    魔改 Vote-K：性价比最优的上下文选择

    核心公式:
        Value(u) = sim(u, q) * sum(rho^(-represent_count[v])) for v in neighbors(u)
        目标: argmax Value(u) / token_count(u)

    Args:
        question_emb: 问题嵌入 (d,)
        candidate_embs: 候选句子嵌入 (n, d)
        candidates_text: 候选句子文本列表
        token_counts: 每个句子的 token 数
        budget_B: 总 token 预算
        k: 投票邻居数
        rho: 衰减系数（>1，值越大越惩罚已覆盖区域）
        min_tokens: 最小句子长度阈值（防碎片化）
        top_k_indices: FAISS 召回的候选索引（可选，用于限制候选范围）

    Returns:
        selected_indices: 被选中的候选索引列表
        used_budget: 使用的 token 预算
    """
    # 如果有 FAISS 召回的限制范围，只用那些候选
    if top_k_indices is not None:
        candidate_embs = candidate_embs[top_k_indices]
        working_text = [candidates_text[i] if i < len(candidates_text) else "" for i in top_k_indices]
        working_tokens = [token_counts[i] if i < len(token_counts) else 1 for i in top_k_indices]
    else:
        working_text = candidates_text
        working_tokens = token_counts

    n = len(working_text)
    if n == 0:
        return [], 0

    # 1. 计算与问题的相关性 sim(u, q)
    question_norm = question_emb / (np.linalg.norm(question_emb) + 1e-10)
    cand_norms = candidate_embs / (np.linalg.norm(candidate_embs, axis=1, keepdims=True) + 1e-10)
    relevance = (cand_norms @ question_norm).astype(np.float64)

    # 2. 构建邻居图：每个节点找 top-k 最相似的邻居
    sim_matrix = cosine_similarity_matrix(candidate_embs.astype(np.float64))
    np.fill_diagonal(sim_matrix, -np.inf)
    actual_k = min(k, n - 1)
    neighbor_graph = np.argsort(-sim_matrix, axis=1)[:, :actual_k]

    # 3. 初始化被代表次数
    represent_count = np.zeros(n, dtype=np.float64)

    # 4. 贪心选择
    selected = []
    used_budget = 0
    remaining = set(range(n))
    valid_length = np.array([working_tokens[i] >= min_tokens for i in range(n)])

    while remaining:
        best_score = -1.0
        best_idx = -1

        for u in remaining:
            if not valid_length[u]:
                continue
            if used_budget + working_tokens[u] > budget_B:
                continue

            neighbors = neighbor_graph[u]
            rep_value = np.sum(rho ** (-represent_count[neighbors]))
            value_u = relevance[u] * rep_value
            cost_effectiveness = value_u / working_tokens[u]

            if cost_effectiveness > best_score:
                best_score = cost_effectiveness
                best_idx = u

        if best_idx == -1:
            break

        selected.append(best_idx)
        used_budget += working_tokens[best_idx]
        remaining.discard(best_idx)
        represent_count[neighbor_graph[best_idx]] += 1

    # 5. 防碎片化补丁
    selected = _anti_fragmentation_patch(
        selected, relevance, represent_count, neighbor_graph,
        working_tokens, budget_B, rho, remaining, valid_length
    )

    # 映射回原始索引
    if top_k_indices is not None:
        mapped = [top_k_indices[i] for i in selected]
        return mapped, used_budget

    return selected, used_budget


def _anti_fragmentation_patch(
    selected, relevance, represent_count, neighbor_graph,
    token_counts, budget_B, rho, remaining, valid_length
):
    """防碎片化补丁：对比碎片集合 vs 高价值大文档"""
    if len(selected) <= 1:
        return selected

    def compute_total_value(indices):
        total = 0.0
        temp_represent = np.zeros_like(represent_count)
        for idx in indices:
            neighbors = neighbor_graph[idx]
            rep_value = np.sum(rho ** (-temp_represent[neighbors]))
            total += relevance[idx] * rep_value
            temp_represent[neighbors] += 1
        return total

    current_value = compute_total_value(selected)
    best_replacement = None
    best_replacement_value = current_value

    for u in remaining:
        if not valid_length[u]:
            continue
        if token_counts[u] > budget_B:
            continue

        single_value = relevance[u] * np.sum(rho ** (-represent_count[neighbor_graph[u]]))
        if token_counts[u] <= budget_B and single_value > current_value * 0.5:
            trial = [u]
            trial_tokens = token_counts[u]
            for s in selected:
                if s != u and trial_tokens + token_counts[s] <= budget_B:
                    overlap = len(set(neighbor_graph[s]) & set(neighbor_graph[u]))
                    if overlap < len(neighbor_graph[s]) // 2:
                        trial.append(s)
                        trial_tokens += token_counts[s]

            trial_value = compute_total_value(trial)
            if trial_value > best_replacement_value:
                best_replacement = trial
                best_replacement_value = trial_value

    if best_replacement is not None and best_replacement_value > current_value:
        return best_replacement

    return selected


def modified_votek_simple(
    question_emb,
    candidate_embs,
    candidates_text,
    token_counts,
    budget_B=BUDGET_TOKENS,
    k=VOTEK_K,
    rho=VOTEK_RHO,
    min_tokens=MIN_TOKENS,
):
    """
    简化版魔改 Vote-K（不做防碎片化补丁，速度更快）

    Returns:
        selected_indices: 被选中的候选索引列表
        used_budget: 使用的 token 预算
    """
    n = len(candidates_text)
    if n == 0:
        return [], 0

    question_norm = question_emb / (np.linalg.norm(question_emb) + 1e-10)
    cand_norms = candidate_embs / (np.linalg.norm(candidate_embs, axis=1, keepdims=True) + 1e-10)
    relevance = (cand_norms @ question_norm).astype(np.float64)

    sim_matrix = cosine_similarity_matrix(candidate_embs.astype(np.float64))
    np.fill_diagonal(sim_matrix, -np.inf)
    actual_k = min(k, n - 1)
    neighbor_graph = np.argsort(-sim_matrix, axis=1)[:, :actual_k]

    represent_count = np.zeros(n, dtype=np.float64)
    selected = []
    used_budget = 0
    remaining = set(range(n))
    valid_length = np.array([token_counts[i] >= min_tokens for i in range(n)])

    while remaining:
        best_score = -1.0
        best_idx = -1

        for u in remaining:
            if not valid_length[u]:
                continue
            if used_budget + token_counts[u] > budget_B:
                continue

            neighbors = neighbor_graph[u]
            rep_value = np.sum(rho ** (-represent_count[neighbors]))
            value_u = relevance[u] * rep_value
            score = value_u / token_counts[u]

            if score > best_score:
                best_score = score
                best_idx = u

        if best_idx == -1:
            break

        selected.append(best_idx)
        used_budget += token_counts[best_idx]
        remaining.discard(best_idx)
        represent_count[neighbor_graph[best_idx]] += 1

    return selected, used_budget
