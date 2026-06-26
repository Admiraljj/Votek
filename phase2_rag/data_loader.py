"""
HotpotQA 数据加载器

功能：
- 从 HuggingFace 加载 HotpotQA distractor 数据
- 将每题的 context 拆分成句子级别的候选池
- 提取金标准事实（Gold Facts）
- 支持采样指定数量的题目
"""

import random
from datasets import load_dataset
from phase2_rag.config import (
    HOTPOTQA_DATASET, HOTPOTQA_CONFIG, HOTPOTQA_SPLIT, EVAL_SAMPLE_SIZE
)


class HotpotQAExample:
    """单条 HotpotQA 数据"""

    def __init__(self, raw_example):
        self.id = raw_example["id"]
        self.question = raw_example["question"]
        self.answer = raw_example["answer"]
        self.type = raw_example["type"]
        self.level = raw_example["level"]

        # 候选句子池：从 context 中提取所有句子
        # 每个元素: {"article_title", "article_idx", "sent_idx", "text", "raw_sentence"}
        self.candidates = []
        self.candidate_texts = []  # 纯文本列表，用于 embedding

        for art_idx, (title, sentences) in enumerate(
            zip(raw_example["context"]["title"], raw_example["context"]["sentences"])
        ):
            for sent_idx, sentence in enumerate(sentences):
                # 候选文本 = 文章标题 + 句子（保留上下文信息）
                candidate_text = f"{title}: {sentence}"
                self.candidates.append({
                    "article_title": title,
                    "article_idx": art_idx,
                    "sent_idx": sent_idx,
                    "text": candidate_text,
                    "raw_sentence": sentence,
                })
                self.candidate_texts.append(candidate_text)

        # 金标准事实
        self.gold_facts = []
        self.gold_texts = set()

        for title, sent_id in zip(
            raw_example["supporting_facts"]["title"],
            raw_example["supporting_facts"]["sent_id"]
        ):
            ctx_idx = raw_example["context"]["title"].index(title)
            gold_sentence = raw_example["context"]["sentences"][ctx_idx][sent_id]
            self.gold_facts.append({
                "article_title": title,
                "sent_idx": sent_id,
                "text": f"{title}: {gold_sentence}",
                "raw_sentence": gold_sentence,
            })
            self.gold_texts.add(f"{title}: {gold_sentence}")

    def count_tokens(self, text):
        """估算 token 数（英文约 1.3 token/word）"""
        words = len(text.split())
        return max(1, int(words * 1.3))

    def get_candidate_token_counts(self):
        """获取所有候选句子的 token 数"""
        return [self.count_tokens(t) for t in self.candidate_texts]


def load_hotpotqa(
    split=None,
    sample_size=None,
    seed=42,
    question_types=None,
):
    """
    加载 HotpotQA 数据

    Args:
        split: 数据集分割（默认 validation）
        sample_size: 采样数量（默认 500）
        seed: 随机种子
        question_types: 过滤问题类型，如 ["bridge", "comparison"]

    Returns:
        examples: HotpotQAExample 列表
    """
    split = split or HOTPOTQA_SPLIT
    sample_size = sample_size or EVAL_SAMPLE_SIZE

    print(f"[DataLoader] 加载 HotpotQA ({HOTPOTQA_DATASET}, config={HOTPOTQA_CONFIG}, split={split})...")

    dataset = load_dataset(HOTPOTQA_DATASET, HOTPOTQA_CONFIG, split=split)

    # 转换为列表
    all_data = list(dataset)
    print(f"[DataLoader] 共 {len(all_data)} 条数据")

    # 按类型过滤
    if question_types:
        all_data = [d for d in all_data if d["type"] in question_types]
        print(f"[DataLoader] 过滤类型 {question_types} 后: {len(all_data)} 条")

    # 采样
    if sample_size and sample_size < len(all_data):
        random.seed(seed)
        all_data = random.sample(all_data, sample_size)
        print(f"[DataLoader] 随机采样 {sample_size} 条")

    # 转换为 HotpotQAExample
    examples = []
    for raw in all_data:
        examples.append(HotpotQAExample(raw))

    # 统计信息
    type_counts = {}
    for ex in examples:
        type_counts[ex.type] = type_counts.get(ex.type, 0) + 1

    avg_candidates = sum(len(ex.candidates) for ex in examples) / len(examples)
    avg_golds = sum(len(ex.gold_facts) for ex in examples) / len(examples)

    print(f"[DataLoader] 加载完成:")
    print(f"  题目数: {len(examples)}")
    print(f"  类型分布: {type_counts}")
    print(f"  平均候选句子数: {avg_candidates:.1f}")
    print(f"  平均金标准事实数: {avg_golds:.1f}")

    return examples


def compute_evidence_coverage(selected_texts, gold_texts):
    """
    计算金标准事实覆盖率

    Args:
        selected_texts: 被选中的候选文本列表
        gold_texts: 金标准文本集合 (set)

    Returns:
        coverage: 覆盖率 (0.0 ~ 1.0)
        covered: 被覆盖的金标准句子列表
        missed: 未被覆盖的金标准句子列表
    """
    selected_set = set(selected_texts)
    covered = [g for g in gold_texts if g in selected_set]
    missed = [g for g in gold_texts if g not in selected_set]

    coverage = len(covered) / len(gold_texts) if gold_texts else 0.0
    return coverage, covered, missed


def compute_context_redundancy(embeddings):
    """
    计算上下文冗余度：两两之间的平均余弦相似度

    Args:
        embeddings: (n, d) numpy array

    Returns:
        redundancy: 平均余弦相似度（越低越好）
    """
    import numpy as np

    if len(embeddings) < 2:
        return 0.0

    # 归一化
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = embeddings / norms

    # 两两余弦相似度
    sim_matrix = normed @ normed.T

    # 取上三角（不含对角线）的平均值
    n = len(embeddings)
    mask = np.triu(np.ones((n, n)), k=1).astype(bool)
    redundancy = sim_matrix[mask].mean()

    return float(redundancy)
