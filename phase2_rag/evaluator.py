"""
评测模块：计算所有实验指标

维度一：生成质量 - F1, EM, Evidence Coverage
维度二：冗余度与信息密度 - Redundancy, F1/Tokens, Efficiency
"""

import re
import json
import numpy as np
from collections import Counter


def normalize_answer(s):
    """标准化答案文本（HotpotQA 风格）"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return re.sub(r'[^\w\s]', '', text)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(prediction, ground_truth):
    """计算单条 F1-Score（词级别）"""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_em(prediction, ground_truth):
    """计算 Exact Match"""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_evidence_coverage(selected_texts, gold_texts):
    """计算金标准事实覆盖率"""
    if not gold_texts:
        return 0.0, 0, 0
    gold_set = set(gold_texts) if not isinstance(gold_texts, set) else gold_texts
    selected_set = set(selected_texts)
    covered = gold_set & selected_set
    return len(covered) / len(gold_set), len(covered), len(gold_set)


def compute_context_redundancy(embeddings):
    """计算上下文冗余度：两两之间的平均余弦相似度"""
    if len(embeddings) < 2:
        return 0.0
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = embeddings / norms
    sim_matrix = normed @ normed.T
    n = len(embeddings)
    mask = np.triu(np.ones((n, n)), k=1).astype(bool)
    return float(sim_matrix[mask].mean())


class ExperimentEvaluator:
    """实验评测器"""

    PIPELINE_NAMES = ["topk", "reranker", "votek", "hybrid"]
    METRIC_KEYS = ["f1", "em", "coverage", "redundancy", "tokens", "evidence_efficiency"]

    def __init__(self):
        self.results = {
            name: {k: [] for k in self.METRIC_KEYS}
            for name in self.PIPELINE_NAMES
        }

    def evaluate_single(self, pipeline_name, prediction, ground_truth,
                        selected_texts, gold_texts, selected_embs=None, used_tokens=0):
        """评测单条样本"""
        f1 = compute_f1(prediction, ground_truth)
        em = compute_em(prediction, ground_truth)
        coverage, _, _ = compute_evidence_coverage(selected_texts, gold_texts)
        redundancy = 0.0
        if selected_embs is not None and len(selected_embs) > 1:
            redundancy = compute_context_redundancy(selected_embs)
        eff = coverage / used_tokens * 1000 if used_tokens > 0 else 0.0

        r = self.results[pipeline_name]
        r["f1"].append(f1)
        r["em"].append(em)
        r["coverage"].append(coverage)
        r["redundancy"].append(redundancy)
        r["tokens"].append(used_tokens)
        r["evidence_efficiency"].append(eff)

    def get_summary(self):
        """获取汇总结果"""
        summary = {}
        for name in self.PIPELINE_NAMES:
            metrics = self.results[name]
            n = len(metrics["f1"])
            if n == 0:
                continue
            avg_f1 = np.mean(metrics["f1"])
            avg_tokens = np.mean(metrics["tokens"])
            summary[name] = {
                "count": n,
                "F1": round(avg_f1, 4),
                "EM": round(np.mean(metrics["em"]), 4),
                "Evidence Coverage": round(np.mean(metrics["coverage"]), 4),
                "Context Redundancy": round(np.mean(metrics["redundancy"]), 4),
                "Avg Tokens": round(avg_tokens, 1),
                "F1 per 1k Tokens": round(avg_f1 / avg_tokens * 1000 if avg_tokens else 0, 6),
                "Evidence Efficiency (per 1k tokens)": round(np.mean(metrics["evidence_efficiency"]), 6),
            }
        return summary

    def print_summary(self):
        """打印汇总表格（支持 4 管线）"""
        summary = self.get_summary()
        has_hybrid = "hybrid" in summary
        W = 14

        sep_len = 37 + W * (5 if has_hybrid else 4)
        print("\n" + "=" * sep_len)
        print("  实验结果汇总")
        print("=" * sep_len)

        display_order = ["topk", "reranker", "hybrid", "votek"] if has_hybrid else ["topk", "reranker", "votek"]
        header_names = {"topk": "Top-k (A)", "reranker": "Reranker (B)", "hybrid": "Hybrid (D)", "votek": "Vote-K (C)"}
        print(f"{'指标':<35}" + "".join(f" {header_names[n]:>{W}}" for n in display_order))
        print("-" * sep_len)

        labels = {
            "count": "样本数", "F1": "F1-Score", "EM": "Exact Match",
            "Evidence Coverage": "Evidence Coverage",
            "Context Redundancy": "Context Redundancy",
            "Avg Tokens": "Avg Tokens Used",
            "F1 per 1k Tokens": "F1 per 1k Tokens",
            "Evidence Efficiency (per 1k tokens)": "Evidence Eff. per 1k Tokens",
        }
        for metric, label in labels.items():
            vals = []
            for name in display_order:
                if name in summary:
                    v = summary[name].get(metric, "N/A")
                    vals.append(f"{v:>{W}}")
                else:
                    vals.append(f"{'N/A':>{W}}")
            print(f"  {label:<33}" + " ".join(vals))

        print("-" * sep_len)

        # Pairwise comparisons
        if "hybrid" in summary and "reranker" in summary:
            h, r = summary["hybrid"], summary["reranker"]
            cov = h["Evidence Coverage"] - r["Evidence Coverage"]
            red = h["Context Redundancy"] - r["Context Redundancy"]
            eff = h["Evidence Efficiency (per 1k tokens)"] - r["Evidence Efficiency (per 1k tokens)"]
            print(f"\n  Hybrid (D) vs Reranker (B):")
            print(f"    Coverage: {'+' if cov>=0 else ''}{cov:.4f} | Redundancy: {'+' if red>=0 else ''}{red:.4f} (负=更好) | Eff/1k: {'+' if eff>=0 else ''}{eff:.4f}")
        if "hybrid" in summary and "votek" in summary:
            h, v = summary["hybrid"], summary["votek"]
            cov = h["Evidence Coverage"] - v["Evidence Coverage"]
            red = h["Context Redundancy"] - v["Context Redundancy"]
            print(f"  Hybrid (D) vs Vote-K (C):")
            print(f"    Coverage: {'+' if cov>=0 else ''}{cov:.4f} | Redundancy: {'+' if red>=0 else ''}{red:.4f} (负=更好)")
        print("=" * sep_len + "\n")

    def save_results(self, path):
        """保存结果到 JSON"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.get_summary(), f, indent=2, ensure_ascii=False)
        print(f"[Evaluator] 结果已保存到 {path}")
