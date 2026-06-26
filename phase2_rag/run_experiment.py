"""
实验主入口：跑完整的三管线对比实验

用法:
    cd icl-selective-annotation
    set GITEE_API_KEY=你的key
    python -m phase2_rag.run_experiment --sample_size 50
    python -m phase2_rag.run_experiment --sample_size 500
"""

import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase2_rag.config import BUDGET_TOKENS, VOTEK_K, VOTEK_RHO, MIN_TOKENS
from phase2_rag.data_loader import load_hotpotqa
from phase2_rag.embedding_client import EmbeddingClient
from phase2_rag.pipeline_topk import run_pipeline_topk
from phase2_rag.pipeline_reranker import run_pipeline_reranker
from phase2_rag.pipeline_votek import run_pipeline_votek
from phase2_rag.pipeline_hybrid import run_pipeline_hybrid
from phase2_rag.evaluator import ExperimentEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="RAG Benchmark Experiment")
    parser.add_argument("--sample_size", type=int, default=50)
    parser.add_argument("--budget", type=int, default=BUDGET_TOKENS)
    parser.add_argument("--skip_llm", action="store_true",
                        help="跳过 LLM（只评测 Coverage 等指标）")
    parser.add_argument("--skip_reranker", action="store_true",
                        help="跳过 Reranker 管线（节省 API 额度）")
    parser.add_argument("--output_dir", type=str, default="phase2_rag/results")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--votek_k", type=int, default=VOTEK_K,
                        help="Vote-K 投票邻居数")
    parser.add_argument("--votek_rho", type=float, default=VOTEK_RHO,
                        help="Vote-K 衰减系数")
    parser.add_argument("--min_tokens", type=int, default=MIN_TOKENS,
                        help="最小句子长度阈值")
    return parser.parse_args()


def run_experiment(args):
    start_time = time.time()

    print("\n" + "=" * 80)
    print("  RAG Benchmark: Modified Vote-K vs Top-k vs Reranker")
    print(f"  samples={args.sample_size}, budget={args.budget}, skip_llm={args.skip_llm}")
    print("=" * 80 + "\n")

    # 1. 加载数据
    examples = load_hotpotqa(sample_size=args.sample_size)

    # 2. 初始化客户端
    emb_client = EmbeddingClient()
    os.makedirs(args.output_dir, exist_ok=True)
    if not args.no_cache:
        emb_client.enable_cache(os.path.join(args.output_dir, "embedding_cache.json"))

    llm_client = None
    if not args.skip_llm:
        from phase2_rag.llm_client import LLMClient
        llm_client = LLMClient()

    # 3. 评测器
    evaluator = ExperimentEvaluator()

    # 4. 逐题运行
    for i, example in enumerate(examples):
        print(f"\n[{i+1}/{len(examples)}] Q: {example.question}")
        print(f"  A: {example.answer} | candidates: {len(example.candidates)} | gold: {len(example.gold_facts)}")

        # 编码
        question_emb = emb_client.encode(example.question)
        candidate_embs = emb_client.encode_batch(example.candidate_texts)
        token_counts = example.get_candidate_token_counts()

        # 三条管线
        topk_idx, topk_txt, topk_tok = run_pipeline_topk(
            example.question, question_emb, candidate_embs,
            example.candidate_texts, token_counts, budget_B=args.budget)

        if not args.skip_reranker:
            rerank_idx, rerank_txt, rerank_tok = run_pipeline_reranker(
                example.question, question_emb, candidate_embs,
                example.candidate_texts, token_counts, budget_B=args.budget)

        votek_idx, votek_txt, votek_tok = run_pipeline_votek(
            example.question, question_emb, candidate_embs,
            example.candidate_texts, token_counts, budget_B=args.budget,
            k=args.votek_k, rho=args.votek_rho, min_tokens=args.min_tokens)

        # 混合管线 (Reranker + Vote-K)
        if not args.skip_reranker:
            hybrid_idx, hybrid_txt, hybrid_tok = run_pipeline_hybrid(
                example.question, question_emb, candidate_embs,
                example.candidate_texts, token_counts, budget_B=args.budget,
                k=args.votek_k, rho=args.votek_rho, min_tokens=args.min_tokens)

        # 评测
        pipelines = [
            ("topk", topk_idx, topk_txt, topk_tok),
            ("votek", votek_idx, votek_txt, votek_tok),
        ]
        if not args.skip_reranker:
            pipelines.insert(1, ("reranker", rerank_idx, rerank_txt, rerank_tok))
            pipelines.append(("hybrid", hybrid_idx, hybrid_txt, hybrid_tok))

        for name, indices, texts, tokens in pipelines:
            selected_embs = candidate_embs[indices] if len(indices) > 0 else None
            prediction = ""
            if llm_client and texts:
                try:
                    prediction = llm_client.generate_answer(
                        example.question, "\n".join(texts))
                except Exception as e:
                    print(f"    [LLM] 失败: {e}")
                    prediction = "unknown"

            evaluator.evaluate_single(
                pipeline_name=name, prediction=prediction,
                ground_truth=example.answer, selected_texts=texts,
                gold_texts=example.gold_texts, selected_embs=selected_embs,
                used_tokens=tokens)

        cov_t = evaluator.results["topk"]["coverage"][-1]
        cov_v = evaluator.results["votek"]["coverage"][-1]
        line = f"  Top-k cov={cov_t:.2f}"
        if not args.skip_reranker:
            cov_r = evaluator.results["reranker"]["coverage"][-1]
            cov_h = evaluator.results["hybrid"]["coverage"][-1]
            line += f" | Reranker cov={cov_r:.2f}"
            line += f" | Hybrid cov={cov_h:.2f}"
        line += f" | Vote-K cov={cov_v:.2f}"
        print(line)

    # 5. 汇总
    evaluator.print_summary()
    evaluator.save_results(os.path.join(args.output_dir, "experiment_results.json"))

    elapsed = time.time() - start_time
    print(f"\n实验完成！耗时 {elapsed:.1f}s ({elapsed/60:.1f}min)")
    return evaluator.get_summary()


if __name__ == "__main__":
    run_experiment(parse_args())
