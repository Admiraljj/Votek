"""
Sprint 1.6 冒烟测试
验证：API 连通性 + FAISS 基本功能 + 数据集加载
"""

import sys
import os
import numpy as np

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_faiss():
    """测试 FAISS 索引构建与搜索"""
    print("=" * 60)
    print("测试 1: FAISS 索引")
    print("=" * 60)

    from phase2_rag.faiss_index import FaissIndex

    # 创建索引
    idx = FaissIndex(dim=8)  # 小维度测试

    # 添加假向量
    vectors = np.random.randn(20, 8).astype(np.float32)
    texts = [f"文本_{i}" for i in range(20)]
    idx.add(vectors, texts)
    print(f"  添加 {idx.ntotal} 条向量 OK")

    # 搜索
    query = np.random.randn(8).astype(np.float32)
    indices, scores = idx.search(query, k=5)
    print(f"  Top-5 搜索结果: indices={indices[0][:3]}..., scores={scores[0][:3]}...")
    top1_idx = int(indices[0][0])
    print(f"  Top-1 文本: {idx._texts[top1_idx]} OK")

    # 保存/加载
    test_path = os.path.join(os.path.dirname(__file__), "test_faiss.bin")
    idx.save(test_path)
    idx2 = FaissIndex(dim=8)
    idx2.load(test_path)
    print(f"  保存/加载: {idx2.ntotal} 条向量, {len(idx2._texts)} 条文本 OK")

    print("  FAISS 测试通过!\n")


def test_embedding_api():
    """测试 Embedding API 连通性"""
    print("=" * 60)
    print("测试 2: Embedding API (Qwen3-Embedding-8B)")
    print("=" * 60)

    if not os.environ.get("GITEE_API_KEY"):
        print("  跳过: GITEE_API_KEY 未设置")
        print("  请运行: set GITEE_API_KEY=你的key\n")
        return

    try:
        from phase2_rag.embedding_client import EmbeddingClient

        client = EmbeddingClient()
        emb = client.encode("Hello, this is a test sentence for embedding.")
        print(f"  返回维度: {emb.shape[0]} (期望 1024)")
        assert emb.shape[0] == 1024, f"维度不对: {emb.shape[0]}"
        print(f"  向量范数: {np.linalg.norm(emb):.4f}")
        print("  Embedding API 测试通过!\n")
    except Exception as e:
        print(f"  Embedding API 测试失败: {e}\n")


def test_reranker_api():
    """测试 Reranker API 连通性"""
    print("=" * 60)
    print("测试 3: Reranker API (Qwen3-Reranker-4B)")
    print("=" * 60)

    if not os.environ.get("GITEE_API_KEY"):
        print("  跳过: GITEE_API_KEY 未设置\n")
        return

    try:
        from phase2_rag.reranker_client import RerankerClient

        client = RerankerClient()
        results = client.rerank(
            query="How to read a CSV file in Python?",
            documents=[
                "Use pandas: import pandas as pd; df = pd.read_csv('data.csv')",
                "You can read CSV files with numpy.loadtxt()",
                "To write JSON files, use json.dump() in Python",
                "CSV means Comma Separated Values."
            ]
        )
        print(f"  返回 {len(results)} 条结果")
        for r in results[:3]:
            score = r.get('relevance_score', 'N/A')
            if isinstance(score, float):
                print(f"    index={r['index']}, score={score:.4f}")
            else:
                print(f"    index={r['index']}, score={score}")
        print("  Reranker API 测试通过!\n")
    except Exception as e:
        print(f"  Reranker API 测试失败: {e}\n")


def test_hotpotqa_load():
    """测试 HotpotQA 数据集加载"""
    print("=" * 60)
    print("测试 4: HotpotQA 数据集加载")
    print("=" * 60)

    try:
        from datasets import load_dataset
        ds = load_dataset(
            "hotpotqa/hotpot_qa", "distractor",
            split="validation", streaming=True
        )
        example = next(iter(ds))

        print(f"  问题: {example['question']}")
        print(f"  答案: {example['answer']}")
        print(f"  类型: {example['type']}")
        print(f"  上下文文章数: {len(example['context']['title'])}")
        print(f"  金标准事实数: {len(example['supporting_facts']['title'])}")

        # 提取金标准句子
        for title, sent_id in zip(
            example["supporting_facts"]["title"],
            example["supporting_facts"]["sent_id"]
        ):
            ctx_idx = example["context"]["title"].index(title)
            gold = example["context"]["sentences"][ctx_idx][sent_id]
            print(f"  金标准: [{title}] {gold[:80]}...")

        print("  HotpotQA 加载测试通过!\n")
    except Exception as e:
        print(f"  HotpotQA 加载失败: {e}")
        print("  可能需要安装: pip install datasets\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  RAG Benchmark Sprint 1 冒烟测试")
    print("=" * 60 + "\n")

    test_faiss()
    test_embedding_api()
    test_reranker_api()
    test_hotpotqa_load()

    print("=" * 60)
    print("  冒烟测试完成!")
    print("=" * 60)
