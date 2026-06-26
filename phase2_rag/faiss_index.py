"""
FAISS 向量索引模块

功能：
- 构建 FAISS IndexFlatIP 索引（余弦相似度 via 内积）
- 添加/搜索向量
- 保存/加载索引到磁盘
"""

import os
import json
import numpy as np
import faiss
from phase2_rag.config import EMBEDDING_DIM, FAISS_TOP_K


class FaissIndex:
    """FAISS 向量索引封装（余弦相似度）"""

    def __init__(self, dim=EMBEDDING_DIM):
        """
        Args:
            dim: 向量维度（默认 1024，对应 Qwen3-Embedding-8B）
        """
        self.dim = dim
        # IndexFlatIP + L2归一化 = 余弦相似度
        self.index = faiss.IndexFlatIP(dim)
        # 存储原始文本，用于结果映射
        self._texts = []

    @property
    def ntotal(self):
        """索引中的向量数量"""
        return self.index.ntotal

    def add(self, vectors: np.ndarray, texts: list = None):
        """
        添加向量到索引

        Args:
            vectors: (n, dim) float32 numpy array
            texts: 对应的文本列表（可选，用于结果映射）
        """
        assert vectors.shape[1] == self.dim, \
            f"向量维度不匹配: 期望 {self.dim}, 得到 {vectors.shape[1]}"

        # L2 归一化（使内积 = 余弦相似度）
        vectors = vectors.astype(np.float32)
        faiss.normalize_L2(vectors)

        self.index.add(vectors)

        if texts is not None:
            assert len(texts) == vectors.shape[0], \
                f"texts 数量 ({len(texts)}) != 向量数量 ({vectors.shape[0]})"
            self._texts.extend(texts)

    def search(self, query_vector: np.ndarray, k: int = FAISS_TOP_K):
        """
        搜索最相似的向量

        Args:
            query_vector: (dim,) 或 (n_queries, dim) float32 array
            k: 返回 top-k 结果

        Returns:
            distances: (n_queries, k) 余弦相似度分数
            indices: (n_queries, k) 索引
        """
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        query_vector = query_vector.astype(np.float32)
        faiss.normalize_L2(query_vector)

        distances, indices = self.index.search(query_vector, k)
        return distances, indices

    def search_with_texts(self, query_vector: np.ndarray, k: int = FAISS_TOP_K):
        """
        搜索并返回 (距离, 索引, 文本)

        Returns:
            results: list of [(index, distance, text), ...] for each query
        """
        distances, indices = self.search(query_vector, k)
        results = []
        for i in range(distances.shape[0]):
            query_results = []
            for j in range(distances.shape[1]):
                idx = int(indices[i][j])
                dist = float(distances[i][j])
                text = self._texts[idx] if idx < len(self._texts) else None
                query_results.append((idx, dist, text))
            results.append(query_results)
        return results

    def save(self, path: str):
        """保存索引到磁盘"""
        faiss.write_index(self.index, path)
        texts_path = path + ".texts.json"
        with open(texts_path, "w", encoding="utf-8") as f:
            json.dump(self._texts, f, ensure_ascii=False)

    def load(self, path: str):
        """从磁盘加载索引"""
        self.index = faiss.read_index(path)
        texts_path = path + ".texts.json"
        if os.path.exists(texts_path):
            with open(texts_path, "r", encoding="utf-8") as f:
                self._texts = json.load(f)

    def get_all_vectors(self):
        """重建并返回索引中的所有向量"""
        vectors = np.array([
            self.index.reconstruct(i) for i in range(self.index.ntotal)
        ], dtype=np.float32)
        return vectors


def build_index_from_embeddings(
    embeddings: np.ndarray,
    texts: list = None,
    save_path: str = None,
) -> FaissIndex:
    """
    从嵌入矩阵快速构建 FAISS 索引

    Args:
        embeddings: (n, dim) numpy array
        texts: 对应文本列表
        save_path: 保存路径（可选）

    Returns:
        FaissIndex 实例
    """
    idx = FaissIndex(dim=embeddings.shape[1])
    idx.add(embeddings, texts)

    if save_path:
        idx.save(save_path)
        print(f"[FAISS] 索引已保存到 {save_path}（{idx.ntotal} 条向量）")

    return idx


def search_top_k(
    index: FaissIndex,
    query_embedding: np.ndarray,
    k: int = FAISS_TOP_K,
):
    """
    便捷搜索函数

    Args:
        index: FaissIndex 实例
        query_embedding: (dim,) 查询向量
        k: top-k

    Returns:
        indices: (k,) 索引数组
        scores: (k,) 余弦相似度分数
    """
    distances, indices = index.search(query_embedding, k)
    return indices[0], distances[0]
