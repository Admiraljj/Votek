"""
Reranker API 客户端：封装 Qwen3-Reranker-4B 调用

功能：
- 对候选文档重排序
- 自动重试 + exponential backoff
- 支持大批量文档分批 rerank
"""

import time
import requests
from phase2_rag.config import (
    GITEE_API_BASE, GITEE_API_KEY,
    RERANKER_MODEL, RERANKER_INSTRUCTION,
    API_MAX_RETRIES, API_RETRY_DELAY, RERANKER_BATCH_SIZE
)


class RerankerClient:
    """Qwen3-Reranker-4B API 客户端"""

    def __init__(self, api_key=None):
        self.api_key = api_key or GITEE_API_KEY
        if not self.api_key:
            raise ValueError(
                "GITEE_API_KEY 未设置！请运行:\n"
                "  Windows: set GITEE_API_KEY=你的key\n"
                "  Linux/Mac: export GITEE_API_KEY=你的key"
            )
        self.api_url = f"{GITEE_API_BASE}/rerank"
        self.headers = {
            "X-Failover-Enabled": "true",
            "Authorization": f"Bearer {self.api_key}"
        }

    def rerank(
        self,
        query: str,
        documents: list,
        top_n=None,
        instruction=None,
    ):
        """
        对候选文档重排序

        Args:
            query: 查询文本（问题）
            documents: 候选文档列表
            top_n: 只返回前 N 个结果（None 表示返回全部）
            instruction: 自定义排序指令

        Returns:
            results: 按 relevance_score 降序排列
                [{"index": int, "relevance_score": float}, ...]
        """
        import re
        # 清理文本中的特殊标签
        clean_docs = [re.sub(r'</?(input|output|system|user|assistant)>', '', d) for d in documents]
        query = re.sub(r'</?(input|output|system|user|assistant)>', '', query)

        if len(clean_docs) > RERANKER_BATCH_SIZE:
            return self._rerank_large_batch(query, documents, top_n, instruction)

        payload = {
            "query": query,
            "documents": clean_docs,
            "model": RERANKER_MODEL,
            "instruction": instruction or RERANKER_INSTRUCTION,
        }
        if top_n is not None:
            payload["top_n"] = top_n

        for attempt in range(API_MAX_RETRIES):
            try:
                response = requests.post(
                    self.api_url, headers=self.headers, json=payload, timeout=60
                )
                response.raise_for_status()
                data = response.json()

                results = data.get("results", [])
                results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
                return results

            except requests.exceptions.RequestException as e:
                delay = API_RETRY_DELAY * (2 ** attempt)
                print(f"[Reranker] API 调用失败 (尝试 {attempt+1}/{API_MAX_RETRIES}): {e}")
                if attempt < API_MAX_RETRIES - 1:
                    time.sleep(delay)
                else:
                    raise RuntimeError(f"Reranker API 调用失败，已重试 {API_MAX_RETRIES} 次") from e

    def _rerank_large_batch(
        self,
        query: str,
        documents: list,
        top_n=None,
        instruction=None,
    ):
        """处理超过 RERANKER_BATCH_SIZE 的文档：分批 rerank，合并结果"""
        all_results = []
        for start in range(0, len(documents), RERANKER_BATCH_SIZE):
            batch_docs = documents[start:start + RERANKER_BATCH_SIZE]
            batch_results = self.rerank(query, batch_docs, instruction=instruction)

            for r in batch_results:
                r["index"] = r["index"] + start
                all_results.append(r)

        all_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        if top_n is not None:
            return all_results[:top_n]
        return all_results

    def rerank_with_texts(
        self,
        query: str,
        documents: list,
        top_n=None,
    ):
        """
        重排序并返回 (原始index, 分数, 文档文本)

        Returns:
            results: [(orig_index, relevance_score, doc_text), ...]
        """
        ranked = self.rerank(query, documents, top_n=top_n)
        return [
            (r["index"], r.get("relevance_score", 0.0), documents[r["index"]])
            for r in ranked
        ]


# 便捷函数
def rerank_documents(query: str, documents: list, top_n=None):
    """
    快速对文档重排序

    Returns:
        [{"index": int, "relevance_score": float}, ...]
    """
    client = RerankerClient()
    return client.rerank(query, documents, top_n=top_n)
