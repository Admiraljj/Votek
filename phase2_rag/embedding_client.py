"""
Embedding API 客户端：封装 Qwen3-Embedding-8B 调用

功能：
- 单条/批量文本编码
- 自动重试 + exponential backoff
- 结果缓存（避免重复调用 API）
"""

import time
import hashlib
import json
import os
import numpy as np
from openai import OpenAI
from phase2_rag.config import (
    GITEE_API_BASE, GITEE_API_KEY,
    EMBEDDING_MODEL, EMBEDDING_DIM, EMBEDDING_INSTRUCTION,
    API_MAX_RETRIES, API_RETRY_DELAY, EMBEDDING_BATCH_SIZE
)


class EmbeddingClient:
    """Qwen3-Embedding-8B API 客户端"""

    def __init__(self, api_key=None):
        self.api_key = api_key or GITEE_API_KEY
        if not self.api_key:
            raise ValueError(
                "GITEE_API_KEY 未设置！请运行:\n"
                "  Windows: set GITEE_API_KEY=你的key\n"
                "  Linux/Mac: export GITEE_API_KEY=你的key"
            )
        self.client = OpenAI(
            base_url=GITEE_API_BASE,
            api_key=self.api_key,
            default_headers={"X-Failover-Enabled": "true"},
        )
        self._cache = {}  # text_hash -> embedding
        self._cache_file = None

    def enable_cache(self, cache_file="embedding_cache.json"):
        """启用磁盘缓存，避免重复调用 API"""
        self._cache_file = cache_file
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
            print(f"[Embedding] 从缓存加载了 {len(self._cache)} 条记录")

    def _save_cache(self):
        """保存缓存到磁盘（原子写入，防止中断导致文件损坏）"""
        if self._cache_file:
            import tempfile
            tmp_path = self._cache_file + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f)
            # 原子替换
            import os
            os.replace(tmp_path, self._cache_file)

    @staticmethod
    def _text_hash(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def encode(self, text: str) -> np.ndarray:
        """
        编码单条文本

        Args:
            text: 输入文本

        Returns:
            embedding: numpy array, shape (dim,)
        """
        text = self._sanitize(text)
        h = self._text_hash(text)
        if h in self._cache:
            return np.array(self._cache[h], dtype=np.float32)

        for attempt in range(API_MAX_RETRIES):
            try:
                response = self.client.embeddings.create(
                    input=text,
                    model=EMBEDDING_MODEL,
                    dimensions=EMBEDDING_DIM,
                    extra_body={"instruction": EMBEDDING_INSTRUCTION},
                )
                embedding = response.data[0].embedding
                result = np.array(embedding, dtype=np.float32)

                self._cache[h] = embedding
                # 不在此处保存，由 encode_batch 统一保存

                return result

            except Exception as e:
                delay = API_RETRY_DELAY * (2 ** attempt)
                print(f"[Embedding] API 调用失败 (尝试 {attempt+1}/{API_MAX_RETRIES}): {e}")
                if attempt < API_MAX_RETRIES - 1:
                    time.sleep(delay)
                else:
                    raise RuntimeError(f"Embedding API 调用失败，已重试 {API_MAX_RETRIES} 次") from e

    @staticmethod
    def _sanitize(text):
        """清理文本中的特殊字符，避免 API 报错"""
        import re
        # 移除可能导致 API 验证失败的标签
        text = re.sub(r'</?(input|output|system|user|assistant)>', '', text)
        # 截断过长文本
        if len(text) > 2000:
            text = text[:2000]
        return text.strip()

    def encode_batch(self, texts: list, show_progress=False) -> np.ndarray:
        """
        批量编码文本（自动降级：批量失败则逐条编码）

        Args:
            texts: 文本列表
            show_progress: 是否显示进度

        Returns:
            embeddings: numpy array, shape (n, dim)
        """
        embeddings = []
        total = len(texts)

        # 先找出未缓存的
        uncached_indices = []
        uncached_texts = []
        for i, text in enumerate(texts):
            clean = self._sanitize(text)
            h = self._text_hash(clean)
            if h in self._cache:
                embeddings.append((i, np.array(self._cache[h], dtype=np.float32)))
            else:
                uncached_indices.append(i)
                uncached_texts.append(clean)

        if uncached_texts:
            print(f"[Embedding] 需要编码 {len(uncached_texts)}/{total} 条文本 "
                  f"（{total - len(uncached_texts)} 条命中缓存）")

        # 逐条编码（批量 API 对某些文本不稳定）
        for j, (text, orig_idx) in enumerate(zip(uncached_texts, uncached_indices)):
            emb = self.encode(text)
            embeddings.append((orig_idx, emb))
            if (j + 1) % 20 == 0:
                self._save_cache()
                print(f"[Embedding] 进度: {j+1}/{len(uncached_texts)}")

        # 最终保存
        self._save_cache()

        # 按原始顺序排列
        embeddings.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in embeddings], dtype=np.float32)

    def _batch_encode(self, uncached_texts, uncached_indices, embeddings):
        """批量调用 API 编码"""
        for batch_start in range(0, len(uncached_texts), EMBEDDING_BATCH_SIZE):
            batch_texts = uncached_texts[batch_start:batch_start + EMBEDDING_BATCH_SIZE]
            batch_indices = uncached_indices[batch_start:batch_start + EMBEDDING_BATCH_SIZE]

            for attempt in range(API_MAX_RETRIES):
                try:
                    response = self.client.embeddings.create(
                        input=batch_texts,
                        model=EMBEDDING_MODEL,
                        dimensions=EMBEDDING_DIM,
                        extra_body={"instruction": EMBEDDING_INSTRUCTION},
                    )
                    for item in response.data:
                        emb = item.embedding
                        idx_in_batch = item.index
                        orig_idx = batch_indices[idx_in_batch]
                        emb_array = np.array(emb, dtype=np.float32)
                        embeddings.append((orig_idx, emb_array))

                        h = self._text_hash(batch_texts[idx_in_batch])
                        self._cache[h] = emb

                    self._save_cache()

                    done = batch_start + len(batch_texts)
                    if done % 100 < EMBEDDING_BATCH_SIZE:
                        print(f"[Embedding] 进度: {done}/{len(uncached_texts)}")
                    break

                except Exception as e:
                    delay = API_RETRY_DELAY * (2 ** attempt)
                    print(f"[Embedding] 批量调用失败 (尝试 {attempt+1}/{API_MAX_RETRIES}): {e}")
                    if attempt < API_MAX_RETRIES - 1:
                        time.sleep(delay)
                    else:
                        raise RuntimeError(f"Embedding API 批量调用失败") from e

        # 按原始顺序排列
        embeddings.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in embeddings], dtype=np.float32)


# 便捷函数
def get_embeddings(texts: list, use_cache=True) -> np.ndarray:
    """
    快速获取文本嵌入

    Args:
        texts: 文本列表
        use_cache: 是否使用缓存

    Returns:
        embeddings: (n, dim) numpy array
    """
    client = EmbeddingClient()
    if use_cache:
        client.enable_cache()
    return client.encode_batch(texts, show_progress=True)
