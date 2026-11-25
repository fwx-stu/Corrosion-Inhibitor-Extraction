"""
SiliconFlow API embedding / reranker 封装

提供与 SentenceTransformer / CrossEncoder 兼容的接口，
无需下载本地模型即可使用 BAAI/bge-m3 和 bge-reranker-v2-m3。

接口兼容性：
  SiliconFlowEmbedder.encode()   ←→ SentenceTransformer.encode()
  SiliconFlowReranker.predict()  ←→ CrossEncoder.predict()
  SiliconFlowReranker.compute_score() ←→ FlagReranker.compute_score()
"""

from __future__ import annotations

import time
from typing import List

import numpy as np
import requests

import config


# ============================================================
# Embedding
# ============================================================
class SiliconFlowEmbedder:
    """
    调用硅基流动 /v1/embeddings 接口，返回 numpy 向量矩阵。

    用法:
        enc = SiliconFlowEmbedder()
        vecs = enc.encode(["text1", "text2"], normalize_embeddings=True)
        # vecs.shape == (2, 1024)；支持 .tolist()
    """

    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        batch_size: int = 32,
    ):
        self.model     = model   or config.SILICONFLOW_EMBED_MODEL
        self.api_key   = api_key or config.SILICONFLOW_API_KEY
        self.base_url  = config.SILICONFLOW_BASE_URL
        self.batch_size = batch_size

    # BGE-M3 最大支持 8192 tokens，约 4000 个中英文混合字符
    MAX_TEXT_CHARS = 4000

    def encode(
        self,
        sentences: List[str],
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """编码文本列表，返回 float32 numpy 数组。"""
        if not sentences:
            return np.array([], dtype=np.float32)

        # 截断过长文本，避免 API 400 错误
        sentences = [s[:self.MAX_TEXT_CHARS] if len(s) > self.MAX_TEXT_CHARS else s for s in sentences]

        all_embeddings: List[List[float]] = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i : i + self.batch_size]
            all_embeddings.extend(self._call_api(batch))

        arr = np.array(all_embeddings, dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            arr = arr / norms
        return arr

    def _call_api(self, texts: List[str], retries: int = 5) -> List[List[float]]:
        url     = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model":           self.model,
            "input":           texts,
            "encoding_format": "float",
        }
        for attempt in range(retries):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=60)
                resp.raise_for_status()
                data  = resp.json()
                items = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in items]
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else 0
                if status == 429 and attempt < retries - 1:
                    wait = 10 * (attempt + 1)
                    print(f"  [embedder] 限速 429，{wait}s 后重试 ({attempt+1}/{retries})...", flush=True)
                    time.sleep(wait)
                    continue
                raise
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.ChunkedEncodingError,
                    ConnectionResetError) as e:
                if attempt < retries - 1:
                    wait = 5 * (attempt + 1)
                    print(f"  [embedder] 网络错误({type(e).__name__})，{wait}s 后重试 ({attempt+1}/{retries})...", flush=True)
                    time.sleep(wait)
                    continue
                raise
            except Exception as e:
                if attempt < retries - 1:
                    wait = 3 * (attempt + 1)
                    print(f"  [embedder] 异常({type(e).__name__})，{wait}s 后重试 ({attempt+1}/{retries})...", flush=True)
                    time.sleep(wait)
                    continue
                raise


# ============================================================
# Reranker
# ============================================================
class SiliconFlowReranker:
    """
    调用硅基流动 /v1/rerank 接口，返回与原始文档列表顺序一致的分数数组。

    用法:
        rr = SiliconFlowReranker()
        scores = rr.predict([[query, doc1], [query, doc2]])
        # scores == [0.92, 0.43]，下标与输入 pairs 对应
    """

    def __init__(
        self,
        model: str = None,
        api_key: str = None,
    ):
        self.model    = model   or config.SILICONFLOW_RERANK_MODEL
        self.api_key  = api_key or config.SILICONFLOW_API_KEY
        self.base_url = config.SILICONFLOW_BASE_URL

    def predict(self, pairs: List[List[str]], **kwargs) -> List[float]:
        """CrossEncoder.predict 兼容接口。pairs = [[query, doc], ...]"""
        if not pairs:
            return []
        query     = pairs[0][0]
        documents = [p[1] for p in pairs]
        return self._call_api(query, documents)

    def compute_score(self, pairs: List[List[str]], **kwargs) -> List[float]:
        """FlagReranker.compute_score 兼容接口。"""
        return self.predict(pairs)

    def _call_api(
        self,
        query: str,
        documents: List[str],
        retries: int = 5,
    ) -> List[float]:
        url     = f"{self.base_url}/rerank"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model":            self.model,
            "query":            query,
            "documents":        documents,
            "top_n":            len(documents),
            "return_documents": False,
        }
        for attempt in range(retries):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                # results 按 relevance_score 降序，需还原原始顺序
                scores_by_index = {
                    r["index"]: r["relevance_score"] for r in data["results"]
                }
                return [scores_by_index.get(i, 0.0) for i in range(len(documents))]
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else 0
                if status == 429 and attempt < retries - 1:
                    wait = 10 * (attempt + 1)
                    print(f"  [reranker] 限速 429，{wait}s 后重试 ({attempt+1}/{retries})...", flush=True)
                    time.sleep(wait)
                    continue
                raise
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.ChunkedEncodingError,
                    ConnectionResetError) as e:
                if attempt < retries - 1:
                    wait = 5 * (attempt + 1)
                    print(f"  [reranker] 网络错误({type(e).__name__})，{wait}s 后重试 ({attempt+1}/{retries})...", flush=True)
                    time.sleep(wait)
                    continue
                raise
            except Exception as e:
                if attempt < retries - 1:
                    wait = 3 * (attempt + 1)
                    print(f"  [reranker] 异常({type(e).__name__})，{wait}s 后重试 ({attempt+1}/{retries})...", flush=True)
                    time.sleep(wait)
                    continue
                raise
