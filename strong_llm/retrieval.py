"""
检索模块 (阶段 B 专用)

提供:
  - LocalRetriever: 基于 ChromaDB 的单文档内本地检索 + 重排序 + 窗口扩展
  - MilvusRetriever: 全局 Milvus 向量检索

架构:
  - ChromaDB  → 轻量级局部 RAG (E3/E4), 每篇文档一个临时集合
  - Milvus    → 全局知识库 (E2/E4), 跨文档检索背景知识

这些组件仅在 E2/E3/E4 策略中被 lazy import。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
from utils import load_prompt, render_prompt


# ============================================================
# LocalRetriever — 基于 ChromaDB 的本地文档检索 + 重排序
# ============================================================
class LocalRetriever:
    """
    基于 ChromaDB 的单文档内检索。
    每次 retrieve() 时将 chunks 写入一个临时 ChromaDB 集合，
    检索完毕后自动清理，无需外部服务。

    用法:
        retriever = LocalRetriever()
        hits = retriever.retrieve(chunks, queries, top_k=30)
        hits = retriever.rerank(query, hits, top_n=5)
        hits = retriever.expand_window(hits, all_chunks, window_size=1)
    """

    def __init__(
        self,
        model_path: str = None,
        reranker_path: str = None,
        device: str = None,
    ):
        self.model_path = model_path or config.EMBEDDING_MODEL
        self.reranker_path = reranker_path or config.RERANKER_MODEL
        self.device = device or config.DEVICE
        self._encoder = None
        self._reranker = None

    def _get_encoder(self):
        if self._encoder is None:
            if config.EMBEDDING_BACKEND == "siliconflow":
                from siliconflow_embedder import SiliconFlowEmbedder
                self._encoder = SiliconFlowEmbedder()
            else:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.model_path, device=self.device)
        return self._encoder

    def _get_reranker(self, backend: str = "cross_encoder"):
        if self._reranker is None:
            if config.EMBEDDING_BACKEND == "siliconflow":
                from siliconflow_embedder import SiliconFlowReranker
                self._reranker = SiliconFlowReranker()
            elif backend == "flagembedding":
                from FlagEmbedding import FlagReranker
                self._reranker = FlagReranker(
                    self.reranker_path,
                    use_fp16=("cuda" in self.device),
                )
            else:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(self.reranker_path, device=self.device)
        return self._reranker

    def _build_bm25_index(self, chunks: List[Dict[str, Any]], cache_key: str = None):
        """
        构建 BM25Okapi 索引，支持持久化缓存（pickle 文件）。
        - cache_key=None : 每次重建，不落盘
        - cache_key=str  : 存到 CHROMA_LOCAL_DIR/bm25_cache/{key}.pkl，跨调用复用
        返回 (bm25, ids)
        """
        import pickle
        import re as _re
        from pathlib import Path as _Path
        from rank_bm25 import BM25Okapi

        def _tokenize(text: str):
            return _re.sub(r"[^\w\s]", " ", text.lower()).split()

        ids = [c.get("chunk_id", f"chunk_{i}") for i, c in enumerate(chunks)]

        if cache_key:
            safe_key = _re.sub(r"[^a-zA-Z0-9_-]", "_", cache_key)[:63].ljust(3, "_")
            cache_dir = _Path(config.CHROMA_LOCAL_DIR) / "bm25_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"{safe_key}.pkl"

            if cache_file.exists():
                try:
                    with open(cache_file, "rb") as fp:
                        cached = pickle.load(fp)
                    if cached.get("count") == len(chunks):
                        return cached["bm25"], ids   # 缓存命中
                except Exception:
                    pass  # 缓存损坏则重建

        tokenized = [_tokenize(c["text"]) for c in chunks]
        bm25 = BM25Okapi(tokenized)

        if cache_key:
            try:
                with open(cache_file, "wb") as fp:
                    pickle.dump({"bm25": bm25, "count": len(chunks)}, fp)
            except Exception:
                pass

        return bm25, ids

    def _build_chroma_collection(self, chunks: List[Dict[str, Any]], cache_key: str = None):
        """
        将 chunks 写入 ChromaDB 集合。
        - cache_key=None  : 纯内存临时集合，用后删除
        - cache_key=str   : 持久化集合（存到 CHROMA_LOCAL_DIR/embed_cache），跨调用复用
        返回 (client, collection, ids, is_persistent)
        """
        import chromadb
        import uuid
        import re as _re
        from pathlib import Path as _Path

        if cache_key:
            # 规范化集合名：只允许字母数字 + _ + -，长度 3~63
            safe_key = _re.sub(r"[^a-zA-Z0-9_-]", "_", cache_key)[:63]
            safe_key = safe_key.ljust(3, "_")

            persist_dir = str(config.CHROMA_LOCAL_DIR / "embed_cache")
            _Path(persist_dir).mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=persist_dir)

            # 尝试命中缓存
            try:
                collection = client.get_collection(name=safe_key, embedding_function=None)
                if collection.count() == len(chunks):
                    # 做一次探针查询验证 HNSW 索引完整性（防止上次中断导致文件损坏）
                    try:
                        collection.query(
                            query_embeddings=[[0.0] * 1024],
                            n_results=1,
                            include=["distances"],
                        )
                        ids = [c.get("chunk_id", f"chunk_{i}") for i, c in enumerate(chunks)]
                        return client, collection, ids, True  # 缓存命中且健康
                    except Exception:
                        print(f"  [cache] {safe_key} HNSW 索引损坏，重建缓存...", flush=True)
                        client.delete_collection(safe_key)
                else:
                    # chunk 数量不匹配（参数变了），删除旧缓存
                    client.delete_collection(safe_key)
            except Exception:
                pass

            col_name = safe_key
            is_persistent = True
        else:
            client = chromadb.Client()  # 纯内存，不落盘
            col_name = f"local_{uuid.uuid4().hex[:8]}"
            is_persistent = False

        collection = client.create_collection(
            name=col_name,
            metadata={"hnsw:space": config.CHROMA_DISTANCE},
            embedding_function=None,
        )

        # 编码并写入
        encoder = self._get_encoder()
        texts = [c["text"] for c in chunks]
        embeddings = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
        ids = [c.get("chunk_id", f"chunk_{i}") for i, c in enumerate(chunks)]
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"index": str(i)} for i in range(len(chunks))],
        )
        return client, collection, ids, is_persistent

    def retrieve(
        self,
        chunks: List[Dict[str, Any]],
        queries: List[str],
        top_k: int = 30,
        cache_key: str = None,
        bm25_weight: float = None,
    ) -> List[Dict[str, Any]]:
        """
        将 chunks 写入 ChromaDB 集合 → 多查询检索 → 合并去重 → 返回 top_k。
        cache_key: 若提供则使用持久化缓存，相同参数下跨调用复用向量（避免 BO 重复编码）。
        bm25_weight: BM25 权重（0=纯向量，1=纯BM25）。None 时从 config.DEFAULT_PARAMS 读取。
        """
        if not chunks or not queries:
            return []

        if bm25_weight is None:
            bm25_weight = config.DEFAULT_PARAMS.get("bm25_weight", 0.0)

        encoder = self._get_encoder()
        client, collection, ids, is_persistent = self._build_chroma_collection(chunks, cache_key)

        # 构建 chunk_id → chunk 映射
        id_to_chunk = {c.get("chunk_id", f"chunk_{i}"): c for i, c in enumerate(chunks)}

        try:
            # --- 向量检索：多查询取最大得分 ---
            vec_scores: Dict[str, float] = {}
            for q in queries:
                qvec = encoder.encode([q], normalize_embeddings=True, show_progress_bar=False).tolist()
                results = collection.query(
                    query_embeddings=qvec,
                    n_results=min(top_k, len(chunks)),
                    include=["distances"],
                )
                for cid, dist in zip(results["ids"][0], results["distances"][0]):
                    sim = 1.0 - dist  # cosine distance → similarity
                    if cid not in vec_scores or sim > vec_scores[cid]:
                        vec_scores[cid] = sim

            if bm25_weight > 0.0:
                # --- BM25 检索：多查询取最大得分，min-max 归一到 [0,1] ---
                import re as _re
                def _tokenize(text: str):
                    return _re.sub(r"[^\w\s]", " ", text.lower()).split()

                bm25, _ = self._build_bm25_index(chunks, cache_key)
                bm25_scores: Dict[str, float] = {}
                for q in queries:
                    q_tokens = _tokenize(q)
                    scores = bm25.get_scores(q_tokens)
                    for cid, score in zip(ids, scores):
                        if cid not in bm25_scores or score > bm25_scores[cid]:
                            bm25_scores[cid] = float(score)

                # 归一化 BM25 分数到 [0, 1]
                if bm25_scores:
                    bm25_min = min(bm25_scores.values())
                    bm25_max = max(bm25_scores.values())
                    span = bm25_max - bm25_min or 1.0
                    bm25_scores = {k: (v - bm25_min) / span for k, v in bm25_scores.items()}

                # 归一化向量分数到 [0, 1]
                if vec_scores:
                    vec_min = min(vec_scores.values())
                    vec_max = max(vec_scores.values())
                    vec_span = vec_max - vec_min or 1.0
                    vec_scores = {k: (v - vec_min) / vec_span for k, v in vec_scores.items()}

                # 融合：所有 chunks 都参与（未被向量检索命中的得 0）
                all_ids = set(vec_scores) | set(bm25_scores)
                best_scores: Dict[str, float] = {
                    cid: (1.0 - bm25_weight) * vec_scores.get(cid, 0.0)
                         + bm25_weight * bm25_scores.get(cid, 0.0)
                    for cid in all_ids
                }
            else:
                best_scores = vec_scores

            # 按得分排序
            ranked_ids = sorted(best_scores, key=best_scores.get, reverse=True)[:top_k]
            out = []
            for cid in ranked_ids:
                c = dict(id_to_chunk.get(cid, {}))
                c["retrieval_score"] = best_scores[cid]
                out.append(c)
            return out
        finally:
            # 仅清理非持久化的临时集合
            if not is_persistent:
                try:
                    client.delete_collection(collection.name)
                except Exception:
                    pass

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_n: int = 5,
        backend: str = None,
    ) -> List[Dict[str, Any]]:
        """重排序 chunks，返回 top_n。"""
        if not chunks:
            return []
        backend = backend or config.DEFAULT_PARAMS.get("reranker_backend", "cross_encoder")
        try:
            reranker = self._get_reranker(backend)
            pairs = [[query, c["text"]] for c in chunks]
            if backend == "flagembedding":
                scores = reranker.compute_score(pairs)
            else:
                scores = reranker.predict(pairs)
            ranked = []
            for c, s in zip(chunks, scores):
                cc = dict(c)
                cc["rerank_score"] = float(s)
                ranked.append(cc)
            ranked.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            return ranked[:top_n]
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "Rerank failed (backend=%s), returning un-reranked top_n: %s",
                backend, e
            )
            return chunks[:top_n]

    @staticmethod
    def expand_window(
        chunks: List[Dict[str, Any]],
        all_chunks: List[Dict[str, Any]],
        window_size: int = 1,
    ) -> List[Dict[str, Any]]:
        """基于 chunk_id 向前后扩展 window_size 个上下文块。"""
        if window_size <= 0:
            return chunks
        idx_map = {c["chunk_id"]: i for i, c in enumerate(all_chunks)}
        selected = set()
        for c in chunks:
            i = idx_map.get(c["chunk_id"])
            if i is None:
                continue
            for j in range(max(0, i - window_size), min(len(all_chunks), i + window_size + 1)):
                selected.add(j)
        return [all_chunks[i] for i in sorted(selected)]


# ============================================================
# MilvusRetriever — 全局混合检索 (Dense + Sparse)
# ============================================================
class MilvusRetriever:
    """
    Milvus 全局知识库检索。
    使用 BGEM3FlagModel 同时编码 Dense + Sparse 向量，
    通过 AnnSearchRequest + WeightedRanker 做原生混合检索。

    与 rag_server_v2.py 的 hybrid_search 逻辑保持一致。

    用法:
        retriever = MilvusRetriever()
        chunks = retriever.search(queries, top_k=30)
    """

    def __init__(
        self,
        uri: str = None,
        collection: str = None,
        token: str = None,
        embedding_model: str = None,
        device: str = None,
    ):
        self.uri = uri or config.MILVUS_URI
        self.collection_name = collection or config.MILVUS_COLLECTION
        self.token = token or config.MILVUS_TOKEN
        self.embedding_model = embedding_model or config.EMBEDDING_MODEL
        self.device = device or config.DEVICE
        self._collection = None
        self._encoder = None

    def _get_collection(self):
        """Lazy 初始化 pymilvus Collection（原生 SDK，支持 hybrid_search）。"""
        if self._collection is None:
            from pymilvus import connections, Collection
            connections.connect(uri=self.uri, token=self.token)
            self._collection = Collection(name=self.collection_name)
            self._collection.load()
        return self._collection

    def _get_encoder(self):
        """Lazy 初始化 BGEM3FlagModel（同时编码 Dense + Sparse）。"""
        if self._encoder is None:
            from FlagEmbedding import BGEM3FlagModel
            self._encoder = BGEM3FlagModel(
                self.embedding_model,
                use_fp16=True,
                normalize_embeddings=True,
            )
        return self._encoder

    def search(
        self,
        queries: List[str],
        top_k: int = 30,
        filter_expr: str = None,
        exclude_paper_id: str = None,
        exclude_doi: str = None,
        dedup_threshold: float = None,
        exclude_tables: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        原生 Milvus 混合检索：Dense + Sparse 双路召回 + WeightedRanker 融合。
        支持批量多 Query 并发检索。

        Args:
            exclude_paper_id: 排除与该 paper_id 关联的 chunks（防自漏泄），
                              匹配规则：chunk_id 以 "{paper_id}#" 或 "{paper_id}_chunk_" 开头，或 metadata.paper_id 相同
            exclude_doi: 排除与该 DOI 关联的 chunks（同一论文不同 paper_id 也能排除）
            dedup_threshold: 余弦相似度  > 此阈值的 chunks 视为重复，只保留得分最高的一条
            exclude_tables: 排除 is_table/content_type 为表格的 chunks，只保留文本
        """
        from pymilvus import AnnSearchRequest, WeightedRanker

        collection = self._get_collection()
        encoder = self._get_encoder()

        # 1. 一次性编码所有 query (Dense + Sparse)
        output = encoder.encode(
            queries,
            batch_size=len(queries),
            max_length=8192,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vecs = [v.tolist() for v in output["dense_vecs"]]
        sparse_vecs = output["lexical_weights"]  # List[dict]

        # 2. 构建 AnnSearchRequest
        dense_req = AnnSearchRequest(
            data=dense_vecs,
            anns_field=config.MILVUS_DENSE_FIELD,
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            expr=filter_expr,
        )
        sparse_req = AnnSearchRequest(
            data=sparse_vecs,
            anns_field=config.MILVUS_SPARSE_FIELD,
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            limit=top_k,
            expr=filter_expr,
        )

        # 3. 混合检索 + 融合
        results = collection.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=WeightedRanker(config.MILVUS_DENSE_WEIGHT, config.MILVUS_SPARSE_WEIGHT),
            limit=top_k,
            output_fields=["text", "*"],
        )

        # 4. 合并去重
        all_hits: Dict[str, Dict[str, Any]] = {}
        for hits in results:
            for hit in hits:
                pk = str(hit.id)
                if pk in all_hits:
                    continue
                entity = hit.entity
                text = entity.get("text", "")
                meta = {}
                for k, v in entity.items():
                    if k in ("text", "vector", "sparse_vector"):
                        continue
                    if isinstance(v, dict):
                        meta.update(v)
                    else:
                        meta[k] = v
                all_hits[pk] = {
                    "chunk_id": pk,
                    "text": text,
                    "metadata": meta,
                    "retrieval_score": float(hit.distance),
                }

        ranked = sorted(all_hits.values(), key=lambda x: x["retrieval_score"], reverse=True)

        # 5. 按 paper_id / DOI 排除（防止目标论文自身 chunks 泄漏）
        if exclude_paper_id or exclude_doi:
            # 兼容两种 chunk_id 格式: "paper_id#chunk_0" 和 "paper_id_chunk_0"
            prefixes = []
            if exclude_paper_id:
                prefixes = [f"{exclude_paper_id}#", f"{exclude_paper_id}_chunk_"]
            filtered = []
            for ch in ranked:
                cid = ch.get("chunk_id", "")
                meta = ch.get("metadata", {})
                meta_pid = meta.get("paper_id", "")
                meta_doi = meta.get("doi", "")
                if prefixes and (any(cid.startswith(p) for p in prefixes) or meta_pid == exclude_paper_id):
                    continue
                if exclude_doi and meta_doi and meta_doi == exclude_doi:
                    continue
                filtered.append(ch)
            ranked = filtered

        # 5.5 排除表格 chunks（防止通用表格噪声干扰全局知识补全）
        if exclude_tables:
            filtered = []
            for ch in ranked:
                meta = ch.get("metadata", {})
                is_table = meta.get("is_table", "")
                content_type = meta.get("content_type", "")
                # 动态字段中 bool 被转为字符串 "True"/"False"
                if str(is_table).lower() == "true" or content_type == "table":
                    continue
                filtered.append(ch)
            ranked = filtered

        # 6. 相似度去重（余弦 > dedup_threshold 的 chunk 只保留第一个）
        if dedup_threshold and dedup_threshold < 1.0 and len(ranked) > 1:
            kept = []
            kept_texts = []
            try:
                import numpy as np
                encoder = self._get_encoder()
                all_texts = [ch["text"] for ch in ranked]
                vecs = encoder.encode(
                    all_texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
                )
                for i, ch in enumerate(ranked):
                    duplicate = False
                    for j in range(len(kept)):
                        sim = float(np.dot(vecs[i], vecs[kept[j]]))
                        if sim > dedup_threshold:
                            duplicate = True
                            break
                    if not duplicate:
                        kept.append(i)
                        kept_texts.append(ch)
                ranked = kept_texts
            except Exception:
                pass  # 去重失败时保留原顺序

        return ranked[:top_k]

    @staticmethod
    def extract_paper_queries(target_text: str, max_queries: int = 5) -> List[str]:
        """
        从目标论文文本中提取关键实体，生成论文特定的检索查询。
        比静态 FIELD_QUERIES 更有针对性，能检索到与该论文真正相关的跨文档知识。

        策略：提取 title/abstract 中的 inhibitor name、substrate、medium 等关键实体，
        组合成语义丰富的查询句。
        """
        import re
        queries = []

        # 取前 3000 字符（title + abstract + intro 区域）
        head = target_text[:3000]

        # 1. 提取标题（# 开头的第一行非空标题）
        title_match = re.search(r'^#\s+(.+)$', head, re.MULTILINE)
        if not title_match:
            title_match = re.search(r'^##?\s+(.+)$', head, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else ""

        # 2. 提取 abstract 段落
        abstract = ""
        abs_match = re.search(
            r'(?:^#+\s*Abstract\s*\n)(.*?)(?=\n#+\s|\Z)',
            head, re.MULTILINE | re.DOTALL | re.IGNORECASE
        )
        if abs_match:
            raw_abs = abs_match.group(1).strip()
            # 跳过作者/地址信息（常见 MinerU 格式问题）
            if not re.match(r'^\d+[\.\)]\s*[-*]?\s*(?:Department|School|College|Center|Institute|Faculty)', raw_abs) \
               and not re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s*\$\^', raw_abs):
                abstract = raw_abs[:500]

        # 3. 提取抑制剂名称模式
        inhibitor_patterns = [
            r'(?:inhibitor|compound)s?\s+(?:namely|such as|including|are)?\s*([A-Z][\w\-()]+(?:\s+and\s+[A-Z][\w\-()]+)*)',
            r'([A-Z][\w\-()]{3,})\s+(?:as|was used as)\s+(?:a\s+)?(?:corrosion\s+)?inhibitor',
            r'(?:inhibit\w+\s+(?:effect|performance|propert)\w*\s+of\s+)([A-Z][\w\-()]+)',
        ]
        inhibitor_names = []
        for pat in inhibitor_patterns:
            for m in re.finditer(pat, head):
                name = m.group(1).strip()
                if len(name) > 2 and name not in inhibitor_names:
                    inhibitor_names.append(name)

        # 4. 提取基底材料
        substrate_patterns = [
            r'((?:mild|carbon|low[- ]carbon|stainless)\s+steel(?:\s+\w+)?)',
            r'(\w+\s+(?:steel|alloy|aluminum|copper|iron|zinc)(?:\s+\w+)?)',
            r'([A-Z]\d{2,4}\s+(?:steel|alloy))',
        ]
        substrates = []
        for pat in substrate_patterns:
            for m in re.finditer(pat, head, re.IGNORECASE):
                sub = m.group(1).strip()
                if len(sub) > 3 and sub not in substrates:
                    substrates.append(sub)
                    break  # 每个模式只取第一个

        # 5. 提取腐蚀介质
        medium_patterns = [
            r'\b(\d+(?:\.\d+)?\s*[MN]\s+(?:HCl|H[₂2]SO[₄4]|NaCl|H[₃3]PO[₄4]))',
            r'\b((?:HCl|H2SO4|NaCl|H3PO4)\s+(?:solution|acid))',
            r'(\d+(?:\.\d+)?\s*(?:wt\.?%|%)\s*(?:HCl|NaCl|H2SO4))',
        ]
        media = []
        for pat in medium_patterns:
            for m in re.finditer(pat, head, re.IGNORECASE):
                med = m.group(1).strip()
                if med not in media:
                    media.append(med)
                    break

        # 6. 组合查询
        # Query 1: 标题作为整体查询（最直接的语义匹配）
        if title:
            queries.append(f"corrosion inhibition {title}")

        # Query 2: 抑制剂 + 基底 + 介质 组合查询
        inh_str = " ".join(inhibitor_names[:2]) if inhibitor_names else "corrosion inhibitor"
        sub_str = substrates[0] if substrates else "steel"
        med_str = media[0] if media else "acid solution"
        queries.append(f"{inh_str} {sub_str} {med_str} inhibition efficiency")

        # Query 3: 抑制剂的吸附/量化参数
        if inhibitor_names:
            queries.append(f"{inhibitor_names[0]} adsorption mechanism DFT quantum chemical HOMO LUMO Gibbs free energy")

        # Query 4: 实验方法相关
        queries.append(f"{inh_str} electrochemical EIS polarization weight loss measurement")

        # Query 5: abstract 截取（如果有的话）
        if abstract and len(abstract) > 50:
            queries.append(abstract[:200])

        return queries[:max_queries]

    def generate_field_queries(
        self,
        llm_client,
        field_names: List[str] = None,
    ) -> List[str]:
        """为每个字段生成语义检索查询。"""
        field_names = field_names or config.FIELD_NAMES
        template = load_prompt(config.PROMPTS_DIR / "query_gen.txt")
        queries = []
        for field_name in field_names:
            prompt = render_prompt(template, field_name=field_name)
            try:
                resp = llm_client.call(prompt)
                q = resp.content.strip().splitlines()[0].strip()
            except Exception:
                q = field_name
            if q:
                queries.append(q)
        return queries
